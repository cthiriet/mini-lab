"""Inference engine: continuous batching on top of the slot-based KV cache.

One ModelRunner per served model owns the model, a KVCache with `max_batch`
slots and a background thread that runs the scheduler loop:

    while there is work:
        1. drop cancelled sequences                (their slot is free again)
        2. admit waiting requests into free slots   (prefill: the whole prompt, one forward)
        3. ONE batched decode step                  (T=1 for every active sequence)
        4. sample a token per sequence, stream it, retire finished sequences

Sequences join and leave the batch at every step, so a short request never
waits for a long one and a free slot is reused immediately. This is
"continuous batching" (iteration-level scheduling, as in Orca / vLLM).

Requests are submitted from the asyncio event loop (the HTTP server). Events
flow back from the engine thread with loop.call_soon_threadsafe into one
asyncio.Queue per request.
"""

from __future__ import annotations

import asyncio
import codecs
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import AsyncIterator

import torch

from minilab.checkpoint import load_checkpoint
from minilab.model.gpt import GPT, KVCache, sample_next
from minilab.obs.metrics import Counter, Gauge, Histogram
from minilab.registry import ModelInfo, list_models
from minilab.settings import get_settings
from minilab.tokenizer.bpe import Tokenizer
from minilab.tokenizer.chat import parse_completion

log = logging.getLogger("minilab.inference")

PROMPT_TOKENS = Counter("minilab_inference_prompt_tokens_total", "Prompt tokens prefilled", ["model"])
COMPLETION_TOKENS = Counter("minilab_inference_completion_tokens_total", "Tokens generated", ["model"])
ACTIVE = Gauge("minilab_inference_active_sequences", "Sequences currently in the batch", ["model"])
QUEUE_DEPTH = Gauge("minilab_inference_queue_depth", "Requests waiting for a free slot", ["model"])
BATCH_SIZE = Histogram("minilab_inference_batch_size", "Sequences per decode step", ["model"],
                       buckets=(1, 2, 4, 8, 16, 32, 64))
TTFT = Histogram("minilab_inference_time_to_first_token_seconds", "Submit to first token", ["model"])
LATENCY = Histogram("minilab_inference_request_latency_seconds", "Submit to last token", ["model"])


class EngineError(Exception):
    status, code = 500, "internal_error"


class ContextLengthExceeded(EngineError):
    status, code = 400, "context_length_exceeded"


class Overloaded(EngineError):
    status, code = 503, "overloaded"


@dataclass
class SamplingParams:
    max_tokens: int | None = None  # None: until the context is full
    temperature: float = 1.0       # 0: greedy
    top_k: int | None = None
    top_p: float = 1.0
    seed: int | None = None        # same seed + same prompt -> same completion
    stop: list[str] = field(default_factory=list)  # stop strings, matched on visible content


# ---------------------------------------------------------------------------
# Streaming parser
# ---------------------------------------------------------------------------

class StreamParser:
    """Incremental twin of tokenizer.chat.parse_completion, fed one token at a time.

    feed() returns deltas ({"content": ...} or {"reasoning": ...}) as soon as text
    is safe to show:
    - content and reasoning (inside <|think_start|>...<|think_end|>) are separate
      channels, each with its own incremental UTF-8 decoder, so a multi-byte
      character split across two tokens is only emitted once complete;
    - tool-call tokens are never streamed (the final event carries the parsed calls);
    - content that could be the start of a stop string is held back until the
      next tokens tell whether it is one.
    The concatenation of all content deltas is always equal to `self.content`.
    """

    def __init__(self, tok: Tokenizer, stop: list[str] | tuple[str, ...] = ()):
        self.tok = tok
        S = tok.special
        self.END, self.THINK, self.THINK_END = S("<|assistant_end|>"), S("<|think_start|>"), S("<|think_end|>")
        self.CALL, self.CALL_END = S("<|tool_call_start|>"), S("<|tool_call_end|>")
        self.stop = [s for s in stop if s]
        self.max_stop = max((len(s) for s in self.stop), default=0)
        self.in_think = self.in_call = False
        self.content = ""               # visible content so far (sent + held back)
        self.sent = 0                   # how much of self.content was already emitted
        self.reasoning: str | None = None
        self.finished = False           # saw <|assistant_end|>
        self.stopped = False            # hit a stop string
        self._content_dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._reasoning_dec = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def feed(self, t: int) -> list[dict]:
        # Same state machine, same order of checks as parse_completion.
        if t == self.END:
            self.finished = True
        elif t == self.THINK:
            self.in_think = True
            self.reasoning = self.reasoning or ""
        elif t == self.THINK_END:
            self.in_think = False
        elif t == self.CALL:
            self.in_call = True
        elif t == self.CALL_END:
            self.in_call = False
        elif self.tok.is_special(t) or self.in_call:
            pass  # stray special token, or tool-call JSON: never streamed
        elif self.in_think:
            text = self._reasoning_dec.decode(self.tok.token_bytes(t))
            self.reasoning += text
            return [{"reasoning": text}] if text else []
        else:
            return self._add_content(self._content_dec.decode(self.tok.token_bytes(t)))
        return []

    def flush(self) -> list[dict]:
        """End of generation: emit bytes still buffered by the decoders and held-back text."""
        out = []
        if not self.stopped:
            out += self._add_content(self._content_dec.decode(b"", final=True))
            if not self.stopped:
                out += self._release(len(self.content))  # nothing more will come: not a stop string
        tail = self._reasoning_dec.decode(b"", final=True)
        if tail:
            self.reasoning += tail
            out.append({"reasoning": tail})
        return out

    def _add_content(self, text: str) -> list[dict]:
        if not text:
            return []
        # A new match must end in the new text, so it starts at most max_stop-1 chars earlier.
        start = max(0, len(self.content) - self.max_stop + 1)
        self.content += text
        if not self.stop:
            return self._release(len(self.content))
        hits = [i for s in self.stop if (i := self.content.find(s, start)) >= 0]
        if hits:
            self.content = self.content[:min(hits)]  # the stop string itself is not returned
            self.stopped = True
            return self._release(len(self.content))
        return self._release(len(self.content) - self._held_back())

    def _held_back(self) -> int:
        """Length of the longest suffix of content that is a proper prefix of a stop string."""
        for k in range(min(self.max_stop - 1, len(self.content)), 0, -1):
            tail = self.content[-k:]
            if any(s.startswith(tail) for s in self.stop):
                return k
        return 0

    def _release(self, upto: int) -> list[dict]:
        # upto >= self.sent always holds: text that could start a stop string is never sent.
        text, self.sent = self.content[self.sent:upto], upto
        return [{"content": text}] if text else []


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------

class Request:
    """One generation. Created by ModelRunner.submit(); read with events() or result().

    Fields below `# engine thread` are only touched by the engine thread; the
    consumer only reads events from the queue (and may call cancel()).
    """

    def __init__(self, runner: "ModelRunner", prompt: list[int], params: SamplingParams,
                 max_tokens: int, stream: bool):
        self.runner, self.prompt, self.params = runner, prompt, params
        self.max_tokens, self.stream = max_tokens, stream
        self.cancelled = False
        self.submitted = time.perf_counter()
        self._loop = asyncio.get_running_loop()
        self._queue: asyncio.Queue[dict] = asyncio.Queue()
        self._closed = False  # the consumer got the final event
        # engine thread
        self.slot: int | None = None
        self.tokens: list[int] = []
        self.parser = StreamParser(runner.tokenizer, params.stop)
        # Every request has its own RNG: sampling never depends on who else is in the batch.
        self.generator = torch.Generator()
        if params.seed is None:
            self.generator.seed()  # a fresh random seed (the default one is a constant)
        else:
            self.generator.manual_seed(params.seed % 2**64)

    # -- engine thread ------------------------------------------------------

    def emit(self, event: dict) -> None:
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, event)
        except RuntimeError:  # the consumer's event loop is closed: nobody is listening
            self.cancelled = True

    def emit_deltas(self, deltas: list[dict]) -> None:
        if self.stream:
            for d in deltas:
                self.emit({"type": "delta", **d})

    # -- consumer (event loop) ----------------------------------------------

    async def events(self) -> AsyncIterator[dict]:
        """Delta events (streaming only), then one final "done" or "error" event."""
        try:
            while not self._closed:
                event = await self._queue.get()
                self._closed = event["type"] != "delta"
                yield event
        finally:
            if not self._closed:  # consumer left early (client disconnected): free the slot
                self.cancel()

    async def result(self) -> dict:
        """Wait for the final "done" event (raises EngineError on failure)."""
        async for event in self.events():
            final = event
        if final["type"] == "error":
            raise EngineError(final["message"])
        return final

    def cancel(self) -> None:
        self.runner.cancel(self)


# ---------------------------------------------------------------------------
# Per-model scheduler
# ---------------------------------------------------------------------------

class ModelRunner:
    """Serves one model: a KV cache with max_batch slots and a scheduler thread."""

    def __init__(self, info: ModelInfo, model: GPT, tokenizer: Tokenizer, max_batch: int = 8, max_queue: int = 64):
        self.info, self.model, self.tokenizer = info, model, tokenizer
        self.max_batch, self.max_queue = max_batch, max_queue
        self.n_ctx = model.config.block_size
        weight = model.lm_head.weight
        self.device = weight.device
        self.cache = KVCache(model.config, batch_size=max_batch, device=weight.device, dtype=weight.dtype)
        self._free = list(range(max_batch - 1, -1, -1))  # stack of free slots (pop() gives slot 0 first)
        self._active: dict[int, Request] = {}              # slot -> request (engine thread only)
        self._waiting: deque[Request] = deque()            # guarded by _cv
        self._cv = threading.Condition()
        self._stopping = False
        self._thread: threading.Thread | None = None

    @property
    def num_active(self) -> int:
        return len(self._active)

    @property
    def num_waiting(self) -> int:
        return len(self._waiting)

    # -- called from the event loop -------------------------------------------

    def submit(self, prompt: list[int], params: SamplingParams, stream: bool = False) -> Request:
        """Queue a request. Must be called from a running event loop."""
        if len(prompt) >= self.n_ctx:
            raise ContextLengthExceeded(
                f"This model's maximum context length is {self.n_ctx} tokens, but the prompt is "
                f"{len(prompt)} tokens long (at least 1 token must be left for the completion)."
            )
        room = self.n_ctx - len(prompt)  # prompt + completion must fit in the context
        max_tokens = room if params.max_tokens is None else max(1, min(params.max_tokens, room))
        req = Request(self, prompt, params, max_tokens, stream)
        with self._cv:
            if self._stopping:
                raise Overloaded("The inference server is shutting down.")
            if len(self._waiting) >= self.max_queue:
                raise Overloaded(f"Model {self.info.id} is overloaded ({len(self._waiting)} requests waiting), "
                                 "please retry later.")
            self._waiting.append(req)
            QUEUE_DEPTH.set(len(self._waiting), model=self.info.id)
            self._cv.notify()
        return req

    def cancel(self, req: Request) -> None:
        """Stop working on `req` (e.g. the client disconnected). Safe from any thread."""
        with self._cv:
            req.cancelled = True
            if req in self._waiting:
                self._waiting.remove(req)
                QUEUE_DEPTH.set(len(self._waiting), model=self.info.id)
            self._cv.notify()  # an active one is dropped at the start of the next step

    # -- thread lifecycle -------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name=f"engine-{self.info.id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        with self._cv:
            self._stopping = True
            self._cv.notify()
        if self._thread is not None:
            self._thread.join()

    def _run(self) -> None:
        with torch.inference_mode():  # grad mode is thread-local: it must be set in this thread
            while True:
                with self._cv:
                    while not (self._stopping or self._waiting or self._active):
                        self._cv.wait()  # idle: sleep until submit() / cancel() / stop()
                    if self._stopping:
                        break
                try:
                    self._step()
                except Exception as e:  # never let one bad step kill the engine thread
                    log.exception("engine step failed")
                    for req in list(self._active.values()):
                        self._fail(req, f"Generation failed: {e}")
        with self._cv:
            leftovers, self._waiting = [*self._active.values(), *self._waiting], deque()
        for req in leftovers:
            self._fail(req, "The inference server is shutting down.")

    # -- engine thread ------------------------------------------------------------

    def _step(self) -> None:
        # 1. Cancelled sequences give their slot back right away.
        for req in [r for r in self._active.values() if r.cancelled]:
            self._release(req)

        # 2. Admit waiting requests into free slots. Prefill runs the whole prompt
        #    through the model in one forward pass (T = prompt length), fills the
        #    slot's keys/values and gives the logits of the first completion token.
        while self._free:
            with self._cv:
                if not self._waiting:
                    break
                req = self._waiting.popleft()
            if not req.cancelled:
                self._prefill(req)

        # 3. One decode step for all active sequences at once (T = 1): every row feeds
        #    its last sampled token, at its own position, in its own cache slot.
        if self._active:
            reqs = list(self._active.values())
            idx = torch.tensor([[r.tokens[-1]] for r in reqs], device=self.device)
            logits = self.model.forward_cached(idx, self.cache, torch.tensor([r.slot for r in reqs]))
            BATCH_SIZE.observe(len(reqs), model=self.info.id)
            self._sample(reqs, logits)

        ACTIVE.set(len(self._active), model=self.info.id)
        QUEUE_DEPTH.set(len(self._waiting), model=self.info.id)

    def _prefill(self, req: Request) -> None:
        req.slot = self._free.pop()
        self.cache.reset(req.slot)
        self._active[req.slot] = req
        idx = torch.tensor([req.prompt], device=self.device)
        logits = self.model.forward_cached(idx, self.cache, torch.tensor([req.slot]))
        PROMPT_TOKENS.inc(len(req.prompt), model=self.info.id)
        self._sample([req], logits)

    def _sample(self, reqs: list[Request], logits: torch.Tensor) -> None:
        # A model may pad its vocab beyond the tokenizer's: never sample those ids.
        logits = logits[:, :self.tokenizer.vocab_size]
        # Rows are sampled one by one because every request has its own temperature,
        # top-k, top-p and RNG: a seeded request gets the same tokens whatever else
        # shares the batch with it.
        for req, row in zip(reqs, logits):
            p = req.params
            try:
                token = int(sample_next(row[None], p.temperature, p.top_k, p.top_p, req.generator))
            except Exception as e:  # one bad request fails alone, not everyone in the batch
                log.exception("sampling failed")
                self._fail(req, f"Sampling failed: {e}")
                continue
            self._on_token(req, token)

    def _on_token(self, req: Request, token: int) -> None:
        req.tokens.append(token)
        COMPLETION_TOKENS.inc(model=self.info.id)
        if len(req.tokens) == 1:
            TTFT.observe(time.perf_counter() - req.submitted, model=self.info.id)
        req.emit_deltas(req.parser.feed(token))
        if req.parser.finished or req.parser.stopped:  # <|assistant_end|> or a stop string
            self._finish(req, "stop")
        elif len(req.tokens) >= req.max_tokens:  # max_tokens, or the context is full (clamped at submit)
            self._finish(req, "length")

    def _finish(self, req: Request, reason: str) -> None:
        req.emit_deltas(req.parser.flush())
        tool_calls = parse_completion(self.tokenizer, req.tokens).tool_calls
        req.emit({
            "type": "done",
            "content": req.parser.content,
            "reasoning": req.parser.reasoning,
            "tool_calls": tool_calls,
            "finish_reason": "tool_calls" if tool_calls else reason,
            "usage": {"prompt_tokens": len(req.prompt), "completion_tokens": len(req.tokens)},
        })
        LATENCY.observe(time.perf_counter() - req.submitted, model=self.info.id)
        self._release(req)

    def _fail(self, req: Request, message: str) -> None:
        req.emit({"type": "error", "message": message})
        if req.slot is not None and self._active.get(req.slot) is req:
            self._release(req)

    def _release(self, req: Request) -> None:
        # Nothing to clear: the next prefill in this slot resets its length, and keys
        # beyond a slot's length are masked out anyway.
        del self._active[req.slot]
        self._free.append(req.slot)


# ---------------------------------------------------------------------------
# All models
# ---------------------------------------------------------------------------

class Engine:
    """Loads the released models (models_dir) and runs one ModelRunner per model.

    Env: MINILAB_SERVE_MODELS (comma list, default: all released models),
    MINILAB_MAX_BATCH (slots per model, default 8), MINILAB_MAX_QUEUE (waiting
    requests per model before 503, default 64).
    """

    def __init__(self, models_dir: str | os.PathLike | None = None, model_ids: list[str] | None = None,
                 max_batch: int | None = None, max_queue: int | None = None):
        self.models_dir = models_dir or get_settings().models_dir
        env_ids = [m.strip() for m in os.environ.get("MINILAB_SERVE_MODELS", "").split(",") if m.strip()]
        self.model_ids = model_ids or env_ids or None
        self.max_batch = max_batch if max_batch is not None else int(os.environ.get("MINILAB_MAX_BATCH", "8"))
        self.max_queue = max_queue if max_queue is not None else int(os.environ.get("MINILAB_MAX_QUEUE", "64"))
        self.runners: dict[str, ModelRunner] = {}

    def start(self) -> None:
        for info in list_models(self.models_dir):
            if info.id in self.runners or (self.model_ids and info.id not in self.model_ids):
                continue
            try:
                model, tok, _ = load_checkpoint(info.path)
            except Exception:
                log.exception("could not load model %s from %s", info.id, info.path)
                continue
            runner = ModelRunner(info, model, tok, self.max_batch, self.max_queue)
            runner.start()
            self.runners[info.id] = runner
            log.info("serving %s: %.2fM params, context %d, max batch %d",
                     info.id, model.num_params() / 1e6, runner.n_ctx, self.max_batch)
        missing = set(self.model_ids or ()) - set(self.runners)
        if missing:
            log.warning("models not found in %s: %s", self.models_dir, ", ".join(sorted(missing)))
        if not self.runners:
            log.warning("no model loaded from %s", self.models_dir)

    def stop(self) -> None:
        for runner in self.runners.values():
            runner.stop()
        self.runners.clear()
