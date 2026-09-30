"""Chat template: turn OpenAI-style messages into token ids, and back.

A rendered conversation looks like this (one line per turn for readability):

    <|bos|>
    <|system_start|>tools: calculator<|system_end|>            # only if tools are enabled
    <|system_start|>You are a helpful assistant.<|system_end|>  # optional system message
    <|user_start|>What is 347 + 58?<|user_end|>
    <|assistant_start|><|tool_call_start|>{"name": "calculator", "arguments": {"expression": "347 + 58"}}<|tool_call_end|><|assistant_end|>
    <|tool_start|>405<|tool_end|>
    <|assistant_start|><|think_start|>...scratchpad...<|think_end|>347 + 58 = 405<|assistant_end|>

Messages use the OpenAI format. Assistant messages may carry a "reasoning" key
(our extension) that is rendered inside <|think_start|>...<|think_end|>.

The tokenizer says which template its model was trained with (`tok.chat_template`), like a
Hugging Face tokenizer config. mini-code's "code" template differs in four ways, all so that a
coding agent's requests (opencode sends ~20k characters of system prompt and ~14k of tool
schemas) fit a model with a 1,024-token context:

- tool calls carry their arguments raw, not as JSON, so code needs no escaping:
      <|tool_call_start|>edit<|arg|>path=calc.py<|arg|>oldString=    return a - b<|arg|>newString=...<|tool_call_end|>
  (every value comes back as a string; the API types them with the client's JSON schemas);
- a system message is cut to its first sentence ("You are an AI agent running in OpenCode, a
  coding agent harness."): the model learned how to behave, it can't read 5,000 tokens of rules;
- paths under the working directory (from the system prompt's "Working directory: ...") are
  shown relative to it, and a tool error in opencode's JSON is shown as "Error: <message>";
- `render_prompt(..., budget=N)` keeps the prompt within N tokens by dropping the oldest turns,
  then shortening tool results: the server does the context management a client would do.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from minilab.tokenizer.bpe import Tokenizer

ARG = "<|arg|>"
MAX_SYSTEM_CHARS = 200
MAX_TOOL_CHARS = 4000
MAX_TEXT_CHARS = 8000   # a user or assistant message: far more than the context holds anyway


class PromptTooLong(ValueError):
    """Even cut down, the current request doesn't fit the budget."""


def _tool_names(tools) -> list[str]:
    names = []
    for t in tools or []:
        if isinstance(t, str):
            names.append(t)
        elif isinstance(t, dict):
            names.append(t.get("function", {}).get("name") or t.get("name"))
    return [n for n in names if n]


def _content_text(content) -> str:
    """OpenAI content can be a string or a list of parts; we only keep text parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")


def _call_arguments(call: dict) -> tuple[str, dict | str]:
    fn = call.get("function", call)
    args = fn.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            pass
    return fn["name"], args


def _tool_call_json(call: dict) -> str:
    name, args = _call_arguments(call)
    return json.dumps({"name": name, "arguments": args})


def _tool_call_ids(tok: Tokenizer, call: dict) -> list[int]:
    """The "code" template: name<|arg|>key=value<|arg|>key=value, values raw."""
    name, args = _call_arguments(call)
    ids = tok.encode(name)
    for key, value in (args.items() if isinstance(args, dict) else []):
        text = value if isinstance(value, str) else json.dumps(value)
        ids += [tok.special(ARG), *tok.encode(f"{key}={text}")]
    return ids


# ---- the "code" template: what a coding agent's messages become -------------------------

def first_sentence(text: str) -> str:
    line = text.strip().split("\n", 1)[0]
    m = re.match(r"(.+?[.!?])(\s|$)", line)
    return (m.group(1) if m else line)[:MAX_SYSTEM_CHARS]


def working_directory(messages: list[dict]) -> str | None:
    for m in messages:
        if m.get("role") in ("system", "developer"):
            found = re.search(r"Working directory: *(\S[^\n]*)", _content_text(m.get("content")))
            if found:
                return found.group(1).strip().rstrip("/")
    return None


def tool_result_text(text: str) -> str:
    """opencode sends a failed tool call's result as JSON: keep the message."""
    if text.startswith('{"error"'):
        try:
            return "Error: " + json.loads(text)["error"]["message"]
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    return text


def shorten(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit("\n", 1)[0] + "\n[... truncated]"


def code_messages(messages: list[dict]) -> list[dict]:
    """The messages as the "code" template shows them (see the module docstring)."""
    cwd = working_directory(messages)

    def rel(text: str) -> str:
        return text.replace(cwd + "/", "").replace(cwd, ".") if cwd else text

    out = []
    for m in messages:
        m = dict(m)
        text = _content_text(m.get("content"))
        if m["role"] in ("system", "developer"):
            m["content"] = first_sentence(text)
        elif m["role"] == "tool":
            m["content"] = shorten(rel(tool_result_text(text)), MAX_TOOL_CHARS)
        elif m["role"] == "user":
            m["content"] = shorten(rel(text), MAX_TEXT_CHARS)
        elif m["role"] == "assistant":
            m["content"] = shorten(text, MAX_TEXT_CHARS)
        out.append(m)
    return out


# ---- rendering ------------------------------------------------------------------------

def _render_header(tok: Tokenizer, tools) -> list[int]:
    S = tok.special
    ids = [tok.bos_id]
    names = _tool_names(tools)
    if names:
        ids += [S("<|system_start|>"), *tok.encode("tools: " + ", ".join(names)), S("<|system_end|>")]
    return ids


def _render_message(tok: Tokenizer, msg: dict) -> tuple[list[int], list[int]]:
    """One message: (ids, mask), mask 1 on what the assistant says."""
    S = tok.special
    role = msg["role"]
    text = _content_text(msg.get("content"))
    if role in ("system", "developer"):
        ids = [S("<|system_start|>"), *tok.encode(text), S("<|system_end|>")]
    elif role == "user":
        ids = [S("<|user_start|>"), *tok.encode(text), S("<|user_end|>")]
    elif role == "tool":
        ids = [S("<|tool_start|>"), *tok.encode(text), S("<|tool_end|>")]
    elif role == "assistant":
        body: list[int] = []
        if msg.get("reasoning"):
            body += [S("<|think_start|>"), *tok.encode(msg["reasoning"]), S("<|think_end|>")]
        body += tok.encode(text)
        for call in msg.get("tool_calls") or []:
            inner = _tool_call_ids(tok, call) if tok.chat_template == "code" else tok.encode(_tool_call_json(call))
            body += [S("<|tool_call_start|>"), *inner, S("<|tool_call_end|>")]
        body.append(S("<|assistant_end|>"))
        train = msg.get("weight", 1) != 0  # OpenAI's fine-tuning format: weight 0 = context, not a target
        return [S("<|assistant_start|>"), *body], [0] + [int(train)] * len(body)
    else:
        raise ValueError(f"unknown role: {role}")
    return ids, [0] * len(ids)


def render_conversation(
    tok: Tokenizer, messages: list[dict], tools=None
) -> tuple[list[int], list[int]]:
    """Render a full conversation for training.

    Returns (ids, mask) where mask[i] == 1 for tokens the model should learn to
    produce (assistant turns, including their <|assistant_end|>), 0 otherwise.
    """
    if tok.chat_template == "code":
        messages = code_messages(messages)
    ids = _render_header(tok, tools)
    mask = [0] * len(ids)
    for msg in messages:
        i, m = _render_message(tok, msg)
        ids += i
        mask += m
    return ids, mask


def render_prompt(tok: Tokenizer, messages: list[dict], tools=None, budget: int | None = None) -> list[int]:
    """Render messages for inference: the result ends with <|assistant_start|>.

    With a budget (the "code" template's context management), the prompt is at most `budget`
    tokens: see fit_messages()."""
    if budget is not None and tok.chat_template == "code":
        messages = fit_messages(tok, code_messages(messages), tools, budget - 1)
        return _render_raw(tok, messages, tools) + [tok.special("<|assistant_start|>")]
    ids, _ = render_conversation(tok, messages, tools)
    return ids + [tok.special("<|assistant_start|>")]


def _render_raw(tok: Tokenizer, messages: list[dict], tools) -> list[int]:
    ids = _render_header(tok, tools)
    for msg in messages:
        ids += _render_message(tok, msg)[0]
    return ids


def fit_messages(tok: Tokenizer, messages: list[dict], tools, budget: int) -> list[dict]:
    """The messages whose rendering fits in `budget` tokens, already in the "code" template's form.

    Kept, in this order: the system messages and the current request (the last user message and
    everything after it: the tool calls made for it so far). Then earlier turns, newest first, a
    whole turn at a time. If the current request alone is too long, its tool results are cut,
    oldest first, then the oldest of its tool round trips are dropped. Raises ValueError if even
    that doesn't fit."""
    system = [m for m in messages if m["role"] in ("system", "developer")]
    rest = [m for m in messages if m["role"] not in ("system", "developer")]
    starts = [i for i, m in enumerate(rest) if m["role"] == "user"] or [0]
    size = {id(m): len(_render_message(tok, m)[0]) for m in rest}
    fixed = len(_render_header(tok, tools)) + sum(len(_render_message(tok, m)[0]) for m in system)

    current = [dict(m) for m in rest[starts[-1]:]]
    for m in current:
        size[id(m)] = len(_render_message(tok, m)[0])

    def used(msgs):
        return fixed + sum(size[id(m)] for m in msgs)

    # 1. The current request must fit: cut its tool results, oldest first, then drop round trips.
    tools_in_turn = [m for m in current if m["role"] == "tool"]
    for m in tools_in_turn:
        if used(current) <= budget:
            break
        excess = used(current) - budget
        keep = max(200, len(m["content"]) - 4 * excess)
        m["content"] = shorten(m["content"], keep)
        size[id(m)] = len(_render_message(tok, m)[0])
    while used(current) > budget and len(current) > 3 and current[1]["role"] == "assistant":
        # drop the oldest (assistant call, tool result) pair after the user message
        n = 2 if current[2]["role"] == "tool" else 1
        del current[1:1 + n]
    if used(current) > budget:
        raise PromptTooLong(f"the last request alone needs {used(current)} tokens, more than the {budget} available")

    # 2. Earlier turns, newest first, while they fit.
    kept = current
    for a, b in zip(reversed(starts[:-1]), reversed(starts[1:])):
        turn = rest[a:b]
        if used(turn + kept) > budget:
            break
        kept = turn + kept
    return system + kept


@dataclass
class ParsedCompletion:
    content: str = ""
    reasoning: str | None = None
    tool_calls: list[dict] = field(default_factory=list)  # [{"name": str, "arguments": str (JSON)}]
    finished: bool = False  # True if <|assistant_end|> was produced


def parse_completion(tok: Tokenizer, ids: list[int]) -> ParsedCompletion:
    """Parse generated assistant tokens (everything after <|assistant_start|>)."""
    S = tok.special
    arg = tok.special_tokens.get(ARG)
    out = ParsedCompletion()
    content: list[int] = []
    reasoning: list[int] | None = None
    call: list[int] | None = None
    in_think = False
    for t in ids:
        if t == S("<|assistant_end|>"):
            out.finished = True
            break
        if t == S("<|think_start|>"):
            in_think, reasoning = True, (reasoning or [])
        elif t == S("<|think_end|>"):
            in_think = False
        elif t == S("<|tool_call_start|>"):
            call = []
        elif t == S("<|tool_call_end|>"):
            if call is not None:
                out.tool_calls.append(_parse_tool_call(tok, call))
            call = None
        elif call is not None and t == arg:
            call.append(t)
        elif tok.is_special(t):
            continue  # stray special token: ignore
        elif call is not None:
            call.append(t)
        elif in_think:
            reasoning.append(t)
        else:
            content.append(t)
    out.content = tok.decode(content)
    out.reasoning = tok.decode(reasoning) if reasoning is not None else None
    return out


def _parse_tool_call(tok: Tokenizer, ids: list[int]) -> dict:
    arg = tok.special_tokens.get(ARG)
    if arg is not None and tok.chat_template == "code":
        parts, cur = [], []
        for t in ids:
            if t == arg:
                parts.append(cur)
                cur = []
            else:
                cur.append(t)
        parts.append(cur)
        name = tok.decode(parts[0]).strip()
        args = {}
        for p in parts[1:]:
            key, sep, value = tok.decode(p).partition("=")
            if not sep or not key.strip().isidentifier():
                return {"name": "invalid", "arguments": json.dumps({"raw": tok.decode(ids)})}
            args[key.strip()] = value
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", name):
            return {"name": "invalid", "arguments": json.dumps({"raw": tok.decode(ids)})}
        return {"name": name, "arguments": json.dumps(args)}
    text = tok.decode(ids)
    try:
        data = json.loads(text)
        return {"name": str(data["name"]), "arguments": json.dumps(data.get("arguments", {}))}
    except (json.JSONDecodeError, KeyError, TypeError):
        return {"name": "invalid", "arguments": json.dumps({"raw": text})}


def coerce_arguments(arguments: str, schema: dict | None) -> str:
    """The "code" template writes every argument as text: give numbers and booleans their JSON
    type again, following the tool's JSON schema (`parameters`). Anything else is left as is."""
    props = (schema or {}).get("properties") or {}
    try:
        args = json.loads(arguments)
    except json.JSONDecodeError:
        return arguments
    if not isinstance(args, dict):
        return arguments
    for key, value in args.items():
        kind = (props.get(key) or {}).get("type")
        if not isinstance(value, str) or kind not in ("integer", "number", "boolean"):
            continue
        v = value.strip()
        if kind == "boolean" and v.lower() in ("true", "false"):
            args[key] = v.lower() == "true"
        elif kind == "integer" and re.fullmatch(r"-?\d+", v):
            args[key] = int(v)
        elif kind == "number":
            try:
                args[key] = float(v) if not re.fullmatch(r"-?\d+", v) else int(v)
            except ValueError:
                pass
    return json.dumps(args)


# ---- history trimming (the chat app, and the eval's real-chat check) ----------

def estimate_tokens(text: str) -> int:
    """~4 characters per token, except digits: the tokenizer always splits them, one token
    each. (Counting them as ~4 per token kept 7 turns of additions, 171 real tokens, and
    left too little room for a 5-digit scratchpad.)"""
    digits = sum(c.isdigit() for c in text)
    return digits + (len(text) - digits) // 4


def recent_turns(messages: list[dict], context_length: int) -> list[dict]:
    """The most recent turns whose prompt leaves about half the context for the answer.

    The model's context is tiny (a few hundred tokens): after a long story, the whole
    conversation would still *fit*, but leave no room to reply. So, like a short memory,
    we keep only the latest turns (a few template tokens per message), always starting at
    a user message."""
    budget, used, start = context_length // 2, 0, len(messages)
    for i in range(len(messages) - 1, -1, -1):
        used += estimate_tokens(messages[i].get("content") or "") + 4
        if used > budget and start < len(messages):
            break
        start = i
    kept = messages[start:]
    while len(kept) > 1 and kept[0]["role"] != "user":
        kept = kept[1:]
    return kept
