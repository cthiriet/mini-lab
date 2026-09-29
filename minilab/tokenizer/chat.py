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
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from minilab.tokenizer.bpe import Tokenizer


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


def _tool_call_json(call: dict) -> str:
    fn = call.get("function", call)
    args = fn.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            pass
    return json.dumps({"name": fn["name"], "arguments": args})


def render_conversation(
    tok: Tokenizer, messages: list[dict], tools=None
) -> tuple[list[int], list[int]]:
    """Render a full conversation for training.

    Returns (ids, mask) where mask[i] == 1 for tokens the model should learn to
    produce (assistant turns, including their <|assistant_end|>), 0 otherwise.
    """
    S = tok.special
    ids: list[int] = []
    mask: list[int] = []

    def add(tokens: list[int], train: bool) -> None:
        ids.extend(tokens)
        mask.extend([int(train)] * len(tokens))

    add([tok.bos_id], False)
    names = _tool_names(tools)
    if names:
        add([S("<|system_start|>"), *tok.encode("tools: " + ", ".join(names)), S("<|system_end|>")], False)

    for msg in messages:
        role = msg["role"]
        text = _content_text(msg.get("content"))
        if role in ("system", "developer"):
            add([S("<|system_start|>"), *tok.encode(text), S("<|system_end|>")], False)
        elif role == "user":
            add([S("<|user_start|>"), *tok.encode(text), S("<|user_end|>")], False)
        elif role == "tool":
            add([S("<|tool_start|>"), *tok.encode(text), S("<|tool_end|>")], False)
        elif role == "assistant":
            add([S("<|assistant_start|>")], False)
            body: list[int] = []
            if msg.get("reasoning"):
                body += [S("<|think_start|>"), *tok.encode(msg["reasoning"]), S("<|think_end|>")]
            body += tok.encode(text)
            for call in msg.get("tool_calls") or []:
                body += [S("<|tool_call_start|>"), *tok.encode(_tool_call_json(call)), S("<|tool_call_end|>")]
            body.append(S("<|assistant_end|>"))
            add(body, True)
        else:
            raise ValueError(f"unknown role: {role}")
    return ids, mask


def render_prompt(tok: Tokenizer, messages: list[dict], tools=None) -> list[int]:
    """Render messages for inference: the result ends with <|assistant_start|>."""
    ids, _ = render_conversation(tok, messages, tools)
    return ids + [tok.special("<|assistant_start|>")]


@dataclass
class ParsedCompletion:
    content: str = ""
    reasoning: str | None = None
    tool_calls: list[dict] = field(default_factory=list)  # [{"name": str, "arguments": str (JSON)}]
    finished: bool = False  # True if <|assistant_end|> was produced


def parse_completion(tok: Tokenizer, ids: list[int]) -> ParsedCompletion:
    """Parse generated assistant tokens (everything after <|assistant_start|>)."""
    S = tok.special
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
                out.tool_calls.append(_parse_tool_call(tok.decode(call)))
            call = None
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


def _parse_tool_call(text: str) -> dict:
    try:
        data = json.loads(text)
        return {"name": str(data["name"]), "arguments": json.dumps(data.get("arguments", {}))}
    except (json.JSONDecodeError, KeyError, TypeError):
        return {"name": "invalid", "arguments": json.dumps({"raw": text})}


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
