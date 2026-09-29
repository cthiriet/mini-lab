"""OpenAI chat completions: request validation, translation for the inference server, response shapes.

Requests are validated with pydantic. Unknown fields are ignored (clients send
harmless extras such as `user`, `metadata` or `store`), but parameters that
would change what OpenAI returns and that mini-lab can't honour (`n=2`,
`logprobs`, JSON mode, ...) are rejected with a clear 400: silently ignoring
them would hand back something the caller didn't ask for.

Responses are plain dicts shaped exactly like OpenAI's, so any OpenAI client can
parse them. One non-standard field: the model's scratchpad reasoning is returned
as `message.reasoning_content` (`delta.reasoning_content` when streaming), the
convention other OpenAI-compatible servers use.
"""

from __future__ import annotations

import secrets
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from minilab.api.errors import APIError, invalid_request


class _Lenient(BaseModel):
    model_config = ConfigDict(extra="ignore")


class FunctionCall(_Lenient):
    name: str
    arguments: str | dict = "{}"  # a JSON string in OpenAI's API; a dict is accepted too


class ToolCall(_Lenient):
    id: str | None = None
    type: Literal["function"] = "function"
    function: FunctionCall


class Message(_Lenient):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[dict] | None = None  # a string, or a list of content parts
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


class FunctionDef(_Lenient):
    # Tool names end up in the chat template ("tools: a, b"), so keep them to OpenAI's charset.
    name: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    description: str | None = None
    parameters: dict | None = None


class Tool(_Lenient):
    type: Literal["function"]
    function: FunctionDef


class StreamOptions(_Lenient):
    include_usage: bool = False


class ChatCompletionRequest(_Lenient):
    model: str
    messages: list[Message] = Field(min_length=1)
    tools: list[Tool] | None = None
    tool_choice: str | dict | None = None
    max_tokens: int | None = Field(None, ge=1)  # deprecated by OpenAI in favour of max_completion_tokens
    max_completion_tokens: int | None = Field(None, ge=1)
    temperature: float | None = Field(None, ge=0, le=2)
    top_p: float | None = Field(None, gt=0, le=1)
    top_k: int | None = Field(None, ge=1)  # not in OpenAI's API, but useful with tiny models
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool | None = False
    stream_options: StreamOptions | None = None

    @property
    def include_usage(self) -> bool:
        return bool(self.stream and self.stream_options and self.stream_options.include_usage)


# Parameters accepted only with their no-op value: name -> (is it a no-op?, error message).
_UNSUPPORTED: dict[str, tuple[Callable[[Any], bool], str]] = {
    "n": (lambda v: v in (None, 1), "Only n=1 is supported."),
    "logprobs": (lambda v: not v, "logprobs are not supported."),
    "top_logprobs": (lambda v: not v, "top_logprobs is not supported."),
    "response_format": (lambda v: v is None or (isinstance(v, dict) and v.get("type") == "text"),
                        "Only response_format {\"type\": \"text\"} is supported (no JSON mode)."),
    "presence_penalty": (lambda v: not v, "presence_penalty is not supported."),
    "frequency_penalty": (lambda v: not v, "frequency_penalty is not supported."),
    "logit_bias": (lambda v: not v, "logit_bias is not supported."),
    "functions": (lambda v: not v, "The legacy 'functions' parameter is not supported; use 'tools'."),
    "function_call": (lambda v: v is None, "The legacy 'function_call' parameter is not supported; use 'tool_choice'."),
    "modalities": (lambda v: v is None or v == ["text"], "Only text output is supported."),
    "audio": (lambda v: v is None, "Audio output is not supported."),
    "prediction": (lambda v: v is None, "Predicted outputs are not supported."),
    "web_search_options": (lambda v: v is None, "Web search is not supported."),
}


def parse_chat_request(body: dict) -> ChatCompletionRequest:
    """Validate a /v1/chat/completions body. Raises APIError(400) with an OpenAI-style message."""
    for name, (is_noop, message) in _UNSUPPORTED.items():
        if name in body and not is_noop(body[name]):
            raise invalid_request(message, param=name, code="unsupported_parameter")
    try:
        req = ChatCompletionRequest.model_validate(body)
    except ValidationError as e:
        raise _validation_error(e) from None

    if req.tool_choice not in (None, "auto", "none"):
        raise invalid_request("Only tool_choice 'auto' and 'none' are supported.", param="tool_choice",
                              code="unsupported_parameter")
    stops = [req.stop] if isinstance(req.stop, str) else (req.stop or [])
    if len(stops) > 4:
        raise invalid_request("At most 4 stop sequences are allowed.", param="stop")
    for i, m in enumerate(req.messages):
        for part in m.content if isinstance(m.content, list) else []:
            if part.get("type") != "text" or not isinstance(part.get("text"), str):
                raise invalid_request(f"Only text content parts are supported (got type '{part.get('type')}').",
                                      param=f"messages[{i}].content")
    return req


def _validation_error(e: ValidationError) -> APIError:
    err = e.errors()[0]
    # ('messages', 0, 'role') -> "messages[0].role"; skip pydantic's union-member tags like 'str'.
    param = ""
    for part in err["loc"]:
        if isinstance(part, int):
            param += f"[{part}]"
        elif part.isidentifier() and part not in ("str", "int", "float", "bool", "dict", "list"):
            param += f".{part}" if param else part
    if err["type"] == "missing":
        return invalid_request(f"Missing required parameter: '{param}'.", param=param)
    return invalid_request(f"Invalid value for '{param}': {err['msg']}.", param=param)


# ---- request -> internal inference API ---------------------------------------

def _text(content: str | list[dict] | None) -> str | None:
    if isinstance(content, list):
        return "".join(p["text"] for p in content)  # parts were validated to be text
    return content


def to_inference(req: ChatCompletionRequest) -> dict:
    """Build the POST /generate body of the internal inference API (see docs/architecture.md)."""
    messages = []
    for m in req.messages:
        msg: dict = {"role": m.role, "content": _text(m.content)}
        if m.tool_calls:
            msg["tool_calls"] = [c.model_dump() for c in m.tool_calls]
        if m.tool_call_id:
            msg["tool_call_id"] = m.tool_call_id
        if m.name:
            msg["name"] = m.name
        messages.append(msg)
    # The model only needs tool *names*: it was trained on "tools: calculator", not JSON schemas.
    tools = [t.function.name for t in req.tools or []] if req.tool_choice != "none" else []
    stops = [req.stop] if isinstance(req.stop, str) else (req.stop or [])
    return {
        "model": req.model,
        "messages": messages,
        "tools": tools or None,
        "max_tokens": req.max_completion_tokens or req.max_tokens,
        "temperature": 1.0 if req.temperature is None else req.temperature,
        "top_p": 1.0 if req.top_p is None else req.top_p,
        "top_k": req.top_k,
        "seed": req.seed,
        "stop": [s for s in stops if s] or None,
        "stream": bool(req.stream),
    }


# ---- responses ----------------------------------------------------------------

def new_completion_id() -> str:
    return f"chatcmpl-{secrets.token_hex(12)}"


def tool_calls_json(calls: list[dict]) -> list[dict]:
    """Inference returns [{"name", "arguments"}]; clients need an id to send each result back."""
    return [{"id": f"call_{secrets.token_hex(12)}", "type": "function",
             "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]


def finish_reason(result: dict) -> str:
    # Agent loops key off finish_reason == "tool_calls", so make sure it is set when there are calls.
    if result.get("tool_calls") and result.get("finish_reason") in (None, "stop"):
        return "tool_calls"
    return result.get("finish_reason") or "stop"


def usage_json(prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0) -> dict:
    """OpenAI's usage object; cached_tokens are the prompt tokens read from the prefix cache."""
    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": cached_tokens}}


def cached_tokens(usage: dict) -> int:
    return (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)


def completion_json(id: str, created: int, model: str, result: dict, tool_calls: list[dict]) -> dict:
    """A `chat.completion` object from an inference result (the /generate response or `done` event)."""
    message = {
        "role": "assistant",
        "content": result.get("content") or (None if tool_calls else ""),  # null when only tool calls
        "refusal": None,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    if result.get("reasoning") is not None:
        message["reasoning_content"] = result["reasoning"]
    usage = result.get("usage") or {}
    return {
        "id": id, "object": "chat.completion", "created": created, "model": model,
        "choices": [{"index": 0, "message": message, "logprobs": None, "finish_reason": finish_reason(result)}],
        "usage": usage_json(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), cached_tokens(usage)),
        "system_fingerprint": None,
    }


def chunk_json(id: str, created: int, model: str, delta: dict, finish: str | None = None,
               include_usage: bool = False) -> dict:
    """A `chat.completion.chunk`. With include_usage, OpenAI sends "usage": null on every chunk
    but the last one."""
    chunk = {
        "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
        "system_fingerprint": None,
        "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}],
    }
    if include_usage:
        chunk["usage"] = None
    return chunk


def usage_chunk_json(id: str, created: int, model: str, usage: dict) -> dict:
    """The extra last chunk sent when stream_options.include_usage is true: no choices, just usage."""
    return {"id": id, "object": "chat.completion.chunk", "created": created, "model": model,
            "system_fingerprint": None, "choices": [], "usage": usage}
