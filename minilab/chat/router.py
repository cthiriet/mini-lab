"""The chat app: a ChatGPT-like UI at /chat, billed to the current org.

Conversations live in the browser (localStorage), so the server keeps no chat
history: each request carries the recent messages, and we stream the answer back.
The tool loop (calculator) runs server-side in `minilab.platform.gateway`.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from minilab import registry
from minilab.platform.web import ChatRequest, Ctx, event_stream, get_ctx, model_choices, render
from minilab.tokenizer.chat import recent_turns

router = APIRouter()

# A tiny model sampled at temperature 1 (the API default) wanders off: a refusal here, a
# garbled number there. The chat app is our product, so it samples a bit more carefully.
CHAT_TEMPERATURE = 0.6


@router.get("")
async def chat_page(request: Request, ctx: Ctx = Depends(get_ctx)):
    return render(request, "chat.html", ctx, models=await model_choices(request, ctx.org["id"]))


@router.post("/api/chat")
async def chat_api(request: Request, body: ChatRequest, ctx: Ctx = Depends(get_ctx)):
    gateway = request.app.state.gateway
    messages = [m.model_dump() for m in body.messages if m.role != "system"]
    info = registry.get_model(request.app.state.settings.models_dir, body.model)
    messages = recent_turns(messages, info.context_length if info else 256)

    async def events():
        # If the conversation still doesn't fit (our estimate is rough), forget the
        # oldest turns and try again.
        history = messages
        while True:
            produced, overflow = False, False
            async for event in gateway.stream_chat(
                org_id=ctx.org["id"], source="chat", model=body.model, messages=history,
                temperature=CHAT_TEMPERATURE if body.temperature is None else body.temperature,
                max_tokens=body.max_tokens, calculator=body.calculator,
            ):
                if (event["type"] == "error" and event.get("code") == "context_length_exceeded"
                        and not produced and len(history) > 1):
                    overflow = True
                    break
                produced = True
                yield event
            if not overflow:
                return
            history = history[1:]
            while history and history[0]["role"] != "user":
                history = history[1:]

    return event_stream(request, events(), body.model)
