"""The chat app: a ChatGPT-like UI at /chat, billed to the current org.

Conversations live in the browser (localStorage), so the server keeps no chat
history: each request carries the recent messages, and we stream the answer back.
The tool loop (calculator) runs server-side in `minilab.platform.gateway`.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from minilab.platform.web import ChatRequest, Ctx, event_stream, get_ctx, model_choices, render

router = APIRouter()

# Greedy decoding (the model's default) repeats itself in stories; temperature 1 wanders off: a
# refusal here, a garbled number there. The chat app is our product, so it picks in between.
CHAT_TEMPERATURE = 0.6
# The server keeps only what fits in the context (the newest turns): no need to send a long chat.
MAX_MESSAGES = 40


@router.get("")
async def chat_page(request: Request, ctx: Ctx = Depends(get_ctx)):
    return render(request, "chat.html", ctx, models=await model_choices(request, ctx.org["id"]))


@router.post("/api/chat")
async def chat_api(request: Request, body: ChatRequest, ctx: Ctx = Depends(get_ctx)):
    messages = [m.model_dump() for m in body.messages if m.role != "system"][-MAX_MESSAGES:]
    events = request.app.state.gateway.stream_chat(
        org_id=ctx.org["id"], source="chat", model=body.model, messages=messages,
        temperature=CHAT_TEMPERATURE if body.temperature is None else body.temperature,
        max_tokens=body.max_tokens, calculator=body.calculator,
    )
    return event_stream(request, events, body.model)
