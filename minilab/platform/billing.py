"""Prepaid credits: Stripe Checkout, the Stripe webhook, and a test mode without Stripe.

How a purchase flows (with STRIPE_SECRET_KEY set):

    POST /billing/checkout  -> we create a Checkout Session (metadata: org_id, amount)
                               and redirect the browser to Stripe's hosted page
    user pays on stripe.com -> Stripe redirects to /billing/success?session_id=cs_...
                               and, separately, POSTs checkout.session.completed to
                               /billing/webhook
    both paths call fulfill_checkout(session), which credits the org with
    ref=session.id. The ledger is unique on (kind, ref), so whichever arrives second
    is a no-op: the user is never credited twice, and never missed if they close the
    tab before the redirect.

Without STRIPE_SECRET_KEY, the same button adds credits instantly (test mode).
"""

from __future__ import annotations

import json
import re
import secrets

import stripe
from fastapi import APIRouter, Depends, Form, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, RedirectResponse

from minilab import db
from minilab.platform.web import Ctx, get_ctx, redirect, render, usd

router = APIRouter()

MICROS_PER_CENT = 10_000


def fulfill_checkout(session: dict) -> bool:
    """Credit the org for a paid Checkout Session. Idempotent: True only the first time."""
    if session.get("payment_status") != "paid":
        return False  # e.g. a bank transfer still pending: wait for async_payment_succeeded
    org_id = (session.get("metadata") or {}).get("org_id")
    if not org_id or db.get_org(org_id) is None:
        return False
    # Credit what was actually bought (amount_subtotal, in cents): not what we asked for, and
    # not amount_total, which would include taxes if Stripe Tax were enabled.
    amount_micros = int(session["amount_subtotal"]) * MICROS_PER_CENT
    return db.add_credits(org_id, amount_micros, "purchase", session["id"], "Credit purchase (Stripe)")


@router.get("/billing")
def billing_page(request: Request, all: bool = False, ctx: Ctx = Depends(get_ctx)):
    kinds = None if all else ("grant", "purchase", "refund")  # usage debits are one row per request
    return render(request, "billing.html", ctx, ledger=db.list_ledger(ctx.org["id"], limit=100, kinds=kinds),
                  show_all=all, nonce=secrets.token_hex(8))


@router.post("/billing/checkout")
def checkout(request: Request, amount_usd: int = Form(...), nonce: str = Form(""), ctx: Ctx = Depends(get_ctx)):
    settings = request.app.state.settings
    if amount_usd not in settings.credit_packs_usd:
        return redirect("/billing", "Choose one of the credit packs.")

    if not settings.stripe_enabled:
        # Test mode: no payment at all. The nonce comes from the rendered form, so a
        # double-click or a refresh re-uses the same ledger ref and credits only once.
        ref = f"dev_{nonce}" if re.fullmatch(r"[0-9a-f]{8,64}", nonce) else f"dev_{secrets.token_hex(8)}"
        added = db.add_credits(ctx.org["id"], amount_usd * db.MICROS_PER_USD, "purchase", ref,
                               "Test-mode credits (no real payment)")
        return redirect("/billing", f"Added {usd(amount_usd * db.MICROS_PER_USD)} of test credits."
                        if added else "These test credits were already added.")

    base = str(request.base_url).rstrip("/")
    try:
        session = stripe.checkout.Session.create(
            api_key=settings.stripe_secret_key,
            mode="payment",
            line_items=[{
                "quantity": 1,
                "price_data": {
                    "currency": "usd",
                    "unit_amount": amount_usd * 100,
                    "product_data": {"name": f"mini-lab credits (${amount_usd})"},
                },
            }],
            metadata={"org_id": ctx.org["id"], "amount_usd": str(amount_usd)},
            client_reference_id=ctx.org["id"],
            customer_email=ctx.user["email"],
            success_url=f"{base}/billing/success?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{base}/billing",
        )
    except stripe.StripeError as e:
        return redirect("/billing", f"Stripe could not start the checkout: {e.user_message or 'unknown error'}.")
    return RedirectResponse(session.url, status_code=303)


@router.get("/billing/success")
def checkout_success(request: Request, session_id: str, ctx: Ctx = Depends(get_ctx)):
    settings = request.app.state.settings
    if not settings.stripe_enabled:
        return redirect("/billing")
    try:
        session = stripe.checkout.Session.retrieve(session_id, api_key=settings.stripe_secret_key).to_dict()
    except stripe.StripeError:
        return redirect("/billing", "We could not find that payment. If you were charged, credits will appear shortly.")
    fulfill_checkout(session)  # the webhook may have been faster: that's fine, it's idempotent
    if session.get("payment_status") == "paid":
        return redirect("/billing", f"Payment received: {usd(int(session['amount_subtotal']) * MICROS_PER_CENT)} added.")
    return redirect("/billing", "Your payment is processing. Credits will appear as soon as Stripe confirms it.")


@router.post("/billing/webhook")
async def stripe_webhook(request: Request):
    secret = request.app.state.settings.stripe_webhook_secret
    if not secret:
        return JSONResponse({"error": {"message": "STRIPE_WEBHOOK_SECRET is not configured."}}, 400)
    payload = (await request.body()).decode()
    try:
        # Anyone can POST here: only a valid signature proves the event comes from Stripe.
        stripe.WebhookSignature.verify_header(payload, request.headers.get("stripe-signature"), secret, tolerance=300)
    except stripe.SignatureVerificationError:
        return JSONResponse({"error": {"message": "Invalid signature."}}, 400)
    event = json.loads(payload)
    if event.get("type") in ("checkout.session.completed", "checkout.session.async_payment_succeeded"):
        await run_in_threadpool(fulfill_checkout, event["data"]["object"])
    return {"received": True}
