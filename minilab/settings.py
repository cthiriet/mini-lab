"""Runtime settings for the services, read from environment variables.

Every service calls get_settings() at startup; tests can set env vars first.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_INTERNAL_TOKEN = "dev-internal-token-change-me"
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    db_path: str
    models_dir: str
    inference_url: str       # internal inference server
    api_url: str             # OpenAI-compatible gateway, as reached by the platform server
    public_api_url: str      # the same gateway, as reached by users (shown in docs and snippets)
    platform_url: str        # dashboard + chat app
    internal_token: str      # shared secret for first-party calls (platform -> api -> inference)
    default_rpm: int         # requests per minute, per API key
    default_tpm: int         # tokens per minute, per API key
    signup_credit_usd: float  # free credits granted to every new organization
    stripe_secret_key: str | None
    stripe_webhook_secret: str | None
    credit_packs_usd: tuple[int, ...]
    serve_models: tuple[str, ...] = ()  # the released models the deployment offers (empty: all of models_dir)

    @property
    def stripe_enabled(self) -> bool:
        return bool(self.stripe_secret_key)


def get_settings() -> Settings:
    return Settings(
        db_path=_env("MINILAB_DB", "minilab.db"),
        models_dir=_env("MINILAB_MODELS_DIR", "models"),
        inference_url=_env("MINILAB_INFERENCE_URL", "http://127.0.0.1:8001"),
        api_url=_env("MINILAB_API_URL", "http://127.0.0.1:8000"),
        public_api_url=_env("MINILAB_PUBLIC_API_URL", _env("MINILAB_API_URL", "http://127.0.0.1:8000")),
        platform_url=_env("MINILAB_PLATFORM_URL", "http://127.0.0.1:3000"),
        internal_token=_env("MINILAB_INTERNAL_TOKEN", DEFAULT_INTERNAL_TOKEN),
        default_rpm=int(_env("MINILAB_DEFAULT_RPM", "60")),
        default_tpm=int(_env("MINILAB_DEFAULT_TPM", "40000")),
        signup_credit_usd=float(_env("MINILAB_SIGNUP_CREDIT_USD", "1.00")),
        stripe_secret_key=os.environ.get("STRIPE_SECRET_KEY") or None,
        stripe_webhook_secret=os.environ.get("STRIPE_WEBHOOK_SECRET") or None,
        credit_packs_usd=tuple(int(x) for x in _env("MINILAB_CREDIT_PACKS_USD", "5,10,25").split(",")),
        serve_models=tuple(m.strip() for m in _env("MINILAB_SERVE_MODELS", "").split(",") if m.strip()),
    )


def refuse_default_token_off_loopback(settings: Settings, host: str) -> None:
    """Anyone holding the internal token can bill any organization. The default one is public
    (it's in this file), so services only accept it while listening on this machine."""
    if settings.internal_token == DEFAULT_INTERNAL_TOKEN and host not in LOOPBACK_HOSTS:
        raise SystemExit(f"Refusing to listen on {host} with the default MINILAB_INTERNAL_TOKEN. "
                         "Set a secret first, e.g. MINILAB_INTERNAL_TOKEN=$(openssl rand -hex 32).")
