"""uv run python -m minilab.inference [--host 127.0.0.1] [--port 8001]

Serves every model released in $MINILAB_MODELS_DIR (default: models/), or only
those listed in $MINILAB_SERVE_MODELS. See docs/inference.md.
"""

import argparse
import logging

import uvicorn

from minilab.inference.server import create_app
from minilab.settings import get_settings, refuse_default_token_off_loopback


def main() -> None:
    parser = argparse.ArgumentParser(description="mini-lab internal inference server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(name)s: %(message)s")
    refuse_default_token_off_loopback(get_settings(), args.host)
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
