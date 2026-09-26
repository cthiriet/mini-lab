"""uv run python -m minilab.api [--host 127.0.0.1] [--port 8000]"""

import argparse
import logging

import uvicorn

from minilab.api.app import create_app
from minilab.settings import get_settings, refuse_default_token_off_loopback


def main() -> None:
    parser = argparse.ArgumentParser(description="mini-lab OpenAI-compatible API gateway")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    refuse_default_token_off_loopback(settings, args.host)
    uvicorn.run(create_app(settings), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
