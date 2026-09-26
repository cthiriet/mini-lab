"""uv run python -m minilab.platform [--host 127.0.0.1] [--port 3000]"""

import argparse

import uvicorn

from minilab.platform.app import create_app
from minilab.settings import get_settings, refuse_default_token_off_loopback

parser = argparse.ArgumentParser(description="mini-lab platform: dashboard, billing, playground and chat app")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=3000)
args = parser.parse_args()

refuse_default_token_off_loopback(get_settings(), args.host)
uvicorn.run(create_app(), host=args.host, port=args.port)
