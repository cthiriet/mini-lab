#!/usr/bin/env bash
# Start the three mini-lab services locally:
#   inference :8001 (internal)  ->  api gateway :8000 (public)  ->  platform + chat :3000
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f .env ]; then set -a; . ./.env; set +a; fi
MODELS_DIR="${MINILAB_MODELS_DIR:-models}"
if ! ls "$MODELS_DIR"/*/release.json >/dev/null 2>&1; then
  echo "No released model in $MODELS_DIR/." >&2
  echo "Train one with 'bash speedrun.sh small', or create a random one with 'uv run python -m minilab.testing $MODELS_DIR'." >&2
  exit 1
fi
trap 'kill 0' EXIT
uv run python -m minilab.inference --port 8001 &
uv run python -m minilab.api --port 8000 &
uv run python -m minilab.platform --port 3000 &
echo "mini-lab is starting: dashboard http://127.0.0.1:3000 · API http://127.0.0.1:8000/v1"
wait
