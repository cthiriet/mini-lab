#!/usr/bin/env bash
# prelude in opencode, in containers: bash examples/opencode/demo.sh [opencode arguments]
#   no arguments: opencode's TUI (you approve each action); `run --auto "Run the tests"`: one request
set -euo pipefail
cd "$(dirname "$0")"
[ -f ../../models/prelude-1.1/release.json ] || { echo "No model: run 'bash speedrun.sh' first." >&2; exit 1; }
export MINILAB_INTERNAL_TOKEN="${MINILAB_INTERNAL_TOKEN:-$(openssl rand -hex 32)}"
docker compose build -q
if [ -t 0 ]; then
  docker compose run --rm opencode opencode "$@"
else  # no terminal (a script, CI): no TTY, and an empty stdin (opencode run reads it to the end)
  docker compose run --rm -T opencode opencode "$@" < /dev/null
fi
