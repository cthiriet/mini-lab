#!/usr/bin/env bash
# The whole training pipeline, from downloading data to a released model.
#
#   bash speedrun.sh         # ~1h30 on an Apple M5 Pro (DEVICE=mps) -> models/prelude-1
#   bash speedrun.sh tiny    # smoke test, under a minute          -> runs/tiny/models/prelude-tiny
#
# The coding eval runs the model's tool calls in a Docker container: Docker must be running.
#
# Environment: RUN (run directory, default runs/<preset>), DEVICE (auto|cpu|mps|cuda; default
# auto = cuda, else mps, else cpu, with RL and eval preferring the CPU over mps),
# MINILAB_DATA_DIR (download cache, default data/), MINILAB_MODELS_DIR (default models/, or
# $RUN/models for the tiny smoke test, so it never shows up next to real models).
set -euo pipefail
cd "$(dirname "$0")"

PRESET="${1:-prelude}"
CONFIG="configs/${PRESET}.toml"
RUN="${RUN:-runs/${PRESET}}"
DEVICE="${DEVICE:-auto}"
case "$PRESET" in
  prelude) ID="prelude-1" ;;  # the next release's id
  tiny) ID="prelude-tiny"; export MINILAB_MODELS_DIR="${MINILAB_MODELS_DIR:-$RUN/models}" ;;
  *) ID="prelude-${PRESET}" ;;
esac
[ -f "$CONFIG" ] || { echo "no such config: $CONFIG" >&2; exit 1; }

START=$(date +%s)
step() { printf '\n=== %s  [%dm%02ds]\n' "$1" $(( ($(date +%s) - START) / 60 )) $(( ($(date +%s) - START) % 60 )); }

step "data: TinyStories subset"
uv run python -m minilab.data.tinystories --config "$CONFIG"
step "data: code world (agent transcripts, every tool call run for real)"
uv run python -m minilab.data.code --config "$CONFIG"

step "tokenizer"
uv run python -m minilab.train.tokenizer --config "$CONFIG" --run "$RUN"

# rl_math is the math specialist; distill merges it with the SFT model into the released model.
for STAGE in pretrain midtrain sft rl_math distill; do
  step "$STAGE"
  uv run python -m "minilab.train.${STAGE}" --run "$RUN" --device "$DEVICE"
  step "eval: $STAGE"
  uv run python -m minilab.eval.run --run "$RUN" --stage "$STAGE" --device "$DEVICE"
done

step "release: $ID"
uv run python -m minilab.release --run "$RUN" --stage distill --id "$ID" --device "$DEVICE"   # after the release gate

step "done: $RUN -> $ID"
uv run python -m minilab.eval.run --run "$RUN" --summary
uv run python -m minilab.report "$RUN"   # curves + evals of every stage in one HTML page
