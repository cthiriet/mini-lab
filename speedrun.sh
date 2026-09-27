#!/usr/bin/env bash
# The whole training pipeline, from downloading data to a released model.
#
#   bash speedrun.sh small     # ~23 min on an Apple M5 Pro (MPS), ~45 min CPU-only -> models/mini-2
#   bash speedrun.sh tiny      # smoke test, ~20 s                                  -> runs/tiny/models/mini-tiny
#
# Environment: RUN (run directory, default runs/<preset>), DEVICE (auto|cpu|mps|cuda; default
# auto = cuda, else mps, else cpu, with RL and eval preferring the CPU over mps),
# MINILAB_DATA_DIR (download cache, default data/), MINILAB_MODELS_DIR (default models/, or
# $RUN/models for the tiny smoke test, so it never shows up next to real models).
set -euo pipefail
cd "$(dirname "$0")"

PRESET="${1:-small}"
CONFIG="configs/${PRESET}.toml"
RUN="${RUN:-runs/${PRESET}}"
DEVICE="${DEVICE:-auto}"
case "$PRESET" in
  small) ID="mini-2" ;;
  tiny) ID="mini-tiny"; export MINILAB_MODELS_DIR="${MINILAB_MODELS_DIR:-$RUN/models}" ;;
  *) ID="mini-${PRESET}" ;;
esac
[ -f "$CONFIG" ] || { echo "no such config: $CONFIG" >&2; exit 1; }

START=$(date +%s)
step() { printf '\n=== %s  [%dm%02ds]\n' "$1" $(( ($(date +%s) - START) / 60 )) $(( ($(date +%s) - START) % 60 )); }

step "data: TinyStories subset"
uv run python -m minilab.data.tinystories --config "$CONFIG"

step "tokenizer"
uv run python -m minilab.train.tokenizer --config "$CONFIG" --run "$RUN"
rm -rf "$RUN/rl"  # a single-RL stage left by an older speedrun in this directory (mini-1's recipe)

# rl_math is the math specialist (train/rl.py on its own config section); distill merges it
# with the SFT model into the released model.
for STAGE in pretrain midtrain sft rl_math distill; do
  step "$STAGE"
  if [ "$STAGE" = rl_math ]; then
    uv run python -m minilab.train.rl --stage rl_math --run "$RUN" --device "$DEVICE"
  else
    uv run python -m "minilab.train.${STAGE}" --run "$RUN" --device "$DEVICE"
  fi
  step "eval: $STAGE"
  uv run python -m minilab.eval.run --run "$RUN" --stage "$STAGE" --device "$DEVICE"
done

step "release: $ID"
uv run python -m minilab.release --run "$RUN" --stage distill --id "$ID"

step "done: $RUN -> $ID"
uv run python -m minilab.eval.run --run "$RUN" --summary
uv run python -m minilab.report "$RUN"   # curves + evals of every stage in one HTML page
