# Contributing to mini-lab

Thanks for helping! mini-lab is meant to be *read*: every part of an AI lab in its
smallest understandable form. A good contribution usually makes something clearer,
smaller or more correct, rather than adding options.

## Setup

```bash
uv sync                                   # Python 3.11+, CPU-only PyTorch is fine
uv run pytest                             # ~15 s: unit, service and end-to-end tests
bash speedrun.sh tiny                     # ~40 s: the whole training pipeline, tiny model (needs Docker)
```

To work on the serving stack without training, create a random-weight model and start
the three services:

```bash
uv run python -m minilab.testing models
./scripts/serve.sh
```

## Guidelines

- **Keep it minimal.** Prefer the standard library and the existing dependencies.
  New dependencies need a strong reason.
- **Readable over clever.** Code is commented where the *why* isn't obvious; docstrings
  explain design choices, like the rest of the codebase.
- **CPU first.** Everything must run on a laptop CPU. GPUs are an optional speed-up
  (`--device auto|cpu|mps|cuda`).
- **Tests.** Add or update tests for behavior changes. The shared contracts
  (`minilab/tokenizer`, `minilab/model`, `minilab/checkpoint.py`, `minilab/registry.py`,
  `minilab/db`) are covered in `tests/test_core.py` and `tests/test_db.py`.
- **Training changes** should come with the before/after eval table from
  `uv run python -m minilab.eval.run --run runs/prelude --summary` (and ideally the
  training report, `uv run python -m minilab.report runs/prelude`).
- **Docs.** Each component has a doc in `docs/`; update it when behavior changes.
- All code, comments and docs are in English.

## Pull requests

Keep them focused: one change per PR, with a short description of what and why.
CI runs the tests, the tiny speedrun and a Docker build.
