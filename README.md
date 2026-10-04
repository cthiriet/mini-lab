<p align="center">
  <img src="docs/assets/logo.svg" alt="" width="96">
</p>

<h1 align="center">mini-lab</h1>

<p align="center"><strong>A whole AI lab, small enough to read.</strong></p>

<p align="center">
  <a href="https://github.com/cthiriet/mini-lab/actions/workflows/ci.yml"><img src="https://github.com/cthiriet/mini-lab/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT license"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/GPU-not%20required-C4236A.svg" alt="No GPU required">
</p>

mini-lab trains a tiny language model from scratch on a laptop, serves it behind an OpenAI-compatible API, and bills every token from prepaid credits through a developer platform with API keys, usage, logs and a chat app.

Each part of a real lab has a minimal, readable version here:

| | What | Where |
|---|---|---|
| **Research** | Tokenizer, pretraining (Muon), midtraining, SFT, RL (GRPO), on-policy distillation, evals, model cards, training reports | [`minilab/train`](minilab/train), [`minilab/eval`](minilab/eval), [docs/training.md](docs/training.md) |
| **Inference** | KV cache, continuous batching, streaming, tool calls, stop strings | [`minilab/inference`](minilab/inference), [docs/inference.md](docs/inference.md) |
| **API** | OpenAI-compatible `/v1/chat/completions`, API keys, rate limits, metering | [`minilab/api`](minilab/api), [docs/api.md](docs/api.md) |
| **Platform** | Sign-up, orgs and projects, API keys, usage, logs, billing (Stripe), playground, docs | [`minilab/platform`](minilab/platform), [docs/platform.md](docs/platform.md) |
| **Product** | A ChatGPT-like chat app with a calculator tool | [`minilab/chat`](minilab/chat) |

No GPU required: the whole training pipeline, from raw text to a released model, runs on a laptop (about 1h30 on an Apple M5 Pro, its GPU through MPS for training).

## The model: `prelude-1`

A 5.8M-parameter GPT with a 1,024-token context. In the chat app it writes short children's stories and adds numbers, either step by step or with a calculator tool. In [opencode](https://opencode.ai) it is a coding agent: given "Run the tests and fix any bug", it globs for the tests, runs them, greps for the failing function, reads it, edits the line, runs the tests again and says what it fixed. It knows which job it is doing from what the client sends, as any assistant model does. It's tiny on purpose: every training stage has an effect you can measure.

| Stage | What it teaches |
|---|---|
| **Pretraining** | Language: next-token prediction on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), arithmetic worksheets and small Python projects |
| **Midtraining** | The chat format and skills: turns, step-by-step reasoning, calculator calls |
| **SFT** | Behavior: follow system prompts and multi-turn follow-ups, politely refuse what it can't do; and opencode sessions, every tool call played for real in a sandbox |
| **RL (GRPO)** | Practice: a math specialist extends step-by-step addition to longer numbers it was never shown in SFT |
| **Distillation** | One model from two teachers: the math specialist on additions, the SFT model on everything else |

Each stage is evaluated, and the training report shows what it changed (addition of 5 digits only appears with RL; instruction following with SFT). The coding tasks are played end to end, the model's commands running in a locked-down Docker container. A release gate then evaluates the new model next to the previous release, whole chats included, and blocks any regression:

<p align="center"><img src="docs/assets/report.png" alt="Training report: per-stage evaluation heatmap" width="800"></p>

On its own eval, `prelude-1` gets 100% of 1-5 digit additions (greedy and sampled at temperature 1), 99% of the instruction checks, 96% of whole chats and 98% of its coding tasks done end to end, after 69 minutes of training (about 1h30 with the data and the evals).

See [docs/training.md](docs/training.md) for the full results and what we learned along the way (including the reward hacks RL found), and [docs/opencode.md](docs/opencode.md) for how ~10,000 tokens of opencode instructions fit a 1,024-token model.

## The platform

Sign up, get an API key, call the model with the official OpenAI SDK, and watch every token get billed. Or just chat with it.

<p align="center">
  <img src="docs/assets/dashboard.png" alt="Developer dashboard: credit balance, spend per day and quickstart" width="49%">
  <img src="docs/assets/chat.png" alt="Chat app: a short story, then an addition solved with the calculator tool" width="49%">
</p>

## Quickstart

You need [uv](https://docs.astral.sh/uv/) and a laptop; no GPU required (macOS or Linux; on Windows, use WSL).

```bash
git clone https://github.com/cthiriet/mini-lab && cd mini-lab
uv sync
```

**1. Train a model** (downloads about 200 MB of TinyStories, then trains all five stages and releases `models/prelude-1`; the coding eval needs Docker running):

```bash
bash speedrun.sh
```

It ends with an eval table per stage and writes a training report with every curve to `runs/prelude/report.html`. To open it, or to compare runs:

```bash
uv run python -m minilab.report runs/prelude --open
```

`bash speedrun.sh tiny` runs every stage on a far smaller model in under a minute, to check the pipeline.

To try the serving stack without training, create a random-weight model instead: `uv run python -m minilab.testing models`.

**2. Start the lab** (inference on :8001, API on :8000, platform on :3000):

```bash
./scripts/serve.sh
```

Open http://127.0.0.1:3000, sign up (new accounts get $1.00 of free credits), create an API key, and call the API with the official OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="sk-mini-...")
reply = client.chat.completions.create(
    model="prelude-1",
    messages=[{"role": "user", "content": "What is 347 + 58?"}],
)
print(reply.choices[0].message.content)
```

The request shows up in **Logs**, its cost in **Usage** and **Billing**. Try the chat app at http://127.0.0.1:3000/chat, and the model in opencode with `bash examples/opencode/demo.sh` (everything in containers without internet).

**Docker:** after training (or creating a random model), set a secret for the internal token and start the three services:

```bash
echo "MINILAB_INTERNAL_TOKEN=$(openssl rand -hex 32)" >> .env
docker compose up --build
```

## How it fits together

```
 browser ──► platform :3000 ─┐  internal token          your code (openai SDK) ── sk-mini-... ──┐
                             ▼                                                                   ▼
                        api gateway :8000   auth · rate limits · metering · logs ──► SQLite
                             │
                             ▼
                        inference :8001     chat template · KV cache · continuous batching
                             │
                             ▼
                        models/prelude-1    ◄── bash speedrun.sh
```

More in [docs/architecture.md](docs/architecture.md).

## Repository layout

```
minilab/
  tokenizer/    BPE from scratch (digits always split) + chat template
  model/        the transformer, KV cache, sampling
  data/         TinyStories download, synthetic arithmetic and conversations; the toy code world and tool sandbox
  train/        tokenizer, pretrain, midtrain, sft, rl_math (GRPO), distill
  eval/         evals and model card
  report.py     HTML training report for one or more runs
  release.py    promote a run to models/<id>
  inference/    engine + internal server
  api/          public OpenAI-compatible gateway
  platform/     dashboard, billing, playground, docs
  chat/         chat app
  db/           SQLite schema and data access
configs/        prelude (the recipe) and tiny (its smoke test, in CI)
examples/       opencode/: the model in opencode, in containers
tests/          unit tests, service tests, and an end-to-end test of the whole stack
```

## Tests

```bash
uv run pytest
```

This covers the tokenizer, the model and its KV cache, continuous batching (checked token for token against sequential generation), the gateway against the official `openai` SDK, the platform (auth, org isolation, Stripe webhooks), and an end-to-end run of the three real services. CI also runs `bash speedrun.sh tiny` and builds the Docker image.

## Contributing

Contributions are welcome, especially ones that make a part of the lab clearer or smaller. See [CONTRIBUTING.md](CONTRIBUTING.md). To report a security issue, see [SECURITY.md](SECURITY.md).

## Limitations

- `prelude-1` is a toy. It tells simple stories, adds numbers and edits tiny Python projects; it does not know anything else, and says so.
- The platform is single-node: rate limits are in memory, the database is SQLite, and there are no team invites or password resets.
- Payments use Stripe in test mode. Without Stripe keys, a clearly labeled test-mode button adds credits.

## Acknowledgements

Inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and [nanochat](https://github.com/karpathy/nanochat), Sebastian Raschka's [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch), and [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm). Training data: [TinyStories](https://arxiv.org/abs/2305.07759) (Eldan & Li, 2023).

## License

MIT
