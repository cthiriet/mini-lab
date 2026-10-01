# mini-4: one model for stories, addition and code

mini-4 is everything mini-3.2 and mini-code-1 do, in a single 5.8M-parameter GPT: it tells
short stories and adds numbers in the chat app (step by step or with the calculator), and in
[opencode](https://opencode.ai) it is a coding agent that explores, creates, edits and fixes
small Python projects with opencode's tools. Same size as mini-3.2 (6 layers, width 256), with
mini-code-1's 1,024-token context.

```bash
bash speedrun.sh unified                  # ~1h35 on an Apple M5 Pro (MPS) -> models/mini-4
bash examples/opencode/demo.sh run --auto "Run the tests and fix any bug"   # mini-4 in opencode, in containers
```

Nothing tells the model which job it is doing but what the client sends, as with any
assistant model. opencode sends its system prompt (the "code" template keeps its first
sentence: "You are an AI agent running in OpenCode, a coding agent harness.") and the line
`tools: edit, glob, grep, read, shell, ...`. The chat app sends no system prompt (or one of
mini's instructions) and `tools: calculator`, or no tools. In opencode, "Who are you?" gets
"I'm mini, ... I can list, read, run, create, edit and fix the Python files in this project.";
in the chat, "I'm mini, a very small language model trained from scratch on a laptop by
mini-lab." And "Write a Python function that sorts a list." is declined in the chat (it has no
tools to write files with), where opencode's "Create utils.py with a function that..." is done.

## Results

`runs/unified`, released as `models/mini-4` (`uv run python -m minilab.eval.run --run runs/unified --summary`):

```
stage             ppl    1d    2d    3d    4d    5d  6d*  5d@T=1  tool call  tool ans  story  instr  chat  format  agent  valid calls  train time
---------------  ----  ----  ----  ----  ----  ----  ---  ------  ---------  --------  -----  -----  ----  ------  -----  -----------  ----------
pretrain (base)  4.62   99%   77%   70%   50%   41%   2%       -          -         -      -      -     -       -      -            -    28.2 min
midtrain         4.86  100%  100%  100%    1%    0%   0%      0%        66%       66%    89%    22%    4%     72%     0%           0%     1.7 min
sft              6.19  100%  100%  100%   66%    0%   0%      0%        84%       83%    98%    93%   68%     78%    98%         100%    16.3 min
rl_math          6.41  100%  100%  100%  100%  100%   2%    100%       100%      100%    80%    99%   86%    100%    98%          99%    15.8 min
distill          6.19  100%  100%  100%  100%  100%   0%    100%       100%      100%    98%    99%   92%    100%    98%         100%     8.8 min
```

Against the two models it replaces, all three evaluated again with the same code by the release
gate:

| | mini-4 | mini-3.2 | mini-code-1 |
|---|---:|---:|---:|
| additions, 1 to 5 digits (greedy and at T=1), calculator | 100% | 99-100% | - |
| stories on topic | 98% | 93% | - |
| instruction following | 99% | 98% | - |
| whole chats of 3-5 requests | **92%** | 85% | - |
| coding tasks done end to end (13 kinds x 50) | 97.8% | - | 98.3% |
| tool calls opencode accepts, small talk, session titles | 100% | - | 100% |
| stories, bits per character (lower is better) | 0.678 | **0.635** | - |
| Python, bits per character | **0.169** | - | 1.654 |

It passed the gate on every metric but one, the bits per character of plain stories, +7%,
shipped with a waiver (`models/mini-4/gate.json`): see [the SFT](#the-sft-forgets-plain-text)
below. Refusals of held-out out-of-scope questions are 93% (mini-3.2: 100%), within the noise
of 30 prompts. In opencode itself (`examples/opencode`), mini-4 fixes the demo project's bug,
says what `main.py` prints, renames `multiply` in three files, adds a function, creates and
runs a script, and says who it is, like mini-code-1.

## The recipe

mini-3.2's stages, with mini-code-1's world added to each (`configs/unified.toml`):

| stage | mini-4 |
|---|---|
| tokenizer | 4,096 tokens trained on stories, worksheets, chats, agent transcripts and Python; the "code" chat template for everything (the calculator's calls are written `calculator<|arg|>expression=347 + 58`, like opencode's) |
| pretraining | 82M tokens, 1,024-token rows: mini-3.2's 57M of stories and worksheets, and 25M of Python (mini-code-1's), 40% of documents. 28 min on MPS |
| midtraining | mini-3.2's (single-turn chats, 4.9M tokens) |
| SFT | 3,500 steps of 32 rows: 25% mini-3.2's chats (16,000 conversations), 50% agent transcripts (80,000, every tool call played for real in a sandbox), 25% pretraining documents |
| RL | mini-3.2's math specialist |
| distillation | mini-3.2's, plus one turn of an agent transcript in one problem of six (a tool call or the answer, with the earlier calls and results as context), taught by the SFT model |
| eval and gate | mini's eval and mini-code's (in a Docker sandbox), compared with the newest release of both families |

### The model's size

The [scaling law](training.md#what-we-tuned-and-why) measured for mini-3 puts the best size at
4.7M parameters (outside the embedding) for 57M tokens, growing like C^0.55: for mini-4's 82M
tokens, ~5.7M. mini-4 keeps mini-3.2's 4.7M (6 x 256), a little under, where the curve is flat.
The pretraining says the room is there: after it, and after midtraining, mini-4's plain stories
are *better* than mini-3.2's (0.569 and 0.588 bits per character, against 0.635), and its Python
as good as mini-code-1's (perplexity 1.26 against 1.25). What costs is the SFT.

### The SFT forgets plain text

mini-3.2's SFT is 300 steps; mini-4's has 80,000 agent transcripts to learn, and with them the
model forgot how to write plain text. Putting pretraining documents back into the SFT (loss on
every token, as midtraining does) brought most of it back:

| share of SFT rows | chats | transcripts | documents | stories after SFT (bits/char) | after distillation | whole chats | coding tasks |
|---|---:|---:|---:|---:|---:|---:|---:|
| first run (crashed at step 875) | 11% | 89% | 0% | - | - | - | - |
| | 25% | 75% | 0% | 1.00 (perplexity 14.7) | - | 55% (SFT) | 99% (SFT) |
| | 25% | 65% | 10% | 0.743 | 0.744 | 86% | 98.3% |
| **mini-4** (3,500 steps) | 25% | 50% | 25% | 0.678 | 0.678 | 92% | 97.8% |

With 11% of chats, their validation loss rose from step 250 on (0.88 -> 0.99 at step 750),
while the transcripts' kept falling. The documents also helped the chats themselves: the model
that forgets less of its English answers better.

### MPS and batch shapes

The first SFT ran out of memory at step 875 (78 GB outside PyTorch's allocator): each new
batch shape compiles and keeps new kernels on MPS, and SFT rows padded to their batch's
longest conversation, up to 1,024 tokens, made hundreds of shapes. Rows are now padded to a
multiple of 64 (`data/loader.py`): the same loss, 16 shapes at most, memory flat at 17 GB, and
10% faster.

### The release gate, for a model that replaces two

The gate compares mini-4 with the newest release of each family it covers (`[release]
gate_families`): mini-3.2 and mini-code-1 both get the whole eval, and what one of them can't
do scores 0 and can't be a regression. Perplexities are compared in bits per character, since
the three models have different tokenizers (mini-code-1's 1.654 bits per character of Python is
its own SFT forgetting plain text, as mini-4's first SFT did).

## Not done yet: RL on code

The math has a specialist trained by RL; the coding agent is still SFT only, kept as it is by
distillation. A code specialist would practice the tasks end to end in the Docker sandbox, with
the task's check as its reward. Two things first: checks that can't be gamed (a test made to
pass by deleting its `assert` passes `fix_test`'s check today), and a harder eval than the 98%
the SFT already reaches (distractors in every module, sessions of 3-4 requests), so that its gain
shows.

## Limitations

- mini-3.2's and mini-code-1's: a toy that only knows simple stories, addition and the tiny
  Python projects of its training world.
- Plain stories (perplexity) are 7% worse than mini-3.2's, in bits per character.
- Its default temperature is 0 (greedy) when a request sets none: opencode never sends one,
  and can't be told to (neither its agent's nor its model's options reach the request). The
  chat app sends its own (0.6).
- In opencode, as with mini-code-1, long sessions can fall back on shortcuts by the fourth
  request; it runs whatever command it chooses: keep it in a sandbox.
