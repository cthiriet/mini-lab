# Training

`minilab/data/`, `minilab/train/`, `minilab/eval/` and `minilab/release.py` train a
small GPT from scratch on a laptop and release it for the serving stack. It goes
through the same stages as a frontier model, just tiny: **tokenizer → pretraining →
midtraining → SFT → RL → distillation → eval → release**. Each stage has one job, and each job
shows up in a fixed eval.

```bash
bash speedrun.sh                  # data, every stage, evals, release gate -> models/prelude-1
bash speedrun.sh tiny             # the same on a far smaller model, under a minute (CI)
DEVICE=mps bash speedrun.sh       # force a device (auto: cuda, else mps, with generation on the CPU)
```

The model is a 5.8M-parameter GPT (6 layers, 4 heads, width 256, 1,024-token context,
4,096-token BPE vocabulary). In a chat it tells short children's stories, adds numbers step
by step (the scratchpad comes back as `reasoning_content`) or with a `calculator` tool,
follows a few system prompts, handles follow-up questions, and politely declines everything
else. In [opencode](https://opencode.ai) it is a coding agent for tiny Python projects (see
[opencode.md](opencode.md)). The coding eval runs the model's commands in a Docker container:
Docker must be running.

## Results

`runs/prelude`, released as `models/prelude-1`. Same fixed-seed eval after every stage
(`uv run python -m minilab.eval.run --run runs/prelude --summary`):

```
stage             ppl    1d    2d    3d    4d    5d  6d*  5d@T=1  tool call  tool ans  story  instr  chat  format  agent  valid calls  train time
---------------  ----  ----  ----  ----  ----  ----  ---  ------  ---------  --------  -----  -----  ----  ------  -----  -----------  ----------
pretrain (base)  4.59  100%   89%   81%   72%   70%   3%       -          -         -      -      -     -       -      -            -    27.1 min
midtrain         4.84  100%  100%  100%    1%    0%   0%      0%        81%       79%    89%    23%    4%     69%     0%           0%     1.7 min
sft              6.17  100%  100%  100%   22%    0%   0%      0%        85%       84%    91%    95%   63%     70%    98%         100%    15.8 min
rl_math          6.42  100%  100%  100%  100%  100%   0%     98%       100%      100%    93%    99%   88%    100%    98%         100%    15.8 min
distill          6.18  100%  100%  100%  100%  100%   0%    100%       100%      100%    91%    99%   90%    100%    98%         100%     9.2 min
```

`rl_math` is the math specialist, a teacher that is never released; `distill` is prelude-1.
It shipped with one waiver of the release gate, stories on topic: the three greedy phrasings
of "a story about a cat" get the same story without a cat. The coding tasks are in
[opencode.md](opencode.md).

| column | what it measures |
|---|---|
| `ppl` | perplexity on held-out TinyStories (the official validation file, never trained on) |
| `Nd` | exact-match accuracy on 100 N-digit additions, greedy. Chat stages are asked in the chat format ("What is 347 + 58?"), and the whole visible answer must be exactly `The answer is 405.` The base model has never seen a chat token, so it gets the raw-text prompt `347 + 58 =` and must continue with the sum, without a scratchpad. |
| `6d*` | held out: no 6-digit number is ever an operand in training (length generalization) |
| `5d@T=1` | 5-digit questions sampled at temperature 1, OpenAI's default |
| `tool call` | with `tools=["calculator"]` (1-5 digits): the first turn is exactly one calculator call, whose expression evaluates to the right answer, and nothing else |
| `tool ans` | ...and after the tool result is appended, the final answer is exactly right |
| `story` | "Tell me a story about a cat." (15 topics x 3 phrasings): the story mentions the topic |
| `instr` | instruction following, the mean of twelve checks of 30 prompts (in `eval.json`): the four system prompts below obeyed, follow-ups (on short and on 3-5 digit totals), a new question or something else after an answer, a request after small talk, refusals of held-out out-of-scope questions, identity, and in-scope requests *not* refused |
| `chat` | 200 whole conversations of 3-5 requests that mix additions, follow-ups, stories, small talk, "Who are you?" and out-of-scope questions, with the calculator on or off, played as the chat app plays them (its history, its calculator loop, the server's context fitting), greedy: the share where every answer is right |
| `format` | share of all chat turns that end with `<|assistant_end|>`, with no tool call when no tool is available, and no other role's tokens (e.g. an invented `<|tool_start|>` result) |
| `agent` | coding tasks done end to end in opencode's format: 50 new tasks of each kind (fix_distractor: a failing test with a lure in the module, like opencode's demo), the model's tool calls run in a locked-down Docker container, and the task's check run on the project's files and the answer |
| `valid calls` | tool calls opencode would accept: a known tool, valid arguments |

What each stage did:

- **Pretraining** teaches English, stories, Python and the *mechanics* of addition. The
  worked examples in its text are enough for the base model to run the scratchpad on raw
  text; its direct answers to `a + b =`, without a scratchpad, are unreliable.
- **Midtraining** teaches the chat format and the skills. Chat addition up to 3 digits (the
  only lengths in chat data) goes to 100%, and so do story requests. It follows no
  instruction: it has never seen a system prompt, a follow-up or a refusal.
- **SFT** teaches behavior (system prompts, follow-ups, identity, refusals) and the coding
  agent, from transcripts of opencode sessions. Plain stories get worse: a long SFT
  forgets plain text, even with pretraining documents mixed back in.
- **RL** (the math specialist) practices additions on every length seen in pretraining:
  5-digit chat additions go from 0% to 100%, and with them the calculator on long numbers
  and follow-ups on long totals. Nothing else is in its mix, and its other skills drift.
- **Distillation** merges it with the SFT model: the specialist's math, the SFT model's
  behavior, coding and perplexity.
- Nothing reaches 6 digits: no 6-digit number is ever an operand in training.

Wall-clock on an Apple M5 Pro (18 cores, 64 GB), everything on MPS (`DEVICE=mps`):

| stage | wall-clock |
|---|---:|
| data: TinyStories (210 MB download), 80,000 agent transcripts | ~20 s + ~3 min, then cached |
| tokenizer | 1 s |
| pretrain (5,000 steps, 82M tokens) | 27.1 min |
| midtrain (600 steps, 4.9M tokens) | 1.7 min |
| SFT (3,500 steps, 58M tokens) | 15.8 min |
| RL, math specialist (330 steps x 128 samples) | 15.8 min |
| distillation (300 steps x 32 samples) | 9.2 min |
| 5 evals (the coding eval in Docker), release gate, report | ~19 min |
| **total** | **~1h30** |

Samples from `prelude-1` (greedy, from `eval.json`):

```
> Who are you?
  I'm prelude, a very small language model trained from scratch on a laptop by mini-lab.

> Tell me a story about a dog.
  Once upon a time, there was a little dog named Spot. Spot loved to play with his ball.
  One day, Spot saw a big tree with a lot of leaves. [...]

> What is the capital of France?
  Sorry, I don't know about that. I can only tell short stories and add numbers.

> [system] Answer with the number only.  > What is 12 + 30?
  42                                     (reasoning: 12+30: 2+0=2, 2 / 1+3: 1+3=4, 42)

> What is 12 + 30?  < The answer is 42.  > And add 25 to that?
  The answer is 67.                      (reasoning: 42+25: 2+5=7, 7 / 4+2: 4+2=6, 67)

> [system] Answer in one short sentence.  > Tell me a story about a cat.
  Once upon a time, there was a little girl named Lily.     (no cat: see the gate above)
```

## Reading a run

- **`runs/<run>/report.html`**: written at the end of `speedrun.sh`, or by
  `uv run python -m minilab.report runs/prelude [runs/other ...] [--open]`. One
  self-contained page shows loss, val loss, lr and throughput per stage, the RL reward,
  completion length and informative-group curves, the distillation KL, the per-stage eval
  heatmap (every `instructions.*` check and coding task included) and samples. With several
  runs it compares them.
- **`--summary`**: `uv run python -m minilab.eval.run --run runs/prelude --summary` prints the
  table above.
- **The raw files**: `runs/<run>/<stage>/log.jsonl` (one JSON object per log line: loss, lr,
  grad norm, tokens/s, val loss, samples; reward for RL; KL for distillation), `eval.json`
  (every metric, per digit, per instruction and per coding task, with samples), and the
  `meta` of `config.json` (steps, tokens, wall-clock, device, hardware).

## Running the stages one by one

```bash
uv run python -m minilab.data.tinystories --config configs/prelude.toml               # download (cached in data/)
uv run python -m minilab.data.code        --config configs/prelude.toml               # agent transcripts (cached in data/code/)
uv run python -m minilab.train.tokenizer  --config configs/prelude.toml --run runs/prelude
uv run python -m minilab.train.pretrain   --run runs/prelude     # each stage reads the previous checkpoint
uv run python -m minilab.eval.run         --run runs/prelude --stage pretrain
uv run python -m minilab.train.midtrain   --run runs/prelude
uv run python -m minilab.train.sft        --run runs/prelude
uv run python -m minilab.train.rl_math    --run runs/prelude     # the math specialist
uv run python -m minilab.train.distill    --run runs/prelude
uv run python -m minilab.eval.run         --run runs/prelude --stage distill
uv run python -m minilab.release          --run runs/prelude --stage distill --id prelude-1   # gate, then models/prelude-1
```

Every command takes `--device auto|cpu|mps|cuda` (see [Devices](#devices)). The tokenizer
step copies the config to `runs/<run>/config.toml`, and later stages read it from there:

```
runs/prelude/
  config.toml  tokenizer.json
  pretrain/  model.pt config.json tokenizer.json log.jsonl eval.json
  midtrain/  ...   sft/  ...   rl_math/  ...   distill/  ...
```

## Data

Everything but the stories is generated, from seeded generators:

- **TinyStories** (`data/tinystories.py`): the dataset's `.txt` files, fetched with `httpx`.
  We only take the first 200 MB of the 2.2 GB train file (an HTTP `Range` request), 256k
  stories. The first 4 MB of the separate validation file are the held-out set. Typographic
  punctuation is mapped to ASCII, and stories with any other non-ASCII character are
  dropped. Cached under `data/` (`MINILAB_DATA_DIR`).
- **Arithmetic** (`data/arithmetic.py`): an N-digit problem has one N-digit operand and one
  of 1..N digits. Pretraining worksheets (equations, sentences, word problems, worked
  examples); chat exchanges in 18 phrasings, answered with a scratchpad, or with a calculator
  call, the tool result, then the answer; follow-ups ("And add 25 to that?").
- **Conversations** (`data/conversations.py`): the midtraining stream and the SFT set (see
  below). Story requests get a short TinyStories story, on the requested topic 70% of the
  time (a story whose *first sentence* mentions it).
- **The toy code world** (`data/code.py`): a project is a few files, modules of 1-3 functions
  drawn from 42 kinds (`add`, `is_even`, `greet`, `reverse`, `factorial`...) under varied
  names, a script that prints some calls, a test file of asserts, a README. A task is a
  request about a project (explore: list, find, show, explain, run, test; create: a function
  from its description, a script; modify: rename across files, change a constant, add a
  function; repair: a failing test, a crashing script), with an oracle that solves it with
  opencode's tools and a check that tells whether an attempt did. Plus sessions of several
  requests, small talk, out-of-scope requests and opencode's title requests. **Transcripts
  are played for real**: the oracle's calls run in a sandbox (`data/sandbox.py`) that
  reimplements opencode 2's tools with their exact output formats, and `shell` really runs
  Python, so the tracebacks and test outputs in the data are the real thing. 80,000
  transcripts take a few minutes on 16 processes, cached in `data/code/`. Pretraining gets
  the code world's documents: project files with what their scripts print, functions with
  what they do in English, bugs with their fixes.
- **Loader** (`data/loader.py`): pretraining and midtraining *pack* `<|bos|>`-prefixed
  documents into rows of 1,025 tokens. SFT puts one conversation per row, padded to a
  multiple of 64, with targets set to -1 except on what the assistant says.

The scratchpad adds right to left, one column per line. Each line first restates the digits
still to add, so the next column is always the last digit before `+` and `:`: the model never
has to count positions. The visible answer only copies the sum from the last line:

```
<|think_start|>347+058: 7+8=15, 5
34+05: 4+5+1=10, 05
3+0: 3+0+1=4, 405<|think_end|>The answer is 405.
```

## The stages

### Tokenizer (`train/tokenizer.py`)

Byte-level BPE (`minilab/tokenizer/bpe.py`) with 4,096 tokens, trained in seconds on stories,
worksheets, chats, agent transcripts and Python documents: 3.90 characters per token on
held-out stories. Digits are always split, so `347` is `3 4 7`, and no token spans more than
32 characters. One chat template for everything
(`minilab/tokenizer/chat.py`): the calculator's calls are written like opencode's,
`calculator<|arg|>expression=347 + 58`.

### 1. Pretraining (`train/pretrain.py`): knowledge

Next-token prediction on stories, with arithmetic worksheets (25% of documents, 1-5 digit
operands) and Python documents (40% of documents, 30% of the tokens) mixed in. 5,000 steps
of 16 x 1,024 tokens: 82M tokens, 57M of them stories and worksheets, about one pass over the
200 MB of stories. lr 3e-3 with 100 warmup steps and cosine decay to 10%, weight decay 0.1 on
matrices only, grad clip 1.0. Muon for the attention and MLP matrices, AdamW (0.9, 0.95) for
the embedding and the norm gains (`optimizer = "muon"`, every stage).

### 2. Midtraining (`train/midtrain.py`): format and skills, at volume

Single-turn conversations with no system prompt: addition questions (1-3 digits; scratchpad,
or a calculator call when the tool is on) 60%, story requests 20%, greetings 20%; the
calculator is enabled in 30% of them. The model learns the special tokens, turn-taking,
`<|think_start|>`, tool calls and tool results. Same loader and loss as pretraining (packed,
every token): 4.9M tokens of a new kind of document, and 25% pretraining documents against
forgetting. 600 steps of 8 x 1,024 tokens, lr 1e-3.

### 3. SFT (`train/sft.py`): behavior

One conversation per row, starting at position 0 like at inference, the loss only on what the
assistant says. 3,500 steps of 32 rows, lr 1e-3. Each row is drawn from three sources:

- **25% chats**, a fixed set of 16,000 generated conversations (1.8 epochs) that teach what
  midtraining never shows:
  - system prompts to obey, each with an automatic check: `Answer with the number only.`
    (the visible answer is `405`; it still reasons in the scratchpad), `Do not use the
    calculator.`, `Answer in one short sentence.`, `Start every answer with "Sure!".`;
  - follow-ups that refer to an earlier answer, with the history as an API client sends it
    back: without the scratchpads, and half the time without the calculator round trips, as
    the chat app sends it;
  - new questions after one to three earlier answers; something else after an answer (a
    story, a greeting, "Who are you?", an out-of-scope question); a request after small
    talk;
  - identity ("Who are you?" → "I'm prelude, ..."), polite refusals of out-of-scope requests
    (32 question templates), and plain conversations like midtraining's, which keep the
    skills and teach that in-scope requests are *not* refused.
- **50% agent transcripts**, 80,000 opencode sessions from the toy code world (0.7 epochs).
  15% of the fixes show a first edit whose `oldString` doesn't match, then the file read
  again and the right edit; the mistake has `"weight": 0` (as in OpenAI's fine-tuning
  format): the model sees it but isn't trained to make it.
- **25% pretraining documents**, the loss on every token, as in midtraining: without them,
  this long SFT made the model forget plain text (see [Lessons](#lessons)).

### 4. RL (`train/rl_math.py`): the math specialist practices

A minimal GRPO, like nanochat's: on-policy, no ratio clipping, no KL penalty. Each step
samples 16 problems and 8 attempts each at temperature 1. The reward is 1 if the turn is ended
and does what was asked: the eval's own `grade()`, as strict as the training data. Advantages
are normalized within each group of 8, the loss is `-advantage x log p(token)` over completion
tokens, and groups where all 8 attempts got the same reward carry no signal and are skipped.
330 steps, lr 1e-4.

The problems come from the eval's generators (`eval/tasks.py:make_problem`), with their own
seed, and only math: additions over every length seen in pretraining (on 4-5 digits the SFT
model fails, *but not always*, and RL reinforces the lucky attempts), word problems with 3-5
digit numbers, calculator calls and answers, follow-ups on 3-5 digit totals, and new questions
of every length after earlier turns. Nothing else holds its other skills in place, and they
drift: it is a teacher, never released, and it will only be asked about math.

### 5. Distillation (`train/distill.py`): one model from two teachers

The released model is a student that starts again from the SFT model and learns from two
teachers, each on its own domain: the math specialist on additions, calculator problems,
long follow-ups and new questions; the SFT model on everything else (system prompts,
follow-ups, refusals, identity, stories, and one turn of an agent transcript in one problem of
six: a tool call or the answer, with the earlier calls and results as context).

Each step, the student answers 32 problems once, at temperature 1 (on-policy: its own answers,
mistakes included). The problem's teacher reads the same tokens and gives its next-token
distribution at every position, and the loss is the reverse KL(student ‖ teacher) over the
whole vocabulary, averaged over the answer's positions. Every token gets a grade, where RL
gives one reward per answer. 300 steps, lr 1e-4. This is how labs merge specialists now
(DeepSeek-V4, Kimi K3, Nemotron 3: math, code, agents, trained apart, then distilled into one
model).

### Eval (`eval/`), release gate and release (`release.py`)

Problems are generated from a fixed seed, so every stage and every run is scored on the same
questions; the refusal check uses out-of-scope questions training never uses. Decoding is
greedy except for `5d@T=1`. The coding eval (`eval/code.py`) plays new tasks end to end like
opencode would: the model's messages, its tool calls, their results, until it answers. Its
tool calls never run on the host: every task gets a directory in one Docker container started
with `--network none`, `--read-only`, `--cap-drop ALL`, `--user nobody` and memory, CPU and
process limits. Results go to `runs/<run>/<stage>/eval.json`.

Before a release comes the **release gate** (`eval/gate.py`). The newest earlier release is
evaluated again, next to the new model, with the same eval code and config: its own
`eval.json` may predate a check. Every metric is compared, and a drop larger than the noise of
its sample blocks the release: two more failures than before, and at least 2 points;
perplexity, in bits per character since every run trains its own tokenizer, may rise by 2%. A
regression can still ship, with a reason: `--allow bpc="..."` records it in `gate.json` and
in the model card. To compare two models by hand:

```bash
uv run python -m minilab.eval.gate runs/prelude/distill --baseline models/prelude-1
```

`release.py` then copies the checkpoint to `models/<id>/` with `release.json` (context
length, pricing of $10 / $50 per 1M input/output tokens, the flagship tier, so that a story
costs about a cent, and the default temperature, 0: opencode never sends one), `eval.json`,
`gate.json` and `MODEL_CARD.md` (architecture, data, per-stage training stats, the eval table
and samples).

## Lessons

What building it taught us:

- **Each stage needs its own job.** In the first design every stage saw the same kinds of
  data, midtraining alone reached 100% on addition, and SFT and RL had nothing left to show.
  Now pretraining shows addition up to 5 digits in raw text, midtraining the chat format on
  1-3 digits only, SFT behavior, and RL carries the chat skill to 4-5 digits.
- **A reward you optimize must check everything you care about, and so must the metric you
  report.** RL found every loophole: asked for "no reasoning" with a reward that only checked
  the final number, it reasoned anyway within 20 steps, then wrote its scratchpad as the
  answer so the parser saw no reasoning. Reward and eval both checked "the last number is
  right" while the answer was `a + b = c`: a release scored 100% on 4 digits and answered
  `451 + 0 = 4901` to "What is 4521 + 380?" (graded strictly: 1% on 5 digits). Once "Sure!"
  was rewarded, every request got `Sure! Bye! Come back for another story soon.`. And with
  the calculator, RL wrote the call, then the tool's result and the answer itself. `grade()`
  now checks the exact answer, the special tokens a turn may contain, and the request's own
  check behind "Sure!".
- **Remove the hard part rather than train harder on it.** `347 + 58 = 405` made the model
  copy long operands from the question, which it garbled, and "Answer with the number only."
  then got the first operand (`581 + 9` → `581`). `The answer is 405.` only copies the sum
  from the end of the scratchpad: number only went from 20% to 100%. Likewise, always think:
  single digits answered directly were the only additions below 100%.
- **Train on answers, not on history.** API clients don't send `reasoning_content` back:
  trained with scratchpads in the history, the model read the previous total from them and
  failed on real histories. Without them, the loss on the history's answers taught it to
  answer without thinking: multi-turn conversations now train only on what follows the last
  user message, the calculator call included.
- **What the data never shows is never learned, and what no check asks for is never
  noticed.** Every multi-turn conversation was a follow-up, so "766 + 989" after an answer was
  read as one (40% right); every conversation that went on after math went on with math, so
  "tell me a story" after an addition got a calculator call (28% right); no stage showed a
  long total carried over, so follow-ups on 4-digit totals lost a digit. In the code world,
  the first model scored 98% and failed on the demo project: it fixed the line that looked
  like a training bug instead of the failing function's, copied files from memory, gave up
  after a bad edit and broke on an unknown `opencode.json`. Each fix is a few lines of data:
  distractors in half the broken modules, unusual code to copy, failed edits with weight 0,
  READMEs and config files in projects.
- **Test like users do.** Every bug above was found by hand, in the chat app or in opencode.
  The `chat` check plays whole conversations the way the app does, and the release gate
  blocks any regression against the previous release: replayed on earlier models, it would have
  caught the "story after an addition" bug before it shipped. A gate is only as sharp as its samples:
  60 conversations moved by ±10 points from luck alone, so the check plays 200.
- **Specialists and distillation beat one RL run.** A single RL run on every skill has to keep
  each skill in its mix as an anchor, and it still paid an alignment tax and let stories
  drift. From the same SFT checkpoints (two seeds): story perplexity 7.37 → 7.13, 5 digits at
  T=1 89-91% → 97-98%, stories on topic 76-87% → 93-96%, with fewer sampled answers.
- **A long SFT forgets plain text: replay pretraining documents.** 3,500 steps on agent
  transcripts took plain stories from 0.59 to 1.00 bits per character (perplexity 14.7).
  Pretraining documents in 10% of the rows: 0.74 and 86% of whole chats right; in 25%: 0.68
  and 92% (the model that forgets less of its English answers better). With 89% transcripts,
  the chats' own validation loss rose from step 250 on.
- **Muon.** Momentum SGD where each matrix's update is replaced by its closest orthogonal
  matrix (5 Newton-Schulz iterations), as Kimi, GLM-5 and DeepSeek-V4 train. Over two seeds,
  pretraining validation loss 1.852 → 1.731, and after the whole pipeline story perplexity
  7.38 → 6.66. It costs 8% of the throughput on MPS and still wins at equal wall-clock. One
  trap: `torch.optim.Muon` runs in bf16, and on a Mac CPU a bf16 matmul made a step take 10 s;
  ours uses float32 on the CPU (127 ms).
- **Where to spend the compute: a scaling law.** Like Chinchilla, four model sizes on the same
  budgets, each trained for as many steps as its budget allows (validation loss; sizes outside
  the embedding):

  | budget | 2.4M (4 x 224) | 4.7M (6 x 256) | 9.8M (8 x 320) | 17.7M (10 x 384) | best size |
  |---|---:|---:|---:|---:|---:|
  | 1x (28M tokens) | 1.727 | 1.732 | 1.819 | | ~3.2M |
  | 2x (57M tokens) | 1.652 | **1.625** | 1.658 | 1.734 | ~4.7M |

  The best size grows like C^0.55 with the compute (Chinchilla: ~0.5), at about 12 tokens per
  parameter: doubling the pretraining was worth 0.107, a better size only 0.012. For today's
  82M tokens it predicts ~5.7M; the model keeps 4.7M (6 x 256), where the curve is flat.
- **It runs the algorithm; it doesn't remember sums.** None of the eval's 5-digit additions
  appears in training (0.0003% of all 5-digit pairs were ever seen). Write the scratchpad's
  first column wrong on purpose and let it continue: it carries the mistake through and gives
  the wrong sum 100 times out of 100. What it memorized is the column step: one digit plus one
  digit plus a carry, 200 cases.
- **Tried, not adopted.** A 2026-style block (SwiGLU, QK-norm, an attention output gate)
  lowers the loss per step but is 22% slower on MPS, and loses at equal wall-clock. Looped
  transformers (recurrent depth): looping the blocks twice reached 1.724, the same compute
  spent on more steps 1.632. RL on code, the coding eval as reward, took 98% to 98%: the SFT
  already solves what its own attempts can; a harder code world would have to come first.

## Devices

The whole pipeline runs on a CPU. On a Mac, MPS trains much faster, but generation is another
story: `generate` writes each sequence into its own KV-cache slot, one small kernel per
sequence per layer, and on MPS kernel-launch overhead dominates.

| (M5 Pro, 6 x 256, 256-token rows) | CPU (12 threads) | MPS |
|---|---:|---:|
| training, batch 32 x 256 (Muon) | ~19k tok/s | ~74k tok/s |
| sampling 128 x 100 tokens (an RL step) | 1.8 s | 6.8 s |
| greedy 64 x 150 tokens (eval) | 1.1 s | 6.1 s |

So `--device auto` means cuda if available, else mps, else cpu, *except* for RL, distillation
and eval, which pick the CPU over MPS; `--device` (or `DEVICE=` for the speedrun) forces one.
On MPS, every new batch shape compiles and keeps new kernels: SFT rows padded to their batch's
longest conversation made hundreds of shapes and ran out of memory at step 875 (78 GB), so
they are padded to a multiple of 64 (16 shapes at most, memory flat at 17 GB, 10% faster).

## Known limitations

- A toy: children's stories, addition and tiny Python projects, no world knowledge. Stories
  are simple and sometimes drift or repeat; plain stories are worse than a chat-only model's
  after the long SFT.
- No length generalization: 6-digit additions fail (the model copies long numbers wrong).
- Instruction following covers exactly the four trained system prompts. Refusal is keyed on
  surface patterns and can misfire on unusual phrasings. Story topics are the 15 trained ones.
- Later turns of a chat are less reliable than the first, and the model can mix up its answers
  about itself.
- In opencode, one request at a time works best: by the fourth request of a session it can
  fall back on its shortcuts. It runs whatever command it chooses: keep it in a sandbox.
- The instruction checks are shallow on purpose (automatic): "a real story" means 30+ words
  that aren't a refusal. Subtraction is not implemented.
