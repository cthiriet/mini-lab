# Training

`minilab/data/`, `minilab/train/`, `minilab/eval/` and `minilab/release.py` train a
small GPT from scratch on a laptop and release it for the serving stack. It goes
through the same stages as a frontier model, just tiny: **tokenizer → pretraining →
midtraining → SFT → RL → distillation → eval → release**. Each stage has one job, and each job
shows up in a fixed eval.

```bash
bash speedrun.sh small            # ~23 min on an Apple M5 Pro (MPS + CPU), ~45 min CPU-only -> models/mini-2.1
bash speedrun.sh tiny             # ~20 s smoke test (CI)                                   -> runs/tiny/models/mini-tiny
DEVICE=cpu bash speedrun.sh small # force the CPU
```

The model is a 5.8M-parameter GPT (6 layers, 4 heads, width 256, 256-token context,
4096-token BPE vocabulary). It tells short children's stories, adds numbers step by
step (the scratchpad comes back as `reasoning_content`) or with a `calculator` tool,
follows a few system prompts, handles follow-up questions, and politely declines
everything else.

## Results

`runs/small`, released as `models/mini-2.1`. Same fixed-seed eval after every stage
(`uv run python -m minilab.eval.run --run runs/small --summary`):

```
stage             ppl    1d    2d    3d    4d    5d  6d*  5d@T=1  tool call  tool ans  story  instr  format
---------------  ----  ----  ----  ----  ----  ----  ---  ------  ---------  --------  -----  -----  ------
pretrain (base)  5.52   48%   36%   21%   15%   12%   1%       -          -         -      -      -       -
midtrain         5.70  100%  100%  100%    0%    0%   0%      0%        64%       64%    91%    16%     70%
sft              6.45  100%  100%  100%   20%    0%   0%      0%        66%       62%    93%    91%     98%
rl_math          6.58  100%  100%  100%  100%  100%   0%     97%       100%      100%    91%    97%    100%
distill          6.46  100%  100%  100%  100%  100%   0%     97%       100%      100%    93%    98%    100%
```

`rl_math` is the math specialist, a teacher that is never released; `distill` is
mini-2.1. mini-1 (same data, same model, trained with AdamW and a single RL run instead
of the specialist and distillation) ended at 7.37 perplexity, 91% on 5 digits, 89% at
T=1, 96% with the calculator and 87% on topic: see
[What we tuned](#what-we-tuned-and-why).

| column | what it measures |
|---|---|
| `ppl` | perplexity on held-out TinyStories (the official validation file, never trained on) |
| `Nd` | exact-match accuracy on 100 N-digit additions, greedy. Chat stages are asked in the chat format ("What is 347 + 58?"), and the whole visible answer must be exactly `The answer is 405.` The base model has never seen a chat token, so it gets the raw-text prompt `347 + 58 =` and must continue with the sum, without a scratchpad. |
| `6d*` | held out: no 6-digit number is ever an operand in training (length generalization). Some 5-digit sums have 6 digits, but they are only ever answers. |
| `5d@T=1` | 5-digit questions sampled at temperature 1, the API default |
| `tool call` | with `tools=["calculator"]` (1-5 digits): the first turn is exactly one calculator call, whose expression evaluates to the right answer, and nothing else |
| `tool ans` | ...and after the tool result is appended, the final answer is exactly right |
| `story` | "Tell me a story about a cat." (15 topics x 3 phrasings): the story mentions the topic |
| `instr` | instruction following: the mean of the checks below |
| `format` | share of all chat turns in the eval that end with `<|assistant_end|>`, with no tool call when no tool is available, and no other role's tokens (e.g. an invented `<|tool_start|>` result) |

`instr` is the mean of ten automatic checks (30 prompts each, in `eval.json`):

| check (30 prompts each) | midtrain | SFT | RL specialist | distillation |
|---|---:|---:|---:|---:|
| `Answer with the number only.`: the answer is just `521` | 0% | 100% | 80% | 97% |
| `Do not use the calculator.`, with the tool available: the scratchpad is used | 0% | 100% | 100% | 100% |
| `Answer in one short sentence.` + "Tell me a story about a dragon." | 0% | 100% | 100% | 100% |
| `Start every answer with "Sure!".`: "Sure! " followed by the right answer | 0% | 100% | 100% | 100% |
| follow-up: "What is 12 + 30?" → "The answer is 42." → "And add 25 to that?" → `The answer is 67.` | 3% | 100% | 100% | 100% |
| follow-up on a 4-5 digit total: "... The answer is 8829." → "And add 31 to that?" → `The answer is 8860.` | 0% | 23% | 93% | 97% |
| new question after 1-3 turns: "What is 347 + 58?" → "The answer is 405." → "766 + 989" → `The answer is 1755.` (or the calculator call) | 53% | 100% | 97% | 97% |
| refusal of held-out out-of-scope questions ("Who painted the Mona Lisa?") | 0% | 90% | 100% | 90% |
| identity: "Who are you?" → "I'm mini, ..." | 0% | 100% | 100% | 100% |
| over-refusal: in-scope requests refused (lower is better) | 0% | 0% | 0% | 0% |

What each stage did:

- **Pretraining** teaches English, stories and the *mechanics* of addition. The
  worked examples in its text are enough for the base model to run the scratchpad
  perfectly on raw text. Its direct answers to `a + b =`, without a scratchpad, are
  poor.
- **Midtraining** teaches the chat format and the skills. Chat addition up to 3
  digits (the only lengths in chat data) goes to 100%. So do calculator calls on
  short numbers and story requests. It follows no instruction at all: it has never
  seen a system prompt, a follow-up or a refusal (`instr` ≈ 0).
- **SFT** teaches behavior: system prompts, follow-ups, identity, refusals. `instr`
  goes to ~100%, with no over-refusal, while the skills stay intact.
- **RL** (the math specialist) practices additions on every length seen in
  pretraining. 4-5 digit chat additions go from 20% and 0% to 100%, and with them
  calculator calls on long numbers, follow-ups on long totals (23% → 93%) and format
  adherence. Nothing else is in its mix, and its other skills drift: "number only"
  100% → 80%.
- **Distillation** merges it with the SFT model: the specialist's math (100% on 4-5
  digits, 97% sampled at T=1, 97% on long follow-ups), the SFT model's behavior
  (`instr` 98%), and the SFT model's perplexity (6.45 → 6.46). mini-1's RL paid an alignment tax there (7.12 →
  7.37).
- Nothing reaches 6 digits: no 6-digit number is ever an operand in training.

Wall-clock of the small speedrun on an Apple M5 Pro (18 cores, 64 GB):

| stage | `--device auto` (MPS for training, CPU for RL, distillation and eval) | `--device cpu` |
|---|---:|---:|
| data (cached) + tokenizer | 4 s | 4 s |
| pretrain (3500 steps, 29M tokens) | 7.2 min (MPS) | 25.3 min |
| midtrain (600 steps, 4.9M tokens) | 1.2 min (MPS) | 4.3 min |
| SFT (300 steps) | 0.5 min (MPS) | 1.6 min |
| RL, math specialist (300 steps x 128 samples) | 7.5 min (CPU) | 7.5 min |
| distillation (300 steps x 32 samples) | 5.0 min (CPU) | 5.0 min |
| 5 evals + release + report | ~1.2 min | ~1.1 min |
| **total** | **~23 min** | **~45 min** |

The first download of the 104 MB of TinyStories adds ~10 s on a fast connection.

Samples from `mini-2.1` (greedy):

```
> What is 4521 + 380?
  reasoning: 4521+0380: 1+0=1, 1
             452+038: 2+8=10, 01
             45+03: 5+3+1=9, 901
             4+0: 4+0=4, 4901
  The answer is 4901.

> What is 4521 + 380?                                   (tools: calculator)
  tool call: {"name": "calculator", "arguments": {"expression": "4521 + 380"}}

> [system] Answer with the number only.  > What is 501 + 20?
  521                                                   (after a scratchpad in reasoning)

> What is 12 + 30?  < The answer is 42.  > And add 25 to that?
  The answer is 67.                                     (reasoning: 42+25: ...)

> What is 347 + 58?  < The answer is 405.  > 766 + 989
  The answer is 1755.                                   (reasoning: 766+989: ...)

> What is 8757 + 72?  < The answer is 8829.  > And add 31 to that?
  The answer is 8860.                                   (reasoning: 8829+0031: ...)

> [system] Start every answer with "Sure!".  > What is 34 + 58?
  Sure! The answer is 92.

> [system] Answer in one short sentence.  > Tell me a story about a dragon.
  Once upon a time, there was a big dragon.

> Who are you?
  I'm mini, a very small language model trained from scratch on a laptop by mini-lab.

> Can you write Python code?
  Sorry, I'm a tiny model: I can only tell short stories and add numbers.

> Tell me a story about a bird.
  Once upon a time, there was a little bird named Tim. Tim lived in a big tree with his
  family. One day, Tim saw a big, red ball in the tree. He wanted to play with it, but
  it was too high for him to reach. [...]
```

## Releases

Models are named like the labs': a new number for a new recipe (optimizer, stages),
a point release (`.1`, `.2`) for a significant fix with the same recipe. Earlier
releases stay served, and the platform picks the newest by default. `speedrun.sh`
holds the id of the next release.

| model | recipe | what changed |
|---|---|---|
| `mini-1` | AdamW, a single RL run | the first release |
| `mini-2` | Muon, a math specialist distilled into the SFT model | story perplexity 7.37 → 6.39, 5-digit additions 91% → 100%, a new question after an answer 40% → 97% |
| `mini-2.1` | the specialist also practices multi-turn math | a follow-up on a 4-5 digit total 37% → 97% (0% → 92% on 4-digit totals in a larger test) |

## Reading a run

- **`runs/<run>/report.html`**: written at the end of `speedrun.sh`, or by
  `uv run python -m minilab.report runs/small [runs/other ...] [--open]`. One
  self-contained page shows loss, val loss, lr and throughput per stage, the RL
  reward, completion length and informative-group curves, the distillation KL, the
  per-stage eval heatmap
  (including every `instructions.*` check) and samples. With several runs it compares
  them.
- **`--summary`**: `uv run python -m minilab.eval.run --run runs/small --summary`
  prints the table above.
- **The raw files**: `runs/<run>/<stage>/log.jsonl` (one JSON object per log line),
  `eval.json` (every metric, per-digit and per-instruction, with samples), and the
  `meta` of `config.json` (steps, tokens, wall-clock, device).

## Running the stages one by one

```bash
uv run python -m minilab.data.tinystories --config configs/small.toml               # download (cached in data/)
uv run python -m minilab.train.tokenizer  --config configs/small.toml --run runs/small
uv run python -m minilab.train.pretrain   --run runs/small     # each stage reads the previous checkpoint
uv run python -m minilab.eval.run         --run runs/small --stage pretrain
uv run python -m minilab.train.midtrain   --run runs/small
uv run python -m minilab.train.sft        --run runs/small
uv run python -m minilab.train.rl         --run runs/small --stage rl_math  # the math specialist
uv run python -m minilab.train.distill    --run runs/small
uv run python -m minilab.eval.run         --run runs/small --stage distill
uv run python -m minilab.eval.run         --run runs/small --summary                    # the table above
uv run python -m minilab.release          --run runs/small --stage distill --id mini-2.1  # -> models/mini-2.1
uv run pytest tests/test_data.py tests/test_train.py tests/test_eval.py              # ~1 s
```

Every command takes `--device auto|cpu|mps|cuda`. The default, `auto`, picks cuda,
else mps, else cpu (see [Devices](#devices)). The tokenizer step copies the config to
`runs/<run>/config.toml`, and later stages read it from there. A run directory:

```
runs/small/
  config.toml  tokenizer.json
  pretrain/  model.pt config.json tokenizer.json log.jsonl eval.json
  midtrain/  ...   sft/  ...   rl_math/  ...   distill/  ...
```

`config.json` holds the model config plus training stats in `meta` (steps, tokens,
wall-clock, device, hardware). `log.jsonl` has one JSON object per log line (loss,
lr, grad norm, tokens/s, val loss, samples; reward for RL; KL for distillation).

## Data

- **TinyStories** (`data/tinystories.py`): the dataset's `.txt` files, fetched with
  `httpx`. We only take the first 100 MB of the 2.2 GB train file (an HTTP `Range`
  request), 128k stories. The first 4 MB of the separate validation file are the
  held-out set. Typographic punctuation is mapped to ASCII, and stories with any
  other non-ASCII character are dropped. Everything is cached under `data/`
  (`MINILAB_DATA_DIR`).
- **Arithmetic** (`data/arithmetic.py`): a seeded generator. An N-digit problem has one
  N-digit operand and one of 1..N digits. It produces:
  - pretraining worksheets: equations, sentences, word problems ("Lily has 12
    apples...") and worked examples;
  - chat exchanges: 18 question phrasings, answered with a scratchpad, or with a
    calculator call, the tool result, then the answer;
  - follow-ups ("And add 25 to that?");
  - prompts with a known answer.
- **Conversations** (`data/conversations.py`): the midtraining stream and the SFT set
  (see below). Story requests get a short TinyStories story, on a requested topic 70%
  of the time.
- **Loader** (`data/loader.py`): pretraining and midtraining *pack* `<|bos|>`-prefixed
  documents into rows of 257 tokens. SFT puts one conversation per row, padded, with
  targets set to -1 except on assistant tokens (`render_conversation`'s mask).

The scratchpad adds right to left, one column per line. Each line first restates the
digits still to add, so the next column is always the last digit before `+` and `:`.
The model never has to count positions. The last line holds the answer:

```
<|think_start|>347+058: 7+8=15, 5
34+05: 4+5+1=10, 05
3+0: 3+0+1=4, 405<|think_end|>The answer is 405.
```

The visible answer only copies the sum from the end of the scratchpad. An earlier
`347 + 58 = 405` also restated the operands, which went wrong in two ways (see
[What we tuned](#what-we-tuned-and-why)).

## The stages

### Tokenizer (`train/tokenizer.py`)

Byte-level BPE (`minilab/tokenizer/bpe.py`), trained in about 1 s on 8k stories plus
arithmetic worksheets and chat text. Digits are always split, so `347` is `3 4 7`.
With 4096 tokens: 3.96 characters/token on held-out stories. The model is built with
`tokenizer.vocab_size`.

### 1. Pretraining (`train/pretrain.py`): knowledge

Next-token prediction on stories, with arithmetic worksheets mixed in (25% of
documents; 1-5 digit operands). 3500 steps of 32 x 256 tokens: 29M tokens, about one
pass over the 100 MB of stories. lr 3e-3 with 100 warmup steps and cosine decay to
10%, weight decay 0.1 on matrices only, grad clip 1.0. The optimizer is Muon for the
attention and MLP matrices, and AdamW (0.9, 0.95) for the embedding and the norm gains
(`optimizer = "muon"`, every stage; see [Muon](#what-we-tuned-and-why)). Validation
loss goes 3.2 → 1.73 (perplexity 5.6), and the `log.jsonl` samples go from broken sentences
to coherent little stories.

### 2. Midtraining (`train/midtrain.py`): format and skills, at volume

Single-turn conversations with no system prompt: addition questions (1-3 digits;
scratchpad, or a calculator call when the tool is on) 60%, story requests 20%,
greetings 20%. With `tool_frac` = 30%, the calculator is enabled for the
conversation. The model learns the special tokens, turn-taking, `<|think_start|>`,
tool calls and tool results. The loader and loss are the same as pretraining (packed,
every token): it is still language modeling, on 4.9M tokens of a new kind of
document. 25% of documents are pretraining text, against forgetting. 600 steps, lr
1e-3.

### 3. SFT (`train/sft.py`): behavior

Labs fine-tune on a small, curated set. Ours is a fixed set of 8,000 generated
conversations (seen 1.2 times), one per row starting at position 0 like at inference,
with the loss only on the assistant's tokens. 300 steps, lr 3e-4. It teaches what
midtraining never shows:

- **system prompts to obey**, each with an automatic check:
  - `Answer with the number only.`: the visible answer is `405` (it still reasons in
    the scratchpad, like a reasoning model's hidden thoughts);
  - `Do not use the calculator.`: with the tool available, it uses the scratchpad;
  - `Answer in one short sentence.`: a one-sentence story;
  - `Start every answer with "Sure!".`
- **follow-ups** that refer to an earlier answer: "What is 12 + 30?" → "The answer is
  42." → "And add 25 to that?" → (scratchpad `42+25: ...`) "The answer is 67.". Earlier
  turns come *without* their scratchpad, as an API client sends the history back
  (OpenAI clients don't return `reasoning_content`). Trained with the scratchpads in
  the history, the model read the previous total from them, and failed on real
  histories (`42` + 25 became `03+25`). Half the time the history is also compact, as
  the chat app sends it: each turn's final answer, without the calculator round trip;
- **new questions** after one to three earlier answers (additions, stories, greetings): "What
  is 347 + 58?" → "The answer is 405." → "766 + 989" → (scratchpad `766+989: ...`) "The
  answer is 1755.". The operands are the new ones, not the last total;
- **identity** ("Who are you?" → "I'm mini, ...");
- **polite refusals** of out-of-scope requests (capitals, code, weather, trivia,
  multiplication...), 32 question templates;
- **plain conversations** like midtraining's (23%), sometimes under a neutral system
  prompt. They keep the skills, and teach that in-scope requests are *not* refused.

The mix is plain 23%, instruction 35%, follow-up 12%, new question 12%, refusal 12%,
identity 6%. Among
instructions, "number only" gets twice the examples.

### 4. RL (`train/rl.py --stage rl_math`): the math specialist practices

A minimal GRPO, like nanochat's simplified version: on-policy, no ratio clipping, no
KL penalty. Each step samples 16 problems and 8 attempts each at temperature 1. The
reward is 1 if the turn is ended and does what was asked. It is the eval's own
`grade()`. Advantages are normalized within each group of 8, and the loss is
`-advantage x log p(token)` over completion tokens. Groups where all 8 attempts got
the same reward carry no signal and are skipped. 300 steps, lr 1e-4.

The problems come from the eval's generators (`eval/tasks.py:make_problem`), with
their own random seed. The specialist only gets math (`[rl_math]` in the config):

- 50% additions over every length seen in pretraining (1-5 digits). The 1-3 digit
  ones are already solved, so they are mostly skipped. On 4-5 digits the SFT model
  fails, *but not always*: among 8 attempts, a few are right. RL finds those lucky
  attempts and reinforces them. The mean reward climbs from ~0.7 to ~0.99 over the
  300 steps.
- 20% calculator (the call, or the final answer after the result).
- 10% follow-ups on a 4-5 digit total ("And add 31 to that?" after "The answer is
  8829."), and 20% new questions of 1-5 digits (each length equally often) after one
  to three earlier turns: SFT only shows those with short numbers.

Nothing else is in its mix, so nothing holds its other skills in place, and they
drift: after RL it follows "Answer with the number only." 80% of the time (SFT:
100%). That is fine, because it is never released: it is a teacher,
and it will only be asked about math.

### 5. Distillation (`train/distill.py`): one model from two teachers

The released model is a student that starts again from the SFT model and learns from
two teachers, each on its own domain:

- the math specialist, on additions, calculator problems, long follow-ups and new
  questions;
- the SFT model itself, on everything else (system prompts, follow-ups, refusals,
  identity): it already does them right.

Each step, the student answers 32 problems once, at temperature 1 (on-policy: its own
answers, mistakes included). The problem's teacher reads the same tokens and gives its
next-token distribution at every position of the answer, and the loss is the reverse
KL(student ‖ teacher) over the whole vocabulary, averaged over the answer's
positions. Every token gets a grade, where RL gives one reward per answer: the KL
falls below 0.001 within ~60 steps. The problem mix is the one mini-1's RL used (50%
additions, 15% calculator, 35% instructions), plus 5% each of new questions, follow-ups
on long totals and story requests. 300 steps, lr 1e-4: 9,600 sampled
answers, a quarter of the specialist's 38,400.

This is how labs merge specialists now (DeepSeek-V4, Kimi K3, Nemotron 3: math,
code, agents, trained apart, then distilled into one model). Here it replaces
mini-1's single RL run on every skill at once (`[rl]` in the config, still runnable:
`uv run python -m minilab.train.rl --run runs/small`). That run had to keep the skills
it wasn't practicing from drifting by putting them all in its mix, as anchors: as soon
as a skill degraded it failed, got a signal, and was pulled back. Every skill left out
drifted: without identity in the mix, rewarding refusals turned "Tell me about
yourself" into "Sorry, I can only..."; without "Sure!", answers to additions under
that system prompt lost their "Sure!" (90% → 63%). And stories, which have no reward,
drifted anyway (on topic 93% → 76-87%). Distillation pulls every answer toward a
teacher, token by token, and nothing drifted, stories included (see
[What we tuned](#what-we-tuned-and-why)).

### Eval (`eval/tasks.py`, `eval/run.py`) and release (`release.py`)

Problems are generated from a fixed seed (`EVAL_SEED`), so every stage and every run
is scored on the same questions. The refusal check uses out-of-scope questions that
training never uses. Decoding is greedy except for `5d@T=1`. Results go to
`runs/<run>/<stage>/eval.json`, with per-digit and per-instruction breakdowns and
samples. `eval/model_card.py` writes `MODEL_CARD.md`: architecture, data, per-stage
training stats, the eval table and samples.

`release.py` copies the checkpoint to `models/<id>/` (`MINILAB_MODELS_DIR`) with
`release.json` (`registry.ModelInfo`: context length 256, pricing $10 / $50 per 1M
input/output tokens, the flagship tier of Claude Fable 5.1 and GPT-6 Astra, so that
a story costs about a cent), `eval.json` and `MODEL_CARD.md`.

## What we tuned, and why

- **Each stage needs its own job.** In our first design every stage saw the same
  kinds of data: 1-3 digit additions, stories, small talk. Midtraining alone reached
  100% on addition, so SFT and RL had nothing left to show. Now:
  - pretraining teaches knowledge: stories, and addition up to 5 digits in raw text;
  - midtraining teaches the chat format and the skills, on 1-3 digit additions only;
  - SFT teaches behavior: instructions, follow-ups, refusals;
  - RL extends the chat skill to 4-5 digits, which only pretraining showed, and
    distillation brings it into the released model.
- **A fixed SFT set overfits fast.** 4,000 conversations seen 3.2 times: train loss
  0.26 vs validation 0.62, and perplexity on raw stories 6.3 → 9.3, because the
  stories get memorized. 8,000 conversations seen 1.2 times, at a lower lr, keep
  instruction following at ~100% with perplexity ~7.1.
- **Refusals generalize from contrast.** Held-out out-of-scope questions were only
  refused 73% of the time. "Who painted the Mona Lisa?" got "I'm mini..." (like
  "Who are you?"), and "Tell me today's news" was treated like "Tell me about
  yourself". Adding out-of-scope training questions that start the same way ("Who
  wrote...", "Tell me the latest...", "How far...") brought it to 93%.
- **Always think.** Single-digit questions were first answered directly, and they
  were the only ones below 100% (`3 + 8 = 10`). The column sums inside the scratchpad
  (`3+8=11`) were always right. Likewise, "Answer with the number only" first meant
  no scratchpad at all, and the bare numbers were almost random (`5 + 3` → `13`).
  Now it only changes the visible answer.
- **RL hacks weak rewards, fast.** With an "answer only, no reasoning" request and a
  reward that only checked the final number, RL learned within 20 steps to ignore the
  request and reason anyway (completion length doubled). It also forgot the
  calculator (tool calls 100% → 2%). Requiring "no reasoning" led to the next hack:
  drop `<|think_start|>` and write the scratchpad as the answer, so the parser saw no
  reasoning.
- **...and the eval had the same blind spot.** Both the reward and the eval checked
  "the last number of the answer is right", and the answer was then `a + b = c`. A
  release that scored 100% on 4-digit additions answered "What is 4521 + 380?" with
  `451 + 0 = 4901`: right sum, garbled operands, correct scratchpad. Nothing rewarded
  restating the question, so RL let that part rot, even on 2-3 digits that SFT had
  right. The coordinator caught it by hand, testing the served model. Graded strictly
  (the answer must be exactly `4521 + 380 = 4901`), that release scored 79% / 52% /
  19% / 1% on 2-5 digits. We made `grade()` as strict as the training data: the exact
  answer and nothing else, for both the eval and the RL reward. With that reward the
  same RL recipe learned to restate the operands (99% on 4 digits). The lesson: **a
  metric you optimize must check everything you care about**, and so must the metric
  you report.
- **Restating the operands also broke "number only".** A second run of the same
  recipe (only the device differed) followed "Answer with the number only." 20% of
  the time instead of 100%. It gave a bare number, but the *first operand* (`581 + 9`
  → `581`): the default answer starts by copying the question's first number, and
  that copy won. More examples didn't fix it (4x: 33%). So the answer became `The
  answer is 4901.`, which only copies the sum from the end of the scratchpad, like the
  number-only answer does. After SFT, number only went to 100% on both runs, and
  4-digit additions went from ~10% to ~50% (no long numbers left to copy). The best
  fix removed the hard part rather than training harder on it.
- **...and again, as soon as a new check was rewarded.** Once "Sure!" was in the RL
  mix, its check (the answer starts with "Sure!") got the cheapest possible answer:
  `Sure! Bye! Come back for another story soon.` for every request, stories and
  greetings alike. Now "Sure!" wraps a real request, and the rest of the answer must
  pass that request's own check: the exact answer, a real story, or a reply that
  fits the greeting.
- **...and again: RL invented the tool's result.** The calculator check was "a correct
  call somewhere in the turn, and the turn ends". RL learned to write the call and
  then keep going: `<|tool_call_end|>2<|tool_end|><|assistant_start|>The answer is 2.`.
  It made up the result (`2 + 4` → 2) and spoke for the tool, sometimes calling twice.
  `parse_completion` skips stray special tokens, so the turn looked fine, and so did
  the `format` metric. Under a strict check, only 44% of the calculator turns were
  clean. Now a turn may only contain the special tokens its answer needs, and a
  calculator turn is exactly one call and nothing else. Retrained with that reward:
  146/150 clean calls.
- **Train on answers, not on history.** Follow-up conversations first kept each
  earlier turn's scratchpad in the history. Real API clients don't send
  `reasoning_content` back, and on real histories the model read the wrong previous
  total. Dropping the scratchpads from the history fixed that, but broke something
  else. The SFT loss covers every assistant turn, so the model also learned from the
  history answers, which now had no scratchpad: 1-2 digit additions after SFT fell to
  55-75% (answered without thinking). For those conversations the loss now only covers
  the last turn (`train_on = "last"` in the loader). The history is context to read,
  not answers to imitate.
- **A second question is not always a follow-up.** Every multi-turn conversation in
  training was a follow-up ("And add 25 to that?"), and the model learned the shortcut:
  in a second addition, the first operand is the last total. In the chat app, "What is
  347 + 58?" → "The answer is 405." → "766 + 989" got the scratchpad `405+989: ...` and
  "The answer is 1394.". No metric saw it, because the eval only asked real
  follow-ups: mini-2's first release answered 40% of the new check (a fresh addition
  after an answer) right. It also stopped calling the calculator after the first
  turn: the chat app sends the history without the tool round trips, a shape it had
  never seen. SFT now has new questions after an answer (12%), half the multi-turn
  histories are compact like the chat app's, and the check is in the eval and in the
  distillation mix: 97% (237 of 240 in a larger test, with and without the
  calculator).
- **Train on the whole answer.** That fix alone brought new questions to 100%
  without the calculator, and 0% with it: the model still never called the
  calculator after the first turn. Multi-turn conversations only train on the last
  answer (`train_on = "last"`), and "last" meant the last assistant turn: after the
  tool's result, "The answer is 1755." The call itself, one assistant turn earlier,
  was never trained on, in follow-ups either. The loss now covers everything after
  the last user message.
- **Follow-ups on long totals.** After those fixes, a new question in a later turn
  was right 97-100% of the time, but a follow-up on a 4-digit total almost never:
  "8829" came back as `882+031: ...` in the scratchpad, one digit short, the same
  failure as 6-digit additions. SFT follow-ups only use short numbers, and the
  specialist had only ever practiced single questions, so no stage showed a long
  total being carried over, and no check asked for it. The specialist now practices
  follow-ups on 4-5 digit totals and new questions after several turns, and the
  distillation routes both to it: 37% → 97% on the check (92-94% on 4-digit totals in
  a larger test, with and without the calculator, where it had been 0-22%). The first
  try made short new questions worse (2 digits after one turn: 98% → 64%): new
  questions were drawn mostly long, like additions, so the specialist rarely practiced
  short ones, and it padded a 1-digit operand wrong ("3 + 69" → `33+69`). The check
  drew them mostly long too, and missed it. Drawing every length equally often, in
  practice and in the check, and doubling their share brought them back.
- **Stories need an anchor too.** With new questions in the distillation mix, the
  "Sure!" check fell from 90% (SFT) to 63%. Every answer started with "Sure!", but
  under greedy decoding the generic story requests ("Write a story for me.") all got
  the same story, and it looped ("She wanted to sleep. She wanted to sleep...") until
  the context ran out. At the chat app's temperature every story ended. Stories
  weren't in the distillation mix, so nothing held them; 5% story requests (taught by
  the SFT model) brought "Sure!" to 100%.
- **Drift.** RL on only some skills slowly erodes the others (5-digit accuracy fell
  from 100% to 87% while practicing something else). Adam turns small, noisy
  gradients into lr-sized steps, and solved problems carry no gradient to hold the
  model in place. A KL penalty to the SFT model fixed it, at twice the cost per step.
  Putting the skills we care about in the RL mix did the same job for free, for the
  skills that have a reward. Distillation now does it on every token (below).
- **Muon.** Kimi, GLM-5 and DeepSeek-V4 train with Muon: momentum SGD where each
  matrix's update is replaced by its closest orthogonal matrix (5 Newton-Schulz
  iterations), so every direction of the matrix moves at the same speed. Scaled to
  AdamW's update size, it takes AdamW's lr and weight decay unchanged. Over two seeds,
  pretraining validation loss went 1.852 → 1.731, and after the whole pipeline story
  perplexity 7.38 → 6.66 and 5-digit additions 91-96% → 99%. It costs 8% of the
  throughput on MPS (13% on the CPU), and it still wins at equal wall-clock: AdamW
  given 37% more steps only reaches 1.780. One trap: `torch.optim.Muon` runs the
  iterations in bf16, and on a Mac CPU a bf16 matmul is so slow that a step took 10 s
  (AdamW: 12 ms). `Muon` in `train/trainer.py` uses float32 on the CPU: 127 ms.
- **One RL run, or a specialist and distillation.** From the same SFT checkpoints
  (two seeds, AdamW), the single RL run on everything (mini-1's recipe) against the
  math specialist + distillation:

  | | story ppl | 5d@T=1 | tool ans | story | refusal |
  |---|---:|---:|---:|---:|---:|
  | one RL run | 7.37 / 7.39 | 89% / 91% | 96% / 93% | 87% / 76% | 90% / 100% |
  | specialist + distillation | 7.13 / 7.17 | 98% / 97% | 99% / 99% | 93% / 96% | 100% / 100% |

  The alignment tax is gone (SFT: 7.12 / 7.16), and so is the drift, stories included. The
  specialist alone is better at math than the single run (5-digit 98-99%), and worse
  at the rest ("number only" 0% and 40%): exactly why it's a teacher and not the
  release. Distillation is also cheap: a 150-step specialist and 100 distillation
  steps (22k sampled answers, against 38k for the RL run) still matched or beat the
  RL run everywhere.
- **The 2026 block didn't pay for itself.** `GPTConfig` has the attention and MLP of
  Qwen3.5 and Kimi K3: SwiGLU, RMSNorm on queries and keys, and a sigmoid gate on the
  attention output (which lets a head output nothing instead of parking its attention
  on the first token). At the same parameter count it lowers the pretraining loss per
  step (1.852 → 1.781, and 1.731 → 1.704 on top of Muon), but its extra small kernels
  make it 22% slower on MPS (8% on the CPU). At equal wall-clock on MPS, Muon alone
  with 4,400 steps reaches 1.694, against 1.699 with the block. It's off by default
  (see `configs/small.toml` to turn it on).
- **Recurrent depth: no better stories, but longer additions.** `GPTConfig` can loop
  its blocks, like Huginn and Ouro, and reportedly GPT-6 Astra. A prelude runs once,
  then a core runs `loops` times with the same weights, then a coda. Each run of a
  block gets its own layers in the KV cache. `train_loops` draws a new loop count at
  every step, and `inject` adds the prelude's output back before each extra loop. We
  pretrained each variant once (seed 0, 3500 steps, MPS). Four baseline runs ended at
  1.727-1.736.

  | pretraining | params | compute per token | validation loss |
  |---|---:|---:|---:|
  | baseline | 5.8M | 1x | 1.732 |
  | the whole stack run twice (Astra-style) | 5.8M | 2x | 1.724 |
  | 1 block, 4 blocks looped 1-4 times, 1 block; injection | 5.8M | ~2x | 1.726 |
  | 12 distinct blocks | 10.5M | 2x | 1.669 |
  | baseline trained twice as long (7000 steps) | 5.8M | 2x | 1.632 |

  At this size the model is short of parameters and data, not of depth. Spent on
  training longer, the same extra compute lowers the loss 12 times more than looping.
  Test-time compute didn't pay either. The model trained at exactly 2 loops falls
  apart when given more (perplexity 5.47 at 2 loops, 7.49 at 3, 11.4 at 4). The model
  trained at 1-4 loops accepts any count but stops improving after 2: perplexity 5.71,
  5.52, 5.52, 5.56, 5.69 and 5.86 at 1, 2, 3, 4, 6 and 8 loops. We couldn't tell
  whether looping helps direct answers (`a + b =` without a scratchpad): the four
  baseline runs answered 1-digit additions right 48%, 57%, 94% and 98% of the time.

  Length generalization is a different story. In a separate toy task, the model
  answers `a+b=` directly, with two n-digit operands, and is trained on 1-5 digits. As
  in "Looped Transformers for Length Generalization" (Fan et al., 2024), the looped
  model fills all n + 1 answer slots at once, after its loops. We write the sum least
  significant digit first, so each loop can pass the carry one digit further. One
  seed each:

  | direct addition, trained on 1-5 digits | params | 5 digits | 6 digits (never seen) | 7 |
  |---|---:|---:|---:|---:|
  | GPT, 6 blocks, RoPE, one token at a time (like mini) | 4.7M | 93% | 0% | 0% |
  | the same without positional encoding (NoPE) | 4.7M | 83% | 4% | 0% |
  | 6 blocks, NoPE, the whole answer at once, no loop | 4.7M | 0% | 0% | 0% |
  | 1 block looped n + 1 times, RoPE | 0.8M | 53% | 0% | 0% |
  | 1 block looped n + 1 times, NoPE | 0.8M | 97% | 53% | 0% |
  | 1 block looped 6 times, NoPE | 0.8M | 97% | 72% | 2% |

  The loops are the computation. A single pass of 6 distinct blocks can't carry
  through 5 digits (3 digits: 23%). The looped block solves 5-digit additions with 6
  loops, but only 4-12% with 4. Only the looped models without positional encoding
  get past the lengths they trained on. Our reading: RoPE lets the model use rules
  tied to exact distances ("the matching digit is 7 tokens back"), and those distances
  change with the length. Without positions, it has to count ("the 3rd digit after
  the +"), and counting still works on a longer number. But the looped models gain one
  digit, not Fan et al.'s ten: they trained on 1-19 digits, we trained on 1-5.

  None of this is in mini. The gain needs direct answers, NoPE and whole-answer
  prediction. mini answers step by step in a scratchpad, with RoPE, and it fails on
  6 digits because it restates the operands wrong, not for lack of depth. Recurrent
  depth is off by default (see `configs/small.toml` to try it).
- **It runs the algorithm; it doesn't remember sums.** Replaying the exact training
  streams: 17 of the eval's 100 3-digit additions appear somewhere in training, 3 of
  the 4-digit ones and none of the 5-digit ones (0.0003% of all 5-digit pairs were
  ever seen). The model gets them right by writing the scratchpad, and the scratchpad
  is what it reads its answer from. On 100 new 5-digit problems, mini-2's scratchpad
  matches the reference line for line 100 times out of 100, with no tool available.
  Write its first column wrong on purpose (`1+0=2`) and let it continue: it carries
  the mistake through and gives the wrong sum 100 times out of 100, never the true
  one. What it did memorize is the column
  step (one digit plus one digit plus a carry: 200 cases, seen hundreds of thousands
  of times).
- **Topic adherence.** "Tell me a story about a cat" was answered with a generic dog
  story. Picking training stories whose *first sentence* mentions the topic took the
  metric from ~75% to ~98%.
- **Sizes.** On this CPU a 6-layer, width-256 model trains at ~19k tokens/s with
  Muon. 3500 pretraining steps (29M tokens, ~25 min CPU-only) is the bulk of the budget.
  Validation loss was still slowly improving, so a longer pretraining buys better
  stories.

## Devices

The whole pipeline runs on a CPU. On a Mac, MPS trains much faster, but generation is
another story: `generate` writes each sequence into its own KV-cache slot, one small
kernel per sequence per layer, and on MPS kernel-launch overhead dominates.

| (M5 Pro, small model) | CPU (12 threads) | MPS |
|---|---:|---:|
| training, batch 32 x 256 (Muon) | ~19k tok/s | ~74k tok/s |
| sampling 128 x 100 tokens (an RL step) | 1.8 s | 6.8 s |
| greedy 64 x 150 tokens (eval) | 1.1 s | 6.1 s |

So `--device auto` means cuda if available, else mps, else cpu, *except* for RL,
distillation and eval, which pick the CPU over MPS. `--device` forces a device. The loop avoids a
`.item()` sync per step, and generators and the KV cache follow the model's device.

## Known limitations

- It is a toy: children's stories and addition, no world knowledge. Stories are
  simple and sometimes drift or repeat. The context is 256 tokens.
- No length generalization: 6-digit additions fail (the model copies the long
  numbers wrong). The scratchpad makes each step local, but restating an operand
  longer than any seen in training is still out of distribution. A looped block
  without positional encoding does reach 6 digits, but only as a separate toy that
  answers directly (see [What we tuned](#what-we-tuned-and-why)).
- Instruction following covers exactly the four trained system prompts. Other
  instructions are ignored. Refusal is keyed on surface patterns and can misfire on
  unusual phrasings.
- Story topics are the 15 trained ones. "Tell me a story about a robot" gets a
  generic story (earlier checkpoints even looped: "a tiny model: a tiny model...").
- Later turns are less reliable than the first. A fresh addition after earlier turns
  is right ~91% of the time on average (every length is 100% on the first turn), and
  less with the calculator on short numbers (~78% for 1-digit ones).
- Follow-ups work for the trained phrasings ("And add 25 to that?", "Plus 9?", ...)
  after a short answer. Other phrasings may be read as a new question, and a new
  question that looks like a follow-up ("Plus 9 and 3?") may be read as one.
- Results vary between runs and devices: mini-1's recipe gave 5-digit accuracy of
  91-99% after RL and stories on topic 76-100%; nine runs of the specialist +
  distillation recipe gave 99-100% on 5 digits and 93-100% on topic.
- The instruction checks are shallow on purpose (automatic): "a real story" means 30+
  words that aren't a refusal.
- Subtraction is not implemented.
