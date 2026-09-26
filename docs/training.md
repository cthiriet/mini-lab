# Training

`minilab/data/`, `minilab/train/`, `minilab/eval/` and `minilab/release.py` train a
small GPT from scratch on a laptop and release it for the serving stack. It goes
through the same stages as a frontier model, just tiny: **tokenizer → pretraining →
midtraining → SFT → RL → eval → release**. Each stage has one job, and each job
shows up in a fixed eval.

```bash
bash speedrun.sh small            # ~16 min on an Apple M5 Pro (MPS + CPU), ~36 min CPU-only -> models/mini-1
bash speedrun.sh tiny             # ~20 s smoke test (CI)                                   -> models/mini-tiny
DEVICE=cpu bash speedrun.sh small # force the CPU
```

The model is a 5.8M-parameter GPT (6 layers, 4 heads, width 256, 256-token context,
4096-token BPE vocabulary). It tells short children's stories, adds numbers step by
step (the scratchpad comes back as `reasoning_content`) or with a `calculator` tool,
follows a few system prompts, handles follow-up questions, and politely declines
everything else.

## Results

`runs/small`, released as `models/mini-1`. Same fixed-seed eval after every stage
(`uv run python -m minilab.eval.run --run runs/small --summary`):

```
stage             ppl    1d    2d    3d    4d   5d  6d*  5d@T=1  tool call  tool ans  story  instr  format
---------------  ----  ----  ----  ----  ----  ---  ---  ------  ---------  --------  -----  -----  ------
pretrain (base)  6.17   34%    7%    4%    7%   2%   0%       -          -         -      -      -       -
midtrain         6.27  100%  100%  100%    0%   0%   0%      0%        65%       65%    96%    15%     67%
sft              7.12  100%  100%  100%    1%   0%   0%      0%        66%       65%    93%    99%     98%
rl               7.37  100%  100%  100%  100%  91%   0%     89%        97%       96%    87%    98%    100%
```

| column | what it measures |
|---|---|
| `ppl` | perplexity on held-out TinyStories (the official validation file, never trained on) |
| `Nd` | exact-match accuracy on 100 N-digit additions, greedy. Chat stages are asked in the chat format ("What is 347 + 58?"), and the whole visible answer must be exactly `The answer is 405.` The base model has never seen a chat token, so it gets the raw-text prompt `347 + 58 =` and must continue with the sum, without a scratchpad. |
| `6d*` | held out: no 6-digit number appears anywhere in training (length generalization) |
| `5d@T=1` | 5-digit questions sampled at temperature 1, the API default |
| `tool call` | with `tools=["calculator"]` (1-5 digits): the first turn is exactly one calculator call, whose expression evaluates to the right answer, and nothing else |
| `tool ans` | ...and after the tool result is appended, the final answer is exactly right |
| `story` | "Tell me a story about a cat." (15 topics x 3 phrasings): the story mentions the topic |
| `instr` | instruction following: the mean of the checks below |
| `format` | share of all chat turns in the eval that end with `<|assistant_end|>`, with no tool call when no tool is available, and no other role's tokens (e.g. an invented `<|tool_start|>` result) |

`instr` is the mean of eight automatic checks (30 prompts each, in `eval.json`):

| check (30 prompts each) | midtrain | SFT | RL |
|---|---:|---:|---:|
| `Answer with the number only.`: the answer is just `521` | 0% | 97% | 100% |
| `Do not use the calculator.`, with the tool available: the scratchpad is used | 10% | 100% | 100% |
| `Answer in one short sentence.` + "Tell me a story about a dragon." | 0% | 100% | 100% |
| `Start every answer with "Sure!".`: "Sure! " followed by the right answer | 0% | 100% | 100% |
| follow-up: "What is 12 + 30?" → "The answer is 42." → "And add 25 to that?" → `The answer is 67.` | 7% | 97% | 97% |
| refusal of held-out out-of-scope questions ("Who painted the Mona Lisa?") | 0% | 100% | 90% |
| identity: "Who are you?" → "I'm mini, ..." | 0% | 100% | 100% |
| over-refusal: in-scope requests refused (lower is better) | 0% | 0% | 0% |

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
- **RL** practices on every length seen in pretraining. 4-5 digit chat additions go
  from ~0% to 91-100%, and with them calculator calls on long numbers and format
  adherence. Instruction following is kept, because it is in the RL mix. Perplexity on
  raw stories creeps up slightly (6.3 → 7.4 over SFT + RL): the alignment tax.
- Nothing reaches 6 digits, which never appear in training.

Wall-clock of the small speedrun on an Apple M5 Pro (18 cores, 64 GB):

| stage | `--device auto` (MPS for training, CPU for RL and eval) | `--device cpu` |
|---|---:|---:|
| data (cached) + tokenizer | 4 s | 4 s |
| pretrain (3500 steps, 29M tokens) | 6.8 min (MPS) | 22.5 min |
| midtrain (600 steps, 4.9M tokens) | 1.1 min (MPS) | 3.9 min |
| SFT (300 steps) | 0.4 min (MPS) | 1.4 min |
| RL (300 steps x 128 samples) | 7.2 min (CPU) | 7.1 min |
| 4 evals + release + report | ~1 min | ~1 min |
| **total** | **~16.5 min** | **~36 min** |

The first download of the 104 MB of TinyStories adds ~10 s on a fast connection.

Samples from `mini-1` (greedy):

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

> [system] Start every answer with "Sure!".  > What is 34 + 58?
  Sure! The answer is 92.

> [system] Answer in one short sentence.  > Tell me a story about a dragon.
  Once upon a time, there was a big dragon.

> Who are you?
  I'm mini, a very small language model trained from scratch on a laptop by mini-lab.

> Can you write Python code?
  Sorry, I don't know about that. I can only tell short stories and add numbers.

> Tell me a story about a bird.
  Once upon a time, there was a little bird named Tim. Tim lived in a big tree. One day,
  Tim saw a big, red ball. He wanted to play with it.
  Tim flew up to the tree and started to play with the ball. He threw the ball high in
  the air. The ball went up and up. Tim was very happy. [...]
```

## Reading a run

- **`runs/<run>/report.html`**: written at the end of `speedrun.sh`, or by
  `uv run python -m minilab.report runs/small [runs/other ...] [--open]`. One
  self-contained page shows loss, val loss, lr and throughput per stage, the RL
  reward, completion length and informative-group curves, the per-stage eval heatmap
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
uv run python -m minilab.train.rl         --run runs/small
uv run python -m minilab.eval.run         --run runs/small --stage rl
uv run python -m minilab.eval.run         --run runs/small --summary                 # the table above
uv run python -m minilab.release          --run runs/small --stage rl --id mini-1    # -> models/mini-1
uv run pytest tests/test_data.py tests/test_train.py tests/test_eval.py              # ~1 s
```

Every command takes `--device auto|cpu|mps|cuda`. The default, `auto`, picks cuda,
else mps, else cpu (see [Devices](#devices)). The tokenizer step copies the config to
`runs/<run>/config.toml`, and later stages read it from there. A run directory:

```
runs/small/
  config.toml  tokenizer.json
  pretrain/  model.pt config.json tokenizer.json log.jsonl eval.json
  midtrain/  ...   sft/  ...   rl/  ...
```

`config.json` holds the model config plus training stats in `meta` (steps, tokens,
wall-clock, device, hardware). `log.jsonl` has one JSON object per log line (loss,
lr, grad norm, tokens/s, val loss, samples; reward for RL).

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
pass over the 100 MB of stories. AdamW (0.9, 0.95), lr 3e-3 with 100 warmup steps and
cosine decay to 10%. Weight decay 0.1 on matrices only, grad clip 1.0. Validation
loss goes 3.2 → 1.84 (perplexity 6.2), and the `log.jsonl` samples go from broken
sentences to coherent little stories.

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
  histories (`42` + 25 became `03+25`);
- **identity** ("Who are you?" → "I'm mini, ...");
- **polite refusals** of out-of-scope requests (capitals, code, weather, trivia,
  multiplication...), 32 question templates;
- **plain conversations** like midtraining's (35%), sometimes under a neutral system
  prompt. They keep the skills, and teach that in-scope requests are *not* refused.

The mix is plain 35%, instruction 35%, follow-up 12%, refusal 12%, identity 6%. Among
instructions, "number only" gets twice the examples.

### 4. RL (`train/rl.py`): practice

A minimal GRPO, like nanochat's simplified version: on-policy, no ratio clipping, no
KL penalty. Each step samples 16 problems and 8 attempts each at temperature 1. The
reward is 1 if the turn is ended and does what was asked. It is the eval's own
`grade()`. Advantages are normalized within each group of 8, and the loss is
`-advantage x log p(token)` over completion tokens. Groups where all 8 attempts got
the same reward carry no signal and are skipped. 300 steps, lr 1e-4.

The problems come from the eval's generators (`eval/tasks.py:make_problem`), with
their own random seed, in a configurable mix:

- 50% additions over every length seen in pretraining (1-5 digits). The 1-3 digit
  ones are already solved, so they are mostly skipped. On 4-5 digits the SFT model
  fails, *but not always*: among 8 attempts, a few are right. RL finds those lucky
  attempts and reinforces them. The mean reward climbs from ~0.6 to ~0.95 over the
  300 steps.
- 15% calculator (the call, or the final answer after the result).
- 35% instruction following at chat lengths: number only 10%, and 5% each for no
  calculator, "Sure!", follow-ups, refusals and identity. These are anchors: as soon
  as a skill degrades it fails, gets a signal, and is pulled back. Every skill left
  out drifted. Without identity in the mix, rewarding refusals turned "Tell me about
  yourself" into "Sorry, I can only...". Without "Sure!", answers to additions under
  that system prompt lost their "Sure!" (90% → 63%).

### Eval (`eval/tasks.py`, `eval/run.py`) and release (`release.py`)

Problems are generated from a fixed seed (`EVAL_SEED`), so every stage and every run
is scored on the same questions. The refusal check uses out-of-scope questions that
training never uses. Decoding is greedy except for `5d@T=1`. Results go to
`runs/<run>/<stage>/eval.json`, with per-digit and per-instruction breakdowns and
samples. `eval/model_card.py` writes `MODEL_CARD.md`: architecture, data, per-stage
training stats, the eval table and samples.

`release.py` copies the checkpoint to `models/<id>/` (`MINILAB_MODELS_DIR`) with
`release.json` (`registry.ModelInfo`: context length 256, pricing $0.50 / $1.50 per
1M input/output tokens), `eval.json` and `MODEL_CARD.md`.

## What we tuned, and why

- **Each stage needs its own job.** In our first design every stage saw the same
  kinds of data: 1-3 digit additions, stories, small talk. Midtraining alone reached
  100% on addition, so SFT and RL had nothing left to show. Now:
  - pretraining teaches knowledge: stories, and addition up to 5 digits in raw text;
  - midtraining teaches the chat format and the skills, on 1-3 digit additions only;
  - SFT teaches behavior: instructions, follow-ups, refusals;
  - RL extends the chat skill to 4-5 digits, which only pretraining showed.
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
- **Drift.** RL on only some skills slowly erodes the others (5-digit accuracy fell
  from 100% to 87% while practicing something else). Adam turns small, noisy
  gradients into lr-sized steps, and solved problems carry no gradient to hold the
  model in place. A KL penalty to the SFT model fixed it, at twice the cost per step.
  Putting the skills we care about in the RL mix does the same job for free.
- **Topic adherence.** "Tell me a story about a cat" was answered with a generic dog
  story. Picking training stories whose *first sentence* mentions the topic took the
  metric from ~75% to ~98%.
- **Sizes.** On this CPU a 6-layer, width-256 model trains at ~21k tokens/s. 3500
  pretraining steps (29M tokens, ~22 min CPU-only) is the bulk of the budget.
  Validation loss was still slowly improving, so a longer pretraining buys better
  stories.

## Devices

The whole pipeline runs on a CPU. On a Mac, MPS trains much faster, but generation is
another story: `generate` writes each sequence into its own KV-cache slot, one small
kernel per sequence per layer, and on MPS kernel-launch overhead dominates.

| (M5 Pro, small model) | CPU (12 threads) | MPS |
|---|---:|---:|
| training, batch 32 x 256 | ~21k tok/s | ~75k tok/s |
| sampling 128 x 100 tokens (an RL step) | 1.8 s | 6.8 s |
| greedy 64 x 150 tokens (eval) | 1.1 s | 6.1 s |

So `--device auto` means cuda if available, else mps, else cpu, *except* for RL and
eval, which pick the CPU over MPS. `--device` forces a device. The loop avoids a
`.item()` sync per step, and generators and the KV cache follow the model's device.

## Known limitations

- It is a toy: children's stories and addition, no world knowledge. Stories are
  simple and sometimes drift or repeat. The context is 256 tokens.
- No length generalization: 6-digit additions fail (the model copies the long
  numbers wrong). The scratchpad makes each step local, but restating a longer
  number than ever seen is still out of distribution.
- Instruction following covers exactly the four trained system prompts. Other
  instructions are ignored. Refusal is keyed on surface patterns and can misfire on
  unusual phrasings.
- Story topics are the 15 trained ones. "Tell me a story about a robot" gets a
  generic story (earlier checkpoints even looped: "a tiny model: a tiny model...").
- Follow-ups work for the trained phrasings ("And add 25 to that?", "Plus 9?", ...)
  after a short answer. Other phrasings may be read as a new question.
- Results vary between runs and devices: the same recipe gave 5-digit accuracy of
  91-99% after RL, and stories on topic 87-100%.
- The instruction checks are shallow on purpose (automatic): "a real story" means 30+
  words that aren't a refusal.
- Subtraction is not implemented.
