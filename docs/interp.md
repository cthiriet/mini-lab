# Interpretability

`minilab/interp/` looks inside the model. A **microscope** (`trace.py`) records everything a
forward pass computes and can change any of it; a **dictionary of features** (`sae.py`), trained
for every release by the speedrun, splits the residual stream into directions that each mean
one thing; the platform shows them on **Features pages**; and requests can **steer** the model
with them, through the API and the playground. Then a worked example: why prelude-2 can add 5-digit
numbers but not 6-digit ones.

```bash
uv run python -m minilab.interp.sae models/prelude-2 --config configs/prelude.toml   # the dictionary, ~7 min on MPS
uv run python -m minilab.interp.addition models/prelude-2 --features 1112 868         # the worked example, ~1 min
```

## The microscope (`trace.py`)

`trace(model, ids)` runs one sequence through the model the way `GPT.forward` does, written out
step by step so that nothing is thrown away:

| | |
|---|---|
| `resid[l]` | the residual stream: `resid[0]` the embeddings, `resid[l + 1]` after block `l` |
| `pattern[l]` | each head's attention weights, `(H, T, T)` |
| `heads[l]` | what each head writes into the residual stream, `(H, T, C)` |
| `neurons[l]`, `mlp[l]` | each MLP's hidden activations and what it writes |
| `logits` | the same as the model's (the tests check it) |

Attention is computed in the open (a softmax of the scores, not a fused kernel), so its weights
exist, and a head's output is its slice of the attention output times its slice of the output
projection, so the heads add up to the layer's output. `patch={(name, layer): fn}` replaces an
activation as it is computed: that's activation patching and ablation. Around it:
`logit_lens` (what the model would predict from each layer), `components` (the final residual
stream split into what each head and MLP wrote, the blocks' x0 mixes included, adding up exactly)
and `attribution` (direct logit attribution: how much each of them pushes one token over another).

```python
from minilab.interp.trace import attribution, trace

tr = trace(model, ids)                                       # everything
att = attribution(model, tr, pos, good=plus_id, bad=digit_id)  # {"embed", "L0.H0", ..., "L5.mlp"}
tr2 = trace(model, ids, {("mlp", 2): lambda x: x * 0})       # the same, without layer 2's MLP
```

## The dictionary (`sae.py`)

The residual stream is 256 numbers per token, each one mixing many things. A sparse autoencoder
learns many more directions (4,096) such that every token's residual stream is close to a sum of
only 32 of them: each **feature** then tends to mean one thing. It is OpenAI's TopK SAE (Gao et
al., 2024): the encoder keeps the 32 largest pre-activations, so sparsity is exact and there is
no penalty to tune, and an auxiliary loss lets dead features learn from what the live ones miss.
It trains on activations streamed from the model on its own kinds of text (stories 25%,
worksheets 10%, Python 10%, chats 30%, agent transcripts 25%): nothing is stored.

On prelude-2, after block 4 of 6 (the `[sae]` section of the recipe), 50M tokens, ~5.5 min on MPS:

| | |
|---|---|
| loss recovered | **98.9%**: next-token loss 2.565, 2.631 with the layer replaced by its reconstruction, 8.67 with it zeroed |
| variance unexplained | 7.3% |
| dead features | 0 of 4,096 |

The speedrun's last step writes it into the release: `models/<id>/features/` holds `sae.pt`
(the inference engine steers with it), `quality.json`, and `features.json`: each feature's 12
strongest examples on held-out text, how often it fires, on which kinds of text, and the tokens
it pushes the model toward. Hand-written labels go in `labels.json` (`{"1112": "..."}`).

## Features pages and steering

`/models/<id>/features` lists a release's features (search by token, label or number; filter by
the kind of text), and `/models/<id>/features/<n>` shows one: where it fires, what it makes the
model say, its examples with each token shaded by the feature's activation. Most read at a
glance: "a" in "Once upon a time", "module" in a Python traceback, "Answer" opening a system
prompt, the end of a scratchpad line.

**Steering** pushes features while the model writes. A request may carry
`steer: [{"feature": 2418, "strength": 15}]` (with the OpenAI SDK, in `extra_body`; see
[api.md](api.md)); the engine adds `strength` times each feature's direction to the residual
stream after block 4, at every position of that sequence only (`GPT.forward_cached(steer=...)`),
so steered and plain requests share a batch. A steered sequence neither reads nor leaves a cached
prefix. The playground has a control for it: a feature number, a strength slider and the
feature's label.

On "Tell me a story." (greedy), prelude-2 starts "a little girl named Lily. She loved to play
with her toys"; with feature 2418, which fires on " cat", at +15: "a little girl named Sue and
her cat, Tom". At +25 most features drive the text into repetitions. Word features this early
mostly encode what comes next ("dog" pushes "named"): they nudge a story more than they set its
topic.

## Worked example: why 6-digit additions fail

prelude-2 adds 1-5 digit numbers right every time and 6-digit numbers never. Its scratchpad
(data/arithmetic.py) restates the operands, zero-padded, then works one column per line:

```
837741+002623: 1+3=4, 4
83774+00262: 4+2=6, 64
...
```

**Where it breaks.** Forcing the right scratchpad on 60 six-digit problems and reading each
prediction (`addition.py`, part 1): every column sum and carry is right; what fails is where
numbers **end**. Line 1 writes "+" after 5 digits of A (6th digit right 12% of the time) and ":"
after 5 of B (12%); line 2 stops after 4 digits (10%); the result after 6. Each is the longest
that line ever is in training, where operands have 1-5 digits.

**Who ends the number** (part 2: direct attribution of logit("+") − logit(the right digit) after
k digits of a 7-digit operand):

| digits written | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| MLPs of layers 1-3 push "+" | −0.4 | 0.0 | +0.8 | +3.2 | **+13.4** | +16.9 |
| heads of layers 4-5 push the digit | −7.4 | −6.9 | −6.9 | −6.8 | −7.6 | −5.3 |

The heads (L4.H3, L5.H3 and L4.H2 do most of it) look at the next digit to copy in the question
and push it, steadily. The MLPs count digits: nothing until 3, then a jump at 5, whatever the
number's real length. After 5 digits the counter wins.

**Causal check** (part 3: one MLP mean-ablated at the one position where the 6th digit is due):
the 6th digit comes out right 4 times in 40 untouched, 13 without layer 1's MLP, **37** without
layer 2's, **40** without layer 3's.

**In the dictionary.** Two features carry it. #868 is off for the first 4 digits of a number and
on from the 5th: the counter's output. #1112 fires on the digit just before "+" or ":" (the end
of a restated operand); on a 4-digit number at the 4th digit, rightly, on a 7-digit number from
the 5th. Taking both out where the 6th digit is due brings it back 31 times in 40; adding them
(x8, about their natural size) after the 3rd digit of a 5-digit operand makes the model write "+"
29 times in 40 (part 4).

**What doesn't fix it.** The counter is not a shortcut on top of a general rule: it is how the
model ends 4- and 5-digit numbers too. Zeroing the 6 neurons that carry most of its push breaks
those (4 digits 100% → 57%, 5 digits → 37%) and 6 digits stay at 0%. Fine-tuning on numbers of 1-12 digits
copied in chat, never an addition, weakens the counter by 60% and repairs line 1, but line 2 has
its own counter (copies of the line above, never longer than 4 digits in training) and the
5-digit scratchpads lose 17 points. Length generalization here needs a model that ends a number at
its delimiter, not one that counts.

## Limitations

- One dictionary, at one layer. Earlier layers hold the token, later ones the next prediction;
  a feature that spans blocks (a circuit) needs the microscope, not the dictionary.
- Features are labeled by hand, a handful of them; the others show their top token and examples.
- Steering adds a direction at every position, prompt included: strong enough to change a story,
  it also breaks things the feature wasn't meant to touch (an addition's whole scratchpad).
