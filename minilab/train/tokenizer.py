"""Step 0: train the BPE tokenizer on a sample of the pretraining mix.

    uv run python -m minilab.train.tokenizer --config configs/prelude.toml --run runs/prelude

The sample contains stories, arithmetic worksheets, the text of a few chat conversations,
agent transcripts and Python documents, so common words, chat phrases, " +", " =", tool
names and Python keywords all get their own tokens. Also copies the config into the run
directory.
"""

from __future__ import annotations

import random
import shutil
import time
from pathlib import Path

from minilab.data import arithmetic, code
from minilab.data.conversations import StoryPool, sft_conversation
from minilab.data.tinystories import load_stories
from minilab.tokenizer.bpe import Tokenizer
from minilab.train.trainer import load_config, parse_args


def main() -> None:
    args = parse_args("Train the BPE tokenizer.")
    if not args.config:
        raise SystemExit("--config is required for the first step of a run")
    cfg = load_config(args)
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, run / "config.toml")

    d, tc = cfg["data"], cfg["tokenizer"]
    rng = random.Random(cfg.get("seed", 0))
    stories = load_stories("train", d["train_mb"])
    sample = rng.sample(stories, min(len(stories), tc["sample_stories"]))
    n_other = len(sample) // 5
    sample += [arithmetic.pretrain_document(rng, d["digits"]) for _ in range(n_other)]
    pool = StoryPool(stories[:1000])
    sft_mix = {"plain": 1.0, "instruction": 1.0, "followup": 1.0, "refusal": 1.0, "identity": 1.0}
    for _ in range(n_other):
        conv = sft_conversation(rng, sft_mix, d["digits"], pool, tool_frac=0.5)
        sample += code.text_of(conv) + ["calculator"]  # tool calls as the template writes them: calculator<|arg|>...
    # the code world: agent transcripts and Python documents
    n_code = tc["code_sample"]
    convs = code.sft_set(cfg, cfg.get("seed", 0) + 2)
    sample += [t for c in rng.sample(convs, min(n_code, len(convs))) for t in code.text_of(c)]
    docs = code.pretrain_documents(cfg.get("seed", 0) + 1)
    sample += [next(docs) for _ in range(n_code)]

    t0 = time.time()
    tok = Tokenizer.train(sample, tc["vocab_size"])
    tok.save(run / "tokenizer.json")
    val = load_stories("val", d["val_mb"])[:500]
    chars_per_token = sum(map(len, val)) / sum(len(tok.encode(s)) for s in val)
    print(f"tokenizer: vocab_size {tok.vocab_size} ({len(tok.merges)} merges) in {time.time() - t0:.1f}s, "
          f"{chars_per_token:.2f} chars/token on held-out stories -> {run / 'tokenizer.json'}")


if __name__ == "__main__":
    main()
