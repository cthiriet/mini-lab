"""Step 0: train the BPE tokenizer on a sample of the pretraining mix.

    uv run python -m minilab.train.tokenizer --config configs/small.toml --run runs/small

The sample contains stories, arithmetic worksheets and the text of a few chat
conversations, so common words, chat phrases, " +", " =" and the JSON of
tool calls all get their own tokens. Also copies the config into the run directory.
"""

from __future__ import annotations

import json
import random
import shutil
import time
from pathlib import Path

from minilab.data import arithmetic
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
        for m in sft_conversation(rng, sft_mix, d["digits"], pool, tool_frac=0.5)["messages"]:
            sample.append(m.get("content") or "")
            for call in m.get("tool_calls") or []:  # the JSON the chat template renders
                fn = call["function"]
                sample.append(json.dumps({"name": fn["name"], "arguments": json.loads(fn["arguments"])}))

    t0 = time.time()
    tok = Tokenizer.train(sample, tc["vocab_size"])
    tok.save(run / "tokenizer.json")
    val = load_stories("val", d["val_mb"])[:500]
    chars_per_token = sum(map(len, val)) / sum(len(tok.encode(s)) for s in val)
    print(f"tokenizer: vocab_size {tok.vocab_size} ({len(tok.merges)} merges) in {time.time() - t0:.1f}s, "
          f"{chars_per_token:.2f} chars/token on held-out stories -> {run / 'tokenizer.json'}")


if __name__ == "__main__":
    main()
