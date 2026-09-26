"""TinyStories: short, simple stories written by GPT-4 in the vocabulary of a 3-year-old.

The official files are plain text with stories separated by <|endoftext|>. The train
file is 2.2 GB, far more than a CPU can digest in an hour, so we download only its
first N megabytes with an HTTP Range request. The validation file is a separate set
of stories, so a prefix of it is a clean held-out set.

Files are cached under data/tinystories/ (override the root with MINILAB_DATA_DIR).
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx

BASE_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
FILES = {"train": "TinyStoriesV2-GPT4-train.txt", "val": "TinyStoriesV2-GPT4-valid.txt"}
SEPARATOR = "<|endoftext|>"

# GPT-4 used typographic punctuation; mapping it to ASCII saves vocabulary slots.
_ASCII = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                        "–": "-", "—": " - ", "…": "...", " ": " "})


def data_dir() -> Path:
    return Path(os.environ.get("MINILAB_DATA_DIR", "data"))


def download(split: str, mb: float) -> Path:
    """Download (once) the first `mb` megabytes of a split and return the local path."""
    path = data_dir() / "tinystories" / f"{split}-{mb:g}MB.txt"
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    n_bytes = int(mb * 2**20)
    url = BASE_URL + FILES[split]
    print(f"downloading {n_bytes / 2**20:.1f} MB of {FILES[split]} ...", flush=True)
    tmp = path.with_suffix(".part")
    with httpx.stream("GET", url, headers={"Range": f"bytes=0-{n_bytes - 1}"},
                      follow_redirects=True, timeout=60) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
                if f.tell() >= n_bytes:  # in case the server ignored the Range header
                    break
    tmp.rename(path)
    return path


def clean(story: str) -> str | None:
    """Normalize punctuation to ASCII and drop stories with any other exotic characters."""
    story = story.translate(_ASCII)
    story = "\n".join(" ".join(line.split()) for line in story.splitlines() if line.strip())
    if not story or not story.isascii():
        return None
    return story


def load_stories(split: str, mb: float) -> list[str]:
    """Clean stories from the first `mb` MB of a split (the last, truncated story is dropped)."""
    text = download(split, mb).read_text(encoding="utf-8", errors="ignore")
    parts = text.split(SEPARATOR)[:-1]  # the last part is cut by the byte range (or empty)
    return [s for p in parts if (s := clean(p))]


if __name__ == "__main__":
    import argparse
    import tomllib

    parser = argparse.ArgumentParser(description="Download the TinyStories subset used by a config.")
    parser.add_argument("--config", required=True)
    with open(parser.parse_args().config, "rb") as f:
        data = tomllib.load(f)["data"]
    for split, mb in (("train", data["train_mb"]), ("val", data["val_mb"])):
        stories = load_stories(split, mb)
        print(f"{split}: {len(stories)} stories, {sum(map(len, stories)) / 1e6:.1f}M characters")
