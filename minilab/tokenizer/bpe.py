"""Byte-level BPE tokenizer, trained from scratch.

Text is first split into chunks with a regex (words, punctuation, whitespace and
*single digits*). Merges never cross chunk boundaries, so numbers are always
tokenized one digit at a time. This makes arithmetic far easier to learn for a
tiny model: "347" is always [3, 4, 7], never an arbitrary [34, 7].

Token ids are laid out as:
    0..255                      raw bytes
    256..256+n_merges-1         learned merges
    after that                  special tokens (see SPECIAL_TOKENS)
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

SPLIT_PATTERN = r"""'(?:s|t|re|ve|m|ll|d)| ?[A-Za-z]+|\d| ?[^\sA-Za-z\d]+|\s+(?!\S)|\s+"""

# Special tokens delimit documents and chat turns. They are never produced by
# encoding plain text (unless allow_special=True), so user content can't forge them.
SPECIAL_TOKENS = [
    "<|bos|>",
    "<|system_start|>",
    "<|system_end|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|think_start|>",
    "<|think_end|>",
    "<|tool_call_start|>",
    "<|tool_call_end|>",
    "<|tool_start|>",
    "<|tool_end|>",
]
# mini-code's chat template (tokenizer/chat.py) needs one more: it separates the arguments of a
# tool call, so that code goes in raw instead of escaped inside JSON.
CODE_SPECIAL_TOKENS = [*SPECIAL_TOKENS, "<|arg|>"]


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class Tokenizer:
    def __init__(self, merges: list[tuple[int, int]], special_tokens: list[str] = SPECIAL_TOKENS,
                 chat_template: str = "default"):
        self.pattern = re.compile(SPLIT_PATTERN)
        self.chat_template = chat_template  # "default" (mini), or "code" (mini-code): see tokenizer/chat.py
        self.merges = {tuple(p): 256 + i for i, p in enumerate(merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), idx in self.merges.items():
            self.vocab[idx] = self.vocab[a] + self.vocab[b]
        first_special = 256 + len(self.merges)
        self.special_tokens = {tok: first_special + i for i, tok in enumerate(special_tokens)}
        self.inverse_special = {i: tok for tok, i in self.special_tokens.items()}
        self._special_re = re.compile("(" + "|".join(re.escape(t) for t in special_tokens) + ")")
        self._cache: dict[str, list[int]] = {}

    # ---- training -----------------------------------------------------------

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int, verbose: bool = False,
              chat_template: str = "default") -> "Tokenizer":
        """Learn merges on chunk frequencies (fast: each unique chunk is processed once)."""
        special_tokens = CODE_SPECIAL_TOKENS if chat_template == "code" else SPECIAL_TOKENS
        n_merges = vocab_size - 256 - len(special_tokens)
        assert n_merges >= 0, f"vocab_size must be >= {256 + len(special_tokens)}"
        pattern = re.compile(SPLIT_PATTERN)
        chunk_counts: Counter[str] = Counter()
        for text in texts:
            chunk_counts.update(pattern.findall(text))
        words = [list(w.encode("utf-8")) for w in chunk_counts]
        freqs = list(chunk_counts.values())

        # pair -> total count, and pair -> indices of words containing it
        pair_counts: Counter[tuple[int, int]] = Counter()
        where: dict[tuple[int, int], set[int]] = {}
        for wi, (w, f) in enumerate(zip(words, freqs)):
            for p in zip(w, w[1:]):
                pair_counts[p] += f
                where.setdefault(p, set()).add(wi)

        merges: list[tuple[int, int]] = []
        for m in range(n_merges):
            if not pair_counts:
                break
            best = max(pair_counts, key=pair_counts.__getitem__)
            if pair_counts[best] < 2:
                break
            new_id = 256 + m
            merges.append(best)
            for wi in list(where.get(best, ())):
                w, f = words[wi], freqs[wi]
                for p in zip(w, w[1:]):
                    pair_counts[p] -= f
                    if pair_counts[p] <= 0:
                        del pair_counts[p]
                w = _merge(w, best, new_id)
                words[wi] = w
                for p in zip(w, w[1:]):
                    pair_counts[p] += f
                    where.setdefault(p, set()).add(wi)
            where.pop(best, None)
            if verbose and (m + 1) % 500 == 0:
                print(f"  merge {m + 1}/{n_merges}: {best} -> {new_id} (count {pair_counts.get(best, 0)})")
        return cls(merges, special_tokens, chat_template)

    # ---- encode / decode ----------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + len(self.special_tokens)

    @property
    def bos_id(self) -> int:
        return self.special_tokens["<|bos|>"]

    def special(self, name: str) -> int:
        """Id of a special token, e.g. tok.special("<|assistant_end|>")."""
        return self.special_tokens[name]

    def is_special(self, token_id: int) -> bool:
        return token_id in self.inverse_special

    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = _merge(ids, pair, self.merges[pair])
        if len(self._cache) < 500_000 and len(chunk) <= 64:
            self._cache[chunk] = ids
        return ids

    def _encode_ordinary(self, text: str) -> list[int]:
        out: list[int] = []
        for chunk in self.pattern.findall(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def encode(self, text: str, allow_special: bool = False) -> list[int]:
        if not allow_special:
            return self._encode_ordinary(text)
        out: list[int] = []
        for part in self._special_re.split(text):
            if part in self.special_tokens:
                out.append(self.special_tokens[part])
            elif part:
                out.extend(self._encode_ordinary(part))
        return out

    def token_bytes(self, token_id: int) -> bytes:
        """Raw bytes of one token (special tokens return their literal text)."""
        if token_id in self.inverse_special:
            return self.inverse_special[token_id].encode("utf-8")
        return self.vocab[token_id]

    def decode(self, ids: Iterable[int]) -> str:
        return b"".join(self.token_bytes(i) for i in ids).decode("utf-8", errors="replace")

    # ---- persistence --------------------------------------------------------

    def save(self, path: str | Path) -> None:
        inverse = {v: k for k, v in self.merges.items()}
        merges = [list(inverse[256 + i]) for i in range(len(self.merges))]
        data = {
            "version": 1,
            "pattern": SPLIT_PATTERN,
            "merges": merges,
            "special_tokens": list(self.special_tokens),
            "chat_template": self.chat_template,
        }
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        data = json.loads(Path(path).read_text())
        return cls([tuple(m) for m in data["merges"]], data["special_tokens"], data.get("chat_template", "default"))
