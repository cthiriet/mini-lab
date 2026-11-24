"""Reasoning traces for story problems, written by a large LLM through OpenRouter (docs/story-math-spec.md §4).

Historical: this script wrote the bank of medium traces that training used up to E8. Since E9 training
uses no teacher (teacher_frac = 0 in configs/prelude.toml): story_math.trace_program writes every
reasoning, at the two levels that remain, low and high. The "high" prompt below, with its "Check:"
sentence, is the old format: story_math.verify_trace now rejects a "Check:" and wants the quoted story
sentences of the current high (reread sentence by sentence), so a new run at high would fail the filter
until the prompt is rewritten. The script is kept as it was.

    op run -- uv run python scripts/teacher_traces.py --n 300 --levels medium high --budget-usd 0.5

Reads OPENROUTER_API_KEY from the environment. Checks first that the model has a zero-data-retention
endpoint, then asks for one trace per (problem, level), checks it with story_math.verify_trace, and
appends one JSON line per try to --out (a failed try is replayed once, at temperature 1.0). The run
stops, cleanly, once the cost (the resumed file's included) reaches --budget-usd.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import sys
import time
import tomllib
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

from minilab.data import story_math as sm
from minilab.tokenizer.bpe import Tokenizer

API = "https://openrouter.ai/api/v1"
BACKOFF = 1.0  # seconds, doubled after each retry of a 429 / 5xx

SYSTEM = (
    "You write the private scratch thoughts of a small, careful assistant that solves a short story problem about "
    "counting things. You get the problem. Write ONLY the thoughts: no greeting, no final answer line, no markdown, "
    "no bullet points, no headings. Use plain ASCII only. Write every number with digits (8, never eight). Write "
    'operations with + - x and =, with the symbol x for "times". Every number you write must be correct. Never write '
    '"The answer is". Follow the style below exactly.')
# The styles of the old formats: medium (until E8, the level that was trained) and high with "Check:" (dropped in E9).
STYLE = {
    "medium": (
        "Style: short and plain, like a quick note to oneself. First sentence: who starts with how many of what. Then "
        "ONE short sentence per event that changes the count (at most 12 words), ending with its equation, for example "
        '"She finds 5, so 3 + 5 = 8." Skip events that change nothing. No introduction and no conclusion: the last '
        "sentence is the last equation. Vary your wording from one problem to the next (finds, gets, picks up, "
        "receives...). Use the character's name or She/He."),
    "high": (
        "Style: a careful person thinking out loud, in complete sentences. Begin by saying who has how many of what. "
        "Then go through the events in order, one or two sentences each: say what happens, then the calculation with "
        'the new total ("so she has 3 + 5 = 8 apples"). If a sentence of the problem does not change the count, say in '
        'a few words that it does not matter. Near the end, add one sentence starting with "Check:" that goes through '
        "the whole story again in a single line of arithmetic. Finish with a sentence stating how many she has, for "
        'example "So Lily has 5 apples." Between 30 and 120 words. Vary your wording from one problem to the next.'),
}
USER = "Problem: {text}\nThoughts:"

# Two fixed examples per level, written by hand (invented problems, never drawn from an eval seed).
_LILY = ("Lily has 3 apples. She finds 5 more. Tom has 4 pears. She gives 2 to Tom. Then she eats 1. "
         "How many apples does Lily have now?")
_BEN = "Ben has 6 stickers. The sky is blue. He buys 4 more. Then his pile doubles. How many stickers does Ben have in all?"
EXAMPLES = {
    "medium": [
        (_LILY, "Lily starts with 3 apples. She finds 5, so 3 + 5 = 8. She gives 2 to Tom, so 8 - 2 = 6. "
                "She eats 1, so 6 - 1 = 5."),
        (_BEN, "Ben starts with 6 stickers. He buys 4, so 6 + 4 = 10. He doubles his stickers, so 10 x 2 = 20."),
    ],
    "high": [
        (_LILY, "Lily begins with 3 apples. She finds 5 more, so she has 3 + 5 = 8 apples. Tom's pears do not matter. "
                "She gives 2 apples to Tom, so 8 - 2 = 6. Then she eats 1, so 6 - 1 = 5 apples. "
                "Check: 3 + 5 - 2 - 1 = 5. So Lily has 5 apples."),
        (_BEN, "Ben begins with 6 stickers. He buys 4 more, so he has 6 + 4 = 10 stickers. The sky does not matter, "
               "it changes nothing. Then his pile doubles, so he has 10 x 2 = 20 stickers. "
               "Check: (6 + 4) x 2 = 20. So Ben has 20 stickers."),
    ],
}


def messages(level: str, problem: dict) -> list[dict]:
    chat = [{"role": "system", "content": f"{SYSTEM}\n\n{STYLE[level]}"}]
    for text, trace in EXAMPLES[level]:
        chat += [{"role": "user", "content": USER.format(text=text)}, {"role": "assistant", "content": trace}]
    return [*chat, {"role": "user", "content": USER.format(text=problem["messages"][0]["content"])}]


def body(model: str, level: str, problem: dict, temperature: float) -> dict:
    return {"model": model, "messages": messages(level, problem), "temperature": temperature, "max_tokens": 400,
            "reasoning": {"enabled": False}, "usage": {"include": True},
            "provider": {"zdr": True, "data_collection": "deny", "allow_fallbacks": True}}


# ---- http (the only code that touches the network; tests give `run` their own `post`) -------------

def http_post(key: str):
    def post(payload: dict) -> tuple[int, dict]:
        req = urllib.request.Request(f"{API}/chat/completions", json.dumps(payload).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            return e.code, {}
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return 599, {}  # network trouble: retried like a 5xx
    return post


def fetch_zdr() -> list[dict]:
    with urllib.request.urlopen(f"{API}/endpoints/zdr", timeout=60) as r:
        return json.load(r)["data"]


def has_zdr(endpoints: list[dict], model: str) -> bool:
    """Does one of the zero-data-retention endpoints serve `model`?"""
    return any(e.get("model_id") == model for e in endpoints)


def call(post, payload: dict) -> tuple[int, dict]:
    """One request, with exponential backoff (3 tries) on 429 and 5xx."""
    for attempt in range(3):
        status, data = post(payload)
        if status != 429 and status < 500:
            break
        if attempt < 2:
            time.sleep(BACKOFF * 2 ** attempt)
    return status, data


# ---- the run -------------------------------------------------------------------------------------

def problems(seed: int, n: int, split: str, k_weights: list[float], min_k: int) -> list[dict]:
    """n problems, the same for the same seed; their id is "<seed>-<k>-<idx>". Only k >= min_k are kept."""
    rng, out = random.Random(seed), []
    for idx in range(n):
        k = rng.choices(range(1, len(k_weights) + 1), k_weights)[0]
        p = sm.make(rng, k, split)
        if k >= min_k:
            out.append({**p, "id": f"{seed}-{k}-{idx}"})
    return out


def read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def attempt(post, model: str, level: str, problem: dict, temperature: float, tok) -> dict:
    """One try: a JSONL row (spec §4.3, plus the problem itself)."""
    row = {"id": problem["id"], "level": level, "model": model, "raw": "", "text": "", "ok": False, "defects": [],
           "cost": 0.0, "problem": problem}
    status, data = call(post, body(model, level, problem, temperature))
    if status != 200:
        return {**row, "defects": [f"http {status}"]}
    row["cost"] = float((data.get("usage") or {}).get("cost") or 0)
    try:
        raw = data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return {**row, "defects": ["bad response"]}
    text = sm.normalize(raw)
    reason = sm.verify_trace(problem, text, level, tok)
    return {**row, "raw": raw, "text": text, "ok": reason is None, "defects": [reason] if reason else []}


def run(post, args, tok=None) -> float:
    """Ask for every missing trace, `args.workers` at a time, appending a row per try. Once the cost reaches
    the budget no new call is made (those in flight finish). Returns the cumulative cost."""
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    old = read_lines(out) if args.resume else []
    done = {(r["id"], r["level"]) for r in old if r["ok"]}
    todo = [(p, level) for p in problems(args.seed, args.n, args.split, args.k_weights, args.min_k)
            for level in args.levels if (p["id"], level) not in done]
    spent, lock = sum(r["cost"] for r in old), Lock()

    with open(out, "a" if args.resume else "w") as f:
        def work(item):
            nonlocal spent
            problem, level = item
            for temperature in (0.8, 1.0):  # a failed try is replayed once, hotter
                with lock:
                    if spent >= args.budget_usd:
                        return
                row = attempt(post, args.model, level, problem, temperature, tok)
                with lock:
                    spent += row["cost"]
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                if row["ok"] or row["defects"][0].startswith(("http", "bad response")):
                    return  # a transport failure is not replayed: a hotter sample would not help
        with ThreadPoolExecutor(args.workers) as pool:
            list(pool.map(work, todo))
    if spent >= args.budget_usd:
        print(f"budget reached: ${spent:.4f} of ${args.budget_usd} spent, stopped. Only the owner raises the cap.")
    return spent


def summary(rows: list[dict], tok=None) -> str:
    lines, medians = [], {}
    for level in ("medium", "high"):
        mine = [r for r in rows if r["level"] == level]
        if not mine:
            continue
        ok = [r for r in mine if r["ok"]]
        why = Counter(d.split(":")[0] for r in mine for d in r["defects"]).most_common(3)
        skeletons = len({re.sub(r"\d+", "#", r["text"]) for r in ok})
        line = (f"{level}: {len(mine)} tries, {len(ok)} accepted ({len(ok) / len(mine):.0%}), "
                f"${sum(r['cost'] for r in mine):.4f}, {skeletons} distinct skeletons, rejections: {why or 'none'}")
        if tok and ok:
            medians[level] = statistics.median(len(tok.encode(r["text"])) for r in ok)
            line += f", median {medians[level]:.0f} tokens"
        lines.append(line)
    if len(medians) == 2:
        lines.append(f"high / medium median length: {medians['high'] / medians['medium']:.2f}")
    return "\n".join(lines)


def dedupe(rows: list[dict], out: Path) -> int:
    """Write the accepted rows, minus the surplus of any one skeleton (story_math.dedupe_skeletons)."""
    ok, kept = [r for r in rows if r["ok"]], Counter()
    for level in {r["level"] for r in ok}:
        kept.update((level, t) for t in sm.dedupe_skeletons([r["text"] for r in ok if r["level"] == level]))
    n = 0
    with open(out, "w") as f:
        for r in ok:
            if kept[(r["level"], r["text"])] > 0:
                kept[(r["level"], r["text"])] -= 1
                f.write(json.dumps(r) + "\n")
                n += 1
    return n


def parse(argv=None) -> argparse.Namespace:
    cfg = tomllib.loads(Path("configs/prelude.toml").read_text())["data"]["reason"]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, required=True, help="problems to draw (those with k < --min-k get no trace)")
    p.add_argument("--levels", nargs="+", default=["medium", "high"], choices=["medium", "high"])
    p.add_argument("--model", default="deepseek/deepseek-v4.1-flash")
    p.add_argument("--budget-usd", type=float, default=8.0, help="hard cap on the cumulative cost")
    p.add_argument("--out", default=cfg["traces"])
    p.add_argument("--resume", action="store_true", help="skip the (id, level) already ok in --out")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=cfg["train_seed"])
    p.add_argument("--split", default="train", choices=["train", "heldout"])
    p.add_argument("--min-k", type=int, default=2)
    p.add_argument("--tokenizer", default="runs/cand5k/tokenizer.json", help="for the length filter")
    p.add_argument("--dedupe", action="store_true", help="also write <out>.dedup.jsonl")
    args = p.parse_args(argv)
    args.k_weights = cfg["k_weights"]
    return args


def main(argv=None, post=None, endpoints=None) -> None:
    args = parse(argv)
    key = os.environ.get("OPENROUTER_API_KEY")
    if post is None and not key:
        sys.exit("OPENROUTER_API_KEY is not set (run through `op run --`)")
    if not has_zdr(endpoints if endpoints is not None else fetch_zdr(), args.model):
        sys.exit(f"{args.model} has no zero-data-retention endpoint ({API}/endpoints/zdr): stopped, nothing was sent")
    tok = Tokenizer.load(args.tokenizer)
    run(post or http_post(key), args, tok)
    rows = read_lines(Path(args.out))
    print(summary(rows, tok))
    if args.dedupe:
        out = Path(str(args.out) + ".dedup.jsonl")
        print(f"{dedupe(rows, out)} traces kept after skeleton dedupe -> {out}")


if __name__ == "__main__":
    main()
