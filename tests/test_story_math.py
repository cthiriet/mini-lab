import random
import re
from pathlib import Path

import pytest

from minilab.data import story_math as sm
from minilab.tokenizer.bpe import Tokenizer
from minilab.tokenizer.chat import render_conversation

LEVELS = ("low", "medium", "high")
TOKENIZER = Path("/Users/clementthiriet/VM/mini-lab/runs/cand5k/tokenizer.json")


def problems(n, split="train", seed=0, ks=range(1, 7)):
    rng = random.Random(seed)
    return [sm.make(rng, rng.choice(list(ks)), split) for _ in range(n)]


@pytest.mark.parametrize("split,ks", [("train", range(1, 7)), ("heldout", range(1, 9))])
def test_problems_are_valid(split, ks):
    for p in problems(10_000, split, ks=ks):
        states = p["states"]
        assert len(states) == p["steps"] + 1 and p["steps"] in ks
        assert all(0 <= s <= 99 for s in states) and states[0] >= 2
        assert p["answer"] == states[-1]
        assert p["bucket"] == sm.bucket(p) == (p["steps"], int(any(op == "dbl" for op, _ in p["ops"])))
        assert (split == "heldout") or all(op != "half" for op, _ in p["ops"])


def test_problem_sentences():
    p = sm.make(random.Random(3), 4, distractor=False)
    assert p["ignore"] is None and p["messages"][0]["content"].count(".") >= 5
    assert sm.make(random.Random(3), 4, distractor=True)["ignore"]
    assert all(p["ignore"] for p in problems(200, "heldout"))
    share = sum(p["ignore"] is not None for p in problems(2000)) / 2000
    assert 0.3 < share < 0.4
    assert sm.answer_text(5) == "The answer is 5."


def test_heldout_is_disjoint_from_train():
    def words(split):
        return {w for p in problems(1000, split) for w in re.findall(r"[A-Za-z']+", p["messages"][0]["content"])}
    held, train = words("heldout"), words("train")
    held_only = set(sm.HELD_HEROES + sm.HELD_OTHERS) | {w for t in sm.HELD_THINGS for w in t}
    train_only = set(sm.HEROES + sm.OTHERS) | {w for t in sm.THINGS for w in t}
    assert not held_only & train_only and held_only <= held
    assert not held & train_only and not train & held_only
    assert any(op == "half" for p in problems(200, "heldout", ks=[4]) for op, _ in p["ops"])


def test_same_seed_same_problem():
    assert problems(50, seed=7) == problems(50, seed=7) != problems(50, seed=8)
    p = problems(1, seed=5)[0]
    assert all(sm.trace_program(p, level, random.Random(1)) == sm.trace_program(p, level, random.Random(1))
               for level in LEVELS)


@pytest.mark.parametrize("level", LEVELS)
def test_program_traces_pass_the_checks(level):
    rng = random.Random(1)
    for p in problems(500, ks=range(1, 7)) + problems(100, "heldout", ks=range(1, 9)):
        text = sm.trace_program(p, level, rng)
        assert sm.verify_trace(p, text, level) is None, (text, p["messages"][0]["content"])
        assert sm.style_ok(level, text) and sm.normalize(text) == text
    assert sm.style_ok("low", "3+5=8\n8-2=6") and not sm.style_ok("low", "3 + 5 = 8")
    assert not sm.style_ok("medium", "Lily has 3. She finds 5, so 3 + 5 = 8.\nDone.")
    assert not sm.style_ok("high", "Lily has 3. She finds 5, so 3 + 5 = 8. So Lily has 8 apples.")


def test_program_wording_varies():
    p = sm.make(random.Random(2), 3, distractor=True)
    rng = random.Random(0)
    for level in ("medium", "high"):
        assert len({sm.trace_program(p, level, rng) for _ in range(50)}) > 20
    assert "not " in sm.trace_program(p, "high", rng) and sm.trace_program(p, "high", rng).count("Check:") == 1


def test_verify_trace_rejects():
    p = sm.make(random.Random(0), 3, distractor=False)
    b = p["states"][1]
    for level in LEVELS:
        good = sm.trace_program(p, level, random.Random(4))
        assert sm.verify_trace(p, good, level) is None
        sep = "\n" if level == "low" else " "
        wrong = good.replace(f"= {b}", f"= {b + 1}") if level != "low" else good.replace(f"={b}", f"={b + 1}", 1)
        assert sm.verify_trace(p, wrong, level).startswith("wrong equation")
        assert sm.verify_trace(p, good + sep + "77", level) == "invented number: 77"
        assert sm.verify_trace(p, good + sep + "five", level) == "number word"
    low = sm.trace_program(p, "low", random.Random(4))
    assert sm.verify_trace(p, "\n".join(low.splitlines()[::2]), "low") == "a state is missing"
    assert sm.verify_trace(p, low.replace("-", "\u2212"), "low") == "forbidden text"
    assert sm.verify_trace(p, "The answer is 5. " + low, "low") == "forbidden text"
    medium = sm.trace_program(p, "medium", random.Random(4)).split(". ")
    assert sm.verify_trace(p, ". ".join(medium[:1] + medium[2:]), "medium") == "a state is missing"
    assert sm.verify_trace(p, ". ".join(medium), "high") == "style"
    q = sm.make(random.Random(0), 3, distractor=True)
    high = sm.trace_program(q, "high", random.Random(4))
    assert sm.verify_trace(q, re.sub(r"[^.]*not (matter|change)[^.]*\.", "", high), "high") == "the distractor is not set aside"
    assert sm.verify_trace(q, high.replace(f"So {q['hero']} has", "So Zed has"), "high") == "wrong conclusion"


def test_normalize_and_dedupe():
    assert sm.normalize(" 3 ×  2 =  6 \n\n 6 − 1 = 5 ") == "3 x 2 = 6\n6 - 1 = 5"
    traces = ["a 1 b", "a 2 b", "a 3 b", "c 1"] * 50  # 150 of one skeleton, 50 of another
    kept = sm.dedupe_skeletons(traces, max_frac=0.1)  # 200 traces, two skeletons: 20 of each
    assert len(kept) == 40 and kept[:3] == traces[:3]


def test_think_follows_the_threshold():
    policy = {"tau": {"low": 0.8, "medium": 0.92, "high": 0.99}, "p": {"1,0": 0.99, "2,0": 0.85, "3,0": 0.5}, "n": 300}
    mk = lambda k: {"steps": k, "ops": [("add", 1)] * k}  # noqa: E731
    assert not sm.think("low", mk(1), policy) and not sm.think("medium", mk(1), policy)
    assert sm.think("high", mk(1), policy)  # 0.99 is not 0.03 above 0.99
    assert not sm.think("low", mk(2), policy)  # 0.85 >= 0.80 + 0.03
    assert sm.think("medium", mk(2), policy)  # under the threshold
    assert sm.think("low", {"steps": 2, "ops": [("add", 1), ("dbl", None)]}, policy)  # bucket (2, 1) is absent
    assert sm.think("low", mk(3), policy) and sm.think("high", mk(3), policy)
    assert not any(sm.think("none", mk(k), policy) for k in (1, 2, 3, 6))
    gray = {**policy, "p": {"2,0": 0.82}}  # within 0.03 of tau_low
    assert sm.think("low", mk(2), gray)


def test_load_policy(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text('{"tau": {"low": 0.8}, "p": {"1,0": 1.0}, "n": 300}')
    assert sm.load_policy(path) == {"tau": {"low": 0.8}, "p": {"1,0": 1.0}, "n": 300}


@pytest.mark.skipif(not TOKENIZER.exists(), reason="runs/cand5k/tokenizer.json is not here")
def test_lengths_in_tokens(capsys):
    tok = Tokenizer.load(TOKENIZER)
    rng = random.Random(0)
    table = {}
    for k in range(1, 9):
        for level in LEVELS:
            lengths = []
            for _ in range(100):
                p = sm.make(rng, k)
                reasoning = sm.trace_program(p, level, rng)
                messages = [p["messages"][0], {"role": "assistant", "content": sm.answer_text(p["answer"]), "reasoning": reasoning}]
                lengths.append(len(render_conversation(tok, messages, None)[0]))
            table[k, level] = max(lengths)
    with capsys.disabled():
        print("\nmax conversation tokens (k: low medium high)")
        for k in range(1, 9):
            print(k, *(table[k, level] for level in LEVELS))
    assert table[6, "high"] <= 350 and max(table.values()) <= 500
    # most program traces also satisfy the teacher's length rule
    ok = [sm.verify_trace(p, t, "high", tok) is None
          for p in problems(300) for t in [sm.trace_program(p, "high", rng)]]
    assert sum(ok) / len(ok) > 0.95
