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
        ops = [op for op, _ in p["ops"]]
        assert p["bucket"] == sm.bucket(p) == (f"1,{ops[0]}" if p["steps"] == 1 else f"{p['steps']},{int('dbl' in ops)}")
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
        return {w for p in problems(5000, split) for w in re.findall(r"[A-Za-z']+", p["messages"][0]["content"])}
    held, train = words("heldout"), words("train")
    held_only = set(sm.HELD_HEROES + sm.HELD_OTHERS) | {w for t in sm.HELD_THINGS for w in t}
    train_only = set(sm.HEROES + sm.OTHERS) | {w for t in sm.THINGS for w in t}
    assert not held_only & train_only and held_only <= held
    assert not held & train_only and not train & held_only
    assert any(op == "half" for p in problems(200, "heldout", ks=[4]) for op, _ in p["ops"])


def test_heldout_problems_are_frozen():
    """The eval's problems must not move (no new draw, no new sentence): two of them, written out."""
    assert sm.make(random.Random(0), 4, "heldout")["messages"][0]["content"] == (
        "Kyle has 3 whistles. Kurt brings him 17 more. He shares half of his whistles with Gail. Hana has 9 crowns. "
        "Kyle drops 5. He drops 5. How many whistles does Kyle have in all?")
    assert sm.make(random.Random(1), 4, "heldout")["messages"][0]["content"] == (
        "Maya has 4 daisies. Nell brings her 16 more. Then her pile doubles. Maya drops 1. Maya loses 7. "
        "Omar has 7 lamps. How many daisies does Maya have left?")


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
    assert not sm.style_ok("high", "Lily has 3. She finds 5, so 3 + 5 = 8. So Lily has 8 apples.")  # no quotes
    assert not sm.style_ok("high", 'Lily has 3. "She finds 5." So 3 + 5 = 8. Check: 3 + 5 = 8. So Lily has 8 apples.')


def test_program_wording_varies():
    p = sm.make(random.Random(2), 3, distractor=True)
    rng = random.Random(0)
    for level in ("medium", "high"):
        assert len({sm.trace_program(p, level, rng) for _ in range(50)}) > 20
    assert "not " in sm.trace_program(p, "high", rng) or "nothing" in sm.trace_program(p, "high", rng)
    assert "Check" not in sm.trace_program(p, "high", rng)


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
    q = next(q for q in (sm.make(random.Random(i), 3, distractor=True) for i in range(20)) if q["at"] < 3)
    high = sm.trace_program(q, "high", random.Random(4))
    quote = q["sentences"][q["at"]]
    effect = high[high.index(quote) + len(quote) + 2:].split(". ")[0] + ". "
    assert "not " in effect or "nothing" in effect
    assert sm.verify_trace(q, high.replace(effect, "", 1), "high") == "the distractor is not set aside"
    assert sm.verify_trace(q, high.replace(f"So {q['hero']} has", "So Zed has"), "high") == "wrong conclusion"


def test_high_reads_the_story_again():
    """Each sentence of the story, in order and as worded, is quoted and followed by its effect."""
    for p in problems(500, ks=range(1, 9)) + problems(500, "heldout", ks=range(1, 9)):
        content = p["messages"][0]["content"]
        start = content[:content.index(".") + 1]
        # the start, then the sentences, then the question: what the story reads is exactly what is quoted
        assert content.startswith(" ".join([start, *p["sentences"]]) + " ")
        assert len(p["sentences"]) == p["steps"] + (p["ignore"] is not None)
        text = sm.trace_program(p, "high", random.Random(3))
        assert re.findall(r'"([^"]*)"', text) == p["sentences"]
        assert sm.verify_trace(p, text, "high") is None and sm.style_ok("high", text)
    p = sm.make(random.Random(1), 3, distractor=True)
    text = sm.trace_program(p, "high", random.Random(3))
    a, b, first = [f'"{s}"' for s in p["sentences"][:2]] + [p["sentences"][0]]
    missing = text.replace(a + " ", "", 1)  # a citation, and its effect word, gone
    swapped = text.replace(a, "@", 1).replace(b, a, 1).replace("@", b, 1)
    altered = text.replace(first, first[:-1] + " today.", 1)
    for bad in (missing, swapped, altered):
        assert sm.verify_trace(p, bad, "high") is not None


def test_normalize_and_dedupe():
    assert sm.normalize(" 3 ×  2 =  6 \n\n 6 − 1 = 5 ") == "3 x 2 = 6\n6 - 1 = 5"
    traces = ["a 1 b", "a 2 b", "a 3 b", "c 1"] * 50  # 150 of one skeleton, 50 of another
    kept = sm.dedupe_skeletons(traces, max_frac=0.1)  # 200 traces, two skeletons: 20 of each
    assert len(kept) == 40 and kept[:3] == traces[:3]


def test_think_follows_the_threshold():
    policy = {"tau": {"low": 0.8, "medium": 0.92, "high": 0.99}, "p": {"1,add": 0.99, "2,0": 0.85, "3,0": 0.5}, "n": 300}
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
    # the program's traces satisfy the length rule (30 k + 60 tokens at most for high), on both splits
    ok = [sm.verify_trace(p, t, "high", tok) is None
          for split, ks in (("train", range(1, 7)), ("heldout", range(1, 9))) for p in problems(300, split, ks=ks)
          for t in [sm.trace_program(p, "high", rng)]]
    assert sum(ok) / len(ok) > 0.99


def test_problems_read_unambiguously():
    """No two doublings in a row; the event after a distractor names the hero (not "She")."""
    rng = random.Random(7)
    for _ in range(2000):
        p = sm.make(rng, rng.randint(2, 8), rng.choice(["train", "heldout"]), distractor=True)
        assert "{" not in p["messages"][0]["content"] and "}" not in " ".join(p["clauses"])
        assert all(not (a == b == "dbl") for (a, _), (b, _) in zip(p["ops"], p["ops"][1:]))
        if p["at"] < p["steps"]:
            after = sm._sentences(p["messages"][0]["content"])[p["at"] + 2]
            assert after.startswith(p["hero"] + " ")


def test_heldout_lists_are_wide():
    assert min(len(sm.HELD_HEROES), len(sm.HELD_OTHERS), len(sm.HELD_THINGS)) >= 20
    assert all(plural == thing + ("es" if thing.endswith(("ch", "sh")) else "s") or plural.endswith("ies")
               for thing, plural in sm.HELD_THINGS)


@pytest.mark.parametrize("level", LEVELS)
def test_anchor_round_trip(level):
    rng = random.Random(2)
    for p in problems(200, ks=range(1, 7)):
        text = sm.trace_program(p, level, rng)
        anchored = sm.anchor(p, level, text)
        assert sm.strip_anchor(level, anchored) == text and sm.strip_anchor(level, text) == text
        assert (anchored == text) == (level == "high")
        assert sm.style_ok(level, sm.strip_anchor(level, anchored))


def test_anchor_format():
    p = sm.make(random.Random(0), 2)
    assert sm.anchor(p, "low", "11-2=9\n9+4=13") == "2 events:\n1: 11-2=9\n2: 9+4=13"
    text = "Sam had 4 flowers. He gets 6 from Kate, so 4 + 6 = 10. Minus 10: 10 - 10 = 0. Plus 17: 0 + 17 = 17."
    assert sm.anchor(sm.make(random.Random(0), 3), "medium", text) == (
        "3 events. Sam had 4 flowers. 1: He gets 6 from Kate, so 4 + 6 = 10. 2: Minus 10: 10 - 10 = 0. 3: Plus 17: 0 + 17 = 17.")
    assert sm.anchor(p, "medium", "Tom has 3. He finds 2.") == "2 events. Tom has 3. He finds 2."
    assert sm.strip_anchor("low", "2 events:\n1: 11-2=9\n2: 9+4=13") == "11-2=9\n9+4=13"


def test_calculator_triggers_after_an_anchor():
    from minilab.data.arithmetic import think_result
    assert think_result("2 events:\n1: 11-2=") == "9" and think_result("2 events:\n1: 11-2=9\n2: 9+4=") == "13"
    assert think_result("3 events. Sam had 4 flowers. 1: He gets 6 from Kate, so 4 + 6 =") == " 10"
    assert think_result("3 events. Sam had 4. 1: He gets 6, so 4 + 6 = 10. 2: Minus 10: 10 - 10 =") == " 0"


def test_medium_round_trip_on_teacher_traces():
    import json
    import os
    path = Path(os.environ.get("TEACHER_TRACES", "/Users/clementthiriet/VM/mini-lab/.claude/worktrees/"
                               "agent-a95ab00c560bea7de/data/reason/traces_v2.jsonl.dedup.jsonl"))
    if not path.is_file():
        pytest.skip("no teacher file")
    rows = [json.loads(line) for line in path.read_text().splitlines()[:300]]
    rows = [r for r in rows if r["level"] == "medium" and r["ok"]]
    assert rows
    for r in rows:
        anchored = sm.anchor(r["problem"], "medium", r["text"])
        assert sm.strip_anchor("medium", anchored) == r["text"]
        numbered = re.findall(r"(?:^|(?<=[.!?] ))(\d+): ", anchored.split(" events. ", 1)[1])
        assert numbered == [str(i) for i in range(1, r["text"].count("=") + 1)]
        assert sm.style_ok("medium", sm.strip_anchor("medium", anchored))


def test_medium_numbers_each_equation_in_order():
    rng = random.Random(5)
    for p in problems(100, ks=range(1, 7)):
        anchored = sm.anchor(p, "medium", sm.trace_program(p, "medium", rng))
        assert anchored.startswith(f"{p['steps']} events. ")
        assert [int(n) for n in re.findall(r"(?:^|(?<=[.!?] ))(\d+): ", anchored.split(" events. ", 1)[1])] == list(range(1, p["steps"] + 1))
        assert all(("=" in s) == bool(re.match(r"\d+: ", s)) for s in sm._sentences(anchored)[1:])


def _names(p):
    """The capitalized words of the story that are not common sentence starters."""
    return set(re.findall(r"[A-Z][a-z]+", p["messages"][0]["content"])) - {"How", "Then", "The", "She", "He", "It", "At"}


def test_wide_lexicon_is_disjoint_and_regular():
    heroes, things = sm.TRAIN_EXTRA_HEROES, [w for t in sm.TRAIN_EXTRA_THINGS for w in t]
    old = {*sm.HEROES, *sm.OTHERS, *sm.HELD_HEROES, *sm.HELD_OTHERS, *(w for t in sm.THINGS + sm.HELD_THINGS for w in t)}
    assert len(heroes) == len(set(heroes)) and len(sm.TRAIN_EXTRA_THINGS) == len({t for t, _ in sm.TRAIN_EXTRA_THINGS})
    assert len(heroes) >= 150 and len(sm.TRAIN_EXTRA_THINGS) >= 200
    assert not old & {*heroes, *things} and not {h.lower() for h in heroes} & set(things)
    assert set(sm.TRAIN_EXTRA_FEMALE) <= sm.FEMALE and not set(sm.TRAIN_EXTRA_MALE) & sm.FEMALE
    assert all(p == t + "s" or p == t + "es" or p == t[:-1] + "ies" for t, p in sm.TRAIN_EXTRA_THINGS)


def test_wide_is_off_by_default_and_draws_the_wide_lexicon():
    for seed in range(5):
        a, b = random.Random(seed), random.Random(seed)
        assert sm.make(a, 5, "train", wide=False) == sm.make(b, 5, "train") and a.random() == b.random()
    rng = random.Random(7)
    ps = [sm.make(rng, 4, "train", distractor=True, wide=True) for _ in range(400)]
    assert any(p["hero"] in sm.TRAIN_EXTRA_HEROES for p in ps) and any(tuple(p["thing"]) in sm.TRAIN_EXTRA_THINGS for p in ps)
    assert any(w in sm.TRAIN_EXTRA_HEROES and w != p["hero"] for p in ps for w in _names(p))
    for level in LEVELS:
        assert all(sm.verify_trace(p, sm.trace_program(p, level, rng), level) is None for p in ps)


def test_pseudo_other_names():
    rng = random.Random(11)
    real = {*sm.HEROES, *sm.OTHERS, *sm.HELD_HEROES, *sm.HELD_OTHERS, *sm.TRAIN_EXTRA_HEROES}
    ps = [sm.make(rng, 4, "train", distractor=True, pseudo=1.0, wide=True) for _ in range(200)]
    assert sum(any(w not in real and w != p["hero"] for w in _names(p)) for p in ps) > 80  # when an event or the distractor names someone else
    assert all(p["hero"] not in real for p in ps)
    assert len({len(sm._pseudo(rng, True)) for _ in range(200)}) >= 4  # one to four syllables


def test_pseudo_is_off_by_default_and_frozen():
    assert sm.make(random.Random(0), 4, "train")["messages"][0]["content"] == (
        "Kai has 3 plums. The sky is blue. Kai picks up 17 more. Then he has twice as many plums as before. "
        "Then he eats 19. The number of his plums doubles. How many plums will Kai have then?")
    for seed in range(5):
        a, b = random.Random(seed), random.Random(seed)
        assert sm.make(a, 5, "train", pseudo=0.0) == sm.make(b, 5, "train") and a.random() == b.random()


def test_pseudo_words_and_names():
    rng = random.Random(3)
    real = {*sm.HEROES, *sm.OTHERS, *sm.HELD_HEROES, *sm.HELD_OTHERS, *(w for t in sm.THINGS + sm.HELD_THINGS for w in t)}
    ps = [sm.make(rng, 3, "train", pseudo=1.0) for _ in range(300)]
    assert all(p["hero"] not in real and p["thing"][0] not in real and p["thing"][1] == p["thing"][0] + "s" for p in ps)
    assert all(p["hero"].istitle() and p["thing"][1] in p["messages"][0]["content"] for p in ps)
    some = [sm.make(rng, 3, "train", pseudo=0.5) for _ in range(300)]
    assert 0.3 < sum(p["thing"][0] not in real for p in some) / 300 < 0.7
    assert 0.3 < sum(p["hero"] not in real for p in some) / 300 < 0.7
    for level in LEVELS:
        assert all(sm.verify_trace(p, sm.trace_program(p, level, rng), level) is None for p in ps)
