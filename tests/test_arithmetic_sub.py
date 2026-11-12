import hashlib
import random

from minilab.data import arithmetic
from minilab.data.arithmetic import pretrain_document, sub_scratchpad
from minilab.data.loader import pretrain_documents
from tests.test_data import STORIES, tok  # noqa: F401  (tok is a fixture)

# sha256 of 200 documents of pretrain_document(random.Random(0), [1, 2, 3]) before sub_frac existed
ADDITION_ONLY_HASH = "9d5244efcc91ce1d3a9f7794dbe7348edf7d21c78b8a2384207c51fd9399c28f"


def test_sub_scratchpad_examples():
    assert sub_scratchpad(74, 21) == "74-21: 4-1=3, 3\n7-2: 7-2=5, 53"
    assert sub_scratchpad(70, 32) == "70-32: 10-2=8, 8\n7-3: 7-3-1=3, 38"
    assert sub_scratchpad(105, 100).splitlines()[-1] == "1-1: 1-1=0, 5"


def test_sub_scratchpad_is_right():
    rng = random.Random(0)
    for _ in range(5000):
        a = rng.randint(0, 999)
        b = rng.randint(0, a)
        lines = sub_scratchpad(a, b).splitlines()
        assert lines[-1].endswith(f", {a - b}")
        for i, line in enumerate(lines):  # every line recomputes its column
            head, calc = line.split(": ")
            expr, rest = calc.split("=")
            terms = [int(t) for t in expr.replace("-", " ").split()]
            assert terms[0] - sum(terms[1:]) == int(rest.split(",")[0])
            assert (a // 10**i) % 10 + 10 * (terms[0] >= 10) == terms[0]


def test_pretrain_document_without_subtractions_is_unchanged():
    rng, h = random.Random(0), hashlib.sha256()
    for _ in range(200):
        h.update(pretrain_document(rng, [1, 2, 3]).encode() + b"\0")
    assert h.hexdigest() == ADDITION_ONLY_HASH
    r1, r2 = random.Random(3), random.Random(3)
    assert [pretrain_document(r1, [1, 2]) for _ in range(20)] == [pretrain_document(r2, [1, 2], 0.0) for _ in range(20)]
    assert r1.random() == r2.random()


def test_pretrain_document_with_only_subtractions():
    rng, seen = random.Random(1), 0
    for _ in range(300):
        for line in pretrain_document(rng, [1, 2, 3], sub_frac=1.0).split("\n"):
            if line.startswith("So "):
                a, b, c = (int(x) for x in line[3:-1].replace(" - ", " ").replace(" = ", " ").split())
                assert a - b == c and a >= b
                seen += 1
    assert seen > 200
    assert "+" not in pretrain_document(random.Random(2), [1, 2, 3], sub_frac=1.0)


def test_midtrain_text_has_subtractions(tok):
    def text(sub_frac):
        docs = pretrain_documents(tok, STORIES, [1, 2, 3], 0.9, seed=0, sub_frac=sub_frac)
        return "\n".join(tok.decode(next(docs)) for _ in range(100))

    assert " - " in text(0.5)
    assert " - " not in text(0.0)
