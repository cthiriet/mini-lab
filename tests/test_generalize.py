"""The generalization panel (eval/generalize.py): its never-seen versions are never trained on."""

import random
import re
from dataclasses import replace

from minilab.data import arithmetic, code
from minilab.data.conversations import TOPIC_REQUESTS, TOPICS
from minilab.eval import generalize as g


def test_new_questions_are_not_the_trained_ones():
    assert not set(g.QUESTIONS) & set(arithmetic.QUESTIONS)
    for two, three in g.THREE:
        assert two in arithmetic.QUESTIONS            # the trained version is a training question
        assert three.replace(" + {c}", "").replace(" plus {c}", "") == two
    texts = [arithmetic.question(random.Random(i), 12, 34) for i in range(500)]
    assert not any("12 + 34 +" in t or "12 plus 34 plus" in t for t in texts)


def test_new_topics_are_never_asked_for():
    assert not set(g.NEW_TOPICS) & set(TOPICS)
    assert not set(g.NEW_TOPICS.values()) & set(TOPICS.values())
    assert all("{t}" in t for t in TOPIC_REQUESTS)   # topics only reach chat data through TOPICS


def test_reworded_code_requests_are_never_training_prompts():
    for kind, reword in [("fix_test", g.reword_repair), ("rename", g.reword_rename)]:
        rng = random.Random(0)
        trained = [code.make_task(rng, kind) for _ in range(150)]
        seen = {t.prompt for t in trained}
        for t in trained:
            new = reword(rng, t)
            assert new not in seen
            for name in re.findall(r"\b\w+\.py\b|\b[a-z_]+\b(?= to )", t.prompt):
                assert name in new                     # as detailed as the request it replaces
            assert replace(t, prompt=new).check is t.check

