import importlib.util
import json
import random
from pathlib import Path

from minilab.data import story_math as sm
from minilab.eval import tasks

spec = importlib.util.spec_from_file_location("probe_direct", Path(__file__).parent.parent / "scripts" / "probe_direct.py")
pd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pd)

TAU = {"low": 0.80, "high": 0.99}
LEVELS = ["low", "high"]


def measured(monkeypatch, tmp_path, correct):
    """The policy file measured with a fake model that is right on a share `correct` of the problems."""
    def fake(model, tok, problems, max_new, level, think=None):
        assert level == "high" and think is False
        return [{"bucket": p["bucket"], "correct": i < correct * len(problems)} for i, p in enumerate(problems)]
    monkeypatch.setattr(tasks, "eval_story_math", fake)
    pol = pd.policy(pd.measure(None, None, 1002, [1, 2, 3], 200), TAU, "runs/x/midtrain", 1002)
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(pol))
    return sm.load_policy(path)


def test_perfect_probe_answers_at_once(monkeypatch, tmp_path):
    pol = measured(monkeypatch, tmp_path, 1.0)
    assert pol["p"] and set(pol["p"].values()) == {1.0} and pol["source"] == "runs/x/midtrain"
    for k in (1, 2, 3):
        problem = sm.make(random.Random(k), k)
        assert [sm.think(level, problem, pol) for level in LEVELS] == [False] * 2


def test_half_right_probe_thinks(monkeypatch, tmp_path):
    pol = measured(monkeypatch, tmp_path, 0.5)
    for k in (1, 2, 3):
        problem = sm.make(random.Random(k), k)
        assert all(sm.think(level, problem, pol) for level in LEVELS)


def test_small_bucket_is_left_out():
    out = pd.policy({"9,0": [True] * 10, "1,0": [True] * 40}, TAU, "s", 1)
    assert out["n"] == {"1,0": 40, "9,0": 10} and list(out["p"]) == ["1,0"]
    assert "n too small" in pd.table(out)
