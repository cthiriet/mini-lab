"""The training report builds from whatever stages a run directory contains."""

import json

from minilab.report import load_run, render


def write_stage(run, stage, rows, meta=None, ev=None):
    d = run / stage
    d.mkdir(parents=True)
    (d / "log.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (d / "config.json").write_text(json.dumps({"model": {}, "meta": meta or {}}))
    if ev is not None:
        (d / "eval.json").write_text(json.dumps(ev))


def test_report_renders_curves_and_progression(tmp_path):
    run = tmp_path / "demo"
    write_stage(run, "pretrain",
                [{"step": 25, "loss": 6.0, "lr": 1e-3, "tok_per_s": 1000},
                 {"step": 50, "val_loss": 4.0, "sample": "Once upon a time <script>alert(1)</script>"}],
                {"steps": 50, "tokens": 12800, "params": 5_770_496, "wall_clock_s": 12.0, "device": "cpu"},
                {"val_ppl": 6.1, "arithmetic": {"1": 0.35}})
    write_stage(run, "rl_math", [{"step": 5, "reward": 0.6, "completion_len": 60}], {"steps": 5},
                {"arithmetic": {"1": 1.0}, "instructions": {"over_refusal": 0.1}, "in_distribution": 1.0,
                 "samples": [{"prompt": "Hi!", "response": "Hello!", "tool_calls": []}]})
    data = load_run(run)
    assert list(data["stages"]) == ["pretrain", "rl_math"]
    assert data["stages"]["pretrain"]["val"][0]["val_loss"] == 4.0

    page = render([data])
    assert "5.8M" in page and "100%" in page and "90%" in page  # over-refusal shown as 1 - x
    assert "<script>alert(1)</script>" not in page  # samples are escaped
    assert "{{" not in page
