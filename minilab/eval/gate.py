"""The release gate: a model only ships if it does at least as well as the release before it.

Both models are evaluated with the same eval code and config: the baseline's own eval.json
may predate a check, or measure it on other prompts. Every metric is compared, and a drop
counts as a regression beyond the noise of its sample: two more failures than before, and at
least 2 points (a check of 30 prompts may lose 6.7 points, one of 100 prompts 2); perplexity
may rise by 2% (measured in bits per character: every run trains its own tokenizer). A
regression blocks the release unless it is waived with a reason, which the release records.

    uv run python -m minilab.eval.gate runs/prelude/distill --baseline models/prelude-1.1 --config configs/prelude.toml
"""

from __future__ import annotations

import argparse
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

from minilab.checkpoint import load_checkpoint
from minilab.eval import tasks
from minilab.eval.run import evaluate
from minilab.registry import list_models
from minilab.train.trainer import DEVICES, resolve_device, setup

PPL_TOLERANCE = 0.02  # relative


@dataclass
class Change:
    metric: str
    old: float
    new: float
    allowed: float        # the largest drop that is still noise
    higher_is_better: bool = True

    @property
    def regressed(self) -> bool:
        return (self.old - self.new if self.higher_is_better else self.new - self.old) > self.allowed + 1e-9


def metrics(result: dict, cfg: dict) -> dict[str, tuple[float, int, bool]]:
    """The eval's numbers as {name: (value, prompts, higher is better)}; prompts 0 = bits per character."""
    ec, digits = cfg["eval"], cfg["data"]["digits"]
    n_instr = ec.get("n_instr", 30)
    out = {"bpc": (result["val_bpc"], 0, False)}
    out |= {f"{n}d": (result["arithmetic"][str(n)], ec["n_per_digit"], True) for n in digits}
    candidates = {
        f"{max(digits)}d@T=1": ("arithmetic_sampled", ec["n_sampled"]),
        "tool call": ("tool_call", ec["n_tool"] * len(digits)),
        "tool ans": ("tool_answer", ec["n_tool"] * len(digits)),
        "story": ("story_topic", len(tasks.TOPICS) * len(tasks.STORY_EVAL_REQUESTS)),
        "chat": ("chat", ec.get("n_chat", 200)),
        "format": ("format", 100),  # hundreds of turns: the 2-point floor applies
    }
    out |= {name: (result[key], n, True) for name, (key, n) in candidates.items() if result.get(key) is not None}
    for kind, value in result.get("instructions", {}).items():
        if value is not None:  # over-refusal is measured on every in-scope prompt: the 2-point floor
            out[kind] = (value, 100, False) if kind == "over_refusal" else (value, n_instr, True)
    if "code" in result:  # the coding agent: "code bpc" and "code chat" apart from the stories' and chats'
        out |= {(f"code {k}" if k in ("bpc", "chat") else k): v
                for k, v in code_metrics(result["code"], cfg["code_eval"]).items()}
    return out


def code_metrics(result: dict, ec: dict) -> dict[str, tuple[float, int, bool]]:
    """The coding eval: every task kind, small talk and titles (see eval/code.py); ec: [code_eval]."""
    out = {"bpc": (result["val_bpc"], 0, False)}
    out |= {kind: (v, ec["n_per_kind"], True) for kind, v in result.get("tasks", {}).items() if v is not None}
    out |= {k: (result[k], ec.get("n_chat", 30), True) for k in ("chat", "title") if result.get(k) is not None}
    if result.get("valid_calls") is not None:
        out["valid calls"] = (result["valid_calls"], 100, True)  # hundreds of calls: the 2-point floor applies
    return out


def compare(new: dict, old: dict, cfg: dict) -> list[Change]:
    """Every metric both evals have, with the drop that is still noise."""
    new_m, old_m = metrics(new, cfg), metrics(old, cfg)
    changes = []
    for name, (value, n, higher) in new_m.items():
        if name not in old_m:
            continue
        old_value = old_m[name][0]
        allowed = old_value * PPL_TOLERANCE if n == 0 else max(0.02, 2 / n)
        changes.append(Change(name, old_value, value, allowed, higher))
    return changes


def newest_release(models_dir: Path, exclude: str) -> Path | None:
    """The most recent release other than `exclude` (the one being released)."""
    return next((m.path for m in list_models(models_dir) if m.id != exclude), None)


def check(new_result: dict, baseline: Path, cfg: dict, device: str, waivers: dict[str, str],
          stage: str = "distill") -> dict:
    """Evaluate the baseline, compare, and say whether the release may go ahead."""
    setup(cfg.get("seed", 0), device)
    model, tok, _ = load_checkpoint(baseline, device=device)
    old = evaluate(model, tok, cfg, stage)
    changes = compare(new_result, old, cfg)
    regressions = [c for c in changes if c.regressed]
    blocking = [c for c in regressions if c.metric not in waivers]
    return {"baseline": baseline.name, "passed": not blocking,
            "regressions": [asdict(c) for c in regressions],
            "waived": {c.metric: waivers[c.metric] for c in regressions if c.metric in waivers},
            "changes": [asdict(c) for c in changes], "baseline_eval": old}


def _fmt(metric: str, value: float) -> str:
    return f"{value:.3f}" if metric.removeprefix("code ") == "bpc" else f"{100 * value:.0f}%"


def report(result: dict) -> str:
    """A few lines for the terminal: the verdict, then every regression."""
    lines = [f"release gate vs {result['baseline']}: " + ("passed" if result["passed"] else "BLOCKED")]
    for c in result["regressions"]:
        waived = result["waived"].get(c["metric"])
        lines.append(f"  {c['metric']}: {_fmt(c['metric'], c['old'])} -> {_fmt(c['metric'], c['new'])}"
                     + (f"  (waived: {waived})" if waived else "  REGRESSION"))
    if not result["regressions"]:
        lines.append("  no regression")
    return "\n".join(lines)


def parse_waivers(items: list[str]) -> dict[str, str]:
    """["sure=one greedy prompt", ...] -> {"sure": "one greedy prompt"}"""
    waivers = {}
    for item in items:
        metric, _, reason = item.partition("=")
        if not reason.strip():
            raise SystemExit(f"--allow {item!r}: give a reason, as METRIC=REASON")
        waivers[metric.strip()] = reason.strip()
    return waivers


def main() -> None:
    p = argparse.ArgumentParser(description="Compare a model with a baseline release, both evaluated now.")
    p.add_argument("model", help="the candidate: a release or checkpoint directory")
    p.add_argument("--baseline", required=True, help="the release to beat, e.g. models/prelude-1.1")
    p.add_argument("--config", default="configs/prelude.toml", help="the eval settings ([data], [eval], [code_eval])")
    p.add_argument("--device", default="auto", choices=DEVICES)
    p.add_argument("--allow", action="append", default=[], metavar="METRIC=REASON")
    args = p.parse_args()
    cfg = tomllib.loads(Path(args.config).read_text())
    device = resolve_device(args.device, generation=True)
    setup(cfg.get("seed", 0), device)
    model, tok, _ = load_checkpoint(args.model, device=device)
    new = evaluate(model, tok, cfg, "distill")
    print(report(check(new, Path(args.baseline), cfg, device, parse_waivers(args.allow))))


if __name__ == "__main__":
    main()
