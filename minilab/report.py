"""Training report: one self-contained HTML page with the curves and evals of a run.

    uv run python -m minilab.report runs/small            # -> runs/small/report.html
    uv run python -m minilab.report runs/exp7 runs/exp8   # compare runs -> runs/compare.html
    uv run python -m minilab.report runs/small --open     # and open it in the browser

It reads what the training stages already write (runs/<run>/<stage>/log.jsonl,
config.json and eval.json). No dependencies and no network: the data is embedded
as JSON in the page and drawn as SVG by a small script (see report.html).
"""

from __future__ import annotations

import argparse
import html
import json
import webbrowser
from pathlib import Path

STAGES = ["pretrain", "midtrain", "sft", "rl", "rl_math", "distill"]  # as minilab.train.trainer.STAGES
META_KEYS = ["steps", "tokens", "tokens_total", "wall_clock_s", "params", "device", "hardware", "final_loss", "val_loss"]

# Rows of the "stage progression" table: (path in eval.json, label, kind).
# kind: "pct" (0..1, colored), "num" (plain number), "inv" (0..1 where lower is better, shown as 1 - x)
EVAL_ROWS = [
    ("val_ppl", "Story perplexity (lower is better)", "num"),
    ("arithmetic.1", "Addition, 1 digit", "pct"),
    ("arithmetic.2", "Addition, 2 digits", "pct"),
    ("arithmetic.3", "Addition, 3 digits", "pct"),
    ("arithmetic.4", "Addition, 4 digits", "pct"),
    ("arithmetic.5", "Addition, 5 digits", "pct"),
    ("arithmetic.6", "Addition, 6 digits (never trained on)", "pct"),
    ("arithmetic_sampled", "Addition, sampled at T=1", "pct"),
    ("direct", "Addition, answer only (no scratchpad)", "pct"),
    ("tool_call", "Calls the calculator correctly", "pct"),
    ("tool_answer", "Correct answer with the calculator", "pct"),
    ("story_topic", "Story on the requested topic", "pct"),
    ("instr", "Instruction following (all)", "pct"),
    ("instructions.number_only", "· system prompt: number only", "pct"),
    ("instructions.no_calculator", "· system prompt: no calculator", "pct"),
    ("instructions.one_sentence", "· system prompt: one sentence", "pct"),
    ("instructions.sure", "· system prompt: start with “Sure”", "pct"),
    ("instructions.followup", "· multi-turn follow-up", "pct"),
    ("instructions.long_followup", "· follow-up on a 4-5 digit total", "pct"),
    ("instructions.new_question", "· new question after an answer", "pct"),
    ("instructions.refusal", "· refuses out-of-scope questions", "pct"),
    ("instructions.identity", "· knows who it is", "pct"),
    ("instructions.over_refusal", "· answers in-scope requests (no over-refusal)", "inv"),
    ("format", "Ends its turn properly", "pct"),
]


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _get(d: dict, path: str):
    for key in path.split("."):
        if not isinstance(d, dict) or key not in d:
            return None
        d = d[key]
    return d if isinstance(d, (int, float)) and not isinstance(d, bool) else None


def load_run(run_dir: Path) -> dict:
    run = {"name": run_dir.name, "path": str(run_dir), "stages": {}}
    for stage in STAGES:
        d = run_dir / stage
        if not d.exists():
            continue
        rows = _read_jsonl(d / "log.jsonl")
        meta = json.loads((d / "config.json").read_text()).get("meta", {}) if (d / "config.json").exists() else {}
        ev = json.loads((d / "eval.json").read_text()) if (d / "eval.json").exists() else {}
        run["stages"][stage] = {
            "train": [r for r in rows if "val_loss" not in r],
            "val": [r for r in rows if "val_loss" in r],
            "meta": {k: meta[k] for k in META_KEYS if k in meta},
            "eval": {k: v for k, v in ev.items() if k != "samples"},
            "samples": ev.get("samples", [])[:8],
        }
    return run


# ---- server-side HTML (tables are static; charts are drawn by report.html's script) ----

def _heat_class(v: float) -> str:
    return f"h{min(8, max(0, int(v * 8 + 0.5)))}"


def progression_table(run: dict) -> str:
    stages = [s for s in STAGES if s in run["stages"]]
    evals = {s: run["stages"][s]["eval"] for s in stages}
    rows = []
    for path, label, kind in EVAL_ROWS:
        values = {s: _get(evals[s], path) for s in stages}
        if all(v is None for v in values.values()):
            continue
        cells = []
        for s in stages:
            v = values[s]
            if v is None:
                cells.append('<td class="na">–</td>')
            elif kind == "num":
                cells.append(f'<td class="num">{v:.2f}</td>')
            else:
                v = 1 - v if kind == "inv" else v
                cells.append(f'<td class="heat {_heat_class(v)}">{v:.0%}</td>')
        rows.append(f"<tr><th scope=row>{html.escape(label)}</th>{''.join(cells)}</tr>")
    head = "".join(f"<th scope=col>{s}</th>" for s in stages)
    return (f'<div class="table-wrap"><table class="progression"><thead><tr><th scope=col>Metric</th>{head}</tr></thead>'
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def _fmt_int(n) -> str:
    n = float(n)
    for unit, div in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if n >= div:
            return f"{n / div:.1f}{unit}"
    return f"{n:.0f}"


def _fmt_time(s) -> str:
    s = float(s)
    return f"{s / 60:.1f} min" if s >= 60 else f"{s:.0f} s"


def summary_tiles(run: dict) -> str:
    stages = run["stages"]
    metas = [st["meta"] for st in stages.values()]
    params = next((m["params"] for m in metas if "params" in m), None)
    tokens = max((m.get("tokens_total", 0) for m in metas), default=0)
    wall = sum(m.get("wall_clock_s", 0) for m in metas)
    last_eval = stages[[s for s in STAGES if s in stages][-1]]["eval"] if stages else {}
    hardware = next((m["hardware"] for m in metas if "hardware" in m), "")
    devices = sorted({m["device"] for m in metas if "device" in m})
    tiles = [
        ("Parameters", _fmt_int(params) if params else "–"),
        ("Tokens trained on", _fmt_int(tokens) if tokens else "–"),
        ("Training time", _fmt_time(wall) if wall else "–"),
        ("Final accuracy (in distribution)", f"{last_eval['in_distribution']:.0%}" if "in_distribution" in last_eval else "–"),
    ]
    out = "".join(f'<div class="tile"><div class="tile-label">{html.escape(k)}</div><div class="tile-value">{v}</div></div>'
                  for k, v in tiles)
    note = " · ".join(x for x in [hardware, "device: " + ", ".join(devices) if devices else ""] if x)
    return f'<div class="tiles">{out}</div><p class="muted small">{html.escape(note)}</p>'


def samples_html(run: dict) -> str:
    parts = []
    pre = run["stages"].get("pretrain")
    if pre:
        with_sample = [r for r in pre["val"] if r.get("sample")]
        picks = [with_sample[i] for i in sorted({0, len(with_sample) // 2, len(with_sample) - 1})] if with_sample else []
        if picks:
            items = "".join(f'<li><div class="muted small">pretrain step {r["step"]}</div><p class="sample">{html.escape(r["sample"])}</p></li>'
                            for r in picks)
            parts.append(f"<h3>The base model learning to write</h3><ul class=samples>{items}</ul>")
    last = next((run["stages"][s] for s in reversed(STAGES) if s in run["stages"] and run["stages"][s]["samples"]), None)
    if last:
        items = []
        for smp in last["samples"]:
            tools = " (calculator on)" if smp.get("tools") else ""
            calls = "".join(f'<div class="muted small">calls {html.escape(c.get("name", ""))}({html.escape(str(c.get("arguments", "")))})</div>'
                            for c in smp.get("tool_calls") or [])
            items.append(f'<li><div class="prompt">{html.escape(smp.get("prompt", ""))}<span class="muted">{tools}</span></div>'
                         f'{calls}<p class="sample">{html.escape(smp.get("response") or "")}</p></li>')
        parts.append(f"<h3>Final model, sample answers</h3><ul class=samples>{''.join(items)}</ul>")
    return "".join(parts)


def render(runs: list[dict]) -> str:
    template = (Path(__file__).parent / "report.html").read_text()
    title = runs[0]["name"] if len(runs) == 1 else " vs ".join(r["name"] for r in runs)
    sections = []
    for run in runs:
        heading = f"<h2>{html.escape(run['name'])}</h2>" if len(runs) > 1 else ""
        sections.append(f'<section class="run">{heading}{summary_tiles(run)}'
                        f"<h3>Stage progression</h3>"
                        f'<p class="muted small">Evaluated after each stage. Darker = better.</p>{progression_table(run)}</section>')
    samples = samples_html(runs[-1])
    data = json.dumps({"runs": runs, "stages": STAGES}).replace("</", "<\\/")
    return (template.replace("{{TITLE}}", html.escape(title))
            .replace("{{SUMMARY}}", "".join(sections))
            .replace("{{SAMPLES}}", samples)
            .replace("{{DATA}}", data))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an HTML training report for one or more runs.")
    parser.add_argument("runs", nargs="+", type=Path, help="run directories, e.g. runs/small")
    parser.add_argument("-o", "--out", type=Path, help="output file (default: <run>/report.html or runs/compare.html)")
    parser.add_argument("--open", action="store_true", help="open the report in the browser")
    args = parser.parse_args()
    runs = [load_run(r) for r in args.runs]
    if not any(r["stages"] for r in runs):
        parser.error("no stage found in these run directories")
    out = args.out or (args.runs[0] / "report.html" if len(runs) == 1 else args.runs[0].parent / "compare.html")
    out.write_text(render(runs))
    print(f"report: {out}")
    if args.open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
