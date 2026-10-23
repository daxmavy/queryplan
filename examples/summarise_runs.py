"""Summarise finished runs: a markdown table of times and loads, and a timeline of what each GPU held.

    python examples/summarise_runs.py runs/naive runs/plan --labels naive planned --png timeline.png --md table.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

COLOURS = {"qwen": "#4C72B0", "gemma": "#DD8452", "gemma-judge": "#55A868"}


def load(run: Path) -> dict:
    report = json.loads((run / "report.json").read_text())
    events = [json.loads(l) for l in (run / "events.jsonl").read_text().splitlines() if l.strip()]
    return {"report": report, "events": events}


def intervals(events: list[dict]) -> tuple[float, dict[str, list[tuple[str, float, float, float]]]]:
    """Per slot: (model, load start, serving start, end) in seconds from the run's start."""
    start = min(e["at"] for e in events)
    open_: dict[str, list] = {}
    out: dict[str, list] = {}
    end = max(e["at"] for e in events)
    for e in sorted(events, key=lambda e: e["at"]):
        slot = e.get("slot")
        if e["event"] == "load_start":
            open_[slot] = [e["model"], e["at"] - start, None]
        elif e["event"] == "load_done" and slot in open_:
            open_[slot][2] = e["at"] - start
        elif e["event"] == "unload" and slot in open_:
            model, a, b = open_.pop(slot)
            out.setdefault(slot, []).append((model, a, b if b is not None else e["at"] - start, e["at"] - start))
    for slot, (model, a, b) in open_.items():
        out.setdefault(slot, []).append((model, a, b if b is not None else end - start, end - start))
    return end - start, out


def table(runs: list[dict], labels: list[str]) -> str:
    exps = runs[0]["report"]["experiments"]
    lines = ["| run | wall time (min) | model loads | time loading (GPU-min) | first experiment done (min) |",
             "|---|---:|---:|---:|---:|"]
    for label, r in zip(labels, runs):
        rep = r["report"]
        first = rep.get("first") or exps[0]
        lines.append(f"| {label} | {rep['wall_seconds'] / 60:.1f} | {rep['loads']} | {rep['load_seconds'] / 60:.1f} | "
                     f"{rep['experiment_seconds'].get(first, float('nan')) / 60:.1f} ({first}) |")
    return "\n".join(lines)


def timeline(runs: list[dict], labels: list[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    spans = [intervals(r["events"]) for r in runs]
    longest = max(s[0] for s in spans) / 60
    fig, axes = plt.subplots(len(runs), 1, figsize=(9, 1.3 + 0.55 * sum(len(s[1]) for s in spans) / len(runs) * len(runs)),
                             sharex=True, squeeze=False)
    for ax, label, (total, slots) in zip(axes[:, 0], labels, spans):
        names = sorted(slots)
        for y, slot in enumerate(names):
            for model, a, b, c in slots[slot]:
                colour = COLOURS.get(model, "#8C8C8C")
                ax.barh(y, (b - a) / 60, left=a / 60, color=colour, alpha=0.35, height=0.7, hatch="///",
                        edgecolor=colour, linewidth=0)
                ax.barh(y, (c - b) / 60, left=b / 60, color=colour, height=0.7)
        ax.axvline(total / 60, color="#333333", linewidth=1)
        ax.set_yticks(range(len(names)), names, fontsize=8)
        ax.set_title(f"{label}: {total / 60:.1f} min", fontsize=9, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(0, longest * 1.02)
    axes[-1, 0].set_xlabel("minutes since start")
    handles = [Patch(color=c, label=m) for m, c in COLOURS.items()]
    handles.append(Patch(facecolor="white", edgecolor="#555555", hatch="///", label="loading"))
    axes[0, 0].legend(handles=handles, fontsize=8, ncol=len(handles), loc="lower right", bbox_to_anchor=(1, 1.05),
                      frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--png", type=Path)
    parser.add_argument("--md", type=Path)
    args = parser.parse_args()
    runs = [load(r) for r in args.runs]
    labels = args.labels or [r.name for r in args.runs]
    md = table(runs, labels)
    print(md)
    if args.md:
        args.md.write_text(md + "\n")
    if args.png:
        timeline(runs, labels, args.png)


if __name__ == "__main__":
    main()
