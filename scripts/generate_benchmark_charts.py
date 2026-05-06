"""
Generate two benchmark charts for reports/findings.md.

Reads:  reports/eval_results_20260425_124153.json
Writes: reports/eval_charts/pareto_scatter.png
        reports/eval_charts/variance_decomposition.png

Run from project root:
    python scripts/generate_benchmark_charts.py
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # must precede any pyplot import
import matplotlib.lines as mlines  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from aetherart.visualization import (  # noqa: E402
    BLUE,
    GREEN,
    GREY,
    ORANGE,
    PURPLE,
    RED,
    ChartCanvas,
)

DATA = ROOT / "reports" / "eval_results_20260425_124153.json"
CHARTS = ROOT / "reports" / "eval_charts"
CHARTS.mkdir(exist_ok=True)

with open(DATA) as f:
    d = json.load(f)
results = d["results"]

SCHEDULERS = ["DDIM", "DPM", "EulerA", "LMS"]
STEPS = [20, 30, 50]
SCHED_COLOR = {"DDIM": BLUE, "DPM": GREEN, "EulerA": ORANGE, "LMS": PURPLE}
MARKER = {20: "o", 30: "s", 50: "^"}

# ── Per-(scheduler, steps) aggregates ────────────────────────────────────────
by_key: dict = defaultdict(list)
for r in results:
    by_key[(r["scheduler"], r["steps"])].append((r["clip_score"], r["latency_s"]))

points = []
for sched in SCHEDULERS:
    for steps in STEPS:
        vals = by_key[(sched, steps)]
        clips = [v[0] for v in vals]
        lats = [v[1] for v in vals]
        points.append(
            {
                "sched": sched,
                "steps": steps,
                "clip": statistics.mean(clips),
                "lat": statistics.mean(lats),
                "se": statistics.stdev(clips) / len(clips) ** 0.5,
            }
        )

# Pareto frontier: minimize latency AND maximize CLIP
sorted_pts = sorted(points, key=lambda p: p["lat"])
frontier: list = []
max_clip = -1.0
for p in sorted_pts:
    if p["clip"] > max_clip:
        frontier.append(p)
        max_clip = p["clip"]

# ── Chart 1: Speed–quality Pareto scatter ────────────────────────────────────
canvas = ChartCanvas(
    figsize=(9, 5.5),
    title="Speed/quality tradeoff: 30 PartiPrompts × RTX 3070",
    ylabel="Mean CLIP score",
    xlabel="Mean latency (s)",
    top_margin_pct=0.22,
)
canvas.set_ylim(0.302, 0.334)

for p in points:
    on_frontier = p in frontier
    canvas.ax.scatter(
        p["lat"],
        p["clip"],
        color=SCHED_COLOR[p["sched"]],
        marker=MARKER[p["steps"]],
        s=150 if on_frontier else 65,
        zorder=3 if on_frontier else 2,
        edgecolors="black" if on_frontier else "none",
        linewidths=1.5 if on_frontier else 0,
    )
    canvas.ax.errorbar(
        p["lat"],
        p["clip"],
        yerr=p["se"],
        fmt="none",
        color=GREY,
        capsize=3,
        linewidth=0.8,
        alpha=0.5,
    )

# Dashed frontier line
fpts = sorted(frontier, key=lambda p: p["lat"])
canvas.ax.plot(
    [p["lat"] for p in fpts],
    [p["clip"] for p in fpts],
    "--",
    color=GREY,
    linewidth=1.2,
    alpha=0.55,
    zorder=1,
)

# RED callout on DPM@20 (the sweet spot)
dpm20 = next(p for p in points if p["sched"] == "DPM" and p["steps"] == 20)
canvas.add_callout(
    target_x=dpm20["lat"],
    target_y=dpm20["clip"],
    text="Sweet spot: DPM-Solver++ at 20 steps\nmatches DDIM at 50 steps within noise",
    placement="top",
    color=RED,
    fontsize=8.5,
    x_offset=2.0,
)

# Legend: color = scheduler, shape = step count, black edge = frontier
sched_handles = [mpatches.Patch(color=SCHED_COLOR[s], label=s) for s in SCHEDULERS]
step_handles = [
    mlines.Line2D(
        [],
        [],
        color=GREY,
        marker=MARKER[st],
        linestyle="None",
        markersize=7,
        label=f"{st} steps",
    )
    for st in STEPS
]
frontier_handle = mlines.Line2D(
    [],
    [],
    color="black",
    marker="o",
    linestyle="None",
    markersize=9,
    markeredgewidth=1.5,
    markerfacecolor="none",
    label="Pareto frontier",
)
canvas.ax.legend(
    handles=sched_handles + step_handles + [frontier_handle],
    fontsize=8,
    loc="lower right",
    framealpha=0.9,
    ncol=2,
)
canvas.ax.set_xlim(7.0, 18.0)
canvas.ax.grid(True, alpha=0.18, linestyle="--")

canvas.save(str(CHARTS / "pareto_scatter.png"))


# ── Chart 2: Variance decomposition ──────────────────────────────────────────
by_prompt: dict = defaultdict(list)
by_sched: dict = defaultdict(list)
for r in results:
    by_prompt[r["prompt_id"]].append(r["clip_score"])
    by_sched[r["scheduler"]].append(r["clip_score"])

prompt_means = sorted(statistics.mean(v) for v in by_prompt.values())

# Sort schedulers by mean CLIP ascending so the spread is most visible
sched_sorted = sorted(
    [(s, statistics.mean(by_sched[s])) for s in SCHEDULERS],
    key=lambda x: x[1],
)
sched_names = [x[0] for x in sched_sorted]
sched_vals = np.array([x[1] for x in sched_sorted])

# Same y-axis scale on both panels — the flat right panel is the point
YMIN, YMAX = 0.238, 0.395

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 5.5))
fig.patch.set_facecolor("white")

# Left: per-prompt strip chart
left = ChartCanvas.from_axes(
    ax_left, fig, title="Prompt-to-prompt variation (n=30)", ylabel="Mean CLIP score"
)
left.set_ylim(YMIN, YMAX)

ax_left.scatter([0] * len(prompt_means), prompt_means, color=BLUE, s=40, alpha=0.75, zorder=3)
ax_left.vlines(0, min(prompt_means), max(prompt_means), color=BLUE, linewidth=2.5, alpha=0.28)
ax_left.set_xlim(-0.5, 1.5)
ax_left.set_xticks([])

p_range = max(prompt_means) - min(prompt_means)
p_mid = (max(prompt_means) + min(prompt_means)) / 2
ax_left.annotate(
    f"Range = {p_range:.3f}",
    xy=(0, p_mid),
    xytext=(0.62, p_mid),
    fontsize=9.5,
    va="center",
    color=BLUE,
    fontweight="bold",
    arrowprops=dict(arrowstyle="->", lw=1.0, color=BLUE),
)

# Right: per-scheduler bars
right = ChartCanvas.from_axes(
    ax_right, fig, title="Scheduler variation (n=4)", ylabel="Mean CLIP score"
)
right.set_ylim(YMIN, YMAX)

bar_x = np.arange(len(sched_names), dtype=float)
right.add_bars(
    bar_x,
    sched_vals,
    colors=[SCHED_COLOR[s] for s in sched_names],
    width=0.55,
    value_fmt="{:.4f}",
    value_pad=0.003,
    value_size=9,
)
right.set_xticks(bar_x, sched_names, fontsize=10)
ax_right.set_xlim(-0.6, len(sched_names) - 0.4 + 1.4)  # room for annotation

s_range = float(max(sched_vals) - min(sched_vals))
s_mid = float((max(sched_vals) + min(sched_vals)) / 2)
ax_right.annotate(
    f"Range = {s_range:.3f}\n(18× smaller than\nprompt range)",
    xy=(bar_x[-1], float(max(sched_vals))),
    xytext=(bar_x[-1] + 0.65, float(max(sched_vals)) + 0.012),
    fontsize=8.5,
    va="center",
    color=GREY,
    arrowprops=dict(arrowstyle="->", lw=0.9, color=GREY),
)

fig.suptitle(
    "Variance decomposition: prompt choice dominates scheduler choice (18×)\n"
    "Both panels share the same y-axis scale — the nearly-flat right panel is the finding",
    fontsize=10.5,
    fontweight="bold",
    y=1.01,
)

ChartCanvas.save_fig(fig, str(CHARTS / "variance_decomposition.png"))
