"""Generate two charts for reports/findings.md."""

import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
CHARTS = ROOT / "reports" / "eval_charts"

with open(ROOT / "reports" / "eval_results_20260425_124153.json") as f:
    d = json.load(f)
results = d["results"]

schedulers = ["DDIM", "DPM", "EulerA", "LMS"]
steps_list = [20, 30, 50]
palette = {"DDIM": "#2196F3", "DPM": "#4CAF50", "EulerA": "#FF9800", "LMS": "#9C27B0"}

# ── Per-(scheduler, steps) aggregates ────────────────────────────────────────
by_key = defaultdict(list)
for r in results:
    by_key[(r["scheduler"], r["steps"])].append((r["clip_score"], r["latency_s"]))

points = []
for sched in schedulers:
    for steps in steps_list:
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

# Pareto frontier (minimize latency, maximize CLIP)
sorted_pts = sorted(points, key=lambda p: p["lat"])
frontier, max_clip = [], -1.0
for p in sorted_pts:
    if p["clip"] > max_clip:
        frontier.append(p)
        max_clip = p["clip"]

# ── Chart 1: Speed–quality Pareto scatter ────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor("#F8F9FA")
ax.set_facecolor("#F8F9FA")

marker_shape = {20: "o", 30: "s", 50: "^"}
for p in points:
    on_frontier = p in frontier
    ax.scatter(
        p["lat"],
        p["clip"],
        color=palette[p["sched"]],
        marker=marker_shape[p["steps"]],
        s=130 if on_frontier else 65,
        zorder=3 if on_frontier else 2,
        edgecolors="black" if on_frontier else "none",
        linewidths=1.4 if on_frontier else 0,
    )
    # Label offsets — tweak to avoid overlaps
    dx, dy = 0.15, 0.0004
    if p["sched"] == "DPM" and p["steps"] == 20:
        dy = 0.0010
    elif p["sched"] == "DDIM" and p["steps"] == 20:
        dy = -0.0012
    elif p["sched"] == "LMS" and p["steps"] == 30:
        dy = -0.0012
    elif p["sched"] == "EulerA" and p["steps"] == 30:
        dy = 0.0008
    elif p["sched"] == "DPM" and p["steps"] == 30:
        dy = 0.0008
    ax.annotate(
        f"{p['sched']}@{p['steps']}",
        (p["lat"] + dx, p["clip"] + dy),
        fontsize=7.5,
        color=palette[p["sched"]],
    )

# Pareto frontier line
frontier_x = [p["lat"] for p in sorted(frontier, key=lambda p: p["lat"])]
frontier_y = [p["clip"] for p in sorted(frontier, key=lambda p: p["lat"])]
ax.plot(frontier_x, frontier_y, "--", color="#555555", linewidth=1.1, alpha=0.55, zorder=1)

# Error bars on frontier points
for p in frontier:
    ax.errorbar(
        p["lat"],
        p["clip"],
        yerr=p["se"],
        fmt="none",
        color="#444444",
        capsize=3,
        linewidth=0.9,
        alpha=0.5,
    )

# Annotation callout for DPM@20
dpm20 = next(p for p in points if p["sched"] == "DPM" and p["steps"] == 20)
ax.annotate(
    "Best value: DPM@20\n8.2 s, CLIP 0.317",
    xy=(dpm20["lat"], dpm20["clip"]),
    xytext=(10.2, 0.3125),
    fontsize=8,
    color="#1B5E20",
    arrowprops=dict(arrowstyle="->", color="#1B5E20", lw=1.0),
)

ax.set_xlabel("Mean latency (s)", fontsize=11)
ax.set_ylabel("Mean CLIP score", fontsize=11)
ax.set_title(
    "Scheduler × step count: speed–quality tradeoff\n"
    "Pareto-optimal configs outlined in black; error bars = ±1 SE (n=30)",
    fontsize=10.5,
)
sched_patches = [mpatches.Patch(color=palette[s], label=s) for s in schedulers]
step_handles = [
    mlines.Line2D(
        [], [], color="gray", marker=m, linestyle="None", markersize=7, label=f"{st} steps"
    )
    for st, m in marker_shape.items()
]
ax.legend(
    handles=sched_patches + step_handles,
    fontsize=8,
    loc="lower right",
    framealpha=0.9,
    ncol=2,
)
ax.grid(True, alpha=0.22, linestyle="--")
ax.set_xlim(7.5, 17.2)
ax.set_ylim(0.303, 0.327)
plt.tight_layout()
out1 = CHARTS / "pareto_scatter.png"
plt.savefig(out1, dpi=140, bbox_inches="tight")
plt.close()
print(f"Saved {out1}")

# ── Chart 2: Prompt spread vs scheduler spread ───────────────────────────────
by_prompt = defaultdict(list)
by_sched = defaultdict(list)
for r in results:
    by_prompt[r["prompt_id"]].append(r["clip_score"])
    by_sched[r["scheduler"]].append(r["clip_score"])

prompt_means = sorted(statistics.mean(v) for v in by_prompt.values())
sched_means = [statistics.mean(by_sched[s]) for s in schedulers]

fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharey=False)
fig.patch.set_facecolor("#F8F9FA")

# Left: per-prompt means (strip + range)
ax0 = axes[0]
ax0.set_facecolor("#F8F9FA")
x_jitter = [0] * len(prompt_means)
ax0.scatter(x_jitter, prompt_means, color="#5C6BC0", s=40, alpha=0.7, zorder=3)
ax0.vlines(0, min(prompt_means), max(prompt_means), color="#5C6BC0", linewidth=2.0, alpha=0.4)
ax0.annotate(
    f"Range = {max(prompt_means) - min(prompt_means):.4f}\n(30 prompts)",
    xy=(0, (max(prompt_means) + min(prompt_means)) / 2),
    xytext=(0.25, (max(prompt_means) + min(prompt_means)) / 2),
    fontsize=9,
    va="center",
    arrowprops=dict(arrowstyle="->", lw=0.8),
)
ax0.set_xlim(-0.5, 1.2)
ax0.set_xticks([])
ax0.set_ylabel("Mean CLIP score", fontsize=10)
ax0.set_title("Prompt-to-prompt variation", fontsize=10)
ax0.grid(True, axis="y", alpha=0.2, linestyle="--")

# Right: per-scheduler means
ax1 = axes[1]
ax1.set_facecolor("#F8F9FA")
bar_x = range(len(schedulers))
bars = ax1.bar(bar_x, sched_means, color=[palette[s] for s in schedulers], width=0.5, alpha=0.85)
ax1.set_xticks(list(bar_x))
ax1.set_xticklabels(schedulers, fontsize=10)
ax1.set_ylim(0.300, 0.325)
ax1.set_ylabel("Mean CLIP score", fontsize=10)
ax1.set_title("Scheduler variation", fontsize=10)
ax1.grid(True, axis="y", alpha=0.2, linestyle="--")
sched_spread = max(sched_means) - min(sched_means)
ax1.annotate(
    f"Range = {sched_spread:.4f}\n(4 schedulers)",
    xy=(1.5, min(sched_means) + sched_spread / 2),
    xytext=(2.5, 0.309),
    fontsize=9,
    va="center",
    arrowprops=dict(arrowstyle="->", lw=0.8),
)

fig.suptitle(
    "What drives CLIP score? Prompt choice (left) vs scheduler choice (right)\n"
    "Prompt spread is 18× larger than scheduler spread",
    fontsize=10.5,
    y=1.01,
)
plt.tight_layout()
out2 = CHARTS / "variance_decomposition.png"
plt.savefig(out2, dpi=140, bbox_inches="tight")
plt.close()
print(f"Saved {out2}")
