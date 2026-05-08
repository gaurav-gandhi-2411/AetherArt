#!/usr/bin/env python3
"""Generate CLIP-blindness summary chart for reports/clip_blindness.md.

Two-panel figure:
  Left:  CLIP delta (standard-error units) per experiment — how much the semantic
         metric moved in response to each parameter change.
  Right: Max LPIPS per experiment — how much the images actually changed.

The contrast is the point: most experiments show large LPIPS while CLIP stays flat
(< 1 SE). Experiment 8 (LoRA alpha) is the partial exception where CLIP registers
the style switch but still cannot resolve within the active range.

Run from project root:
    python scripts/generate_clip_blindness_chart.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from aetherart.visualization.charts import BLUE, GREY, ORANGE, ChartCanvas

OUT_PATH = Path("reports/clip_blindness_chart.png")

# --- experiment labels (two lines each for readability at 9-bar density) ---
EXPERIMENTS = [
    "Exp 1\nQuant",
    "Exp 2\nNeg prompt",
    "Exp 3\nCFG",
    "Exp 4\nScheduler",
    "Exp 5\nControlNet",
    "Exp 6\nLoRA rank",
    "Exp 7\nLoRA data",
    "Exp 8\nLoRA alpha",
    "Exp 9\nTrigger",
]

# CLIP delta in standard-error units.
# "Maximum meaningful delta" = the largest delta / pooled SE for that experiment.
# Exp 8 is the partial-sensitivity case: CLIP rises 4 SE from no-LoRA → active-LoRA,
# then plateaus — it detects the style switch but cannot resolve within the active range.
CLIP_SE = np.array([0.94, 0.83, 1.10, 1.80, 2.20, 1.00, 0.80, 4.00, 0.12])

# Maximum LPIPS observed within the practical parameter range for each experiment.
# (Exp 3: CFG=15 vs CFG=7 = 0.47; CFG=1 no-guidance baseline excluded as non-practical.)
MAX_LPIPS = np.array([0.40, 0.46, 0.47, 0.73, 0.72, 0.50, 0.66, 0.67, 0.41])

x = np.arange(len(EXPERIMENTS))

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(17, 6))
fig.patch.set_facecolor("white")

# ── Left panel: CLIP sensitivity ──────────────────────────────────────────────
left = ChartCanvas.from_axes(
    ax_left,
    fig,
    title="CLIP sensitivity: delta (standard-error units)",
    ylabel="CLIP Δ (SE units)",
    top_margin_pct=0.22,
)
left.add_bars(
    x,
    CLIP_SE,
    colors=BLUE,
    width=0.55,
    value_fmt="{:.2f}",
    value_size=8.0,
    value_pad=0.06,
)
left.set_ylim(0, 5.8)
left.set_xticks(x, EXPERIMENTS, fontsize=8.5)

# 1 SE detection threshold
ax_left.axhline(1.0, color=GREY, lw=1.5, ls="--", alpha=0.85, zorder=2)
ax_left.text(
    0.015,
    1.07,
    "1 SE — detectable signal",
    color=GREY,
    fontsize=8,
    fontstyle="italic",
    transform=ax_left.get_yaxis_transform(),
)

# ── Right panel: LPIPS perceptual change ──────────────────────────────────────
right = ChartCanvas.from_axes(
    ax_right,
    fig,
    title="Perceptual change: max LPIPS vs reference condition",
    ylabel="LPIPS (higher = more perceptually different)",
    top_margin_pct=0.22,
)
right.add_bars(
    x,
    MAX_LPIPS,
    colors=ORANGE,
    width=0.55,
    value_fmt="{:.2f}",
    value_size=8.0,
    value_pad=0.012,
)
right.set_ylim(0, 1.0)
right.set_xticks(x, EXPERIMENTS, fontsize=8.5)

# Rough "just-perceptible" reference line (LPIPS ~ 0.20 is typically visible)
ax_right.axhline(0.20, color=GREY, lw=1.5, ls="--", alpha=0.85, zorder=2)
ax_right.text(
    0.015,
    0.215,
    "~0.20 — perceptible threshold",
    color=GREY,
    fontsize=8,
    fontstyle="italic",
    transform=ax_right.get_yaxis_transform(),
)

fig.suptitle(
    "CLIP-Blindness Series (9 experiments) — large perceptual change, small CLIP signal",
    fontsize=13,
    fontweight="bold",
    y=1.01,
)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
ChartCanvas.save_fig(fig, str(OUT_PATH), dpi=130, bottom_adjust=0.20)
print(f"Saved: {OUT_PATH}")
