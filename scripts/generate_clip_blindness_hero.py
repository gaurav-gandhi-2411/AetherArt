#!/usr/bin/env python3
"""Generate hero CLIP-blindness chart for README top section.

Reads from raw experiment result files (reports/experiments/exp*_sdxl/results.json)
and produces a clean, publication-quality paired horizontal bar chart.

Visual point: across 7 SDXL experiments, CLIP Δ stays below 1 SE for rendering-
level parameters (BLIND) while LPIPS shows large perceptual changes — CLIP cannot
see what other metrics catch. Semantic parameters (CFG, LoRA alpha) do break through.

Output: reports/showcase/clip_blindness_hero.png

Run from project root:
    python scripts/generate_clip_blindness_hero.py
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402

OUT_PATH = ROOT / "reports" / "showcase" / "clip_blindness_hero.png"

# ── Palette (consistent with aetherart.visualization.charts) ──────────────────
TEAL = "#1A7F7A"  # BLIND experiments
ORANGE = "#E07B39"  # RESPONDS experiments
GREY_LINE = "#999999"
LPIPS_BAR = "#2C6FAC"  # LPIPS panel bars (blue)
BG = "#FFFFFF"
GRID = "#EEEEEE"

# ── Data extraction helpers ────────────────────────────────────────────────────


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals)


def _se(vals: list[float]) -> float:
    if len(vals) < 2:
        return float("nan")
    return statistics.stdev(vals) / math.sqrt(len(vals))


def _group(rows: list[dict], key: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        k = str(r[key])
        out.setdefault(k, []).append(r)
    return out


def _delta_and_se(
    rows: list[dict], cond_key: str, clip_col: str = "clip_score"
) -> tuple[float, float]:
    """Return (clip_delta_SE, pooled_SE) using the same formula as generate_clip_blindness_sdxl."""
    groups = _group(rows, cond_key)
    cond_means, cond_ses = [], []
    for g in groups.values():
        clips = [r[clip_col] for r in g]
        cond_means.append(_mean(clips))
        cond_ses.append(_se(clips))
    clip_delta = max(cond_means) - min(cond_means)
    pooled_se = _mean([s for s in cond_ses if not math.isnan(s)])
    clip_delta_se = clip_delta / pooled_se if pooled_se > 0 else float("nan")
    return clip_delta_se, pooled_se


# Source: reports/experiments/exp1_sdxl/results.json
def load_exp1() -> tuple[float, float]:
    """exp1: quantization fp16/int8/nf4.  Returns (clip_delta_se, lpips_range)."""
    data = json.loads((ROOT / "reports/experiments/exp1_sdxl/results.json").read_text())
    rows = data["results"]
    clip_dse, _ = _delta_and_se(rows, "condition")
    # LPIPS range = max(non-ref condition means) - min(non-ref condition means)
    # fp16 is the reference (lpips=0 by construction, excluded from range)
    groups = _group(rows, "condition")
    lp_means = []
    for cond, g in groups.items():
        if cond == "fp16":
            continue
        lp_means.append(_mean([r["lpips"] for r in g]))
    lpips_range = max(lp_means) - min(lp_means)
    return clip_dse, lpips_range


# Source: reports/experiments/exp2_sdxl/results.json
def load_exp2() -> tuple[float, float]:
    """exp2: negative prompt absent/present.  Returns (clip_delta_se, lpips_range)."""
    data = json.loads((ROOT / "reports/experiments/exp2_sdxl/results.json").read_text())
    rows = data["results"]
    clip_dse, _ = _delta_and_se(rows, "condition")
    # LPIPS is mean paired cross-condition distance
    lpips_range = _mean([r["lpips_vs_no_neg"] for r in rows])
    return clip_dse, lpips_range


# Source: reports/experiments/exp3_sdxl/results.json
def load_exp3() -> tuple[float, float]:
    """exp3: CFG scale 1/3/5/7/9/12/15.  Returns (clip_delta_se, lpips_range)."""
    data = json.loads((ROOT / "reports/experiments/exp3_sdxl/results.json").read_text())
    rows = data["results"]
    clip_dse, _ = _delta_and_se(rows, "cfg_value")
    # LPIPS vs ref (cfg=7); exclude ref condition from range calculation
    groups_cfg = _group(rows, "cfg_value")
    cfg_lp = []
    for cfg, g in groups_cfg.items():
        if int(float(cfg)) == 7:
            continue
        vs = [r["lpips_vs_ref"] for r in g if r.get("lpips_vs_ref") is not None]
        if vs:
            cfg_lp.append(_mean(vs))
    # Range = max - min across non-ref condition means
    lpips_range = max(cfg_lp) - min(cfg_lp) if cfg_lp else float("nan")
    return clip_dse, lpips_range


# Source: reports/experiments/exp4_sdxl/results.json
def load_exp4() -> tuple[float, float]:
    """exp4: scheduler DDIM/DPM/EulerA/LMS.  Returns (clip_delta_se, lpips_range)."""
    data = json.loads((ROOT / "reports/experiments/exp4_sdxl/results.json").read_text())
    sched_agg = data["sched_agg"]
    pair_agg = data["pair_agg"]
    # CLIP delta/SE from per-scheduler aggregates (pre-computed in JSON)
    means = [v["mean_clip"] for v in sched_agg.values()]
    ses = [v["se_clip"] for v in sched_agg.values()]
    clip_delta = max(means) - min(means)
    pooled_se = _mean(ses)
    clip_dse = clip_delta / pooled_se
    # LPIPS range = max pairwise - min pairwise
    lp = [v["mean_lpips"] for v in pair_agg.values()]
    lpips_range = max(lp) - min(lp)
    return clip_dse, lpips_range


# Source: reports/experiments/exp5_sdxl/results.json
def load_exp5() -> tuple[float, float]:
    """exp5: ControlNet strength 0.0–1.5.  Returns (clip_delta_se, lpips_range)."""
    data = json.loads((ROOT / "reports/experiments/exp5_sdxl/results.json").read_text())
    rows = data["results"]
    clip_dse, _ = _delta_and_se(rows, "strength")
    groups_str = _group(rows, "strength")
    lp_means = []
    for s, g in groups_str.items():
        if float(s) == 1.0:
            continue
        vs = [r["lpips_vs_ref"] for r in g if r.get("lpips_vs_ref") is not None]
        if vs:
            lp_means.append(_mean(vs))
    # Range = max - min across non-ref condition means
    lpips_range = max(lp_means) - min(lp_means) if lp_means else float("nan")
    return clip_dse, lpips_range


# Source: reports/experiments/exp8_sdxl/results.json
def load_exp8() -> tuple[float, float]:
    """exp8: LoRA alpha 0.0–1.5.  Returns (clip_delta_se, lpips_range)."""
    data = json.loads((ROOT / "reports/experiments/exp8_sdxl/results.json").read_text())
    rows = data["results"]
    clip_dse, _ = _delta_and_se(rows, "alpha")
    groups_a = _group(rows, "alpha")
    lp_means = []
    for a, g in groups_a.items():
        if float(a) == 1.0:
            continue
        vs = [r["lpips_vs_ref"] for r in g if r.get("lpips_vs_ref") is not None]
        if vs:
            lp_means.append(_mean(vs))
    # Range = max - min across non-ref condition means
    lpips_range = max(lp_means) - min(lp_means) if lp_means else float("nan")
    return clip_dse, lpips_range


# Source: reports/experiments/exp9_sdxl/results.json
def load_exp9() -> tuple[float, float]:
    """exp9: trigger token absent/present.  Returns (clip_delta_se, lpips_range)."""
    data = json.loads((ROOT / "reports/experiments/exp9_sdxl/results.json").read_text())
    rows = data["results"]
    clip_dse, _ = _delta_and_se(rows, "condition")
    lpips_range = _mean([r["lpips_vs_no_trigger"] for r in rows])
    return clip_dse, lpips_range


# ── Chart ─────────────────────────────────────────────────────────────────────

# Verified expected values (from reports/clip_blindness_sdxl.md, lines 36–43)
# Used only for assertion; chart always plots from re-derived values above.
_EXPECTED_CLIP_SE = {
    "exp1": 0.24,
    "exp2": 1.09,
    "exp3": 7.01,
    "exp4": 0.67,
    "exp5": 1.66,
    "exp8": 7.21,
    "exp9": 0.84,
}
_EXPECTED_LPIPS = {
    "exp1": 0.203,
    "exp2": 0.374,
    "exp3": 0.343,
    "exp4": 0.452,
    "exp5": 0.618,
    "exp8": 0.295,
    "exp9": 0.301,
}


def _assert_close(label: str, derived: float, expected: float, tol: float = 0.02) -> None:
    if math.isnan(derived):
        print(f"  WARNING: {label} derived value is NaN")
        return
    if abs(derived - expected) > tol:
        print(f"  MISMATCH {label}: derived={derived:.3f} expected={expected:.3f}")
    else:
        print(f"  OK {label}: {derived:.3f} (expected {expected:.3f})")


def main() -> None:
    print("Loading experiment data from reports/experiments/exp*_sdxl/results.json …")
    exp1 = load_exp1()
    exp2 = load_exp2()
    exp3 = load_exp3()
    exp4 = load_exp4()
    exp5 = load_exp5()
    exp8 = load_exp8()
    exp9 = load_exp9()

    raw: dict[str, tuple[float, float]] = {
        "exp1": exp1,
        "exp2": exp2,
        "exp3": exp3,
        "exp4": exp4,
        "exp5": exp5,
        "exp8": exp8,
        "exp9": exp9,
    }

    print("\nVerification against reports/clip_blindness_sdxl.md (lines 36–43):")
    for eid, (clip_dse, lpips) in raw.items():
        _assert_close(f"{eid} CLIP_SE", clip_dse, _EXPECTED_CLIP_SE[eid])
        _assert_close(f"{eid} LPIPS", lpips, _EXPECTED_LPIPS[eid], tol=0.03)

    # ── Assemble rows sorted ascending by CLIP Δ SE ───────────────────────────
    rows = [
        # (short label,        clip_dse,    lpips_range, verdict)
        ("Quantization", raw["exp1"][0], raw["exp1"][1], "BLIND"),
        ("Scheduler", raw["exp4"][0], raw["exp4"][1], "BLIND"),
        ("Trigger token", raw["exp9"][0], raw["exp9"][1], "BLIND"),
        ("Neg prompt", raw["exp2"][0], raw["exp2"][1], "RESPONDS"),
        ("ControlNet strength", raw["exp5"][0], raw["exp5"][1], "RESPONDS"),
        ("CFG scale (1–15)", raw["exp3"][0], raw["exp3"][1], "RESPONDS"),
        ("LoRA alpha (0–1.5)", raw["exp8"][0], raw["exp8"][1], "RESPONDS"),
    ]
    rows.sort(key=lambda r: r[1])  # ascending clip_dse

    labels = [r[0] for r in rows]
    clip_ses = np.array([r[1] for r in rows])
    lpips_rngs = np.array([r[2] for r in rows])
    verdicts = [r[3] for r in rows]

    clip_colors = [TEAL if v == "BLIND" else ORANGE for v in verdicts]

    print("\nPlot data (sorted ascending by CLIP delta SE):")
    for lbl, cs, lr, vd in zip(labels, clip_ses, lpips_rngs, verdicts, strict=True):
        print(f"  {lbl:25s}  CLIP={cs:.2f} SE   LPIPS={lr:.3f}   [{vd}]")

    # ── Figure: two horizontal bar panels sharing Y axis ─────────────────────
    fig = plt.figure(figsize=(13, 5), facecolor=BG)
    fig.subplots_adjust(wspace=0.04)

    # Left panel (70% width): CLIP Δ SE
    ax_clip = fig.add_axes([0.22, 0.13, 0.50, 0.73])
    # Right panel (25% width): LPIPS range
    ax_lp = fig.add_axes([0.74, 0.13, 0.22, 0.73])

    y = np.arange(len(labels))

    # ── Left: CLIP Δ SE ───────────────────────────────────────────────────────
    bars = ax_clip.barh(y, clip_ses, color=clip_colors, height=0.55, edgecolor="none", zorder=3)
    ax_clip.set_facecolor(BG)
    ax_clip.grid(axis="x", color=GRID, linewidth=0.8, zorder=1)
    ax_clip.spines[["top", "right", "left"]].set_visible(False)
    ax_clip.spines["bottom"].set_color(GREY_LINE)
    ax_clip.tick_params(left=False)

    # Y-axis labels (experiment names)
    ax_clip.set_yticks(y)
    ax_clip.set_yticklabels(labels, fontsize=11)
    ax_clip.set_ylim(-0.6, len(labels) - 0.4)

    # X-axis
    ax_clip.set_xlabel("CLIP Δ  (standard-error units)", fontsize=11)
    ax_clip.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    # Threshold line at 1 SE
    ax_clip.axvline(1.0, color=GREY_LINE, linestyle="--", linewidth=1.4, zorder=2)
    ax_clip.text(
        1.05,
        len(labels) - 0.2,
        "1 SE\nthreshold",
        color=GREY_LINE,
        fontsize=8.5,
        fontstyle="italic",
        va="top",
    )

    # Value labels on bars
    for bar, cs in zip(bars, clip_ses, strict=True):
        offset = 0.15 if cs < 1.0 else 0.15
        ax_clip.text(
            cs + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{cs:.2f}",
            va="center",
            fontsize=9.5,
            color="#444444",
        )

    # Legend
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=TEAL, label="CLIP-BLIND   (< 1 SE)"),
        Patch(facecolor=ORANGE, label="CLIP RESPONDS (≥ 1 SE)"),
    ]
    ax_clip.legend(
        handles=legend_handles, loc="lower right", fontsize=9, framealpha=0.85, edgecolor=GRID
    )

    # ── Right: LPIPS range ────────────────────────────────────────────────────
    ax_lp.barh(y, lpips_rngs, color=LPIPS_BAR, height=0.55, alpha=0.75, edgecolor="none", zorder=3)
    ax_lp.set_facecolor(BG)
    ax_lp.grid(axis="x", color=GRID, linewidth=0.8, zorder=1)
    ax_lp.spines[["top", "right", "left"]].set_visible(False)
    ax_lp.spines["bottom"].set_color(GREY_LINE)
    ax_lp.tick_params(left=False)
    ax_lp.set_yticks([])
    ax_lp.set_ylim(-0.6, len(labels) - 0.4)
    ax_lp.set_xlabel("LPIPS range\n(perceptual change)", fontsize=10)
    ax_lp.set_xlim(0, 0.80)
    ax_lp.xaxis.set_major_locator(mticker.MultipleLocator(0.2))

    for bar, lr in zip(ax_lp.patches, lpips_rngs, strict=False):
        ax_lp.text(
            lr + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{lr:.3f}",
            va="center",
            fontsize=9,
            color="#444444",
        )

    # ── Titles ────────────────────────────────────────────────────────────────
    fig.text(
        0.47,
        0.96,
        "CLIP-Blindness Study — SDXL (7 experiments)",
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
        color="#111111",
    )
    fig.text(
        0.47,
        0.91,
        "Teal bars: CLIP cannot detect quality changes other metrics catch."
        "  |  Right panel: all experiments show large perceptual variation (LPIPS ≥ 0.20).",
        ha="center",
        va="top",
        fontsize=9.5,
        color="#555555",
    )

    # Source note
    fig.text(
        0.47,
        0.01,
        "Source: reports/experiments/exp*_sdxl/results.json  |  "
        "Reproduced by scripts/generate_clip_blindness_hero.py",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#999999",
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"\nHero chart saved: {OUT_PATH}")
    print(f"  Size: {OUT_PATH.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
