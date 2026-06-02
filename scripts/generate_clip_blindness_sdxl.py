#!/usr/bin/env python3
"""Generate CLIP-blindness replication report for SDXL experiments.

Reads results.json from each of 9 SDXL experiments and compares against
their SD 2.1 counterparts to answer: does the CLIP-blindness finding replicate
on SDXL, or is it architecture-dependent?

Outputs:
    reports/clip_blindness_sdxl.md   -- narrative report with per-experiment tables
    reports/clip_blindness_sdxl_chart.png -- two-panel bar chart (CLIP Δ SE vs max LPIPS)

Run from project root:
    python scripts/generate_clip_blindness_sdxl.py
"""

from __future__ import annotations

import json
import statistics
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Attempt to import project visualization; fall back gracefully to plain matplotlib.
try:
    from aetherart.visualization.charts import BLUE, GREY, ORANGE, ChartCanvas

    _HAS_CHART_CANVAS = True
except Exception:  # noqa: BLE001
    _HAS_CHART_CANVAS = False
    BLUE = "#2563EB"
    ORANGE = "#F97316"
    GREY = "#6B7280"
    ChartCanvas = None  # type: ignore[assignment]

# ── Experiment metadata ───────────────────────────────────────────────────────

# Human-readable label and the variable swept, aligned with clip_blindness.md.
EXP_META: list[dict[str, str]] = [
    {"id": "exp1", "label": "Exp 1\nQuant", "variable": "Quantization (fp16 / INT8 / NF4)"},
    {"id": "exp2", "label": "Exp 2\nNeg prompt", "variable": "Negative prompt (off / on)"},
    {"id": "exp3", "label": "Exp 3\nCFG", "variable": "CFG scale"},
    {"id": "exp4", "label": "Exp 4\nScheduler", "variable": "Scheduler"},
    {"id": "exp5", "label": "Exp 5\nControlNet", "variable": "ControlNet strength"},
    {"id": "exp6", "label": "Exp 6\nLoRA rank", "variable": "LoRA rank"},
    {"id": "exp7", "label": "Exp 7\nLoRA data", "variable": "LoRA training data size"},
    {"id": "exp8", "label": "Exp 8\nLoRA alpha", "variable": "LoRA alpha / style scale"},
    {"id": "exp9", "label": "Exp 9\nTrigger", "variable": "LoRA trigger token"},
]

# SD 2.1 reference data from reports/clip_blindness.md (frozen at PR-12 merge).
# Index 0 = exp1, index 8 = exp9.
SD21_CLIP_SE = [0.94, 0.83, 1.10, 1.80, 2.20, 1.00, 0.80, 4.00, 0.12]
SD21_MAX_LPIPS = [0.40, 0.46, 0.47, 0.73, 0.72, 0.50, 0.66, 0.67, 0.41]

# CLIP-blindness criterion (mirrors the SD 2.1 study):
#   CLIP Δ < 2 SE  AND  LPIPS range > 0.10
CLIP_BLIND_SE_THRESHOLD = 2.0
CLIP_BLIND_LPIPS_MIN = 0.10

# Output paths
REPORT_PATH = ROOT / "reports" / "clip_blindness_sdxl.md"
CHART_PATH = ROOT / "reports" / "clip_blindness_sdxl_chart.png"


# ── Data loading ──────────────────────────────────────────────────────────────


def load_exp_results(exp_json: Path) -> dict | None:
    """Load and return parsed JSON from an experiment results file.

    Returns None if the file does not exist or cannot be parsed, printing a
    warning so the caller can skip the experiment gracefully.
    """
    if not exp_json.exists():
        warnings.warn(f"results.json not found: {exp_json}", stacklevel=2)
        return None
    try:
        with open(exp_json, encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        warnings.warn(f"JSON parse error in {exp_json}: {exc}", stacklevel=2)
        return None


# ── Statistics helpers ────────────────────────────────────────────────────────


def _se(values: list[float]) -> float:
    """Standard error of the mean."""
    n = len(values)
    if n < 2:
        return 0.0
    return statistics.stdev(values) / n**0.5


def _safe_mean(values: list[float]) -> float:
    """Mean of a list; returns 0.0 for empty lists."""
    return statistics.mean(values) if values else 0.0


# ── Per-experiment computation ────────────────────────────────────────────────


def compute_exp_stats(results: dict) -> dict:
    """Compute per-condition means/SEs and derive CLIP-blindness verdict.

    Args:
        results: Parsed results.json dict from one SDXL experiment.

    Returns:
        Dict with keys:
            "conditions"      -- list of condition names (in order)
            "per_condition"   -- {cond: {mean_clip, se_clip, mean_hps, se_hps,
                                         mean_ir, se_ir, mean_lpips}}
            "clip_delta_se"   -- max |CLIP Δ| expressed in SE units (pooled SE)
            "clip_delta_raw"  -- max |CLIP Δ| in raw score units
            "hps_delta"       -- max |HPS Δ| from reference condition
            "ir_delta"        -- max |IR Δ| from reference condition
            "lpips_range"     -- max LPIPS observed across conditions
            "verdict"         -- True if CLIP-blind by the 2-SE / LPIPS criterion
            "reference_cond"  -- name of the reference/baseline condition
    """
    rows: list[dict] = results.get("results", [])
    conditions: list[str] = results.get("conditions", [])

    if not rows or not conditions:
        return {
            "conditions": [],
            "per_condition": {},
            "clip_delta_se": 0.0,
            "clip_delta_raw": 0.0,
            "hps_delta": 0.0,
            "ir_delta": 0.0,
            "lpips_range": 0.0,
            "verdict": False,
            "reference_cond": "",
        }

    # Group rows by condition.
    by_cond: dict[str, list[dict]] = {c: [] for c in conditions}
    for row in rows:
        cond = row.get("condition", "")
        if cond in by_cond:
            by_cond[cond].append(row)

    # Per-condition aggregates.
    per_cond: dict[str, dict] = {}
    for cond, cond_rows in by_cond.items():
        clips = [r["clip_score"] for r in cond_rows if r.get("clip_score") is not None]
        hps_vals = [r["hps_score"] for r in cond_rows if r.get("hps_score") is not None]
        ir_vals = [r["ir_score"] for r in cond_rows if r.get("ir_score") is not None]
        # LPIPS may be 0.0 for reference condition (by convention); include all.
        lpips_vals = [r["lpips"] for r in cond_rows if r.get("lpips") is not None]

        per_cond[cond] = {
            "n": len(cond_rows),
            "mean_clip": _safe_mean(clips),
            "se_clip": _se(clips),
            "mean_hps": _safe_mean(hps_vals),
            "se_hps": _se(hps_vals),
            "mean_ir": _safe_mean(ir_vals),
            "se_ir": _se(ir_vals),
            "mean_lpips": _safe_mean(lpips_vals),
        }

    # Reference = first condition listed (matches all exp scripts' convention).
    ref_cond = conditions[0]
    ref = per_cond[ref_cond]

    # Pooled SE across conditions (use reference SE as denominator, matching SD 2.1 study).
    pooled_se = ref["se_clip"] if ref["se_clip"] > 0 else 1e-9

    # Max absolute CLIP delta (any condition vs reference).
    clip_deltas_raw = [
        abs(per_cond[c]["mean_clip"] - ref["mean_clip"])
        for c in conditions
        if c != ref_cond
    ]
    max_clip_delta_raw = max(clip_deltas_raw) if clip_deltas_raw else 0.0
    max_clip_delta_se = max_clip_delta_raw / pooled_se

    # Max absolute HPS / IR delta.
    hps_deltas = [
        abs(per_cond[c]["mean_hps"] - ref["mean_hps"])
        for c in conditions
        if c != ref_cond
    ]
    ir_deltas = [
        abs(per_cond[c]["mean_ir"] - ref["mean_ir"])
        for c in conditions
        if c != ref_cond
    ]
    max_hps_delta = max(hps_deltas) if hps_deltas else 0.0
    max_ir_delta = max(ir_deltas) if ir_deltas else 0.0

    # LPIPS range = max mean LPIPS across all non-reference conditions.
    lpips_vals_nonref = [
        per_cond[c]["mean_lpips"]
        for c in conditions
        if c != ref_cond and per_cond[c]["mean_lpips"] > 0
    ]
    lpips_range = max(lpips_vals_nonref) if lpips_vals_nonref else 0.0

    # Verdict: CLIP-blind if Δ < 2 SE AND LPIPS > 0.10 threshold.
    verdict = (max_clip_delta_se < CLIP_BLIND_SE_THRESHOLD) and (
        lpips_range > CLIP_BLIND_LPIPS_MIN
    )

    return {
        "conditions": conditions,
        "per_condition": per_cond,
        "clip_delta_se": round(max_clip_delta_se, 3),
        "clip_delta_raw": round(max_clip_delta_raw, 6),
        "hps_delta": round(max_hps_delta, 6),
        "ir_delta": round(max_ir_delta, 6),
        "lpips_range": round(lpips_range, 4),
        "verdict": verdict,
        "reference_cond": ref_cond,
    }


# ── Chart generation ──────────────────────────────────────────────────────────


def generate_chart(
    exp_labels: list[str],
    clip_se_sdxl: list[float],
    lpips_sdxl: list[float],
    clip_se_sd21: list[float],
    lpips_sd21: list[float],
) -> None:
    """Write a two-panel comparison chart to CHART_PATH.

    Left panel: CLIP Δ (SE units) — SDXL vs SD 2.1 grouped bars.
    Right panel: Max LPIPS — SDXL vs SD 2.1 grouped bars.

    Falls back to plain matplotlib if ChartCanvas is unavailable.
    """
    n = len(exp_labels)
    x = np.arange(n)
    width = 0.35

    # Colours: SDXL = BLUE, SD 2.1 = ORANGE
    c_sdxl = BLUE
    c_sd21 = ORANGE

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(17, 6))
    fig.patch.set_facecolor("white")

    def _draw_grouped(
        ax: plt.Axes,
        vals_a: list[float],
        vals_b: list[float],
        label_a: str,
        label_b: str,
        title: str,
        ylabel: str,
        threshold: float | None = None,
        threshold_label: str = "",
    ) -> None:
        bars_a = ax.bar(x - width / 2, vals_a, width, color=c_sdxl, label=label_a, zorder=3)
        bars_b = ax.bar(x + width / 2, vals_b, width, color=c_sd21, label=label_b, zorder=3)

        # Value annotations above each bar.
        for bar in (*bars_a, *bars_b):
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 0.03,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

        ax.set_title(title, fontsize=10, fontweight="bold", pad=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(exp_labels, fontsize=8)
        ax.legend(fontsize=8)
        ax.set_facecolor("#F9FAFB")
        ax.grid(axis="y", alpha=0.4, zorder=0)

        if threshold is not None:
            ax.axhline(threshold, color=GREY, lw=1.5, ls="--", alpha=0.85, zorder=2)
            ax.text(
                0.015,
                threshold + 0.04,
                threshold_label,
                color=GREY,
                fontsize=7.5,
                fontstyle="italic",
                transform=ax.get_yaxis_transform(),
            )

    _draw_grouped(
        ax_left,
        clip_se_sdxl,
        clip_se_sd21,
        label_a="SDXL",
        label_b="SD 2.1",
        title="CLIP sensitivity: delta (standard-error units)",
        ylabel="CLIP Δ (SE units)",
        threshold=CLIP_BLIND_SE_THRESHOLD,
        threshold_label=f"{CLIP_BLIND_SE_THRESHOLD} SE — blind threshold",
    )
    ax_left.set_ylim(0, max(max(clip_se_sdxl + clip_se_sd21, default=1.0), 5.0) * 1.25)

    _draw_grouped(
        ax_right,
        lpips_sdxl,
        lpips_sd21,
        label_a="SDXL",
        label_b="SD 2.1",
        title="Perceptual change: max LPIPS vs reference condition",
        ylabel="Max LPIPS (higher = more perceptually different)",
        threshold=CLIP_BLIND_LPIPS_MIN,
        threshold_label=f"{CLIP_BLIND_LPIPS_MIN} LPIPS — blindness criterion floor",
    )
    ax_right.set_ylim(0, 1.0)

    fig.suptitle(
        "CLIP-Blindness Replication: SDXL vs SD 2.1 — 9 experiments",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
    if _HAS_CHART_CANVAS and ChartCanvas is not None:
        ChartCanvas.save_fig(fig, str(CHART_PATH), dpi=130, bottom_adjust=0.22)
    else:
        fig.tight_layout()
        fig.savefig(str(CHART_PATH), dpi=130, bbox_inches="tight")
        plt.close(fig)
    print(f"Chart written: {CHART_PATH}")


# ── Report generation ─────────────────────────────────────────────────────────


def _verdict_str(v: bool) -> str:
    return "YES" if v else "no"


def _condition_table(conditions: list[str], per_cond: dict[str, dict]) -> str:
    """Format a per-condition metric table for the Experiment Detail section."""
    header = "| Condition | CLIP (mean ± SE) | HPS (mean ± SE) | IR (mean ± SE) | LPIPS (mean) |"
    sep = "|-----------|:----------------:|:---------------:|:--------------:|:------------:|"
    lines = [header, sep]
    for cond in conditions:
        pc = per_cond[cond]
        clip_str = f"{pc['mean_clip']:.4f} ± {pc['se_clip']:.4f}"
        hps_str = f"{pc['mean_hps']:.4f} ± {pc['se_hps']:.4f}"
        ir_str = f"{pc['mean_ir']:.4f} ± {pc['se_ir']:.4f}"
        lpips_str = f"{pc['mean_lpips']:.4f}"
        lines.append(f"| {cond} | {clip_str} | {hps_str} | {ir_str} | {lpips_str} |")
    return "\n".join(lines)


def generate_report(
    sdxl_stats: list[dict | None],
    sd21_paths: list[Path | None],
) -> None:
    """Write the full Markdown report to REPORT_PATH."""
    # Summary verdict row data.
    verdict_rows: list[str] = []
    exp_labels_short: list[str] = []
    clip_se_sdxl: list[float] = []
    lpips_sdxl: list[float] = []

    for i, stats in enumerate(sdxl_stats):
        meta = EXP_META[i]
        exp_id = f"exp{i + 1}_sdxl"
        if stats is None:
            verdict_rows.append(
                f"| {exp_id} | {meta['variable']} | — | — | — | — | MISSING |"
            )
            exp_labels_short.append(meta["label"])
            clip_se_sdxl.append(0.0)
            lpips_sdxl.append(0.0)
        else:
            verdict = _verdict_str(stats["verdict"])
            verdict_rows.append(
                f"| {exp_id} | {meta['variable']} "
                f"| {stats['clip_delta_se']:.2f} SE "
                f"| {stats['hps_delta']:.4f} "
                f"| {stats['ir_delta']:.4f} "
                f"| {stats['lpips_range']:.4f} "
                f"| **{verdict}** |"
            )
            exp_labels_short.append(meta["label"])
            clip_se_sdxl.append(stats["clip_delta_se"])
            lpips_sdxl.append(stats["lpips_range"])

    # Count CLIP-blind experiments.
    n_loaded = sum(1 for s in sdxl_stats if s is not None)
    n_blind_sdxl = sum(1 for s in sdxl_stats if s is not None and s["verdict"])
    n_blind_sd21 = sum(
        1
        for clip_se, lpips in zip(SD21_CLIP_SE, SD21_MAX_LPIPS)
        if clip_se < CLIP_BLIND_SE_THRESHOLD and lpips > CLIP_BLIND_LPIPS_MIN
    )

    # Overall conclusion paragraph.
    if n_loaded == 0:
        conclusion = (
            "No SDXL results found. Run exp1_sdxl.py through exp9_sdxl.py first."
        )
    elif n_blind_sdxl == n_loaded:
        conclusion = (
            f"CLIP-blindness **fully replicates** on SDXL: {n_blind_sdxl}/{n_loaded} loaded "
            f"experiments are CLIP-blind. This matches the SD 2.1 finding "
            f"({n_blind_sd21}/9 on SD 2.1) and confirms the result is not architecture-dependent "
            f"— it is a structural property of the CLIP metric itself, independent of model scale, "
            f"resolution (512px vs 1024px), or VAE precision."
        )
    elif n_blind_sdxl > n_loaded // 2:
        conclusion = (
            f"CLIP-blindness **mostly replicates** on SDXL: {n_blind_sdxl}/{n_loaded} loaded "
            f"experiments are CLIP-blind (SD 2.1 baseline: {n_blind_sd21}/9). "
            f"The minority of non-blind experiments may reflect SDXL's higher expressive capacity "
            f"at 1024×1024 producing more semantically distinctive outputs for certain parameter "
            f"ranges. The core finding — CLIP is insensitive to perceptual variation when semantic "
            f"content is preserved — holds on SDXL."
        )
    else:
        conclusion = (
            f"CLIP-blindness **does not fully replicate** on SDXL: only {n_blind_sdxl}/{n_loaded} "
            f"loaded experiments are CLIP-blind (SD 2.1 baseline: {n_blind_sd21}/9). "
            f"This suggests the finding may be architecture-dependent. SDXL's larger capacity "
            f"and 1024×1024 resolution may produce outputs where CLIP can differentiate more "
            f"parameter conditions. Further investigation is warranted."
        )

    # Per-experiment detail sections.
    detail_sections: list[str] = []
    for i, stats in enumerate(sdxl_stats):
        meta = EXP_META[i]
        exp_id = f"exp{i + 1}_sdxl"
        if stats is None:
            detail_sections.append(
                f"### {exp_id} — {meta['variable']}\n\n"
                f"*Results not found — experiment did not run or output is missing.*\n"
            )
            continue

        sd21_clip_se = SD21_CLIP_SE[i]
        sd21_lpips = SD21_MAX_LPIPS[i]
        table = _condition_table(stats["conditions"], stats["per_condition"])
        verdict_word = "CLIP-blind" if stats["verdict"] else "not CLIP-blind"

        detail_sections.append(
            f"### {exp_id} — {meta['variable']}\n\n"
            f"**CLIP Δ:** {stats['clip_delta_se']:.2f} SE "
            f"(raw: {stats['clip_delta_raw']:.4f}) | "
            f"**HPS Δ:** {stats['hps_delta']:.4f} | "
            f"**IR Δ:** {stats['ir_delta']:.4f} | "
            f"**LPIPS range:** {stats['lpips_range']:.4f}\n\n"
            f"**Verdict: {verdict_word.upper()}** "
            f"({'CLIP Δ < 2 SE' if stats['clip_delta_se'] < CLIP_BLIND_SE_THRESHOLD else 'CLIP Δ ≥ 2 SE'}"
            f" AND "
            f"{'LPIPS > 0.10' if stats['lpips_range'] > CLIP_BLIND_LPIPS_MIN else 'LPIPS ≤ 0.10'})\n\n"
            f"SD 2.1 comparison: CLIP Δ was {sd21_clip_se:.2f} SE, LPIPS {sd21_lpips:.2f} "
            f"({'blind' if sd21_clip_se < CLIP_BLIND_SE_THRESHOLD and sd21_lpips > CLIP_BLIND_LPIPS_MIN else 'not blind'})\n\n"
            f"{table}\n"
        )

    # Raw data links.
    raw_data_lines: list[str] = []
    for i in range(9):
        exp_id = f"exp{i + 1}_sdxl"
        raw_data_lines.append(
            f"- `reports/experiments/{exp_id}/results.json` "
            f"/ `reports/experiments/{exp_id}/results.csv`"
        )

    report = f"""\
# CLIP Blindness Replication — SDXL vs SD 2.1

**Run:** GCP L4, us-central1-a, g2-standard-4
**Date:** 2026-06-02
**Model:** stabilityai/stable-diffusion-xl-base-1.0 + madebyollin/sdxl-vae-fp16-fix (G1)
**Scorers:** CLIP (comparison-only), HPSv2.1, ImageReward, LPIPS
**Resolution:** 1024×1024 (SD 2.1 baseline: 512×512)

---

## Research Question

Does the CLIP-blindness finding from SD 2.1 (reported in reports/clip_blindness.md)
replicate on SDXL, or is it architecture-dependent?

**SD 2.1 finding:** {n_blind_sd21}/9 experiments were CLIP-blind — CLIP score stayed
flat (< 2 SE delta) while images changed substantially (LPIPS 0.40–0.73). The one
partial exception was Exp 8 (LoRA alpha: 4.00 SE for the no-LoRA → active-LoRA jump,
then blind within the active range).

---

## Per-Experiment Verdict

| Exp | Variable swept | CLIP Δ (SEs) | HPS Δ | IR Δ | LPIPS range | CLIP-blind? |
|-----|----------------|:------------:|------:|-----:|:-----------:|:-----------:|
{chr(10).join(verdict_rows)}

*CLIP-blind criterion: |CLIP Δ| < {CLIP_BLIND_SE_THRESHOLD} SE AND LPIPS > {CLIP_BLIND_LPIPS_MIN}*

---

## Overall Conclusion

{conclusion}

![CLIP-blindness replication chart: SDXL vs SD 2.1](clip_blindness_sdxl_chart.png)

---

## Experiment Detail Tables

{chr(10).join(detail_sections)}

---

## Methodology Notes

- **CLIP is comparison-only** in this study. The PR 12 finding (CLIP-blindness across SD 2.1
  experiments) established that CLIP is not suitable as a primary parameter-tuning metric for
  this style-transfer domain. CLIP values are reported here for completeness and cross-study
  comparison, not as a quality signal.
- **Primary evaluation metrics: HPSv2.1 and ImageReward.** Both are human-preference-aligned
  scorers trained on rated image-text pairs.
- **LPIPS** (Learned Perceptual Image Patch Similarity, AlexNet backbone) measures perceptual
  distance from the reference condition. Higher = more perceptually different.
- **CLIP-blindness criterion:** |CLIP Δ| < {CLIP_BLIND_SE_THRESHOLD}×SE while LPIPS > {CLIP_BLIND_LPIPS_MIN}. This is the same
  criterion used in the SD 2.1 study, enabling direct comparison.
- **Resolution difference:** SDXL runs at 1024×1024 vs SD 2.1 at 512×512. LPIPS values are
  not directly comparable across architectures (different reference images, different style
  expressions at different resolutions). The comparison is ordinal, not exact.
- **Scorer versions:** HPSv2.1 (hpsv2==1.2.0 with turtle import fix applied), ImageReward 1.5,
  CLIP openai/clip-vit-base-patch32 via openai-clip==1.0.1.

---

## Raw Data

{chr(10).join(raw_data_lines)}

Reproduce this report:

```bash
python scripts/generate_clip_blindness_sdxl.py
```
"""

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(f"Report written: {REPORT_PATH}")

    # Generate companion chart.
    generate_chart(
        exp_labels=exp_labels_short,
        clip_se_sdxl=clip_se_sdxl,
        lpips_sdxl=lpips_sdxl,
        clip_se_sd21=SD21_CLIP_SE,
        lpips_sd21=SD21_MAX_LPIPS,
    )


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    """Locate SDXL results, compute stats, and write report + chart."""
    # Match exp{N}_*_sdxl/results.json — the glob picks up the descriptive suffix
    # (e.g. exp1_quantization_quality_sdxl) while still requiring the _sdxl suffix.
    sdxl_stats: list[dict | None] = []
    sd21_paths: list[Path | None] = []

    for meta in EXP_META:
        exp_id_prefix = meta["id"]  # e.g. "exp1"

        # SDXL result: find a directory matching exp{N}_*_sdxl under reports/experiments.
        sdxl_matches = sorted(
            ROOT.glob(f"reports/experiments/{exp_id_prefix}_*_sdxl/results.json")
        )
        # Fall back to exact exp{N}_sdxl directory (as written in exp1_sdxl.py: "exp1_sdxl").
        if not sdxl_matches:
            sdxl_matches = sorted(
                ROOT.glob(f"reports/experiments/{exp_id_prefix}_sdxl/results.json")
            )

        if sdxl_matches:
            sdxl_json = sdxl_matches[0]  # take first if multiple
            if len(sdxl_matches) > 1:
                warnings.warn(
                    f"Multiple SDXL results for {exp_id_prefix}: {sdxl_matches}. "
                    f"Using {sdxl_json}.",
                    stacklevel=1,
                )
            data = load_exp_results(sdxl_json)
            sdxl_stats.append(compute_exp_stats(data) if data is not None else None)
        else:
            warnings.warn(
                f"No SDXL results directory found for {exp_id_prefix} "
                f"(expected: reports/experiments/{exp_id_prefix}_*_sdxl/results.json)",
                stacklevel=1,
            )
            sdxl_stats.append(None)

        # SD 2.1 counterpart (for reference; cross-arch comparison in report).
        sd21_matches = sorted(
            ROOT.glob(
                f"reports/experiments/{exp_id_prefix}_*/results.json"
            )
        )
        # Exclude any _sdxl directories from SD 2.1 matches.
        sd21_matches = [p for p in sd21_matches if "_sdxl" not in p.parent.name]
        sd21_paths.append(sd21_matches[0] if sd21_matches else None)

    n_found = sum(1 for s in sdxl_stats if s is not None)
    print(f"Loaded {n_found}/9 SDXL experiment result files.")

    generate_report(sdxl_stats, sd21_paths)
    print("Done.")


if __name__ == "__main__":
    main()
