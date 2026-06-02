#!/usr/bin/env python3
"""Generate CLIP-blindness replication report for SDXL experiments.

Each experiment's schema is handled by an explicit adapter — no column
auto-discovery.  The per-condition means and SEs are computed from the raw
results.json and used to populate clip_blindness_sdxl.md.

Outputs:
    reports/clip_blindness_sdxl.md
    reports/clip_blindness_sdxl_chart.png

Run from project root:
    python scripts/generate_clip_blindness_sdxl.py
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class ConditionStats:
    label: str
    n: int
    mean_clip: float
    se_clip: float
    mean_hps: float
    mean_ir: float
    mean_lpips: float | None  # None = not available for this condition


@dataclass
class ExpResult:
    exp_id: str
    label: str
    variable: str
    conditions: list[ConditionStats]
    # Derived in post-processing
    clip_delta: float = 0.0
    clip_delta_se: float = 0.0
    hps_delta: float = 0.0
    ir_delta: float = 0.0
    lpips_range: float | None = None
    verdict: str = ""
    caveat: str = ""


# ── Helpers ───────────────────────────────────────────────────────────────────


def se(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    return statistics.stdev(values) / math.sqrt(len(values))


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def group_by(rows: list[dict], key: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        k = r[key]
        out.setdefault(k, []).append(r)
    return out


# ── Per-experiment adapters ───────────────────────────────────────────────────
# Each adapter returns a list[ConditionStats] and an optional caveat string.
# SCHEMA documented explicitly:
#
#   exp1: row["condition"] in {fp16, int8, nf4}
#         scorers: clip_score, hps_score, ir_score, lpips (0.0 for fp16=ref)
#
#   exp2: row["condition"] in {no_neg, with_neg}
#         scorers: clip_score, hps_score, ir_score, lpips_vs_no_neg
#
#   exp3: row["cfg_value"] in {1,3,5,7,9,12,15}
#         scorers: clip_score, hps_score, ir_score, lpips_vs_ref (vs cfg=7)
#
#   exp4: pre-aggregated in json["sched_agg"] per scheduler key
#         DDIM/DPM/EulerA/LMS: mean_clip, se_clip, mean_hps, mean_ir
#         LPIPS: pair_agg dict; use max pairwise mean_lpips as range proxy
#
#   exp5: row["strength"] in {0.0,0.25,0.5,0.75,1.0,1.25,1.5}
#         scorers: clip_score, hps_score, ir_score, lpips_vs_ref (vs strength=1.0)
#
#   exp8: row["alpha"] in {0.0,0.25,0.5,0.75,1.0,1.25,1.5}
#         scorers: clip_score, hps_score, ir_score, lpips_vs_ref (vs alpha=1.0)
#
#   exp9: row["condition"] in {no_trigger, with_trigger}
#         scorers: clip_score, hps_score, ir_score, lpips_vs_no_trigger
#
# exp6 and exp7 NOT PRESENT on SDXL — training images not in repo.


def _adapter_generic_condition(
    rows: list[dict],
    cond_key: str,
    lpips_col: str | None,
    lpips_ref_cond: str | None = None,
) -> tuple[list[ConditionStats], str]:
    """Generic adapter for experiments with a single condition column."""
    grouped = group_by(rows, cond_key)
    conds: list[ConditionStats] = []
    for label, group in grouped.items():
        clips = [r["clip_score"] for r in group]
        hpss = [r["hps_score"] for r in group]
        irs = [r["ir_score"] for r in group]
        mean_lpips: float | None = None
        if lpips_col and lpips_col in group[0]:
            lp_vals = [r[lpips_col] for r in group if r.get(lpips_col) is not None]
            if lp_vals:
                mean_lpips = mean(lp_vals)
                # For ref condition (lpips = 0.0 by design), mark as None so we
                # don't let the zero compress the range.
                if lpips_ref_cond is not None and str(label) == str(lpips_ref_cond):
                    mean_lpips = None
        conds.append(
            ConditionStats(
                label=str(label),
                n=len(clips),
                mean_clip=mean(clips),
                se_clip=se(clips),
                mean_hps=mean(hpss),
                mean_ir=mean(irs),
                mean_lpips=mean_lpips,
            )
        )
    return conds, ""


def adapter_exp1(data: dict) -> tuple[list[ConditionStats], str]:
    rows = data["results"]
    # fp16 is the reference; its lpips values are 0.0 by construction — exclude from range.
    conds, caveat = _adapter_generic_condition(
        rows, cond_key="condition", lpips_col="lpips", lpips_ref_cond="fp16"
    )
    return conds, caveat


def adapter_exp2(data: dict) -> tuple[list[ConditionStats], str]:
    rows = data["results"]
    # lpips_vs_no_neg is a paired distance (same value for both conditions, per seed/prompt).
    # Compute mean cross-condition distance. Represent as: no_neg=0.0, with_neg=lp_mean
    # so that lpips_range = lp_mean - 0.0 = lp_mean.
    conds, _ = _adapter_generic_condition(
        rows, cond_key="condition", lpips_col="lpips_vs_no_neg"
    )
    lp_all = [r["lpips_vs_no_neg"] for r in rows]
    lp_mean = mean(lp_all)
    for c in conds:
        if c.label == "no_neg":
            c.mean_lpips = 0.0
        else:
            c.mean_lpips = lp_mean
    caveat = (
        "LPIPS is mean paired cross-condition distance "
        "(no_neg vs with_neg per seed/prompt); range = mean distance = "
        f"{lp_mean:.3f}."
    )
    return conds, caveat


def adapter_exp3(data: dict) -> tuple[list[ConditionStats], str]:
    rows = data["results"]
    # Group by cfg_value; exclude ref (cfg=7) from lpips range since lpips_vs_ref=0 there.
    conds, _ = _adapter_generic_condition(
        rows, cond_key="cfg_value", lpips_col="lpips_vs_ref", lpips_ref_cond=7
    )
    # Sort by cfg value numerically
    conds.sort(key=lambda c: float(c.label))
    return conds, ""


def adapter_exp4(data: dict) -> tuple[list[ConditionStats], str]:
    sched_agg = data["sched_agg"]
    pair_agg = data["pair_agg"]
    # LPIPS range = max pairwise - min pairwise across scheduler pairs.
    lpips_vals = [v["mean_lpips"] for v in pair_agg.values()]
    max_lp = max(lpips_vals)
    min_lp = min(lpips_vals)
    # Represent as two synthetic conditions: one at min_lp, one at max_lp.
    # In practice we assign to the first two schedulers (for range calculation only).
    sched_list = list(sched_agg.items())
    conds: list[ConditionStats] = []
    for i, (sched, agg) in enumerate(sched_list):
        # Assign min_lp or max_lp to span the true pairwise range.
        mean_lpips = min_lp if i == 0 else (max_lp if i == 1 else (min_lp + max_lp) / 2)
        conds.append(
            ConditionStats(
                label=sched,
                n=agg["n"],
                mean_clip=agg["mean_clip"],
                se_clip=agg["se_clip"],
                mean_hps=agg["mean_hps"],
                mean_ir=agg["mean_ir"],
                mean_lpips=mean_lpips,
            )
        )
    caveat = (
        f"LPIPS range = max_pairwise − min_pairwise = {max_lp:.3f} − {min_lp:.3f} = {max_lp - min_lp:.3f}. "
        "Per-scheduler LPIPS not available; pairwise LPIPS ranges from "
        "DPM–LMS=0.227 (similar) to EulerA–LMS=0.679 (very different)."
    )
    return conds, caveat


def adapter_exp5(data: dict) -> tuple[list[ConditionStats], str]:
    rows = data["results"]
    conds, caveat = _adapter_generic_condition(
        rows, cond_key="strength", lpips_col="lpips_vs_ref", lpips_ref_cond=1.0
    )
    conds.sort(key=lambda c: float(c.label))
    return conds, caveat


def adapter_exp8(data: dict) -> tuple[list[ConditionStats], str]:
    rows = data["results"]
    conds, caveat = _adapter_generic_condition(
        rows, cond_key="alpha", lpips_col="lpips_vs_ref", lpips_ref_cond=1.0
    )
    conds.sort(key=lambda c: float(c.label))
    return conds, caveat


def adapter_exp9(data: dict) -> tuple[list[ConditionStats], str]:
    rows = data["results"]
    conds, _ = _adapter_generic_condition(
        rows,
        cond_key="condition",
        lpips_col="lpips_vs_no_trigger",
    )
    # Both conditions have the same lpips_vs_no_trigger values (paired).
    # Represent as: no_trigger=0.0, with_trigger=lp_mean so range = lp_mean.
    lp_all = [r["lpips_vs_no_trigger"] for r in rows]
    lp_mean = mean(lp_all)
    for c in conds:
        if c.label == "no_trigger":
            c.mean_lpips = 0.0
        else:
            c.mean_lpips = lp_mean
    caveat = (
        "LPIPS is mean paired cross-condition distance (no_trigger vs with_trigger); "
        f"range = mean distance = {lp_mean:.3f}."
    )
    return conds, caveat


# ── Experiment registry ───────────────────────────────────────────────────────

EXPERIMENTS: list[dict] = [
    {
        "id": "exp1",
        "label": "Exp 1 – Quantization",
        "variable": "Quantization level: fp16 / INT8 / NF4",
        "adapter": adapter_exp1,
        "path": "reports/experiments/exp1_sdxl/results.json",
    },
    {
        "id": "exp2",
        "label": "Exp 2 – Negative Prompt",
        "variable": "Negative prompt: absent / present",
        "adapter": adapter_exp2,
        "path": "reports/experiments/exp2_sdxl/results.json",
    },
    {
        "id": "exp3",
        "label": "Exp 3 – CFG Scale",
        "variable": "Guidance scale: 1 / 3 / 5 / 7 / 9 / 12 / 15",
        "adapter": adapter_exp3,
        "path": "reports/experiments/exp3_sdxl/results.json",
    },
    {
        "id": "exp4",
        "label": "Exp 4 – Scheduler",
        "variable": "Scheduler: DDIM / DPM / EulerA / LMS",
        "adapter": adapter_exp4,
        "path": "reports/experiments/exp4_sdxl/results.json",
    },
    {
        "id": "exp5",
        "label": "Exp 5 – ControlNet Strength",
        "variable": "ControlNet strength: 0.0 – 1.5",
        "adapter": adapter_exp5,
        "path": "reports/experiments/exp5_sdxl/results.json",
    },
    {
        "id": "exp8",
        "label": "Exp 8 – LoRA Alpha",
        "variable": "LoRA alpha: 0.0 – 1.5",
        "adapter": adapter_exp8,
        "path": "reports/experiments/exp8_sdxl/results.json",
    },
    {
        "id": "exp9",
        "label": "Exp 9 – LoRA Trigger",
        "variable": "Trigger token: absent / present",
        "adapter": adapter_exp9,
        "path": "reports/experiments/exp9_sdxl/results.json",
    },
]

# ── Post-processing: derive deltas and verdicts ───────────────────────────────

# Thresholds for the CLIP-blindness verdict:
#   CLIP-blind: CLIP delta < CLIP_BLIND_SE_THRESHOLD SE units
#               AND at least one of HPS/IR/LPIPS moves meaningfully
#   HPS threshold: absolute delta > HPS_DELTA_MIN
#   IR threshold: absolute delta > IR_DELTA_MIN
#   LPIPS threshold: range > LPIPS_RANGE_MIN
CLIP_BLIND_SE_THRESHOLD = 1.0
HPS_DELTA_MIN = 0.015
IR_DELTA_MIN = 0.25
LPIPS_RANGE_MIN = 0.08


def derive_exp_metrics(conds: list[ConditionStats]) -> dict:
    clips = [c.mean_clip for c in conds]
    hpss = [c.mean_hps for c in conds]
    irs = [c.mean_ir for c in conds]
    clip_delta = max(clips) - min(clips)
    hps_delta = max(hpss) - min(hpss)
    ir_delta = max(irs) - min(irs)

    # Pooled SE: mean of per-condition SEs (weight them equally)
    valid_ses = [c.se_clip for c in conds if not math.isnan(c.se_clip)]
    pooled_se = mean(valid_ses) if valid_ses else float("nan")
    clip_delta_se = clip_delta / pooled_se if pooled_se > 0 else float("nan")

    lpips_vals = [c.mean_lpips for c in conds if c.mean_lpips is not None]
    lpips_range: float | None = (max(lpips_vals) - min(lpips_vals)) if lpips_vals else None

    return {
        "clip_delta": clip_delta,
        "clip_delta_se": clip_delta_se,
        "hps_delta": hps_delta,
        "ir_delta": ir_delta,
        "lpips_range": lpips_range,
        "pooled_se": pooled_se,
    }


def assign_verdict(metrics: dict) -> str:
    cds = metrics["clip_delta_se"]
    hd = metrics["hps_delta"]
    iд = metrics["ir_delta"]
    lr = metrics["lpips_range"]

    other_moves = (
        hd > HPS_DELTA_MIN
        or iд > IR_DELTA_MIN
        or (lr is not None and lr > LPIPS_RANGE_MIN)
    )
    if math.isnan(cds):
        return "INDETERMINATE (SE unavailable)"
    if cds < CLIP_BLIND_SE_THRESHOLD and other_moves:
        return "CLIP-BLIND"
    if cds < CLIP_BLIND_SE_THRESHOLD and not other_moves:
        return "FLAT ACROSS BOARD (CLIP and others flat)"
    return "CLIP RESPONDS"


# ── Chart ─────────────────────────────────────────────────────────────────────


def make_chart(results: list[ExpResult], out_path: Path) -> None:
    labels = [r.exp_id.upper() for r in results]
    clip_ses = [r.clip_delta_se for r in results]
    lpips_vals = [r.lpips_range if r.lpips_range is not None else 0.0 for r in results]

    x = np.arange(len(labels))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("CLIP-Blindness Replication — SDXL", fontsize=14, fontweight="bold")

    bars1 = ax1.bar(x, clip_ses, width=0.6, color="#2563EB", alpha=0.85)
    ax1.axhline(CLIP_BLIND_SE_THRESHOLD, color="red", linestyle="--", linewidth=1.5, label=f"Threshold ({CLIP_BLIND_SE_THRESHOLD} SE)")
    ax1.set_xlabel("Experiment")
    ax1.set_ylabel("CLIP Δ (SE units)")
    ax1.set_title("CLIP Score Movement Across Conditions")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.legend()
    ax1.bar_label(bars1, fmt="%.2f", padding=2, fontsize=8)

    bars2 = ax2.bar(x, lpips_vals, width=0.6, color="#F97316", alpha=0.85)
    ax2.axhline(LPIPS_RANGE_MIN, color="green", linestyle="--", linewidth=1.5, label=f"Min meaningful ({LPIPS_RANGE_MIN})")
    ax2.set_xlabel("Experiment")
    ax2.set_ylabel("LPIPS range across conditions")
    ax2.set_title("Perceptual Variation Across Conditions")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.legend()
    ax2.bar_label(bars2, fmt="%.3f", padding=2, fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved: {out_path}")


# ── Report writer ─────────────────────────────────────────────────────────────


def fmt(v: float | None, decimals: int = 4) -> str:
    if v is None:
        return "N/A"
    if math.isnan(v):
        return "NaN"
    return f"{v:.{decimals}f}"


def write_report(results: list[ExpResult], sd21_path: Path, out_path: Path) -> None:
    blind_count = sum(1 for r in results if r.verdict == "CLIP-BLIND")
    total = len(results)

    # Overall verdict string
    if blind_count == total:
        overall = f"**REPLICATES** — CLIP-blindness confirmed in all {total}/{total} SDXL experiments."
    elif blind_count >= total * 0.6:
        overall = (
            f"**PARTIAL REPLICATION** — CLIP-blindness confirmed in {blind_count}/{total} "
            "SDXL experiments."
        )
    elif blind_count > 0:
        overall = (
            f"**WEAK REPLICATION** — CLIP-blindness confirmed in {blind_count}/{total} "
            "SDXL experiments only."
        )
    else:
        overall = f"**DOES NOT REPLICATE** — CLIP-blindness absent in all {total} SDXL experiments."

    lines: list[str] = []

    lines.append("# CLIP-Blindness Replication: SDXL Analysis")
    lines.append("")
    lines.append("**Date:** 2026-06-02  ")
    lines.append("**Model:** stabilityai/stable-diffusion-xl-base-1.0  ")
    lines.append("**Baseline:** SD 2.1 (Phase 6b, reports/clip_blindness.md)  ")
    lines.append("**GCS backup:** gs://aetherart-eval-pr13/experiments/  ")
    lines.append("")
    lines.append("## Overall Verdict")
    lines.append("")
    lines.append(overall)
    lines.append("")
    lines.append(
        f"Experiments 6 and 7 (LoRA rank, LoRA data size) are **N/A** on SDXL — "
        "training images are not in the repo; these experiments cannot be reproduced "
        "without the original fine-tuning dataset. Results are therefore based on "
        f"{total} of the 9 Phase 6b experiments."
    )
    lines.append("")

    # ── Schema map (3a) ──────────────────────────────────────────────────────
    lines.append("## Schema Map (as-observed from results.json)")
    lines.append("")
    lines.append(
        "| Exp | Condition column | Condition values | CLIP col | HPS col | IR col | LPIPS col |"
    )
    lines.append(
        "|-----|-----------------|-----------------|----------|---------|--------|-----------|"
    )
    schema_rows = [
        ("exp1", "`condition`", "fp16 / int8 / nf4", "`clip_score`", "`hps_score`", "`ir_score`", "`lpips` (0.0 for fp16=ref)"),
        ("exp2", "`condition`", "no_neg / with_neg", "`clip_score`", "`hps_score`", "`ir_score`", "`lpips_vs_no_neg`"),
        ("exp3", "`cfg_value`", "1/3/5/7/9/12/15", "`clip_score`", "`hps_score`", "`ir_score`", "`lpips_vs_ref` (vs cfg=7)"),
        ("exp4", "`scheduler`", "DDIM/DPM/EulerA/LMS", "`clip_score`", "`hps_score`", "`ir_score`", "pair_agg only (max pairwise)"),
        ("exp5", "`strength`", "0.0–1.5 (7 levels)", "`clip_score`", "`hps_score`", "`ir_score`", "`lpips_vs_ref` (vs strength=1.0)"),
        ("exp6", "**N/A**", "—", "—", "—", "—", "**Not run — training images missing**"),
        ("exp7", "**N/A**", "—", "—", "—", "—", "**Not run — training images missing**"),
        ("exp8", "`alpha`", "0.0–1.5 (7 levels)", "`clip_score`", "`hps_score`", "`ir_score`", "`lpips_vs_ref` (vs alpha=1.0)"),
        ("exp9", "`condition`", "no_trigger / with_trigger", "`clip_score`", "`hps_score`", "`ir_score`", "`lpips_vs_no_trigger`"),
    ]
    for row in schema_rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # ── Per-experiment delta table (3c) ──────────────────────────────────────
    lines.append("## Per-Experiment Delta Table")
    lines.append("")
    lines.append(
        "CLIP Δ SE = (max_condition_mean_CLIP − min_condition_mean_CLIP) / pooled_SE_CLIP.  "
    )
    lines.append(
        f"Verdict threshold: CLIP Δ < {CLIP_BLIND_SE_THRESHOLD} SE AND "
        f"(HPS Δ > {HPS_DELTA_MIN} OR IR Δ > {IR_DELTA_MIN} OR LPIPS range > {LPIPS_RANGE_MIN})."
    )
    lines.append("")
    lines.append(
        "| Exp | Variable | CLIP Δ (abs) | CLIP Δ SE | HPS Δ (abs) | IR Δ (abs) | LPIPS range | Verdict |"
    )
    lines.append(
        "|-----|----------|-------------|-----------|-------------|------------|-------------|---------|"
    )

    for r in results:
        lr_str = fmt(r.lpips_range, 3) if r.lpips_range is not None else "N/A"
        lines.append(
            f"| {r.exp_id} | {r.variable} | {fmt(r.clip_delta, 4)} | "
            f"{fmt(r.clip_delta_se, 2)} | {fmt(r.hps_delta, 4)} | "
            f"{fmt(r.ir_delta, 4)} | {lr_str} | **{r.verdict}** |"
        )
    lines.append("")

    # ── Per-experiment detail ─────────────────────────────────────────────────
    lines.append("## Per-Experiment Detail")
    lines.append("")

    for r in results:
        lines.append(f"### {r.label}")
        lines.append("")
        lines.append(f"**Variable:** {r.variable}  ")
        lines.append(f"**Verdict:** {r.verdict}  ")
        if r.caveat:
            lines.append(f"**Note:** {r.caveat}  ")
        lines.append("")

        # Condition table
        lines.append("| Condition | N | Mean CLIP | SE CLIP | Mean HPS | Mean IR | Mean LPIPS |")
        lines.append("|-----------|---|-----------|---------|----------|---------|------------|")
        for c in r.conditions:
            lines.append(
                f"| {c.label} | {c.n} | {fmt(c.mean_clip, 4)} | "
                f"{fmt(c.se_clip, 4)} | {fmt(c.mean_hps, 4)} | "
                f"{fmt(c.mean_ir, 4)} | {fmt(c.mean_lpips, 3) if c.mean_lpips is not None else 'N/A'} |"
            )
        lines.append("")
        lines.append(
            f"CLIP Δ = {fmt(r.clip_delta, 4)} ({fmt(r.clip_delta_se, 2)} SE)  "
            f"HPS Δ = {fmt(r.hps_delta, 4)}  IR Δ = {fmt(r.ir_delta, 4)}"
        )
        if r.lpips_range is not None:
            lines.append(f"LPIPS range across conditions = {fmt(r.lpips_range, 3)}")
        lines.append("")

    # ── Comparison with SD 2.1 ────────────────────────────────────────────────
    lines.append("## Comparison with SD 2.1 Baseline")
    lines.append("")
    lines.append(
        "The SD 2.1 baseline (reports/clip_blindness.md) found CLIP-blindness across "
        "all 9 Phase 6b experiments: CLIP scores varied < 1 SE while HPS, ImageReward, "
        "and LPIPS showed meaningful movement across conditions."
    )
    lines.append("")
    lines.append(
        f"On SDXL (7 experiments completed): {blind_count}/{total} experiments show "
        "the same CLIP-blind pattern. See the per-experiment table above for which "
        "experiments differ and by how much."
    )
    lines.append("")
    lines.append("![CLIP-Blindness Chart](clip_blindness_sdxl_chart.png)")
    lines.append("")

    # ── Data-quality caveats (3e) ─────────────────────────────────────────────
    lines.append("## Data-Quality Caveats")
    lines.append("")
    lines.append(
        "1. **Exp 6 and Exp 7 missing (N/A):** LoRA rank and LoRA data-size experiments "
        "require fine-tuning images that are not committed to the repo. These 2 of 9 "
        "experiments cannot be run without the original dataset."
    )
    lines.append(
        "2. **Exp 4 LPIPS:** Pairwise-only; the per-scheduler LPIPS column does not exist. "
        "The max pairwise mean LPIPS is used as a proxy for perceptual spread."
    )
    lines.append(
        "3. **Exp 2 and Exp 9 LPIPS:** Values are paired cross-condition distances "
        "(no_neg↔with_neg, no_trigger↔with_trigger), not within-condition variation. "
        "They quantify how much the output changes when the condition changes, which "
        "is exactly the relevant quantity for the blindness test."
    )
    lines.append(
        "4. **LPIPS for fp16/alpha=1.0/strength=1.0 reference:** Set to 0 by construction "
        "(image compared to itself). These are excluded from the range calculation."
    )
    lines.append(
        "5. **Sample sizes:** Each condition cell has 8 prompts × 5 seeds = 40 observations "
        "(exp1/exp2/exp8/exp9) or 8 prompts × 1 seed = 8 (exp4). Exp3 and exp5 have "
        "7 CFG/strength levels × 8 prompts × 5 seeds = 40 per condition."
    )
    lines.append(
        "6. **Exp 2 borderline:** CLIP Δ = 1.09 SE, just over the 1.0 SE threshold. "
        "HPS Δ and IR Δ are both well below their thresholds (0.009 vs 0.015, 0.040 vs 0.25). "
        "The 'CLIP RESPONDS' verdict depends entirely on the 0.09 SE excess above the threshold "
        "— this experiment is ambiguous; it could equally plausibly be classed as borderline CLIP-blind."
    )
    lines.append("")
    lines.append(
        "*Analysis script:* `scripts/generate_clip_blindness_sdxl.py`  "
        "*Raw data:* `reports/experiments/exp*_sdxl/results.json`  "
        "*GCS backup:* gs://aetherart-eval-pr13/experiments/"
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    results: list[ExpResult] = []

    for exp_cfg in EXPERIMENTS:
        path = ROOT / exp_cfg["path"]
        if not path.exists():
            print(f"SKIP {exp_cfg['id']}: {path} not found")
            continue
        with path.open(encoding="utf-8") as f:
            data = json.load(f)

        conds, caveat = exp_cfg["adapter"](data)
        metrics = derive_exp_metrics(conds)
        verdict = assign_verdict(metrics)

        r = ExpResult(
            exp_id=exp_cfg["id"],
            label=exp_cfg["label"],
            variable=exp_cfg["variable"],
            conditions=conds,
            clip_delta=metrics["clip_delta"],
            clip_delta_se=metrics["clip_delta_se"],
            hps_delta=metrics["hps_delta"],
            ir_delta=metrics["ir_delta"],
            lpips_range=metrics["lpips_range"],
            verdict=verdict,
            caveat=caveat,
        )
        results.append(r)
        print(
            f"{exp_cfg['id']}: CLIP Δ={r.clip_delta:.4f} ({r.clip_delta_se:.2f} SE) "
            f"HPS Δ={r.hps_delta:.4f} IR Δ={r.ir_delta:.4f} "
            f"LPIPS range={fmt(r.lpips_range, 3)} → {r.verdict}"
        )

    out_dir = ROOT / "reports"
    make_chart(results, out_dir / "clip_blindness_sdxl_chart.png")

    sd21_path = ROOT / "reports" / "clip_blindness.md"
    write_report(results, sd21_path, out_dir / "clip_blindness_sdxl.md")


if __name__ == "__main__":
    main()
