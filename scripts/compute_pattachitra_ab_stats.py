#!/usr/bin/env python
"""Paired-diff stats + power/MDE for the Pattachitra curated-LoRA-vs-sdxl_base evaluation, per
docs/PATTACHITRA_AB_PREREGISTRATION.md's amended design. Same methodology as
scripts/compute_lora_ab_power.py - paired diff/SEM computed directly on the 90 per-pair
differences, MDE at 80%/90% power reported alongside every endpoint regardless of significance.

Scores ALL THREE checkpoints (500/1000/1500) vs. base, per the pre-registration's "checkpoint-
select at steps 500/1000/1500, as with ukiyo-e" - ukiyo-e's own precedent explicitly REJECTED its
checkpoint-1500 for figure dropout/mode collapse and selected checkpoint-1000
(docs/lab_notebook.md). Checkpoint selection rule (stated here, applied mechanically, not tuned
after seeing results): among checkpoints where figure_preservation does NOT regress vs. base by
more than 2xSEM, select the one with the best (highest) style_adherence diff/SEM. If ALL
checkpoints regress figure_preservation by >2xSEM, no checkpoint clears the bar and that is
reported as the finding - not grounds to relax the guardrail.

Usage:
    python scripts/compute_pattachitra_ab_stats.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).parent.parent
SOURCE_JSON = ROOT / "reports" / "pattachitra_ab_base_comparison.json"
AXES = ("style_adherence", "figure_preservation", "artifact_absence")
CHECKPOINTS = ("curated500", "curated1000", "curated1500")

Z_ALPHA_2 = 1.96
Z_BETA_80 = 0.8416
Z_BETA_90 = 1.2816


def paired_stats(base: dict, arm: dict, keys: list[str]) -> dict:
    out = {}
    for axis in AXES:
        b = [base[k]["independent_calls"][axis] for k in keys]
        c = [arm[k]["independent_calls"][axis] for k in keys]
        diffs = [ci - bi for ci, bi in zip(c, b, strict=True)]
        mean_b, mean_c = statistics.fmean(b), statistics.fmean(c)
        diff_mean = statistics.fmean(diffs)
        sem = statistics.stdev(diffs) / (len(diffs) ** 0.5) if statistics.stdev(diffs) else 0.0
        ratio = diff_mean / sem if sem else float("inf") if diff_mean != 0 else 0.0
        ci_lo, ci_hi = diff_mean - Z_ALPHA_2 * sem, diff_mean + Z_ALPHA_2 * sem
        mde_80 = (Z_ALPHA_2 + Z_BETA_80) * sem
        mde_90 = (Z_ALPHA_2 + Z_BETA_90) * sem
        out[axis] = {
            "base_mean": round(mean_b, 4),
            "arm_mean": round(mean_c, 4),
            "diff": round(diff_mean, 4),
            "sem": round(sem, 4),
            "diff_over_sem": round(ratio, 3),
            "ci95_lo": round(ci_lo, 4),
            "ci95_hi": round(ci_hi, 4),
            "mde_80pct_power": round(mde_80, 4),
            "mde_90pct_power": round(mde_90, 4),
        }
    return out


def main() -> None:
    records = json.loads(SOURCE_JSON.read_text())
    base = {f"{r['prompt_id']}_{r['seed']}": r for r in records if r["checkpoint"] == "base"}
    assert len(base) == 90, f"expected 90 base records, got {len(base)}"

    all_results = {}
    print("=== Per-checkpoint results vs. sdxl_base (n=90 paired each) ===\n")
    for ckpt in CHECKPOINTS:
        arm = {f"{r['prompt_id']}_{r['seed']}": r for r in records if r["checkpoint"] == ckpt}
        if len(arm) < 90:
            print(f"{ckpt}: only {len(arm)}/90 scored - skipping (not yet complete)\n")
            continue
        keys = sorted(set(base) & set(arm))
        assert len(keys) == 90, f"{ckpt}: expected 90 matched pairs, got {len(keys)}"
        stats = paired_stats(base, arm, keys)
        all_results[ckpt] = stats

        print(f"--- {ckpt} ---")
        print(
            f"{'Axis':<22}{'base':>9}{'arm':>9}{'diff':>10}{'SEM':>9}{'diff/SEM':>10}{'MDE@80%':>10}"
        )
        for axis in AXES:
            s = stats[axis]
            print(
                f"{axis:<22}{s['base_mean']:>9.4f}{s['arm_mean']:>9.4f}{s['diff']:>+10.4f}"
                f"{s['sem']:>9.4f}{s['diff_over_sem']:>10.3f}{s['mde_80pct_power']:>10.4f}"
            )
        fp = stats["figure_preservation"]
        regressed = fp["diff"] < 0 and fp["diff_over_sem"] < -2.0
        print(f"  figure_preservation guardrail: {'REGRESSED' if regressed else 'no regression'}\n")

    if not all_results:
        print("No checkpoints fully scored yet.")
        return

    print("=== Checkpoint selection (rule fixed before seeing results, applied mechanically) ===")
    print("Rule: among checkpoints where figure_preservation does not regress >2xSEM vs. base,")
    print("select the one with the best (highest) style_adherence diff/SEM.\n")

    eligible = []
    for ckpt, stats in all_results.items():
        fp = stats["figure_preservation"]
        regressed = fp["diff"] < 0 and fp["diff_over_sem"] < -2.0
        status = "REGRESSED (excluded)" if regressed else "eligible"
        print(f"  {ckpt}: figure_preservation diff/SEM = {fp['diff_over_sem']:+.3f} -> {status}")
        if not regressed:
            eligible.append(ckpt)

    if not eligible:
        print("\n*** NO checkpoint clears the figure_preservation guardrail. ***")
        print("This is the finding, not grounds to relax the guardrail or pick the least-bad one.")
        selected = None
    else:
        selected = max(eligible, key=lambda c: all_results[c]["style_adherence"]["diff_over_sem"])
        selected_diff_over_sem = all_results[selected]["style_adherence"]["diff_over_sem"]
        print(
            f"\nSelected checkpoint: {selected} "
            f"(style_adherence diff/SEM = {selected_diff_over_sem:+.3f} "
            f"among eligible checkpoints)"
        )

    out_path = ROOT / "reports" / "pattachitra_ab_stats.json"
    out_path.write_text(
        json.dumps(
            {
                "n_pairs": 90,
                "per_checkpoint": all_results,
                "eligible_checkpoints": eligible,
                "selected_checkpoint": selected,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
