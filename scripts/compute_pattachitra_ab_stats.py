#!/usr/bin/env python
"""Paired-diff stats + power/MDE for the Pattachitra curated-LoRA-vs-sdxl_base evaluation, per
docs/PATTACHITRA_AB_PREREGISTRATION.md's amended design. Same methodology as
scripts/compute_lora_ab_power.py (the ukiyo-e SS4.7/SS4.8 template this pre-registration commits
to following) - paired diff/SEM computed directly on the 90 per-pair differences, MDE at 80%/90%
power reported alongside every endpoint regardless of significance.

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

Z_ALPHA_2 = 1.96
Z_BETA_80 = 0.8416
Z_BETA_90 = 1.2816


def main() -> None:
    records = json.loads(SOURCE_JSON.read_text())
    base = {f"{r['prompt_id']}_{r['seed']}": r for r in records if r["checkpoint"] == "base"}
    curated = {f"{r['prompt_id']}_{r['seed']}": r for r in records if r["checkpoint"] == "curated"}
    assert len(base) == len(curated) == 90, f"expected 90/90, got {len(base)}/{len(curated)}"
    keys = sorted(set(base) & set(curated))
    assert len(keys) == 90

    print(f"n = {len(keys)} paired observations (curated vs. sdxl_base)\n")
    print(f"{'Axis':<22}{'base mean':>11}{'curated mean':>14}{'diff':>10}{'SEM':>9}{'diff/SEM':>10}"
          f"{'MDE@80%':>10}{'MDE@90%':>10}")

    out = {}
    for axis in AXES:
        b = [base[k]["independent_calls"][axis] for k in keys]
        c = [curated[k]["independent_calls"][axis] for k in keys]
        diffs = [ci - bi for ci, bi in zip(c, b)]
        mean_b, mean_c = statistics.fmean(b), statistics.fmean(c)
        diff_mean = statistics.fmean(diffs)
        sem = statistics.stdev(diffs) / (len(diffs) ** 0.5)
        ratio = diff_mean / sem if sem else float("inf")
        ci_lo, ci_hi = diff_mean - Z_ALPHA_2 * sem, diff_mean + Z_ALPHA_2 * sem
        mde_80 = (Z_ALPHA_2 + Z_BETA_80) * sem
        mde_90 = (Z_ALPHA_2 + Z_BETA_90) * sem

        out[axis] = {
            "base_mean": round(mean_b, 4), "curated_mean": round(mean_c, 4),
            "diff": round(diff_mean, 4), "sem": round(sem, 4), "diff_over_sem": round(ratio, 3),
            "ci95_lo": round(ci_lo, 4), "ci95_hi": round(ci_hi, 4),
            "mde_80pct_power": round(mde_80, 4), "mde_90pct_power": round(mde_90, 4),
            "clears_2sem": abs(ratio) > 2.0,
        }
        print(f"{axis:<22}{mean_b:>11.4f}{mean_c:>14.4f}{diff_mean:>+10.4f}{sem:>9.4f}"
              f"{ratio:>10.3f}{mde_80:>10.4f}{mde_90:>10.4f}")

    print("\n=== Per-endpoint characterization (per the pre-registered decision rule) ===")
    sa = out["style_adherence"]
    print(f"Primary A (style_adherence, want > +2xSEM): "
          f"{'CLEARS the bar - demonstrated style lift' if sa['diff'] > 0 and sa['diff_over_sem'] > 2.0 else 'DOES NOT clear the bar'} "
          f"(diff={sa['diff']:+.4f}, {sa['diff_over_sem']:.3f}xSEM, 95% CI [{sa['ci95_lo']:+.4f}, {sa['ci95_hi']:+.4f}], "
          f"MDE@80%={sa['mde_80pct_power']:.4f})")

    aa = out["artifact_absence"]
    direction = "regresses" if aa["diff"] < 0 else "improves"
    significant = abs(aa["diff_over_sem"]) > 2.0
    print(f"Primary B (artifact_absence, characterized not gated): "
          f"{direction} vs. base by {aa['diff']:+.4f} "
          f"({'a significant' if significant else 'a non-significant'} {abs(aa['diff_over_sem']):.3f}xSEM effect, "
          f"95% CI [{aa['ci95_lo']:+.4f}, {aa['ci95_hi']:+.4f}], MDE@80%={aa['mde_80pct_power']:.4f})")

    fp = out["figure_preservation"]
    regressed = fp["diff"] < 0 and fp["diff_over_sem"] < -2.0
    print(f"Guardrail (figure_preservation, must not regress >2xSEM): "
          f"{'REGRESSED' if regressed else 'no regression'} "
          f"(diff={fp['diff']:+.4f}, {fp['diff_over_sem']:.3f}xSEM)")

    out_path = ROOT / "reports" / "pattachitra_ab_stats.json"
    out_path.write_text(json.dumps({"n_pairs": 90, "results": out}, indent=2), encoding="utf-8")
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
