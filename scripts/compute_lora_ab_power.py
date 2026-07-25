#!/usr/bin/env python
"""Power/sensitivity audit for the LoRA A/B primary endpoint (task 3 of the null-result audit,
docs/MODEL_VERDICT.md SS4.6): compute the minimum detectable effect (MDE) at n=90 paired
under independent-axis scoring, from the observed per-pair variance, and the 95% CI on the true
paired diff. Distinguishes "underpowered" (can't rule out a real small-to-moderate effect) from
"demonstrated null" (data actively rules out the originally-claimed effect size).

MDE formula (paired design, approx-normal): MDE = (z_{alpha/2} + z_beta) * SEM_diff, using the
OBSERVED per-pair SEM (not a pre-registered assumption) since this is a post-hoc sensitivity
check, not a pre-registered power analysis.

Usage:
    python scripts/compute_lora_ab_power.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).parent.parent
SOURCE_JSON = ROOT / "reports" / "lora_ab_30prompt_independent.json"
AXES = ("style_adherence", "figure_preservation", "artifact_absence")

Z_ALPHA_2 = 1.96  # two-sided alpha=0.05
Z_BETA_80 = 0.8416  # 80% power
Z_BETA_90 = 1.2816  # 90% power


def main() -> None:
    records = json.loads(SOURCE_JSON.read_text())
    by_key: dict[str, dict] = {}
    for r in records:
        key = f"{r['prompt_id']}_{r['seed']}"
        by_key.setdefault(key, {})[r["checkpoint"]] = r
    pairs = [v for v in by_key.values() if "published" in v and "curated" in v]
    assert len(pairs) == 90

    print(f"n = {len(pairs)} paired observations\n")
    print(f"{'Metric':<22} {'Observed diff':>14} {'SEM':>8} {'diff/SEM':>9} "
          f"{'95% CI':>24} {'MDE@80%':>9} {'MDE@90%':>9}")

    out = {}
    for axis in AXES:
        diffs = [p["curated"]["independent_calls"][axis] - p["published"]["independent_calls"][axis]
                 for p in pairs]
        n = len(diffs)
        mean_diff = statistics.fmean(diffs)
        sem = statistics.stdev(diffs) / (n ** 0.5)
        ratio = mean_diff / sem if sem else float("inf")
        ci_lo = mean_diff - Z_ALPHA_2 * sem
        ci_hi = mean_diff + Z_ALPHA_2 * sem
        mde_80 = (Z_ALPHA_2 + Z_BETA_80) * sem
        mde_90 = (Z_ALPHA_2 + Z_BETA_90) * sem

        out[axis] = {
            "observed_diff": round(mean_diff, 4), "sem": round(sem, 4),
            "diff_over_sem": round(ratio, 3),
            "ci95_lo": round(ci_lo, 4), "ci95_hi": round(ci_hi, 4),
            "mde_80pct_power": round(mde_80, 4), "mde_90pct_power": round(mde_90, 4),
        }
        print(f"{axis:<22} {mean_diff:>+14.4f} {sem:>8.4f} {ratio:>9.3f} "
              f"[{ci_lo:>+.4f}, {ci_hi:>+.4f}]      {mde_80:>9.4f} {mde_90:>9.4f}")

    print()
    primary = out["artifact_absence"]
    original_claim = 0.0400
    print("=== Interpretation: artifact_absence (PRIMARY) ===")
    print(f"Observed diff: {primary['observed_diff']:+.4f}, SEM: {primary['sem']:.4f}, "
          f"diff/SEM: {primary['diff_over_sem']:.3f}")
    print(f"95% CI on true diff: [{primary['ci95_lo']:+.4f}, {primary['ci95_hi']:+.4f}]")
    print(f"MDE at 80% power (this n, this observed variance): {primary['mde_80pct_power']:.4f}")
    print(f"MDE at 90% power: {primary['mde_90pct_power']:.4f}")
    print(f"Originally-claimed (correlated-regime) effect size: +{original_claim:.4f}")

    ci_excludes_original_claim = primary["ci95_hi"] < original_claim
    verdict = (
        "RULES OUT the originally-claimed effect size (95% CI upper bound is below it)"
        if ci_excludes_original_claim else
        "does NOT rule out the originally-claimed effect size (falls within the 95% CI) - "
        "the design is UNDERPOWERED for an effect of that size, not a clean demonstrated null"
    )
    print(f"\nVerdict: this n=90 independent-axis result {verdict}.")

    smaller_than_mde = abs(primary["observed_diff"]) < primary["mde_80pct_power"]
    print(
        f"The observed effect ({primary['observed_diff']:+.4f}) is "
        f"{'SMALLER' if smaller_than_mde else 'LARGER'} than this design's own 80%-power MDE "
        f"({primary['mde_80pct_power']:.4f}) - this design cannot reliably distinguish "
        f"'no effect' from 'a true effect smaller than the MDE' at the observed magnitude."
    )

    out_path = ROOT / "reports" / "lora_ab_power_audit.json"
    out_path.write_text(json.dumps({
        "n_pairs": 90,
        "z_alpha_2_two_sided_005": Z_ALPHA_2,
        "z_beta_80pct_power": Z_BETA_80,
        "z_beta_90pct_power": Z_BETA_90,
        "original_correlated_regime_claim": original_claim,
        "results": out,
    }, indent=2), encoding="utf-8")
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
