#!/usr/bin/env python
"""Pattachitra adapter-weight sweep stats - tests whether a lower `adapter_weights` scale
recovers `figure_preservation` while keeping a positive `style_adherence` lift over `sdxl_base`,
per docs/MODEL_VERDICT.md SS7.2(5)'s overtraining/recipe hypothesis.

Single source of truth: reports/pattachitra_uniform_rescore.json
(scripts/rescore_pattachitra_uniform.py) - every weight (0.3/0.5/0.7/1.0) and `base` are scored
in one uniform pass with the corrected PATTACHITRA_STYLE_QUESTION, so no number here is a mix of
the old (void) hardcoded-ukiyo-e scoring and new corrected scoring.

Same paired-diff/SEM/MDE methodology as scripts/compute_pattachitra_ab_stats.py - reused directly
(imported, not reimplemented) so both reports are computed identically.

An "operating point" is a (checkpoint, weight) where BOTH conditions hold at the same weight
(docs/WEIGHT_SWEEP_PREREGISTRATION.md's joint criterion, guarding the weight->0 tautology):
style_adherence diff/SEM > +2.0 (significant lift) AND figure_preservation diff/SEM >= -2.0
(non-inferior). If none exists at any tested weight, the regression is not a dosage/overtraining
artifact recoverable by lowering the adapter scale - and per the standing directive, no result
here reopens the publication decision either way (already settled by figure_preservation at
weight=1.0).

Usage:
    python scripts/compute_pattachitra_weight_sweep_stats.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
UNIFORM_RESCORE_JSON = ROOT / "reports" / "pattachitra_uniform_rescore.json"
OUT_JSON = ROOT / "reports" / "pattachitra_weight_sweep_stats.json"
CHECKPOINTS = ("curated500", "curated1000")
ALL_WEIGHTS = (0.3, 0.5, 0.7, 1.0)
AXES = ("style_adherence", "figure_preservation")


def _import_ab_stats():
    spec = importlib.util.spec_from_file_location(
        "compute_pattachitra_ab_stats", ROOT / "scripts" / "compute_pattachitra_ab_stats.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["compute_pattachitra_ab_stats"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ab_stats = _import_ab_stats()

    all_records = json.loads(UNIFORM_RESCORE_JSON.read_text(encoding="utf-8"))
    clean = [r for r in all_records if not r.get("error") and r.get("independent_calls")]

    base = {f"{r['prompt_id']}_{r['seed']}": r for r in clean if r["checkpoint"] == "base"}
    assert len(base) == 90, f"expected 90 base records, got {len(base)}"

    results = {}
    operating_points = []
    print("=== Pattachitra adapter-weight sweep vs. sdxl_base "
          "(n=90 paired each, where complete; uniform re-score, corrected style question) ===\n")

    for ckpt in CHECKPOINTS:
        results[ckpt] = {}
        for weight in ALL_WEIGHTS:
            arm = {
                f"{r['prompt_id']}_{r['seed']}": r for r in clean
                if r["checkpoint"] == ckpt and r.get("weight") == weight
            }
            if len(arm) < 90:
                print(f"{ckpt} @ weight={weight}: only {len(arm)}/90 scored - "
                      f"skipping (incomplete)\n")
                continue
            keys = sorted(set(base) & set(arm))
            assert len(keys) == 90, f"{ckpt}@{weight}: expected 90 matched pairs, got {len(keys)}"
            stats = ab_stats.paired_stats(base, arm, keys)
            results[ckpt][str(weight)] = stats

            fp = stats["figure_preservation"]
            sa = stats["style_adherence"]
            # Joint criterion per docs/WEIGHT_SWEEP_PREREGISTRATION.md, fixed BEFORE this script
            # was run against complete sweep data: recovering figure_preservation alone is
            # guaranteed by construction as weight -> 0 (the model converges to sdxl_base) and is
            # NOT evidence of a usable adapter on its own - both conditions must hold at the same
            # weight. Both thresholds reuse existing project thresholds, not new ones invented for
            # this sweep: +2xSEM is PATTACHITRA_AB_PREREGISTRATION.md's own primary-endpoint bar;
            # -2xSEM is the same figure_preservation guardrail already in force throughout SS7.
            sa_significant_positive = sa["diff_over_sem"] > 2.0
            fp_non_inferior = fp["diff_over_sem"] >= -2.0
            is_operating_point = fp_non_inferior and sa_significant_positive
            if is_operating_point:
                operating_points.append((ckpt, weight))

            print(f"--- {ckpt} @ weight={weight} ---")
            print(f"{'Axis':<20}{'base':>9}{'arm':>9}{'diff':>10}{'SEM':>9}{'diff/SEM':>10}{'MDE@80%':>10}")
            for axis in AXES:
                s = stats[axis]
                print(f"{axis:<20}{s['base_mean']:>9.4f}{s['arm_mean']:>9.4f}{s['diff']:>+10.4f}"
                      f"{s['sem']:>9.4f}{s['diff_over_sem']:>10.3f}{s['mde_80pct_power']:>10.4f}")
            print(f"  figure_preservation non-inferior (>=-2.0xSEM): {fp_non_inferior}, "
                  f"style_adherence significant positive (>+2.0xSEM): {sa_significant_positive} "
                  f"-> {'OPERATING POINT' if is_operating_point else 'not viable'}\n")

    print("=== Joint curve (both axes together, per docs/WEIGHT_SWEEP_PREREGISTRATION.md - "
          "never reported in isolation). fp MDE columns are required alongside every "
          "non-inferiority claim per the same pre-registration - the >=-2xSEM screen is a "
          "'failed to detect regression' test, not demonstrated non-inferiority. ===\n")
    for ckpt in CHECKPOINTS:
        if not results.get(ckpt):
            continue
        print(f"--- {ckpt} ---")
        print(f"{'weight':>8}{'sa diff/SEM':>13}{'fp diff/SEM':>13}"
              f"{'fp MDE@80%':>12}{'fp MDE@90%':>12}")
        for weight_str in sorted(results[ckpt], key=float):
            stats = results[ckpt][weight_str]
            fp = stats["figure_preservation"]
            print(f"{weight_str:>8}{stats['style_adherence']['diff_over_sem']:>13.3f}"
                  f"{fp['diff_over_sem']:>13.3f}{fp['mde_80pct_power']:>12.4f}"
                  f"{fp['mde_90pct_power']:>12.4f}")
        print()

    print("=== Verdict ===")
    if operating_points:
        print(f"Operating point(s) found: {operating_points}")
        print("The adapter is over-applied/overtrained at weight=1.0, not intrinsically broken - a")
        print("lower adapter weight recovers figure_preservation without losing the style signal.")
    else:
        print("No (checkpoint, weight) combination is both figure_preservation-non-inferior and")
        print("style_adherence-positive. The regression does not resolve by lowering adapter scale")
        print("- not a dosage/overtraining artifact recoverable this way.")

    OUT_JSON.write_text(json.dumps({
        "n_pairs": 90, "per_checkpoint_weight": results, "operating_points": operating_points,
    }, indent=2), encoding="utf-8")
    print(f"\nWritten: {OUT_JSON}")


if __name__ == "__main__":
    main()
