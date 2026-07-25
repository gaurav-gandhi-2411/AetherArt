#!/usr/bin/env python
"""Compute the paired-diff effect/SEM for the LoRA A/B under the TRUSTED independent-axis
scoring regime (reports/lora_ab_30prompt_independent.json), replicating the exact paired-diff
methodology docs/MODEL_VERDICT.md SS4.3 used for the original single-call-regime headline: diff =
curated - published per matching prompt_id+seed pair, SEM computed on the n per-pair differences
directly (not independent-sample quadrature) via sample stdev / sqrt(n).

This is the provenance script for the model card's primary claim (rule 65b) - re-run it against
the raw JSON any time the headline number is quoted, don't recompute by hand.

Usage:
    python scripts/compute_lora_ab_independent_stats.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).parent.parent
SOURCE_JSON = ROOT / "reports" / "lora_ab_30prompt_independent.json"
AXES = ("style_adherence", "figure_preservation", "artifact_absence")


def main() -> None:
    records = json.loads(SOURCE_JSON.read_text())
    assert len(records) == 180, f"expected 180 records (90 published + 90 curated), got {len(records)}"
    assert all(not r.get("error") for r in records), "unresolved errors in the independent-regime dataset"
    assert all(r.get("independent_calls") is not None for r in records), (
        "not all records have independent_calls scored yet - re-run "
        "scripts/_lora_ab_30prompt_independent.py for both checkpoints first"
    )

    by_key: dict[str, dict[str, dict]] = {}
    for r in records:
        key = f"{r['prompt_id']}_{r['seed']}"
        by_key.setdefault(key, {})[r["checkpoint"]] = r

    pairs = [v for v in by_key.values() if "published" in v and "curated" in v]
    assert len(pairs) == 90, f"expected 90 matched pairs, got {len(pairs)}"

    print(f"n = {len(pairs)} paired prompt+seed matches\n")
    print(f"{'Metric':<22} {'Published (mean)':>17} {'Curated (mean)':>15} "
          f"{'Paired diff':>12} {'Paired SEM':>11} {'diff/SEM':>9}")

    results = {}
    for axis in AXES:
        pub_scores = [p["published"]["independent_calls"][axis] for p in pairs]
        cur_scores = [p["curated"]["independent_calls"][axis] for p in pairs]
        diffs = [c - p for c, p in zip(cur_scores, pub_scores)]

        pub_mean = statistics.fmean(pub_scores)
        cur_mean = statistics.fmean(cur_scores)
        diff_mean = statistics.fmean(diffs)
        diff_sem = statistics.stdev(diffs) / (len(diffs) ** 0.5)
        ratio = diff_mean / diff_sem if diff_sem else float("inf")

        results[axis] = {
            "published_mean": round(pub_mean, 4),
            "curated_mean": round(cur_mean, 4),
            "paired_diff": round(diff_mean, 4),
            "paired_sem": round(diff_sem, 4),
            "diff_over_sem": round(ratio, 3),
        }
        print(f"{axis:<22} {pub_mean:>17.4f} {cur_mean:>15.4f} "
              f"{diff_mean:>+12.4f} {diff_sem:>11.4f} {ratio:>9.3f}")

    out_path = ROOT / "reports" / "lora_ab_independent_stats.json"
    out_path.write_text(json.dumps({
        "n_pairs": len(pairs),
        "scoring_regime": "independent single-axis (one Ollama call per axis)",
        "source": str(SOURCE_JSON.relative_to(ROOT)),
        "results": results,
    }, indent=2), encoding="utf-8")
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
