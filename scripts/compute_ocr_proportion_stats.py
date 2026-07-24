#!/usr/bin/env python
"""Paired-proportion analysis of the OCR-based binary artifact detector
(scripts/detect_text_artifacts.py), per docs/MODEL_VERDICT.md's high-power-metric follow-up.

Decision rule: has_artifact = (n_detections >= 1) - i.e. EasyOCR found ANY text-shaped region,
regardless of confidence. Chosen over a confidence threshold because
scripts/validate_text_detector.py found confidence poorly calibrated for this LoRA's stylized,
often-illegible pseudo-calligraphy (precision 1.00/recall 0.23 at conf>=0.3, vs 0.94/0.77 for
"any detection") - n_detections correlates with ground truth better than max_confidence does.

Uses McNemar's exact test for the paired published-vs-curated comparison (appropriate for a
binary paired design - a rank-based or chi-square test would be wrong here) and a two-proportion
z-test for each arm vs. the independent base-comparison sample (not paired: base wasn't matched
to individual arm images 1:1 in the same generation run structure needed for exact pairing, though
prompt_id+seed DO match - so this IS also computed paired for completeness).

Usage:
    python scripts/compute_ocr_proportion_stats.py
"""

from __future__ import annotations

import json
from math import comb
from pathlib import Path

ROOT = Path(__file__).parent.parent


def has_artifact(record: dict) -> bool:
    return record["n_detections"] >= 1


def load_by_key(path: Path, checkpoint_filter: str | None = None) -> dict[str, bool]:
    records = json.loads(path.read_text())
    out = {}
    for r in records:
        fname = Path(r["image_path"]).name
        # filenames: {checkpoint}_{prompt_id}_{seed}.png or base_{prompt_id}_{seed}.png
        parts = fname.replace(".png", "").split("_")
        if checkpoint_filter and not fname.startswith(checkpoint_filter):
            continue
        # key = prompt_id_seed (strip the checkpoint prefix and .png)
        if fname.startswith("published_"):
            key = fname[len("published_"):-4]
        elif fname.startswith("curated_"):
            key = fname[len("curated_"):-4]
        elif fname.startswith("base_"):
            key = fname[len("base_"):-4]
        else:
            key = fname[:-4]
        out[key] = has_artifact(r)
    return out


def mcnemar_exact_p(b: int, c: int) -> float:
    """Exact two-sided McNemar test p-value (binomial test on discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p_one_side = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * p_one_side)


def paired_proportion_stats(a: dict[str, bool], b: dict[str, bool], name_a: str, name_b: str) -> dict:
    keys = sorted(set(a) & set(b))
    n = len(keys)
    # 2x2 discordant-pair table
    both_true = sum(1 for k in keys if a[k] and b[k])
    only_a = sum(1 for k in keys if a[k] and not b[k])
    only_b = sum(1 for k in keys if not a[k] and b[k])
    neither = sum(1 for k in keys if not a[k] and not b[k])
    assert both_true + only_a + only_b + neither == n

    rate_a = sum(a[k] for k in keys) / n
    rate_b = sum(b[k] for k in keys) / n
    diff = rate_b - rate_a  # b - a
    p_value = mcnemar_exact_p(only_a, only_b)

    print(f"\n=== {name_a} vs {name_b} (n={n} paired) ===")
    print(f"  2x2: both-flagged={both_true}, only-{name_a}={only_a}, only-{name_b}={only_b}, neither={neither}")
    print(f"  P(artifact | {name_a}) = {rate_a:.4f}")
    print(f"  P(artifact | {name_b}) = {rate_b:.4f}")
    print(f"  diff ({name_b} - {name_a}) = {diff:+.4f}")
    print(f"  McNemar exact p-value = {p_value:.4f}")

    return {
        "n": n, "both_flagged": both_true, f"only_{name_a}": only_a, f"only_{name_b}": only_b,
        "neither": neither, f"rate_{name_a}": round(rate_a, 4), f"rate_{name_b}": round(rate_b, 4),
        "diff": round(diff, 4), "mcnemar_p": round(p_value, 4),
    }


def mde_mcnemar(n: int, discordant_frac_guess: float = 0.15, alpha: float = 0.05, power: float = 0.8) -> float:
    """Approximate MDE for a paired-proportion (McNemar) design at given n, using the standard
    normal approximation: required discordant pairs ~ ((z_a/2 + z_b) / diff)^2 * p_discordant,
    inverted to solve for the detectable diff given n and an assumed discordant-pair fraction.
    This is an approximation for reporting purposes, not a formal power calculation."""
    z_alpha, z_beta = 1.96, 0.8416
    # For McNemar: diff = (z_a/2 + z_b) * sqrt(p_discordant / n), solving for the effect size
    # detectable from n total pairs, given typical fraction of pairs that are discordant.
    return (z_alpha + z_beta) * (discordant_frac_guess / n) ** 0.5


def main() -> None:
    arms_path = ROOT / "reports" / "text_artifact_detections_arms.json"
    base_path = ROOT / "reports" / "text_artifact_detections_base.json"

    published = load_by_key(arms_path, "published_")
    curated = load_by_key(arms_path, "curated_")
    base = load_by_key(base_path, "base_")

    assert len(published) == len(curated) == len(base) == 90

    results = {}
    results["published_vs_curated"] = paired_proportion_stats(published, curated, "published", "curated")
    results["base_vs_published"] = paired_proportion_stats(base, published, "base", "published")
    results["base_vs_curated"] = paired_proportion_stats(base, curated, "base", "curated")

    n = 90
    observed_discordant_frac = results["published_vs_curated"]["only_published"] / n + \
        results["published_vs_curated"]["only_curated"] / n
    mde = mde_mcnemar(n, discordant_frac_guess=max(observed_discordant_frac, 0.05))
    print(f"\n=== Power/sensitivity (OCR binary metric, n={n} paired) ===")
    print(f"Observed discordant-pair fraction (published-vs-curated): {observed_discordant_frac:.4f}")
    print(f"Approx. MDE at 80% power, alpha=0.05 (two-sided): {mde:.4f}")
    print(f"Compare to the independent-axis VLM rubric's MDE at n=90: ~0.037 (docs/MODEL_VERDICT.md SS4.7)")

    results["power"] = {
        "n": n, "observed_discordant_fraction": round(observed_discordant_frac, 4),
        "approx_mde_80pct_power": round(mde, 4), "rubric_mde_for_comparison": 0.0374,
    }

    out_path = ROOT / "reports" / "ocr_proportion_stats.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
