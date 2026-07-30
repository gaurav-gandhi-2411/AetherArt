#!/usr/bin/env python
"""Compare FLUX.1-schnell's canonical 30-prompt x 3-seed eval against the 5 existing base
families (docs/MODEL_VERDICT.md SS2). Independent-samples (NOT paired) comparison: the 5
baselines' raw per-record JSON files are not resident on this machine (generated in an earlier
session, not committed — reports/*.json is a local-only artifact), so only their published
mean+/-SEM (n=90 each, itself independently re-verified per SS2's own note) is available. Using
that means the diff/SEM below is computed via independent-sample SEM quadrature
(sqrt(SEM_flux^2 + SEM_base^2)), not this project's stronger paired-diff design — stated
explicitly in the output, not silently presented as if it were paired.

Usage:
    python scripts/compute_flux_schnell_comparison.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).parent.parent
FLUX_JSON = ROOT / "reports" / "verdict_flux_schnell.json"

Z_ALPHA_2 = 1.96
Z_BETA_80 = 0.8416
Z_BETA_90 = 1.2816

# docs/MODEL_VERDICT.md SS2 — mean +/- SEM, n=90 each, independently re-verified there.
BASELINES = {
    "sd21_base": {"clip_mean": 0.3167, "clip_sem": 0.0036, "hps_mean": 0.2528, "hps_sem": 0.0042},
    "sdxl_base": {"clip_mean": 0.3280, "clip_sem": 0.0034, "hps_mean": 0.2876, "hps_sem": 0.0034},
    "hyper_4step": {
        "clip_mean": 0.3269,
        "clip_sem": 0.0037,
        "hps_mean": 0.3138,
        "hps_sem": 0.0034,
    },
    "hyper_8step": {
        "clip_mean": 0.3136,
        "clip_sem": 0.0042,
        "hps_mean": 0.2369,
        "hps_sem": 0.0044,
    },
    "sdxl_controlnet_union": {
        "clip_mean": 0.3281,
        "clip_sem": 0.0031,
        "hps_mean": 0.2802,
        "hps_sem": 0.0033,
    },
}


def mean_sem(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = statistics.fmean(values)
    sem = statistics.stdev(values) / (n**0.5) if n > 1 else float("nan")
    return mean, sem


def main() -> None:
    records = json.loads(FLUX_JSON.read_text())
    n_total = len(records)
    errors = [r for r in records if r.get("error")]
    clean = [r for r in records if not r.get("error")]

    clip_vals = [r["clip_score"] for r in clean if r.get("clip_score") is not None]
    hps_vals = [r["hps_score"] for r in clean if r.get("hps_score") is not None]
    lat_vals = [r["latency_s"] for r in clean if r.get("latency_s") is not None]

    clip_mean, clip_sem = mean_sem(clip_vals)
    hps_mean, hps_sem = mean_sem(hps_vals)

    print(f"flux_schnell: n_total={n_total} n_errors={len(errors)} n_clean={len(clean)}")
    print(f"  CLIP:  n={len(clip_vals)} mean={clip_mean:.4f} SEM={clip_sem:.4f}")
    print(f"  HPS:   n={len(hps_vals)} mean={hps_mean:.4f} SEM={hps_sem:.4f}")
    if lat_vals:
        print(f"  Latency: mean={statistics.fmean(lat_vals):.1f}s, n={len(lat_vals)}")
    print()
    print(
        "Independent-samples comparison vs. docs/MODEL_VERDICT.md SS2's 5 baselines "
        "(NOT paired -- baselines' raw per-record files aren't resident locally):"
    )
    header = (
        f"{'Baseline':<24} {'Axis':<6} {'flux mean':>10} {'base mean':>10} "
        f"{'diff':>8} {'SEM_diff':>9} {'diff/SEM':>9} {'MDE@80%':>9} {'MDE@90%':>9}"
    )
    print(header)
    for name, b in BASELINES.items():
        for axis, f_mean, f_sem, b_mean, b_sem in (
            ("CLIP", clip_mean, clip_sem, b["clip_mean"], b["clip_sem"]),
            ("HPS", hps_mean, hps_sem, b["hps_mean"], b["hps_sem"]),
        ):
            diff = f_mean - b_mean
            sem_diff = (f_sem**2 + b_sem**2) ** 0.5
            ratio = diff / sem_diff if sem_diff else float("inf")
            mde_80 = (Z_ALPHA_2 + Z_BETA_80) * sem_diff
            mde_90 = (Z_ALPHA_2 + Z_BETA_90) * sem_diff
            print(
                f"{name:<24} {axis:<6} {f_mean:>10.4f} {b_mean:>10.4f} "
                f"{diff:>8.4f} {sem_diff:>9.4f} {ratio:>9.3f} {mde_80:>9.4f} {mde_90:>9.4f}"
            )

    out = {
        "flux_schnell": {
            "n_total": n_total,
            "n_errors": len(errors),
            "clip_mean": clip_mean,
            "clip_sem": clip_sem,
            "hps_mean": hps_mean,
            "hps_sem": hps_sem,
            "latency_mean_s": statistics.fmean(lat_vals) if lat_vals else None,
        },
        "comparisons": [],
    }
    for name, b in BASELINES.items():
        for axis, f_mean, f_sem, b_mean, b_sem in (
            ("clip", clip_mean, clip_sem, b["clip_mean"], b["clip_sem"]),
            ("hps", hps_mean, hps_sem, b["hps_mean"], b["hps_sem"]),
        ):
            diff = f_mean - b_mean
            sem_diff = (f_sem**2 + b_sem**2) ** 0.5
            out["comparisons"].append(
                {
                    "baseline": name,
                    "axis": axis,
                    "flux_mean": f_mean,
                    "baseline_mean": b_mean,
                    "diff": diff,
                    "sem_diff": sem_diff,
                    "diff_over_sem": diff / sem_diff if sem_diff else None,
                    "mde_80": (Z_ALPHA_2 + Z_BETA_80) * sem_diff,
                    "mde_90": (Z_ALPHA_2 + Z_BETA_90) * sem_diff,
                }
            )
    out_path = ROOT / "reports" / "flux_schnell_comparison.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
