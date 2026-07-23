#!/usr/bin/env python
"""CI quality gate: assert a fresh eval run's CLIP score hasn't regressed vs. the frozen
SD 2.1 30-prompt / seed-42 baseline.

The tolerance band is NOT a hardcoded/invented number — it is computed at run time from the
baseline file's own 30 individual per-prompt CLIP scores (mean and standard error of the mean),
so the threshold always traces to real measured data (reports/eval_results_20260425_124153.json)
rather than a guess. See PLAN.md PR 14 / docs/MODEL_AUDIT.md §4 for the audit finding this closes.

Usage:
    python scripts/check_eval_gate.py --candidate reports/eval_results_<run_id>.json
    python scripts/check_eval_gate.py --candidate <path> --baseline <path> \
        --scheduler DPM --steps 30 --n-sigma 2
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DEFAULT_BASELINE = ROOT / "reports" / "eval_results_20260425_124153.json"


def _cell_clip_scores(results: list[dict], scheduler: str, steps: int) -> list[float]:
    return [
        r["clip_score"]
        for r in results
        if r.get("scheduler") == scheduler and r.get("steps") == steps and not r.get("error")
    ]


def compute_baseline(path: Path, scheduler: str, steps: int) -> tuple[float, float, int]:
    """Return (mean, standard_error_of_mean, n) for one scheduler/steps cell of a baseline file.

    SEM is derived from the sample standard deviation of the cell's own CLIP scores
    (statistics.stdev, i.e. the n-1 unbiased estimator) divided by sqrt(n) — not invented.
    """
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    scores = _cell_clip_scores(data["results"], scheduler, steps)
    if len(scores) < 2:
        raise ValueError(
            f"Baseline cell scheduler={scheduler} steps={steps} in {path} has {len(scores)} "
            "usable scores (need >= 2 to compute a standard error)."
        )
    mean = statistics.mean(scores)
    sem = statistics.stdev(scores) / (len(scores) ** 0.5)
    return mean, sem, len(scores)


def compute_candidate_mean(path: Path, scheduler: str, steps: int) -> tuple[float, int]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    scores = _cell_clip_scores(data["results"], scheduler, steps)
    if not scores:
        raise ValueError(
            f"Candidate file {path} has no usable scheduler={scheduler} steps={steps} scores."
        )
    return statistics.mean(scores), len(scores)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--candidate", required=True, type=Path, help="Fresh eval_results_*.json to check"
    )
    p.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE, help="Frozen baseline JSON")
    p.add_argument("--scheduler", default="DPM")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument(
        "--n-sigma",
        type=float,
        default=2.0,
        help=(
            "Regression threshold = baseline_mean - n_sigma * baseline_SEM. n_sigma=2 is a "
            "standard one-sided ~97.5%% confidence bound under a normal approximation (n=30 "
            "is large enough that Student's t ~ z here); NOT tuned to make the gate pass."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    baseline_mean, baseline_sem, baseline_n = compute_baseline(
        args.baseline, args.scheduler, args.steps
    )
    candidate_mean, candidate_n = compute_candidate_mean(args.candidate, args.scheduler, args.steps)

    threshold = baseline_mean - args.n_sigma * baseline_sem
    passed = candidate_mean >= threshold

    print("=== CI Quality Gate: CLIP score regression check ===")
    print(f"Baseline:  {args.baseline} ({args.scheduler}/{args.steps}steps, n={baseline_n})")
    print(f"  mean = {baseline_mean:.6f}   SEM = {baseline_sem:.6f}")
    print(f"Candidate: {args.candidate} ({args.scheduler}/{args.steps}steps, n={candidate_n})")
    print(f"  mean = {candidate_mean:.6f}")
    print(f"Threshold = baseline_mean - {args.n_sigma}*SEM = {threshold:.6f}")
    print()
    if passed:
        print(f"PASS: candidate mean {candidate_mean:.6f} >= threshold {threshold:.6f}")
    else:
        print(f"FAIL: candidate mean {candidate_mean:.6f} < threshold {threshold:.6f}")
        print(
            f"Regression of {threshold - candidate_mean:.6f} "
            f"({(threshold - candidate_mean) / baseline_sem:.2f} SEM below the threshold)."
        )

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
