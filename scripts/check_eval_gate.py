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
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DEFAULT_BASELINE = ROOT / "reports" / "eval_results_20260425_124153.json"

# Content hash of the exact 30 DPM/30steps prompt entries reports/eval_results_20260425_124153.json
# (this repo's frozen baseline run) was computed on -- pinned separately from the mean/SEM
# threshold below because --num-images 30 only pins the COUNT of prompts a candidate run uses,
# not their IDENTITY. If scripts/eval_prompts.yaml is reordered or its first 30 entries are
# edited, a candidate run would keep producing 30 CLIP scores and the gate would keep passing
# green while silently comparing against a different prompt set than the one mean=0.319903 /
# SEM=0.005917 (see compute_baseline) was measured on -- the same defect class as
# docs/paper's inconsistent-reference-arms finding, just in eval data instead of paper prose.
# Computed 2026-07-31 via: hashlib.sha256(json.dumps(records, sort_keys=True,
# separators=(",", ":")).encode()).hexdigest() over each result's
# (prompt_id, prompt, category, expected_difficulty), IN RESULT ORDER (not sorted) -- a pure
# reorder of the same 30 IDs must still change this hash, since it's evidence
# eval_prompts.yaml was edited even if the measured mean would come out numerically identical.
#
# DO NOT update this constant to silence a mismatch. A mismatch means the candidate no longer
# measures the same 30 prompts the mean/SEM above were computed from -- the fix is to recompute
# a fresh baseline (new mean, new SEM, new hash from that run), not to re-pin this hash alone.
BASELINE_PROMPT_SET_SHA256 = "fb821f7bb3e9308b2511f71162bf762ae31c6b3f05c8f5034f29b79be526aed4"


def _cell_clip_scores(results: list[dict], scheduler: str, steps: int) -> list[float]:
    return [
        r["clip_score"]
        for r in results
        if r.get("scheduler") == scheduler and r.get("steps") == steps and not r.get("error")
    ]


def _cell_prompt_identity_records(results: list[dict], scheduler: str, steps: int) -> list[dict]:
    """Order-preserving (prompt_id, prompt, category, expected_difficulty) records for one
    scheduler/steps cell -- the exact content the CLIP mean was computed from. List order is
    intentionally NOT sorted: a reorder of eval_prompts.yaml's early entries must change the
    resulting hash even when the same 30 IDs end up included."""
    return [
        {
            "prompt_id": r.get("prompt_id"),
            "prompt": r.get("prompt"),
            "category": r.get("category"),
            "expected_difficulty": r.get("expected_difficulty"),
        }
        for r in results
        if r.get("scheduler") == scheduler and r.get("steps") == steps and not r.get("error")
    ]


def compute_prompt_set_hash(results: list[dict], scheduler: str, steps: int) -> str:
    records = _cell_prompt_identity_records(results, scheduler, steps)
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_candidate_prompt_set_hash(path: Path, scheduler: str, steps: int) -> str:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return compute_prompt_set_hash(data["results"], scheduler, steps)


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

    print("=== CI Quality Gate: CLIP score regression check ===")

    candidate_prompt_hash = compute_candidate_prompt_set_hash(
        args.candidate, args.scheduler, args.steps
    )
    if candidate_prompt_hash != BASELINE_PROMPT_SET_SHA256:
        print("FAIL: baseline prompt-set identity mismatch.")
        print(f"  pinned SHA256    (from {DEFAULT_BASELINE.name}): {BASELINE_PROMPT_SET_SHA256}")
        print(f"  candidate SHA256 ({args.scheduler}/{args.steps}steps): {candidate_prompt_hash}")
        print()
        print(
            f"The candidate's {args.scheduler}/{args.steps}steps prompt content "
            "(prompt_id + prompt + category + expected_difficulty, in order) no longer matches "
            "the 30 prompts the baseline mean/SEM were computed from. scripts/eval_prompts.yaml "
            "has been reordered or its early entries edited since the baseline was frozen -- the "
            "candidate and baseline are no longer measuring the same prompt set, so a mean/SEM "
            "comparison would be meaningless and is not performed.\n"
            "DO NOT edit BASELINE_PROMPT_SET_SHA256 to make this pass. If the prompt-set change "
            "is intentional, run a fresh baseline against the new prompt set and recompute the "
            "mean, SEM, AND this hash from that run -- see the comment above "
            "BASELINE_PROMPT_SET_SHA256 in this file for exactly how it's derived."
        )
        sys.exit(1)

    print(f"PASS: baseline prompt-set identity check ({candidate_prompt_hash[:16]}...)")

    baseline_mean, baseline_sem, baseline_n = compute_baseline(
        args.baseline, args.scheduler, args.steps
    )
    candidate_mean, candidate_n = compute_candidate_mean(args.candidate, args.scheduler, args.steps)

    threshold = baseline_mean - args.n_sigma * baseline_sem
    passed = candidate_mean >= threshold

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
