#!/usr/bin/env python
"""Retroactive scan for VLM judge scores outside [0, 1], non-numeric, or NaN.

scripts/model_verdict_harness.py's `validate_judge_scores` range guard (added after the
Pattachitra CUDA-corruption audit, docs/MODEL_VERDICT.md SS7.2(4)) is new - every verdict/A-B
report on disk was written before it existed. This scans every reports/*.json file that carries
a judge-score dict (any of "vlm_judge", "independent_calls", "original_single_call") for values
that guard would have rejected, so a silent bad value already baked into a published mean can be
caught and corrected rather than assumed clean by construction.

Usage:
    python scripts/scan_verdict_judge_ranges.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).parent.parent
REPORTS_DIR = ROOT / "reports"
JUDGE_FIELDS = ("vlm_judge", "independent_calls", "original_single_call")
AXES = ("style_adherence", "figure_preservation", "artifact_absence")


def scan_scores_dict(scores: dict) -> list[str]:
    issues = []
    for axis, value in scores.items():
        if axis not in AXES:
            issues.append(f"unknown_axis={axis!r}")
            continue
        if not isinstance(value, int | float) or isinstance(value, bool):
            issues.append(f"{axis}=non_numeric({value!r})")
        elif math.isnan(value):
            issues.append(f"{axis}=nan")
        elif not 0.0 <= value <= 1.0:
            issues.append(f"{axis}=out_of_range({value})")
    return issues


def scan_file(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []
    violations = []
    for i, record in enumerate(data):
        if not isinstance(record, dict):
            continue
        for field in JUDGE_FIELDS:
            scores = record.get(field)
            if not isinstance(scores, dict):
                continue
            issues = scan_scores_dict(scores)
            if issues:
                violations.append(
                    {
                        "file": path.name,
                        "index": i,
                        "field": field,
                        "key": f"{record.get('checkpoint', record.get('family', ''))}"
                        f"_{record.get('prompt_id')}_{record.get('seed')}",
                        "issues": issues,
                    }
                )
    return violations


def main() -> None:
    all_violations = []
    scanned = 0
    for path in sorted(REPORTS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not (isinstance(data, list) and data and isinstance(data[0], dict)):
            continue
        has_judge_field = any(
            isinstance(r, dict) and any(f in r for f in JUDGE_FIELDS) for r in data
        )
        if not has_judge_field:
            continue
        scanned += 1
        violations = scan_file(path)
        status = f"{len(violations)} violation(s)" if violations else "clean"
        print(f"{path.name}: {len(data)} records - {status}")
        all_violations.extend(violations)

    print(f"\nScanned {scanned} files with judge-score fields.")
    if all_violations:
        print(f"\n*** {len(all_violations)} TOTAL VIOLATIONS ***")
        for v in all_violations:
            print(f"  {v['file']} [{v['index']}] {v['field']} {v['key']}: {v['issues']}")
    else:
        print("\nNo out-of-range, non-numeric, or NaN judge scores found in any file.")


if __name__ == "__main__":
    main()
