#!/usr/bin/env python
"""Guardrail for GCP instance teardown: verify a result file was actually copied to local and
is non-empty/non-trivial BEFORE deleting the instance that produced it.

Exists because of a real incident (2026-07-23, docs/MODEL_VERDICT.md's provenance note /
PLAN.md's "Gotchas" entry): a GCP VM was deleted to free capacity for a retry before its
completed results (90 generations + 90 VLM scores) were pulled to local. `gcloud compute
instances delete` removes the boot disk by default, so that data was gone and had to be
regenerated from scratch. An exit-code check on the `gcloud compute scp` call would not have
caught this class of mistake (the delete happened as a SEPARATE, later command, with no
automated link back to "did the copy actually land and look right") - what was missing was a
positive, structural check of the copied file's content before the point of no return.

Usage (from repo root, after `gcloud compute scp <instance>:<remote_path> <local_path> ...`):
    python scripts/gcp_verify_before_teardown.py <local_path> --min-records N
    python scripts/gcp_verify_before_teardown.py <local_path> --min-bytes N

For a JSON records file (list of dicts, e.g. reports/*.json from this project's harnesses):
    python scripts/gcp_verify_before_teardown.py reports/lora_ab_30prompt.json --min-records 180 --no-errors

Exits 0 (safe to delete the instance) only if every requested check passes. Exits 1 and prints
exactly what failed otherwise - never silently proceeds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("local_path", type=Path, help="Local file that was just scp'd down")
    ap.add_argument("--min-bytes", type=int, default=1,
                     help="Minimum file size in bytes (default: 1, i.e. just non-empty)")
    ap.add_argument("--min-records", type=int, default=None,
                     help="If set, parse local_path as JSON (a list) and require at least this many records")
    ap.add_argument("--no-errors", action="store_true",
                     help="If set with --min-records, also require every record's 'error' field to be falsy")
    args = ap.parse_args()

    failures: list[str] = []

    if not args.local_path.exists():
        print(f"FAIL: {args.local_path} does not exist locally — nothing was copied.")
        return 1

    size = args.local_path.stat().st_size
    if size < args.min_bytes:
        failures.append(f"file size {size} bytes < required minimum {args.min_bytes} bytes")

    if args.min_records is not None:
        try:
            data = json.loads(args.local_path.read_text(encoding="utf-8"))
        except Exception as e:
            failures.append(f"could not parse as JSON: {e}")
            data = None
        if data is not None:
            if not isinstance(data, list):
                failures.append(f"expected a JSON list, got {type(data).__name__}")
            else:
                n = len(data)
                if n < args.min_records:
                    failures.append(f"record count {n} < required minimum {args.min_records}")
                if args.no_errors:
                    n_err = sum(1 for r in data if isinstance(r, dict) and r.get("error"))
                    if n_err:
                        failures.append(f"{n_err} record(s) have a non-empty 'error' field")

    if failures:
        print("FAIL — do NOT delete the instance. Issues found:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print(f"PASS: {args.local_path} verified ({size} bytes"
          + (f", {args.min_records}+ records, 0 errors" if args.min_records is not None else "")
          + ") — safe to tear down the source instance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
