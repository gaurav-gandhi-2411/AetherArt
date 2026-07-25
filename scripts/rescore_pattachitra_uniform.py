#!/usr/bin/env python
"""Uniform re-score of every Pattachitra weight/checkpoint comparison image on all three axes,
using the corrected PATTACHITRA_STYLE_QUESTION, in one consistent pass - per
docs/WEIGHT_SWEEP_PREREGISTRATION.md's "uniform re-score" plan and the standing directive's Q3.

Scope (all from already-generated, already-persisted images - no new SDXL generation):
  - base (n=90): reports/pattachitra_ab_base_comparison.json, checkpoint == "base".
  - curated500 / curated1000 @ weight=1.0 (n=90 each): same file, checkpoint == the arm name.
    Images and figure_preservation/artifact_absence were already CUDA-corruption-audited
    (docs/MODEL_VERDICT.md SS7.2(4)) - only style_adherence was ever wrong for these.
  - curated500 / curated1000 @ weights 0.3/0.5/0.7 (n=90 each):
    reports/pattachitra_weight_sweep.json. These have never been scored on any axis (inline
    scoring was disabled before it reached them).

"Uniform" means literal: every one of these 810 images is scored fresh, on all three axes, in
this one pass - including axes (figure_preservation, artifact_absence) that were already valid
at weight=1.0/base. This is deliberately not an optimization to reuse those valid prior scores:
the point is that every number feeding the joint-criterion stats comes from one consistent
method/run, not a patchwork of old-clean and new-fresh scoring passes. The cost is redundant
Ollama calls for the already-valid axes; the benefit is zero ambiguity about which regime any
given number came from.

Output: reports/pattachitra_uniform_rescore.json (a new file - the original
pattachitra_ab_base_comparison.json and pattachitra_weight_sweep.json are left untouched as the
historical/provenance record of what was actually run and when).

Usage:
    python scripts/rescore_pattachitra_uniform.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from aetherart.logger import get_logger

logger = get_logger(__name__)

ROOT = Path(__file__).parent.parent
BASE_COMPARISON_JSON = ROOT / "reports" / "pattachitra_ab_base_comparison.json"
WEIGHT_SWEEP_JSON = ROOT / "reports" / "pattachitra_weight_sweep.json"
OUT_JSON = ROOT / "reports" / "pattachitra_uniform_rescore.json"

PATTACHITRA_STYLE_QUESTION = (
    "does this image look like an authentic Pattachitra painting (traditional Odisha folk art) - "
    "flat colour, dense ornamental border, circular iconographic composition?"
)


def _import_harness():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "model_verdict_harness", ROOT / "scripts" / "model_verdict_harness.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["model_verdict_harness"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_partial() -> list[dict]:
    if OUT_JSON.exists():
        try:
            return json.loads(OUT_JSON.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def save_partial(results: list[dict]) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    tmp.replace(OUT_JSON)


def build_source() -> list[dict]:
    """Every (checkpoint, weight, prompt_id, seed) this uniform pass must score, with the image
    path to score it from. Weight is always recorded, including 1.0 and "base" (base has no
    adapter, weight is not a meaningful axis for it but is set to None for schema uniformity)."""
    base_records = json.loads(BASE_COMPARISON_JSON.read_text(encoding="utf-8"))
    sweep_records = json.loads(WEIGHT_SWEEP_JSON.read_text(encoding="utf-8"))
    # curated1500 is NOT part of the weight sweep (only 500/1000 were ever swept) - excluded here
    # so it isn't wastefully re-scored as part of this diagnostic's scope.
    in_scope_checkpoints = {"base", "curated500", "curated1000"}

    source = []
    for r in base_records:
        if r.get("error") or r["checkpoint"] not in in_scope_checkpoints:
            continue
        weight = None if r["checkpoint"] == "base" else 1.0
        source.append(
            {
                "checkpoint": r["checkpoint"],
                "weight": weight,
                "prompt_id": r["prompt_id"],
                "prompt": r["prompt"],
                "seed": r["seed"],
                "image_path": r["image_path"],
            }
        )
    for r in sweep_records:
        if r.get("error"):
            continue
        source.append(
            {
                "checkpoint": r["checkpoint"],
                "weight": r["weight"],
                "prompt_id": r["prompt_id"],
                "prompt": r["prompt"],
                "seed": r["seed"],
                "image_path": r["image_path"],
            }
        )
    return source


def main() -> None:
    harness = _import_harness()

    source = build_source()
    logger.info("[rescore] %d images in scope for uniform re-score.", len(source))

    results = load_partial()
    done = {
        f"{r['checkpoint']}_{r['weight']}_{r['prompt_id']}_{r['seed']}"
        for r in results
        if not r.get("error")
    }
    stale_error_keys = {
        f"{r['checkpoint']}_{r['weight']}_{r['prompt_id']}_{r['seed']}"
        for r in results
        if r.get("error")
    }
    if stale_error_keys:
        logger.info(
            "[rescore] Dropping %d stale errored record(s) for retry.", len(stale_error_keys)
        )
    results = [r for r in results if not r.get("error")]

    to_process = [
        src
        for src in source
        if f"{src['checkpoint']}_{src['weight']}_{src['prompt_id']}_{src['seed']}" not in done
    ]
    logger.info(
        "[rescore] %d already done, %d remaining this run.",
        len(source) - len(to_process),
        len(to_process),
    )

    for n_done, src in enumerate(to_process, start=1):
        key = f"{src['checkpoint']}_{src['weight']}_{src['prompt_id']}_{src['seed']}"
        record = {
            "checkpoint": src["checkpoint"],
            "weight": src["weight"],
            "prompt_id": src["prompt_id"],
            "prompt": src["prompt"],
            "seed": src["seed"],
            "image_path": src["image_path"],
            "independent_calls": None,
            "error": None,
        }
        try:
            img = Image.open(src["image_path"])
            record["independent_calls"] = harness.score_vlm_judge(
                img, src["prompt"], style_question=PATTACHITRA_STYLE_QUESTION
            )
            if record["independent_calls"] is None:
                record["error"] = "score_vlm_judge returned None (a judge call failed)"
        except Exception as e:
            logger.exception("[rescore] Error on %s", key)
            record["error"] = str(e)
        results.append(record)
        save_partial(results)
        if n_done % 10 == 0 or n_done == len(to_process):
            logger.info(
                "[rescore] [%d/%d] %s: %s",
                n_done,
                len(to_process),
                key,
                record["independent_calls"],
            )

    logger.info("[rescore] Done. %d total records.", len(results))
    harness.assert_unique_records(results, ("checkpoint", "weight", "prompt_id", "seed"))
    n_errors = sum(1 for r in results if r.get("error"))
    logger.info("[rescore] %d errors out of %d records.", n_errors, len(results))


if __name__ == "__main__":
    main()
