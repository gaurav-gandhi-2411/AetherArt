#!/usr/bin/env python
"""Validates scripts/detect_text_artifacts.py's EasyOCR-based binary artifact detector against
INDEPENDENT ground truth (direct visual inspection of each image, not derived from the VLM judge
or the detector itself) before it is trusted for anything. Per the task: if it doesn't correlate,
report that as a fast negative and stop.

Ground truth labels (`GROUND_TRUTH`, keyed by image_path) were assigned by directly viewing each
of the 29 stratified-sample images (reports/_ocr_validation_sample.json) - a genuinely independent
check, not a re-derivation of the VLM's own artifact_absence score.

Usage:
    python scripts/validate_text_detector.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Ground truth from direct visual inspection - True = image contains visible embedded text /
# calligraphy / cartouche / seal mark, however small or stylized. False = no such mark visible.
GROUND_TRUTH: dict[str, bool] = {
    "published_lora_016_42.png": True,
    "published_lora_012_42.png": True,
    "published_lora_028_44.png": True,
    "published_lora_006_43.png": True,
    "published_lora_002_43.png": True,
    "published_lora_016_44.png": True,
    "published_lora_003_42.png": False,
    "published_lora_015_44.png": True,
    "published_lora_026_42.png": False,
    "curated_lora_012_44.png": True,
    "curated_lora_001_42.png": True,
    "curated_lora_002_44.png": True,
    "curated_lora_009_44.png": True,
    "curated_lora_004_43.png": False,
    "curated_lora_020_43.png": False,
    "curated_lora_018_44.png": True,
    "curated_lora_027_44.png": False,
    "curated_lora_013_44.png": True,
    "curated_lora_025_44.png": True,
    "base_lora_025_42.png": False,
    "base_lora_003_43.png": True,
    "base_lora_004_43.png": False,
    "base_lora_022_44.png": True,
    "base_lora_006_44.png": True,
    "base_lora_001_44.png": True,
    "base_lora_006_43.png": True,
    "base_lora_005_43.png": True,
    "base_lora_030_44.png": True,
    "base_lora_030_42.png": True,
}


def main() -> None:
    detections = json.loads((ROOT / "reports" / "_ocr_validation_detections.json").read_text())
    assert len(detections) == len(GROUND_TRUTH) == 29

    tp = fp = tn = fn = 0
    gt_vals, conf_vals, ndet_vals, vlm_vals = [], [], [], []
    rows = []
    for r in detections:
        fname = Path(r["image_path"]).name
        gt = GROUND_TRUTH[fname]
        pred = r["has_detected_text"]
        if gt and pred:
            tp += 1
        elif gt and not pred:
            fn += 1
        elif not gt and pred:
            fp += 1
        else:
            tn += 1
        gt_vals.append(1 if gt else 0)
        conf_vals.append(r["max_confidence"])
        ndet_vals.append(r["n_detections"])
        vlm_vals.append(r["artifact_absence"])
        rows.append(
            (fname, gt, pred, r["max_confidence"], r["n_detections"], r["artifact_absence"])
        )

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")

    n_pos, n_neg = sum(gt_vals), len(gt_vals) - sum(gt_vals)
    print(f"n = {len(detections)}  (GT positive: {n_pos}, GT negative: {n_neg})")
    print(f"TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"Precision (at conf>=0.3 threshold): {precision:.3f}")
    print(f"Recall    (at conf>=0.3 threshold): {recall:.3f}")

    def pearson(xs, ys):
        mx, my = statistics.fmean(xs), statistics.fmean(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
        dx = (sum((x - mx) ** 2 for x in xs)) ** 0.5
        dy = (sum((y - my) ** 2 for y in ys)) ** 0.5
        return num / (dx * dy) if dx and dy else float("nan")

    r_conf_gt = pearson(conf_vals, gt_vals)
    r_ndet_gt = pearson(ndet_vals, gt_vals)
    # invert: higher artifact presence -> lower artifact_absence
    r_conf_vlm = pearson(conf_vals, [1 - v for v in vlm_vals])
    r_ndet_vlm = pearson(ndet_vals, [1 - v for v in vlm_vals])

    print(f"\nPearson r(max_confidence, ground_truth_has_text)  = {r_conf_gt:.3f}")
    print(f"Pearson r(n_detections,   ground_truth_has_text)  = {r_ndet_gt:.3f}")
    print(f"Pearson r(max_confidence, 1 - VLM artifact_absence) = {r_conf_vlm:.3f}")
    print(f"Pearson r(n_detections,   1 - VLM artifact_absence) = {r_ndet_vlm:.3f}")

    # Threshold sensitivity: recall/precision at a few candidate thresholds
    print("\nThreshold sensitivity (recomputed from raw confidences, not re-running OCR):")
    print(f"{'threshold':>10}{'precision':>11}{'recall':>9}{'n_flagged':>11}")
    for thresh in (0.5, 0.3, 0.2, 0.1, 0.05, 0.01):
        tp2 = fp2 = fn2 = 0
        for _fname, gt, _pred, conf, _n, _vlm in rows:
            pred2 = conf >= thresh
            if gt and pred2:
                tp2 += 1
            elif gt and not pred2:
                fn2 += 1
            elif not gt and pred2:
                fp2 += 1
        prec2 = tp2 / (tp2 + fp2) if (tp2 + fp2) else float("nan")
        rec2 = tp2 / (tp2 + fn2) if (tp2 + fn2) else float("nan")
        n_flagged = sum(1 for *_r, conf, _n, _v in rows if conf >= thresh)
        print(f"{thresh:>10.2f}{prec2:>11.3f}{rec2:>9.3f}{n_flagged:>11d}")

    out = {
        "n": len(detections),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision_at_0.3": round(precision, 4),
        "recall_at_0.3": round(recall, 4),
        "pearson_maxconf_vs_groundtruth": round(r_conf_gt, 4),
        "pearson_ndet_vs_groundtruth": round(r_ndet_gt, 4),
        "pearson_maxconf_vs_vlm_artifact": round(r_conf_vlm, 4),
        "pearson_ndet_vs_vlm_artifact": round(r_ndet_vlm, 4),
    }
    out_path = ROOT / "reports" / "text_detector_validation.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
