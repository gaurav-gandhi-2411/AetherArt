#!/usr/bin/env python
"""Binary text/calligraphy-artifact detector for GENERATED Ukiyo-e LoRA outputs, using EasyOCR.

This is a DIFFERENT task from scripts/curate_ukiyo_e_dataset.py's EasyOCR attempt, which was
discarded there for false-positiving on real WikiArt woodblock-print TRAINING images (English-only
reader misreading brushwork texture as Latin glyphs). Per the task instruction, that failure does
not imply EasyOCR fails here too - it must be independently validated on THIS image population
(SDXL-generated LoRA outputs, not scanned woodblock prints) before being trusted for anything.

Uses both English and Japanese readers (['en', 'ja']) since the artifact this LoRA learned could
plausibly render as either script - unlike the training-curation attempt, which used English only
and is documented as a specific, avoidable mistake.

Binary decision rule: an image "has_detected_text" if EasyOCR returns at least one detection with
confidence >= CONF_THRESHOLD. Raw detections (all boxes + confidences, not just the binary verdict)
are saved per image so the threshold can be re-examined without re-running OCR.

Usage:
    python scripts/detect_text_artifacts.py --manifest reports/_ocr_validation_sample.json
    python scripts/detect_text_artifacts.py --dir outputs/verdict/lora_ab_30prompt_independent \
        --out reports/text_artifact_detections_arms.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CONF_THRESHOLD = 0.3


def build_reader():
    import easyocr

    return easyocr.Reader(["en", "ja"], gpu=True)


def detect(reader, image_path: str) -> dict:
    results = reader.readtext(image_path)
    detections = [
        {"text": text, "confidence": round(float(conf), 4)} for (_bbox, text, conf) in results
    ]
    has_text = any(d["confidence"] >= CONF_THRESHOLD for d in detections)
    max_conf = max((d["confidence"] for d in detections), default=0.0)
    return {
        "has_detected_text": has_text,
        "n_detections": len(detections),
        "max_confidence": max_conf,
        "detections": detections,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", help="JSON list of {image_path: ...} records to score")
    ap.add_argument("--dir", help="Directory of images to score (all files)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.manifest:
        items = json.loads(Path(args.manifest).read_text())
        image_paths = [it["image_path"] for it in items]
    elif args.dir:
        image_paths = [str(p) for p in sorted(Path(args.dir).glob("*.png"))]
        items = [{"image_path": p} for p in image_paths]
    else:
        raise SystemExit("must pass --manifest or --dir")

    reader = build_reader()

    out_path = Path(args.out)
    results = []
    if out_path.exists():
        try:
            results = json.loads(out_path.read_text())
        except Exception:
            results = []
    done = {r["image_path"] for r in results}

    for i, (item, img_path) in enumerate(zip(items, image_paths, strict=True)):
        if img_path in done:
            continue
        det = detect(reader, img_path)
        record = {**item, "image_path": img_path, **det}
        results.append(record)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(results, indent=2), encoding="utf-8")
        tmp.replace(out_path)
        print(
            f"[{i + 1}/{len(image_paths)}] {Path(img_path).name}: "
            f"has_text={det['has_detected_text']} n_det={det['n_detections']} "
            f"max_conf={det['max_confidence']:.3f}",
            flush=True,
        )

    print(f"\nDone. {len(results)} total records. Written: {out_path}")


if __name__ == "__main__":
    main()
