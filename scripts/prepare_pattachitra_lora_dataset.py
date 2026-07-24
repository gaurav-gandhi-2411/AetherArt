#!/usr/bin/env python
"""Dataset preparation for the Pattachitra LoRA training run, per
docs/PATTACHITRA_AB_PREREGISTRATION.md - curated corpus ONLY (no uncurated arm is trained, per
that document's amendment).

Source : the 111 images the VLM curation filter kept (reports/pattachitra_classifications.json,
         flagged=False), already sitting locally at data/lora/pattachitra-precheck/images/.
Target : data/lora/pattachitra-curated/{images/,metadata.jsonl}, 1024x1024 (SDXL native res,
         matching ukiyo-e's curated SDXL retrain, not the 512x512 SD2.1 recipe).

Usage:
    python scripts/prepare_pattachitra_lora_dataset.py            # full run (BLIP captions)
    python scripts/prepare_pattachitra_lora_dataset.py --no-blip  # template captions (fast)
    python scripts/prepare_pattachitra_lora_dataset.py --dry-run  # count only, no I/O
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

TRIGGER_TOKEN = "pattascroll"
SRC_IMAGES_DIR = Path("data/lora/pattachitra-precheck/images")
CLASSIFICATIONS_PATH = Path("reports/pattachitra_classifications.json")
OUT_DIR = Path("data/lora/pattachitra-curated")
OUTPUT_SIZE = (1024, 1024)
BLIP_MODEL_ID = "Salesforce/blip-image-captioning-large"

# Manual QA pass (2026-07-24, before committing GCP training spend): the automated VLM curation
# filter's "clean" verdict was spot-checked by direct visual inspection before trusting it for
# training. 11 of the 111 "clean" images were found, on inspection, to be documentary/vendor
# photographs where a person's face or torso dominates a substantial fraction of the frame (the
# curation prompt's own stated exclusion criterion - filenames containing "artist at work",
# "stall", "book fair" turned out to be a strong predictor the automated filter missed), or a
# different visual genre entirely (painted spherical objects, not flat scroll paintings). This is
# NOT a re-run of the automated filter - it is a targeted, disclosed manual exclusion on top of it,
# proportionate to real GCP spend being committed on this corpus. See
# docs/NEXT_MODEL_SPEC.md for the writeup.
MANUAL_EXCLUDE: set[str] = {
    "Artist with Odisha Pattachitra DSCN1052 01.jpg",
    "Patachitra stall.jpg",
    "Patachitra artist at work.jpg",
    "Patachitra artists work with immense care and passion.jpg",
    "Pattachitra artist at work in Odisha, India.jpg",
    "Patua - International Kolkata Book Fair 2013 - Milan Mela Complex - Kolkata 2013-02-03 4284.JPG",
    "Patua - International Kolkata Book Fair 2013 - Milan Mela Complex - Kolkata 2013-02-03 4285.JPG",
    "Patua - International Kolkata Book Fair 2013 - Milan Mela Complex - Kolkata 2013-02-03 4286.JPG",
    "Patua - International Kolkata Book Fair 2013 - Milan Mela Complex - Kolkata 2013-02-03 4291.JPG",
    "Raghurajpur Artist.JPG",
    "The Paintings on Betel nut (Areca nut) by the artists of Raghurajpur.JPG",
}

_blip_pipe = None


def _load_blip():
    global _blip_pipe
    if _blip_pipe is None:
        import torch
        from transformers import pipeline as hf_pipeline

        device = 0 if torch.cuda.is_available() else -1
        print(f"Loading BLIP ({BLIP_MODEL_ID}) on {'GPU' if device == 0 else 'CPU'}...", flush=True)
        _blip_pipe = hf_pipeline("image-to-text", model=BLIP_MODEL_ID, device=device)
        print("BLIP ready.", flush=True)
    return _blip_pipe


def caption_blip(img: Image.Image) -> str:
    result = _load_blip()(img, max_new_tokens=60)
    raw = result[0]["generated_text"].strip() if result else ""
    base = raw if raw else "traditional Odisha folk painting"
    return f"{TRIGGER_TOKEN}, Pattachitra scroll painting, {base}"


def caption_template() -> str:
    return f"{TRIGGER_TOKEN}, Pattachitra scroll painting, traditional Odisha folk art"


def center_crop_resize(img: Image.Image, size: tuple = OUTPUT_SIZE) -> Image.Image:
    w, h = img.size
    short = min(w, h)
    left, top = (w - short) // 2, (h - short) // 2
    return img.crop((left, top, left + short, top + short)).resize(size, Image.LANCZOS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-blip", action="store_true")
    args = ap.parse_args()

    use_blip = not args.no_blip and not args.dry_run

    classifications = json.loads(CLASSIFICATIONS_PATH.read_text(encoding="utf-8"))
    clean_filenames = [c["file_name"] for c in classifications if not c["flagged"]]
    assert len(clean_filenames) == 111, f"expected 111 clean images, got {len(clean_filenames)}"
    clean_filenames = [f for f in clean_filenames if f not in MANUAL_EXCLUDE]
    print(f"Clean per automated pre-check: 111; after manual QA exclusion of "
          f"{len(MANUAL_EXCLUDE)} documentary/genre-mismatch photos: {len(clean_filenames)}", flush=True)
    assert len(clean_filenames) == 100, f"expected 100 images after manual exclusion, got {len(clean_filenames)}"

    img_dir = OUT_DIR / "images"
    if not args.dry_run:
        img_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    missing = []
    for i, fname in enumerate(sorted(clean_filenames)):
        src_path = SRC_IMAGES_DIR / fname
        if not src_path.exists():
            missing.append(fname)
            continue
        idx = len(records) + 1
        out_filename = f"{idx:03d}.jpg"
        try:
            img = Image.open(src_path).convert("RGB")
        except Exception as e:
            print(f"  decode error on {fname.encode('ascii', 'replace').decode('ascii')}: {e}", flush=True)
            missing.append(fname)
            continue

        if not args.dry_run:
            processed = center_crop_resize(img)
            processed.save(img_dir / out_filename, "JPEG", quality=90)
            caption = caption_blip(img) if use_blip else caption_template()
        else:
            caption = caption_template()

        records.append({"file_name": f"images/{out_filename}", "text": caption})
        if len(records) % 10 == 0 or len(records) == 1:
            print(f"  [{len(records):3d}/{len(clean_filenames)}] "
                  f"{fname.encode('ascii', 'replace').decode('ascii')} -> {out_filename}", flush=True)

    print(f"\nProcessed {len(records)} images, {len(missing)} missing/failed.", flush=True)
    if missing:
        print("Missing files:", [m.encode("ascii", "replace").decode("ascii") for m in missing])

    if args.dry_run:
        print("\nDry-run: no files written.")
        for r in records[:5]:
            print(f"  {r['file_name']}  ->  {r['text']}")
        return

    jsonl_path = OUT_DIR / "metadata.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    saved = sorted(img_dir.glob("*.jpg"))
    total_mb = sum(p.stat().st_size for p in saved) / 1024**2
    print(f"\n-- Dataset saved: {OUT_DIR.resolve()} --")
    print(f"  Images        : {len(saved)}")
    print(f"  Disk size     : {total_mb:.1f} MB")
    print(f"  metadata.jsonl: {jsonl_path}")
    print("\n-- Sample captions (first 5) --")
    for rec in records[:5]:
        print(f"  {rec['file_name']}  ->  {rec['text']}")


if __name__ == "__main__":
    main()
