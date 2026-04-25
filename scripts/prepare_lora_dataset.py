#!/usr/bin/env python
"""
Stage 1 dataset preparation for Ukiyo-e LoRA fine-tuning.

Source  : huggan/wikiart parquet cache (style=26 = Ukiyo_e)
Target  : 50-80 images at 512x512 RGB + metadata.jsonl
Output  : data/lora/ukiyo-e/
           ├-- images/001.jpg … 080.jpg
           └-- metadata.jsonl

Fast path reads cached .parquet files directly via pyarrow instead of
re-downloading through the datasets streaming API.

Usage
-----
  python scripts/prepare_lora_dataset.py            # full run (BLIP captions)
  python scripts/prepare_lora_dataset.py --no-blip  # template captions (fast)
  python scripts/prepare_lora_dataset.py --dry-run  # count only, no I/O
"""
import argparse
import glob
import io
import json
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

# -- constants ----------------------------------------------------------------

UKIYO_E_STYLE_IDX = 26
HF_CACHE_GLOB     = (
    "C:/Users/gaura/.cache/huggingface/hub/"
    "datasets--huggan--wikiart/snapshots/*/data/train-*.parquet"
)
TRIGGER_TOKEN     = "ukyowood"
TARGET_MAX        = 80
MIN_SHORTEST_EDGE = 512
MIN_ASPECT        = 0.7
MAX_ASPECT        = 1.4
OUTPUT_SIZE       = (512, 512)
OUT_DIR           = Path("data/lora/ukiyo-e")
BLIP_MODEL_ID     = "Salesforce/blip-image-captioning-large"

# wikiart artist label names (index = artist int)
ARTIST_NAMES: list[str] = []  # populated on first use

# -- image helpers ------------------------------------------------------------─

def qualifies(img: Image.Image) -> bool:
    w, h = img.size
    if min(w, h) < MIN_SHORTEST_EDGE:
        return False
    ratio = w / h
    return MIN_ASPECT < ratio < MAX_ASPECT


def center_crop_resize(img: Image.Image, size: tuple = OUTPUT_SIZE) -> Image.Image:
    w, h = img.size
    short = min(w, h)
    left, top = (w - short) // 2, (h - short) // 2
    return img.crop((left, top, left + short, top + short)).resize(size, Image.LANCZOS)


def bytes_to_pil(raw: bytes) -> Image.Image:
    return Image.open(io.BytesIO(raw)).convert("RGB")


# -- captioning ----------------------------------------------------------------

_blip_pipe = None


def _load_blip():
    global _blip_pipe
    if _blip_pipe is None:
        from transformers import pipeline as hf_pipeline
        import torch
        device = 0 if torch.cuda.is_available() else -1
        print(f"Loading BLIP ({BLIP_MODEL_ID}) on {'GPU' if device == 0 else 'CPU'}...",
              flush=True)
        _blip_pipe = hf_pipeline("image-to-text", model=BLIP_MODEL_ID, device=device)
        print("BLIP ready.", flush=True)
    return _blip_pipe


def caption_blip(img: Image.Image) -> str:
    result = _load_blip()(img, max_new_tokens=60)
    raw = result[0]["generated_text"].strip() if result else ""
    base = raw if raw else "traditional Japanese art"
    return f"{TRIGGER_TOKEN}, ukiyo-e woodblock print, {base}"


def caption_template(artist_int: int) -> str:
    if ARTIST_NAMES and 0 <= artist_int < len(ARTIST_NAMES):
        artist = ARTIST_NAMES[artist_int].replace("_", " ")
        return f"{TRIGGER_TOKEN}, ukiyo-e woodblock print by {artist}, traditional Japanese woodblock art"
    return f"{TRIGGER_TOKEN}, ukiyo-e woodblock print, traditional Japanese woodblock art"


# -- main ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run",  action="store_true")
    ap.add_argument("--no-blip",  action="store_true")
    ap.add_argument("--max",      type=int, default=TARGET_MAX)
    args = ap.parse_args()

    use_blip = not args.no_blip and not args.dry_run

    # -- locate cached parquet shards ----------------------------------------
    shards = sorted(glob.glob(HF_CACHE_GLOB))
    if not shards:
        print("ERROR: no cached wikiart parquet shards found at:", HF_CACHE_GLOB, file=sys.stderr)
        print("Run: from datasets import load_dataset; load_dataset('huggan/wikiart', split='train')",
              file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(shards)} cached parquet shards.", flush=True)

    # -- populate artist names ------------------------------------------------
    try:
        from datasets import load_dataset
        ds_tmp = load_dataset("huggan/wikiart", split="train", streaming=True)
        ARTIST_NAMES.extend(ds_tmp.features["artist"].names)
        print(f"Artist names loaded ({len(ARTIST_NAMES)} artists).", flush=True)
    except Exception as e:
        print(f"Could not load artist names ({e}); using generic captions.", flush=True)

    img_dir = OUT_DIR / "images"
    if not args.dry_run:
        img_dir.mkdir(parents=True, exist_ok=True)

    t0             = time.time()
    records: list[dict] = []
    total_scanned  = 0
    total_ukiyoe   = 0
    total_filtered = 0

    print(f"Scanning {len(shards)} shards for Ukiyo-e (style={UKIYO_E_STYLE_IDX})...", flush=True)

    for shard_path in shards:
        if len(records) >= args.max:
            break

        shard_name = Path(shard_path).name
        table = pq.read_table(shard_path, columns=["image", "artist", "style"])
        styles  = table["style"].to_pylist()
        artists = table["artist"].to_pylist()
        images  = table["image"].to_pylist()

        shard_ukiyo = 0
        for style, artist_int, img_struct in zip(styles, artists, images):
            total_scanned += 1
            if style != UKIYO_E_STYLE_IDX:
                continue
            total_ukiyoe += 1

            raw_bytes = img_struct["bytes"] if isinstance(img_struct, dict) else bytes(img_struct["bytes"])
            try:
                img = bytes_to_pil(raw_bytes)
            except Exception as e:
                print(f"  decode error in {shard_name}: {e}", flush=True)
                continue

            if not qualifies(img):
                total_filtered += 1
                continue

            idx      = len(records) + 1
            filename = f"{idx:03d}.jpg"

            if not args.dry_run:
                processed = center_crop_resize(img, OUTPUT_SIZE)
                processed.save(img_dir / filename, "JPEG", quality=90)
                if use_blip:
                    caption = caption_blip(img)
                else:
                    caption = caption_template(artist_int)
            else:
                caption = caption_template(artist_int)

            records.append({"file_name": f"images/{filename}", "text": caption})
            shard_ukiyo += 1

            if len(records) % 10 == 0 or len(records) == 1:
                elapsed = time.time() - t0
                print(f"  [{elapsed:5.0f}s] {len(records):3d} collected | "
                      f"shard {shard_name} | scanned {total_scanned} | "
                      f"ukiyo-e so far {total_ukiyoe}", flush=True)

            if len(records) >= args.max:
                break

    elapsed = time.time() - t0
    print(f"\n-- Scan complete ({elapsed:.0f}s) --", flush=True)
    print(f"  Shards scanned : {len(shards)}", flush=True)
    print(f"  Total scanned  : {total_scanned}", flush=True)
    print(f"  Ukiyo-e found  : {total_ukiyoe}", flush=True)
    print(f"  Filtered out   : {total_filtered} (res<{MIN_SHORTEST_EDGE}px or aspect out of range)", flush=True)
    print(f"  Collected      : {len(records)} images", flush=True)

    if args.dry_run:
        print("\nDry-run: no files written.", flush=True)
        if records:
            print("\nSample captions:", flush=True)
            for r in records[:5]:
                print(f"  {r['file_name']}  ->  {r['text']}", flush=True)
        return

    if not records:
        print("ERROR: no images collected.", file=sys.stderr)
        sys.exit(1)

    # -- save metadata.jsonl ------------------------------------------------─
    jsonl_path = OUT_DIR / "metadata.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # -- final summary ------------------------------------------------------─
    saved = sorted(img_dir.glob("*.jpg"))
    total_mb = sum(p.stat().st_size for p in saved) / 1024**2
    print(f"\n-- Dataset saved: {OUT_DIR.resolve()} --", flush=True)
    print(f"  Images        : {len(saved)}", flush=True)
    print(f"  Disk size     : {total_mb:.1f} MB", flush=True)
    print(f"  metadata.jsonl: {jsonl_path}", flush=True)

    print("\n-- Sample captions (first 5) --", flush=True)
    for rec in records[:5]:
        print(f"  {rec['file_name']}  ->  {rec['text']}", flush=True)

    print("\n-- Sample filenames (first 5) --", flush=True)
    for p in saved[:5]:
        print(f"  {p.name}  ({p.stat().st_size // 1024} KB)", flush=True)

    if len(records) < 50:
        print(f"\nWARNING: only {len(records)} images (target ≥ 50).", flush=True)


if __name__ == "__main__":
    main()
