"""One-off dataset curation pass for the calligraphy-artifact retrain.

The published Ukiyo-e LoRAs (SD 2.1 and SDXL, checkpoint-1000) learned visible text
artifacts (signatures, dates, script, title cartouches) because the WikiArt source images
carry embedded text in the image margins. The documented fix (both HF model cards, and
docs/lab_notebook.md:23-25) is retraining on a curated subset with those images removed.

First attempt used EasyOCR (English) and was WRONG: it flagged 31/80 images at high
confidence on single out-of-context Latin glyphs ('{', '#', '84', 'H'...) — noise from an
English-only Latin-character detector misfiring on Japanese woodblock texture/brushwork, not
real detections. It also cannot read the actual artifact, which is Japanese script/kanji
cartouches and seals, not Latin text. Discarded.

This version uses the local Ollama vision-language judge (qwen2.5vl:7b, zero-cost, no paid
API) to directly ask the semantic question EasyOCR couldn't answer: "does this image contain
visible embedded text, calligraphy, a signature, or a cartouche/seal mark?" — which is robust
to script (kanji, kana, or Latin) since it's a vision-language judgment, not character
recognition.

Usage:
    python scripts/curate_ukiyo_e_dataset.py
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent
METADATA_IN = ROOT / "data" / "lora" / "ukiyo-e" / "metadata.jsonl"
CURATED_DIR = ROOT / "data" / "lora" / "ukiyo-e-curated"
CURATED_IMAGES_DIR = CURATED_DIR / "images"
CURATED_METADATA = CURATED_DIR / "metadata.jsonl"
REPORT_PATH = ROOT / "reports" / "ukiyo_e_curation_report.json"
# Incremental per-image checkpoint — survives an interrupted run; --resume skips done images.
CLASSIFICATIONS_PATH = ROOT / "reports" / "ukiyo_e_classifications.json"

OLLAMA_URL = "http://localhost:11434/api/generate"
VLM_MODEL = "qwen2.5vl:7b"

DETECT_PROMPT = """You are screening a training image for a Japanese Ukiyo-e woodblock print
style LoRA adapter. The dataset must exclude images with visible embedded text so the model
does not learn to reproduce text artifacts.

Look at this image carefully. Does it contain ANY visible embedded text, calligraphy,
signature, date stamp, title cartouche, or seal/stamp mark anywhere in the image (including
small text in a corner or margin)? This includes Japanese kanji/kana script, Latin text, or
any stylized mark that represents writing (not decorative pattern/texture that merely
resembles brushwork without being legible marks).

Respond with ONLY a JSON object: {"has_text_artifact": <true|false>, "confidence": <float 0-1>,
"location": "<brief description or 'none'>"}"""


def classify_image(img_path: Path) -> dict:
    with img_path.open("rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": VLM_MODEL,
            "prompt": DETECT_PROMPT,
            "images": [b64],
            "stream": False,
            "format": "json",
        },
        timeout=120,
    )
    resp.raise_for_status()
    return json.loads(resp.json()["response"])


def load_classifications() -> dict[str, dict]:
    if not CLASSIFICATIONS_PATH.exists():
        return {}
    try:
        return {c["file_name"]: c for c in json.loads(CLASSIFICATIONS_PATH.read_text(encoding="utf-8"))}
    except Exception:
        return {}


def save_classifications(classifications: dict[str, dict]) -> None:
    CLASSIFICATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CLASSIFICATIONS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(list(classifications.values()), indent=2), encoding="utf-8")
    tmp.replace(CLASSIFICATIONS_PATH)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    with METADATA_IN.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(records)} source records from {METADATA_IN}", flush=True)

    classifications = load_classifications() if args.resume else {}
    if classifications:
        print(f"Resuming: {len(classifications)} already classified.", flush=True)

    for i, rec in enumerate(records):
        if rec["file_name"] in classifications:
            continue
        img_path = ROOT / "data" / "lora" / "ukiyo-e" / rec["file_name"]
        try:
            verdict = classify_image(img_path)
            has_artifact = bool(verdict.get("has_text_artifact"))
            status = "flagged" if has_artifact else "kept"
            print(
                f"[{i + 1}/{len(records)}] {rec['file_name']}: {status} "
                f"(confidence={verdict.get('confidence')}, location={verdict.get('location')})",
                flush=True,
            )
            classifications[rec["file_name"]] = {**rec, "vlm_verdict": verdict, "has_artifact": has_artifact}
        except Exception as e:
            print(f"[{i + 1}/{len(records)}] {rec['file_name']}: VLM call failed ({e}); keeping by default", flush=True)
            classifications[rec["file_name"]] = {**rec, "vlm_verdict": None, "has_artifact": False}

        save_classifications(classifications)

    kept = [c for c in classifications.values() if not c["has_artifact"]]
    flagged = [c for c in classifications.values() if c["has_artifact"]]
    print(f"\n=== Curation complete: {len(kept)} kept, {len(flagged)} flagged/excluded ===", flush=True)

    CURATED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    with CURATED_METADATA.open("w", encoding="utf-8") as f:
        for rec in kept:
            src = ROOT / "data" / "lora" / "ukiyo-e" / rec["file_name"]
            dst = CURATED_IMAGES_DIR / Path(rec["file_name"]).name
            dst.write_bytes(src.read_bytes())
            f.write(json.dumps({"file_name": f"images/{dst.name}", "text": rec["text"]}) + "\n")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(
            {
                "detector": "qwen2.5vl:7b (local Ollama VLM judge)",
                "total_source_images": len(records),
                "kept": len(kept),
                "flagged_excluded": len(flagged),
                "flagged_detail": flagged,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Curated dataset: {CURATED_METADATA} ({len(kept)} images)", flush=True)
    print(f"Report: {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
