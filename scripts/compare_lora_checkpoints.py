#!/usr/bin/env python
"""
Compare LoRA checkpoints across multiple prompts to pick the best adapter.

Generates:
  reports/lora_checkpoint_comparison.png  — 6-row × 6-col grid
  reports/lora_validation/{row}/prompt_NN.png  — individual images

Rows:  baseline (no LoRA), checkpoint-500, 750, 1000, 1250, 1500
Cols:  6 test prompts (same seed across all rows for direct comparison)

Usage:
    python scripts/compare_lora_checkpoints.py
"""
import time
from pathlib import Path

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
VAL_DIR = REPO_ROOT / "reports" / "lora_validation"
GRID_PATH = REPO_ROOT / "reports" / "lora_checkpoint_comparison.png"
CKPT_ROOT = REPO_ROOT / "data" / "lora" / "ukiyo-e" / "training_output"

MODEL_ID = "sd2-community/stable-diffusion-2-1"
SEED = 42
STEPS = 30
GUIDANCE = 7.5
IMG_SIZE = 512
THUMB_SIZE = 384

CHECKPOINTS = [500, 750, 1000, 1250, 1500]

PROMPTS = [
    "ukyowood ukiyo-e print of Mount Fuji at sunset",
    "ukyowood ukiyo-e print of a samurai warrior on horseback",
    "ukyowood ukiyo-e print of cherry blossoms in spring",
    "ukyowood ukiyo-e print of a tea ceremony in a traditional garden",
    "ukyowood ukiyo-e print of a dragon over Tokyo",
    "ukyowood ukiyo-e print of a woman in kimono walking in rain",
]

PROMPT_LABELS = [
    "Fuji sunset",
    "Samurai",
    "Cherry blossoms",
    "Tea ceremony",
    "Dragon / Tokyo",
    "Kimono rain",
]


def load_pipeline() -> StableDiffusionPipeline:
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
    ).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate_row(pipe: StableDiffusionPipeline, lora_path: Path | None) -> list[Image.Image]:
    if lora_path is not None:
        pipe.load_lora_weights(str(lora_path.parent), weight_name=lora_path.name)

    images = []
    for prompt in PROMPTS:
        img = pipe(
            prompt,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            generator=torch.Generator(device="cuda").manual_seed(SEED),
            height=IMG_SIZE,
            width=IMG_SIZE,
        ).images[0]
        images.append(img)

    if lora_path is not None:
        pipe.unload_lora_weights()

    return images


def save_row(images: list[Image.Image], row_name: str) -> None:
    row_dir = VAL_DIR / row_name
    row_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        img.save(row_dir / f"prompt_{i + 1:02d}.png")


def build_grid(rows: list[tuple[str, list[Image.Image]]]) -> Image.Image:
    label_w = 140
    label_h = 44
    pad = 3
    n_rows, n_cols = len(rows), len(PROMPTS)
    w = label_w + n_cols * (THUMB_SIZE + pad)
    h = label_h + n_rows * (THUMB_SIZE + pad)

    grid = Image.new("RGB", (w, h), color=(230, 230, 230))
    draw = ImageDraw.Draw(grid)

    try:
        font_hdr = ImageFont.truetype("arial.ttf", 13)
        font_row = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font_hdr = ImageFont.load_default()
        font_row = font_hdr

    # Column headers
    for j, lbl in enumerate(PROMPT_LABELS):
        x = label_w + j * (THUMB_SIZE + pad) + THUMB_SIZE // 2
        draw.text((x, label_h // 2), lbl, fill=(40, 40, 40), font=font_hdr, anchor="mm")

    # Rows
    for i, (row_label, images) in enumerate(rows):
        y = label_h + i * (THUMB_SIZE + pad)
        draw.text(
            (label_w // 2, y + THUMB_SIZE // 2),
            row_label,
            fill=(40, 40, 40),
            font=font_row,
            anchor="mm",
        )
        for j, img in enumerate(images):
            thumb = img.resize((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
            grid.paste(thumb, (label_w + j * (THUMB_SIZE + pad), y))

    return grid


def main() -> None:
    print(f"[compare] Loading pipeline: {MODEL_ID}")
    pipe = load_pipeline()
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"[compare] Ready — {vram:.1f} GB VRAM allocated\n")

    rows: list[tuple[str, list[Image.Image]]] = []

    # --- baseline (no LoRA) ---
    print("[compare] Row 1/6: baseline (no LoRA)")
    t0 = time.monotonic()
    imgs = generate_row(pipe, lora_path=None)
    elapsed = time.monotonic() - t0
    print(f"          done in {elapsed:.0f}s ({elapsed/len(PROMPTS):.1f}s/img)")
    save_row(imgs, "baseline")
    rows.append(("baseline\n(no LoRA)", imgs))

    # --- LoRA checkpoints ---
    for n, step in enumerate(CHECKPOINTS, start=2):
        lora_path = CKPT_ROOT / f"checkpoint-{step}" / "pytorch_lora_weights.safetensors"
        if not lora_path.exists():
            print(f"[compare] Row {n}/6: checkpoint-{step} — MISSING, skipped")
            continue

        print(f"[compare] Row {n}/6: checkpoint-{step}")
        t0 = time.monotonic()
        imgs = generate_row(pipe, lora_path=lora_path)
        elapsed = time.monotonic() - t0
        print(f"          done in {elapsed:.0f}s ({elapsed/len(PROMPTS):.1f}s/img)")
        save_row(imgs, f"checkpoint-{step}")
        rows.append((f"ckpt-{step}", imgs))

    # --- Grid ---
    print("\n[compare] Building grid...")
    GRID_PATH.parent.mkdir(parents=True, exist_ok=True)
    grid = build_grid(rows)
    grid.save(str(GRID_PATH))
    size_mb = GRID_PATH.stat().st_size / 1e6
    print(f"[compare] Grid saved: {GRID_PATH}")
    print(f"          {grid.size[0]} x {grid.size[1]} px, {size_mb:.1f} MB")
    print("\n[compare] Individual images: reports/lora_validation/")
    for row_label, _ in rows:
        print(f"          {row_label.splitlines()[0]}/")


if __name__ == "__main__":
    main()
