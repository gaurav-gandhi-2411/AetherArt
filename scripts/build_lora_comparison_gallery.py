#!/usr/bin/env python
"""
Generate a 2-row × 4-col comparison gallery: base SD 2.1 vs Ukiyo-e LoRA.

Output: reports/lora_comparison_gallery.png
        reports/lora_comparison_gallery/{base,lora}/prompt_NN.png

Usage:
    python scripts/build_lora_comparison_gallery.py
"""
import time
from pathlib import Path

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "reports" / "lora_comparison_gallery"
GRID_PATH = REPO_ROOT / "reports" / "lora_comparison_gallery.png"
LORA_PATH = REPO_ROOT / "data" / "lora" / "ukiyo-e" / "ukiyo-e-lora.safetensors"

MODEL_ID = "sd2-community/stable-diffusion-2-1"
SEED = 42
STEPS = 30
GUIDANCE = 7.5
IMG_SIZE = 512
THUMB_SIZE = 400
TRIGGER = "ukyowood"
LORA_NEG = "text, watermark, calligraphy, signature, words, letters"

# 4 prompts — without trigger (added for LoRA row automatically)
PROMPTS = [
    "a samurai warrior on horseback",
    "Mount Fuji at sunset with cherry blossoms",
    "a dragon flying over Tokyo",
    "a woman in kimono in a garden",
]
PROMPT_LABELS = [
    "Samurai on horseback",
    "Fuji + cherry blossoms",
    "Dragon over Tokyo",
    "Kimono in garden",
]
ROW_LABELS = ["base SD 2.1", "Ukiyo-e LoRA"]


def load_pipeline() -> StableDiffusionPipeline:
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate_row(
    pipe: StableDiffusionPipeline,
    prompts: list[str],
    negative: str | None,
) -> list[Image.Image]:
    images = []
    for prompt in prompts:
        img = pipe(
            prompt,
            negative_prompt=negative,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            generator=torch.Generator(device="cuda").manual_seed(SEED),
            height=IMG_SIZE,
            width=IMG_SIZE,
        ).images[0]
        images.append(img)
    return images


def build_grid(rows: list[tuple[str, list[Image.Image]]]) -> Image.Image:
    label_w = 130
    label_h = 40
    pad = 3
    n_rows, n_cols = len(rows), len(PROMPTS)
    w = label_w + n_cols * (THUMB_SIZE + pad)
    h = label_h + n_rows * (THUMB_SIZE + pad)
    grid = Image.new("RGB", (w, h), (230, 230, 230))
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for j, lbl in enumerate(PROMPT_LABELS):
        x = label_w + j * (THUMB_SIZE + pad) + THUMB_SIZE // 2
        draw.text((x, label_h // 2), lbl, fill=(40, 40, 40), font=font, anchor="mm")

    for i, (row_label, images) in enumerate(rows):
        y = label_h + i * (THUMB_SIZE + pad)
        draw.text(
            (label_w // 2, y + THUMB_SIZE // 2),
            row_label,
            fill=(40, 40, 40),
            font=font,
            anchor="mm",
        )
        for j, img in enumerate(images):
            thumb = img.resize((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
            grid.paste(thumb, (label_w + j * (THUMB_SIZE + pad), y))
    return grid


def main() -> None:
    if not LORA_PATH.exists():
        raise FileNotFoundError(f"LoRA weights not found: {LORA_PATH}")

    print(f"[gallery] Loading pipeline: {MODEL_ID}")
    pipe = load_pipeline()
    print(f"[gallery] VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB\n")

    rows: list[tuple[str, list[Image.Image]]] = []

    # Base row
    print("[gallery] Row 1/2: base SD 2.1 (no LoRA)")
    t0 = time.monotonic()
    base_imgs = generate_row(pipe, PROMPTS, negative=None)
    print(f"          done in {time.monotonic()-t0:.0f}s")
    (OUT_DIR / "base").mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(base_imgs):
        img.save(OUT_DIR / "base" / f"prompt_{i+1:02d}.png")
    rows.append((ROW_LABELS[0], base_imgs))

    # LoRA row
    print("[gallery] Row 2/2: Ukiyo-e LoRA")
    pipe.load_lora_weights(str(LORA_PATH.parent), weight_name=LORA_PATH.name)
    lora_prompts = [f"{TRIGGER} {p}" for p in PROMPTS]
    t0 = time.monotonic()
    lora_imgs = generate_row(pipe, lora_prompts, negative=LORA_NEG)
    print(f"          done in {time.monotonic()-t0:.0f}s")
    pipe.unload_lora_weights()
    (OUT_DIR / "lora").mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(lora_imgs):
        img.save(OUT_DIR / "lora" / f"prompt_{i+1:02d}.png")
    rows.append((ROW_LABELS[1], lora_imgs))

    # Grid
    grid = build_grid(rows)
    GRID_PATH.parent.mkdir(parents=True, exist_ok=True)
    grid.save(str(GRID_PATH))
    mb = GRID_PATH.stat().st_size / 1e6
    print(f"\n[gallery] Grid saved: {GRID_PATH}  ({grid.size[0]}×{grid.size[1]} px, {mb:.1f} MB)")


if __name__ == "__main__":
    main()
