#!/usr/bin/env python
"""
Generate a 2×2 hero grid for the README.

Output: docs/hero.png  (individual tiles also saved to docs/hero_tiles/)

Usage:
    python scripts/generate_hero_image.py
    python scripts/generate_hero_image.py --steps 80   # higher quality
"""
import argparse
import time
from pathlib import Path

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "docs" / "hero_tiles"
HERO_PATH = REPO_ROOT / "docs" / "hero.png"
LORA_PATH = REPO_ROOT / "data" / "lora" / "ukiyo-e" / "ukiyo-e-lora.safetensors"

MODEL_ID = "sd2-community/stable-diffusion-2-1"
SEED = 42
GUIDANCE = 7.5
IMG_SIZE = 512
THUMB_W = 640
THUMB_H = 640
TRIGGER = "ukyowood"
NEGATIVE = "text, watermark, calligraphy, signature, words, letters, low quality, blurry"

PROMPTS = [
    "ukyowood ukiyo-e woodblock print of Mount Fuji at sunset with reflections in calm water, cherry blossoms",  # noqa: E501
    "ukyowood ukiyo-e woodblock print of two samurai warriors duelling under cherry blossoms, dramatic pose",  # noqa: E501
    "ukyowood ukiyo-e woodblock print of a tiger leaping through a bamboo forest, bold colours",
    "ukyowood ukiyo-e woodblock print of a tea ceremony in a moonlit pavilion overlooking a garden",
]
LABELS = [
    "Mount Fuji at sunset",
    "Samurai duel",
    "Tiger in bamboo",
    "Tea ceremony",
]


def load_pipeline(steps: int) -> StableDiffusionPipeline:
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=False)
    return pipe


def generate(pipe: StableDiffusionPipeline, prompt: str, steps: int) -> Image.Image:
    return pipe(
        prompt,
        negative_prompt=NEGATIVE,
        num_inference_steps=steps,
        guidance_scale=GUIDANCE,
        generator=torch.Generator(device="cuda").manual_seed(SEED),
        height=IMG_SIZE,
        width=IMG_SIZE,
    ).images[0]


def build_grid(images: list[Image.Image]) -> Image.Image:
    pad = 4
    label_h = 32
    cols, rows = 2, 2
    w = cols * THUMB_W + (cols + 1) * pad
    h = rows * (THUMB_H + label_h) + (rows + 1) * pad
    grid = Image.new("RGB", (w, h), (20, 20, 20))
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for idx, (img, label) in enumerate(zip(images, LABELS)):
        row, col = divmod(idx, 2)
        x = pad + col * (THUMB_W + pad)
        y = pad + row * (THUMB_H + label_h + pad)
        thumb = img.resize((THUMB_W, THUMB_H), Image.LANCZOS)
        grid.paste(thumb, (x, y))
        draw.text(
            (x + THUMB_W // 2, y + THUMB_H + label_h // 2),
            label,
            fill=(200, 200, 200),
            font=font,
            anchor="mm",
        )
    return grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    if not LORA_PATH.exists():
        raise FileNotFoundError(f"LoRA weights not found: {LORA_PATH}")

    print(f"[hero] Loading pipeline: {MODEL_ID}")
    pipe = load_pipeline(args.steps)
    pipe.load_lora_weights(str(LORA_PATH.parent), weight_name=LORA_PATH.name)
    pipe.set_progress_bar_config(disable=True)
    print(f"[hero] VRAM after load: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    images: list[Image.Image] = []

    for i, prompt in enumerate(PROMPTS):
        label = LABELS[i]
        print(f"[hero] {i+1}/4: {label}")
        t0 = time.monotonic()
        img = generate(pipe, prompt, args.steps)
        elapsed = time.monotonic() - t0
        tile_path = OUT_DIR / f"tile_{i+1:02d}.png"
        img.save(tile_path)
        print(f"          saved {tile_path.name}  ({elapsed:.0f}s)")
        images.append(img)

    grid = build_grid(images)
    HERO_PATH.parent.mkdir(parents=True, exist_ok=True)
    grid.save(str(HERO_PATH))
    mb = HERO_PATH.stat().st_size / 1e6
    print(f"\n[hero] Grid saved: {HERO_PATH}  ({grid.size[0]}×{grid.size[1]} px, {mb:.1f} MB)")


if __name__ == "__main__":
    main()
