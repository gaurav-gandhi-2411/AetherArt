#!/usr/bin/env python
"""
Generate standard vs LCM side-by-side comparison.

Output: docs/lcm_comparison.png

Usage:
    python scripts/generate_lcm_comparison.py
"""
import time
from pathlib import Path

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont

from aetherart.lcm import LCM_STEPS, LCM_GUIDANCE, apply_lcm_mode, restore_standard_mode

REPO_ROOT  = Path(__file__).resolve().parent.parent
OUT_PATH   = REPO_ROOT / "docs" / "lcm_comparison.png"
MODEL_ID   = "sd2-community/stable-diffusion-2-1"

PROMPTS = [
    "a samurai warrior on horseback, dramatic lighting, cinematic",
    "a tea house at the base of Mount Fuji at dusk",
    "a tiger prowling through a bamboo forest",
]
STANDARD_STEPS = 30
SEED = 42
GUIDANCE_STD = 7.5
IMG_SIZE = 512
THUMB_W, THUMB_H = 400, 400
LABEL_H = 28


def build_grid(rows: list[tuple[str, list]]) -> Image.Image:
    pad = 4
    n_rows = len(rows)
    n_cols = len(PROMPTS)
    label_col_w = 180
    w = label_col_w + n_cols * (THUMB_W + pad)
    h = n_rows * (THUMB_H + LABEL_H + pad)
    grid = Image.new("RGB", (w, h), (20, 20, 20))
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    for i, (row_label, images) in enumerate(rows):
        y = i * (THUMB_H + LABEL_H + pad)
        draw.text((label_col_w // 2, y + THUMB_H // 2), row_label,
                  fill=(200, 200, 200), font=font, anchor="mm")
        for j, img in enumerate(images):
            x = label_col_w + j * (THUMB_W + pad)
            grid.paste(img.resize((THUMB_W, THUMB_H), Image.LANCZOS), (x, y))
    return grid


def main() -> None:
    print(f"[lcm] Loading {MODEL_ID}")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16
    ).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    results = {}

    for label, n_steps, guidance, mode in [
        (f"Standard ({STANDARD_STEPS}-step DPM++)", STANDARD_STEPS, GUIDANCE_STD, "standard"),
        (f"LCM ({LCM_STEPS}-step LCMScheduler)", LCM_STEPS, LCM_GUIDANCE, "lcm"),
    ]:
        if mode == "lcm":
            apply_lcm_mode(pipe)
        else:
            restore_standard_mode(pipe)

        images = []
        t0 = time.monotonic()
        for prompt in PROMPTS:
            img = pipe(
                prompt,
                num_inference_steps=n_steps,
                guidance_scale=guidance,
                generator=torch.Generator(device="cuda").manual_seed(SEED),
                height=IMG_SIZE, width=IMG_SIZE,
            ).images[0]
            images.append(img)
        elapsed = time.monotonic() - t0
        per_img = elapsed / len(PROMPTS)
        print(f"[lcm]  {label}: {elapsed:.1f}s total, {per_img:.1f}s/image")
        results[label] = (images, elapsed, per_img)

    rows = [(lbl, data[0]) for lbl, data in results.items()]
    grid = build_grid(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    grid.save(str(OUT_PATH))
    mb = OUT_PATH.stat().st_size / 1e6
    print(f"\n[lcm] Saved: {OUT_PATH}  ({grid.size[0]}×{grid.size[1]} px, {mb:.1f} MB)")

    std_per = list(results.values())[0][2]
    lcm_per = list(results.values())[1][2]
    print(f"[lcm] Speedup: {std_per/lcm_per:.1f}×  ({std_per:.1f}s → {lcm_per:.1f}s per image)")


if __name__ == "__main__":
    main()
