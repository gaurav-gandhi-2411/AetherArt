#!/usr/bin/env python
"""
Generate standard vs LCM vs SDXL Turbo three-tier comparison.

Output: docs/three_tier_comparison.png

Usage:
    python scripts/generate_three_tier_comparison.py

NOTE: First run downloads SDXL Turbo (~6.7 GB).
"""
import time
from pathlib import Path

import torch
from diffusers import (
    AutoPipelineForText2Image,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
)
from PIL import Image, ImageDraw, ImageFont

from aetherart.lcm import LCM_GUIDANCE, LCM_STEPS, apply_lcm_mode, restore_standard_mode
from aetherart.sdxl_turbo import TURBO_GUIDANCE, TURBO_MODEL_ID, TURBO_STEPS, generate_turbo

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH  = REPO_ROOT / "docs" / "three_tier_comparison.png"
SD21_ID   = "sd2-community/stable-diffusion-2-1"

PROMPTS = [
    "a samurai warrior on horseback, dramatic lighting, cinematic",
    "a tea house at the base of Mount Fuji at dusk",
    "a tiger prowling through a bamboo forest",
]
STANDARD_STEPS = 30
SEED = 42
GUIDANCE_STD = 7.5
IMG_SIZE = 512
THUMB_W, THUMB_H = 360, 360
LABEL_COL_W = 220
PAD = 4
LABEL_H = 24


def build_grid(rows: list[tuple[str, list, float]]) -> Image.Image:
    n_rows = len(rows)
    n_cols = len(PROMPTS)
    w = LABEL_COL_W + n_cols * (THUMB_W + PAD)
    h = n_rows * (THUMB_H + LABEL_H + PAD)
    grid = Image.new("RGB", (w, h), (20, 20, 20))
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
        font_sm = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
        font_sm = font

    for i, (row_label, images, per_img_s) in enumerate(rows):
        y = i * (THUMB_H + LABEL_H + PAD)
        draw.text(
            (LABEL_COL_W // 2, y + THUMB_H // 2 - 10),
            row_label, fill=(220, 220, 220), font=font, anchor="mm",
        )
        draw.text(
            (LABEL_COL_W // 2, y + THUMB_H // 2 + 10),
            f"{per_img_s:.1f}s/img", fill=(160, 200, 160), font=font_sm, anchor="mm",
        )
        for j, img in enumerate(images):
            x = LABEL_COL_W + j * (THUMB_W + PAD)
            grid.paste(img.resize((THUMB_W, THUMB_H), Image.LANCZOS), (x, y))
    return grid


def main() -> None:
    rows = []

    # ── SD 2.1 standard + LCM ──────────────────────────────────────────────
    print(f"[tier] Loading {SD21_ID}")
    pipe_sd21 = StableDiffusionPipeline.from_pretrained(
        SD21_ID, torch_dtype=torch.float16
    ).to("cuda")
    pipe_sd21.scheduler = DPMSolverMultistepScheduler.from_config(pipe_sd21.scheduler.config)
    pipe_sd21.set_progress_bar_config(disable=True)

    for label, n_steps, guidance, mode in [
        (f"Standard  ({STANDARD_STEPS}-step DPM++)", STANDARD_STEPS, GUIDANCE_STD, "standard"),
        (f"LCM  ({LCM_STEPS}-step scheduler)", LCM_STEPS, LCM_GUIDANCE, "lcm"),
    ]:
        if mode == "lcm":
            apply_lcm_mode(pipe_sd21)
        else:
            restore_standard_mode(pipe_sd21)

        images, t0 = [], time.monotonic()
        for prompt in PROMPTS:
            img = pipe_sd21(
                prompt,
                num_inference_steps=n_steps,
                guidance_scale=guidance,
                generator=torch.Generator(device="cuda").manual_seed(SEED),
                height=IMG_SIZE, width=IMG_SIZE,
            ).images[0]
            images.append(img)
        per_img = (time.monotonic() - t0) / len(PROMPTS)
        print(f"[tier]  {label}: {per_img:.1f}s/image")
        rows.append((label, images, per_img))

    # free SD 2.1
    pipe_sd21.to("cpu")
    del pipe_sd21
    torch.cuda.empty_cache()

    # ── SDXL Turbo ─────────────────────────────────────────────────────────
    print(f"[tier] Loading {TURBO_MODEL_ID}  (downloads ~6.7 GB on first run)")
    pipe_turbo = AutoPipelineForText2Image.from_pretrained(
        TURBO_MODEL_ID, torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")
    pipe_turbo.set_progress_bar_config(disable=True)

    turbo_label = f"SDXL Turbo  ({TURBO_STEPS}-step ADD)"
    images, t0 = [], time.monotonic()
    for prompt in PROMPTS:
        img, _ = generate_turbo(pipe_turbo, prompt, seed=SEED, width=IMG_SIZE, height=IMG_SIZE)
        images.append(img)
    per_img = (time.monotonic() - t0) / len(PROMPTS)
    print(f"[tier]  {turbo_label}: {per_img:.1f}s/image")
    rows.append((turbo_label, images, per_img))

    pipe_turbo.to("cpu")
    del pipe_turbo
    torch.cuda.empty_cache()

    # ── assemble + save ────────────────────────────────────────────────────
    grid = build_grid(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    grid.save(str(OUT_PATH))
    mb = OUT_PATH.stat().st_size / 1e6
    print(f"\n[tier] Saved: {OUT_PATH}  ({grid.size[0]}×{grid.size[1]} px, {mb:.1f} MB)")

    std_per  = rows[0][2]
    lcm_per  = rows[1][2]
    turb_per = rows[2][2]
    print(f"[tier] LCM speedup:   {std_per/lcm_per:.1f}×  ({std_per:.1f}s → {lcm_per:.1f}s)")
    print(f"[tier] Turbo speedup: {std_per/turb_per:.1f}×  ({std_per:.1f}s → {turb_per:.1f}s)")


if __name__ == "__main__":
    main()
