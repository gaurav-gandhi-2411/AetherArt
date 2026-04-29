#!/usr/bin/env python
"""
Four-tier showcase: Standard fp16 · LCM 4-step · 8-bit INT8 · SDXL Turbo.

Generates a 4-row × 4-column grid demonstrating each speed/memory tier.
Output: docs/four_tier_showcase.png

Usage:
    python scripts/generate_four_tier_showcase.py

NOTE: First run downloads SDXL Turbo (~6.7 GB) and loads all four pipelines sequentially.
      Expects ~20 min on RTX 3070 8 GB.
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

from aetherart.lcm import LCM_GUIDANCE, LCM_STEPS, apply_lcm_mode
from aetherart.quantization import load_sd21_quantized
from aetherart.sdxl_turbo import TURBO_MODEL_ID, generate_turbo

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH  = REPO_ROOT / "docs" / "four_tier_showcase.png"
SD21_ID   = "sd2-community/stable-diffusion-2-1"

PROMPTS = [
    "a samurai warrior on horseback, dramatic lighting, cinematic",
    "cherry blossoms over a stone bridge at night, moonlight reflections",
    "a tiger prowling through a bamboo forest, painterly",
    "a tea house at the base of Mount Fuji at dusk, ukiyo-e style",
]
SEED      = 42
IMG_SIZE  = 512
THUMB_W, THUMB_H = 340, 340
LABEL_COL_W = 240
PAD = 4
LABEL_H = 26


def build_grid(rows: list[tuple]) -> Image.Image:
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

    for i, (row_label, sub_label, images) in enumerate(rows):
        y = i * (THUMB_H + LABEL_H + PAD)
        draw.text(
            (LABEL_COL_W // 2, y + THUMB_H // 2 - 12),
            row_label, fill=(220, 220, 220), font=font, anchor="mm",
        )
        draw.text(
            (LABEL_COL_W // 2, y + THUMB_H // 2 + 8),
            sub_label, fill=(160, 200, 160), font=font_sm, anchor="mm",
        )
        for j, img in enumerate(images):
            x = LABEL_COL_W + j * (THUMB_W + PAD)
            grid.paste(img.resize((THUMB_W, THUMB_H), Image.LANCZOS), (x, y))
    return grid


def gen_batch(pipe, prompts, n_steps, guidance, device="cuda"):
    images = []
    for prompt in prompts:
        img = pipe(
            prompt,
            num_inference_steps=n_steps,
            guidance_scale=guidance,
            generator=torch.Generator(device=device).manual_seed(SEED),
            height=IMG_SIZE, width=IMG_SIZE,
        ).images[0]
        images.append(img)
    return images


def main() -> None:
    rows = []

    # ── Tier 1: Standard fp16 (30-step DPM++) ────────────────────────────
    print("[4tier] Tier 1: Standard fp16")
    pipe = StableDiffusionPipeline.from_pretrained(SD21_ID, torch_dtype=torch.float16).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    t0 = time.monotonic()
    imgs = gen_batch(pipe, PROMPTS, n_steps=30, guidance=7.5)
    per = (time.monotonic() - t0) / len(PROMPTS)
    print(f"  {per:.1f}s/img")
    rows.append(("Standard fp16", f"30-step DPM++  {per:.1f}s/img", imgs))

    # ── Tier 2: LCM 4-step ────────────────────────────────────────────────
    print("[4tier] Tier 2: LCM 4-step")
    apply_lcm_mode(pipe)
    t0 = time.monotonic()
    imgs = gen_batch(pipe, PROMPTS, n_steps=LCM_STEPS, guidance=LCM_GUIDANCE)
    per = (time.monotonic() - t0) / len(PROMPTS)
    print(f"  {per:.1f}s/img")
    rows.append(("LCM fast", f"4-step LCMScheduler  {per:.1f}s/img", imgs))

    pipe.to("cpu"); del pipe; torch.cuda.empty_cache()

    # ── Tier 3: 8-bit INT8 ────────────────────────────────────────────────
    print("[4tier] Tier 3: 8-bit INT8 quantized")
    pipe_q = load_sd21_quantized(bits=8)
    pipe_q.scheduler = DPMSolverMultistepScheduler.from_config(pipe_q.scheduler.config)
    pipe_q.set_progress_bar_config(disable=True)
    t0 = time.monotonic()
    imgs = gen_batch(pipe_q, PROMPTS, n_steps=30, guidance=7.5, device="cpu")
    per = (time.monotonic() - t0) / len(PROMPTS)
    peak_vram = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  {per:.1f}s/img  |  {peak_vram:.0f} MB peak VRAM")
    rows.append(("8-bit INT8", f"30-step  {per:.1f}s/img  {peak_vram:.0f} MB VRAM", imgs))
    del pipe_q; torch.cuda.empty_cache()

    # ── Tier 4: SDXL Turbo ────────────────────────────────────────────────
    print(f"[4tier] Tier 4: SDXL Turbo  (downloads ~6.7 GB on first run)")
    pipe_t = AutoPipelineForText2Image.from_pretrained(
        TURBO_MODEL_ID, torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")
    pipe_t.set_progress_bar_config(disable=True)
    t0 = time.monotonic()
    imgs = []
    for prompt in PROMPTS:
        img, _ = generate_turbo(pipe_t, prompt, seed=SEED, width=IMG_SIZE, height=IMG_SIZE)
        imgs.append(img)
    per = (time.monotonic() - t0) / len(PROMPTS)
    print(f"  {per:.1f}s/img")
    rows.append(("SDXL Turbo", f"1-step ADD  {per:.1f}s/img", imgs))
    pipe_t.to("cpu"); del pipe_t; torch.cuda.empty_cache()

    # ── assemble + save ───────────────────────────────────────────────────
    grid = build_grid(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    grid.save(str(OUT_PATH))
    mb = OUT_PATH.stat().st_size / 1e6
    print(f"\n[4tier] Saved: {OUT_PATH}  ({grid.size[0]}×{grid.size[1]} px, {mb:.1f} MB)")


if __name__ == "__main__":
    main()
