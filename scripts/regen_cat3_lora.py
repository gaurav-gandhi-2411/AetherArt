"""Regenerate gallery category 3 with working LoRA.

Also produces a WITH vs WITHOUT LoRA comparison pair at seed 42 for verification.
"""

from __future__ import annotations

import atexit
import json
import time
from pathlib import Path

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

from aetherart.gpu_hygiene import cleanup_gpu

atexit.register(cleanup_gpu)

LORA_PATH = "data/lora/ukiyo-e/ukiyo-e-lora.safetensors"
SD21_MODEL = "sd2-community/stable-diffusion-2-1"
OUT = Path("docs/gallery_candidates/03_lora")
OUT.mkdir(parents=True, exist_ok=True)
COMPARE_OUT = Path("docs/gallery_candidates/03_lora_comparison")
COMPARE_OUT.mkdir(exist_ok=True)

NEG = (
    "blurry, low quality, deformed, distorted, ugly, bad anatomy, "
    "watermark, signature, text, cropped, out of frame"
)
SEEDS = [42, 1337, 7777, 2024, 9999, 314, 888, 5050]

PROMPTS = {
    "fuji_waves": (
        "ukyowood Mount Fuji with crashing waves and cherry blossoms in the style of Hokusai, "
        "traditional Japanese woodblock print, vibrant colors, bold outlines, masterpiece"
    ),
    "samurai_sunset": (
        "ukyowood a samurai warrior in flowing silk robes against a blazing sunset, "
        "traditional ukiyo-e woodblock style, bold graphic lines, rich colors, atmospheric"
    ),
}

COMPARE_PROMPT = (
    "ukyowood Mount Fuji with crashing waves and cherry blossoms in the style of Hokusai, "
    "traditional Japanese woodblock print, vibrant colors, bold outlines, masterpiece"
)
COMPARE_PROMPT_NO_TRIGGER = (
    "Mount Fuji with crashing waves and cherry blossoms in the style of Hokusai, "
    "traditional Japanese woodblock print, vibrant colors, bold outlines, masterpiece"
)


def _timer():
    t = time.time()
    return lambda: round(time.time() - t, 1)


def _gen(pipe, prompt, seed):
    elapsed = _timer()
    img = pipe(
        prompt,
        negative_prompt=NEG,
        num_inference_steps=50,
        guidance_scale=7.5,
        height=768,
        width=768,
        generator=torch.Generator("cuda").manual_seed(seed),
    ).images[0]
    return img, elapsed()


def main():
    pipe = None
    try:
        print("Loading SD 2.1 fp16...")
        pipe = StableDiffusionPipeline.from_pretrained(
            SD21_MODEL,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()

        # ── Step 1: Comparison WITHOUT LoRA ─────────────────────────────────
        print("\nGenerating WITHOUT LoRA (seed 42)...")
        img_no_lora, t = _gen(pipe, COMPARE_PROMPT_NO_TRIGGER, 42)
        img_no_lora.save(COMPARE_OUT / "seed42_WITHOUT_lora.png")
        print(f"  saved seed42_WITHOUT_lora.png ({t}s)")

        # ── Step 2: Load LoRA ────────────────────────────────────────────────
        print(f"\nLoading LoRA from {LORA_PATH}...")
        pipe.load_lora_weights(LORA_PATH, adapter_name="ukiyo_e")
        pipe.set_adapters(["ukiyo_e"], adapter_weights=[1.0])
        print("  LoRA loaded, active adapters:", pipe.get_list_adapters())

        # ── Step 3: Comparison WITH LoRA ─────────────────────────────────────
        print("\nGenerating WITH LoRA (seed 42, trigger token active)...")
        img_with_lora, t = _gen(pipe, COMPARE_PROMPT, 42)
        img_with_lora.save(COMPARE_OUT / "seed42_WITH_lora.png")
        print(f"  saved seed42_WITH_lora.png ({t}s)")

        # ── Step 4: Full Cat 3 batch WITH LoRA ───────────────────────────────
        print("\n=== Generating Cat 3 batch (WITH LoRA) ===")
        for slug, prompt in PROMPTS.items():
            print(f"  prompt: {slug}")
            for seed in SEEDS:
                img, t = _gen(pipe, prompt, seed)
                meta = dict(
                    prompt=prompt,
                    negative_prompt=NEG,
                    seed=seed,
                    steps=50,
                    scheduler="DPM-Solver++",
                    guidance_scale=7.5,
                    width=768,
                    height=768,
                    model=SD21_MODEL,
                    lora=LORA_PATH,
                    adapter_name="ukiyo_e",
                    lora_weight=1.0,
                    generation_time_seconds=t,
                    device="RTX 3070 8GB",
                    dtype="fp16",
                )
                p = OUT / f"{slug}_seed{seed}.png"
                img.save(p)
                p.with_suffix(".json").write_text(
                    json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                print(f"    saved {p.name} ({t}s)")

        pipe.unload_lora_weights()
        print("\nDONE — compare images in docs/gallery_candidates/03_lora_comparison/")
        print("Full batch in docs/gallery_candidates/03_lora/")
    finally:
        if pipe is not None:
            del pipe
        cleanup_gpu(verbose=True)


if __name__ == "__main__":
    main()
