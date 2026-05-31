"""Visual evaluation script for SDXL Ukiyo-e LoRA checkpoint selection.

Generates 4 images per checkpoint at fixed seed 42 across 4 prompts.
Saves to reports/lora_checkpoint_eval_sdxl/ for visual comparison.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, StableDiffusionXLPipeline
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = REPO_ROOT / "data" / "lora" / "ukiyo-e" / "training_output_sdxl" / "checkpoints"
OUTPUT_DIR = REPO_ROOT / "reports" / "lora_checkpoint_eval_sdxl"
SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_MODEL = "madebyollin/sdxl-vae-fp16-fix"

PROMPTS = [
    "ukyowood ukiyo-e woodblock print of Mount Fuji at sunset",
    "ukyowood ukiyo-e print of a crane over ocean waves",
    "ukyowood ukiyo-e woodblock print of a samurai in a bamboo forest",
    "ukyowood ukiyo-e print of cherry blossoms along a river",
]
NEGATIVE_PROMPT = (
    "text, watermark, calligraphy, writing, letters, words, signature, "
    "blurry, low quality, western art, photograph, 3d render"
)
SEED = 42
STEPS = 25
GUIDANCE = 7.5
SIZE = 1024

# Evaluate these three checkpoints for selection (250/750/1250 skipped)
EVAL_CHECKPOINTS = ["checkpoint-500", "checkpoint-1000", "checkpoint-1500"]


def load_base_pipeline() -> StableDiffusionXLPipeline:
    vae = AutoencoderKL.from_pretrained(VAE_MODEL, torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        SDXL_MODEL,
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    # Do NOT move to CUDA yet — LoRA will be fused first, then offload enabled
    return pipe


def generate_grid(pipe: StableDiffusionXLPipeline, lora_path: str, ckpt_name: str) -> None:
    out_dir = OUTPUT_DIR / ckpt_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if lora_path:
        pipe.load_lora_weights(lora_path, adapter_name="ukiyo_sdxl")
        pipe.set_adapters(["ukiyo_sdxl"], adapter_weights=[1.0])
        # Fuse LoRA weights into base model before enabling offload.
        # This removes LoRA hooks so cpu_offload runs at baseline speed.
        pipe.fuse_lora()

    pipe.enable_model_cpu_offload()

    images = []
    for i, prompt in enumerate(PROMPTS):
        gen = torch.Generator(device="cpu").manual_seed(SEED)
        t0 = time.perf_counter()
        img = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            height=SIZE,
            width=SIZE,
            generator=gen,
        ).images[0]
        lat = time.perf_counter() - t0
        arr = np.array(img)
        print(f"  [{ckpt_name}] prompt {i + 1}/4 mean={arr.mean():.1f} lat={lat:.1f}s")
        img.save(out_dir / f"prompt_{i + 1:02d}.png")
        images.append(img)

    # 2×2 grid
    grid = Image.new("RGB", (SIZE * 2, SIZE * 2))
    for idx, img in enumerate(images):
        r, c = divmod(idx, 2)
        grid.paste(img, (c * SIZE, r * SIZE))
    grid.save(OUTPUT_DIR / f"{ckpt_name}_grid.png")
    print(f"  [{ckpt_name}] grid saved: {OUTPUT_DIR / f'{ckpt_name}_grid.png'}")

    import gc

    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Baseline: base model only — skip if already generated
    baseline_grid = OUTPUT_DIR / "baseline_grid.png"
    if not baseline_grid.exists():
        print("\n[baseline] loading pipeline...")
        pipe = load_base_pipeline()
        generate_grid(pipe, "", "baseline")
    else:
        print("\n[baseline] already generated, skipping.")

    for ckpt in EVAL_CHECKPOINTS:
        grid_path = OUTPUT_DIR / f"{ckpt}_grid.png"
        if grid_path.exists():
            print(f"\n[{ckpt}] already generated, skipping.")
            continue
        lora_path = str(CHECKPOINTS_DIR / ckpt / "pytorch_lora_weights.safetensors")
        print(f"\n[{ckpt}] loading fresh pipeline + LoRA from {lora_path}...")
        pipe = load_base_pipeline()
        generate_grid(pipe, lora_path, ckpt)

    print(f"\nAll grids written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
