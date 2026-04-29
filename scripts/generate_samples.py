#!/usr/bin/env python
"""Generate sample images covering every feature tier for the Sample Outputs gallery.

Outputs: docs/samples/{tier}/{n:02d}_{slug}.png + .json

Usage (from repo root, aetherart conda env, requires CUDA GPU):
    python scripts/generate_samples.py

Estimated time on RTX 3070 8 GB: ~15 minutes (all tiers).
"""
import json
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

from aetherart.lcm import apply_lcm_mode, LCM_STEPS, LCM_GUIDANCE
from aetherart.lora import load_lora
from aetherart.quantization import load_sd21_quantized
from aetherart import controlnet as cn

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = REPO_ROOT / "docs" / "samples"
SD21_ID = "sd2-community/stable-diffusion-2-1"
SEED = 42
IMG_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NEGATIVE = "blurry, low quality, artifacts, watermark"

PROMPTS_4 = [
    ("samurai",         "a lone samurai standing in a misty forest, dramatic lighting, cinematic"),
    ("mountain_lake",   "a serene mountain lake at sunrise, reflected peaks, peaceful atmosphere"),
    ("cyberpunk_city",  "a futuristic city at night, neon lights, rain-slicked streets, cyberpunk"),
    ("cherry_blossoms", "cherry blossoms over a stone bridge, spring morning, soft golden light"),
]
PROMPTS_2 = PROMPTS_4[:2]


def _reset_vram():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _peak_vram() -> float:
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / 1024**2, 0)
    return 0.0


def _gen(device=None):
    d = device or DEVICE
    return torch.Generator(device=d).manual_seed(SEED)


def save_sample(img: Image.Image, meta: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path), format="PNG")
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    t = meta.get("inference_time_rtx3070_s", "?")
    v = meta.get("vram_peak_mb", "?")
    print(f"  {path.name}  [{t}s / {v} MB VRAM]")


def free_pipe(pipe):
    try:
        pipe.to("cpu")
    except Exception:
        pass
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def make_canny_source() -> Image.Image:
    """Synthetic conditioning image: building silhouette + moon for clear Canny edges."""
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (80, 90, 110))
    draw = ImageDraw.Draw(img)
    draw.rectangle([(70, 180), (200, IMG_SIZE)],  fill=(55, 65, 85))
    draw.rectangle([(210, 230), (380, IMG_SIZE)], fill=(45, 55, 75))
    draw.rectangle([(390, 260), (460, IMG_SIZE)], fill=(50, 60, 80))
    draw.ellipse([(340, 40), (430, 130)], fill=(225, 215, 175))
    draw.line([(0, 360), (IMG_SIZE, 360)], fill=(130, 140, 160), width=2)
    return img


def make_depth_source() -> Image.Image:
    """Synthetic depth gradient: light top (far) → dark bottom (near)."""
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
    pixels = img.load()
    for y in range(IMG_SIZE):
        val = int(210 * (1 - y / IMG_SIZE) + 20)
        for x in range(IMG_SIZE):
            pixels[x, y] = (val, val, val)
    return img


# ── Tier 1: Standard fp16 ─────────────────────────────────────────────────

def generate_standard_fp16():
    print("\n[samples] Standard fp16 — 30-step DPM++")
    pipe = StableDiffusionPipeline.from_pretrained(SD21_ID, torch_dtype=torch.float16).to(DEVICE)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    tier_dir = SAMPLES_DIR / "standard_fp16"
    for i, (slug, prompt) in enumerate(PROMPTS_4, 1):
        _reset_vram()
        t0 = time.time()
        img = pipe(prompt, negative_prompt=NEGATIVE, num_inference_steps=30, guidance_scale=7.5,
                   generator=_gen(), height=IMG_SIZE, width=IMG_SIZE).images[0]
        meta = dict(prompt=prompt, negative_prompt=NEGATIVE, seed=SEED,
                    tier="standard_fp16", steps=30, guidance=7.5,
                    inference_time_rtx3070_s=round(time.time() - t0, 2),
                    vram_peak_mb=_peak_vram())
        save_sample(img, meta, tier_dir / f"{i:02d}_{slug}.png")
    free_pipe(pipe)


# ── Tier 2: LCM 4-step ───────────────────────────────────────────────────

def generate_lcm():
    print(f"\n[samples] LCM {LCM_STEPS}-step LCMScheduler (guidance={LCM_GUIDANCE})")
    pipe = StableDiffusionPipeline.from_pretrained(SD21_ID, torch_dtype=torch.float16).to(DEVICE)
    apply_lcm_mode(pipe)
    pipe.set_progress_bar_config(disable=True)

    tier_dir = SAMPLES_DIR / "lcm"
    for i, (slug, prompt) in enumerate(PROMPTS_4, 1):
        _reset_vram()
        t0 = time.time()
        img = pipe(prompt, num_inference_steps=LCM_STEPS, guidance_scale=LCM_GUIDANCE,
                   generator=_gen(), height=IMG_SIZE, width=IMG_SIZE).images[0]
        meta = dict(prompt=prompt, seed=SEED,
                    tier="lcm", steps=LCM_STEPS, guidance=LCM_GUIDANCE,
                    inference_time_rtx3070_s=round(time.time() - t0, 2),
                    vram_peak_mb=_peak_vram())
        save_sample(img, meta, tier_dir / f"{i:02d}_{slug}.png")
    free_pipe(pipe)


# ── Tier 3: SDXL Turbo ───────────────────────────────────────────────────

def generate_turbo():
    print("\n[samples] SDXL Turbo — 1-step ADD")
    from aetherart.sdxl_turbo import load_turbo_pipeline, generate_turbo as _turbo_gen
    pipe = load_turbo_pipeline()
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)

    tier_dir = SAMPLES_DIR / "turbo"
    for i, (slug, prompt) in enumerate(PROMPTS_4, 1):
        _reset_vram()
        t0 = time.time()
        img, _ = _turbo_gen(pipe, prompt=prompt, seed=SEED, width=IMG_SIZE, height=IMG_SIZE)
        meta = dict(prompt=prompt, seed=SEED,
                    tier="turbo", model="stabilityai/sdxl-turbo", steps=1, guidance=0.0,
                    inference_time_rtx3070_s=round(time.time() - t0, 2),
                    vram_peak_mb=_peak_vram())
        save_sample(img, meta, tier_dir / f"{i:02d}_{slug}.png")
    free_pipe(pipe)


# ── Tier 4: Ukiyo-e LoRA ─────────────────────────────────────────────────

def generate_lora_ukiyo_e():
    print("\n[samples] Ukiyo-e LoRA — rank-8, checkpoint-1000")
    pipe = StableDiffusionPipeline.from_pretrained(SD21_ID, torch_dtype=torch.float16).to(DEVICE)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    load_lora(pipe, "ukiyo-e", alpha=1.0)
    pipe.set_progress_bar_config(disable=True)

    trigger = "ukyowood"
    negative = "text, watermark, calligraphy, signature, words, letters, blurry"
    tier_dir = SAMPLES_DIR / "lora_ukiyo_e"
    for i, (slug, prompt) in enumerate(PROMPTS_4, 1):
        _reset_vram()
        lora_prompt = f"{trigger} {prompt}"
        t0 = time.time()
        img = pipe(lora_prompt, negative_prompt=negative, num_inference_steps=30, guidance_scale=7.5,
                   generator=_gen(), height=IMG_SIZE, width=IMG_SIZE).images[0]
        meta = dict(prompt=lora_prompt, original_prompt=prompt, negative_prompt=negative,
                    seed=SEED, tier="lora_ukiyo_e", lora="ukiyo-e", lora_alpha=1.0,
                    steps=30, guidance=7.5,
                    inference_time_rtx3070_s=round(time.time() - t0, 2),
                    vram_peak_mb=_peak_vram())
        save_sample(img, meta, tier_dir / f"{i:02d}_{slug}.png")
    free_pipe(pipe)


# ── Tier 5: ControlNet Canny ─────────────────────────────────────────────

def generate_controlnet_canny():
    print("\n[samples] ControlNet Canny — edge conditioning")
    tier_dir = SAMPLES_DIR / "controlnet_canny"
    tier_dir.mkdir(parents=True, exist_ok=True)

    source = make_canny_source()
    source.save(str(tier_dir / "00_source.png"))
    ctrl_map = cn.preprocess(source, "canny", canny_low=100, canny_high=200)
    ctrl_map.save(str(tier_dir / "00_canny_map.png"))
    print("  Conditioning source + Canny map saved.")

    canny_prompts = [
        ("city_night",       "a futuristic cyberpunk city at night, neon signs, rain-wet streets"),
        ("japanese_temple",  "a Japanese temple complex at sunset, dramatic clouds, lanterns lit"),
    ]

    pipe = cn.get_pipeline("canny")
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)

    for i, (slug, prompt) in enumerate(canny_prompts, 1):
        _reset_vram()
        t0 = time.time()
        out = pipe(prompt, image=ctrl_map, negative_prompt=NEGATIVE,
                   num_inference_steps=30, guidance_scale=7.5,
                   generator=_gen(), height=IMG_SIZE, width=IMG_SIZE,
                   controlnet_conditioning_scale=1.0)
        img = out.images[0]
        meta = dict(prompt=prompt, seed=SEED, tier="controlnet_canny",
                    steps=30, guidance=7.5, conditioning_scale=1.0,
                    inference_time_rtx3070_s=round(time.time() - t0, 2),
                    vram_peak_mb=_peak_vram())
        save_sample(img, meta, tier_dir / f"{i:02d}_{slug}.png")


# ── Tier 6: ControlNet Depth ─────────────────────────────────────────────

def generate_controlnet_depth():
    print("\n[samples] ControlNet Depth — depth conditioning")
    tier_dir = SAMPLES_DIR / "controlnet_depth"
    tier_dir.mkdir(parents=True, exist_ok=True)

    source = make_depth_source()
    source.save(str(tier_dir / "00_source.png"))
    # Use the synthetic gradient directly — it's already a depth-appropriate greyscale map.
    # Bypasses Intel/dpt-hybrid-midas which requires torch >= 2.6 for its pickle weights.
    ctrl_map = source.convert("RGB")
    ctrl_map.save(str(tier_dir / "00_depth_map.png"))
    print("  Depth source + depth map saved.")

    depth_prompts = [
        ("mountain_valley", "a lush mountain valley at dawn, mist rolling through the trees, sunrise"),
        ("misty_forest",    "an ancient misty forest, massive trees, shafts of sunlight through fog"),
    ]

    pipe = cn.get_pipeline("depth")
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)

    for i, (slug, prompt) in enumerate(depth_prompts, 1):
        _reset_vram()
        t0 = time.time()
        out = pipe(prompt, image=ctrl_map, negative_prompt=NEGATIVE,
                   num_inference_steps=30, guidance_scale=7.5,
                   generator=_gen(), height=IMG_SIZE, width=IMG_SIZE,
                   controlnet_conditioning_scale=1.0)
        img = out.images[0]
        meta = dict(prompt=prompt, seed=SEED, tier="controlnet_depth",
                    steps=30, guidance=7.5, conditioning_scale=1.0,
                    inference_time_rtx3070_s=round(time.time() - t0, 2),
                    vram_peak_mb=_peak_vram())
        save_sample(img, meta, tier_dir / f"{i:02d}_{slug}.png")


# ── Tier 7: 8-bit INT8 quantized ─────────────────────────────────────────

def generate_quantized_8bit():
    print("\n[samples] 8-bit INT8 quantized U-Net")
    pipe = load_sd21_quantized(bits=8)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    tier_dir = SAMPLES_DIR / "quantized_8bit"
    for i, (slug, prompt) in enumerate(PROMPTS_2, 1):
        _reset_vram()
        t0 = time.time()
        img = pipe(prompt, negative_prompt=NEGATIVE, num_inference_steps=30, guidance_scale=7.5,
                   generator=_gen("cpu"), height=IMG_SIZE, width=IMG_SIZE).images[0]
        meta = dict(prompt=prompt, seed=SEED, tier="quantized_8bit",
                    steps=30, guidance=7.5, precision="8bit_int8",
                    inference_time_rtx3070_s=round(time.time() - t0, 2),
                    vram_peak_mb=_peak_vram())
        save_sample(img, meta, tier_dir / f"{i:02d}_{slug}.png")


# ── Tier 8: 4-bit NF4 quantized ──────────────────────────────────────────

def generate_quantized_4bit():
    print("\n[samples] 4-bit NF4 quantized U-Net")
    pipe = load_sd21_quantized(bits=4)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    tier_dir = SAMPLES_DIR / "quantized_4bit"
    for i, (slug, prompt) in enumerate(PROMPTS_2, 1):
        _reset_vram()
        t0 = time.time()
        img = pipe(prompt, negative_prompt=NEGATIVE, num_inference_steps=30, guidance_scale=7.5,
                   generator=_gen("cpu"), height=IMG_SIZE, width=IMG_SIZE).images[0]
        meta = dict(prompt=prompt, seed=SEED, tier="quantized_4bit",
                    steps=30, guidance=7.5, precision="4bit_nf4",
                    inference_time_rtx3070_s=round(time.time() - t0, 2),
                    vram_peak_mb=_peak_vram())
        save_sample(img, meta, tier_dir / f"{i:02d}_{slug}.png")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("WARNING: no CUDA GPU detected. Generation will be slow.")

    generate_standard_fp16()
    generate_lcm()
    generate_turbo()
    generate_lora_ukiyo_e()
    generate_controlnet_canny()
    generate_controlnet_depth()
    generate_quantized_8bit()
    generate_quantized_4bit()

    total = sum(len(list((SAMPLES_DIR / t).glob("*.png")))
                for t in ["standard_fp16", "lcm", "turbo", "lora_ukiyo_e",
                           "controlnet_canny", "controlnet_depth",
                           "quantized_8bit", "quantized_4bit"]
                if (SAMPLES_DIR / t).is_dir())
    print(f"\n[samples] Done. {total} sample images in {SAMPLES_DIR}")
