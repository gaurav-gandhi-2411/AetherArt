"""
Gallery candidate generator — 6 capability categories, 8-12 seeds each.
Saves every candidate + picks nothing automatically (human reviews output).

Usage:
    conda run -n aetherart python scripts/generate_gallery.py
"""

from __future__ import annotations

import atexit
import json
import time
from pathlib import Path

import torch
from PIL import Image

from aetherart.gpu_hygiene import cleanup_gpu

atexit.register(cleanup_gpu)

# ── constants ───────────────────────────────────────────────────────────────

OUT = Path("docs/gallery_candidates")
OUT.mkdir(exist_ok=True)

SD21_MODEL = "sd2-community/stable-diffusion-2-1"
SDXL_MODEL = "stabilityai/sdxl-turbo"
LORA_PATH = "data/lora/ukiyo-e/ukiyo-e-lora.safetensors"

NEG = (
    "blurry, low quality, deformed, distorted, ugly, bad anatomy, "
    "watermark, signature, text, cropped, out of frame"
)
GUIDANCE = 7.5
STEPS_SD = 50
SEEDS = [42, 1337, 7777, 2024, 9999, 314, 888, 5050]

# ── helpers ─────────────────────────────────────────────────────────────────


def _timer():
    t = time.time()
    return lambda: round(time.time() - t, 1)


def _save(img: Image.Image, meta: dict, path: Path):
    img.save(path)
    path.with_suffix(".json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"    saved {path.name}  ({meta['generation_time_seconds']}s)")


def _load_sd21():
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

    print("Loading SD 2.1...")
    pipe = StableDiffusionPipeline.from_pretrained(
        SD21_MODEL,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    return pipe


def _gen_sd21(
    pipe, prompt: str, seed: int, width=768, height=768, steps=STEPS_SD, guidance=GUIDANCE
) -> tuple[Image.Image, float]:
    elapsed = _timer()
    img = pipe(
        prompt,
        negative_prompt=NEG,
        num_inference_steps=steps,
        guidance_scale=guidance,
        height=height,
        width=width,
        generator=torch.Generator("cuda").manual_seed(seed),
    ).images[0]
    return img, elapsed()


def _batch(pipe, category: str, prompt: str, seeds: list[int], **kw) -> list[Path]:
    cat_dir = OUT / category
    cat_dir.mkdir(exist_ok=True)
    paths = []
    for seed in seeds:
        img, t = _gen_sd21(pipe, prompt, seed, **kw)
        meta = dict(
            prompt=prompt,
            negative_prompt=NEG,
            seed=seed,
            steps=kw.get("steps", STEPS_SD),
            scheduler="DPM-Solver++",
            guidance_scale=kw.get("guidance", GUIDANCE),
            width=kw.get("width", 768),
            height=kw.get("height", 768),
            model=SD21_MODEL,
            generation_time_seconds=t,
            device="RTX 3070 8GB",
            dtype="fp16",
        )
        p = cat_dir / f"seed{seed}.png"
        _save(img, meta, p)
        paths.append(p)
    return paths


# ── CATEGORY 1: Hero ────────────────────────────────────────────────────────


def gen_hero(pipe):
    print("\n=== CATEGORY 1: Hero (multiple prompts) ===")
    prompts = {
        "fuji_blossom": (
            "majestic Mount Fuji at golden hour, reflections in a perfectly still lake, "
            "foreground cherry blossom branches, traditional Japanese woodblock print "
            "fused with photorealism, ultra detailed, cinematic, masterpiece"
        ),
        "samurai_dawn": (
            "an ancient samurai meditating on a cliff at dawn, swirling cherry blossom petals, "
            "atmospheric mist, traditional ink painting style merged with cinematic photography, "
            "ultra detailed, dramatic golden light, masterpiece"
        ),
        "astronaut_nebula": (
            "a lone astronaut floating through a nebula made of cherry blossoms, "
            "cosmic, ethereal, surreal masterpiece, ultra detailed, highly atmospheric, "
            "dreamlike, vivid colors"
        ),
    }
    for slug, prompt in prompts.items():
        print(f"  prompt: {slug}")
        _batch(pipe, f"01_hero_{slug}", prompt, SEEDS[:6])


# ── CATEGORY 2: Standard fp16 ───────────────────────────────────────────────


def gen_standard(pipe):
    print("\n=== CATEGORY 2: Standard fp16 ===")
    prompts = {
        "wizard_candlelight": (
            "a wise old wizard reading an ancient leather-bound book by candlelight, "
            "intricate magical symbols floating around him, warm golden light, "
            "photorealistic fantasy, ultra detailed, dramatic shadows"
        ),
        "dragon_crystal": (
            "a majestic dragon coiled around a crystal mountain peak, glowing iridescent scales, "
            "atmospheric mist, dramatic cinematic lighting, fantasy art masterpiece, ultra detailed"
        ),
    }
    for slug, prompt in prompts.items():
        print(f"  prompt: {slug}")
        _batch(pipe, f"02_standard_{slug}", prompt, SEEDS)


# ── CATEGORY 3: Ukiyo-e LoRA ────────────────────────────────────────────────


def gen_lora(pipe):
    print("\n=== CATEGORY 3: Ukiyo-e LoRA ===")
    from peft import PeftModel  # noqa: F401 — ensure peft available

    # Load LoRA weights
    try:
        pipe.load_lora_weights(LORA_PATH, adapter_name="ukiyo_e")
        pipe.set_adapters(["ukiyo_e"], adapter_weights=[1.0])
        lora_loaded = True
        print("  LoRA loaded from", LORA_PATH)
    except Exception as e:
        print(f"  LoRA load failed ({e}) — generating without LoRA for fallback")
        lora_loaded = False

    trigger = "ukyowood" if lora_loaded else ""

    prompts = {
        "fuji_waves": (
            f"{trigger} Mount Fuji with crashing waves and cherry blossoms in the style of Hokusai, "  # noqa: E501
            "traditional Japanese woodblock print, vibrant colors, bold outlines, masterpiece"
        ),
        "samurai_sunset": (
            f"{trigger} a samurai warrior in flowing silk robes against a blazing sunset, "
            "traditional ukiyo-e woodblock style, bold graphic lines, rich colors, atmospheric"
        ),
    }

    cat_dir = OUT / "03_lora"
    cat_dir.mkdir(exist_ok=True)

    for slug, prompt in prompts.items():
        print(f"  prompt: {slug}")
        for seed in SEEDS:
            img, t = _gen_sd21(pipe, prompt, seed)
            meta = dict(
                prompt=prompt,
                negative_prompt=NEG,
                seed=seed,
                steps=STEPS_SD,
                scheduler="DPM-Solver++",
                guidance_scale=GUIDANCE,
                width=768,
                height=768,
                model=SD21_MODEL,
                lora=LORA_PATH if lora_loaded else None,
                generation_time_seconds=t,
                device="RTX 3070 8GB",
                dtype="fp16",
            )
            p = cat_dir / f"{slug}_seed{seed}.png"
            _save(img, meta, p)

    if lora_loaded:
        pipe.unload_lora_weights()


# ── CATEGORY 4: ControlNet Canny ────────────────────────────────────────────


def gen_canny(pipe):
    print("\n=== CATEGORY 4: ControlNet Canny ===")
    import cv2
    import numpy as np
    from diffusers import (
        ControlNetModel,
        DPMSolverMultistepScheduler,
        StableDiffusionControlNetPipeline,
    )

    cn_id = "thibaud/controlnet-sd21-canny-diffusers"
    print(f"  Loading ControlNet ({cn_id})...")
    controlnet = ControlNetModel.from_pretrained(cn_id, torch_dtype=torch.float16)
    cn_pipe = StableDiffusionControlNetPipeline.from_pretrained(
        SD21_MODEL,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    cn_pipe.scheduler = DPMSolverMultistepScheduler.from_config(cn_pipe.scheduler.config)
    cn_pipe = cn_pipe.to("cuda")
    cn_pipe.enable_attention_slicing()

    cat_dir = OUT / "04_canny"
    cat_dir.mkdir(exist_ok=True)

    # Source image: use existing controlnet sample source
    src_path = Path("docs/samples/controlnet_canny/00_source.png")
    if not src_path.exists():
        print(f"  Source not found at {src_path} — skipping canny")
        return

    src_img = Image.open(src_path).convert("RGB").resize((768, 768))
    src_img.save(cat_dir / "source.png")

    arr = np.array(src_img)
    edges = cv2.Canny(arr, 100, 200)
    canny_map = Image.fromarray(np.stack([edges] * 3, axis=-1))
    canny_map.save(cat_dir / "canny_map.png")

    prompt = (
        "an ornate ancient temple in mystical mountain mist, fantasy art, "
        "ultra detailed, atmospheric, dramatic lighting, cinematic"
    )

    for seed in SEEDS:
        elapsed = _timer()
        img = cn_pipe(
            prompt,
            negative_prompt=NEG,
            image=canny_map,
            num_inference_steps=STEPS_SD,
            guidance_scale=GUIDANCE,
            height=768,
            width=768,
            generator=torch.Generator("cuda").manual_seed(seed),
            controlnet_conditioning_scale=1.0,
        ).images[0]
        t = elapsed()
        meta = dict(
            prompt=prompt,
            negative_prompt=NEG,
            seed=seed,
            steps=STEPS_SD,
            scheduler="DPM-Solver++",
            guidance_scale=GUIDANCE,
            width=768,
            height=768,
            model=SD21_MODEL,
            controlnet="canny",
            controlnet_source=str(src_path),
            generation_time_seconds=t,
            device="RTX 3070 8GB",
            dtype="fp16",
        )
        p = cat_dir / f"seed{seed}.png"
        _save(img, meta, p)

    del cn_pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── CATEGORY 5: ControlNet Depth ────────────────────────────────────────────


def gen_depth(pipe):
    print("\n=== CATEGORY 5: ControlNet Depth ===")
    from diffusers import (
        ControlNetModel,
        DPMSolverMultistepScheduler,
        StableDiffusionControlNetPipeline,
    )

    cn_id = "thibaud/controlnet-sd21-depth-diffusers"
    print(f"  Loading ControlNet ({cn_id})...")
    controlnet = ControlNetModel.from_pretrained(cn_id, torch_dtype=torch.float16)
    cn_pipe = StableDiffusionControlNetPipeline.from_pretrained(
        SD21_MODEL,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    cn_pipe.scheduler = DPMSolverMultistepScheduler.from_config(cn_pipe.scheduler.config)
    cn_pipe = cn_pipe.to("cuda")
    cn_pipe.enable_attention_slicing()

    cat_dir = OUT / "05_depth"
    cat_dir.mkdir(exist_ok=True)

    src_path = Path("docs/samples/controlnet_depth/00_source.png")
    if not src_path.exists():
        print(f"  Source not found at {src_path} — skipping depth")
        return

    src_img = Image.open(src_path).convert("RGB").resize((768, 768))
    src_img.save(cat_dir / "source.png")
    depth_map = (
        Image.open("docs/samples/controlnet_depth/00_depth_map.png")
        .convert("RGB")
        .resize((768, 768))
    )
    depth_map.save(cat_dir / "depth_map.png")

    prompt = (
        "a futuristic neon-lit Asian metropolis at night, cyberpunk aesthetic, "
        "rain-slicked streets reflecting holographic advertisements, ultra detailed, cinematic"
    )

    for seed in SEEDS:
        elapsed = _timer()
        img = cn_pipe(
            prompt,
            negative_prompt=NEG,
            image=depth_map,
            num_inference_steps=STEPS_SD,
            guidance_scale=GUIDANCE,
            height=768,
            width=768,
            generator=torch.Generator("cuda").manual_seed(seed),
            controlnet_conditioning_scale=1.0,
        ).images[0]
        t = elapsed()
        meta = dict(
            prompt=prompt,
            negative_prompt=NEG,
            seed=seed,
            steps=STEPS_SD,
            scheduler="DPM-Solver++",
            guidance_scale=GUIDANCE,
            width=768,
            height=768,
            model=SD21_MODEL,
            controlnet="depth",
            controlnet_source=str(src_path),
            generation_time_seconds=t,
            device="RTX 3070 8GB",
            dtype="fp16",
        )
        p = cat_dir / f"seed{seed}.png"
        _save(img, meta, p)

    del cn_pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── CATEGORY 6: SDXL Turbo ──────────────────────────────────────────────────


def gen_turbo():
    print("\n=== CATEGORY 6: SDXL Turbo ===")
    from diffusers import AutoPipelineForText2Image

    local_dir = Path("models/sdxl-turbo")
    src = (
        str(local_dir)
        if (local_dir / "unet" / "diffusion_pytorch_model.fp16.safetensors").exists()
        else SDXL_MODEL
    )
    print(f"  Loading SDXL Turbo from {src}...")

    turbo = AutoPipelineForText2Image.from_pretrained(
        src, torch_dtype=torch.float16, variant="fp16"
    )
    turbo.enable_model_cpu_offload()

    cat_dir = OUT / "06_turbo"
    cat_dir.mkdir(exist_ok=True)

    prompts = {
        "cosmic_library": (
            "a cosmic library with infinite shelves of glowing ancient books, "
            "ethereal light beams, mystical atmosphere, ultra detailed, dreamlike"
        ),
        "bioluminescent_city": (
            "an underwater city of bioluminescent coral and ancient ruins, "
            "mystical sea creatures, divine light rays, ultra detailed fantasy, epic"
        ),
    }

    for slug, prompt in prompts.items():
        print(f"  prompt: {slug}")
        for seed in SEEDS:
            elapsed = _timer()
            img = turbo(
                prompt=prompt,
                num_inference_steps=1,
                guidance_scale=0.0,
                height=512,
                width=512,
                generator=torch.Generator("cuda").manual_seed(seed),
            ).images[0]
            t = elapsed()
            meta = dict(
                prompt=prompt,
                negative_prompt="",
                seed=seed,
                steps=1,
                scheduler="ADD-1step (Turbo)",
                guidance_scale=0.0,
                width=512,
                height=512,
                model="stabilityai/sdxl-turbo",
                generation_time_seconds=t,
                device="RTX 3070 8GB",
                dtype="fp16",
            )
            p = cat_dir / f"{slug}_seed{seed}.png"
            _save(img, meta, p)

    del turbo
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipe = None
    try:
        t_total = _timer()

        # SD 2.1 pipe shared across cats 1-3; ControlNet loads its own
        pipe = _load_sd21()

        gen_hero(pipe)
        gen_standard(pipe)
        gen_lora(pipe)

        # Unload SD 2.1 to free VRAM for ControlNet pipelines
        del pipe
        pipe = None
        cleanup_gpu(verbose=True)

        gen_canny(None)
        gen_depth(None)

        # SDXL Turbo last (separate pipeline)
        gen_turbo()

        print(f"\n=== DONE — total GPU time: {t_total()}s ===")
        print(f"Candidates in: {OUT}/")
        print("Review each category subfolder and pick your favourite.")
    finally:
        if pipe is not None:
            del pipe
        cleanup_gpu(verbose=True)
