#!/usr/bin/env python
"""
Benchmark SD 2.1 at fp16, 8-bit, and 4-bit quantization.

Measures per-image latency, VRAM peak, and CLIP image-text alignment score.
Output: reports/quantization_benchmark.md

Usage:
    python scripts/benchmark_quantization.py
"""
import time
from pathlib import Path

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "reports" / "quantization_benchmark.md"
MODEL_ID = "sd2-community/stable-diffusion-2-1"

PROMPTS = [
    "a samurai warrior on horseback, dramatic lighting, cinematic",
    "a tea house at the base of Mount Fuji at dusk",
    "a tiger prowling through a bamboo forest",
    "cherry blossoms falling over a stone bridge at night",
]
STEPS = 30
GUIDANCE = 7.5
SEED = 42
IMG_SIZE = 512


def measure_clip_score(images: list, prompts: list) -> float:
    """Return mean CLIP cosine similarity for the image-text pairs."""
    try:
        import clip

        model, preprocess = clip.load("ViT-B/32", device="cuda")
        import numpy as np

        scores = []
        for img, text in zip(images, prompts):
            img_t = preprocess(img).unsqueeze(0).to("cuda")
            tok = clip.tokenize([text], truncate=True).to("cuda")
            with torch.no_grad():
                img_f = model.encode_image(img_t)
                text_f = model.encode_text(tok)
                img_f = img_f / img_f.norm(dim=-1, keepdim=True)
                text_f = text_f / text_f.norm(dim=-1, keepdim=True)
                scores.append((img_f * text_f).sum().item())
        return float(np.mean(scores))
    except Exception:
        return float("nan")


def run_benchmark(label: str, pipe, device: str = "cuda") -> dict:
    pipe.set_progress_bar_config(disable=True)
    torch.cuda.reset_peak_memory_stats()

    images, latencies = [], []
    for prompt in PROMPTS:
        t0 = time.monotonic()
        img = pipe(
            prompt,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            generator=torch.Generator(device=device).manual_seed(SEED),
            height=IMG_SIZE,
            width=IMG_SIZE,
        ).images[0]
        latencies.append(time.monotonic() - t0)
        images.append(img)

    avg_lat = sum(latencies) / len(latencies)
    peak_vram = torch.cuda.max_memory_allocated() / 1024**2
    clip_score = measure_clip_score(images, PROMPTS)

    print(
        f"  {label}: {avg_lat:.1f}s/img  |  {peak_vram:.0f} MB peak VRAM  |  CLIP={clip_score:.3f}"
    )
    return {"label": label, "avg_lat": avg_lat, "peak_vram": peak_vram, "clip": clip_score}


def load_fp16() -> "StableDiffusionPipeline":
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe


def load_quantized(bits: int) -> "StableDiffusionPipeline":
    from aetherart.quantization import load_sd21_quantized

    pipe = load_sd21_quantized(bits=bits)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe


def write_report(results: list[dict]) -> None:
    fp16 = next(r for r in results if r["label"] == "fp16")

    lines = [
        "# Quantization Benchmark — SD 2.1 U-Net",
        "",
        f"**Model**: `{MODEL_ID}`  ",
        f"**Steps**: {STEPS}  **Guidance**: {GUIDANCE}  **Seed**: {SEED}  ",
        f"**Prompts**: {len(PROMPTS)}  **Resolution**: {IMG_SIZE}×{IMG_SIZE}",
        "",
        "| Mode | Avg latency (s/img) | Peak VRAM (MB) | CLIP score | VRAM vs fp16 |",
        "|------|--------------------:|---------------:|:----------:|:------------|",
    ]
    for r in results:
        vram_delta = f"{r['peak_vram'] - fp16['peak_vram']:+.0f} MB"
        clip_str = f"{r['clip']:.3f}" if r["clip"] == r["clip"] else "n/a"
        lines.append(
            f"| {r['label']} | {r['avg_lat']:.1f} | {r['peak_vram']:.0f} | {clip_str} | {vram_delta} |"  # noqa: E501
        )

    lines += [
        "",
        "## Notes",
        "",
        "- Quantization applied to **U-Net only**; text encoder and VAE remain at fp16.",
        "- 4-bit (NF4) enables SD 2.1 on GPUs with ≥ 4 GB VRAM.",
        "- Latency overhead is bitsandbytes dequantization cost; varies by GPU.",
        "- CLIP score uses `openai/clip-vit-base-patch32`; N/A if `clip` package absent.",
    ]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[quant] Report saved: {OUT_PATH}")


def main() -> None:
    results = []

    print("[quant] fp16 baseline")
    pipe = load_fp16()
    results.append(run_benchmark("fp16", pipe))
    pipe.to("cpu")
    del pipe
    torch.cuda.empty_cache()

    print("[quant] 8-bit INT8")
    pipe = load_quantized(8)
    results.append(run_benchmark("8-bit INT8", pipe, device="cpu"))
    del pipe
    torch.cuda.empty_cache()

    print("[quant] 4-bit NF4")
    pipe = load_quantized(4)
    results.append(run_benchmark("4-bit NF4", pipe, device="cpu"))
    del pipe
    torch.cuda.empty_cache()

    write_report(results)


if __name__ == "__main__":
    main()
