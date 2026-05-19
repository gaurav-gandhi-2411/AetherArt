"""GPU smoke test for SDXL ControlNet Union pipeline (PR 08 proof-of-landing).

Run in the aetherart conda env:
  conda run -n aetherart python scripts/smoke_controlnet_sdxl.py

Asserts:
  - canny output is not all-black (mean > 10)
  - depth output is not all-black (mean > 10)
  - VRAM peaks reported per control type
  - Latency reported per control type

Exits 0 on success and prints CONTROLNET_SDXL_SMOKE_PASSED.
Exits 1 with a traceback on any failure.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from aetherart.controlnet_sdxl import (
    generate_sdxl_controlnet,
    load_sdxl_controlnet_pipeline,
    preprocess_canny,
    release_controlnet_union_model,
    release_sdxl_controlnet_pipeline,
)

PROMPT = "a futuristic city skyline at sunset, cinematic, highly detailed"
NEG_PROMPT = "blurry, low quality, cartoon"
SEED = 42
SIZE = 1024
STEPS = 20
GUIDANCE = 7.5
COND_SCALE = 0.7


def _make_test_image(size: int = SIZE) -> Image.Image:
    """Gradient image that produces non-trivial canny edges and depth map."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(size):
        arr[i, :, :] = int(i / size * 200) + 30
    for j in range(size):
        arr[:, j, 0] = np.clip(arr[:, j, 0].astype(int) + int(j / size * 80), 0, 255)
    return Image.fromarray(arr)


def run_ctype(pipe: object, control_img: Image.Image, ctype: str) -> dict:
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    result = generate_sdxl_controlnet(
        pipe,
        PROMPT,
        control_img,
        ctype=ctype,
        negative_prompt=NEG_PROMPT,
        conditioning_scale=COND_SCALE,
        guidance_scale=GUIDANCE,
        num_inference_steps=STEPS,
        width=SIZE,
        height=SIZE,
        seed=SEED,
    )
    # control_image and control_mode are the verified diffusers 0.35 param names
    latency = time.perf_counter() - t0
    arr = np.array(result)
    vram_mb = torch.cuda.max_memory_allocated() / 1024**2
    return {
        "shape": arr.shape,
        "mean": float(arr.mean()),
        "vram_mb": vram_mb,
        "latency_s": latency,
    }


def main() -> None:
    print("=== PR 08 SDXL ControlNet Union GPU smoke test ===")
    print(f"torch {torch.__version__}  CUDA {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available — cannot run GPU smoke test")
        sys.exit(1)

    source_img = _make_test_image()

    print("\n[canny] preprocessing...")
    canny_img = preprocess_canny(source_img, low_threshold=100, high_threshold=200)
    print(f"  canny image size: {canny_img.size}")

    print("[depth] preprocessing is skipped for smoke test — reusing canny map as control image")
    depth_img = canny_img  # depth estimator download is optional for basic smoke

    print("\nLoading SDXL ControlNet Union pipeline (singleton model)...")
    pipe = load_sdxl_controlnet_pipeline()

    results = {}
    for ctype, ctrl_img in [("canny", canny_img), ("depth", depth_img)]:
        print(f"\n[{ctype}] generating {SIZE}×{SIZE} image ({STEPS} steps)...")
        m = run_ctype(pipe, ctrl_img, ctype)
        results[ctype] = m
        print(f"  shape:   {m['shape']}")
        print(f"  mean:    {m['mean']:.2f}")
        print(f"  VRAM:    {m['vram_mb']:.0f} MB")
        print(f"  latency: {m['latency_s']:.1f}s")

    release_sdxl_controlnet_pipeline(pipe)
    release_controlnet_union_model()

    print("\n=== Assertions ===")
    failed = []
    for ctype, m in results.items():
        if m["mean"] <= 10:
            failed.append(f"{ctype}: mean={m['mean']:.2f} ≤ 10 (all-black output)")
        else:
            print(f"  [{ctype}] mean={m['mean']:.2f} > 10  OK")

    if failed:
        for msg in failed:
            print(f"  FAIL: {msg}")
        sys.exit(1)

    print("\nCONTROLNET_SDXL_SMOKE_PASSED")


if __name__ == "__main__":
    main()
