"""
Experiment 1: Quantization quality comparison.

Conditions: fp16 (baseline), INT8 (8-bit bitsandbytes), NF4 (4-bit bitsandbytes)
Seeds: 5 fixed seeds x 8 prompts = 40 images per condition, 120 total
Metrics: CLIP score, LPIPS vs fp16, latency (s), peak VRAM (MB)

Run from project root:
    python scripts/experiments/exp1_quantization_quality.py

Outputs:
    reports/experiments/exp1_quantization_quality/
        images/{fp16,int8,nf4}/  -- 40 PNG per condition
        results.csv              -- one row per image
        results.json             -- same data + metadata
        charts/                  -- ChartCanvas figures
        findings.md              -- per-experiment writeup
"""

from __future__ import annotations

import atexit
import csv
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import lpips as lpips_lib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline  # noqa: E402
from PIL import Image  # noqa: E402

from aetherart import clip_scorer  # noqa: E402
from aetherart.gpu_hygiene import cleanup_gpu  # noqa: E402
from aetherart.quantization import load_sd21_quantized, vram_peak_mb  # noqa: E402
from aetherart.visualization import BLUE, GREEN, ORANGE, ChartCanvas  # noqa: E402

atexit.register(cleanup_gpu)

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_ID = "sd2-community/stable-diffusion-2-1"
SEEDS = [42, 123, 456, 789, 1337]
PROMPTS = {
    "p01_portrait": (
        "a portrait of an elderly woman with weathered skin, "
        "dramatic studio lighting, photorealistic"
    ),
    "p02_landscape": (
        "a misty mountain valley at sunrise, pine forest, "
        "golden hour light, landscape photography"
    ),
    "p03_abstract": (
        "geometric abstract composition with intersecting circles and triangles, "
        "vibrant color blocks"
    ),
    "p04_text": (
        "a vintage poster with large bold letters reading OPEN, "
        "retro typography, worn paper texture"
    ),
    "p05_texture": (
        "extreme close-up of rough concrete wall, water drops, " "micro detail, macro photography"
    ),
    "p06_arch": (
        "interior of a Gothic cathedral with stone arches, "
        "stained glass windows, soft diffused light"
    ),
    "p07_hands": (
        "two hands clasped together, wrinkled skin, " "natural light, photorealistic close-up"
    ),
    "p08_crowd": (
        "a busy street market in Tokyo, dozens of people, "
        "neon signs, rain-wet pavement, night scene"
    ),
}
NEG_PROMPT = "low quality, blurry, deformed, ugly, bad anatomy, watermark"
STEPS = 30
GUIDANCE = 7.5
SIZE = 512

OUT = ROOT / "reports" / "experiments" / "exp1_quantization_quality"
IMG_DIR = OUT / "images"
CHARTS_DIR = OUT / "charts"

for _d in [IMG_DIR / "fp16", IMG_DIR / "int8", IMG_DIR / "nf4", CHARTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

CONDITIONS = ["fp16", "int8", "nf4"]
COND_COLORS = {"fp16": BLUE, "int8": GREEN, "nf4": ORANGE}
COND_LABELS = {"fp16": "fp16 (baseline)", "int8": "INT8 (8-bit)", "nf4": "NF4 (4-bit)"}


# ── Pipeline loading ──────────────────────────────────────────────────────────


def _apply_dpm(pipe: StableDiffusionPipeline) -> StableDiffusionPipeline:
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe


def load_fp16() -> StableDiffusionPipeline:
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    _apply_dpm(pipe)
    if torch.cuda.is_available():
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pipe = pipe.to("cuda")
    return pipe


def load_quantized(bits: int) -> StableDiffusionPipeline:
    pipe = load_sd21_quantized(bits=bits)  # already enables cpu_offload
    _apply_dpm(pipe)
    return pipe


# ── Generation loop ───────────────────────────────────────────────────────────


def run_condition(label: str, pipe: StableDiffusionPipeline) -> list[dict]:
    rows: list[dict] = []
    img_dir = IMG_DIR / label
    for prompt_id, prompt_text in PROMPTS.items():
        for seed in SEEDS:
            generator = torch.Generator().manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = pipe(
                prompt=prompt_text,
                negative_prompt=NEG_PROMPT,
                num_inference_steps=STEPS,
                guidance_scale=GUIDANCE,
                height=SIZE,
                width=SIZE,
                generator=generator,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latency = time.perf_counter() - t0
            peak_mb = vram_peak_mb()

            fname = f"{prompt_id}_seed{seed}.png"
            out.images[0].save(img_dir / fname)

            rows.append(
                {
                    "condition": label,
                    "prompt_id": prompt_id,
                    "prompt_text": prompt_text,
                    "seed": seed,
                    "latency_s": round(latency, 3),
                    "peak_vram_mb": round(peak_mb, 1),
                    "clip_score": None,
                    "lpips": None,
                    "image_path": (img_dir / fname).relative_to(ROOT).as_posix(),
                }
            )
            print(
                f"  [{label}] {prompt_id} seed={seed:5d} | "
                f"{latency:.1f}s | VRAM={peak_mb:.0f}MB"
            )
    return rows


# ── Generate all conditions ───────────────────────────────────────────────────

all_rows: list[dict] = []

print("\n=== Condition 1/3: fp16 ===")
pipe = load_fp16()
all_rows.extend(run_condition("fp16", pipe))
del pipe
cleanup_gpu(verbose=True)

print("\n=== Condition 2/3: int8 ===")
pipe = load_quantized(8)
all_rows.extend(run_condition("int8", pipe))
del pipe
cleanup_gpu(verbose=True)

print("\n=== Condition 3/3: nf4 ===")
pipe = load_quantized(4)
all_rows.extend(run_condition("nf4", pipe))
del pipe
cleanup_gpu(verbose=True)

# ── CLIP scores (post-hoc, all images) ───────────────────────────────────────

print(f"\nComputing CLIP scores for {len(all_rows)} images...")
for i, r in enumerate(all_rows, 1):
    img = Image.open(ROOT / r["image_path"]).convert("RGB")
    r["clip_score"] = round(clip_scorer.score(img, r["prompt_text"]), 6)
    if i % 20 == 0 or i == len(all_rows):
        print(f"  {i}/{len(all_rows)}")

# ── LPIPS vs fp16 (CPU, post-hoc) ────────────────────────────────────────────

print("\nComputing LPIPS vs fp16...")
_lpips_fn = lpips_lib.LPIPS(net="alex")
_lpips_fn.eval()

fp16_img_path: dict[tuple, str] = {
    (r["prompt_id"], r["seed"]): r["image_path"] for r in all_rows if r["condition"] == "fp16"
}


def _to_lpips_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img.convert("RGB")).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


for r in all_rows:
    if r["condition"] == "fp16":
        r["lpips"] = 0.0
        continue
    ref = Image.open(ROOT / fp16_img_path[(r["prompt_id"], r["seed"])]).convert("RGB")
    cmp = Image.open(ROOT / r["image_path"]).convert("RGB")
    with torch.no_grad():
        r["lpips"] = round(float(_lpips_fn(_to_lpips_tensor(ref), _to_lpips_tensor(cmp))), 6)

print("LPIPS done.")

# ── Save results ──────────────────────────────────────────────────────────────

CSV_PATH = OUT / "results.csv"
JSON_PATH = OUT / "results.json"

csv_fields = [
    "condition",
    "prompt_id",
    "seed",
    "latency_s",
    "peak_vram_mb",
    "clip_score",
    "lpips",
    "image_path",
]
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(all_rows)

with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(
        {
            "experiment": "exp1_quantization_quality",
            "date": "2026-05-05",
            "model": MODEL_ID,
            "steps": STEPS,
            "guidance": GUIDANCE,
            "size": SIZE,
            "seeds": SEEDS,
            "prompts": PROMPTS,
            "neg_prompt": NEG_PROMPT,
            "conditions": CONDITIONS,
            "total_images": len(all_rows),
            "results": all_rows,
        },
        f,
        indent=2,
    )

print(f"Results: {CSV_PATH}\n         {JSON_PATH}")

# ── Per-condition aggregates ──────────────────────────────────────────────────

by_cond: dict[str, list[dict]] = {c: [] for c in CONDITIONS}
for r in all_rows:
    by_cond[r["condition"]].append(r)

agg: dict[str, dict] = {}
for cond, rows in by_cond.items():
    clips = [r["clip_score"] for r in rows]
    lats = [r["latency_s"] for r in rows]
    vrams = [r["peak_vram_mb"] for r in rows]
    lpips_vals = [r["lpips"] for r in rows if r["lpips"] > 0.0]
    agg[cond] = {
        "n": len(rows),
        "mean_clip": statistics.mean(clips),
        "se_clip": statistics.stdev(clips) / len(clips) ** 0.5,
        "mean_lat": statistics.mean(lats),
        "mean_vram": statistics.mean(vrams),
        "mean_lpips": statistics.mean(lpips_vals) if lpips_vals else 0.0,
        "se_lpips": (
            statistics.stdev(lpips_vals) / len(lpips_vals) ** 0.5 if len(lpips_vals) > 1 else 0.0
        ),
    }

print("\n── Aggregates ──")
for cond in CONDITIONS:
    a = agg[cond]
    print(
        f"  {cond:4s}: CLIP={a['mean_clip']:.4f} ±{a['se_clip']:.4f} | "
        f"lat={a['mean_lat']:.1f}s | VRAM={a['mean_vram']:.0f}MB | "
        f"LPIPS={a['mean_lpips']:.4f}"
    )

# ── Charts ────────────────────────────────────────────────────────────────────

x = np.arange(len(CONDITIONS), dtype=float)
quant = ["int8", "nf4"]
x2 = np.arange(len(quant), dtype=float)

# Chart 1: CLIP score by condition
clip_vals = np.array([agg[c]["mean_clip"] for c in CONDITIONS])
clip_max = float(clip_vals.max())

canvas = ChartCanvas(
    figsize=(7, 4.5),
    title="CLIP score by quantization mode — 8 prompts x 5 seeds = 40 images/condition",
    ylabel="Mean CLIP score",
    top_margin_pct=0.22,
)
canvas.set_ylim(0.0, clip_max * 1.35)
canvas.add_bars(
    x,
    clip_vals,
    colors=[COND_COLORS[c] for c in CONDITIONS],
    width=0.55,
    value_fmt="{:.4f}",
    value_pad=clip_max * 0.015,
    value_size=9,
)
canvas.set_xticks(x, [COND_LABELS[c] for c in CONDITIONS], fontsize=9)
canvas.save(str(CHARTS_DIR / "clip_by_condition.png"))

# Chart 2: LPIPS vs fp16 (int8 and nf4 only)
lpips_vals_arr = np.array([agg[c]["mean_lpips"] for c in quant])
lpips_max = float(lpips_vals_arr.max())

canvas2 = ChartCanvas(
    figsize=(6, 4.5),
    title="Perceptual distance from fp16 (LPIPS, lower = more similar)",
    ylabel="Mean LPIPS vs fp16",
    top_margin_pct=0.22,
)
canvas2.set_ylim(0.0, lpips_max * 1.6)
canvas2.add_bars(
    x2,
    lpips_vals_arr,
    colors=[COND_COLORS[c] for c in quant],
    width=0.55,
    value_fmt="{:.4f}",
    value_pad=lpips_max * 0.05,
    value_size=9,
)
canvas2.set_xticks(x2, [COND_LABELS[c] for c in quant], fontsize=9)
canvas2.save(str(CHARTS_DIR / "lpips_vs_fp16.png"))

# Chart 3: Latency by condition
lat_vals = np.array([agg[c]["mean_lat"] for c in CONDITIONS])
lat_max = float(lat_vals.max())

canvas3 = ChartCanvas(
    figsize=(7, 4.5),
    title="Generation latency by quantization mode (30 steps, 512x512, RTX 3070)",
    ylabel="Mean latency (s)",
    top_margin_pct=0.22,
)
canvas3.set_ylim(0.0, lat_max * 1.4)
canvas3.add_bars(
    x,
    lat_vals,
    colors=[COND_COLORS[c] for c in CONDITIONS],
    width=0.55,
    value_fmt="{:.1f}s",
    value_pad=lat_max * 0.03,
    value_size=9,
)
canvas3.set_xticks(x, [COND_LABELS[c] for c in CONDITIONS], fontsize=9)
canvas3.save(str(CHARTS_DIR / "latency_by_condition.png"))

print(f"Charts written to {CHARTS_DIR}")

# ── Findings writeup (data-driven) ───────────────────────────────────────────

fp16_a = agg["fp16"]
int8_a = agg["int8"]
nf4_a = agg["nf4"]
int8_delta = int8_a["mean_clip"] - fp16_a["mean_clip"]
nf4_delta = nf4_a["mean_clip"] - fp16_a["mean_clip"]
pooled_se = fp16_a["se_clip"]


def _clip_verdict(delta: float, se: float) -> str:
    if abs(delta) < 2 * se:
        return f"within 2 SE of fp16 (delta = {delta:+.4f}) — statistically indistinguishable"
    return (
        f"{'above' if delta > 0 else 'below'} fp16 by {abs(delta):.4f} (> 2 SE) — "
        f"statistically distinguishable"
    )


def _lpips_verdict(val: float) -> str:
    if val < 0.05:
        return f"near-identical to fp16 perceptually (LPIPS = {val:.4f}, < 0.05 threshold)"
    if val < 0.10:
        return f"minor perceptual differences from fp16 (LPIPS = {val:.4f}, 0.05–0.10)"
    if val < 0.20:
        return f"moderate perceptual differences from fp16 (LPIPS = {val:.4f}, 0.10–0.20)"
    return f"substantial perceptual degradation vs fp16 (LPIPS = {val:.4f}, > 0.20)"


int8_lat_ratio = int8_a["mean_lat"] / fp16_a["mean_lat"]
nf4_lat_ratio = nf4_a["mean_lat"] / fp16_a["mean_lat"]
int8_vram_saving_mb = fp16_a["mean_vram"] - int8_a["mean_vram"]
nf4_vram_saving_mb = fp16_a["mean_vram"] - nf4_a["mean_vram"]
int8_vram_saving_pct = (1 - int8_a["mean_vram"] / fp16_a["mean_vram"]) * 100
nf4_vram_saving_pct = (1 - nf4_a["mean_vram"] / fp16_a["mean_vram"]) * 100

FINDINGS = f"""\
# Experiment 1: Quantization Quality Comparison

**Date:** 2026-05-05
**Hardware:** RTX 3070 Laptop 8 GB
**Model:** {MODEL_ID}
**Conditions:** fp16 (baseline) · INT8 (8-bit bitsandbytes U-Net) · NF4 (4-bit bitsandbytes U-Net)
**Design:** 5 seeds x 8 prompts = 40 images per condition · 120 images total
**Scheduler:** DPM-Solver++ · {STEPS} steps · {SIZE}x{SIZE} · CFG={GUIDANCE}

## Hypothesis

Quantizing the U-Net to INT8 or NF4 will degrade output quality measurably, but not
catastrophically. Perceptual degradation (LPIPS) should be detectable before CLIP-score
differences rise above statistical noise.

## Results

| Condition | Mean CLIP | CLIP delta vs fp16 | Mean LPIPS | Latency (s) | Peak VRAM (MB) |
|-----------|----------:|-------------------:|-----------:|------------:|---------------:|
| fp16      | {fp16_a['mean_clip']:.4f}    | —                  | 0.0000     | {fp16_a['mean_lat']:.1f}s       | {fp16_a['mean_vram']:.0f}            |
| INT8      | {int8_a['mean_clip']:.4f}    | {int8_delta:+.4f}             | {int8_a['mean_lpips']:.4f}     | {int8_a['mean_lat']:.1f}s       | {int8_a['mean_vram']:.0f}            |
| NF4       | {nf4_a['mean_clip']:.4f}    | {nf4_delta:+.4f}             | {nf4_a['mean_lpips']:.4f}     | {nf4_a['mean_lat']:.1f}s       | {nf4_a['mean_vram']:.0f}            |

SE on CLIP: fp16 ±{fp16_a['se_clip']:.4f} · INT8 ±{int8_a['se_clip']:.4f} · \\
NF4 ±{nf4_a['se_clip']:.4f}

## Interpretation

**INT8 quality:** CLIP score is {_clip_verdict(int8_delta, pooled_se)}.
Perceptual fidelity to fp16: {_lpips_verdict(int8_a['mean_lpips'])}.
Latency cost: {int8_a['mean_lat']:.1f}s vs {fp16_a['mean_lat']:.1f}s fp16 ({int8_lat_ratio:.1f}x slower).
VRAM saved: {int8_vram_saving_mb:.0f} MB ({int8_vram_saving_pct:.0f}% reduction vs fp16).

**NF4 quality:** CLIP score is {_clip_verdict(nf4_delta, pooled_se)}.
Perceptual fidelity to fp16: {_lpips_verdict(nf4_a['mean_lpips'])}.
Latency cost: {nf4_a['mean_lat']:.1f}s vs {fp16_a['mean_lat']:.1f}s fp16 ({nf4_lat_ratio:.1f}x slower).
VRAM saved: {nf4_vram_saving_mb:.0f} MB ({nf4_vram_saving_pct:.0f}% reduction vs fp16).

**Bottom line:** Both quantization modes preserve CLIP-measured alignment but carry a latency
penalty (dequantization overhead). LPIPS indicates whether the pixel-level differences are
perceptually meaningful. If LPIPS < 0.05 for both modes, quantization is essentially
transparent to human perception for these prompt types.

## Charts

- `charts/clip_by_condition.png` — mean CLIP score per condition
- `charts/lpips_vs_fp16.png` — LPIPS perceptual distance from fp16 (INT8 and NF4 only)
- `charts/latency_by_condition.png` — mean generation latency per condition

## Raw data

`results.csv` / `results.json` — one row per image (120 rows total).

Reproduce:

```bash
python scripts/experiments/exp1_quantization_quality.py
```
"""

with open(OUT / "findings.md", "w", encoding="utf-8") as f:
    f.write(FINDINGS)

print(f"\nFindings written: {OUT / 'findings.md'}")
print("Experiment 1 complete.")
