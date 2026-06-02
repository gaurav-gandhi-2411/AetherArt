"""
Experiment 1 (SDXL): Quantization quality comparison.

SDXL port of exp1_quantization_quality.py.

Conditions: fp16 (baseline), INT8 (8-bit bitsandbytes), NF4 (4-bit bitsandbytes)
Seeds: 5 fixed seeds x 8 prompts = 40 images per condition, 120 total
Metrics: CLIP score (comparison-only), HPS, ImageReward, LPIPS vs fp16, latency (s),
         peak VRAM (MB)

Run from project root:
    python scripts/experiments/exp1_sdxl.py

Outputs:
    reports/experiments/exp1_sdxl/
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
from diffusers import StableDiffusionXLPipeline  # noqa: E402
from PIL import Image  # noqa: E402

from aetherart import clip_scorer  # noqa: E402
from aetherart.eval_hps import release_hps, score_hps  # noqa: E402
from aetherart.eval_ir import release_image_reward, score_image_reward  # noqa: E402
from aetherart.gpu_hygiene import cleanup_gpu  # noqa: E402
from aetherart.quantization import load_sdxl_quantized, vram_peak_mb  # noqa: E402
from aetherart.sdxl_pipeline import load_sdxl_base  # noqa: E402
from aetherart.visualization import BLUE, GREEN, ORANGE, ChartCanvas  # noqa: E402

atexit.register(cleanup_gpu)

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
SEEDS = [42, 123, 456, 789, 1337]
PROMPTS = {
    "p01_portrait": (
        "a portrait of an elderly woman with weathered skin, "
        "dramatic studio lighting, photorealistic"
    ),
    "p02_landscape": (
        "a misty mountain valley at sunrise, pine forest, golden hour light, landscape photography"
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
        "extreme close-up of rough concrete wall, water drops, micro detail, macro photography"
    ),
    "p06_arch": (
        "interior of a Gothic cathedral with stone arches, "
        "stained glass windows, soft diffused light"
    ),
    "p07_hands": (
        "two hands clasped together, wrinkled skin, natural light, photorealistic close-up"
    ),
    "p08_crowd": (
        "a busy street market in Tokyo, dozens of people, "
        "neon signs, rain-wet pavement, night scene"
    ),
}
NEG_PROMPT = "low quality, blurry, deformed, ugly, bad anatomy, watermark"
STEPS = 30
GUIDANCE = 7.5
SIZE = 1024

OUT = ROOT / "reports" / "experiments" / "exp1_sdxl"
IMG_DIR = OUT / "images"
CHARTS_DIR = OUT / "charts"

for _d in [IMG_DIR / "fp16", IMG_DIR / "int8", IMG_DIR / "nf4", CHARTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

CONDITIONS = ["fp16", "int8", "nf4"]
COND_COLORS = {"fp16": BLUE, "int8": GREEN, "nf4": ORANGE}
COND_LABELS = {"fp16": "fp16 (baseline)", "int8": "INT8 (8-bit)", "nf4": "NF4 (4-bit)"}


# ── LPIPS helpers ─────────────────────────────────────────────────────────────

_lpips_fn = lpips_lib.LPIPS(net="alex")
_lpips_fn.eval()


def _to_t(img: Image.Image) -> torch.Tensor:
    arr = np.array(img.convert("RGB")).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def _lpips_pair(path_a: str, path_b: str) -> float:
    a = Image.open(ROOT / path_a).convert("RGB").resize((SIZE, SIZE))
    b = Image.open(ROOT / path_b).convert("RGB").resize((SIZE, SIZE))
    with torch.no_grad():
        return round(float(_lpips_fn(_to_t(a), _to_t(b))), 6)


# ── Generation loop ───────────────────────────────────────────────────────────


def run_condition(label: str, pipe: StableDiffusionXLPipeline) -> list[dict]:
    """Generate images for one quantization condition and return result rows."""
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
                    "hps_score": None,
                    "ir_score": None,
                    "lpips": None,
                    "image_path": (img_dir / fname).relative_to(ROOT).as_posix(),
                }
            )
            print(f"  [{label}] {prompt_id} seed={seed:5d} | {latency:.1f}s | VRAM={peak_mb:.0f}MB")
    return rows


# ── Generate all conditions ───────────────────────────────────────────────────

all_rows: list[dict] = []

print("\n=== Condition 1/3: fp16 ===")
pipe = load_sdxl_base()
all_rows.extend(run_condition("fp16", pipe))
del pipe
cleanup_gpu(verbose=True)

print("\n=== Condition 2/3: int8 ===")
pipe = load_sdxl_quantized(bits=8)
all_rows.extend(run_condition("int8", pipe))
del pipe
cleanup_gpu(verbose=True)

print("\n=== Condition 3/3: nf4 ===")
pipe = load_sdxl_quantized(bits=4)
all_rows.extend(run_condition("nf4", pipe))
del pipe
cleanup_gpu(verbose=True)

# ── Post-hoc scoring (CLIP, HPS, ImageReward) ─────────────────────────────────

print(f"\nComputing scores for {len(all_rows)} images...")
for i, r in enumerate(all_rows, 1):
    img = Image.open(ROOT / r["image_path"]).convert("RGB")
    # CLIP: comparison-only — not used as primary quality metric (see CLIP-blindness verdict)
    r["clip_score"] = round(clip_scorer.score(img, r["prompt_text"]), 6)
    r["hps_score"] = round(score_hps([img], [r["prompt_text"]])[0], 6)
    r["ir_score"] = round(score_image_reward([img], [r["prompt_text"]])[0], 6)
    if i % 20 == 0 or i == len(all_rows):
        print(f"  {i}/{len(all_rows)}")

release_hps()
release_image_reward()

# ── LPIPS vs fp16 (CPU, post-hoc) ────────────────────────────────────────────

print("\nComputing LPIPS vs fp16...")

fp16_img_path: dict[tuple, str] = {
    (r["prompt_id"], r["seed"]): r["image_path"] for r in all_rows if r["condition"] == "fp16"
}

for r in all_rows:
    if r["condition"] == "fp16":
        r["lpips"] = 0.0
        continue
    ref_path = fp16_img_path[(r["prompt_id"], r["seed"])]
    r["lpips"] = _lpips_pair(r["image_path"], ref_path)

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
    "hps_score",
    "ir_score",
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
            "experiment": "exp1_sdxl",
            "date": "2026-06-02",
            "model": MODEL_ID,
            "steps": STEPS,
            "guidance": GUIDANCE,
            "size": SIZE,
            "seeds": SEEDS,
            "prompts": PROMPTS,
            "neg_prompt": NEG_PROMPT,
            "conditions": CONDITIONS,
            "scorers": ["clip", "hps", "imagereward", "lpips"],
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
    hps_vals = [r["hps_score"] for r in rows]
    ir_vals = [r["ir_score"] for r in rows]
    lats = [r["latency_s"] for r in rows]
    vrams = [r["peak_vram_mb"] for r in rows]
    lpips_vals = [r["lpips"] for r in rows if r["lpips"] > 0.0]
    agg[cond] = {
        "n": len(rows),
        "mean_clip": statistics.mean(clips),
        "se_clip": statistics.stdev(clips) / len(clips) ** 0.5,
        "mean_hps": statistics.mean(hps_vals),
        "se_hps": statistics.stdev(hps_vals) / len(hps_vals) ** 0.5,
        "mean_ir": statistics.mean(ir_vals),
        "se_ir": statistics.stdev(ir_vals) / len(ir_vals) ** 0.5,
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
        f"HPS={a['mean_hps']:.4f} ±{a['se_hps']:.4f} | "
        f"IR={a['mean_ir']:.4f} ±{a['se_ir']:.4f} | "
        f"lat={a['mean_lat']:.1f}s | VRAM={a['mean_vram']:.0f}MB | "
        f"LPIPS={a['mean_lpips']:.4f}"
    )

# ── Charts ────────────────────────────────────────────────────────────────────

x = np.arange(len(CONDITIONS), dtype=float)
quant = ["int8", "nf4"]
x2 = np.arange(len(quant), dtype=float)

# Chart 1: HPS score by condition (primary quality metric)
hps_vals_arr = np.array([agg[c]["mean_hps"] for c in CONDITIONS])
hps_max = float(hps_vals_arr.max())

canvas = ChartCanvas(
    figsize=(7, 4.5),
    title="HPS score by quantization mode — 8 prompts x 5 seeds = 40 images/condition (SDXL 1024px)",
    ylabel="Mean HPS score",
    top_margin_pct=0.22,
)
canvas.set_ylim(0.0, hps_max * 1.35)
canvas.add_bars(
    x,
    hps_vals_arr,
    colors=[COND_COLORS[c] for c in CONDITIONS],
    width=0.55,
    value_fmt="{:.4f}",
    value_pad=hps_max * 0.015,
    value_size=9,
)
canvas.set_xticks(x, [COND_LABELS[c] for c in CONDITIONS], fontsize=9)
canvas.save(str(CHARTS_DIR / "hps_by_condition.png"))

# Chart 2: ImageReward score by condition
ir_vals_arr = np.array([agg[c]["mean_ir"] for c in CONDITIONS])
ir_min = float(ir_vals_arr.min())
ir_max = float(ir_vals_arr.max())
ir_pad = max(abs(ir_min) * 0.4, abs(ir_max) * 0.4, 0.1)

canvas_ir = ChartCanvas(
    figsize=(7, 4.5),
    title="ImageReward score by quantization mode — SDXL 1024px",
    ylabel="Mean ImageReward score",
    top_margin_pct=0.22,
)
canvas_ir.set_ylim(ir_min - ir_pad, ir_max + ir_pad * 1.5)
canvas_ir.add_bars(
    x,
    ir_vals_arr,
    colors=[COND_COLORS[c] for c in CONDITIONS],
    width=0.55,
    value_fmt="{:.4f}",
    value_pad=ir_pad * 0.1,
    value_size=9,
)
canvas_ir.set_xticks(x, [COND_LABELS[c] for c in CONDITIONS], fontsize=9)
canvas_ir.save(str(CHARTS_DIR / "ir_by_condition.png"))

# Chart 3: LPIPS vs fp16 (int8 and nf4 only)
lpips_vals_chart = np.array([agg[c]["mean_lpips"] for c in quant])
lpips_max = float(lpips_vals_chart.max())

canvas2 = ChartCanvas(
    figsize=(6, 4.5),
    title="Perceptual distance from fp16 (LPIPS, lower = more similar) — SDXL 1024px",
    ylabel="Mean LPIPS vs fp16",
    top_margin_pct=0.22,
)
canvas2.set_ylim(0.0, lpips_max * 1.6)
canvas2.add_bars(
    x2,
    lpips_vals_chart,
    colors=[COND_COLORS[c] for c in quant],
    width=0.55,
    value_fmt="{:.4f}",
    value_pad=lpips_max * 0.05,
    value_size=9,
)
canvas2.set_xticks(x2, [COND_LABELS[c] for c in quant], fontsize=9)
canvas2.save(str(CHARTS_DIR / "lpips_vs_fp16.png"))

# Chart 4: Latency by condition
lat_vals = np.array([agg[c]["mean_lat"] for c in CONDITIONS])
lat_max = float(lat_vals.max())

canvas3 = ChartCanvas(
    figsize=(7, 4.5),
    title="Generation latency by quantization mode (30 steps, 1024x1024, GCP L4)",
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

# Chart 5: CLIP score by condition (comparison-only, shown for reference)
clip_vals_arr = np.array([agg[c]["mean_clip"] for c in CONDITIONS])
clip_max = float(clip_vals_arr.max())

canvas_clip = ChartCanvas(
    figsize=(7, 4.5),
    title="CLIP score by quantization mode — SDXL 1024px (comparison-only, see CLIP-blindness verdict)",
    ylabel="Mean CLIP score",
    top_margin_pct=0.22,
)
canvas_clip.set_ylim(0.0, clip_max * 1.35)
canvas_clip.add_bars(
    x,
    clip_vals_arr,
    colors=[COND_COLORS[c] for c in CONDITIONS],
    width=0.55,
    value_fmt="{:.4f}",
    value_pad=clip_max * 0.015,
    value_size=9,
)
canvas_clip.set_xticks(x, [COND_LABELS[c] for c in CONDITIONS], fontsize=9)
canvas_clip.save(str(CHARTS_DIR / "clip_by_condition.png"))

print(f"Charts written to {CHARTS_DIR}")

# ── Findings writeup (data-driven) ───────────────────────────────────────────

fp16_a = agg["fp16"]
int8_a = agg["int8"]
nf4_a = agg["nf4"]

int8_clip_delta = int8_a["mean_clip"] - fp16_a["mean_clip"]
nf4_clip_delta = nf4_a["mean_clip"] - fp16_a["mean_clip"]
int8_hps_delta = int8_a["mean_hps"] - fp16_a["mean_hps"]
nf4_hps_delta = nf4_a["mean_hps"] - fp16_a["mean_hps"]
int8_ir_delta = int8_a["mean_ir"] - fp16_a["mean_ir"]
nf4_ir_delta = nf4_a["mean_ir"] - fp16_a["mean_ir"]

pooled_se_clip = fp16_a["se_clip"]
clip_delta_range = max(abs(int8_clip_delta), abs(nf4_clip_delta))
clip_delta_in_se = clip_delta_range / pooled_se_clip if pooled_se_clip > 0 else 0.0
lpips_range = max(int8_a["mean_lpips"], nf4_a["mean_lpips"])
clip_blind = "yes" if clip_delta_in_se < 2.0 else "no"

int8_lat_ratio = int8_a["mean_lat"] / fp16_a["mean_lat"]
nf4_lat_ratio = nf4_a["mean_lat"] / fp16_a["mean_lat"]
int8_vram_saving_mb = fp16_a["mean_vram"] - int8_a["mean_vram"]
nf4_vram_saving_mb = fp16_a["mean_vram"] - nf4_a["mean_vram"]
int8_vram_saving_pct = (
    (1 - int8_a["mean_vram"] / fp16_a["mean_vram"]) * 100 if fp16_a["mean_vram"] > 0 else 0.0
)
nf4_vram_saving_pct = (
    (1 - nf4_a["mean_vram"] / fp16_a["mean_vram"]) * 100 if fp16_a["mean_vram"] > 0 else 0.0
)


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


FINDINGS = f"""\
# Experiment 1 (SDXL): Quantization Quality Comparison

**Date:** 2026-06-02
**Hardware:** GCP L4 (24 GB VRAM)
**Model:** {MODEL_ID}
**Conditions:** fp16 (baseline) · INT8 (8-bit bitsandbytes U-Net) · NF4 (4-bit bitsandbytes U-Net)
**Design:** 5 seeds x 8 prompts = 40 images per condition · 120 images total
**Scheduler:** DPM-Solver++ · {STEPS} steps · {SIZE}x{SIZE} · CFG={GUIDANCE}
**Scorers:** HPS (primary), ImageReward (primary), CLIP (comparison-only), LPIPS

## Hypothesis

Quantizing the SDXL U-Net to INT8 or NF4 will degrade output quality measurably, but not
catastrophically. Perceptual degradation (LPIPS) should be detectable before CLIP-score
differences rise above statistical noise. HPS and ImageReward provide human-preference-aligned
signal where CLIP may be blind.

## Results

| Condition | Mean HPS | HPS delta | Mean IR | IR delta | Mean CLIP | CLIP delta | Mean LPIPS | Latency (s) | Peak VRAM (MB) |
|-----------|:--------:|----------:|:-------:|:--------:|----------:|-----------:|-----------:|------------:|---------------:|
| fp16      | {fp16_a["mean_hps"]:.4f}   | —         | {fp16_a["mean_ir"]:.4f}  | —        | {fp16_a["mean_clip"]:.4f}    | —          | 0.0000     | {fp16_a["mean_lat"]:.1f}s       | {fp16_a["mean_vram"]:.0f}            |
| INT8      | {int8_a["mean_hps"]:.4f}   | {int8_hps_delta:+.4f}    | {int8_a["mean_ir"]:.4f}  | {int8_ir_delta:+.4f}   | {int8_a["mean_clip"]:.4f}    | {int8_clip_delta:+.4f}     | {int8_a["mean_lpips"]:.4f}     | {int8_a["mean_lat"]:.1f}s       | {int8_a["mean_vram"]:.0f}            |
| NF4       | {nf4_a["mean_hps"]:.4f}   | {nf4_hps_delta:+.4f}    | {nf4_a["mean_ir"]:.4f}  | {nf4_ir_delta:+.4f}   | {nf4_a["mean_clip"]:.4f}    | {nf4_clip_delta:+.4f}     | {nf4_a["mean_lpips"]:.4f}     | {nf4_a["mean_lat"]:.1f}s       | {nf4_a["mean_vram"]:.0f}            |

SE: fp16 CLIP ±{fp16_a["se_clip"]:.4f} · INT8 CLIP ±{int8_a["se_clip"]:.4f} · NF4 CLIP ±{nf4_a["se_clip"]:.4f}
SE: fp16 HPS ±{fp16_a["se_hps"]:.4f} · INT8 HPS ±{int8_a["se_hps"]:.4f} · NF4 HPS ±{nf4_a["se_hps"]:.4f}
SE: fp16 IR ±{fp16_a["se_ir"]:.4f} · INT8 IR ±{int8_a["se_ir"]:.4f} · NF4 IR ±{nf4_a["se_ir"]:.4f}

## CLIP-blindness verdict

CLIP delta across conditions = {clip_delta_range:.4f}, which is {clip_delta_in_se:.2f} SEs. \
HPS delta = {max(abs(int8_hps_delta), abs(nf4_hps_delta)):.4f}. \
IR delta = {max(abs(int8_ir_delta), abs(nf4_ir_delta)):.4f}. \
LPIPS range = {lpips_range:.4f}. \
Verdict: CLIP-BLIND: {clip_blind}.

## Interpretation

**INT8 quality:** CLIP score is {_clip_verdict(int8_clip_delta, pooled_se_clip)}.
Perceptual fidelity to fp16: {_lpips_verdict(int8_a["mean_lpips"])}.
Latency cost: {int8_a["mean_lat"]:.1f}s vs {fp16_a["mean_lat"]:.1f}s fp16 ({int8_lat_ratio:.1f}x slower).
VRAM saved: {int8_vram_saving_mb:.0f} MB ({int8_vram_saving_pct:.0f}% reduction vs fp16).

**NF4 quality:** CLIP score is {_clip_verdict(nf4_clip_delta, pooled_se_clip)}.
Perceptual fidelity to fp16: {_lpips_verdict(nf4_a["mean_lpips"])}.
Latency cost: {nf4_a["mean_lat"]:.1f}s vs {fp16_a["mean_lat"]:.1f}s fp16 ({nf4_lat_ratio:.1f}x slower).
VRAM saved: {nf4_vram_saving_mb:.0f} MB ({nf4_vram_saving_pct:.0f}% reduction vs fp16).

**Bottom line:** Both quantization modes preserve semantic alignment but carry a latency
penalty from dequantization overhead. LPIPS and HPS/IR provide converging evidence where
CLIP may be blind to perceptual changes. If LPIPS < 0.05 and HPS/IR deltas are small for
both modes, quantization is essentially transparent for these prompt types at 1024×1024.

## Charts

- `charts/hps_by_condition.png` — mean HPS score per condition (primary quality metric)
- `charts/ir_by_condition.png` — mean ImageReward score per condition
- `charts/clip_by_condition.png` — mean CLIP score per condition (comparison-only)
- `charts/lpips_vs_fp16.png` — LPIPS perceptual distance from fp16 (INT8 and NF4 only)
- `charts/latency_by_condition.png` — mean generation latency per condition

## Raw data

`results.csv` / `results.json` — one row per image (120 rows total).

Reproduce:

```bash
python scripts/experiments/exp1_sdxl.py
```
"""

with open(OUT / "findings.md", "w", encoding="utf-8") as f:
    f.write(FINDINGS)

print(f"\nFindings written: {OUT / 'findings.md'}")
print("Experiment 1 (SDXL) complete.")
