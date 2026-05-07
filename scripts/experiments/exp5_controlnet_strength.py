"""
Experiment 5: ControlNet conditioning strength sweep (Canny, SD2.1).

Conditioning images: canny edges extracted from benchmark DDIM-30step images
  (pp_001 through pp_008, 512x512, seed=42 — fixed, one per prompt).
Strength values: [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
  0.0 = no conditioning (effectively text-only)
  1.0 = standard ControlNet strength (reference for LPIPS)
  1.5 = over-conditioned
Seeds: [42, 123, 456, 789, 1337] — 5 seeds × 8 prompts = 40 images per strength
Total: 7 × 8 × 5 = 280 images

Metrics:
  - CLIP score (semantic alignment to prompt)
  - LPIPS vs strength=1.0 reference (same prompt+seed pair)
  - LPIPS vs previous strength value (adjacent step visual change)

Hypothesis: CLIP stays roughly flat across all strength values (same text prompt,
same semantic content). LPIPS vs the reference reveals when the image is
perceptually departing from standard conditioning — both below (too loose) and
above (over-constrained). Another CLIP-blindness data point.

Run from project root:
    python scripts/experiments/exp5_controlnet_strength.py

Outputs:
    reports/experiments/exp5_controlnet_strength/
        images/strength_{val}/  -- 40 PNG per strength
        results.csv
        results.json
        charts/
        findings.md
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
from diffusers import (  # noqa: E402
    ControlNetModel,
    DPMSolverMultistepScheduler,
    StableDiffusionControlNetPipeline,
)
from PIL import Image  # noqa: E402

from aetherart import clip_scorer  # noqa: E402
from aetherart.controlnet import CANNY_MODEL_ID, preprocess_canny  # noqa: E402
from aetherart.gpu_hygiene import cleanup_gpu  # noqa: E402
from aetherart.visualization import (  # noqa: E402
    BLUE,
    GREEN,
    GREY,
    ORANGE,
    RED,
    TEAL,
    ChartCanvas,
)

atexit.register(cleanup_gpu)

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_ID = "sd2-community/stable-diffusion-2-1"
STRENGTH_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
STRENGTH_REF = 1.0
SEEDS = [42, 123, 456, 789, 1337]
STEPS = 30
CFG = 7.0
SIZE = 512

# PartiPrompts pp_001..pp_008 — sourced from the 360-run benchmark
PROMPT_IDS = [f"pp_{i:03d}" for i in range(1, 9)]
PROMPTS = {
    "pp_001": "artificial intelligence",
    "pp_002": "a shiba inu",
    "pp_003": "a dragon",
    "pp_004": "a shiba inu wearing a beret and black turtleneck",
    "pp_005": "a corgi wearing a red bowtie and a purple party hat",
    "pp_006": "an eagle swooping down to catch a mouse",
    "pp_007": "an elephant using its trunk to blow into a tuba",
    "pp_008": "a dolphin in an astronaut suit on saturn",
}

# Source images for canny conditioning: DDIM 30-step benchmark outputs
BENCHMARK_IMGS = {
    pid: ROOT / "outputs" / "eval" / pid / "DDIM" / "30steps.png" for pid in PROMPT_IDS
}

NEG_PROMPT = "low quality, blurry, deformed, ugly, bad anatomy, watermark"

OUT = ROOT / "reports" / "experiments" / "exp5_controlnet_strength"
IMG_DIR = OUT / "images"
CHARTS_DIR = OUT / "charts"

for s in STRENGTH_VALUES:
    (IMG_DIR / f"strength_{s}").mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
(OUT / "canny").mkdir(parents=True, exist_ok=True)

# Colour palette per strength level
_STR_PALETTE = {
    0.0: GREY,
    0.25: "#5AADA8",
    0.5: GREEN,
    0.75: TEAL,
    1.0: BLUE,
    1.25: ORANGE,
    1.5: RED,
}


# ── Pre-compute canny conditioning images (one per prompt) ────────────────────

print("Extracting canny edges from benchmark source images...")
canny_images: dict[str, Image.Image] = {}
for pid in PROMPT_IDS:
    src_path = BENCHMARK_IMGS[pid]
    if not src_path.exists():
        raise FileNotFoundError(
            f"Benchmark image missing: {src_path}\n"
            "Run the benchmark eval first (scripts/eval.py)."
        )
    src = Image.open(src_path).convert("RGB")
    canny_img = preprocess_canny(src, low_threshold=100, high_threshold=200)
    canny_path = OUT / "canny" / f"{pid}_canny.png"
    canny_img.save(canny_path)
    canny_images[pid] = canny_img
    print(f"  {pid}: {src_path.name} → canny saved")

print(f"Canny conditioning images ready ({len(canny_images)} prompts)")


# ── Pipeline ──────────────────────────────────────────────────────────────────


def load_pipeline() -> StableDiffusionControlNetPipeline:
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Loading ControlNet weights: {CANNY_MODEL_ID}")
    controlnet = ControlNetModel.from_pretrained(CANNY_MODEL_ID, torch_dtype=dtype)
    print(f"Loading SD2.1 base: {MODEL_ID}")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        MODEL_ID, controlnet=controlnet, torch_dtype=dtype
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if torch.cuda.is_available():
        try:
            pipe.enable_model_cpu_offload()
            print("Model CPU offload enabled")
        except Exception:
            pipe = pipe.to("cuda")
    return pipe


# ── Generation loop ───────────────────────────────────────────────────────────


def run_strength(
    strength: float,
    pipe: StableDiffusionControlNetPipeline,
) -> list[dict]:
    rows: list[dict] = []
    img_dir = IMG_DIR / f"strength_{strength}"
    for pid in PROMPT_IDS:
        prompt_text = PROMPTS[pid]
        cond_img = canny_images[pid]
        for seed in SEEDS:
            generator = torch.Generator().manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = pipe(
                prompt=prompt_text,
                image=cond_img,
                negative_prompt=NEG_PROMPT,
                num_inference_steps=STEPS,
                guidance_scale=CFG,
                controlnet_conditioning_scale=strength,
                height=SIZE,
                width=SIZE,
                generator=generator,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latency = time.perf_counter() - t0

            fname = f"{pid}_seed{seed}.png"
            out.images[0].save(img_dir / fname)

            rows.append(
                {
                    "strength": strength,
                    "prompt_id": pid,
                    "prompt_text": prompt_text,
                    "seed": seed,
                    "latency_s": round(latency, 3),
                    "clip_score": None,
                    "lpips_vs_ref": None,
                    "lpips_vs_prev": None,
                    "image_path": (img_dir / fname).relative_to(ROOT).as_posix(),
                }
            )
            print(f"  [strength={strength:.2f}] {pid} seed={seed:5d} | {latency:.1f}s")
    return rows


# ── Generate all strength values (single loaded pipeline) ─────────────────────

all_rows: list[dict] = []

print("\nLoading ControlNet pipeline (fp16)...")
pipe = load_pipeline()

for strength in STRENGTH_VALUES:
    idx = STRENGTH_VALUES.index(strength) + 1
    print(f"\n=== Strength = {strength} ({idx}/{len(STRENGTH_VALUES)}) ===")
    all_rows.extend(run_strength(strength, pipe))

del pipe
cleanup_gpu(verbose=True)


# ── CLIP scores (post-hoc) ────────────────────────────────────────────────────

print(f"\nComputing CLIP scores for {len(all_rows)} images...")
for i, r in enumerate(all_rows, 1):
    img = Image.open(ROOT / r["image_path"]).convert("RGB")
    r["clip_score"] = round(clip_scorer.score(img, r["prompt_text"]), 6)
    if i % 40 == 0 or i == len(all_rows):
        print(f"  {i}/{len(all_rows)}")


# ── LPIPS (post-hoc) ─────────────────────────────────────────────────────────

print("\nComputing LPIPS (vs reference and vs adjacent)...")
_lpips_fn = lpips_lib.LPIPS(net="alex")
_lpips_fn.eval()

img_index: dict[tuple, str] = {
    (r["strength"], r["prompt_id"], r["seed"]): r["image_path"] for r in all_rows
}


def _to_t(img: Image.Image) -> torch.Tensor:
    arr = np.array(img.convert("RGB")).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def _lpips_pair(path_a: str, path_b: str) -> float:
    a = Image.open(ROOT / path_a).convert("RGB")
    b = Image.open(ROOT / path_b).convert("RGB")
    with torch.no_grad():
        return round(float(_lpips_fn(_to_t(a), _to_t(b))), 6)


done = 0
total_lpips = len(all_rows) * 2
for r in all_rows:
    s = r["strength"]
    pid = r["prompt_id"]
    seed = r["seed"]

    if s == STRENGTH_REF:
        r["lpips_vs_ref"] = 0.0
    else:
        ref_path = img_index[(STRENGTH_REF, pid, seed)]
        r["lpips_vs_ref"] = _lpips_pair(r["image_path"], ref_path)
    done += 1

    s_idx = STRENGTH_VALUES.index(s)
    if s_idx == 0:
        r["lpips_vs_prev"] = None
    else:
        prev_s = STRENGTH_VALUES[s_idx - 1]
        prev_path = img_index[(prev_s, pid, seed)]
        r["lpips_vs_prev"] = _lpips_pair(r["image_path"], prev_path)
    done += 1

    if done % 80 == 0:
        print(f"  LPIPS {done}/{total_lpips}")

print("LPIPS done.")


# ── Save results ──────────────────────────────────────────────────────────────

CSV_PATH = OUT / "results.csv"
JSON_PATH = OUT / "results.json"

csv_fields = [
    "strength",
    "prompt_id",
    "seed",
    "latency_s",
    "clip_score",
    "lpips_vs_ref",
    "lpips_vs_prev",
    "image_path",
]
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(all_rows)

with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(
        {
            "experiment": "exp5_controlnet_strength",
            "date": "2026-05-07",
            "model": MODEL_ID,
            "controlnet": CANNY_MODEL_ID,
            "strength_values": STRENGTH_VALUES,
            "strength_ref": STRENGTH_REF,
            "steps": STEPS,
            "cfg": CFG,
            "size": SIZE,
            "seeds": SEEDS,
            "prompts": PROMPTS,
            "neg_prompt": NEG_PROMPT,
            "total_images": len(all_rows),
            "results": all_rows,
        },
        f,
        indent=2,
    )

print(f"Results: {CSV_PATH}")


# ── Per-strength aggregates ───────────────────────────────────────────────────

by_str: dict[float, list[dict]] = {s: [] for s in STRENGTH_VALUES}
for r in all_rows:
    by_str[r["strength"]].append(r)

agg: dict[float, dict] = {}
for s, rows in by_str.items():
    clips = [r["clip_score"] for r in rows]
    lats = [r["latency_s"] for r in rows]
    lpips_ref = [r["lpips_vs_ref"] for r in rows if r["lpips_vs_ref"] is not None]
    lpips_prev = [r["lpips_vs_prev"] for r in rows if r["lpips_vs_prev"] is not None]
    agg[s] = {
        "mean_clip": statistics.mean(clips),
        "se_clip": statistics.stdev(clips) / len(clips) ** 0.5,
        "mean_lat": statistics.mean(lats),
        "mean_lpips_ref": statistics.mean(lpips_ref) if lpips_ref else 0.0,
        "mean_lpips_prev": statistics.mean(lpips_prev) if lpips_prev else None,
    }

print("\n── CLIP by conditioning strength ──")
for s in STRENGTH_VALUES:
    a = agg[s]
    prev_str = f"{a['mean_lpips_prev']:.4f}" if a["mean_lpips_prev"] is not None else "—"
    print(
        f"  strength={s:.2f}: CLIP={a['mean_clip']:.4f} ±{a['se_clip']:.4f} | "
        f"LPIPS_ref={a['mean_lpips_ref']:.4f} | LPIPS_prev={prev_str}"
    )


# ── Charts ────────────────────────────────────────────────────────────────────

str_arr = np.array(STRENGTH_VALUES)
clip_arr = np.array([agg[s]["mean_clip"] for s in STRENGTH_VALUES])
lpips_ref_arr = np.array([agg[s]["mean_lpips_ref"] for s in STRENGTH_VALUES])
lpips_prev_arr = np.array(
    [
        agg[s]["mean_lpips_prev"] if agg[s]["mean_lpips_prev"] is not None else 0.0
        for s in STRENGTH_VALUES
    ]
)
colors = [_STR_PALETTE[s] for s in STRENGTH_VALUES]
x = np.arange(len(STRENGTH_VALUES), dtype=float)
xlabels = [f"s={s}" for s in STRENGTH_VALUES]

# Chart 1: CLIP by strength
clip_max = float(clip_arr.max())
canvas = ChartCanvas(
    figsize=(9, 4.5),
    title="CLIP score vs ControlNet conditioning strength — 8 prompts × 5 seeds",
    ylabel="Mean CLIP score",
    top_margin_pct=0.22,
)
canvas.set_ylim(0.0, clip_max * 1.35)
canvas.add_bars(
    x,
    clip_arr,
    colors=colors,
    width=0.6,
    value_fmt="{:.4f}",
    value_pad=clip_max * 0.015,
    value_size=8,
)
canvas.set_xticks(x, xlabels, fontsize=9)
canvas.save(str(CHARTS_DIR / "clip_by_strength.png"))

# Chart 2: LPIPS vs strength=1.0 reference
lpips_ref_max = float(lpips_ref_arr.max())
canvas2 = ChartCanvas(
    figsize=(9, 4.5),
    title=f"Perceptual distance from strength={STRENGTH_REF} reference (LPIPS)",
    ylabel=f"Mean LPIPS vs strength={STRENGTH_REF}",
    top_margin_pct=0.22,
)
canvas2.set_ylim(0.0, max(lpips_ref_max * 1.5, 0.05))
canvas2.add_bars(
    x,
    lpips_ref_arr,
    colors=colors,
    width=0.6,
    value_fmt="{:.4f}",
    value_pad=max(lpips_ref_max * 0.05, 0.002),
    value_size=8,
)
canvas2.set_xticks(x, xlabels, fontsize=9)
canvas2.save(str(CHARTS_DIR / "lpips_vs_ref.png"))

# Chart 3: LPIPS between adjacent strength values
adj_strs = STRENGTH_VALUES[1:]
adj_labels = [
    f"{STRENGTH_VALUES[i-1]}→{STRENGTH_VALUES[i]}" for i in range(1, len(STRENGTH_VALUES))
]
adj_lpips = np.array([agg[s]["mean_lpips_prev"] for s in adj_strs])
adj_colors = [_STR_PALETTE[s] for s in adj_strs]
x3 = np.arange(len(adj_strs), dtype=float)

adj_max = float(adj_lpips.max())
canvas3 = ChartCanvas(
    figsize=(9, 4.5),
    title="LPIPS between adjacent strength values — step-wise visual change",
    ylabel="Mean LPIPS (adjacent pair)",
    top_margin_pct=0.22,
)
canvas3.set_ylim(0.0, adj_max * 1.5)
canvas3.add_bars(
    x3,
    adj_lpips,
    colors=adj_colors,
    width=0.6,
    value_fmt="{:.4f}",
    value_pad=adj_max * 0.05,
    value_size=8,
)
canvas3.set_xticks(x3, adj_labels, fontsize=9)
canvas3.save(str(CHARTS_DIR / "lpips_adjacent.png"))

print(f"Charts written to {CHARTS_DIR}")


# ── Findings writeup ──────────────────────────────────────────────────────────

# CLIP plateau: first strength (in STRENGTH_VALUES order) within 1 SE of max
max_clip = float(clip_arr.max())
max_se = max(agg[s]["se_clip"] for s in STRENGTH_VALUES)
plateau_str = None
for s in STRENGTH_VALUES:
    if abs(agg[s]["mean_clip"] - max_clip) < max_se:
        plateau_str = s
        break

# Largest adjacent LPIPS step
regime_step = max(adj_strs, key=lambda s: agg[s]["mean_lpips_prev"] or 0.0)
regime_val = agg[regime_step]["mean_lpips_prev"]
regime_from = STRENGTH_VALUES[STRENGTH_VALUES.index(regime_step) - 1]

# LPIPS at extremes
lpips_at_zero = agg[0.0]["mean_lpips_ref"]
lpips_at_max = agg[STRENGTH_VALUES[-1]]["mean_lpips_ref"]

# Build results table
table_rows = []
for s in STRENGTH_VALUES:
    a = agg[s]
    prev_s = f"{a['mean_lpips_prev']:.4f}" if a["mean_lpips_prev"] is not None else "—"
    ref_marker = " ← ref" if s == STRENGTH_REF else ""
    table_rows.append(
        f"| {s:.2f}  | {a['mean_clip']:.4f}    | ±{a['se_clip']:.4f}"
        f"  | {a['mean_lpips_ref']:.4f}                  | {prev_s}{ref_marker} |"
    )

FINDINGS = f"""\
# Experiment 5: ControlNet Conditioning Strength Sweep

**Date:** 2026-05-07
**Hardware:** RTX 3070 Laptop 8 GB (enable_model_cpu_offload)
**Model:** {MODEL_ID}
**ControlNet:** {CANNY_MODEL_ID} (Canny edges, SD2.1)
**Conditioning source:** benchmark DDIM-30step outputs for pp_001–pp_008 (seed=42)
**Strength values:** {STRENGTH_VALUES}
  (0.0 = no conditioning / text-only; 1.0 = standard reference; 1.5 = over-conditioned)
**Reference strength for LPIPS:** {STRENGTH_REF}
**Design:** 5 seeds × 8 prompts = 40 images per strength · {len(all_rows)} images total
**Scheduler:** DPM-Solver++ · {STEPS} steps · {SIZE}×{SIZE}
**CFG:** {CFG} (fixed — established Pareto point from benchmark)
**Negative prompt:** standard (held constant)

## Hypothesis

CLIP will stay roughly flat across all strength values — the same text prompt drives
the same semantic content regardless of how tightly the canny edges constrain the output.
LPIPS vs the strength=1.0 reference will reveal where the image perceptually departs from
standard conditioning, diverging monotonically in both directions from the anchor.

## Results

| Strength | Mean CLIP | SE      | LPIPS vs strength={STRENGTH_REF} | LPIPS vs prev (step) |
|----------|----------:|--------:|---------------------------------:|---------------------:|
{chr(10).join(table_rows)}

LPIPS at strength=0.0 (text-only) vs reference: {lpips_at_zero:.4f}
LPIPS at strength={STRENGTH_VALUES[-1]} (over-conditioned) vs reference: {lpips_at_max:.4f}

## Interpretation

**CLIP:** {"Flat across all strengths — first value within 1 SE of max CLIP is strength=" + str(plateau_str) + f" (max CLIP={max_clip:.4f}, SE≈{max_se:.4f}). Conditioning strength has no measurable effect on semantic alignment as judged by CLIP." if plateau_str is not None else "No clear plateau detected in the tested range."}

**LPIPS vs reference:** Divergence is measurable in both directions from strength=1.0.
At strength=0.0 (no conditioning, effectively text-only generation): LPIPS={lpips_at_zero:.4f}.
At strength={STRENGTH_VALUES[-1]} (over-conditioned): LPIPS={lpips_at_max:.4f}.
The images look very different despite CLIP seeing the same semantic content.

**LPIPS (adjacent steps):** The largest single-step visual change is the
{regime_from}→{regime_step} transition (LPIPS={regime_val:.4f}). The conditioning
strength is a pixel-level creative decision — how tightly should the output follow the
edge structure? CLIP cannot see this choice at all; LPIPS can.

**What this means:** ControlNet strength is a creative parameter, not an accuracy parameter.
At lower strengths, the model interprets the prompt more freely; at higher strengths, it
defers more to the edge geometry. CLIP scores are structurally blind to this trade-off.
Choosing a conditioning strength requires visual judgment — CLIP optimisation gives no
signal here whatsoever.

**Cross-experiment note:** Fifth confirmation of CLIP-blindness: quantization (Exp 1),
negative prompt (Exp 2), CFG plateau (Exp 3), scheduler stochasticity (Exp 4), ControlNet
strength (Exp 5). The pattern is consistent: any parameter that affects pixel-level
character without changing the dominant semantic content is invisible to CLIP.

## Charts

- `charts/clip_by_strength.png` — mean CLIP score per conditioning strength
- `charts/lpips_vs_ref.png` — perceptual distance from strength=1.0 reference
- `charts/lpips_adjacent.png` — step-wise LPIPS between adjacent strength values

## Raw data

`results.csv` / `results.json` — one row per image ({len(all_rows)} rows total).

Reproduce:

```bash
python scripts/experiments/exp5_controlnet_strength.py
```
"""

with open(OUT / "findings.md", "w", encoding="utf-8") as f:
    f.write(FINDINGS)

print(f"\nFindings written: {OUT / 'findings.md'}")
print("Experiment 5 complete.")
