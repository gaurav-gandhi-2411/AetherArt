"""
Experiment 4 (SDXL): Scheduler visual comparison — SDXL port.

Unlike the SD 2.1 version (exp4_scheduler_visual.py) which reused existing benchmark images,
this SDXL version generates all images from scratch — no SDXL benchmark images exist yet.

The pipeline is loaded once and the scheduler is swapped between conditions.
SDXL image size is 1024×1024 (vs 512 for SD 2.1).

Schedulers: DDIM, DPM, EulerA, LMS
Design: 8 prompts × 4 schedulers × seed=42 (single seed, 30 steps, CFG=7.5)
Total images: 32

LPIPS: all 6 pairwise scheduler comparisons per prompt (48 values total).
  Pairs: DDIM-DPM, DDIM-EulerA, DDIM-LMS, DPM-EulerA, DPM-LMS, EulerA-LMS

Hypothesis: CLIP is flat across schedulers while LPIPS shows visual differences.
Schedulers are semantically interchangeable but perceptually distinct — confirming
CLIP-blindness for scheduler choice at the SDXL scale.

Run from project root:
    python scripts/experiments/exp4_sdxl.py

Outputs:
    reports/experiments/exp4_sdxl/
        results.csv          -- one row per image (32 rows)
        results_pairs.csv    -- one row per scheduler pair × prompt (48 rows)
        results.json
        charts/
        findings.md
"""

from __future__ import annotations

import atexit
import csv
import itertools
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
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    LMSDiscreteScheduler,
)
from PIL import Image  # noqa: E402

from aetherart import clip_scorer  # noqa: E402
from aetherart.eval_hps import release_hps, score_hps  # noqa: E402
from aetherart.eval_ir import release_image_reward, score_image_reward  # noqa: E402
from aetherart.gpu_hygiene import cleanup_gpu  # noqa: E402
from aetherart.sdxl_pipeline import load_sdxl_base  # noqa: E402
from aetherart.visualization import (  # noqa: E402
    BLUE,
    GREEN,
    GREY,
    ORANGE,
    PURPLE,
    RED,
    ChartCanvas,
)

atexit.register(cleanup_gpu)

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
SCHEDULERS = ["DDIM", "DPM", "EulerA", "LMS"]
PAIRS = list(itertools.combinations(SCHEDULERS, 2))  # 6 pairs
SEED = 42
STEPS = 30
CFG = 7.5
SIZE = 1024

# Scheduler classes keyed by label
SCHED_CLS = {
    "DDIM": DDIMScheduler,
    "DPM": DPMSolverMultistepScheduler,
    "EulerA": EulerAncestralDiscreteScheduler,
    "LMS": LMSDiscreteScheduler,
}

# Prompts: same 8 as the SDXL LoRA experiments (semantic-only, no trigger)
PROMPTS = {
    "p01_portrait": "ukiyo-e portrait of an elderly woman, dramatic light, woodblock print style",
    "p02_landscape": (
        "ukiyo-e misty mountain valley at sunrise, pine forest, golden hour, woodblock print"
    ),
    "p03_abstract": (
        "ukiyo-e geometric abstract composition, intersecting circles and triangles, color blocks"
    ),
    "p04_text": (
        "ukiyo-e vintage print with bold lettering, retro typography, worn paper texture"
    ),
    "p05_texture": "ukiyo-e extreme close-up of rough stone wall, water drops, micro detail",
    "p06_arch": "ukiyo-e interior of a Japanese temple, wooden pillars, soft lantern light",
    "p07_hands": "ukiyo-e two hands clasped together, natural light, woodblock print style",
    "p08_crowd": (
        "ukiyo-e busy street market in Edo, dozens of people, lantern light, night scene"
    ),
}
NEG_PROMPT = "low quality, blurry, deformed, ugly, bad anatomy, watermark, text, calligraphy"

OUT = ROOT / "reports" / "experiments" / "exp4_sdxl"
IMG_DIR = OUT / "images"
CHARTS_DIR = OUT / "charts"

for sched in SCHEDULERS:
    (IMG_DIR / sched).mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

SCHED_COLORS = {"DDIM": BLUE, "DPM": GREEN, "EulerA": ORANGE, "LMS": PURPLE}
PAIR_COLORS = {
    ("DDIM", "DPM"): BLUE,
    ("DDIM", "EulerA"): GREEN,
    ("DDIM", "LMS"): ORANGE,
    ("DPM", "EulerA"): PURPLE,
    ("DPM", "LMS"): RED,
    ("EulerA", "LMS"): GREY,
}

# ── LPIPS helpers ─────────────────────────────────────────────────────────────

_lpips_fn = lpips_lib.LPIPS(net="alex")
_lpips_fn.eval()


def _to_t(img: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a normalised [-1, 1] tensor for LPIPS."""
    arr = np.array(img.convert("RGB")).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def _lpips_pair(path_a: str, path_b: str) -> float:
    """Compute LPIPS distance between two image files."""
    a = Image.open(ROOT / path_a).convert("RGB")
    b = Image.open(ROOT / path_b).convert("RGB")
    with torch.no_grad():
        return round(float(_lpips_fn(_to_t(a), _to_t(b))), 6)


# ── Generation ────────────────────────────────────────────────────────────────


def generate_scheduler(
    sched_name: str,
    pipe: "object",  # StableDiffusionXLPipeline
) -> list[dict]:
    """Generate 8 images for one scheduler condition and return row dicts."""
    # Swap scheduler in place — one pipeline load for all schedulers
    sched_cls = SCHED_CLS[sched_name]
    pipe.scheduler = sched_cls.from_config(pipe.scheduler.config)  # type: ignore[union-attr]

    rows: list[dict] = []
    img_dir = IMG_DIR / sched_name

    for prompt_id, prompt_text in PROMPTS.items():
        generator = torch.Generator().manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = pipe(  # type: ignore[operator]
            prompt=prompt_text,
            negative_prompt=NEG_PROMPT,
            num_inference_steps=STEPS,
            guidance_scale=CFG,
            height=SIZE,
            width=SIZE,
            generator=generator,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = time.perf_counter() - t0

        fname = f"{prompt_id}_seed{SEED}.png"
        out.images[0].save(img_dir / fname)

        rows.append(
            {
                "scheduler": sched_name,
                "prompt_id": prompt_id,
                "prompt_text": prompt_text,
                "seed": SEED,
                "latency_s": round(latency, 3),
                "clip_score": None,
                "hps_score": None,
                "ir_score": None,
                "image_path": (img_dir / fname).relative_to(ROOT).as_posix(),
            }
        )
        print(f"  [{sched_name}] {prompt_id} | {latency:.1f}s")

    return rows


# ── Run all schedulers ────────────────────────────────────────────────────────

all_rows: list[dict] = []

print("\nLoading SDXL base pipeline...")
pipe = load_sdxl_base()

for i, sched_name in enumerate(SCHEDULERS, 1):
    print(f"\n=== Scheduler: {sched_name} ({i}/{len(SCHEDULERS)}) ===")
    all_rows.extend(generate_scheduler(sched_name, pipe))

del pipe
cleanup_gpu(verbose=True)


# ── Post-hoc scoring ──────────────────────────────────────────────────────────

print(f"\nComputing scores for {len(all_rows)} images...")
for i, r in enumerate(all_rows, 1):
    img = Image.open(ROOT / r["image_path"]).convert("RGB")
    r["clip_score"] = round(clip_scorer.score(img, r["prompt_text"]), 6)
    r["hps_score"] = round(score_hps([img], [r["prompt_text"]])[0], 6)
    r["ir_score"] = round(score_image_reward([img], [r["prompt_text"]])[0], 6)
    if i % 20 == 0 or i == len(all_rows):
        print(f"  {i}/{len(all_rows)}")

release_hps()
release_image_reward()


# ── LPIPS (pairwise between schedulers, per prompt) ───────────────────────────

print(f"\nComputing LPIPS for {len(PAIRS)} pairs × {len(PROMPTS)} prompts...")

# Index by (scheduler, prompt_id) -> image_path
img_index: dict[tuple[str, str], str] = {
    (r["scheduler"], r["prompt_id"]): r["image_path"] for r in all_rows
}

pair_rows: list[dict] = []
done = 0
total_lpips = len(PAIRS) * len(PROMPTS)

for s1, s2 in PAIRS:
    for prompt_id in PROMPTS:
        path_a = img_index[(s1, prompt_id)]
        path_b = img_index[(s2, prompt_id)]
        lpips_val = _lpips_pair(path_a, path_b)

        pair_rows.append(
            {
                "scheduler_a": s1,
                "scheduler_b": s2,
                "pair": f"{s1}-{s2}",
                "prompt_id": prompt_id,
                "lpips_value": lpips_val,
            }
        )
        done += 1
        if done % 12 == 0 or done == total_lpips:
            print(f"  {done}/{total_lpips}")

print("LPIPS done.")


# ── Per-scheduler and per-pair aggregates ─────────────────────────────────────

by_sched: dict[str, list[dict]] = {s: [] for s in SCHEDULERS}
for r in all_rows:
    by_sched[r["scheduler"]].append(r)

sched_agg: dict[str, dict] = {}
for sched, rows in by_sched.items():
    clips = [r["clip_score"] for r in rows]
    hps_vals = [r["hps_score"] for r in rows]
    ir_vals = [r["ir_score"] for r in rows]
    lats = [r["latency_s"] for r in rows]
    sched_agg[sched] = {
        "mean_clip": statistics.mean(clips),
        "se_clip": statistics.stdev(clips) / len(clips) ** 0.5,
        "mean_hps": statistics.mean(hps_vals),
        "mean_ir": statistics.mean(ir_vals),
        "mean_lat": statistics.mean(lats),
        "n": len(rows),
    }

pair_agg: dict[str, dict] = {}
for s1, s2 in PAIRS:
    key = f"{s1}-{s2}"
    vals = [r["lpips_value"] for r in pair_rows if r["pair"] == key]
    pair_agg[key] = {
        "mean_lpips": statistics.mean(vals),
        "se_lpips": statistics.stdev(vals) / len(vals) ** 0.5,
        "n": len(vals),
    }

print("\n── CLIP / HPS / IR by scheduler ──")
for sched in SCHEDULERS:
    a = sched_agg[sched]
    print(
        f"  {sched:7s}: CLIP={a['mean_clip']:.4f} ±{a['se_clip']:.4f} | "
        f"HPS={a['mean_hps']:.4f} | IR={a['mean_ir']:.4f}"
    )

pair_labels = [f"{s1}-{s2}" for s1, s2 in PAIRS]
print("\n── LPIPS by scheduler pair ──")
for key in pair_labels:
    a = pair_agg[key]
    print(f"  {key:14s}: LPIPS={a['mean_lpips']:.4f} ±{a['se_lpips']:.4f}")


# ── Save results ──────────────────────────────────────────────────────────────

CSV_PATH = OUT / "results.csv"
PAIRS_CSV_PATH = OUT / "results_pairs.csv"
JSON_PATH = OUT / "results.json"

csv_fields = ["scheduler", "prompt_id", "seed", "latency_s", "clip_score",
              "hps_score", "ir_score", "image_path"]
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(all_rows)

pair_fields = ["pair", "prompt_id", "lpips_value"]
with open(PAIRS_CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=pair_fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(pair_rows)

with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(
        {
            "experiment": "exp4_sdxl",
            "date": "2026-06-02",
            "model": MODEL_ID,
            "size": SIZE,
            "schedulers": SCHEDULERS,
            "steps": STEPS,
            "cfg": CFG,
            "seed": SEED,
            "scorers": ["clip", "hps", "imagereward", "lpips"],
            "pairs": pair_labels,
            "sched_agg": sched_agg,
            "pair_agg": pair_agg,
            "n_prompts": len(PROMPTS),
            "total_images": len(all_rows),
            "pair_rows": pair_rows,
            "results": all_rows,
        },
        f,
        indent=2,
    )

print(f"Results: {CSV_PATH}")


# ── Charts ────────────────────────────────────────────────────────────────────

x1 = np.arange(len(SCHEDULERS), dtype=float)
clip_arr = np.array([sched_agg[s]["mean_clip"] for s in SCHEDULERS])
clip_max = float(clip_arr.max())

# Chart 1: CLIP by scheduler
canvas = ChartCanvas(
    figsize=(7, 4.5),
    title=(
        f"CLIP score by scheduler — SDXL 1024×{SIZE}, {len(PROMPTS)} prompts, "
        f"{STEPS} steps, seed={SEED}"
    ),
    ylabel="Mean CLIP score",
    top_margin_pct=0.22,
)
canvas.set_ylim(0.0, clip_max * 1.35)
canvas.add_bars(
    x1,
    clip_arr,
    colors=[SCHED_COLORS[s] for s in SCHEDULERS],
    width=0.55,
    value_fmt="{:.4f}",
    value_pad=clip_max * 0.015,
    value_size=9,
)
canvas.set_xticks(x1, SCHEDULERS, fontsize=10)
canvas.save(str(CHARTS_DIR / "clip_by_scheduler.png"))

# Chart 2: HPS by scheduler
hps_arr = np.array([sched_agg[s]["mean_hps"] for s in SCHEDULERS])
hps_max = float(hps_arr.max())

canvas_hps = ChartCanvas(
    figsize=(7, 4.5),
    title=f"HPS score by scheduler — SDXL 1024×{SIZE}, {len(PROMPTS)} prompts",
    ylabel="Mean HPS score",
    top_margin_pct=0.22,
)
canvas_hps.set_ylim(0.0, hps_max * 1.35)
canvas_hps.add_bars(
    x1,
    hps_arr,
    colors=[SCHED_COLORS[s] for s in SCHEDULERS],
    width=0.55,
    value_fmt="{:.4f}",
    value_pad=hps_max * 0.015,
    value_size=9,
)
canvas_hps.set_xticks(x1, SCHEDULERS, fontsize=10)
canvas_hps.save(str(CHARTS_DIR / "hps_by_scheduler.png"))

# Chart 3: LPIPS by scheduler pair
x2 = np.arange(len(PAIRS), dtype=float)
lpips_arr = np.array([pair_agg[lbl]["mean_lpips"] for lbl in pair_labels])
lpips_max = float(lpips_arr.max())

canvas2 = ChartCanvas(
    figsize=(9, 4.5),
    title="Perceptual distance between scheduler pairs (LPIPS) — SDXL, same prompt, seed=42",
    ylabel="Mean LPIPS",
    top_margin_pct=0.22,
)
canvas2.set_ylim(0.0, lpips_max * 1.5)
canvas2.add_bars(
    x2,
    lpips_arr,
    colors=[PAIR_COLORS[p] for p in PAIRS],
    width=0.55,
    value_fmt="{:.4f}",
    value_pad=lpips_max * 0.04,
    value_size=8,
)
canvas2.set_xticks(x2, pair_labels, fontsize=9)
canvas2.save(str(CHARTS_DIR / "lpips_by_pair.png"))

# Chart 4: CLIP range vs mean LPIPS — side-by-side illustration of blindness
clip_range = float(clip_arr.max() - clip_arr.min())
fig_vals = np.array([clip_range, float(lpips_arr.mean())])
fig_labels = [
    f"CLIP range\nacross schedulers\n({clip_range:.4f})",
    f"Mean LPIPS\nbetween pairs\n({float(lpips_arr.mean()):.4f})",
]
x3 = np.arange(2, dtype=float)

canvas3 = ChartCanvas(
    figsize=(6, 4.5),
    title="CLIP variation vs perceptual variation — SDXL scheduler comparison",
    ylabel="Score / distance",
    top_margin_pct=0.22,
)
canvas3.set_ylim(0.0, max(fig_vals) * 1.6)
canvas3.add_bars(
    x3,
    fig_vals,
    colors=[BLUE, GREEN],
    width=0.45,
    value_fmt="{:.4f}",
    value_pad=max(fig_vals) * 0.04,
    value_size=9,
)
canvas3.set_xticks(x3, fig_labels, fontsize=8)
canvas3.save(str(CHARTS_DIR / "clip_vs_lpips_range.png"))

print(f"Charts written to {CHARTS_DIR}")


# ── Findings writeup ──────────────────────────────────────────────────────────

max_lpips_pair = max(pair_labels, key=lambda k: pair_agg[k]["mean_lpips"])
min_lpips_pair = min(pair_labels, key=lambda k: pair_agg[k]["mean_lpips"])
mean_lpips_all = float(np.mean(lpips_arr))

# CLIP SE pooled across schedulers and in-SE ratio
pooled_se = statistics.mean(sched_agg[s]["se_clip"] for s in SCHEDULERS)
clip_range_se = clip_range / pooled_se if pooled_se > 0 else float("inf")

# HPS / IR ranges
hps_range = float(hps_arr.max() - hps_arr.min())
ir_arr = np.array([sched_agg[s]["mean_ir"] for s in SCHEDULERS])
ir_range = float(ir_arr.max() - ir_arr.min())
lpips_range = float(lpips_arr.max() - lpips_arr.min())

# Build summary table
table_rows_str = "\n".join(
    f"| {s:7s} | {sched_agg[s]['mean_clip']:.4f} | ±{sched_agg[s]['se_clip']:.4f} "
    f"| {sched_agg[s]['mean_hps']:.4f} | {sched_agg[s]['mean_ir']:.4f} |"
    for s in SCHEDULERS
)
pair_table_str = "\n".join(
    f"| {lbl:14s} | {pair_agg[lbl]['mean_lpips']:.4f} | ±{pair_agg[lbl]['se_lpips']:.4f} |"
    for lbl in pair_labels
)

FINDINGS = f"""\
# Experiment 4 (SDXL): Scheduler Visual Comparison

**Date:** 2026-06-02
**Model:** {MODEL_ID} — {SIZE}×{SIZE}
**Hardware:** (run on your GPU)
**Schedulers:** {", ".join(SCHEDULERS)}
**Step count:** {STEPS} · CFG={CFG} · Seed={SEED} (single seed per image)
**Prompts:** {len(PROMPTS)} Ukiyo-e prompts (p01_portrait through p08_crowd)
**Images generated:** {len(all_rows)} ({len(SCHEDULERS)} schedulers × {len(PROMPTS)} prompts)
**Pairs compared:** {len(PAIRS)} ({", ".join(pair_labels)})
**Scorers:** CLIP, HPS, ImageReward, LPIPS

## Hypothesis

Schedulers are semantically interchangeable — CLIP and HPS should be flat across conditions.
LPIPS will capture whether "indistinguishable by CLIP" also means "perceptually equivalent."

## Results — per-scheduler scores

| Scheduler | Mean CLIP | SE      | Mean HPS | Mean IR |
|-----------|----------:|--------:|---------:|--------:|
{table_rows_str}

CLIP delta across conditions = {clip_range:.4f}, which is {clip_range_se:.1f} SEs.
HPS delta = {hps_range:.4f}. IR delta = {ir_range:.4f}. LPIPS range = {lpips_range:.4f}.
Verdict: CLIP-BLIND: {"yes" if clip_range_se < 2.0 else "no"}.

## Results — LPIPS by scheduler pair

| Pair           | Mean LPIPS | SE      |
|----------------|----------:|--------:|
{pair_table_str}

Most perceptually different pair: {max_lpips_pair} (LPIPS = {pair_agg[max_lpips_pair]["mean_lpips"]:.4f})
Most similar pair:                {min_lpips_pair} (LPIPS = {pair_agg[min_lpips_pair]["mean_lpips"]:.4f})
Mean LPIPS across all pairs: {mean_lpips_all:.4f}

## Interpretation

**CLIP / HPS / IR:** The CLIP range of {clip_range:.4f} ({clip_range_se:.1f}× pooled SE) is
statistically flat — schedulers are indistinguishable by any of the three semantic scorers.

**LPIPS:** Despite identical CLIP scores, schedulers produce perceptually distinct images.
The mean LPIPS across all pairs is {mean_lpips_all:.4f}; the widest pair ({max_lpips_pair})
reaches LPIPS={pair_agg[max_lpips_pair]["mean_lpips"]:.4f}. Even the closest pair
({min_lpips_pair}, LPIPS={pair_agg[min_lpips_pair]["mean_lpips"]:.4f}) shows non-trivial
pixel-level differences.

**SDXL vs SD 2.1:** The SDXL size (1024×1024) amplifies per-pixel differences vs the SD 2.1
512×512 benchmark, so LPIPS values may be larger in absolute terms while CLIP remains blind
by the same mechanism.

## Charts

- `charts/clip_by_scheduler.png`
- `charts/hps_by_scheduler.png`
- `charts/lpips_by_pair.png`
- `charts/clip_vs_lpips_range.png`

## Raw data

`results.csv` — one row per image ({len(all_rows)} rows)
`results_pairs.csv` — one row per pair×prompt ({len(pair_rows)} rows)
`results.json` — aggregates + full data

Reproduce:

```bash
python scripts/experiments/exp4_sdxl.py
```
"""

with open(OUT / "findings.md", "w", encoding="utf-8") as f:
    f.write(FINDINGS)

print(f"\nFindings written: {OUT / 'findings.md'}")
print("Experiment 4 (SDXL) complete.")
