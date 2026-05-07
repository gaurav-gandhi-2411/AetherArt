"""
Experiment 2: Negative prompt impact.

Conditions: no_neg (empty negative prompt), with_neg (standard negative prompt)
Seeds: 5 fixed seeds x 8 prompts = 40 images per condition, 80 total
Metrics: CLIP score, LPIPS between conditions (same seed/prompt pair), latency

Hypothesis: the negative prompt reduces visual artifacts but CLIP score may
decrease slightly because guidance away from "bad" tokens reduces the energy
available for positive alignment. An increase would be equally informative.

Run from project root:
    python scripts/experiments/exp2_negative_prompt.py

Outputs:
    reports/experiments/exp2_negative_prompt/
        images/{no_neg,with_neg}/
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
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline  # noqa: E402
from PIL import Image  # noqa: E402

from aetherart import clip_scorer  # noqa: E402
from aetherart.gpu_hygiene import cleanup_gpu  # noqa: E402
from aetherart.visualization import BLUE, GREEN, GREY, ChartCanvas  # noqa: E402

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

CONDITIONS = {
    "no_neg": "",
    "with_neg": NEG_PROMPT,
}

OUT = ROOT / "reports" / "experiments" / "exp2_negative_prompt"
IMG_DIR = OUT / "images"
CHARTS_DIR = OUT / "charts"

for _d in [IMG_DIR / "no_neg", IMG_DIR / "with_neg", CHARTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

COND_COLORS = {"no_neg": GREY, "with_neg": BLUE}
COND_LABELS = {"no_neg": "No negative prompt", "with_neg": "Standard negative prompt"}


# ── Pipeline ──────────────────────────────────────────────────────────────────


def load_pipeline() -> StableDiffusionPipeline:
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if torch.cuda.is_available():
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pipe = pipe.to("cuda")
    return pipe


# ── Generation loop ───────────────────────────────────────────────────────────


def run_condition(label: str, neg_prompt: str, pipe: StableDiffusionPipeline) -> list[dict]:
    rows: list[dict] = []
    img_dir = IMG_DIR / label
    for prompt_id, prompt_text in PROMPTS.items():
        for seed in SEEDS:
            generator = torch.Generator().manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = pipe(
                prompt=prompt_text,
                negative_prompt=neg_prompt if neg_prompt else None,
                num_inference_steps=STEPS,
                guidance_scale=GUIDANCE,
                height=SIZE,
                width=SIZE,
                generator=generator,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latency = time.perf_counter() - t0

            fname = f"{prompt_id}_seed{seed}.png"
            out.images[0].save(img_dir / fname)

            rows.append(
                {
                    "condition": label,
                    "prompt_id": prompt_id,
                    "prompt_text": prompt_text,
                    "seed": seed,
                    "latency_s": round(latency, 3),
                    "clip_score": None,
                    "lpips": None,
                    "image_path": (img_dir / fname).relative_to(ROOT).as_posix(),
                }
            )
            print(f"  [{label}] {prompt_id} seed={seed:5d} | {latency:.1f}s")
    return rows


# ── Main: single pipeline, both conditions ────────────────────────────────────

all_rows: list[dict] = []

print("\nLoading fp16 pipeline (shared across both conditions)...")
pipe = load_pipeline()

print("\n=== Condition 1/2: no_neg ===")
all_rows.extend(run_condition("no_neg", CONDITIONS["no_neg"], pipe))

print("\n=== Condition 2/2: with_neg ===")
all_rows.extend(run_condition("with_neg", CONDITIONS["with_neg"], pipe))

del pipe
cleanup_gpu(verbose=True)

# ── CLIP scores (post-hoc) ────────────────────────────────────────────────────

print(f"\nComputing CLIP scores for {len(all_rows)} images...")
for i, r in enumerate(all_rows, 1):
    img = Image.open(ROOT / r["image_path"]).convert("RGB")
    r["clip_score"] = round(clip_scorer.score(img, r["prompt_text"]), 6)
    if i % 20 == 0 or i == len(all_rows):
        print(f"  {i}/{len(all_rows)}")

# ── LPIPS between conditions (same seed/prompt pair) ─────────────────────────

print("\nComputing LPIPS between conditions (no_neg vs with_neg)...")
_lpips_fn = lpips_lib.LPIPS(net="alex")
_lpips_fn.eval()

no_neg_by_key: dict[tuple, str] = {
    (r["prompt_id"], r["seed"]): r["image_path"] for r in all_rows if r["condition"] == "no_neg"
}


def _to_lpips_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img.convert("RGB")).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


for r in all_rows:
    if r["condition"] == "no_neg":
        # no reference to compare against for this direction; fill symmetrically below
        r["lpips"] = None
        continue
    ref = Image.open(ROOT / no_neg_by_key[(r["prompt_id"], r["seed"])]).convert("RGB")
    cmp = Image.open(ROOT / r["image_path"]).convert("RGB")
    with torch.no_grad():
        val = round(float(_lpips_fn(_to_lpips_tensor(ref), _to_lpips_tensor(cmp))), 6)
    r["lpips"] = val
    # mirror onto the no_neg row so every row has a value
    no_neg_row = next(
        x
        for x in all_rows
        if x["condition"] == "no_neg"
        and x["prompt_id"] == r["prompt_id"]
        and x["seed"] == r["seed"]
    )
    no_neg_row["lpips"] = val

print("LPIPS done.")

# ── Save results ──────────────────────────────────────────────────────────────

CSV_PATH = OUT / "results.csv"
JSON_PATH = OUT / "results.json"

csv_fields = ["condition", "prompt_id", "seed", "latency_s", "clip_score", "lpips", "image_path"]
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(all_rows)

with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(
        {
            "experiment": "exp2_negative_prompt",
            "date": "2026-05-07",
            "model": MODEL_ID,
            "steps": STEPS,
            "guidance": GUIDANCE,
            "size": SIZE,
            "seeds": SEEDS,
            "prompts": PROMPTS,
            "conditions": {k: v for k, v in CONDITIONS.items()},
            "total_images": len(all_rows),
            "results": all_rows,
        },
        f,
        indent=2,
    )

print(f"Results: {CSV_PATH}\n         {JSON_PATH}")

# ── Per-condition aggregates ──────────────────────────────────────────────────

by_cond: dict[str, list[dict]] = {"no_neg": [], "with_neg": []}
for r in all_rows:
    by_cond[r["condition"]].append(r)

agg: dict[str, dict] = {}
for cond, rows in by_cond.items():
    clips = [r["clip_score"] for r in rows]
    lats = [r["latency_s"] for r in rows]
    lpips_vals = [r["lpips"] for r in rows if r["lpips"] is not None]
    agg[cond] = {
        "n": len(rows),
        "mean_clip": statistics.mean(clips),
        "se_clip": statistics.stdev(clips) / len(clips) ** 0.5,
        "mean_lat": statistics.mean(lats),
        "mean_lpips": statistics.mean(lpips_vals) if lpips_vals else 0.0,
        "se_lpips": (
            statistics.stdev(lpips_vals) / len(lpips_vals) ** 0.5 if len(lpips_vals) > 1 else 0.0
        ),
    }

print("\n── Aggregates ──")
for cond in ["no_neg", "with_neg"]:
    a = agg[cond]
    print(
        f"  {cond:8s}: CLIP={a['mean_clip']:.4f} ±{a['se_clip']:.4f} | "
        f"lat={a['mean_lat']:.1f}s | LPIPS={a['mean_lpips']:.4f}"
    )

clip_delta = agg["with_neg"]["mean_clip"] - agg["no_neg"]["mean_clip"]
lpips_mean = agg["with_neg"]["mean_lpips"]  # symmetric; same for both conditions
print(f"\n  CLIP delta (with_neg − no_neg): {clip_delta:+.4f}")
print(f"  Mean LPIPS between conditions:  {lpips_mean:.4f}")

# ── Charts ────────────────────────────────────────────────────────────────────

COND_ORDER = ["no_neg", "with_neg"]
x = np.arange(len(COND_ORDER), dtype=float)

# Chart 1: CLIP score by condition
clip_vals = np.array([agg[c]["mean_clip"] for c in COND_ORDER])
clip_max = float(clip_vals.max())
clip_min = float(clip_vals.min())
clip_range = clip_max - clip_min

canvas = ChartCanvas(
    figsize=(6, 4.5),
    title="CLIP score: no negative prompt vs standard negative prompt",
    ylabel="Mean CLIP score",
    top_margin_pct=0.22,
)
canvas.set_ylim(0.0, clip_max * 1.35)
canvas.add_bars(
    x,
    clip_vals,
    colors=[COND_COLORS[c] for c in COND_ORDER],
    width=0.55,
    value_fmt="{:.4f}",
    value_pad=clip_max * 0.015,
    value_size=9,
)
canvas.set_xticks(x, [COND_LABELS[c] for c in COND_ORDER], fontsize=9)
canvas.save(str(CHARTS_DIR / "clip_by_condition.png"))

# Chart 2: LPIPS between conditions (single bar — symmetric)
canvas2 = ChartCanvas(
    figsize=(5, 4.5),
    title="Perceptual distance between conditions (LPIPS)",
    ylabel="Mean LPIPS (no_neg vs with_neg)",
    top_margin_pct=0.22,
)
canvas2.set_ylim(0.0, lpips_mean * 1.7)
canvas2.add_bars(
    np.array([0.0]),
    np.array([lpips_mean]),
    colors=[GREEN],
    width=0.45,
    value_fmt="{:.4f}",
    value_pad=lpips_mean * 0.06,
    value_size=10,
)
canvas2.set_xticks(np.array([0.0]), ["No neg vs With neg"], fontsize=9)
canvas2.save(str(CHARTS_DIR / "lpips_between_conditions.png"))

# Chart 3: Per-prompt CLIP delta (with_neg − no_neg)
prompt_ids = list(PROMPTS.keys())
by_prompt_cond: dict[tuple, list[float]] = {}
for r in all_rows:
    key = (r["condition"], r["prompt_id"])
    by_prompt_cond.setdefault(key, []).append(r["clip_score"])

prompt_deltas = np.array(
    [
        statistics.mean(by_prompt_cond[("with_neg", pid)])
        - statistics.mean(by_prompt_cond[("no_neg", pid)])
        for pid in prompt_ids
    ]
)
colors_delta = [BLUE if d >= 0 else GREY for d in prompt_deltas]

x3 = np.arange(len(prompt_ids), dtype=float)
canvas3 = ChartCanvas(
    figsize=(10, 4.5),
    title="Per-prompt CLIP delta: with_neg minus no_neg (positive = negative prompt helped)",
    ylabel="CLIP delta",
    top_margin_pct=0.22,
)
# Include negative bars: use a range that accommodates both directions
delta_abs_max = float(np.abs(prompt_deltas).max())
canvas3.set_ylim(-delta_abs_max * 2.5, delta_abs_max * 2.5)
canvas3.add_bars(
    x3,
    prompt_deltas,
    colors=colors_delta,
    width=0.6,
    value_fmt="{:+.4f}",
    value_pad=delta_abs_max * 0.08,
    value_size=8,
)
canvas3.set_xticks(x3, [pid.replace("_", " ") for pid in prompt_ids], fontsize=8)
canvas3.save(str(CHARTS_DIR / "clip_delta_by_prompt.png"))

print(f"Charts written to {CHARTS_DIR}")

# ── Findings writeup ──────────────────────────────────────────────────────────

no_a = agg["no_neg"]
wn_a = agg["with_neg"]
pooled_se = (no_a["se_clip"] + wn_a["se_clip"]) / 2


def _clip_delta_verdict(delta: float, se: float) -> str:
    if abs(delta) < se:
        return f"within 1 SE (delta = {delta:+.4f}) — no reliable effect"
    if abs(delta) < 2 * se:
        return f"between 1 and 2 SE (delta = {delta:+.4f}) — marginal, not reliable"
    return (
        f"{'above' if delta > 0 else 'below'} 2 SE (delta = {delta:+.4f}) — "
        f"statistically distinguishable"
    )


def _lpips_context(val: float) -> str:
    if val < 0.05:
        return f"near-identical images regardless of negative prompt (LPIPS = {val:.4f})"
    if val < 0.15:
        return f"minor pixel differences between conditions (LPIPS = {val:.4f})"
    if val < 0.30:
        return f"moderate pixel differences between conditions (LPIPS = {val:.4f})"
    return f"substantial pixel differences between conditions (LPIPS = {val:.4f})"


FINDINGS = f"""\
# Experiment 2: Negative Prompt Impact

**Date:** 2026-05-07
**Hardware:** RTX 3070 Laptop 8 GB (enable_model_cpu_offload)
**Model:** {MODEL_ID}
**Conditions:** no_neg (empty negative prompt) · with_neg (standard negative prompt)
**Design:** 5 seeds x 8 prompts = 40 images per condition · 80 images total
**Scheduler:** DPM-Solver++ · {STEPS} steps · {SIZE}x{SIZE} · CFG={GUIDANCE}
**Negative prompt tested:** `{NEG_PROMPT}`

## Hypothesis

The standard negative prompt guides the model away from degenerate outputs (blurry, deformed,
watermarked). This should reduce artifacts and may increase CLIP score if the negative tokens
overlap with semantically poor regions, or decrease it if the guidance energy is reallocated
away from positive alignment. A null result (no reliable CLIP difference) is also plausible —
negative prompts primarily reshape the output distribution at the tails.

## Results

| Condition | Mean CLIP | Mean latency (s) | Mean LPIPS (vs other condition) |
|-----------|----------:|-----------------:|--------------------------------:|
| no_neg    | {no_a['mean_clip']:.4f}    | {no_a['mean_lat']:.1f}s             | {no_a['mean_lpips']:.4f}                         |
| with_neg  | {wn_a['mean_clip']:.4f}    | {wn_a['mean_lat']:.1f}s             | {wn_a['mean_lpips']:.4f}                         |

SE on CLIP: no_neg ±{no_a['se_clip']:.4f} · with_neg ±{wn_a['se_clip']:.4f}
CLIP delta (with_neg − no_neg): {clip_delta:+.4f} — {_clip_delta_verdict(clip_delta, pooled_se)}

LPIPS between conditions (same seed/prompt pair): {lpips_mean:.4f} — {_lpips_context(lpips_mean)}

## Per-prompt breakdown

See `charts/clip_delta_by_prompt.png`. Positive bars = negative prompt improved CLIP for that
prompt category; negative bars = negative prompt hurt CLIP. Variance across prompts reveals
whether the effect is consistent or prompt-dependent.

## Interpretation

The CLIP delta is {_clip_delta_verdict(clip_delta, pooled_se)}.
The LPIPS of {lpips_mean:.4f} between conditions tells us the negative prompt {_lpips_context(lpips_mean)}.

Latency difference: {wn_a['mean_lat'] - no_a['mean_lat']:+.2f}s — negative prompt text adds
minimal compute overhead (classifier-free guidance already processes a null embedding; replacing
it with a non-empty prompt does not change the number of forward passes).

## Charts

- `charts/clip_by_condition.png` — mean CLIP score per condition
- `charts/lpips_between_conditions.png` — perceptual distance between matched pairs
- `charts/clip_delta_by_prompt.png` — per-prompt CLIP delta (with_neg minus no_neg)

## Raw data

`results.csv` / `results.json` — one row per image (80 rows total).

Reproduce:

```bash
python scripts/experiments/exp2_negative_prompt.py
```
"""

with open(OUT / "findings.md", "w", encoding="utf-8") as f:
    f.write(FINDINGS)

print(f"\nFindings written: {OUT / 'findings.md'}")
print("Experiment 2 complete.")
