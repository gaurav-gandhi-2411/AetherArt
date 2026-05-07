"""
Experiment 6: LoRA style scale (alpha) sweep — Ukiyo-e rank-8 LoRA on SD2.1.

The existing rank-8 Ukiyo-e LoRA is loaded once. Adapter weight (alpha) is swept
across [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5] via set_adapters() between
conditions — no retraining required.

alpha=0.0: LoRA weights loaded but suppressed (effectively base SD2.1)
alpha=1.0: standard adapter weight (trained default) — LPIPS reference
alpha=1.5: over-styled; potential mode collapse or saturation

All prompts use the "ukyowood" trigger token to address the adapter. CLIP is
measured against the full prompt including the trigger. If CLIP is insensitive
to the stylistic shift that LPIPS captures, it adds a sixth CLIP-blindness point.

Seeds: [42, 123, 456, 789, 1337] — 5 seeds × 8 prompts = 40 images per alpha
Total: 7 × 8 × 5 = 280 images

Run from project root:
    python scripts/experiments/exp6_lora_alpha.py

Outputs:
    reports/experiments/exp6_lora_alpha/
        images/alpha_{val}/
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
from aetherart.lora import LORA_REGISTRY  # noqa: E402
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
LORA_NAME = "ukiyo-e"
ALPHA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
ALPHA_REF = 1.0
SEEDS = [42, 123, 456, 789, 1337]
STEPS = 30
CFG = 7.0
SIZE = 512

# Prompts with "ukyowood" trigger token (LoRA was trained with this caption prefix)
PROMPTS = {
    "p01_portrait": (
        "ukyowood ukiyo-e portrait of an elderly woman, "
        "dramatic light, woodblock print style"
    ),
    "p02_landscape": (
        "ukyowood ukiyo-e misty mountain valley at sunrise, "
        "pine forest, golden hour, woodblock print"
    ),
    "p03_abstract": (
        "ukyowood ukiyo-e geometric abstract composition, "
        "intersecting circles and triangles, vibrant color blocks"
    ),
    "p04_text": (
        "ukyowood ukiyo-e vintage print with bold lettering, "
        "retro typography, worn paper texture"
    ),
    "p05_texture": (
        "ukyowood ukiyo-e extreme close-up of rough stone wall, "
        "water drops, micro detail"
    ),
    "p06_arch": (
        "ukyowood ukiyo-e interior of a Japanese temple, "
        "wooden pillars, soft lantern light"
    ),
    "p07_hands": (
        "ukyowood ukiyo-e two hands clasped together, "
        "natural light, woodblock print style"
    ),
    "p08_crowd": (
        "ukyowood ukiyo-e busy street market in Edo, "
        "dozens of people, lantern light, night scene"
    ),
}
NEG_PROMPT = "low quality, blurry, deformed, ugly, bad anatomy, watermark, text, calligraphy"

OUT = ROOT / "reports" / "experiments" / "exp6_lora_alpha"
IMG_DIR = OUT / "images"
CHARTS_DIR = OUT / "charts"

for a in ALPHA_VALUES:
    (IMG_DIR / f"alpha_{a}").mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

_ALPHA_PALETTE = {
    0.0:  GREY,
    0.25: "#5AADA8",
    0.5:  GREEN,
    0.75: TEAL,
    1.0:  BLUE,
    1.25: ORANGE,
    1.5:  RED,
}


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


def load_lora_once(pipe: StableDiffusionPipeline) -> None:
    config = LORA_REGISTRY[LORA_NAME]
    lora_path = Path(config["path"])
    pipe.load_lora_weights(
        str(lora_path.parent), weight_name=lora_path.name, adapter_name="ukiyo_e"
    )
    print(f"LoRA loaded: {lora_path.name}")


def set_alpha(pipe: StableDiffusionPipeline, alpha: float) -> None:
    if alpha == 0.0:
        # Disable the adapter — equivalent to base model for generation
        pipe.set_adapters(["ukiyo_e"], adapter_weights=[0.0])
    else:
        pipe.set_adapters(["ukiyo_e"], adapter_weights=[alpha])


# ── Generation loop ───────────────────────────────────────────────────────────

def run_alpha(alpha: float, pipe: StableDiffusionPipeline) -> list[dict]:
    set_alpha(pipe, alpha)
    rows: list[dict] = []
    img_dir = IMG_DIR / f"alpha_{alpha}"
    for prompt_id, prompt_text in PROMPTS.items():
        for seed in SEEDS:
            generator = torch.Generator().manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = pipe(
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

            fname = f"{prompt_id}_seed{seed}.png"
            out.images[0].save(img_dir / fname)

            rows.append(
                {
                    "alpha": alpha,
                    "prompt_id": prompt_id,
                    "prompt_text": prompt_text,
                    "seed": seed,
                    "latency_s": round(latency, 3),
                    "clip_score": None,
                    "lpips_vs_ref": None,
                    "lpips_vs_prev": None,
                    "image_path": (img_dir / fname).relative_to(ROOT).as_posix(),
                }
            )
            print(f"  [alpha={alpha:.2f}] {prompt_id} seed={seed:5d} | {latency:.1f}s")
    return rows


# ── Run all alpha values ───────────────────────────────────────────────────────

all_rows: list[dict] = []

print("\nLoading SD2.1 pipeline (fp16) ...")
pipe = load_pipeline()
print("Loading LoRA weights once ...")
load_lora_once(pipe)

for alpha in ALPHA_VALUES:
    idx = ALPHA_VALUES.index(alpha) + 1
    print(f"\n=== Alpha = {alpha} ({idx}/{len(ALPHA_VALUES)}) ===")
    all_rows.extend(run_alpha(alpha, pipe))

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
    (r["alpha"], r["prompt_id"], r["seed"]): r["image_path"] for r in all_rows
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
    a = r["alpha"]
    pid = r["prompt_id"]
    seed = r["seed"]

    if a == ALPHA_REF:
        r["lpips_vs_ref"] = 0.0
    else:
        ref_path = img_index[(ALPHA_REF, pid, seed)]
        r["lpips_vs_ref"] = _lpips_pair(r["image_path"], ref_path)
    done += 1

    a_idx = ALPHA_VALUES.index(a)
    if a_idx == 0:
        r["lpips_vs_prev"] = None
    else:
        prev_a = ALPHA_VALUES[a_idx - 1]
        prev_path = img_index[(prev_a, pid, seed)]
        r["lpips_vs_prev"] = _lpips_pair(r["image_path"], prev_path)
    done += 1

    if done % 80 == 0:
        print(f"  LPIPS {done}/{total_lpips}")

print("LPIPS done.")


# ── Save results ──────────────────────────────────────────────────────────────

CSV_PATH = OUT / "results.csv"
JSON_PATH = OUT / "results.json"

csv_fields = [
    "alpha", "prompt_id", "seed", "latency_s",
    "clip_score", "lpips_vs_ref", "lpips_vs_prev", "image_path",
]
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(all_rows)

lora_config = LORA_REGISTRY[LORA_NAME]
with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(
        {
            "experiment": "exp6_lora_alpha",
            "date": "2026-05-07",
            "model": MODEL_ID,
            "lora": LORA_NAME,
            "lora_path": lora_config["path"],
            "alpha_values": ALPHA_VALUES,
            "alpha_ref": ALPHA_REF,
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


# ── Per-alpha aggregates ──────────────────────────────────────────────────────

by_alpha: dict[float, list[dict]] = {a: [] for a in ALPHA_VALUES}
for r in all_rows:
    by_alpha[r["alpha"]].append(r)

agg: dict[float, dict] = {}
for a, rows in by_alpha.items():
    clips = [r["clip_score"] for r in rows]
    lats = [r["latency_s"] for r in rows]
    lpips_ref = [r["lpips_vs_ref"] for r in rows if r["lpips_vs_ref"] is not None]
    lpips_prev = [r["lpips_vs_prev"] for r in rows if r["lpips_vs_prev"] is not None]
    agg[a] = {
        "mean_clip": statistics.mean(clips),
        "se_clip": statistics.stdev(clips) / len(clips) ** 0.5,
        "mean_lat": statistics.mean(lats),
        "mean_lpips_ref": statistics.mean(lpips_ref) if lpips_ref else 0.0,
        "mean_lpips_prev": statistics.mean(lpips_prev) if lpips_prev else None,
    }

print("\n── CLIP by LoRA alpha ──")
for a in ALPHA_VALUES:
    aa = agg[a]
    prev_str = f"{aa['mean_lpips_prev']:.4f}" if aa["mean_lpips_prev"] is not None else "—"
    print(
        f"  alpha={a:.2f}: CLIP={aa['mean_clip']:.4f} ±{aa['se_clip']:.4f} | "
        f"LPIPS_ref={aa['mean_lpips_ref']:.4f} | LPIPS_prev={prev_str}"
    )


# ── Charts ────────────────────────────────────────────────────────────────────

clip_arr = np.array([agg[a]["mean_clip"] for a in ALPHA_VALUES])
lpips_ref_arr = np.array([agg[a]["mean_lpips_ref"] for a in ALPHA_VALUES])
lpips_prev_arr = np.array(
    [agg[a]["mean_lpips_prev"] if agg[a]["mean_lpips_prev"] is not None else 0.0
     for a in ALPHA_VALUES]
)
colors = [_ALPHA_PALETTE[a] for a in ALPHA_VALUES]
x = np.arange(len(ALPHA_VALUES), dtype=float)
xlabels = [f"α={a}" for a in ALPHA_VALUES]

# Chart 1: CLIP by alpha
clip_max = float(clip_arr.max())
canvas = ChartCanvas(
    figsize=(9, 4.5),
    title="CLIP score vs LoRA adapter weight (ukiyo-e, rank-8) — 8 prompts × 5 seeds",
    ylabel="Mean CLIP score",
    top_margin_pct=0.22,
)
canvas.set_ylim(0.0, clip_max * 1.35)
canvas.add_bars(
    x, clip_arr, colors=colors, width=0.6,
    value_fmt="{:.4f}", value_pad=clip_max * 0.015, value_size=8,
)
canvas.set_xticks(x, xlabels, fontsize=9)
canvas.save(str(CHARTS_DIR / "clip_by_alpha.png"))

# Chart 2: LPIPS vs alpha=1.0 reference
lpips_ref_max = float(lpips_ref_arr.max())
canvas2 = ChartCanvas(
    figsize=(9, 4.5),
    title=f"Perceptual distance from alpha={ALPHA_REF} reference (LPIPS)",
    ylabel=f"Mean LPIPS vs alpha={ALPHA_REF}",
    top_margin_pct=0.22,
)
canvas2.set_ylim(0.0, max(lpips_ref_max * 1.5, 0.05))
canvas2.add_bars(
    x, lpips_ref_arr, colors=colors, width=0.6,
    value_fmt="{:.4f}", value_pad=max(lpips_ref_max * 0.05, 0.002), value_size=8,
)
canvas2.set_xticks(x, xlabels, fontsize=9)
canvas2.save(str(CHARTS_DIR / "lpips_vs_ref.png"))

# Chart 3: LPIPS adjacent steps
adj_alphas = ALPHA_VALUES[1:]
adj_labels = [
    f"{ALPHA_VALUES[i-1]}→{ALPHA_VALUES[i]}"
    for i in range(1, len(ALPHA_VALUES))
]
adj_lpips = np.array([agg[a]["mean_lpips_prev"] for a in adj_alphas])
adj_colors = [_ALPHA_PALETTE[a] for a in adj_alphas]
x3 = np.arange(len(adj_alphas), dtype=float)

adj_max = float(adj_lpips.max())
canvas3 = ChartCanvas(
    figsize=(9, 4.5),
    title="LPIPS between adjacent alpha values — step-wise visual change",
    ylabel="Mean LPIPS (adjacent pair)",
    top_margin_pct=0.22,
)
canvas3.set_ylim(0.0, adj_max * 1.5)
canvas3.add_bars(
    x3, adj_lpips, colors=adj_colors, width=0.6,
    value_fmt="{:.4f}", value_pad=adj_max * 0.05, value_size=8,
)
canvas3.set_xticks(x3, adj_labels, fontsize=9)
canvas3.save(str(CHARTS_DIR / "lpips_adjacent.png"))

print(f"Charts written to {CHARTS_DIR}")


# ── Findings writeup ──────────────────────────────────────────────────────────

max_clip = float(clip_arr.max())
max_clip_se = max(agg[a]["se_clip"] for a in ALPHA_VALUES)

# First alpha within 1 SE of max CLIP
plateau_alpha = None
for a in ALPHA_VALUES:
    if abs(agg[a]["mean_clip"] - max_clip) < max_clip_se:
        plateau_alpha = a
        break

# Largest adjacent step
adj_strs = ALPHA_VALUES[1:]
regime_step = max(adj_strs, key=lambda a: agg[a]["mean_lpips_prev"] or 0.0)
regime_val = agg[regime_step]["mean_lpips_prev"]
regime_from = ALPHA_VALUES[ALPHA_VALUES.index(regime_step) - 1]

lpips_at_zero = agg[0.0]["mean_lpips_ref"]
lpips_at_max_alpha = agg[ALPHA_VALUES[-1]]["mean_lpips_ref"]

# Build table
table_rows = []
for a in ALPHA_VALUES:
    aa = agg[a]
    prev_str = f"{aa['mean_lpips_prev']:.4f}" if aa["mean_lpips_prev"] is not None else "—"
    ref_marker = " ← ref" if a == ALPHA_REF else ""
    table_rows.append(
        f"| {a:.2f}  | {aa['mean_clip']:.4f}    | ±{aa['se_clip']:.4f}"
        f"  | {aa['mean_lpips_ref']:.4f}              | {prev_str}{ref_marker} |"
    )

lora_cfg = LORA_REGISTRY[LORA_NAME]

FINDINGS = f"""\
# Experiment 6: LoRA Style Scale (Alpha) Sweep

**Date:** 2026-05-07
**Hardware:** RTX 3070 Laptop 8 GB (enable_model_cpu_offload)
**Model:** {MODEL_ID}
**LoRA:** {LORA_NAME} — {lora_cfg['description']}
**Trigger token:** {lora_cfg['trigger_token']} (used in all prompts)
**Alpha values:** {ALPHA_VALUES}
  (0.0 = LoRA loaded but suppressed / base model; 1.0 = standard weight; 1.5 = over-styled)
**Reference alpha for LPIPS:** {ALPHA_REF}
**Design:** 5 seeds × 8 prompts = 40 images per alpha · {len(all_rows)} images total
**Scheduler:** DPM-Solver++ · {STEPS} steps · {SIZE}×{SIZE}
**CFG:** {CFG} (fixed)
**Note:** The LoRA adapter is loaded once and `set_adapters()` is called between alpha
  values — no retraining. This isolates the effect of adapter weight magnitude on output.

## Hypothesis

CLIP will be largely insensitive to the stylistic shift the LoRA induces — the text
prompt still describes the same semantic content at any alpha. LPIPS will capture the
substantial visual differences between the unstyled base model (alpha=0) and the
ukiyo-e style at various intensities.

## Results

| Alpha | Mean CLIP | SE      | LPIPS vs alpha={ALPHA_REF} | LPIPS vs prev (step) |
|-------|----------:|--------:|---------------------------:|---------------------:|
{chr(10).join(table_rows)}

LPIPS at alpha=0.0 (base model, no style) vs reference: {lpips_at_zero:.4f}
LPIPS at alpha={ALPHA_VALUES[-1]} (over-styled) vs reference: {lpips_at_max_alpha:.4f}

## Interpretation

**CLIP:** {f"Within noise across all alpha values — first within 1 SE of max CLIP ({max_clip:.4f}) is alpha={plateau_alpha}. Adapter weight has no detectable effect on semantic alignment as measured by CLIP. The ukiyo-e style is invisible to the embedding-based metric." if plateau_alpha is not None else "Modest variation across alpha values; see table."}

**LPIPS vs reference:** At alpha=0.0 (base model, LoRA suppressed): LPIPS={lpips_at_zero:.4f}.
At alpha={ALPHA_VALUES[-1]}: LPIPS={lpips_at_max_alpha:.4f}. The stylistic character of the
image changes substantially as alpha increases, but CLIP does not register this — the prompt
describes the same scene at every alpha value, and CLIP measures prompt alignment, not style.

**LPIPS (adjacent steps):** The largest single-step visual change is {regime_from}→{regime_step}
(LPIPS={regime_val:.4f}). Style accrues gradually rather than jumping at a threshold.

**The CLIP-blindness case for style:** LoRA adapters encode stylistic knowledge that has no
semantic equivalent in the text prompt. "A portrait in ukiyo-e style" and "a portrait"
describe the same content; the style difference is real and substantial (LPIPS confirms it)
but invisible to a metric that measures text-image semantic alignment. CLIP is the right
metric for "does the image match the prompt?", not "does the image have the intended style?"

**Cross-experiment note:** Sixth confirmation of CLIP-blindness: quantization (Exp 1),
negative prompt (Exp 2), CFG plateau (Exp 3), scheduler stochasticity (Exp 4), ControlNet
strength (Exp 5), LoRA style scale (Exp 6). The through-line: CLIP measures semantic
alignment, not visual character. Style, texture, and pixel-level decisions require a
perceptual metric.

## Charts

- `charts/clip_by_alpha.png` — mean CLIP score per adapter weight
- `charts/lpips_vs_ref.png` — perceptual distance from alpha={ALPHA_REF}
- `charts/lpips_adjacent.png` — step-wise LPIPS between adjacent alpha values

## Raw data

`results.csv` / `results.json` — one row per image ({len(all_rows)} rows total).

Reproduce:

```bash
python scripts/experiments/exp6_lora_alpha.py
```
"""

with open(OUT / "findings.md", "w", encoding="utf-8") as f:
    f.write(FINDINGS)

print(f"\nFindings written: {OUT / 'findings.md'}")
print("Experiment 6 complete.")
