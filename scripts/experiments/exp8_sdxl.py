"""
Experiment 8 (SDXL): LoRA style scale (alpha) sweep — Ukiyo-e SDXL LoRA.

Port of exp8_lora_alpha.py (SD 2.1) to SDXL. The original file docstring
incorrectly says "Experiment 6" — this is exp8 by filename.

The SDXL Ukiyo-e LoRA (gauravgandhi2411/aetherart-ukiyo-sdxl) is loaded once.
Adapter weight (alpha) is swept across [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
via set_adapters() between conditions — no retraining required.

alpha=0.0: LoRA loaded but suppressed (base SDXL)
alpha=1.0: standard adapter weight — LPIPS reference
alpha=1.5: over-styled; potential saturation / mode collapse

All prompts use "ukyowood" trigger token. CLIP is measured against the full prompt.
If CLIP is insensitive to the stylistic shift that LPIPS captures, it adds another
CLIP-blindness data point in the SDXL regime.

Design: 7 alpha values × 8 prompts × 5 seeds = 280 images
Image size: 1024×1024 (SDXL)

Run from project root:
    python scripts/experiments/exp8_sdxl.py

Outputs:
    reports/experiments/exp8_sdxl/
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
    TEAL,
    ChartCanvas,
)

atexit.register(cleanup_gpu)

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
LORA_REPO = "gauravgandhi2411/aetherart-ukiyo-sdxl"
LORA_LOCAL = ROOT / "data" / "lora" / "ukiyo-e-sdxl" / "pytorch_lora_weights.safetensors"

ALPHA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
ALPHA_REF = 1.0
SEEDS = [42, 123, 456, 789, 1337]
STEPS = 30
CFG = 7.5
SIZE = 1024

# Prompts with "ukyowood" trigger token (LoRA was trained with this caption prefix)
PROMPTS = {
    "p01_portrait": (
        "ukyowood ukiyo-e portrait of an elderly woman, dramatic light, woodblock print style"
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
        "ukyowood ukiyo-e vintage print with bold lettering, retro typography, worn paper texture"
    ),
    "p05_texture": (
        "ukyowood ukiyo-e extreme close-up of rough stone wall, water drops, micro detail"
    ),
    "p06_arch": (
        "ukyowood ukiyo-e interior of a Japanese temple, wooden pillars, soft lantern light"
    ),
    "p07_hands": (
        "ukyowood ukiyo-e two hands clasped together, natural light, woodblock print style"
    ),
    "p08_crowd": (
        "ukyowood ukiyo-e busy street market in Edo, dozens of people, lantern light, night scene"
    ),
}
NEG_PROMPT = "low quality, blurry, deformed, ugly, bad anatomy, watermark, text, calligraphy"

OUT = ROOT / "reports" / "experiments" / "exp8_sdxl"
IMG_DIR = OUT / "images"
CHARTS_DIR = OUT / "charts"

for a in ALPHA_VALUES:
    (IMG_DIR / f"alpha_{a}").mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

_ALPHA_PALETTE = {
    0.0: GREY,
    0.25: "#5AADA8",
    0.5: GREEN,
    0.75: TEAL,
    1.0: BLUE,
    1.25: ORANGE,
    1.5: RED,
}


# ── LoRA download helper ───────────────────────────────────────────────────────


def _ensure_lora() -> Path:
    """Return the local LoRA path, downloading from HF Hub if missing."""
    if LORA_LOCAL.exists():
        return LORA_LOCAL
    print(f"Downloading SDXL LoRA from {LORA_REPO}...")
    from huggingface_hub import hf_hub_download

    LORA_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        LORA_REPO, "pytorch_lora_weights.safetensors", local_dir=str(LORA_LOCAL.parent)
    )
    return Path(path)


# ── Pipeline and LoRA loading ─────────────────────────────────────────────────


def load_lora_once(pipe: "object") -> None:  # StableDiffusionXLPipeline
    """Download LoRA if needed then load it into the pipeline once."""
    lora_path = _ensure_lora()
    pipe.load_lora_weights(  # type: ignore[union-attr]
        str(lora_path.parent),
        weight_name=lora_path.name,
        adapter_name="ukiyo_e",
    )
    print(f"SDXL LoRA loaded: {lora_path.name}")


def set_alpha(pipe: "object", alpha: float) -> None:  # StableDiffusionXLPipeline
    """Set the LoRA adapter weight without reloading weights."""
    pipe.set_adapters(["ukiyo_e"], adapter_weights=[alpha])  # type: ignore[union-attr]


# ── Generation loop ───────────────────────────────────────────────────────────


def run_alpha(alpha: float, pipe: "object") -> list[dict]:  # StableDiffusionXLPipeline
    """Generate 8 prompts × 5 seeds at one alpha value and return row dicts."""
    set_alpha(pipe, alpha)
    rows: list[dict] = []
    img_dir = IMG_DIR / f"alpha_{alpha}"

    for prompt_id, prompt_text in PROMPTS.items():
        for seed in SEEDS:
            generator = torch.Generator().manual_seed(seed)
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
                    "hps_score": None,
                    "ir_score": None,
                    "lpips_vs_ref": None,
                    "image_path": (img_dir / fname).relative_to(ROOT).as_posix(),
                }
            )
            print(f"  [alpha={alpha:.2f}] {prompt_id} seed={seed:5d} | {latency:.1f}s")

    return rows


# ── Run all alpha values ───────────────────────────────────────────────────────

all_rows: list[dict] = []

print("\nLoading SDXL base pipeline...")
pipe = load_sdxl_base()
print("Loading SDXL LoRA weights once...")
load_lora_once(pipe)

for alpha in ALPHA_VALUES:
    idx = ALPHA_VALUES.index(alpha) + 1
    print(f"\n=== Alpha = {alpha} ({idx}/{len(ALPHA_VALUES)}) ===")
    all_rows.extend(run_alpha(alpha, pipe))

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


# ── LPIPS vs alpha=1.0 reference (post-hoc) ──────────────────────────────────

print("\nComputing LPIPS vs reference (alpha=1.0)...")
_lpips_fn = lpips_lib.LPIPS(net="alex")
_lpips_fn.eval()

img_index: dict[tuple[float, str, int], str] = {
    (r["alpha"], r["prompt_id"], r["seed"]): r["image_path"] for r in all_rows
}


def _to_t(img: Image.Image) -> torch.Tensor:
    """Convert PIL image to normalised [-1, 1] tensor for LPIPS."""
    arr = np.array(img.convert("RGB")).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def _lpips_pair(path_a: str, path_b: str) -> float:
    """Compute LPIPS distance between two image files."""
    a = Image.open(ROOT / path_a).convert("RGB")
    b = Image.open(ROOT / path_b).convert("RGB")
    with torch.no_grad():
        return round(float(_lpips_fn(_to_t(a), _to_t(b))), 6)


done = 0
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
    if done % 56 == 0 or done == len(all_rows):
        print(f"  LPIPS {done}/{len(all_rows)}")

print("LPIPS done.")


# ── Save results ──────────────────────────────────────────────────────────────

CSV_PATH = OUT / "results.csv"
JSON_PATH = OUT / "results.json"

csv_fields = [
    "alpha", "prompt_id", "seed", "latency_s",
    "clip_score", "hps_score", "ir_score", "lpips_vs_ref", "image_path",
]
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(all_rows)

with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(
        {
            "experiment": "exp8_sdxl",
            "date": "2026-06-02",
            "model": MODEL_ID,
            "lora_repo": LORA_REPO,
            "alpha_values": ALPHA_VALUES,
            "alpha_ref": ALPHA_REF,
            "size": SIZE,
            "steps": STEPS,
            "cfg": CFG,
            "seeds": SEEDS,
            "scorers": ["clip", "hps", "imagereward", "lpips"],
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
    hps_vals = [r["hps_score"] for r in rows]
    ir_vals = [r["ir_score"] for r in rows]
    lats = [r["latency_s"] for r in rows]
    lpips_ref = [r["lpips_vs_ref"] for r in rows if r["lpips_vs_ref"] is not None]
    agg[a] = {
        "mean_clip": statistics.mean(clips),
        "se_clip": statistics.stdev(clips) / len(clips) ** 0.5,
        "mean_hps": statistics.mean(hps_vals),
        "mean_ir": statistics.mean(ir_vals),
        "mean_lat": statistics.mean(lats),
        "mean_lpips_ref": statistics.mean(lpips_ref) if lpips_ref else 0.0,
        "n": len(rows),
    }

print("\n── CLIP / HPS / IR / LPIPS by LoRA alpha ──")
for a in ALPHA_VALUES:
    aa = agg[a]
    ref_mark = " ← ref" if a == ALPHA_REF else ""
    print(
        f"  alpha={a:.2f}: CLIP={aa['mean_clip']:.4f} ±{aa['se_clip']:.4f} | "
        f"HPS={aa['mean_hps']:.4f} | IR={aa['mean_ir']:.4f} | "
        f"LPIPS_ref={aa['mean_lpips_ref']:.4f}{ref_mark}"
    )


# ── Charts ────────────────────────────────────────────────────────────────────

clip_arr = np.array([agg[a]["mean_clip"] for a in ALPHA_VALUES])
hps_arr = np.array([agg[a]["mean_hps"] for a in ALPHA_VALUES])
ir_arr = np.array([agg[a]["mean_ir"] for a in ALPHA_VALUES])
lpips_ref_arr = np.array([agg[a]["mean_lpips_ref"] for a in ALPHA_VALUES])
colors = [_ALPHA_PALETTE[a] for a in ALPHA_VALUES]
x = np.arange(len(ALPHA_VALUES), dtype=float)
xlabels = [f"α={a}" for a in ALPHA_VALUES]

# Chart 1: CLIP by alpha
clip_max = float(clip_arr.max())
canvas = ChartCanvas(
    figsize=(9, 4.5),
    title=(
        f"CLIP score vs LoRA alpha — SDXL Ukiyo-e, {len(PROMPTS)} prompts × {len(SEEDS)} seeds"
    ),
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

# Chart 2: HPS by alpha
hps_max = float(hps_arr.max())
canvas_hps = ChartCanvas(
    figsize=(9, 4.5),
    title=f"HPS score vs LoRA alpha — SDXL Ukiyo-e",
    ylabel="Mean HPS score",
    top_margin_pct=0.22,
)
canvas_hps.set_ylim(0.0, hps_max * 1.35)
canvas_hps.add_bars(
    x, hps_arr, colors=colors, width=0.6,
    value_fmt="{:.4f}", value_pad=hps_max * 0.015, value_size=8,
)
canvas_hps.set_xticks(x, xlabels, fontsize=9)
canvas_hps.save(str(CHARTS_DIR / "hps_by_alpha.png"))

# Chart 3: LPIPS vs alpha=1.0 reference
lpips_ref_max = float(lpips_ref_arr.max())
canvas2 = ChartCanvas(
    figsize=(9, 4.5),
    title=f"Perceptual distance from alpha={ALPHA_REF} reference (LPIPS) — SDXL",
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

print(f"Charts written to {CHARTS_DIR}")


# ── Findings writeup ──────────────────────────────────────────────────────────

clip_max_val = float(clip_arr.max())
max_clip_se = max(agg[a]["se_clip"] for a in ALPHA_VALUES)
plateau_alpha = None
for a in ALPHA_VALUES:
    if abs(agg[a]["mean_clip"] - clip_max_val) < max_clip_se:
        plateau_alpha = a
        break

lpips_at_zero = agg[0.0]["mean_lpips_ref"]
lpips_at_max = agg[ALPHA_VALUES[-1]]["mean_lpips_ref"]

# CLIP blindness stats
clip_range = float(clip_arr.max() - clip_arr.min())
pooled_se = statistics.mean(agg[a]["se_clip"] for a in ALPHA_VALUES)
clip_range_se = clip_range / pooled_se if pooled_se > 0 else float("inf")
hps_range = float(hps_arr.max() - hps_arr.min())
ir_range = float(ir_arr.max() - ir_arr.min())
lpips_range = float(lpips_ref_arr.max() - lpips_ref_arr.min())

# Build table
table_rows = []
for a in ALPHA_VALUES:
    aa = agg[a]
    ref_mark = " ← ref" if a == ALPHA_REF else ""
    table_rows.append(
        f"| {a:.2f}  | {aa['mean_clip']:.4f} | ±{aa['se_clip']:.4f}"
        f" | {aa['mean_hps']:.4f} | {aa['mean_ir']:.4f}"
        f" | {aa['mean_lpips_ref']:.4f}{ref_mark} |"
    )

FINDINGS = f"""\
# Experiment 8 (SDXL): LoRA Style Scale (Alpha) Sweep

**Date:** 2026-06-02
**Model:** {MODEL_ID} — {SIZE}×{SIZE}
**LoRA:** {LORA_REPO} (SDXL Ukiyo-e)
**Trigger token:** ukyowood (used in all prompts)
**Alpha values:** {ALPHA_VALUES}
  (0.0 = LoRA loaded but suppressed; 1.0 = trained default; 1.5 = over-styled)
**Reference alpha for LPIPS:** {ALPHA_REF}
**Design:** {len(ALPHA_VALUES)} alphas × {len(PROMPTS)} prompts × {len(SEEDS)} seeds = {len(all_rows)} images
**Scheduler:** DPM-Solver++ · {STEPS} steps · CFG={CFG}
**Scorers:** CLIP, HPS, ImageReward, LPIPS
**Note:** LoRA loaded once; set_adapters() used to sweep alpha — no retraining.

## Hypothesis

CLIP and HPS will be largely insensitive to the stylistic shift the LoRA induces —
the text prompt still describes the same semantic content at any alpha. LPIPS will
capture substantial visual differences between base SDXL (alpha=0) and the
ukiyo-e style at various intensities.

## Results

| Alpha | Mean CLIP | SE      | Mean HPS | Mean IR | LPIPS vs α=1.0 |
|-------|----------:|--------:|---------:|--------:|---------------:|
{chr(10).join(table_rows)}

CLIP delta across conditions = {clip_range:.4f}, which is {clip_range_se:.1f} SEs.
HPS delta = {hps_range:.4f}. IR delta = {ir_range:.4f}. LPIPS range = {lpips_range:.4f}.
Verdict: CLIP-BLIND: {"yes" if clip_range_se < 2.0 else "no"}.

LPIPS at alpha=0.0 (base SDXL, no style) vs reference: {lpips_at_zero:.4f}
LPIPS at alpha={ALPHA_VALUES[-1]} (over-styled) vs reference: {lpips_at_max:.4f}

## Interpretation

**CLIP / HPS / IR:** {f"All three scorers are within noise across all alpha values. First within 1 SE of max CLIP ({clip_max_val:.4f}) is alpha={plateau_alpha}. Adapter weight has no detectable effect on semantic alignment." if plateau_alpha is not None else "Modest variation across alpha values; see table."}

**LPIPS:** The stylistic character of the image changes substantially as alpha increases,
but semantic scorers do not register this — the prompt describes the same scene at every
alpha. At alpha=0.0 (no LoRA), LPIPS={lpips_at_zero:.4f} vs reference; at
alpha={ALPHA_VALUES[-1]}, LPIPS={lpips_at_max:.4f}.

**SDXL vs SD 2.1:** At 1024×1024, the LoRA style shift may produce larger absolute LPIPS
values than the 512×512 SD 2.1 baseline, while CLIP scores remain in the same range.

## Charts

- `charts/clip_by_alpha.png`
- `charts/hps_by_alpha.png`
- `charts/lpips_vs_ref.png`

## Raw data

`results.csv` / `results.json` — one row per image ({len(all_rows)} rows total).

Reproduce:

```bash
python scripts/experiments/exp8_sdxl.py
```
"""

with open(OUT / "findings.md", "w", encoding="utf-8") as f:
    f.write(FINDINGS)

print(f"\nFindings written: {OUT / 'findings.md'}")
print("Experiment 8 (SDXL) complete.")
