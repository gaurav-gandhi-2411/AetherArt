"""
Experiment 2 (SDXL): Negative prompt impact.

SDXL port of exp2_negative_prompt.py.

Conditions: no_neg (empty negative prompt), with_neg (standard negative prompt)
Seeds: 5 fixed seeds x 8 prompts = 40 images per condition, 80 total
Metrics: CLIP score (comparison-only), HPS, ImageReward, LPIPS between conditions
         (same seed/prompt pair), latency

Hypothesis: the negative prompt reduces visual artifacts but CLIP score may
decrease slightly because guidance away from "bad" tokens reduces the energy
available for positive alignment. HPS and ImageReward capture human-preference
shifts that CLIP may miss.

Run from project root:
    python scripts/experiments/exp2_sdxl.py

Outputs:
    reports/experiments/exp2_sdxl/
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
from diffusers import StableDiffusionXLPipeline  # noqa: E402
from PIL import Image  # noqa: E402

from aetherart import clip_scorer  # noqa: E402
from aetherart.eval_hps import release_hps, score_hps  # noqa: E402
from aetherart.eval_ir import release_image_reward, score_image_reward  # noqa: E402
from aetherart.gpu_hygiene import cleanup_gpu  # noqa: E402
from aetherart.sdxl_pipeline import load_sdxl_base  # noqa: E402
from aetherart.visualization import BLUE, GREEN, GREY, ChartCanvas  # noqa: E402

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
# with_neg uses a more thorough negative prompt than SD 2.1 exp2 to better cover
# SDXL's common failure modes (calligraphy artefacts, signatures at 1024px).
NEG_PROMPT = (
    "low quality, blurry, deformed, ugly, bad anatomy, watermark, text, calligraphy, signature"
)
STEPS = 30
CFG = 7.5
SIZE = 1024

CONDITIONS = {
    "no_neg": "",
    "with_neg": NEG_PROMPT,
}

OUT = ROOT / "reports" / "experiments" / "exp2_sdxl"
IMG_DIR = OUT / "images"
CHARTS_DIR = OUT / "charts"

for _d in [IMG_DIR / "no_neg", IMG_DIR / "with_neg", CHARTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

COND_COLORS = {"no_neg": GREY, "with_neg": BLUE}
COND_LABELS = {"no_neg": "No negative prompt", "with_neg": "Standard negative prompt"}

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


def run_condition(label: str, neg_prompt: str, pipe: StableDiffusionXLPipeline) -> list[dict]:
    """Generate images for one negative-prompt condition and return result rows."""
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
                    "condition": label,
                    "prompt_id": prompt_id,
                    "prompt_text": prompt_text,
                    "seed": seed,
                    "latency_s": round(latency, 3),
                    "clip_score": None,
                    "hps_score": None,
                    "ir_score": None,
                    "lpips_vs_no_neg": None,
                    "image_path": (img_dir / fname).relative_to(ROOT).as_posix(),
                }
            )
            print(f"  [{label}] {prompt_id} seed={seed:5d} | {latency:.1f}s")
    return rows


# ── Main: single pipeline, both conditions ────────────────────────────────────

all_rows: list[dict] = []

print("\nLoading fp16 SDXL pipeline (shared across both conditions)...")
pipe = load_sdxl_base()

print("\n=== Condition 1/2: no_neg ===")
all_rows.extend(run_condition("no_neg", CONDITIONS["no_neg"], pipe))

print("\n=== Condition 2/2: with_neg ===")
all_rows.extend(run_condition("with_neg", CONDITIONS["with_neg"], pipe))

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

# ── LPIPS between conditions (same seed/prompt pair) ─────────────────────────

print("\nComputing LPIPS between conditions (no_neg vs with_neg)...")

no_neg_by_key: dict[tuple, str] = {
    (r["prompt_id"], r["seed"]): r["image_path"] for r in all_rows if r["condition"] == "no_neg"
}

for r in all_rows:
    if r["condition"] == "no_neg":
        # Will be filled symmetrically from the with_neg pass below
        r["lpips_vs_no_neg"] = None
        continue
    ref_path = no_neg_by_key[(r["prompt_id"], r["seed"])]
    val = _lpips_pair(ref_path, r["image_path"])
    r["lpips_vs_no_neg"] = val
    # Mirror onto the no_neg row so every row has a value
    no_neg_row = next(
        x
        for x in all_rows
        if x["condition"] == "no_neg"
        and x["prompt_id"] == r["prompt_id"]
        and x["seed"] == r["seed"]
    )
    no_neg_row["lpips_vs_no_neg"] = val

print("LPIPS done.")

# ── Save results ──────────────────────────────────────────────────────────────

CSV_PATH = OUT / "results.csv"
JSON_PATH = OUT / "results.json"

csv_fields = [
    "condition",
    "prompt_id",
    "seed",
    "latency_s",
    "clip_score",
    "hps_score",
    "ir_score",
    "lpips_vs_no_neg",
    "image_path",
]
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(all_rows)

with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(
        {
            "experiment": "exp2_sdxl",
            "date": "2026-06-02",
            "model": MODEL_ID,
            "steps": STEPS,
            "guidance": CFG,
            "size": SIZE,
            "seeds": SEEDS,
            "prompts": PROMPTS,
            "conditions": {k: v for k, v in CONDITIONS.items()},
            "scorers": ["clip", "hps", "imagereward", "lpips"],
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
    hps_vals = [r["hps_score"] for r in rows]
    ir_vals = [r["ir_score"] for r in rows]
    lats = [r["latency_s"] for r in rows]
    lpips_vals = [r["lpips_vs_no_neg"] for r in rows if r["lpips_vs_no_neg"] is not None]
    agg[cond] = {
        "n": len(rows),
        "mean_clip": statistics.mean(clips),
        "se_clip": statistics.stdev(clips) / len(clips) ** 0.5,
        "mean_hps": statistics.mean(hps_vals),
        "se_hps": statistics.stdev(hps_vals) / len(hps_vals) ** 0.5,
        "mean_ir": statistics.mean(ir_vals),
        "se_ir": statistics.stdev(ir_vals) / len(ir_vals) ** 0.5,
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
        f"HPS={a['mean_hps']:.4f} ±{a['se_hps']:.4f} | "
        f"IR={a['mean_ir']:.4f} ±{a['se_ir']:.4f} | "
        f"lat={a['mean_lat']:.1f}s | LPIPS={a['mean_lpips']:.4f}"
    )

clip_delta = agg["with_neg"]["mean_clip"] - agg["no_neg"]["mean_clip"]
hps_delta = agg["with_neg"]["mean_hps"] - agg["no_neg"]["mean_hps"]
ir_delta = agg["with_neg"]["mean_ir"] - agg["no_neg"]["mean_ir"]
lpips_mean = agg["with_neg"]["mean_lpips"]  # symmetric; same for both conditions
print(f"\n  CLIP delta (with_neg − no_neg): {clip_delta:+.4f}")
print(f"  HPS delta  (with_neg − no_neg): {hps_delta:+.4f}")
print(f"  IR delta   (with_neg − no_neg): {ir_delta:+.4f}")
print(f"  Mean LPIPS between conditions:  {lpips_mean:.4f}")

# ── Charts ────────────────────────────────────────────────────────────────────

COND_ORDER = ["no_neg", "with_neg"]
x = np.arange(len(COND_ORDER), dtype=float)

# Chart 1: HPS score by condition (primary quality metric)
hps_vals_chart = np.array([agg[c]["mean_hps"] for c in COND_ORDER])
hps_max = float(hps_vals_chart.max())

canvas_hps = ChartCanvas(
    figsize=(6, 4.5),
    title="HPS score: no negative prompt vs standard negative prompt (SDXL 1024px)",
    ylabel="Mean HPS score",
    top_margin_pct=0.22,
)
canvas_hps.set_ylim(0.0, hps_max * 1.35)
canvas_hps.add_bars(
    x,
    hps_vals_chart,
    colors=[COND_COLORS[c] for c in COND_ORDER],
    width=0.55,
    value_fmt="{:.4f}",
    value_pad=hps_max * 0.015,
    value_size=9,
)
canvas_hps.set_xticks(x, [COND_LABELS[c] for c in COND_ORDER], fontsize=9)
canvas_hps.save(str(CHARTS_DIR / "hps_by_condition.png"))

# Chart 2: ImageReward score by condition
ir_vals_chart = np.array([agg[c]["mean_ir"] for c in COND_ORDER])
ir_min = float(ir_vals_chart.min())
ir_max_val = float(ir_vals_chart.max())
ir_pad = max(abs(ir_min) * 0.4, abs(ir_max_val) * 0.4, 0.1)

canvas_ir = ChartCanvas(
    figsize=(6, 4.5),
    title="ImageReward score: no negative prompt vs standard negative prompt (SDXL 1024px)",
    ylabel="Mean ImageReward score",
    top_margin_pct=0.22,
)
canvas_ir.set_ylim(ir_min - ir_pad, ir_max_val + ir_pad * 1.5)
canvas_ir.add_bars(
    x,
    ir_vals_chart,
    colors=[COND_COLORS[c] for c in COND_ORDER],
    width=0.55,
    value_fmt="{:.4f}",
    value_pad=ir_pad * 0.1,
    value_size=9,
)
canvas_ir.set_xticks(x, [COND_LABELS[c] for c in COND_ORDER], fontsize=9)
canvas_ir.save(str(CHARTS_DIR / "ir_by_condition.png"))

# Chart 3: CLIP score by condition (comparison-only)
clip_vals_chart = np.array([agg[c]["mean_clip"] for c in COND_ORDER])
clip_max = float(clip_vals_chart.max())

canvas_clip = ChartCanvas(
    figsize=(6, 4.5),
    title="CLIP score: no negative prompt vs standard negative prompt — SDXL 1024px (comparison-only)",
    ylabel="Mean CLIP score",
    top_margin_pct=0.22,
)
canvas_clip.set_ylim(0.0, clip_max * 1.35)
canvas_clip.add_bars(
    x,
    clip_vals_chart,
    colors=[COND_COLORS[c] for c in COND_ORDER],
    width=0.55,
    value_fmt="{:.4f}",
    value_pad=clip_max * 0.015,
    value_size=9,
)
canvas_clip.set_xticks(x, [COND_LABELS[c] for c in COND_ORDER], fontsize=9)
canvas_clip.save(str(CHARTS_DIR / "clip_by_condition.png"))

# Chart 4: LPIPS between conditions (single bar — symmetric)
canvas_lpips = ChartCanvas(
    figsize=(5, 4.5),
    title="Perceptual distance between conditions (LPIPS) — SDXL 1024px",
    ylabel="Mean LPIPS (no_neg vs with_neg)",
    top_margin_pct=0.22,
)
canvas_lpips.set_ylim(0.0, max(lpips_mean * 1.7, 0.05))
canvas_lpips.add_bars(
    np.array([0.0]),
    np.array([lpips_mean]),
    colors=[GREEN],
    width=0.45,
    value_fmt="{:.4f}",
    value_pad=max(lpips_mean * 0.06, 0.002),
    value_size=10,
)
canvas_lpips.set_xticks(np.array([0.0]), ["No neg vs With neg"], fontsize=9)
canvas_lpips.save(str(CHARTS_DIR / "lpips_between_conditions.png"))

# Chart 5: Per-prompt CLIP delta (with_neg − no_neg)
prompt_ids = list(PROMPTS.keys())
by_prompt_cond: dict[tuple, list[float]] = {}
for r in all_rows:
    key = (r["condition"], r["prompt_id"])
    by_prompt_cond.setdefault(key, []).append(r["clip_score"])

prompt_clip_deltas = np.array(
    [
        statistics.mean(by_prompt_cond[("with_neg", pid)])
        - statistics.mean(by_prompt_cond[("no_neg", pid)])
        for pid in prompt_ids
    ]
)
colors_delta = [BLUE if d >= 0 else GREY for d in prompt_clip_deltas]

x3 = np.arange(len(prompt_ids), dtype=float)
canvas_delta = ChartCanvas(
    figsize=(10, 4.5),
    title="Per-prompt CLIP delta: with_neg minus no_neg (positive = negative prompt helped) — SDXL",
    ylabel="CLIP delta",
    top_margin_pct=0.22,
)
delta_abs_max = float(np.abs(prompt_clip_deltas).max())
canvas_delta.set_ylim(-delta_abs_max * 2.5, delta_abs_max * 2.5)
canvas_delta.add_bars(
    x3,
    prompt_clip_deltas,
    colors=colors_delta,
    width=0.6,
    value_fmt="{:+.4f}",
    value_pad=delta_abs_max * 0.08,
    value_size=8,
)
canvas_delta.set_xticks(x3, [pid.replace("_", " ") for pid in prompt_ids], fontsize=8)
canvas_delta.save(str(CHARTS_DIR / "clip_delta_by_prompt.png"))

print(f"Charts written to {CHARTS_DIR}")

# ── Findings writeup ──────────────────────────────────────────────────────────

no_a = agg["no_neg"]
wn_a = agg["with_neg"]
pooled_se_clip = (no_a["se_clip"] + wn_a["se_clip"]) / 2
clip_delta_in_se = abs(clip_delta) / pooled_se_clip if pooled_se_clip > 0 else 0.0
clip_blind = "yes" if clip_delta_in_se < 2.0 else "no"


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
# Experiment 2 (SDXL): Negative Prompt Impact

**Date:** 2026-06-02
**Hardware:** GCP L4 (24 GB VRAM)
**Model:** {MODEL_ID}
**Conditions:** no_neg (empty negative prompt) · with_neg (standard negative prompt)
**Design:** 5 seeds x 8 prompts = 40 images per condition · 80 images total
**Scheduler:** DPM-Solver++ · {STEPS} steps · {SIZE}x{SIZE} · CFG={CFG}
**Negative prompt tested:** `{NEG_PROMPT}`
**Scorers:** HPS (primary), ImageReward (primary), CLIP (comparison-only), LPIPS

## Hypothesis

The standard negative prompt guides the model away from degenerate outputs (blurry, deformed,
watermarked). This should reduce artifacts and may increase HPS/IR if the negative tokens
overlap with semantically poor regions, or decrease them if the guidance energy is reallocated
away from positive alignment. A null result (no reliable difference) is also plausible —
negative prompts primarily reshape the output distribution at the tails.

## Results

| Condition | Mean HPS | Mean IR | Mean CLIP | Latency (s) | Mean LPIPS (vs other cond.) |
|-----------|:--------:|:-------:|----------:|--------------:|----------------------------:|
| no_neg    | {no_a["mean_hps"]:.4f}   | {no_a["mean_ir"]:.4f}  | {no_a["mean_clip"]:.4f}    | {no_a["mean_lat"]:.1f}s         | {no_a["mean_lpips"]:.4f}                       |
| with_neg  | {wn_a["mean_hps"]:.4f}   | {wn_a["mean_ir"]:.4f}  | {wn_a["mean_clip"]:.4f}    | {wn_a["mean_lat"]:.1f}s         | {wn_a["mean_lpips"]:.4f}                       |

SE on CLIP: no_neg ±{no_a["se_clip"]:.4f} · with_neg ±{wn_a["se_clip"]:.4f}
SE on HPS:  no_neg ±{no_a["se_hps"]:.4f} · with_neg ±{wn_a["se_hps"]:.4f}
SE on IR:   no_neg ±{no_a["se_ir"]:.4f} · with_neg ±{wn_a["se_ir"]:.4f}

CLIP delta (with_neg − no_neg): {clip_delta:+.4f} — {_clip_delta_verdict(clip_delta, pooled_se_clip)}
HPS delta  (with_neg − no_neg): {hps_delta:+.4f}
IR delta   (with_neg − no_neg): {ir_delta:+.4f}

LPIPS between conditions (same seed/prompt pair): {lpips_mean:.4f} — {_lpips_context(lpips_mean)}

## CLIP-blindness verdict

CLIP delta across conditions = {abs(clip_delta):.4f}, which is {clip_delta_in_se:.2f} SEs. \
HPS delta = {abs(hps_delta):.4f}. \
IR delta = {abs(ir_delta):.4f}. \
LPIPS range = {lpips_mean:.4f}. \
Verdict: CLIP-BLIND: {clip_blind}.

## Per-prompt breakdown

See `charts/clip_delta_by_prompt.png`. Positive bars = negative prompt improved CLIP for that
prompt category; negative bars = negative prompt hurt CLIP. Variance across prompts reveals
whether the effect is consistent or prompt-dependent.

## Interpretation

The CLIP delta is {_clip_delta_verdict(clip_delta, pooled_se_clip)}.
The LPIPS of {lpips_mean:.4f} between conditions tells us the negative prompt {_lpips_context(lpips_mean)}.

HPS and ImageReward provide human-preference-aligned signal: HPS delta = {hps_delta:+.4f},
IR delta = {ir_delta:+.4f}. Where CLIP is blind to perceptual differences, these scorers
reveal whether the negative prompt improves or degrades the output quality experienced by
a human observer.

Latency difference: {wn_a["mean_lat"] - no_a["mean_lat"]:+.2f}s — negative prompt text adds
minimal compute overhead (classifier-free guidance already processes a null embedding; replacing
it with a non-empty prompt does not change the number of forward passes).

## Charts

- `charts/hps_by_condition.png` — mean HPS score per condition (primary quality metric)
- `charts/ir_by_condition.png` — mean ImageReward score per condition
- `charts/clip_by_condition.png` — mean CLIP score per condition (comparison-only)
- `charts/lpips_between_conditions.png` — perceptual distance between matched pairs
- `charts/clip_delta_by_prompt.png` — per-prompt CLIP delta (with_neg minus no_neg)

## Raw data

`results.csv` / `results.json` — one row per image (80 rows total).

Reproduce:

```bash
python scripts/experiments/exp2_sdxl.py
```
"""

with open(OUT / "findings.md", "w", encoding="utf-8") as f:
    f.write(FINDINGS)

print(f"\nFindings written: {OUT / 'findings.md'}")
print("Experiment 2 (SDXL) complete.")
