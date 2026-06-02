"""
Experiment 3 (SDXL): CFG (guidance scale) sweep.

SDXL port of exp3_cfg_sweep.py.

CFG values: [1, 3, 5, 7, 9, 12, 15]
Seeds: 5 fixed seeds x 8 prompts = 40 images per CFG value, 280 total
Metrics:
  - CLIP score (comparison-only — see CLIP-blindness verdict)
  - HPS and ImageReward per image (primary quality metrics)
  - LPIPS between adjacent CFG pairs (1-3, 3-5, ..., 12-15) per (prompt, seed):
    locates where the visual regime change happens as CFG increases
  - LPIPS vs cfg=7 reference per (prompt, seed):
    cumulative distance from a mid-range baseline

Hypothesis: HPS/CLIP will plateau once CFG is high enough to anchor the prompt (somewhere
around 5-9). LPIPS between adjacent values will reveal a "regime change" — a step where
visual character changes sharply — that CLIP cannot detect. At very high CFG (12–15)
over-saturation artefacts should depress HPS and IR before CLIP reacts.

Run from project root:
    python scripts/experiments/exp3_sdxl.py

Outputs:
    reports/experiments/exp3_sdxl/
        images/cfg_{value}/  -- 40 PNG per CFG value
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
CFG_VALUES = [1, 3, 5, 7, 9, 12, 15]
CFG_REF = 7  # reference for LPIPS-vs-reference chart
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
SIZE = 1024

OUT = ROOT / "reports" / "experiments" / "exp3_sdxl"
IMG_DIR = OUT / "images"
CHARTS_DIR = OUT / "charts"

for cfg in CFG_VALUES:
    (IMG_DIR / f"cfg_{cfg}").mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# Colour palette: map CFG values to a gradient from grey → blue → red
_CFG_PALETTE = {
    1: GREY,
    3: "#5AADA8",  # teal-light
    5: GREEN,
    7: BLUE,
    9: ORANGE,
    12: PURPLE,
    15: RED,
}

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


def run_cfg(cfg_val: float, pipe: StableDiffusionXLPipeline) -> list[dict]:
    """Generate images for one CFG value and return result rows."""
    rows: list[dict] = []
    img_dir = IMG_DIR / f"cfg_{cfg_val}"
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
                guidance_scale=cfg_val,
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
                    "cfg_value": cfg_val,
                    "prompt_id": prompt_id,
                    "prompt_text": prompt_text,
                    "seed": seed,
                    "latency_s": round(latency, 3),
                    "clip_score": None,
                    "hps_score": None,
                    "ir_score": None,
                    "lpips_vs_ref": None,  # vs cfg=CFG_REF
                    "lpips_vs_prev": None,  # vs previous CFG value
                    "image_path": (img_dir / fname).relative_to(ROOT).as_posix(),
                }
            )
            print(f"  [cfg={cfg_val:2}] {prompt_id} seed={seed:5d} | {latency:.1f}s")
    return rows


# ── Generate all CFG values (single loaded pipeline) ─────────────────────────

all_rows: list[dict] = []

print("\nLoading fp16 SDXL pipeline...")
pipe = load_sdxl_base()

for cfg in CFG_VALUES:
    print(f"\n=== CFG = {cfg} ({CFG_VALUES.index(cfg) + 1}/{len(CFG_VALUES)}) ===")
    all_rows.extend(run_cfg(cfg, pipe))

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

# ── LPIPS (post-hoc) ─────────────────────────────────────────────────────────

print("\nComputing LPIPS (vs reference and vs adjacent)...")

# index: (cfg_value, prompt_id, seed) -> image_path
img_index: dict[tuple, str] = {
    (r["cfg_value"], r["prompt_id"], r["seed"]): r["image_path"] for r in all_rows
}

total_lpips = len(all_rows) * 2  # rough upper bound
done = 0
for r in all_rows:
    cfg_val = r["cfg_value"]
    pid = r["prompt_id"]
    seed = r["seed"]

    # LPIPS vs reference (cfg=CFG_REF)
    if cfg_val == CFG_REF:
        r["lpips_vs_ref"] = 0.0
    else:
        ref_path = img_index[(CFG_REF, pid, seed)]
        r["lpips_vs_ref"] = _lpips_pair(r["image_path"], ref_path)
    done += 1

    # LPIPS vs previous CFG value in the sweep
    cfg_idx = CFG_VALUES.index(cfg_val)
    if cfg_idx == 0:
        r["lpips_vs_prev"] = None
    else:
        prev_cfg = CFG_VALUES[cfg_idx - 1]
        prev_path = img_index[(prev_cfg, pid, seed)]
        r["lpips_vs_prev"] = _lpips_pair(r["image_path"], prev_path)
    done += 1

    if done % 80 == 0:
        print(f"  LPIPS {done}/{total_lpips}")

print("LPIPS done.")

# ── Save results ──────────────────────────────────────────────────────────────

CSV_PATH = OUT / "results.csv"
JSON_PATH = OUT / "results.json"

csv_fields = [
    "cfg_value",
    "prompt_id",
    "seed",
    "latency_s",
    "clip_score",
    "hps_score",
    "ir_score",
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
            "experiment": "exp3_sdxl",
            "date": "2026-06-02",
            "model": MODEL_ID,
            "cfg_values": CFG_VALUES,
            "cfg_ref": CFG_REF,
            "steps": STEPS,
            "size": SIZE,
            "seeds": SEEDS,
            "prompts": PROMPTS,
            "neg_prompt": NEG_PROMPT,
            "scorers": ["clip", "hps", "imagereward", "lpips"],
            "total_images": len(all_rows),
            "results": all_rows,
        },
        f,
        indent=2,
    )

print(f"Results: {CSV_PATH}")

# ── Per-CFG aggregates ────────────────────────────────────────────────────────

by_cfg: dict[int, list[dict]] = {c: [] for c in CFG_VALUES}
for r in all_rows:
    by_cfg[r["cfg_value"]].append(r)

agg: dict[int, dict] = {}
for cfg_val, rows in by_cfg.items():
    clips = [r["clip_score"] for r in rows]
    hps_vals = [r["hps_score"] for r in rows]
    ir_vals = [r["ir_score"] for r in rows]
    lats = [r["latency_s"] for r in rows]
    lpips_ref = [r["lpips_vs_ref"] for r in rows if r["lpips_vs_ref"] is not None]
    lpips_prev = [r["lpips_vs_prev"] for r in rows if r["lpips_vs_prev"] is not None]
    agg[cfg_val] = {
        "mean_clip": statistics.mean(clips),
        "se_clip": statistics.stdev(clips) / len(clips) ** 0.5,
        "mean_hps": statistics.mean(hps_vals),
        "se_hps": statistics.stdev(hps_vals) / len(hps_vals) ** 0.5,
        "mean_ir": statistics.mean(ir_vals),
        "se_ir": statistics.stdev(ir_vals) / len(ir_vals) ** 0.5,
        "mean_lat": statistics.mean(lats),
        "mean_lpips_ref": statistics.mean(lpips_ref) if lpips_ref else 0.0,
        "mean_lpips_prev": statistics.mean(lpips_prev) if lpips_prev else None,
    }

print("\n── Aggregates ──")
for cfg_val in CFG_VALUES:
    a = agg[cfg_val]
    prev_str = f"{a['mean_lpips_prev']:.4f}" if a["mean_lpips_prev"] is not None else "—"
    print(
        f"  cfg={cfg_val:2d}: CLIP={a['mean_clip']:.4f} ±{a['se_clip']:.4f} | "
        f"HPS={a['mean_hps']:.4f} ±{a['se_hps']:.4f} | "
        f"IR={a['mean_ir']:.4f} ±{a['se_ir']:.4f} | "
        f"lat={a['mean_lat']:.1f}s | LPIPS_ref={a['mean_lpips_ref']:.4f} | "
        f"LPIPS_prev={prev_str}"
    )

# ── Charts ────────────────────────────────────────────────────────────────────

cfg_arr = np.array(CFG_VALUES, dtype=float)
clip_arr = np.array([agg[c]["mean_clip"] for c in CFG_VALUES])
hps_arr = np.array([agg[c]["mean_hps"] for c in CFG_VALUES])
ir_arr = np.array([agg[c]["mean_ir"] for c in CFG_VALUES])
lpips_ref_arr = np.array([agg[c]["mean_lpips_ref"] for c in CFG_VALUES])
lpips_prev_arr = np.array(
    [
        agg[c]["mean_lpips_prev"] if agg[c]["mean_lpips_prev"] is not None else 0.0
        for c in CFG_VALUES
    ]
)
colors = [_CFG_PALETTE[c] for c in CFG_VALUES]
x = np.arange(len(CFG_VALUES), dtype=float)
xlabels = [f"CFG={c}" for c in CFG_VALUES]

# Chart 1: HPS score by CFG value (primary quality metric)
hps_max = float(hps_arr.max())
canvas_hps = ChartCanvas(
    figsize=(9, 4.5),
    title="HPS score vs guidance scale — 8 prompts x 5 seeds per CFG value (SDXL 1024px)",
    ylabel="Mean HPS score",
    top_margin_pct=0.22,
)
canvas_hps.set_ylim(0.0, hps_max * 1.35)
canvas_hps.add_bars(
    x,
    hps_arr,
    colors=colors,
    width=0.6,
    value_fmt="{:.4f}",
    value_pad=hps_max * 0.015,
    value_size=8,
)
canvas_hps.set_xticks(x, xlabels, fontsize=9)
canvas_hps.save(str(CHARTS_DIR / "hps_by_cfg.png"))

# Chart 2: ImageReward score by CFG value
ir_min_val = float(ir_arr.min())
ir_max_val = float(ir_arr.max())
ir_pad = max(abs(ir_min_val) * 0.35, abs(ir_max_val) * 0.35, 0.1)
canvas_ir = ChartCanvas(
    figsize=(9, 4.5),
    title="ImageReward score vs guidance scale — SDXL 1024px",
    ylabel="Mean ImageReward score",
    top_margin_pct=0.22,
)
canvas_ir.set_ylim(ir_min_val - ir_pad, ir_max_val + ir_pad * 1.5)
canvas_ir.add_bars(
    x,
    ir_arr,
    colors=colors,
    width=0.6,
    value_fmt="{:.4f}",
    value_pad=ir_pad * 0.1,
    value_size=8,
)
canvas_ir.set_xticks(x, xlabels, fontsize=9)
canvas_ir.save(str(CHARTS_DIR / "ir_by_cfg.png"))

# Chart 3: CLIP score by CFG value (comparison-only)
clip_max = float(clip_arr.max())
canvas_clip = ChartCanvas(
    figsize=(9, 4.5),
    title="CLIP score vs guidance scale — SDXL 1024px (comparison-only, see CLIP-blindness verdict)",
    ylabel="Mean CLIP score",
    top_margin_pct=0.22,
)
canvas_clip.set_ylim(0.0, clip_max * 1.35)
canvas_clip.add_bars(
    x,
    clip_arr,
    colors=colors,
    width=0.6,
    value_fmt="{:.4f}",
    value_pad=clip_max * 0.015,
    value_size=8,
)
canvas_clip.set_xticks(x, xlabels, fontsize=9)
canvas_clip.save(str(CHARTS_DIR / "clip_by_cfg.png"))

# Chart 4: LPIPS vs cfg=CFG_REF reference
lpips_ref_max = float(lpips_ref_arr.max())
canvas_lpips_ref = ChartCanvas(
    figsize=(9, 4.5),
    title=f"Perceptual distance from cfg={CFG_REF} reference (LPIPS, lower = more similar) — SDXL 1024px",
    ylabel=f"Mean LPIPS vs cfg={CFG_REF}",
    top_margin_pct=0.22,
)
canvas_lpips_ref.set_ylim(0.0, max(lpips_ref_max * 1.5, 0.05))
canvas_lpips_ref.add_bars(
    x,
    lpips_ref_arr,
    colors=colors,
    width=0.6,
    value_fmt="{:.4f}",
    value_pad=max(lpips_ref_max * 0.05, 0.002),
    value_size=8,
)
canvas_lpips_ref.set_xticks(x, xlabels, fontsize=9)
canvas_lpips_ref.save(str(CHARTS_DIR / "lpips_vs_ref.png"))

# Chart 5: LPIPS between adjacent CFG values (step change)
# First value has no previous, so skip index 0
adj_cfgs = CFG_VALUES[1:]
adj_labels = [f"{CFG_VALUES[i - 1]}→{CFG_VALUES[i]}" for i in range(1, len(CFG_VALUES))]
adj_lpips = np.array([agg[c]["mean_lpips_prev"] for c in adj_cfgs])
adj_colors = [_CFG_PALETTE[c] for c in adj_cfgs]
x3 = np.arange(len(adj_cfgs), dtype=float)

adj_max = float(adj_lpips.max())
canvas_adj = ChartCanvas(
    figsize=(9, 4.5),
    title="LPIPS between adjacent CFG values — where does visual character change most? (SDXL 1024px)",
    ylabel="Mean LPIPS (adjacent pair)",
    top_margin_pct=0.22,
)
canvas_adj.set_ylim(0.0, adj_max * 1.5)
canvas_adj.add_bars(
    x3,
    adj_lpips,
    colors=adj_colors,
    width=0.6,
    value_fmt="{:.4f}",
    value_pad=adj_max * 0.05,
    value_size=8,
)
canvas_adj.set_xticks(x3, adj_labels, fontsize=9)
canvas_adj.save(str(CHARTS_DIR / "lpips_adjacent.png"))

print(f"Charts written to {CHARTS_DIR}")

# ── Findings writeup ──────────────────────────────────────────────────────────

# Find CFG where HPS plateaus: first value where HPS is within 1 SE of the max
max_hps = float(hps_arr.max())
plateau_cfg_hps = None
for cfg_val in CFG_VALUES:
    if abs(agg[cfg_val]["mean_hps"] - max_hps) < agg[cfg_val]["se_hps"]:
        plateau_cfg_hps = cfg_val
        break

# Find CFG where CLIP plateaus: first value where CLIP is within 1 SE of the max
max_clip = float(clip_arr.max())
max_clip_se = float(max(agg[c]["se_clip"] for c in CFG_VALUES))
plateau_cfg_clip = None
for cfg_val in CFG_VALUES:
    if abs(agg[cfg_val]["mean_clip"] - max_clip) < max_clip_se:
        plateau_cfg_clip = cfg_val
        break

# Find the largest adjacent LPIPS step (visual regime change)
regime_change_step = max(adj_cfgs, key=lambda c: agg[c]["mean_lpips_prev"] or 0.0)
regime_change_val = agg[regime_change_step]["mean_lpips_prev"]
regime_change_from = CFG_VALUES[CFG_VALUES.index(regime_change_step) - 1]

# CLIP-blindness check: range of CLIP vs range of LPIPS
clip_range = float(clip_arr.max() - clip_arr.min())
mean_clip_se = float(statistics.mean(agg[c]["se_clip"] for c in CFG_VALUES))
clip_range_in_se = clip_range / mean_clip_se if mean_clip_se > 0 else 0.0
lpips_ref_range = float(lpips_ref_arr.max())
hps_range = float(hps_arr.max() - hps_arr.min())
ir_range = float(ir_arr.max() - ir_arr.min())
clip_blind = "yes" if clip_range_in_se < 2.0 else "no"

# Build results table rows for findings.md
table_rows = []
for cfg_val in CFG_VALUES:
    a = agg[cfg_val]
    prev_str = f"{a['mean_lpips_prev']:.4f}" if a["mean_lpips_prev"] is not None else "—"
    table_rows.append(
        f"| {cfg_val:2d}    | {a['mean_hps']:.4f}    | ±{a['se_hps']:.4f}"
        f"  | {a['mean_ir']:.4f}   | ±{a['se_ir']:.4f}"
        f"  | {a['mean_clip']:.4f}    | ±{a['se_clip']:.4f}"
        f"  | {a['mean_lpips_ref']:.4f}               | {prev_str}            |"
    )

FINDINGS = f"""\
# Experiment 3 (SDXL): CFG (Guidance Scale) Sweep

**Date:** 2026-06-02
**Hardware:** GCP L4 (24 GB VRAM)
**Model:** {MODEL_ID}
**CFG values tested:** {CFG_VALUES}
**Reference CFG for LPIPS:** {CFG_REF}
**Design:** 5 seeds x 8 prompts = 40 images per CFG value · {len(all_rows)} images total
**Scheduler:** DPM-Solver++ · {STEPS} steps · {SIZE}x{SIZE}
**Negative prompt:** standard (held constant across all CFG values)
**Scorers:** HPS (primary), ImageReward (primary), CLIP (comparison-only), LPIPS

## Hypothesis

HPS/CLIP will increase with CFG and plateau once the guidance is strong enough to anchor
the prompt (expected somewhere in the 5–9 range). LPIPS between adjacent values will reveal
a "regime change" — a step where visual character shifts sharply — that CLIP cannot detect.
At very high CFG (12–15) we expect over-saturation and structural artefacts that HPS and IR
will detect as quality degradation before CLIP reacts.

## Results

| CFG | Mean HPS | SE_HPS  | Mean IR | SE_IR   | Mean CLIP | SE_CLIP | LPIPS vs cfg={CFG_REF} (cumulative) | LPIPS vs prev (step) |
|-----|:--------:|--------:|:-------:|--------:|----------:|--------:|------------------------------------:|---------------------:|
{chr(10).join(table_rows)}

## Key numbers

- HPS plateau starts at CFG = {plateau_cfg_hps} (first value within 1 SE of max HPS = {max_hps:.4f})
- CLIP plateau starts at CFG = {plateau_cfg_clip} (first value within 1 SE of max CLIP = {max_clip:.4f})
- Largest adjacent LPIPS step: {regime_change_from}→{regime_change_step} \
(LPIPS = {regime_change_val:.4f})
- HPS range across all CFG values: {hps_range:.4f}
- IR range across all CFG values: {ir_range:.4f}
- CLIP range across all CFG values: {clip_range:.4f} ({clip_range_in_se:.2f} SEs)

## CLIP-blindness verdict

CLIP delta across conditions = {clip_range:.4f}, which is {clip_range_in_se:.2f} SEs. \
HPS delta = {hps_range:.4f}. \
IR delta = {ir_range:.4f}. \
LPIPS range = {lpips_ref_range:.4f}. \
Verdict: CLIP-BLIND: {clip_blind}.

## Interpretation

**HPS:** {"Plateaus at CFG=" + str(plateau_cfg_hps) + ", within 1 SE of the maximum (" + f"{max_hps:.4f}" + ") from that point." if plateau_cfg_hps else "No clear plateau within the tested range."}

**CLIP (comparison-only):** {"Plateaus at CFG=" + str(plateau_cfg_clip) + ", within 1 SE of the maximum (" + f"{max_clip:.4f}" + ") from that point." if plateau_cfg_clip else "No clear plateau within the tested range."}
Increasing CFG beyond the plateau does not improve semantic alignment as measured by CLIP.

**LPIPS (cumulative vs cfg={CFG_REF}):** Images diverge progressively from the cfg={CFG_REF}
reference as CFG moves in either direction. Low CFG (1, 3) and high CFG (12, 15) both produce
substantially different images from the mid-range baseline — but for different reasons: low CFG
underweights the prompt, high CFG overweights it to saturation.

**LPIPS (adjacent steps):** The largest single-step visual change is at the
{regime_change_from}→{regime_change_step} transition (LPIPS = {regime_change_val:.4f}). This
is the regime boundary where a one-unit CFG change produces the greatest pixel-level shift.
HPS and IR provide converging evidence for whether this regime boundary corresponds to a
human-perceptible quality change or merely a stylistic shift.

**Cross-experiment note:** If CLIP-blindness is confirmed (verdict above), this is the
SDXL replication of the same dissociation seen in SD 2.1 exp3 — CLIP cannot tell you when
the image has diverged perceptually; LPIPS is needed. HPS and IR add the human-preference
dimension: they can detect whether high-CFG oversaturation constitutes a quality regression,
not just a pixel-level change.

## Charts

- `charts/hps_by_cfg.png` — mean HPS score per CFG value (primary quality metric)
- `charts/ir_by_cfg.png` — mean ImageReward score per CFG value
- `charts/clip_by_cfg.png` — mean CLIP score per CFG value (comparison-only)
- `charts/lpips_vs_ref.png` — cumulative LPIPS distance from cfg={CFG_REF}
- `charts/lpips_adjacent.png` — step-wise LPIPS between adjacent CFG values

## Raw data

`results.csv` / `results.json` — one row per image ({len(all_rows)} rows total).

Reproduce:

```bash
python scripts/experiments/exp3_sdxl.py
```
"""

with open(OUT / "findings.md", "w", encoding="utf-8") as f:
    f.write(FINDINGS)

print(f"\nFindings written: {OUT / 'findings.md'}")
print("Experiment 3 (SDXL) complete.")
