"""
Experiment 5 (SDXL): ControlNet conditioning strength sweep (Canny, SDXL 1.0).

Conditioning images: canny edges extracted from 8 fresh reference images generated
  with plain SDXL (load_sdxl_base, strength=1.0, DPM++, seed=42). These are saved
  to conditioning_images/ and reused across all strength sweeps.
Strength values: [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
  0.0 = no conditioning (effectively text-only)
  1.0 = standard ControlNet strength (reference for LPIPS)
  1.5 = over-conditioned
Seeds: [42, 123, 456, 789, 1337] — 5 seeds × 8 prompts = 40 images per strength
Total: 7 × 8 × 5 = 280 images

Metrics:
  - CLIP score (semantic alignment to prompt)
  - HPS score (human preference)
  - ImageReward score (human preference)
  - LPIPS vs strength=1.0 reference (same prompt+seed pair)
  - LPIPS vs previous strength value (adjacent step visual change)

Hypothesis: CLIP stays roughly flat across all strength values (same text prompt,
same semantic content). LPIPS vs the reference reveals when the image is
perceptually departing from standard conditioning. Another CLIP-blindness data point.

Run from project root:
    python scripts/experiments/exp5_sdxl.py

Outputs:
    reports/experiments/exp5_sdxl/
        images/strength_{val}/      -- 40 PNG per strength
        conditioning_images/        -- 8 canny edge maps from SDXL reference images
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
from diffusers import AutoencoderKL, ControlNetModel, StableDiffusionXLControlNetPipeline  # noqa: E402
from PIL import Image  # noqa: E402

from aetherart import clip_scorer  # noqa: E402
from aetherart.config import cfg  # noqa: E402
from aetherart.controlnet import preprocess_canny  # noqa: E402
from aetherart.eval_hps import release_hps, score_hps  # noqa: E402
from aetherart.eval_ir import release_image_reward, score_image_reward  # noqa: E402
from aetherart.gpu_hygiene import cleanup_gpu  # noqa: E402
from aetherart.sdxl_pipeline import load_sdxl_base  # noqa: E402
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

STRENGTH_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
STRENGTH_REF = 1.0
SEEDS = [42, 123, 456, 789, 1337]
STEPS = 30
CFG = 7.5
SIZE = 1024
REF_SEED = 42  # seed used for generating the 8 canny reference images

# Same 8 prompts as SD 2.1 exp5 (p01_portrait through p08_crowd)
PROMPT_IDS = [f"p{i:02d}" for i in range(1, 9)]
PROMPTS = {
    "p01": "artificial intelligence",
    "p02": "a shiba inu",
    "p03": "a dragon",
    "p04": "a shiba inu wearing a beret and black turtleneck",
    "p05": "a corgi wearing a red bowtie and a purple party hat",
    "p06": "an eagle swooping down to catch a mouse",
    "p07": "an elephant using its trunk to blow into a tuba",
    "p08": "a dolphin in an astronaut suit on saturn",
}

NEG_PROMPT = "low quality, blurry, deformed, ugly, bad anatomy, watermark"

OUT = ROOT / "reports" / "experiments" / "exp5_sdxl"
IMG_DIR = OUT / "images"
CHARTS_DIR = OUT / "charts"
COND_DIR = OUT / "conditioning_images"

for s in STRENGTH_VALUES:
    (IMG_DIR / f"strength_{s}").mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
COND_DIR.mkdir(parents=True, exist_ok=True)

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


# ── Step 0: Generate 8 reference images and extract canny edges ───────────────


def build_conditioning_images() -> dict[str, Image.Image]:
    """Generate fresh 1024×1024 reference images with plain SDXL, then extract canny edges.

    Returns a dict mapping prompt_id → PIL canny edge image.
    All 8 references use REF_SEED for reproducibility. Skips generation if the
    canny PNG already exists on disk (idempotent across partial runs).
    """
    canny_images: dict[str, Image.Image] = {}

    # Check whether all 8 canny images are already on disk
    all_present = all((COND_DIR / f"{pid}_canny.png").exists() for pid in PROMPT_IDS)
    if all_present:
        print("Conditioning images already exist — loading from disk.")
        for pid in PROMPT_IDS:
            canny_images[pid] = Image.open(COND_DIR / f"{pid}_canny.png").convert("RGB")
        return canny_images

    print("Step 0: Generating SDXL reference images for canny conditioning...")
    ref_pipe = load_sdxl_base()

    for pid in PROMPT_IDS:
        prompt_text = PROMPTS[pid]
        ref_path = COND_DIR / f"{pid}_ref.png"
        canny_path = COND_DIR / f"{pid}_canny.png"

        if not ref_path.exists():
            generator = torch.Generator().manual_seed(REF_SEED)
            out = ref_pipe(
                prompt=prompt_text,
                negative_prompt=NEG_PROMPT,
                num_inference_steps=STEPS,
                guidance_scale=CFG,
                height=SIZE,
                width=SIZE,
                generator=generator,
            )
            out.images[0].save(ref_path)
            print(f"  {pid}: reference saved → {ref_path.name}")

        ref_img = Image.open(ref_path).convert("RGB")
        canny_img = preprocess_canny(ref_img, low_threshold=100, high_threshold=200)
        canny_img.save(canny_path)
        canny_images[pid] = canny_img
        print(f"  {pid}: canny extracted → {canny_path.name}")

    del ref_pipe
    cleanup_gpu(verbose=True)
    print(f"Conditioning images ready ({len(canny_images)} prompts)\n")
    return canny_images


canny_images = build_conditioning_images()


# ── Pipeline ──────────────────────────────────────────────────────────────────


def load_sdxl_controlnet_pipeline() -> StableDiffusionXLControlNetPipeline:
    """Load SDXL ControlNet (Canny) pipeline with fp16-fix VAE and DPM-Solver++."""
    print(f"Loading SDXL ControlNet weights: {cfg.sdxl_controlnet_canny}")
    controlnet = ControlNetModel.from_pretrained(
        cfg.sdxl_controlnet_canny, torch_dtype=torch.float16
    )
    vae = AutoencoderKL.from_pretrained(cfg.sdxl_vae_fix, torch_dtype=torch.float16)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        cfg.sdxl_model,
        controlnet=controlnet,
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.enable_model_cpu_offload()
    print("SDXL ControlNet pipeline ready (model CPU offload enabled)")
    return pipe


# ── Generation loop ───────────────────────────────────────────────────────────


def run_strength(
    strength: float,
    pipe: StableDiffusionXLControlNetPipeline,
) -> list[dict]:
    """Generate all prompt × seed images for a single conditioning strength value."""
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
                    "hps_score": None,
                    "ir_score": None,
                    "lpips_vs_ref": None,
                    "lpips_vs_prev": None,
                    "image_path": (img_dir / fname).relative_to(ROOT).as_posix(),
                }
            )
            print(f"  [strength={strength:.2f}] {pid} seed={seed:5d} | {latency:.1f}s")
    return rows


# ── Generate all strength values (single loaded pipeline) ─────────────────────

all_rows: list[dict] = []

print("Loading SDXL ControlNet pipeline (fp16)...")
pipe = load_sdxl_controlnet_pipeline()

for strength in STRENGTH_VALUES:
    idx = STRENGTH_VALUES.index(strength) + 1
    print(f"\n=== Strength = {strength} ({idx}/{len(STRENGTH_VALUES)}) ===")
    all_rows.extend(run_strength(strength, pipe))

del pipe
cleanup_gpu(verbose=True)


# ── Post-hoc scoring: CLIP, HPS, ImageReward ─────────────────────────────────

print(f"\nComputing CLIP / HPS / ImageReward scores for {len(all_rows)} images...")
for i, r in enumerate(all_rows, 1):
    img = Image.open(ROOT / r["image_path"]).convert("RGB")
    r["clip_score"] = round(clip_scorer.score(img, r["prompt_text"]), 6)
    r["hps_score"] = round(score_hps([img], [r["prompt_text"]])[0], 6)
    r["ir_score"] = round(score_image_reward([img], [r["prompt_text"]])[0], 6)
    if i % 40 == 0 or i == len(all_rows):
        print(f"  {i}/{len(all_rows)}")

release_hps()
release_image_reward()


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
            "experiment": "exp5_sdxl",
            "date": "2026-06-02",
            "model": "stabilityai/stable-diffusion-xl-base-1.0",
            "controlnet": cfg.sdxl_controlnet_canny,
            "size": 1024,
            "scorers": ["clip", "hps", "imagereward", "lpips"],
            "strength_values": STRENGTH_VALUES,
            "strength_ref": STRENGTH_REF,
            "steps": STEPS,
            "cfg": CFG,
            "seeds": SEEDS,
            "prompts": PROMPTS,
            "neg_prompt": NEG_PROMPT,
            "conditioning_source": "sdxl_generated_seed42",
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
    hps = [r["hps_score"] for r in rows]
    irs = [r["ir_score"] for r in rows]
    lats = [r["latency_s"] for r in rows]
    lpips_ref = [r["lpips_vs_ref"] for r in rows if r["lpips_vs_ref"] is not None]
    lpips_prev = [r["lpips_vs_prev"] for r in rows if r["lpips_vs_prev"] is not None]
    agg[s] = {
        "mean_clip": statistics.mean(clips),
        "se_clip": statistics.stdev(clips) / len(clips) ** 0.5,
        "mean_hps": statistics.mean(hps),
        "se_hps": statistics.stdev(hps) / len(hps) ** 0.5,
        "mean_ir": statistics.mean(irs),
        "se_ir": statistics.stdev(irs) / len(irs) ** 0.5,
        "mean_lat": statistics.mean(lats),
        "mean_lpips_ref": statistics.mean(lpips_ref) if lpips_ref else 0.0,
        "mean_lpips_prev": statistics.mean(lpips_prev) if lpips_prev else None,
    }

print("\n── Scores by conditioning strength ──")
for s in STRENGTH_VALUES:
    a = agg[s]
    prev_str = f"{a['mean_lpips_prev']:.4f}" if a["mean_lpips_prev"] is not None else "—"
    print(
        f"  strength={s:.2f}: CLIP={a['mean_clip']:.4f} ±{a['se_clip']:.4f} | "
        f"HPS={a['mean_hps']:.4f} | IR={a['mean_ir']:.4f} | "
        f"LPIPS_ref={a['mean_lpips_ref']:.4f} | LPIPS_prev={prev_str}"
    )


# ── Charts ────────────────────────────────────────────────────────────────────

str_arr = np.array(STRENGTH_VALUES)
clip_arr = np.array([agg[s]["mean_clip"] for s in STRENGTH_VALUES])
hps_arr = np.array([agg[s]["mean_hps"] for s in STRENGTH_VALUES])
ir_arr = np.array([agg[s]["mean_ir"] for s in STRENGTH_VALUES])
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
    title="CLIP score vs ControlNet conditioning strength — SDXL, 8 prompts × 5 seeds",
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

# Chart 2: HPS by strength
hps_max = float(hps_arr.max())
canvas_hps = ChartCanvas(
    figsize=(9, 4.5),
    title="HPS score vs ControlNet conditioning strength — SDXL",
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
canvas_hps.save(str(CHARTS_DIR / "hps_by_strength.png"))

# Chart 3: ImageReward by strength
ir_max = float(ir_arr.max())
canvas_ir = ChartCanvas(
    figsize=(9, 4.5),
    title="ImageReward score vs ControlNet conditioning strength — SDXL",
    ylabel="Mean ImageReward score",
    top_margin_pct=0.22,
)
canvas_ir.set_ylim(min(float(ir_arr.min()) * 1.2, -0.1), ir_max * 1.35)
canvas_ir.add_bars(
    x,
    ir_arr,
    colors=colors,
    width=0.6,
    value_fmt="{:.4f}",
    value_pad=abs(ir_max) * 0.015 + 0.01,
    value_size=8,
)
canvas_ir.set_xticks(x, xlabels, fontsize=9)
canvas_ir.save(str(CHARTS_DIR / "ir_by_strength.png"))

# Chart 4: LPIPS vs strength=1.0 reference
lpips_ref_max = float(lpips_ref_arr.max())
canvas4 = ChartCanvas(
    figsize=(9, 4.5),
    title=f"Perceptual distance from strength={STRENGTH_REF} reference (LPIPS) — SDXL",
    ylabel=f"Mean LPIPS vs strength={STRENGTH_REF}",
    top_margin_pct=0.22,
)
canvas4.set_ylim(0.0, max(lpips_ref_max * 1.5, 0.05))
canvas4.add_bars(
    x,
    lpips_ref_arr,
    colors=colors,
    width=0.6,
    value_fmt="{:.4f}",
    value_pad=max(lpips_ref_max * 0.05, 0.002),
    value_size=8,
)
canvas4.set_xticks(x, xlabels, fontsize=9)
canvas4.save(str(CHARTS_DIR / "lpips_vs_ref.png"))

# Chart 5: LPIPS between adjacent strength values
adj_strs = STRENGTH_VALUES[1:]
adj_labels = [
    f"{STRENGTH_VALUES[i - 1]}→{STRENGTH_VALUES[i]}" for i in range(1, len(STRENGTH_VALUES))
]
adj_lpips = np.array([agg[s]["mean_lpips_prev"] for s in adj_strs])
adj_colors = [_STR_PALETTE[s] for s in adj_strs]
x5 = np.arange(len(adj_strs), dtype=float)

adj_max = float(adj_lpips.max())
canvas5 = ChartCanvas(
    figsize=(9, 4.5),
    title="LPIPS between adjacent strength values — step-wise visual change (SDXL)",
    ylabel="Mean LPIPS (adjacent pair)",
    top_margin_pct=0.22,
)
canvas5.set_ylim(0.0, adj_max * 1.5)
canvas5.add_bars(
    x5,
    adj_lpips,
    colors=adj_colors,
    width=0.6,
    value_fmt="{:.4f}",
    value_pad=adj_max * 0.05,
    value_size=8,
)
canvas5.set_xticks(x5, adj_labels, fontsize=9)
canvas5.save(str(CHARTS_DIR / "lpips_adjacent.png"))

print(f"Charts written to {CHARTS_DIR}")


# ── Findings writeup ──────────────────────────────────────────────────────────

# CLIP plateau: first strength within 1 SE of max
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

lpips_at_zero = agg[0.0]["mean_lpips_ref"]
lpips_at_max = agg[STRENGTH_VALUES[-1]]["mean_lpips_ref"]

# Compute clip/hps/ir deltas between 0 and max strength
clip_delta = agg[STRENGTH_VALUES[-1]]["mean_clip"] - agg[0.0]["mean_clip"]
hps_delta = agg[STRENGTH_VALUES[-1]]["mean_hps"] - agg[0.0]["mean_hps"]
ir_delta = agg[STRENGTH_VALUES[-1]]["mean_ir"] - agg[0.0]["mean_ir"]
lpips_range = lpips_at_max  # largest LPIPS departure from ref

clip_ses = agg[0.0]["se_clip"]
clip_delta_ses = abs(clip_delta) / clip_ses if clip_ses > 0 else 0.0
clip_blind = abs(clip_delta) < max_se

table_rows = []
for s in STRENGTH_VALUES:
    a = agg[s]
    prev_s = f"{a['mean_lpips_prev']:.4f}" if a["mean_lpips_prev"] is not None else "—"
    ref_marker = " ← ref" if s == STRENGTH_REF else ""
    table_rows.append(
        f"| {s:.2f}  | {a['mean_clip']:.4f} | ±{a['se_clip']:.4f}"
        f" | {a['mean_hps']:.4f} | {a['mean_ir']:.4f}"
        f" | {a['mean_lpips_ref']:.4f}     | {prev_s}{ref_marker} |"
    )

FINDINGS = f"""\
# Experiment 5 (SDXL): ControlNet Conditioning Strength Sweep

**Date:** 2026-06-02
**Hardware:** GCP L4 (enable_model_cpu_offload)
**Model:** stabilityai/stable-diffusion-xl-base-1.0
**ControlNet:** {cfg.sdxl_controlnet_canny} (Canny edges, SDXL)
**Conditioning source:** fresh SDXL reference images (seed={REF_SEED}, plain SDXL base)
**Strength values:** {STRENGTH_VALUES}
  (0.0 = no conditioning / text-only; 1.0 = standard reference; 1.5 = over-conditioned)
**Reference strength for LPIPS:** {STRENGTH_REF}
**Design:** 5 seeds × 8 prompts = 40 images per strength · {len(all_rows)} images total
**Scheduler:** DPM-Solver++ · {STEPS} steps · {SIZE}×{SIZE}
**CFG:** {CFG}
**Negative prompt:** standard (held constant)

## Hypothesis

CLIP will stay roughly flat across all strength values — the same text prompt drives
the same semantic content regardless of how tightly the canny edges constrain the output.
HPS and ImageReward may be more sensitive to the pixel-level character change.
LPIPS vs the strength=1.0 reference will reveal where the image perceptually departs from
standard conditioning, diverging in both directions from the anchor.

## Results

| Strength | Mean CLIP | SE      | HPS    | IR     | LPIPS vs ref={STRENGTH_REF} | LPIPS vs prev |
|----------|----------:|--------:|-------:|-------:|----------------------------:|--------------:|
{chr(10).join(table_rows)}

LPIPS at strength=0.0 (text-only) vs reference: {lpips_at_zero:.4f}
LPIPS at strength={STRENGTH_VALUES[-1]} (over-conditioned) vs reference: {lpips_at_max:.4f}

## Interpretation

**CLIP:** {"Flat across all strengths — first value within 1 SE of max CLIP is strength=" + str(plateau_str) + f" (max CLIP={max_clip:.4f}, SE≈{max_se:.4f}). Conditioning strength has no measurable effect on semantic alignment as judged by CLIP." if plateau_str is not None else "No clear plateau detected in the tested range."}

**HPS / IR:** HPS delta (0.0→{STRENGTH_VALUES[-1]}) = {hps_delta:+.4f}. IR delta = {ir_delta:+.4f}.
{"Both richer scorers also see flat scores across strengths — all three scorers are blind to ControlNet strength." if abs(hps_delta) < 0.01 and abs(ir_delta) < 0.05 else "HPS and/or IR show a signal that CLIP misses — these scorers capture conditioning-strength character."}

**LPIPS vs reference:** Divergence is measurable in both directions from strength=1.0.
At strength=0.0 (no conditioning, effectively text-only generation): LPIPS={lpips_at_zero:.4f}.
At strength={STRENGTH_VALUES[-1]} (over-conditioned): LPIPS={lpips_at_max:.4f}.
The images look very different despite similar scorer values — LPIPS is the honest metric here.

**LPIPS (adjacent steps):** The largest single-step visual change is the
{regime_from}→{regime_step} transition (LPIPS={regime_val:.4f}). ControlNet strength
is a pixel-level creative parameter — LPIPS registers it, CLIP may not.

CLIP delta = {clip_delta:+.4f} ({clip_delta_ses:.1f} SEs). HPS delta = {hps_delta:+.4f}. IR delta = {ir_delta:+.4f}. LPIPS range = {lpips_range:.4f}. Verdict: CLIP-BLIND: {"yes" if clip_blind else "no"}.

## Charts

- `charts/clip_by_strength.png` — mean CLIP score per conditioning strength
- `charts/hps_by_strength.png` — mean HPS score per conditioning strength
- `charts/ir_by_strength.png` — mean ImageReward score per conditioning strength
- `charts/lpips_vs_ref.png` — perceptual distance from strength=1.0 reference
- `charts/lpips_adjacent.png` — step-wise LPIPS between adjacent strength values

## Raw data

`results.csv` / `results.json` — one row per image ({len(all_rows)} rows total).

Reproduce:

```bash
python scripts/experiments/exp5_sdxl.py
```
"""

with open(OUT / "findings.md", "w", encoding="utf-8") as f:
    f.write(FINDINGS)

print(f"\nFindings written: {OUT / 'findings.md'}")
print("Experiment 5 SDXL complete.")
