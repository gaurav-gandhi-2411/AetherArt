"""
Experiment 7 (SDXL): LoRA training data size ablation — 20 / 40 / 80 images.

Trains rank-8 SDXL Ukiyo-e LoRAs on fixed-seed random subsets of the 80-image
WikiArt dataset (seed=42). The 80-image model uses the existing SDXL checkpoint
(downloaded from HF if absent). 20-image and 40-image subsets are subsampled
(without replacement) from the 80-image pool so each larger set strictly contains
all images from the smaller.

Training is skipped if checkpoint already exists (idempotent).

Expected runtime: ~3-4h training + ~1h generation+eval on GCP L4.

Run from project root:
    python scripts/experiments/exp7_sdxl.py

Outputs:
    data/lora/ukiyo-e-sdxl-data20/   (subset data + checkpoint trained here)
    data/lora/ukiyo-e-sdxl-data40/   (subset data + checkpoint trained here)
    reports/experiments/exp7_sdxl/
        images/data_{n}/
        results.csv
        results.json
        charts/
        findings.md
"""

from __future__ import annotations

import atexit
import csv
import json
import os
import random
import shutil
import statistics
import subprocess
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
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline  # noqa: E402
from PIL import Image  # noqa: E402

from aetherart import clip_scorer  # noqa: E402
from aetherart.eval_hps import release_hps, score_hps  # noqa: E402
from aetherart.eval_ir import release_image_reward, score_image_reward  # noqa: E402
from aetherart.gpu_hygiene import cleanup_gpu  # noqa: E402
from aetherart.visualization import BLUE, GREEN, ORANGE, ChartCanvas  # noqa: E402

atexit.register(cleanup_gpu)

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DATA_SIZES = [20, 40, 80]
SIZE_COLORS = {20: ORANGE, 40: BLUE, 80: GREEN}
SUBSET_SEED = 42
SEEDS = [42, 123, 456, 789, 1337]
STEPS = 50
CFG = 7.0
SIZE = 1024
TRAIN_STEPS = 1500
TRAIN_SEED = 42
RANK = 8

# Source data: the canonical 80-image SDXL Ukiyo-e training set
SOURCE_DATA_DIR = ROOT / "data" / "lora" / "ukiyo-e"
SOURCE_METADATA = SOURCE_DATA_DIR / "metadata.jsonl"
SOURCE_IMAGES_DIR = SOURCE_DATA_DIR / "images"

# Rank-8 SDXL checkpoint repo (80-image baseline)
LORA_REPO = "gauravgandhi2411/aetherart-ukiyo-sdxl"

# 80-image model: existing SDXL rank-8; 20 and 40 are trained here.
# train_lora.py --base sdxl writes to training_output_sdxl/ inside --output-dir.
LORA_DIRS = {
    20: ROOT / "data" / "lora" / "ukiyo-e-sdxl-data20",
    40: ROOT / "data" / "lora" / "ukiyo-e-sdxl-data40",
    80: ROOT / "data" / "lora" / "ukiyo-e-sdxl",
}
LORA_PATHS = {
    20: ROOT / "data" / "lora" / "ukiyo-e-sdxl-data20" / "pytorch_lora_weights.safetensors",
    40: ROOT / "data" / "lora" / "ukiyo-e-sdxl-data40" / "pytorch_lora_weights.safetensors",
    80: ROOT / "data" / "lora" / "ukiyo-e-sdxl" / "pytorch_lora_weights.safetensors",
}

# 6 prompts: same as SD 2.1 exp7 (with "ukyowood" trigger prefix)
PROMPTS = {
    "p01_portrait": (
        "ukyowood ukiyo-e portrait of an elderly woman, dramatic light, woodblock print style"
    ),
    "p02_landscape": (
        "ukyowood ukiyo-e misty mountain valley at sunrise, pine forest, golden hour, woodblock print"
    ),
    "p03_wave": (
        "ukyowood ukiyo-e great wave crashing on rocks, foaming water, dramatic sky, Hokusai style"
    ),
    "p04_arch": (
        "ukyowood ukiyo-e interior of a Japanese temple, wooden pillars, soft lantern light"
    ),
    "p05_texture": (
        "ukyowood ukiyo-e extreme close-up of rough stone wall, water drops, micro detail, woodblock"
    ),
    "p06_crowd": (
        "ukyowood ukiyo-e busy street market in Edo, dozens of people, lantern light, night scene"
    ),
}
NEG_PROMPT = "low quality, blurry, deformed, ugly, bad anatomy, watermark, text, calligraphy"

OUT = ROOT / "reports" / "experiments" / "exp7_sdxl"
IMG_DIR = OUT / "images"
CHARTS_DIR = OUT / "charts"

for n in DATA_SIZES:
    (IMG_DIR / f"data_{n}").mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)


# ── 80-image checkpoint download (if not already local) ───────────────────────


def _ensure_rank8_lora() -> Path:
    """Download the rank-8 SDXL LoRA (80-image baseline) from HF if not on disk."""
    path = LORA_PATHS[80]
    if path.exists():
        return path
    from huggingface_hub import hf_hub_download

    print(f"[data 80] Downloading checkpoint from {LORA_REPO}...")
    path.parent.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        LORA_REPO,
        "pytorch_lora_weights.safetensors",
        local_dir=str(path.parent),
    )
    return path


# ── Subset data preparation ───────────────────────────────────────────────────


def build_subset(n: int) -> None:
    """Create data/lora/ukiyo-e-sdxl-dataN/ with N images and matching metadata.jsonl.

    The subset is a deterministic fixed-seed sample of the 80-image source, so
    20 ⊆ 40 ⊆ 80 (each larger set contains all images from the smaller).
    """
    subset_dir = LORA_DIRS[n]
    subset_images_dir = subset_dir / "images"
    subset_metadata = subset_dir / "metadata.jsonl"

    existing_images = list(subset_images_dir.glob("*.jpg")) if subset_images_dir.exists() else []
    if subset_metadata.exists() and len(existing_images) >= n:
        print(f"[data {n}] subset data already exists — skipping creation")
        return

    # Load all 80 metadata entries from the canonical SDXL training set
    all_entries = []
    with open(SOURCE_METADATA, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                all_entries.append(json.loads(stripped))

    assert len(all_entries) == 80, f"Expected 80 entries, got {len(all_entries)}"

    # Fixed-seed sample: 20 ⊆ 40 ⊆ 80
    rng = random.Random(SUBSET_SEED)
    indices_80 = list(range(80))
    rng.shuffle(indices_80)
    selected_indices = sorted(indices_80[:n])
    selected = [all_entries[i] for i in selected_indices]

    subset_images_dir.mkdir(parents=True, exist_ok=True)

    for entry in selected:
        src = SOURCE_DATA_DIR / entry["file_name"]
        dst = subset_images_dir / Path(entry["file_name"]).name
        if not dst.exists():
            shutil.copy2(src, dst)

    with open(subset_metadata, "w", encoding="utf-8") as f:
        for entry in selected:
            adjusted = dict(entry)
            adjusted["file_name"] = f"images/{Path(entry['file_name']).name}"
            f.write(json.dumps(adjusted) + "\n")

    print(f"[data {n}] subset prepared: {n} images at {subset_dir}")


# ── Training (skipped if checkpoint exists) ───────────────────────────────────


def train_subset(n: int) -> None:
    """Train a rank-8 SDXL LoRA on the N-image subset via train_lora.py --base sdxl.

    The training wrapper writes checkpoints to training_output_sdxl/ inside the
    specified --output-dir. After training we expect the final weights at
    LORA_PATHS[n] (directly in the output dir, not in the subdirectory).
    """
    ckpt = LORA_PATHS[n]
    if ckpt.exists():
        print(f"[data {n}] checkpoint exists — skipping training: {ckpt}")
        return

    out_dir = LORA_DIRS[n]
    data_dir = LORA_DIRS[n]
    print(f"\n[data {n}] Starting SDXL training → {out_dir}")
    print(f"[data {n}] Expected time: ~1.5-2h on GCP L4")

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train_lora.py"),
        "--base", "sdxl",
        "--rank", str(RANK),
        "--max-train-steps", str(TRAIN_STEPS),
        "--seed", str(TRAIN_SEED),
        "--output-dir", str(out_dir),
        "--data-dir", str(data_dir),
    ]

    t0 = time.monotonic()
    proc = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.monotonic() - t0
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)

    if proc.returncode != 0:
        print(f"[data {n}] TRAINING FAILED (exit {proc.returncode}) after {h:02d}:{m:02d}:{s:02d}")
        sys.exit(proc.returncode)
    print(f"[data {n}] Training complete in {h:02d}:{m:02d}:{s:02d}")

    # The diffusers SDXL training script outputs to training_output_sdxl/ within out_dir.
    # If the final weights landed there instead of directly in out_dir, surface the path.
    if not ckpt.exists():
        fallback = out_dir / "training_output_sdxl" / "pytorch_lora_weights.safetensors"
        if fallback.exists():
            print(f"[data {n}] Note: checkpoint found at {fallback} (not {ckpt})")
        else:
            print(f"[data {n}] WARNING: expected checkpoint not found at {ckpt}")


# ── Pipeline ──────────────────────────────────────────────────────────────────


def load_pipeline_with_lora(n: int) -> StableDiffusionXLPipeline:
    """Load SDXL base pipeline with fp16-fix VAE and attach the given data-size LoRA."""
    from diffusers import AutoencoderKL

    from aetherart.config import cfg

    ckpt = LORA_PATHS[n]
    if not ckpt.exists():
        raise FileNotFoundError(f"LoRA checkpoint missing for data size {n}: {ckpt}")

    vae = AutoencoderKL.from_pretrained(cfg.sdxl_vae_fix, torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if torch.cuda.is_available():
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pipe = pipe.to("cuda")

    pipe.load_lora_weights(str(ckpt.parent), weight_name=ckpt.name, adapter_name="ukiyo_e")
    pipe.set_adapters(["ukiyo_e"], adapter_weights=[1.0])
    print(f"[data {n}] SDXL Pipeline + LoRA ready: {ckpt.name}")
    return pipe


# ── Generation loop ───────────────────────────────────────────────────────────


def run_data_size(n: int) -> list[dict]:
    """Generate all prompt × seed images for a single training data size."""
    pipe = load_pipeline_with_lora(n)
    rows: list[dict] = []
    img_dir = IMG_DIR / f"data_{n}"

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
                    "data_size": n,
                    "prompt_id": prompt_id,
                    "prompt_text": prompt_text,
                    "seed": seed,
                    "latency_s": round(latency, 3),
                    "clip_score": None,
                    "hps_score": None,
                    "ir_score": None,
                    "lpips_vs_80": None,
                    "image_path": (img_dir / fname).relative_to(ROOT).as_posix(),
                }
            )
            print(f"  [data {n}] {prompt_id} seed={seed:5d} | {latency:.1f}s")

    del pipe
    cleanup_gpu(verbose=False)
    return rows


# ── Phase 1: Ensure 80-image checkpoint ──────────────────────────────────────

print("\n=== Phase 1: Checkpoint setup (80-image baseline) ===")
_ensure_rank8_lora()

# ── Phase 2: Subset data preparation ─────────────────────────────────────────

print("\n=== Phase 2: Subset data preparation ===")
for n in [20, 40]:
    build_subset(n)

# ── Phase 3: Training ─────────────────────────────────────────────────────────

print("\n=== Phase 3: Training (data sizes 20 and 40) ===")
for n in [20, 40]:
    train_subset(n)

# Verify all checkpoints exist
for n in DATA_SIZES:
    if not LORA_PATHS[n].exists():
        print(f"ERROR: checkpoint missing for data size {n}: {LORA_PATHS[n]}")
        sys.exit(1)
    size_mb = os.path.getsize(LORA_PATHS[n]) / 1e6
    print(f"[data {n}] checkpoint: {LORA_PATHS[n].name}  ({size_mb:.1f} MB)")

# ── Phase 4: Generation ────────────────────────────────────────────────────────

print("\n=== Phase 4: Generation (90 images) ===")
all_rows: list[dict] = []
for n in DATA_SIZES:
    print(f"\n--- Data size {n} ---")
    all_rows.extend(run_data_size(n))

# ── Post-hoc scoring: CLIP, HPS, ImageReward ─────────────────────────────────

print(f"\nComputing CLIP / HPS / ImageReward scores for {len(all_rows)} images...")
for i, r in enumerate(all_rows, 1):
    img = Image.open(ROOT / r["image_path"]).convert("RGB")
    r["clip_score"] = round(clip_scorer.score(img, r["prompt_text"]), 6)
    r["hps_score"] = round(score_hps([img], [r["prompt_text"]])[0], 6)
    r["ir_score"] = round(score_image_reward([img], [r["prompt_text"]])[0], 6)
    if i % 30 == 0 or i == len(all_rows):
        print(f"  {i}/{len(all_rows)}")

release_hps()
release_image_reward()


# ── LPIPS (post-hoc) ─────────────────────────────────────────────────────────

print("\nComputing LPIPS vs 80-image baseline...")
_lpips_fn = lpips_lib.LPIPS(net="alex")
_lpips_fn.eval()

img_index: dict[tuple, str] = {
    (r["data_size"], r["prompt_id"], r["seed"]): r["image_path"] for r in all_rows
}


def _to_t(img: Image.Image) -> torch.Tensor:
    arr = np.array(img.convert("RGB")).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def _lpips_pair(path_a: str, path_b: str) -> float:
    a = Image.open(ROOT / path_a).convert("RGB")
    b = Image.open(ROOT / path_b).convert("RGB")
    with torch.no_grad():
        return round(float(_lpips_fn(_to_t(a), _to_t(b))), 6)


data80_idx = {
    (r["prompt_id"], r["seed"]): r["image_path"] for r in all_rows if r["data_size"] == 80
}

for r in all_rows:
    r80_path = data80_idx[(r["prompt_id"], r["seed"])]
    r["lpips_vs_80"] = 0.0 if r["data_size"] == 80 else _lpips_pair(r["image_path"], r80_path)

print("LPIPS done.")


# ── Save results ──────────────────────────────────────────────────────────────

CSV_PATH = OUT / "results.csv"
JSON_PATH = OUT / "results.json"

csv_fields = [
    "data_size",
    "prompt_id",
    "seed",
    "latency_s",
    "clip_score",
    "hps_score",
    "ir_score",
    "lpips_vs_80",
    "image_path",
]
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(all_rows)

ckpt_sizes = {n: round(os.path.getsize(LORA_PATHS[n]) / 1e6, 2) for n in DATA_SIZES}

with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(
        {
            "experiment": "exp7_sdxl",
            "date": "2026-06-02",
            "model": MODEL_ID,
            "size": 1024,
            "scorers": ["clip", "hps", "imagereward", "lpips"],
            "data_sizes": DATA_SIZES,
            "subset_seed": SUBSET_SEED,
            "rank": RANK,
            "train_steps": TRAIN_STEPS,
            "train_seed": TRAIN_SEED,
            "eval_steps": STEPS,
            "cfg": CFG,
            "seeds": SEEDS,
            "prompts": PROMPTS,
            "neg_prompt": NEG_PROMPT,
            "lora_repo": LORA_REPO,
            "checkpoint_sizes_mb": ckpt_sizes,
            "total_images": len(all_rows),
            "results": all_rows,
        },
        f,
        indent=2,
    )

print(f"Results: {CSV_PATH}")


# ── Per-size aggregates ───────────────────────────────────────────────────────

by_size: dict[int, list[dict]] = {n: [] for n in DATA_SIZES}
for row in all_rows:
    by_size[row["data_size"]].append(row)

agg: dict[int, dict] = {}
for n, rows in by_size.items():
    clips = [r["clip_score"] for r in rows]
    hps = [r["hps_score"] for r in rows]
    irs = [r["ir_score"] for r in rows]
    lats = [r["latency_s"] for r in rows]
    lpips_80 = [r["lpips_vs_80"] for r in rows if r["lpips_vs_80"] is not None]
    agg[n] = {
        "mean_clip": statistics.mean(clips),
        "se_clip": statistics.stdev(clips) / len(clips) ** 0.5,
        "mean_hps": statistics.mean(hps),
        "se_hps": statistics.stdev(hps) / len(hps) ** 0.5,
        "mean_ir": statistics.mean(irs),
        "se_ir": statistics.stdev(irs) / len(irs) ** 0.5,
        "mean_lat": statistics.mean(lats),
        "mean_lpips_80": statistics.mean(lpips_80) if lpips_80 else 0.0,
        "ckpt_mb": ckpt_sizes[n],
    }

print("\n── Scores by training data size ──")
for n in DATA_SIZES:
    aa = agg[n]
    print(
        f"  data={n:3d}: CLIP={aa['mean_clip']:.4f} ±{aa['se_clip']:.4f} | "
        f"HPS={aa['mean_hps']:.4f} | IR={aa['mean_ir']:.4f} | "
        f"LPIPS_vs_80={aa['mean_lpips_80']:.4f}"
    )


# ── Charts ────────────────────────────────────────────────────────────────────

x = np.arange(len(DATA_SIZES), dtype=float)
xlabels = [f"{n} images" for n in DATA_SIZES]
colors = [SIZE_COLORS[n] for n in DATA_SIZES]
clip_arr = np.array([agg[n]["mean_clip"] for n in DATA_SIZES])
hps_arr = np.array([agg[n]["mean_hps"] for n in DATA_SIZES])
ir_arr = np.array([agg[n]["mean_ir"] for n in DATA_SIZES])
lpips_arr = np.array([agg[n]["mean_lpips_80"] for n in DATA_SIZES])

clip_max = float(clip_arr.max())
canvas = ChartCanvas(
    figsize=(7, 4.5),
    title="CLIP score by training data size (rank-8, ukiyo-e, SDXL) — 6 prompts × 5 seeds × 50 steps",
    ylabel="Mean CLIP score",
    top_margin_pct=0.22,
)
canvas.set_ylim(0.0, clip_max * 1.35)
canvas.add_bars(
    x,
    clip_arr,
    colors=colors,
    width=0.5,
    value_fmt="{:.4f}",
    value_pad=clip_max * 0.015,
    value_size=9,
)
canvas.set_xticks(x, xlabels, fontsize=10)
canvas.save(str(CHARTS_DIR / "clip_by_data_size.png"))

hps_max = float(hps_arr.max())
canvas_hps = ChartCanvas(
    figsize=(7, 4.5),
    title="HPS score by training data size (rank-8, ukiyo-e, SDXL)",
    ylabel="Mean HPS score",
    top_margin_pct=0.22,
)
canvas_hps.set_ylim(0.0, hps_max * 1.35)
canvas_hps.add_bars(
    x,
    hps_arr,
    colors=colors,
    width=0.5,
    value_fmt="{:.4f}",
    value_pad=hps_max * 0.015,
    value_size=9,
)
canvas_hps.set_xticks(x, xlabels, fontsize=10)
canvas_hps.save(str(CHARTS_DIR / "hps_by_data_size.png"))

ir_max = float(ir_arr.max())
canvas_ir = ChartCanvas(
    figsize=(7, 4.5),
    title="ImageReward score by training data size (rank-8, ukiyo-e, SDXL)",
    ylabel="Mean ImageReward score",
    top_margin_pct=0.22,
)
canvas_ir.set_ylim(min(float(ir_arr.min()) * 1.2, -0.1), ir_max * 1.35)
canvas_ir.add_bars(
    x,
    ir_arr,
    colors=colors,
    width=0.5,
    value_fmt="{:.4f}",
    value_pad=abs(ir_max) * 0.015 + 0.01,
    value_size=9,
)
canvas_ir.set_xticks(x, xlabels, fontsize=10)
canvas_ir.save(str(CHARTS_DIR / "ir_by_data_size.png"))

lpips_max = float(lpips_arr.max())
canvas4 = ChartCanvas(
    figsize=(7, 4.5),
    title="Perceptual distance from 80-image baseline (LPIPS) — SDXL",
    ylabel="Mean LPIPS vs 80-image model",
    top_margin_pct=0.22,
)
canvas4.set_ylim(0.0, max(lpips_max * 1.5, 0.05))
canvas4.add_bars(
    x,
    lpips_arr,
    colors=colors,
    width=0.5,
    value_fmt="{:.4f}",
    value_pad=max(lpips_max * 0.05, 0.002),
    value_size=9,
)
canvas4.set_xticks(x, xlabels, fontsize=10)
canvas4.save(str(CHARTS_DIR / "lpips_vs_data80.png"))

print(f"Charts written to {CHARTS_DIR}")


# ── Findings writeup ──────────────────────────────────────────────────────────

clip_delta_20_80 = agg[80]["mean_clip"] - agg[20]["mean_clip"]
clip_delta_40_80 = agg[80]["mean_clip"] - agg[40]["mean_clip"]
hps_delta_20_80 = agg[80]["mean_hps"] - agg[20]["mean_hps"]
ir_delta_20_80 = agg[80]["mean_ir"] - agg[20]["mean_ir"]
pooled_se = max(agg[n]["se_clip"] for n in DATA_SIZES)
clip_delta_ses = abs(clip_delta_20_80) / pooled_se if pooled_se > 0 else 0.0
clip_blind = abs(clip_delta_20_80) < pooled_se
lpips_range = agg[20]["mean_lpips_80"]

table_rows = []
for n in DATA_SIZES:
    aa = agg[n]
    baseline_marker = " ← baseline" if n == 80 else ""
    table_rows.append(
        f"| {n:3d}    | {aa['mean_clip']:.4f} | ±{aa['se_clip']:.4f}"
        f" | {aa['mean_hps']:.4f} | {aa['mean_ir']:.4f}"
        f" | {aa['mean_lpips_80']:.4f}       | {aa['ckpt_mb']:.1f} MB{baseline_marker} |"
    )

FINDINGS = f"""\
# Experiment 7 (SDXL): LoRA Training Data Size Ablation

**Date:** 2026-06-02
**Hardware:** GCP L4 (enable_model_cpu_offload)
**Model:** {MODEL_ID}
**LoRA:** ukiyo-e — Japanese woodblock print style — rank-8, SDXL
**Data sizes tested:** {DATA_SIZES} images
**Note:** 200-image condition dropped — dataset contains only 80 images.
**Subset selection:** fixed-seed random sample (seed={SUBSET_SEED}); 20 ⊆ 40 ⊆ 80
**Training:** {TRAIN_STEPS} steps, seed {TRAIN_SEED}, same rank/LR for all sizes
**80-image model:** downloaded from {LORA_REPO} — not retrained here
**Design:** 5 seeds × 6 prompts = 30 images per data size · {len(all_rows)} images total
**Scheduler:** DPM-Solver++ · {STEPS} steps · {SIZE}×{SIZE}
**CFG:** {CFG} (fixed)
**LPIPS reference:** 80-image model (full dataset)

## Hypothesis

More data → better style capture: 20-image model underfits (lower quality, less consistent
style), 80-image model is the best. LPIPS will show the 20-image model diverges more
from the 80-image reference than the 40-image model does. CLIP may or may not detect
this — if it doesn't, this is another CLIP-blindness case. HPS and ImageReward may be
more sensitive to training data volume effects on style quality.

## Results

| Data  | Mean CLIP | SE      | HPS    | IR     | LPIPS vs 80-img    | File size |
|-------|----------:|--------:|-------:|-------:|-------------------:|----------:|
{chr(10).join(table_rows)}

## Interpretation

**CLIP (20-image vs 80-image):** Delta = {clip_delta_20_80:+.4f} ({abs(clip_delta_20_80) / pooled_se:.1f} pooled SEs).
{"Within noise — 20-image SDXL model matches 80-image model semantically by CLIP." if abs(clip_delta_20_80) < pooled_se else "Detectable — data size affects semantic alignment as measured by CLIP."}

**CLIP (40-image vs 80-image):** Delta = {clip_delta_40_80:+.4f} ({abs(clip_delta_40_80) / pooled_se:.1f} pooled SEs).
{"Within noise — 40-image SDXL model matches 80-image model semantically by CLIP." if abs(clip_delta_40_80) < pooled_se else "Detectable — 40 images shows a different CLIP profile than 80."}

**HPS (20-img vs 80-img):** Delta = {hps_delta_20_80:+.4f}.
**IR (20-img vs 80-img):** Delta = {ir_delta_20_80:+.4f}.

**LPIPS vs 80-image baseline:** 20-image LPIPS = {agg[20]["mean_lpips_80"]:.4f};
40-image LPIPS = {agg[40]["mean_lpips_80"]:.4f}.
{"Small LPIPS: data size differences are perceptually minor at SDXL scale." if max(agg[20]["mean_lpips_80"], agg[40]["mean_lpips_80"]) < 0.3 else "Moderate-to-large LPIPS: data size produces visually distinct outputs despite similar CLIP."}

**Checkpoint sizes:** all three are {agg[20]["ckpt_mb"]:.1f} MB — file size is determined
by rank, not training data size. Expected: checkpoint stores rank × hidden_dim weight deltas.

CLIP delta = {clip_delta_20_80:+.4f} ({clip_delta_ses:.1f} SEs). HPS delta = {hps_delta_20_80:+.4f}. IR delta = {ir_delta_20_80:+.4f}. LPIPS range = {lpips_range:.4f}. Verdict: CLIP-BLIND: {"yes" if clip_blind else "no"}.

## Charts

- `charts/clip_by_data_size.png` — mean CLIP score per training data size
- `charts/hps_by_data_size.png` — mean HPS score per training data size
- `charts/ir_by_data_size.png` — mean ImageReward score per training data size
- `charts/lpips_vs_data80.png` — perceptual distance from 80-image baseline

## Raw data

`results.csv` / `results.json` — one row per image ({len(all_rows)} rows total).

Reproduce:

```bash
python scripts/experiments/exp7_sdxl.py
```
"""

with open(OUT / "findings.md", "w", encoding="utf-8") as f:
    f.write(FINDINGS)

print(f"\nFindings written: {OUT / 'findings.md'}")
print("Experiment 7 SDXL complete.")
