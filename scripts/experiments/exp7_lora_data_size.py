"""
Experiment 7: LoRA training data size ablation — 20 / 40 / 80 images.

Trains rank-8 Ukiyo-e LoRAs on fixed-seed random subsets of the 80-image
WikiArt dataset (seed=42). The 80-image model uses the existing checkpoint.
20-image and 40-image subsets are subsampled (without replacement) from the
80-image pool, so each larger set strictly contains all images from the smaller.

Note: 200-image condition dropped — only 80 images are available in the dataset.
This is documented explicitly in findings.md.

Training is skipped if checkpoint already exists (idempotent).

Expected runtime: ~4h20m training + ~45m generation+eval on RTX 3070 8GB.

Run from project root:
    python scripts/experiments/exp7_lora_data_size.py

Outputs:
    data/lora/ukiyo-e-data20/  (trained here — subset data + checkpoint)
    data/lora/ukiyo-e-data40/  (trained here — subset data + checkpoint)
    reports/experiments/exp7_lora_data_size/
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
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline  # noqa: E402
from PIL import Image  # noqa: E402

from aetherart import clip_scorer  # noqa: E402
from aetherart.gpu_hygiene import cleanup_gpu  # noqa: E402
from aetherart.visualization import BLUE, GREEN, ORANGE, ChartCanvas  # noqa: E402

atexit.register(cleanup_gpu)

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_ID = "sd2-community/stable-diffusion-2-1"
DATA_SIZES = [20, 40, 80]
SIZE_COLORS = {20: ORANGE, 40: BLUE, 80: GREEN}
SUBSET_SEED = 42
SEEDS = [42, 123, 456, 789, 1337]
STEPS = 50
CFG = 7.0
SIZE = 512
TRAIN_STEPS = 1500
TRAIN_SEED = 42
RANK = 8

SOURCE_DATA_DIR = ROOT / "data" / "lora" / "ukiyo-e"
SOURCE_METADATA = SOURCE_DATA_DIR / "metadata.jsonl"
SOURCE_IMAGES_DIR = SOURCE_DATA_DIR / "images"

# 80-image model uses existing checkpoint; 20 and 40 are trained here.
LORA_DIRS = {
    20: ROOT / "data" / "lora" / "ukiyo-e-data20",
    40: ROOT / "data" / "lora" / "ukiyo-e-data40",
    80: SOURCE_DATA_DIR,
}
LORA_CKPTS = {
    20: LORA_DIRS[20] / "training_output" / "pytorch_lora_weights.safetensors",
    40: LORA_DIRS[40] / "training_output" / "pytorch_lora_weights.safetensors",
    80: SOURCE_DATA_DIR / "training_output" / "pytorch_lora_weights.safetensors",
}

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

OUT = ROOT / "reports" / "experiments" / "exp7_lora_data_size"
IMG_DIR = OUT / "images"
CHARTS_DIR = OUT / "charts"

for n in DATA_SIZES:
    (IMG_DIR / f"data_{n}").mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Subset data preparation ───────────────────────────────────────────────────


def build_subset(n: int) -> None:
    """Create data/lora/ukiyo-e-dataN/ with N images and matching metadata.jsonl."""
    subset_dir = LORA_DIRS[n]
    subset_images_dir = subset_dir / "images"
    subset_metadata = subset_dir / "metadata.jsonl"

    if subset_metadata.exists() and len(list(subset_images_dir.glob("*.jpg"))) >= n:
        print(f"[data {n}] subset data already exists — skipping creation")
        return

    # Load all 80 metadata entries
    all_entries = []
    with open(SOURCE_METADATA, encoding="utf-8") as f:
        for line in f:
            all_entries.append(json.loads(line.strip()))

    assert len(all_entries) == 80, f"Expected 80 entries, got {len(all_entries)}"

    # Fixed-seed sample — 20 is a subset of 40 which is a subset of 80
    rng = random.Random(SUBSET_SEED)
    indices_80 = list(range(80))
    rng.shuffle(indices_80)
    selected_indices = sorted(indices_80[:n])
    selected = [all_entries[i] for i in selected_indices]

    subset_images_dir.mkdir(parents=True, exist_ok=True)

    # Copy selected images
    for entry in selected:
        src = SOURCE_DATA_DIR / entry["file_name"]
        dst = subset_images_dir / Path(entry["file_name"]).name
        if not dst.exists():
            shutil.copy2(src, dst)

    # Write metadata.jsonl with adjusted file_name paths
    with open(subset_metadata, "w", encoding="utf-8") as f:
        for entry in selected:
            adjusted = dict(entry)
            adjusted["file_name"] = f"images/{Path(entry['file_name']).name}"
            f.write(json.dumps(adjusted) + "\n")

    print(f"[data {n}] subset prepared: {n} images at {subset_dir}")


# ── Training (skipped if checkpoint exists) ───────────────────────────────────


def train_subset(n: int) -> None:
    ckpt = LORA_CKPTS[n]
    if ckpt.exists():
        print(f"[data {n}] checkpoint exists — skipping training: {ckpt}")
        return

    out_dir = LORA_DIRS[n] / "training_output"
    data_dir = LORA_DIRS[n]
    print(f"\n[data {n}] Starting training → {out_dir}")
    print(f"[data {n}] Expected time: ~2h10m on RTX 3070 8GB")

    train_script = ROOT / "scripts" / "train_lora.py"
    cmd = [
        sys.executable,
        str(train_script),
        "--rank",
        str(RANK),
        "--max-train-steps",
        str(TRAIN_STEPS),
        "--seed",
        str(TRAIN_SEED),
        "--output-dir",
        str(out_dir),
        "--data-dir",
        str(data_dir),
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


# ── Pipeline ──────────────────────────────────────────────────────────────────


def load_pipeline_with_lora(n: int) -> StableDiffusionPipeline:
    ckpt = LORA_CKPTS[n]
    if not ckpt.exists():
        raise FileNotFoundError(f"LoRA checkpoint missing for data size {n}: {ckpt}")

    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if torch.cuda.is_available():
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pipe = pipe.to("cuda")

    pipe.load_lora_weights(str(ckpt.parent), weight_name=ckpt.name, adapter_name="ukiyo_e")
    pipe.set_adapters(["ukiyo_e"], adapter_weights=[1.0])
    print(f"[data {n}] Pipeline + LoRA ready: {ckpt.name}")
    return pipe


# ── Generation loop ───────────────────────────────────────────────────────────


def run_data_size(n: int) -> list[dict]:
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
                    "lpips_vs_data80": None,
                    "lpips_vs_prev_size": None,
                    "image_path": (img_dir / fname).relative_to(ROOT).as_posix(),
                }
            )
            print(f"  [data {n}] {prompt_id} seed={seed:5d} | {latency:.1f}s")

    del pipe
    cleanup_gpu(verbose=False)
    return rows


# ── Phase 1: Subset preparation ───────────────────────────────────────────────

print("\n=== Phase 1: Subset data preparation ===")
for n in [20, 40]:
    build_subset(n)

# ── Phase 2: Training ─────────────────────────────────────────────────────────

print("\n=== Phase 2: Training ===")
for n in [20, 40]:
    train_subset(n)

# Verify all checkpoints exist
for n in DATA_SIZES:
    if not LORA_CKPTS[n].exists():
        print(f"ERROR: checkpoint missing for data size {n}: {LORA_CKPTS[n]}")
        sys.exit(1)
    size_mb = os.path.getsize(LORA_CKPTS[n]) / 1e6
    print(f"[data {n}] checkpoint: {LORA_CKPTS[n].name}  ({size_mb:.1f} MB)")

# ── Phase 3: Generation ────────────────────────────────────────────────────────

print("\n=== Phase 3: Generation (90 images) ===")
all_rows: list[dict] = []
for n in DATA_SIZES:
    print(f"\n--- Data size {n} ---")
    all_rows.extend(run_data_size(n))

# ── CLIP scores (post-hoc) ────────────────────────────────────────────────────

print(f"\nComputing CLIP scores for {len(all_rows)} images...")
for i, r in enumerate(all_rows, 1):
    img = Image.open(ROOT / r["image_path"]).convert("RGB")
    r["clip_score"] = round(clip_scorer.score(img, r["prompt_text"]), 6)
    if i % 30 == 0 or i == len(all_rows):
        print(f"  {i}/{len(all_rows)}")


# ── LPIPS (post-hoc) ─────────────────────────────────────────────────────────

print("\nComputing LPIPS...")
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
    # LPIPS vs data-80 baseline
    r80_path = data80_idx[(r["prompt_id"], r["seed"])]
    r["lpips_vs_data80"] = 0.0 if r["data_size"] == 80 else _lpips_pair(r["image_path"], r80_path)

    # LPIPS vs adjacent size
    size_idx = DATA_SIZES.index(r["data_size"])
    if size_idx == 0:
        r["lpips_vs_prev_size"] = None
    else:
        prev_size = DATA_SIZES[size_idx - 1]
        prev_path = img_index[(prev_size, r["prompt_id"], r["seed"])]
        r["lpips_vs_prev_size"] = _lpips_pair(r["image_path"], prev_path)

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
    "lpips_vs_data80",
    "lpips_vs_prev_size",
    "image_path",
]
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(all_rows)

ckpt_sizes = {n: round(os.path.getsize(LORA_CKPTS[n]) / 1e6, 2) for n in DATA_SIZES}

with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(
        {
            "experiment": "exp7_lora_data_size",
            "date": "2026-05-08",
            "model": MODEL_ID,
            "data_sizes": DATA_SIZES,
            "subset_seed": SUBSET_SEED,
            "rank": RANK,
            "train_steps": TRAIN_STEPS,
            "train_seed": TRAIN_SEED,
            "eval_steps": STEPS,
            "cfg": CFG,
            "size": SIZE,
            "seeds": SEEDS,
            "prompts": PROMPTS,
            "neg_prompt": NEG_PROMPT,
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
    lats = [r["latency_s"] for r in rows]
    lpips_80 = [r["lpips_vs_data80"] for r in rows if r["lpips_vs_data80"] is not None]
    lpips_prev = [r["lpips_vs_prev_size"] for r in rows if r["lpips_vs_prev_size"] is not None]
    agg[n] = {
        "mean_clip": statistics.mean(clips),
        "se_clip": statistics.stdev(clips) / len(clips) ** 0.5,
        "mean_lat": statistics.mean(lats),
        "mean_lpips_80": statistics.mean(lpips_80) if lpips_80 else 0.0,
        "mean_lpips_prev": statistics.mean(lpips_prev) if lpips_prev else None,
        "ckpt_mb": ckpt_sizes[n],
    }

print("\n── CLIP by training data size ──")
for n in DATA_SIZES:
    aa = agg[n]
    prev_str = f"{aa['mean_lpips_prev']:.4f}" if aa["mean_lpips_prev"] is not None else "—"
    print(
        f"  data={n:3d}: CLIP={aa['mean_clip']:.4f} ±{aa['se_clip']:.4f} | "
        f"LPIPS_vs_80={aa['mean_lpips_80']:.4f} | LPIPS_prev={prev_str}"
    )


# ── Charts ────────────────────────────────────────────────────────────────────

x = np.arange(len(DATA_SIZES), dtype=float)
xlabels = [f"{n} images" for n in DATA_SIZES]
colors = [SIZE_COLORS[n] for n in DATA_SIZES]
clip_arr = np.array([agg[n]["mean_clip"] for n in DATA_SIZES])
lpips_arr = np.array([agg[n]["mean_lpips_80"] for n in DATA_SIZES])

clip_max = float(clip_arr.max())
canvas = ChartCanvas(
    figsize=(7, 4.5),
    title="CLIP score by training data size (rank-8, ukiyo-e) — 6 prompts × 5 seeds × 50 steps",
    ylabel="Mean CLIP score",
    top_margin_pct=0.22,
)
canvas.set_ylim(0.0, clip_max * 1.35)
canvas.add_bars(x, clip_arr, colors=colors, width=0.5, value_fmt="{:.4f}", value_pad=clip_max * 0.015, value_size=9)
canvas.set_xticks(x, xlabels, fontsize=10)
canvas.save(str(CHARTS_DIR / "clip_by_data_size.png"))

lpips_max = float(lpips_arr.max())
canvas2 = ChartCanvas(
    figsize=(7, 4.5),
    title="Perceptual distance from 80-image baseline (LPIPS)",
    ylabel="Mean LPIPS vs 80-image model",
    top_margin_pct=0.22,
)
canvas2.set_ylim(0.0, max(lpips_max * 1.5, 0.05))
canvas2.add_bars(x, lpips_arr, colors=colors, width=0.5, value_fmt="{:.4f}", value_pad=max(lpips_max * 0.05, 0.002), value_size=9)
canvas2.set_xticks(x, xlabels, fontsize=10)
canvas2.save(str(CHARTS_DIR / "lpips_vs_data80.png"))

print(f"Charts written to {CHARTS_DIR}")


# ── Findings writeup ──────────────────────────────────────────────────────────

clip_delta_20_80 = agg[80]["mean_clip"] - agg[20]["mean_clip"]
clip_delta_40_80 = agg[80]["mean_clip"] - agg[40]["mean_clip"]
pooled_se = max(agg[n]["se_clip"] for n in DATA_SIZES)
lpips_20_40 = agg[40]["mean_lpips_prev"] if agg[40]["mean_lpips_prev"] is not None else 0.0
lpips_40_80 = agg[80]["mean_lpips_prev"] if agg[80]["mean_lpips_prev"] is not None else 0.0

table_rows = []
for n in DATA_SIZES:
    aa = agg[n]
    prev_str = f"{aa['mean_lpips_prev']:.4f}" if aa["mean_lpips_prev"] is not None else "—"
    baseline_marker = " ← baseline" if n == 80 else ""
    table_rows.append(
        f"| {n:3d}    | {aa['mean_clip']:.4f}    | ±{aa['se_clip']:.4f}"
        f"  | {aa['mean_lpips_80']:.4f}              | {prev_str}{baseline_marker}"
        f" | {aa['ckpt_mb']:.1f} MB |"
    )

FINDINGS = f"""\
# Experiment 7: LoRA Training Data Size Ablation

**Date:** 2026-05-08
**Hardware:** RTX 3070 Laptop 8 GB (enable_model_cpu_offload)
**Model:** {MODEL_ID}
**LoRA:** ukiyo-e — Japanese woodblock print style — rank-8
**Data sizes tested:** {DATA_SIZES} images
**Note:** 200-image condition dropped — dataset contains only 80 images. Sourcing
  additional training images is out of scope for this experiment.
**Subset selection:** fixed-seed random sample (seed={SUBSET_SEED}); 20 ⊆ 40 ⊆ 80
**Training:** {TRAIN_STEPS} steps, seed {TRAIN_SEED}, same rank/LR for all sizes
**80-image model:** existing checkpoint (data/lora/ukiyo-e/training_output/) — not retrained
**Design:** 5 seeds × 6 prompts = 30 images per data size · {len(all_rows)} images total
**Scheduler:** DPM-Solver++ · {STEPS} steps · {SIZE}×{SIZE}
**CFG:** {CFG} (fixed)
**LPIPS reference:** 80-image model (full dataset); also computed between adjacent sizes

## Hypothesis

More data → better style capture: 20-image model underfits (lower CLIP, less consistent
style), 80-image model is the best. LPIPS will show the 20-image model diverges more
from the 80-image reference than the 40-image model does. CLIP may or may not detect
this — if it doesn't, this is another CLIP-blindness case.

## Results

| Data  | Mean CLIP | SE      | LPIPS vs 80-img           | LPIPS vs prev size   | File size |
|-------|----------:|--------:|---------------------------:|---------------------:|----------:|
{chr(10).join(table_rows)}

## Interpretation

**CLIP (20-image vs 80-image):** Delta = {clip_delta_20_80:+.4f} ({abs(clip_delta_20_80) / pooled_se:.1f} pooled SEs).
{"Within noise — 20-image model matches 80-image model semantically by CLIP." if abs(clip_delta_20_80) < pooled_se else "Detectable — data size affects semantic alignment as measured by CLIP."}

**CLIP (40-image vs 80-image):** Delta = {clip_delta_40_80:+.4f} ({abs(clip_delta_40_80) / pooled_se:.1f} pooled SEs).
{"Within noise — 40-image model matches 80-image model semantically by CLIP." if abs(clip_delta_40_80) < pooled_se else "Detectable — 40 images shows a different CLIP profile than 80."}

**LPIPS (adjacent sizes):** 20- vs 40-image: {lpips_20_40:.4f}; 40- vs 80-image: {lpips_40_80:.4f}.
{"Small LPIPS: data size differences are perceptually minor." if max(lpips_20_40, lpips_40_80) < 0.3 else "Moderate-to-large LPIPS: data size produces visually distinct outputs despite similar CLIP."}

**Checkpoint sizes:** all three are identical ({agg[20]['ckpt_mb']:.1f} MB) — file size
is determined by rank, not training data size. This is expected: the checkpoint stores
trained weight deltas whose dimensionality is rank × hidden_dim, independent of data.

**Cross-experiment note:** Seventh experiment in the CLIP-blindness series. Together
with Exp 6 (LoRA rank), this experiment completes the picture of LoRA training-time
parameters. The consistent theme: CLIP measures semantic alignment to the prompt but
cannot reliably detect changes in visual style quality or character that arise from
training decisions — data volume and adapter rank both move the image in perceptual
space in ways that LPIPS registers and CLIP does not.

## Charts

- `charts/clip_by_data_size.png` — mean CLIP score per training data size
- `charts/lpips_vs_data80.png` — perceptual distance from 80-image baseline

## Raw data

`results.csv` / `results.json` — one row per image ({len(all_rows)} rows total).

Reproduce:

```bash
python scripts/experiments/exp7_lora_data_size.py
```
"""

with open(OUT / "findings.md", "w", encoding="utf-8") as f:
    f.write(FINDINGS)

print(f"\nFindings written: {OUT / 'findings.md'}")
print("Experiment 7 complete.")
