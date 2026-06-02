"""
Experiment 6 (SDXL): LoRA rank ablation — rank 4 / 8 / 16 on the Ukiyo-e SDXL LoRA.

Trains rank-4 and rank-16 SDXL LoRAs from scratch (1500 steps each, same data/seed as
rank-8 baseline). Rank-8 uses the existing checkpoint (downloaded from HF if absent).
Generates 5 seeds × 6 prompts × 3 ranks = 90 images at 50 steps DPM-Solver++,
then computes CLIP, HPS, ImageReward, and LPIPS.

Training is skipped if checkpoint already exists (idempotent).

Expected runtime: ~3-4h training + ~1h generation+eval on GCP L4.

Run from project root:
    python scripts/experiments/exp6_sdxl.py

Outputs:
    data/lora/ukiyo-e-sdxl-rank4/pytorch_lora_weights.safetensors  (trained here)
    data/lora/ukiyo-e-sdxl-rank16/pytorch_lora_weights.safetensors (trained here)
    reports/experiments/exp6_sdxl/
        images/rank_{r}/
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
RANKS = [4, 8, 16]
RANK_COLORS = {4: ORANGE, 8: BLUE, 16: GREEN}
SEEDS = [42, 123, 456, 789, 1337]
STEPS = 50
CFG = 7.0
SIZE = 1024
TRAIN_STEPS = 1500
TRAIN_SEED = 42

# Rank-8 is downloaded from HF (or already local); 4 and 16 are trained here.
LORA_REPO = "gauravgandhi2411/aetherart-ukiyo-sdxl"
LORA_PATHS = {
    4: ROOT / "data" / "lora" / "ukiyo-e-sdxl-rank4" / "pytorch_lora_weights.safetensors",
    8: ROOT / "data" / "lora" / "ukiyo-e-sdxl" / "ukiyo-e-sdxl-lora.safetensors",
    16: ROOT / "data" / "lora" / "ukiyo-e-sdxl-rank16" / "pytorch_lora_weights.safetensors",
}
LORA_OUTPUT_DIRS = {
    4: ROOT / "data" / "lora" / "ukiyo-e-sdxl-rank4",
    16: ROOT / "data" / "lora" / "ukiyo-e-sdxl-rank16",
}

# 6 prompts: mix of detail-heavy and simple scenes where rank differences show
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

OUT = ROOT / "reports" / "experiments" / "exp6_sdxl"
IMG_DIR = OUT / "images"
CHARTS_DIR = OUT / "charts"

for r in RANKS:
    (IMG_DIR / f"rank_{r}").mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Rank-8 download (if not already local) ────────────────────────────────────


def _ensure_rank8_lora() -> Path:
    """Download the rank-8 SDXL LoRA from HF if it is not already on disk."""
    path = LORA_PATHS[8]
    if path.exists():
        return path
    from huggingface_hub import hf_hub_download

    print(f"[rank 8] Downloading checkpoint from {LORA_REPO}...")
    path.parent.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        LORA_REPO,
        "ukiyo-e-sdxl-lora.safetensors",
        local_dir=str(path.parent),
    )
    return path


# ── Training (skipped if checkpoint exists) ───────────────────────────────────


def train_rank(rank: int) -> None:
    """Train a rank-N SDXL LoRA via train_lora.py --base sdxl.

    Skipped idempotently if the checkpoint already exists.
    """
    ckpt = LORA_PATHS[rank]
    if ckpt.exists():
        print(f"[rank {rank}] checkpoint exists — skipping training: {ckpt}")
        return

    out_dir = LORA_OUTPUT_DIRS[rank]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[rank {rank}] Starting SDXL training → {out_dir}")
    print(f"[rank {rank}] Expected time: ~1.5-2h on GCP L4")

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train_lora.py"),
        "--base", "sdxl",
        "--rank", str(rank),
        "--max-train-steps", str(TRAIN_STEPS),
        "--seed", str(TRAIN_SEED),
        "--output-dir", str(out_dir),
    ]

    t0 = time.monotonic()
    proc = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.monotonic() - t0
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)

    if proc.returncode != 0:
        print(
            f"[rank {rank}] TRAINING FAILED (exit {proc.returncode}) after {h:02d}:{m:02d}:{s:02d}"
        )
        sys.exit(proc.returncode)
    print(f"[rank {rank}] Training complete in {h:02d}:{m:02d}:{s:02d}")


# ── Pipeline ──────────────────────────────────────────────────────────────────


def load_pipeline_with_lora(rank: int) -> StableDiffusionXLPipeline:
    """Load SDXL base pipeline with fp16-fix VAE and attach the given rank's LoRA."""
    from diffusers import AutoencoderKL

    from aetherart.config import cfg

    ckpt = LORA_PATHS[rank]
    if not ckpt.exists():
        raise FileNotFoundError(f"LoRA checkpoint missing for rank {rank}: {ckpt}")

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
    print(f"[rank {rank}] SDXL Pipeline + LoRA ready: {ckpt.name}")
    return pipe


# ── Generation loop ───────────────────────────────────────────────────────────


def run_rank(rank: int) -> list[dict]:
    """Generate all prompt × seed images for a single LoRA rank."""
    pipe = load_pipeline_with_lora(rank)
    rows: list[dict] = []
    img_dir = IMG_DIR / f"rank_{rank}"

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
                    "rank": rank,
                    "prompt_id": prompt_id,
                    "prompt_text": prompt_text,
                    "seed": seed,
                    "latency_s": round(latency, 3),
                    "clip_score": None,
                    "hps_score": None,
                    "ir_score": None,
                    "lpips_vs_rank8": None,
                    "lpips_vs_prev_rank": None,
                    "image_path": (img_dir / fname).relative_to(ROOT).as_posix(),
                }
            )
            print(f"  [rank {rank}] {prompt_id} seed={seed:5d} | {latency:.1f}s")

    del pipe
    cleanup_gpu(verbose=False)
    return rows


# ── Phase 1: Ensure rank-8 checkpoint ────────────────────────────────────────

print("\n=== Phase 1: Checkpoint setup ===")
_ensure_rank8_lora()

# ── Phase 2: Training ─────────────────────────────────────────────────────────

print("\n=== Phase 2: Training (ranks 4 and 16) ===")
for rank in [4, 16]:
    train_rank(rank)

# Verify all checkpoints exist before proceeding
for rank in RANKS:
    if not LORA_PATHS[rank].exists():
        print(f"ERROR: checkpoint missing for rank {rank}: {LORA_PATHS[rank]}")
        sys.exit(1)
    size_mb = os.path.getsize(LORA_PATHS[rank]) / 1e6
    print(f"[rank {rank}] checkpoint: {LORA_PATHS[rank].name}  ({size_mb:.1f} MB)")

# ── Phase 3: Generation ────────────────────────────────────────────────────────

print("\n=== Phase 3: Generation (90 images) ===")
all_rows: list[dict] = []
for rank in RANKS:
    print(f"\n--- Rank {rank} ---")
    all_rows.extend(run_rank(rank))

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

print("\nComputing LPIPS...")
_lpips_fn = lpips_lib.LPIPS(net="alex")
_lpips_fn.eval()

img_index: dict[tuple, str] = {
    (r["rank"], r["prompt_id"], r["seed"]): r["image_path"] for r in all_rows
}


def _to_t(img: Image.Image) -> torch.Tensor:
    arr = np.array(img.convert("RGB")).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def _lpips_pair(path_a: str, path_b: str) -> float:
    a = Image.open(ROOT / path_a).convert("RGB")
    b = Image.open(ROOT / path_b).convert("RGB")
    with torch.no_grad():
        return round(float(_lpips_fn(_to_t(a), _to_t(b))), 6)


rank8_idx = {(r["prompt_id"], r["seed"]): r["image_path"] for r in all_rows if r["rank"] == 8}

for r in all_rows:
    # LPIPS vs rank-8 (baseline)
    r8_path = rank8_idx[(r["prompt_id"], r["seed"])]
    r["lpips_vs_rank8"] = 0.0 if r["rank"] == 8 else _lpips_pair(r["image_path"], r8_path)

    # LPIPS vs adjacent rank (4→8, 8→16)
    rank_idx = RANKS.index(r["rank"])
    if rank_idx == 0:
        r["lpips_vs_prev_rank"] = None
    else:
        prev_rank = RANKS[rank_idx - 1]
        prev_path = img_index[(prev_rank, r["prompt_id"], r["seed"])]
        r["lpips_vs_prev_rank"] = _lpips_pair(r["image_path"], prev_path)

print("LPIPS done.")


# ── Save results ──────────────────────────────────────────────────────────────

CSV_PATH = OUT / "results.csv"
JSON_PATH = OUT / "results.json"

csv_fields = [
    "rank",
    "prompt_id",
    "seed",
    "latency_s",
    "clip_score",
    "hps_score",
    "ir_score",
    "lpips_vs_rank8",
    "lpips_vs_prev_rank",
    "image_path",
]
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(all_rows)

ckpt_sizes = {r: round(os.path.getsize(LORA_PATHS[r]) / 1e6, 2) for r in RANKS}

with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(
        {
            "experiment": "exp6_sdxl",
            "date": "2026-06-02",
            "model": MODEL_ID,
            "size": 1024,
            "scorers": ["clip", "hps", "imagereward", "lpips"],
            "ranks": RANKS,
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


# ── Per-rank aggregates ───────────────────────────────────────────────────────

by_rank: dict[int, list[dict]] = {r: [] for r in RANKS}
for row in all_rows:
    by_rank[row["rank"]].append(row)

agg: dict[int, dict] = {}
for rank, rows in by_rank.items():
    clips = [r["clip_score"] for r in rows]
    hps = [r["hps_score"] for r in rows]
    irs = [r["ir_score"] for r in rows]
    lats = [r["latency_s"] for r in rows]
    lpips_r8 = [r["lpips_vs_rank8"] for r in rows if r["lpips_vs_rank8"] is not None]
    lpips_prev = [r["lpips_vs_prev_rank"] for r in rows if r["lpips_vs_prev_rank"] is not None]
    agg[rank] = {
        "mean_clip": statistics.mean(clips),
        "se_clip": statistics.stdev(clips) / len(clips) ** 0.5,
        "mean_hps": statistics.mean(hps),
        "se_hps": statistics.stdev(hps) / len(hps) ** 0.5,
        "mean_ir": statistics.mean(irs),
        "se_ir": statistics.stdev(irs) / len(irs) ** 0.5,
        "mean_lat": statistics.mean(lats),
        "mean_lpips_r8": statistics.mean(lpips_r8) if lpips_r8 else 0.0,
        "mean_lpips_prev": statistics.mean(lpips_prev) if lpips_prev else None,
        "ckpt_mb": ckpt_sizes[rank],
    }

print("\n── Scores by LoRA rank ──")
for rank in RANKS:
    aa = agg[rank]
    prev_str = f"{aa['mean_lpips_prev']:.4f}" if aa["mean_lpips_prev"] is not None else "—"
    print(
        f"  rank={rank:2d}: CLIP={aa['mean_clip']:.4f} ±{aa['se_clip']:.4f} | "
        f"HPS={aa['mean_hps']:.4f} | IR={aa['mean_ir']:.4f} | "
        f"LPIPS_vs_8={aa['mean_lpips_r8']:.4f} | LPIPS_prev={prev_str} | "
        f"{aa['ckpt_mb']:.1f} MB"
    )


# ── Charts ────────────────────────────────────────────────────────────────────

x = np.arange(len(RANKS), dtype=float)
xlabels = [f"rank-{r}" for r in RANKS]
colors = [RANK_COLORS[r] for r in RANKS]
clip_arr = np.array([agg[r]["mean_clip"] for r in RANKS])
hps_arr = np.array([agg[r]["mean_hps"] for r in RANKS])
ir_arr = np.array([agg[r]["mean_ir"] for r in RANKS])
lpips_arr = np.array([agg[r]["mean_lpips_r8"] for r in RANKS])

clip_max = float(clip_arr.max())
canvas = ChartCanvas(
    figsize=(7, 4.5),
    title="CLIP score by LoRA rank (ukiyo-e, SDXL) — 6 prompts × 5 seeds × 50 steps",
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
canvas.save(str(CHARTS_DIR / "clip_by_rank.png"))

hps_max = float(hps_arr.max())
canvas_hps = ChartCanvas(
    figsize=(7, 4.5),
    title="HPS score by LoRA rank (ukiyo-e, SDXL)",
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
canvas_hps.save(str(CHARTS_DIR / "hps_by_rank.png"))

ir_max = float(ir_arr.max())
canvas_ir = ChartCanvas(
    figsize=(7, 4.5),
    title="ImageReward score by LoRA rank (ukiyo-e, SDXL)",
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
canvas_ir.save(str(CHARTS_DIR / "ir_by_rank.png"))

lpips_max = float(lpips_arr.max())
canvas4 = ChartCanvas(
    figsize=(7, 4.5),
    title="Perceptual distance from rank-8 baseline (LPIPS) — SDXL",
    ylabel="Mean LPIPS vs rank-8",
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
canvas4.save(str(CHARTS_DIR / "lpips_vs_rank8.png"))

print(f"Charts written to {CHARTS_DIR}")


# ── Findings writeup ──────────────────────────────────────────────────────────

clip_delta_4_to_8 = agg[8]["mean_clip"] - agg[4]["mean_clip"]
clip_delta_8_to_16 = agg[16]["mean_clip"] - agg[8]["mean_clip"]
hps_delta_4_to_8 = agg[8]["mean_hps"] - agg[4]["mean_hps"]
hps_delta_8_to_16 = agg[16]["mean_hps"] - agg[8]["mean_hps"]
ir_delta_4_to_8 = agg[8]["mean_ir"] - agg[4]["mean_ir"]
ir_delta_8_to_16 = agg[16]["mean_ir"] - agg[8]["mean_ir"]
pooled_se = max(agg[r]["se_clip"] for r in RANKS)
lpips_4_8 = agg[8]["mean_lpips_prev"] if agg[8]["mean_lpips_prev"] is not None else 0.0
lpips_8_16 = agg[16]["mean_lpips_prev"] if agg[16]["mean_lpips_prev"] is not None else 0.0
clip_delta_ses = abs(clip_delta_4_to_8) / pooled_se if pooled_se > 0 else 0.0
clip_blind = abs(clip_delta_4_to_8) < pooled_se
lpips_range = max(lpips_4_8, lpips_8_16)

table_rows = []
for rank in RANKS:
    aa = agg[rank]
    prev_str = f"{aa['mean_lpips_prev']:.4f}" if aa["mean_lpips_prev"] is not None else "—"
    baseline_marker = " ← baseline" if rank == 8 else ""
    table_rows.append(
        f"| {rank:2d}     | {aa['mean_clip']:.4f} | ±{aa['se_clip']:.4f}"
        f" | {aa['mean_hps']:.4f} | {aa['mean_ir']:.4f}"
        f" | {aa['mean_lpips_r8']:.4f}       | {prev_str}{baseline_marker}"
        f" | {aa['ckpt_mb']:.1f} MB |"
    )

FINDINGS = f"""\
# Experiment 6 (SDXL): LoRA Rank Ablation

**Date:** 2026-06-02
**Hardware:** GCP L4 (enable_model_cpu_offload)
**Model:** {MODEL_ID}
**LoRA:** ukiyo-e — Japanese woodblock print style — 80 WikiArt images, SDXL
**Ranks tested:** {RANKS}
**Training:** {TRAIN_STEPS} steps, seed {TRAIN_SEED}, same data for all ranks
**Rank-8 checkpoint:** downloaded from {LORA_REPO} (or local) — not retrained
**Design:** 5 seeds × 6 prompts = 30 images per rank · {len(all_rows)} images total
**Scheduler:** DPM-Solver++ · {STEPS} steps · {SIZE}×{SIZE}
**CFG:** {CFG} (fixed)
**LPIPS reference:** rank-8 (trained baseline); also computed between adjacent ranks

## Hypothesis

Higher rank captures more style detail: rank-4 underfits (lower quality on fine detail),
rank-16 overfits or matches rank-8 closely. File size scales linearly with rank.
CLIP may or may not register the quality difference — if it doesn't, CLIP-blindness
confirmed again. HPS and ImageReward may be more sensitive to style quality.

## Results

| Rank  | Mean CLIP | SE      | HPS    | IR     | LPIPS vs rank-8    | LPIPS vs prev rank   | File size |
|-------|----------:|--------:|-------:|-------:|-------------------:|---------------------:|----------:|
{chr(10).join(table_rows)}

## Interpretation

**CLIP (rank-4 vs rank-8):** Delta = {clip_delta_4_to_8:+.4f} ({abs(clip_delta_4_to_8) / pooled_se:.1f} pooled SEs).
{"Within noise — rank-4 matches rank-8 semantically by CLIP." if abs(clip_delta_4_to_8) < pooled_se else "Detectable difference — rank affects semantic alignment as measured by CLIP."}

**CLIP (rank-8 vs rank-16):** Delta = {clip_delta_8_to_16:+.4f} ({abs(clip_delta_8_to_16) / pooled_se:.1f} pooled SEs).
{"Within noise — rank-16 does not improve CLIP over rank-8." if abs(clip_delta_8_to_16) < pooled_se else "Detectable difference — rank-16 changes semantic alignment vs rank-8."}

**HPS (rank-4 vs rank-8):** Delta = {hps_delta_4_to_8:+.4f}. (rank-8 vs rank-16): Delta = {hps_delta_8_to_16:+.4f}.
**IR (rank-4 vs rank-8):** Delta = {ir_delta_4_to_8:+.4f}. (rank-8 vs rank-16): Delta = {ir_delta_8_to_16:+.4f}.

**LPIPS (adjacent ranks):** rank-4 vs rank-8: {lpips_4_8:.4f}; rank-8 vs rank-16: {lpips_8_16:.4f}.
{"Small LPIPS values: rank differences are perceptually minor at the pixel level." if max(lpips_4_8, lpips_8_16) < 0.3 else "Moderate-to-large LPIPS: ranks produce visually distinct outputs despite similar CLIP."}

**File size:** rank-4 = {agg[4]["ckpt_mb"]:.1f} MB, rank-8 = {agg[8]["ckpt_mb"]:.1f} MB, rank-16 = {agg[16]["ckpt_mb"]:.1f} MB.
Scales approximately linearly with rank (attention layer matrices scale as rank × hidden_dim).

CLIP delta = {clip_delta_4_to_8:+.4f} ({clip_delta_ses:.1f} SEs). HPS delta = {hps_delta_4_to_8:+.4f}. IR delta = {ir_delta_4_to_8:+.4f}. LPIPS range = {lpips_range:.4f}. Verdict: CLIP-BLIND: {"yes" if clip_blind else "no"}.

## Charts

- `charts/clip_by_rank.png` — mean CLIP score per rank
- `charts/hps_by_rank.png` — mean HPS score per rank
- `charts/ir_by_rank.png` — mean ImageReward score per rank
- `charts/lpips_vs_rank8.png` — perceptual distance from rank-8 baseline

## Raw data

`results.csv` / `results.json` — one row per image ({len(all_rows)} rows total).

Reproduce:

```bash
python scripts/experiments/exp6_sdxl.py
```
"""

with open(OUT / "findings.md", "w", encoding="utf-8") as f:
    f.write(FINDINGS)

print(f"\nFindings written: {OUT / 'findings.md'}")
print("Experiment 6 SDXL complete.")
