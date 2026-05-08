# Experiment 1: Quantization Quality Comparison

**Date:** 2026-05-07
**Hardware:** RTX 3070 Laptop 8 GB
**Model:** sd2-community/stable-diffusion-2-1
**Conditions:** fp16 (baseline) · INT8 (8-bit bitsandbytes U-Net) · NF4 (4-bit bitsandbytes U-Net)
**Design:** 5 seeds x 8 prompts = 40 images per condition · 120 images total
**Scheduler:** DPM-Solver++ · 30 steps · 512x512 · CFG=7.5

## Hypothesis

Quantizing the U-Net to INT8 or NF4 will degrade output quality measurably, but not
catastrophically. Perceptual degradation (LPIPS) should be detectable before CLIP-score
differences rise above statistical noise.

## Results

| Condition | Mean CLIP | CLIP delta vs fp16 | Mean LPIPS | Latency (s) | Peak VRAM (MB) |
|-----------|----------:|-------------------:|-----------:|------------:|---------------:|
| fp16      | 0.3124    | —                  | 0.0000     | 4.4s       | 1803            |
| INT8      | 0.3112    | -0.0012             | 0.1729     | 12.3s       | 2210            |
| NF4       | 0.3158    | +0.0035             | 0.3966     | 6.4s       | 1382            |

SE on CLIP: fp16 ±0.0037 · INT8 ±0.0038 · NF4 ±0.0034

All three CLIP means are within 1 SE of fp16. The apparent ordering (INT8 < fp16 < NF4) is
entirely within measurement noise; NF4's numerically highest CLIP score is not a real signal.

## Interpretation

**INT8 quality:** CLIP score is within 1 SE of fp16 (delta = −0.0012) — statistically indistinguishable.
Perceptual fidelity to fp16: moderate perceptual differences (LPIPS = 0.1729, in the 0.10–0.20 range).
Latency cost: 12.3s vs 4.4s fp16 (2.8× slower).
VRAM: 2210 MB vs 1803 MB fp16 — INT8 uses 407 MB *more* VRAM under the CPU-offload inference
path (bitsandbytes allocates a fp16 compute buffer for dequantization during the forward pass).

**NF4 quality:** CLIP score is within 1 SE of fp16 (delta = +0.0035) — statistically indistinguishable.
Perceptual fidelity to fp16: substantial perceptual degradation (LPIPS = 0.3966, > 0.20 threshold).
Latency cost: 6.4s vs 4.4s fp16 (1.5× slower).
VRAM: 1382 MB vs 1803 MB fp16 — NF4 saves 421 MB (23% reduction); this survives the CPU-offload
path because the stored 4-bit weight footprint is smaller even after accounting for compute buffers.

**Bottom line:** All three modes are statistically indistinguishable by CLIP score, but LPIPS
tells a different story. NF4 causes substantial pixel-level degradation (LPIPS = 0.40) that
CLIP cannot detect — the images have the same semantic content but measurably different pixels.
INT8 is moderate (LPIPS = 0.17). These conclusions are hardware-specific: all measurements used
`enable_model_cpu_offload()` on an RTX 3070 8 GB, which changes the VRAM and latency profile
vs running fully on-device. On a card with enough VRAM to load the full model (≥12 GB fp16,
≥6 GB INT8), INT8 would recover its stored-weight savings without the compute-buffer inflation
seen here, and may show better latency. On this hardware path — CPU offload on 8 GB — INT8
uses more VRAM than fp16 and runs 2.8× slower, leaving no compelling reason to prefer it.
NF4 saves 23% VRAM and runs 1.5× slower, but the pixel-fidelity loss is substantial.

**Cross-experiment note:** Experiment 2 (negative prompt ablation) finds the same pattern:
LPIPS = 0.46 between conditions while CLIP delta is within 1 SE. CLIP-blindness to pixel-level
change appears to be a consistent property of the metric, not an artifact of this experiment.

## Charts

- `charts/clip_by_condition.png` — mean CLIP score per condition
- `charts/lpips_vs_fp16.png` — LPIPS perceptual distance from fp16 (INT8 and NF4 only)
- `charts/latency_by_condition.png` — mean generation latency per condition

## Raw data

`results.csv` / `results.json` — one row per image (120 rows total).

Reproduce:

```bash
python scripts/experiments/exp1_quantization_quality.py
```

---

**Cross-experiment context:** [`reports/clip_blindness.md`](../../clip_blindness.md) — first confirmation of the pattern; see the full 9-experiment evidence table and the "Why CLIP fails" explanation.
