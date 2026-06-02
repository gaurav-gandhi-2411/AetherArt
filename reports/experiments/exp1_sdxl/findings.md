# Experiment 1 (SDXL): Quantization Quality Comparison

**Date:** 2026-06-02
**Hardware:** GCP L4 (24 GB VRAM)
**Model:** stabilityai/stable-diffusion-xl-base-1.0
**Conditions:** fp16 (baseline) · INT8 (8-bit bitsandbytes U-Net) · NF4 (4-bit bitsandbytes U-Net)
**Design:** 5 seeds x 8 prompts = 40 images per condition · 120 images total
**Scheduler:** DPM-Solver++ · 30 steps · 1024x1024 · CFG=7.5
**Scorers:** HPS (primary), ImageReward (primary), CLIP (comparison-only), LPIPS

## Hypothesis

Quantizing the SDXL U-Net to INT8 or NF4 will degrade output quality measurably, but not
catastrophically. Perceptual degradation (LPIPS) should be detectable before CLIP-score
differences rise above statistical noise. HPS and ImageReward provide human-preference-aligned
signal where CLIP may be blind.

## Results

| Condition | Mean HPS | HPS delta | Mean IR | IR delta | Mean CLIP | CLIP delta | Mean LPIPS | Latency (s) | Peak VRAM (MB) |
|-----------|:--------:|----------:|:-------:|:--------:|----------:|-----------:|-----------:|------------:|---------------:|
| fp16      | 0.2820   | —         | 0.6274  | —        | 0.3284    | —          | 0.0000     | 16.6s       | 5335            |
| INT8      | 0.2815   | -0.0005    | 0.6197  | -0.0077   | 0.3275    | -0.0009     | 0.1124     | 26.7s       | 5498            |
| NF4       | 0.2779   | -0.0041    | 0.6174  | -0.0099   | 0.3279    | -0.0005     | 0.3158     | 15.2s       | 2611            |

SE: fp16 CLIP ±0.0035 · INT8 CLIP ±0.0037 · NF4 CLIP ±0.0036
SE: fp16 HPS ±0.0060 · INT8 HPS ±0.0061 · NF4 HPS ±0.0060
SE: fp16 IR ±0.1019 · INT8 IR ±0.1057 · NF4 IR ±0.1082

## CLIP-blindness verdict

CLIP delta across conditions = 0.0009, which is 0.25 SEs. HPS delta = 0.0041. IR delta = 0.0099. LPIPS range = 0.3158. Verdict: CLIP-BLIND: yes.

## Interpretation

**INT8 quality:** CLIP score is within 2 SE of fp16 (delta = -0.0009) — statistically indistinguishable.
Perceptual fidelity to fp16: moderate perceptual differences from fp16 (LPIPS = 0.1124, 0.10–0.20).
Latency cost: 26.7s vs 16.6s fp16 (1.6x slower).
VRAM saved: -163 MB (-3% reduction vs fp16).

**NF4 quality:** CLIP score is within 2 SE of fp16 (delta = -0.0005) — statistically indistinguishable.
Perceptual fidelity to fp16: substantial perceptual degradation vs fp16 (LPIPS = 0.3158, > 0.20).
Latency cost: 15.2s vs 16.6s fp16 (0.9x slower).
VRAM saved: 2724 MB (51% reduction vs fp16).

**Bottom line:** Both quantization modes preserve semantic alignment but carry a latency
penalty from dequantization overhead. LPIPS and HPS/IR provide converging evidence where
CLIP may be blind to perceptual changes. If LPIPS < 0.05 and HPS/IR deltas are small for
both modes, quantization is essentially transparent for these prompt types at 1024×1024.

## Charts

- `charts/hps_by_condition.png` — mean HPS score per condition (primary quality metric)
- `charts/ir_by_condition.png` — mean ImageReward score per condition
- `charts/clip_by_condition.png` — mean CLIP score per condition (comparison-only)
- `charts/lpips_vs_fp16.png` — LPIPS perceptual distance from fp16 (INT8 and NF4 only)
- `charts/latency_by_condition.png` — mean generation latency per condition

## Raw data

`results.csv` / `results.json` — one row per image (120 rows total).

Reproduce:

```bash
python scripts/experiments/exp1_sdxl.py
```
