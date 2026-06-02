# Experiment 5 (SDXL): ControlNet Conditioning Strength Sweep

**Date:** 2026-06-02
**Hardware:** GCP L4 (enable_model_cpu_offload)
**Model:** stabilityai/stable-diffusion-xl-base-1.0
**ControlNet:** xinsir/controlnet-canny-sdxl-1.0 (Canny edges, SDXL)
**Conditioning source:** fresh SDXL reference images (seed=42, plain SDXL base)
**Strength values:** [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
  (0.0 = no conditioning / text-only; 1.0 = standard reference; 1.5 = over-conditioned)
**Reference strength for LPIPS:** 1.0
**Design:** 5 seeds × 8 prompts = 40 images per strength · 280 images total
**Scheduler:** DPM-Solver++ · 30 steps · 1024×1024
**CFG:** 7.5
**Negative prompt:** standard (held constant)

## Hypothesis

CLIP will stay roughly flat across all strength values — the same text prompt drives
the same semantic content regardless of how tightly the canny edges constrain the output.
HPS and ImageReward may be more sensitive to the pixel-level character change.
LPIPS vs the strength=1.0 reference will reveal where the image perceptually departs from
standard conditioning, diverging in both directions from the anchor.

## Results

| Strength | Mean CLIP | SE      | HPS    | IR     | LPIPS vs ref=1.0 | LPIPS vs prev |
|----------|----------:|--------:|-------:|-------:|----------------------------:|--------------:|
| 0.00  | 0.3110 | ±0.0071 | 0.2833 | 0.1127 | 0.7035     | — |
| 0.25  | 0.3181 | ±0.0066 | 0.2906 | 0.0847 | 0.4898     | 0.6310 |
| 0.50  | 0.3175 | ±0.0056 | 0.2835 | 0.1250 | 0.2699     | 0.3828 |
| 0.75  | 0.3134 | ±0.0053 | 0.2800 | 0.0523 | 0.1172     | 0.1908 |
| 1.00  | 0.3131 | ±0.0054 | 0.2791 | -0.0099 | 0.0000     | 0.1172 ← ref |
| 1.25  | 0.3103 | ±0.0052 | 0.2778 | -0.0825 | 0.0852     | 0.0852 |
| 1.50  | 0.3086 | ±0.0048 | 0.2757 | -0.1108 | 0.1455     | 0.0591 |

LPIPS at strength=0.0 (text-only) vs reference: 0.7035
LPIPS at strength=1.5 (over-conditioned) vs reference: 0.1455

## Interpretation

**CLIP:** Flat across all strengths — first value within 1 SE of max CLIP is strength=0.25 (max CLIP=0.3181, SE≈0.0071). Conditioning strength has no measurable effect on semantic alignment as judged by CLIP.

**HPS / IR:** HPS delta (0.0→1.5) = -0.0076. IR delta = -0.2235.
HPS and/or IR show a signal that CLIP misses — these scorers capture conditioning-strength character.

**LPIPS vs reference:** Divergence is measurable in both directions from strength=1.0.
At strength=0.0 (no conditioning, effectively text-only generation): LPIPS=0.7035.
At strength=1.5 (over-conditioned): LPIPS=0.1455.
The images look very different despite similar scorer values — LPIPS is the honest metric here.

**LPIPS (adjacent steps):** The largest single-step visual change is the
0.0→0.25 transition (LPIPS=0.6310). ControlNet strength
is a pixel-level creative parameter — LPIPS registers it, CLIP may not.

CLIP delta = -0.0023 (0.3 SEs). HPS delta = -0.0076. IR delta = -0.2235. LPIPS range = 0.1455. Verdict: CLIP-BLIND: yes.

## Charts

- `charts/clip_by_strength.png` — mean CLIP score per conditioning strength
- `charts/hps_by_strength.png` — mean HPS score per conditioning strength
- `charts/ir_by_strength.png` — mean ImageReward score per conditioning strength
- `charts/lpips_vs_ref.png` — perceptual distance from strength=1.0 reference
- `charts/lpips_adjacent.png` — step-wise LPIPS between adjacent strength values

## Raw data

`results.csv` / `results.json` — one row per image (280 rows total).

Reproduce:

```bash
python scripts/experiments/exp5_sdxl.py
```
