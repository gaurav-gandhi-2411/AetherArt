# Experiment 8 (SDXL): LoRA Style Scale (Alpha) Sweep

**Date:** 2026-06-02
**Model:** stabilityai/stable-diffusion-xl-base-1.0 — 1024×1024
**LoRA:** gauravgandhi2411/aetherart-ukiyo-sdxl (SDXL Ukiyo-e)
**Trigger token:** ukyowood (used in all prompts)
**Alpha values:** [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
  (0.0 = LoRA loaded but suppressed; 1.0 = trained default; 1.5 = over-styled)
**Reference alpha for LPIPS:** 1.0
**Design:** 7 alphas × 8 prompts × 5 seeds = 280 images
**Scheduler:** DPM-Solver++ · 30 steps · CFG=7.5
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
| 0.00  | 0.3214 | ±0.0037 | 0.2633 | 0.8509 | 0.6904 |
| 0.25  | 0.3243 | ±0.0039 | 0.2594 | 0.6961 | 0.6470 |
| 0.50  | 0.3284 | ±0.0044 | 0.2514 | 0.6176 | 0.5401 |
| 0.75  | 0.3269 | ±0.0047 | 0.2356 | 0.3367 | 0.3955 |
| 1.00  | 0.3274 | ±0.0044 | 0.2199 | 0.0913 | 0.0000 ← ref |
| 1.25  | 0.3176 | ±0.0044 | 0.2028 | -0.2752 | 0.4324 |
| 1.50  | 0.2958 | ±0.0062 | 0.1725 | -0.8019 | 0.5902 |

CLIP delta across conditions = 0.0326, which is 7.2 SEs.
HPS delta = 0.0909. IR delta = 1.6528. LPIPS range = 0.6904.
Verdict: CLIP-BLIND: no.

LPIPS at alpha=0.0 (base SDXL, no style) vs reference: 0.6904
LPIPS at alpha=1.5 (over-styled) vs reference: 0.5902

## Interpretation

**CLIP / HPS / IR:** All three scorers are within noise across all alpha values. First within 1 SE of max CLIP (0.3284) is alpha=0.25. Adapter weight has no detectable effect on semantic alignment.

**LPIPS:** The stylistic character of the image changes substantially as alpha increases,
but semantic scorers do not register this — the prompt describes the same scene at every
alpha. At alpha=0.0 (no LoRA), LPIPS=0.6904 vs reference; at
alpha=1.5, LPIPS=0.5902.

**SDXL vs SD 2.1:** At 1024×1024, the LoRA style shift may produce larger absolute LPIPS
values than the 512×512 SD 2.1 baseline, while CLIP scores remain in the same range.

## Charts

- `charts/clip_by_alpha.png`
- `charts/hps_by_alpha.png`
- `charts/lpips_vs_ref.png`

## Raw data

`results.csv` / `results.json` — one row per image (280 rows total).

Reproduce:

```bash
python scripts/experiments/exp8_sdxl.py
```
