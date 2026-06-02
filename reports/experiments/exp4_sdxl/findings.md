# Experiment 4 (SDXL): Scheduler Visual Comparison

**Date:** 2026-06-02
**Model:** stabilityai/stable-diffusion-xl-base-1.0 — 1024×1024
**Hardware:** (run on your GPU)
**Schedulers:** DDIM, DPM, EulerA, LMS
**Step count:** 30 · CFG=7.5 · Seed=42 (single seed per image)
**Prompts:** 8 Ukiyo-e prompts (p01_portrait through p08_crowd)
**Images generated:** 32 (4 schedulers × 8 prompts)
**Pairs compared:** 6 (DDIM-DPM, DDIM-EulerA, DDIM-LMS, DPM-EulerA, DPM-LMS, EulerA-LMS)
**Scorers:** CLIP, HPS, ImageReward, LPIPS

## Hypothesis

Schedulers are semantically interchangeable — CLIP and HPS should be flat across conditions.
LPIPS will capture whether "indistinguishable by CLIP" also means "perceptually equivalent."

## Results — per-scheduler scores

| Scheduler | Mean CLIP | SE      | Mean HPS | Mean IR |
|-----------|----------:|--------:|---------:|--------:|
| DDIM    | 0.3175 | ±0.0090 | 0.2586 | 0.4809 |
| DPM     | 0.3116 | ±0.0102 | 0.2586 | 0.5518 |
| EulerA  | 0.3112 | ±0.0092 | 0.2570 | 0.5973 |
| LMS     | 0.3156 | ±0.0095 | 0.2595 | 0.5371 |

CLIP delta across conditions = 0.0063, which is 0.7 SEs.
HPS delta = 0.0025. IR delta = 0.1164. LPIPS range = 0.4518.
Verdict: CLIP-BLIND: yes.

## Results — LPIPS by scheduler pair

| Pair           | Mean LPIPS | SE      |
|----------------|----------:|--------:|
| DDIM-DPM       | 0.2731 | ±0.0417 |
| DDIM-EulerA    | 0.6759 | ±0.0380 |
| DDIM-LMS       | 0.2801 | ±0.0316 |
| DPM-EulerA     | 0.6769 | ±0.0415 |
| DPM-LMS        | 0.2268 | ±0.0424 |
| EulerA-LMS     | 0.6787 | ±0.0415 |

Most perceptually different pair: EulerA-LMS (LPIPS = 0.6787)
Most similar pair:                DPM-LMS (LPIPS = 0.2268)
Mean LPIPS across all pairs: 0.4686

## Interpretation

**CLIP / HPS / IR:** The CLIP range of 0.0063 (0.7× pooled SE) is
statistically flat — schedulers are indistinguishable by any of the three semantic scorers.

**LPIPS:** Despite identical CLIP scores, schedulers produce perceptually distinct images.
The mean LPIPS across all pairs is 0.4686; the widest pair (EulerA-LMS)
reaches LPIPS=0.6787. Even the closest pair
(DPM-LMS, LPIPS=0.2268) shows non-trivial
pixel-level differences.

**SDXL vs SD 2.1:** The SDXL size (1024×1024) amplifies per-pixel differences vs the SD 2.1
512×512 benchmark, so LPIPS values may be larger in absolute terms while CLIP remains blind
by the same mechanism.

## Charts

- `charts/clip_by_scheduler.png`
- `charts/hps_by_scheduler.png`
- `charts/lpips_by_pair.png`
- `charts/clip_vs_lpips_range.png`

## Raw data

`results.csv` — one row per image (32 rows)
`results_pairs.csv` — one row per pair×prompt (48 rows)
`results.json` — aggregates + full data

Reproduce:

```bash
python scripts/experiments/exp4_sdxl.py
```
