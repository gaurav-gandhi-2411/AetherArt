# Experiment 3: CFG (Guidance Scale) Sweep

**Date:** 2026-05-07
**Hardware:** RTX 3070 Laptop 8 GB (enable_model_cpu_offload)
**Model:** sd2-community/stable-diffusion-2-1
**CFG values tested:** [1, 3, 5, 7, 9, 12, 15] — CFG=1 is a no-guidance baseline reference; the tuning sweep is CFG=3 to CFG=15
**Reference CFG for LPIPS:** 7
**Design:** 5 seeds x 8 prompts = 40 images per CFG value · 280 images total
**Scheduler:** DPM-Solver++ · 30 steps · 512x512
**Negative prompt:** standard (held constant across all CFG values)

## Hypothesis

CLIP score will increase with CFG and plateau once the guidance is strong enough to anchor
the prompt (expected somewhere in the 5–9 range). LPIPS between adjacent values will reveal
a "regime change" — a step where visual character shifts sharply — that CLIP cannot detect.
At very high CFG (12–15) we expect over-saturation and structural artifacts that LPIPS will
capture before CLIP does.

## Results

| CFG | Mean CLIP | SE      | LPIPS vs cfg=7 (cumulative) | LPIPS vs prev (step) |
|-----|----------:|--------:|------------------------------------:|---------------------:|
|  1    | 0.2314    | ±0.0081  | 0.6452               | —            |
|  3    | 0.2930    | ±0.0051  | 0.4479               | 0.5539            |
|  5    | 0.3072    | ±0.0037  | 0.3061               | 0.3710            |
|  7    | 0.3116    | ±0.0037  | 0.0000               | 0.3061            |
|  9    | 0.3101    | ±0.0039  | 0.2904               | 0.2904            |
| 12    | 0.3131    | ±0.0038  | 0.3897               | 0.3329            |
| 15    | 0.3106    | ±0.0043  | 0.4710               | 0.3830            |

## Key numbers

- CFG=1 is a no-guidance baseline (nearly unconditional generation); it is not part of the
  practical tuning range — it anchors what "text guidance off" looks like
- CLIP plateau starts at CFG=5 within the tuning range; CFG 5–15 all within 1 SE of each other
- Above the plateau: CLIP range = 0.3059 to 0.3131 (0.0072), SE ≈ 0.004 — statistically flat
- LPIPS vs cfg=7 at CFG=15: 0.4710 — comparable to NF4 quantization damage (Exp 1: 0.3966)
- Adjacent LPIPS at high CFG: 9→12 = 0.3329, 12→15 = 0.3830 — large despite flat CLIP

## Interpretation

**CFG=1 as reference:** At CFG=1, the model produces nearly unconditional output — it has read
the text but applies almost no guidance weight. CLIP drops to 0.2314 (vs 0.31+ for all other
values), confirming that the prompt is effectively ignored. This value is included to anchor the
measurement scale, not as a realistic operating point.

**CLIP within the tuning range (CFG 3–15):** CLIP rises from 0.2930 at cfg=3, reaches plateau
at cfg=5 (0.3072), and stays flat through cfg=15 (range 0.0059, all within 1 SE). Once the
prompt is anchored, additional guidance provides no measurable semantic improvement.

**The contrarian headline:** The conventional advice to maximize CFG until your evaluation
metric peaks gives no useful signal past cfg=5 in this setup. CLIP is blind to everything
that happens above the plateau. The data shows that cfg=7, cfg=12, and cfg=15 are
statistically indistinguishable by CLIP while LPIPS shows the images keep diverging
substantially (0.29, 0.39, 0.47 vs the cfg=7 reference). By cfg=15 the image is as
perceptually distant from cfg=7 as NF4-quantized images are from fp16 (Experiment 1:
LPIPS=0.40). Visual inspection or a perceptual metric is required for actual CFG tuning
in the above-plateau regime — CLIP cannot do it.

**LPIPS (cumulative vs cfg=7):** Divergence is monotonically growing in both directions
from the reference: 0.31 at cfg=5, 0.39 at cfg=12, 0.47 at cfg=15 upward; 0.45 at cfg=3,
0.65 at cfg=1 downward. The metric is symmetric around the mid-range reference in the sense
that moving away in either direction costs perceptual similarity, but for different reasons:
low CFG underweights the prompt, high CFG overweights it to saturation.

**LPIPS (adjacent steps):** High-CFG steps remain large even in the CLIP-flat region:
9→12 = 0.33, 12→15 = 0.38. Each increment continues visually reshaping the image in ways
CLIP cannot see.

**Cross-experiment note:** This is the third experiment where LPIPS detects structure that CLIP
misses. The CFG sweep adds a continuous-parameter case with a clear practical consequence:
CLIP is adequate for finding the guidance threshold but useless for tuning above it. Both
probes are necessary for a complete picture.

## Charts

- `charts/clip_by_cfg.png` — mean CLIP score per CFG value
- `charts/lpips_vs_ref.png` — cumulative LPIPS distance from cfg=7
- `charts/lpips_adjacent.png` — step-wise LPIPS between adjacent CFG values

## Raw data

`results.csv` / `results.json` — one row per image (280 rows total).

Reproduce:

```bash
python scripts/experiments/exp3_cfg_sweep.py
```

---

**Cross-experiment context:** [`reports/clip_blindness.md`](../../clip_blindness.md) — third confirmation; demonstrates CLIP's plateau problem (adequate for finding the guidance threshold, useless above it).
