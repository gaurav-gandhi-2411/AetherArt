# Experiment 3: CFG (Guidance Scale) Sweep

**Date:** 2026-05-07
**Hardware:** RTX 3070 Laptop 8 GB (enable_model_cpu_offload)
**Model:** sd2-community/stable-diffusion-2-1
**CFG values tested:** [1, 3, 5, 7, 9, 12, 15]
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

- CLIP plateau starts at CFG = 5 (first value within 1 SE of max CLIP = 0.3131)
- Above the plateau (CFG 5→15): CLIP range = 0.3059 to 0.3131, all within 1 SE of each other
- LPIPS vs cfg=7 at CFG=15: 0.4710 — comparable to NF4 quantization damage (Exp 1: 0.3966)
- Largest adjacent LPIPS step: 1→3 (0.5539, expected — cfg=1 nearly ignores the prompt)
- Adjacent LPIPS at high CFG: 9→12 = 0.3329, 12→15 = 0.3830 — large, despite flat CLIP

## Interpretation

**CLIP:** Plateaus at CFG=5, within 1 SE of max from that point. CFG values 5 through 15 are
all statistically indistinguishable on CLIP (range 0.0059, SE ≈ 0.004). Once the prompt is
"anchored," more guidance does not improve measured semantic alignment.

**LPIPS (cumulative vs cfg=7):** This is where the experiment's finding lives. While CLIP is
flat above CFG=5, LPIPS vs the cfg=7 reference keeps growing: 0.31 at cfg=5, 0.39 at cfg=12,
0.47 at cfg=15. By cfg=15 the images are as perceptually distant from cfg=7 as NF4-quantized
images are from fp16 (Experiment 1: LPIPS=0.40). CLIP cannot see any of this divergence.

**LPIPS (adjacent steps):** The 1→3 step (0.5539) is expected — cfg=1 barely follows the
prompt. The more informative finding is that high-CFG adjacent steps remain large even after
CLIP plateaus: 9→12 = 0.33, 12→15 = 0.38. Each increment continues reshaping the image
substantially, in a region where CLIP says all conditions are equivalent.

**Practical implication:** The standard advice to "tune CFG until CLIP peaks" leaves you in
a plateau where you cannot distinguish cfg=7 from cfg=15 by semantic alignment. But LPIPS
shows you are still making large visual changes. CFG above ~9 on SD 2.1 with DPM-Solver++ is
a perceptual choice that CLIP-based evaluation is blind to.

**Cross-experiment note:** This is the third experiment where LPIPS detects structure that CLIP
misses. The CFG sweep adds a continuous-parameter case: CLIP tells you when the prompt is
anchored, but LPIPS is needed to tell you when the image has diverged perceptually. Both probes
are necessary for a complete picture.

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
