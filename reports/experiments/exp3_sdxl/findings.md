# Experiment 3 (SDXL): CFG (Guidance Scale) Sweep

**Date:** 2026-06-02
**Hardware:** GCP L4 (24 GB VRAM)
**Model:** stabilityai/stable-diffusion-xl-base-1.0
**CFG values tested:** [1, 3, 5, 7, 9, 12, 15]
**Reference CFG for LPIPS:** 7
**Design:** 5 seeds x 8 prompts = 40 images per CFG value · 280 images total
**Scheduler:** DPM-Solver++ · 30 steps · 1024x1024
**Negative prompt:** standard (held constant across all CFG values)
**Scorers:** HPS (primary), ImageReward (primary), CLIP (comparison-only), LPIPS

## Hypothesis

HPS/CLIP will increase with CFG and plateau once the guidance is strong enough to anchor
the prompt (expected somewhere in the 5–9 range). LPIPS between adjacent values will reveal
a "regime change" — a step where visual character shifts sharply — that CLIP cannot detect.
At very high CFG (12–15) we expect over-saturation and structural artefacts that HPS and IR
will detect as quality degradation before CLIP reacts.

## Results

| CFG | Mean HPS | SE_HPS  | Mean IR | SE_IR   | Mean CLIP | SE_CLIP | LPIPS vs cfg=7 (cumulative) | LPIPS vs prev (step) |
|-----|:--------:|--------:|:-------:|--------:|----------:|--------:|------------------------------------:|---------------------:|
|  1    | 0.2049    | ±0.0043  | -0.6078   | ±0.1340  | 0.3015    | ±0.0058  | 0.5744               | —            |
|  3    | 0.2596    | ±0.0055  | 0.4916   | ±0.0916  | 0.3258    | ±0.0039  | 0.3743               | 0.4290            |
|  5    | 0.2724    | ±0.0059  | 0.5965   | ±0.0993  | 0.3242    | ±0.0035  | 0.2456               | 0.2523            |
|  7    | 0.2808    | ±0.0060  | 0.6525   | ±0.1005  | 0.3283    | ±0.0035  | 0.0000               | 0.2456            |
|  9    | 0.2828    | ±0.0060  | 0.6373   | ±0.0983  | 0.3274    | ±0.0035  | 0.2309               | 0.2309            |
| 12    | 0.2861    | ±0.0059  | 0.5924   | ±0.1135  | 0.3272    | ±0.0040  | 0.3574               | 0.2950            |
| 15    | 0.2900    | ±0.0055  | 0.6819   | ±0.1089  | 0.3298    | ±0.0040  | 0.4293               | 0.3253            |

## Key numbers

- HPS plateau starts at CFG = 12 (first value within 1 SE of max HPS = 0.2900)
- CLIP plateau starts at CFG = 3 (first value within 1 SE of max CLIP = 0.3298)
- Largest adjacent LPIPS step: 1→3 (LPIPS = 0.4290)
- HPS range across all CFG values: 0.0852
- IR range across all CFG values: 1.2897
- CLIP range across all CFG values: 0.0283 (7.01 SEs)

## CLIP-blindness verdict

CLIP delta across conditions = 0.0283, which is 7.01 SEs. HPS delta = 0.0852. IR delta = 1.2897. LPIPS range = 0.5744. Verdict: CLIP-BLIND: no.

## Interpretation

**HPS:** Plateaus at CFG=12, within 1 SE of the maximum (0.2900) from that point.

**CLIP (comparison-only):** Plateaus at CFG=3, within 1 SE of the maximum (0.3298) from that point.
Increasing CFG beyond the plateau does not improve semantic alignment as measured by CLIP.

**LPIPS (cumulative vs cfg=7):** Images diverge progressively from the cfg=7
reference as CFG moves in either direction. Low CFG (1, 3) and high CFG (12, 15) both produce
substantially different images from the mid-range baseline — but for different reasons: low CFG
underweights the prompt, high CFG overweights it to saturation.

**LPIPS (adjacent steps):** The largest single-step visual change is at the
1→3 transition (LPIPS = 0.4290). This
is the regime boundary where a one-unit CFG change produces the greatest pixel-level shift.
HPS and IR provide converging evidence for whether this regime boundary corresponds to a
human-perceptible quality change or merely a stylistic shift.

**Cross-experiment note:** If CLIP-blindness is confirmed (verdict above), this is the
SDXL replication of the same dissociation seen in SD 2.1 exp3 — CLIP cannot tell you when
the image has diverged perceptually; LPIPS is needed. HPS and IR add the human-preference
dimension: they can detect whether high-CFG oversaturation constitutes a quality regression,
not just a pixel-level change.

## Charts

- `charts/hps_by_cfg.png` — mean HPS score per CFG value (primary quality metric)
- `charts/ir_by_cfg.png` — mean ImageReward score per CFG value
- `charts/clip_by_cfg.png` — mean CLIP score per CFG value (comparison-only)
- `charts/lpips_vs_ref.png` — cumulative LPIPS distance from cfg=7
- `charts/lpips_adjacent.png` — step-wise LPIPS between adjacent CFG values

## Raw data

`results.csv` / `results.json` — one row per image (280 rows total).

Reproduce:

```bash
python scripts/experiments/exp3_sdxl.py
```
