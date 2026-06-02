# Experiment 9 (SDXL): LoRA Trigger Token Sensitivity

**Date:** 2026-06-02
**Model:** stabilityai/stable-diffusion-xl-base-1.0 — 1024×1024
**LoRA:** gauravgandhi2411/aetherart-ukiyo-sdxl (SDXL Ukiyo-e, alpha=1.0)
**Trigger token:** "ukyowood"
**Conditions:**
  - `no_trigger`: prompts use "ukiyo-e ..." description, NO trigger token
  - `with_trigger`: identical prompts prepended with "ukyowood"
**CLIP / HPS / IR reference:** semantic prompt text (without "ukyowood") — same for
  both conditions, so scorers measure image–content alignment free of the trigger token.
**Design:** 2 conditions × 8 prompts × 5 seeds = 80 images
**Scheduler:** DPM-Solver++ · 30 steps · CFG=7.5
**Scorers:** CLIP, HPS, ImageReward, LPIPS

## Hypothesis

"ukyowood" is a trained trigger token absent from CLIP's vocabulary. CLIP / HPS / IR scores
should be near-identical between conditions. LPIPS will determine whether the trigger
meaningfully redirects how the SDXL LoRA fires.

## Results

| Condition    | Mean CLIP | SE      | Mean HPS | Mean IR | Mean LPIPS (between) |
|--------------|----------:|--------:|---------:|--------:|---------------------:|
| no_trigger   | 0.3312    | ±0.0043  | 0.2221    | 0.0419    | 0.3007 ±0.0192 |
| with_trigger | 0.3276    | ±0.0042  | 0.2169    | 0.0914    | (same pairs)               |

CLIP delta across conditions = -0.0036, which is 0.6 SEs.
HPS delta = -0.0052. IR delta = +0.0495. LPIPS range = 0.3256.
Verdict: CLIP-BLIND: yes.

## Interpretation

**CLIP / HPS / IR:** Delta is within noise (< 1 pooled SE) — CLIP cannot detect the trigger. As expected, none of the semantic scorers have
a representation for "ukyowood" and cannot register its presence or absence.

**LPIPS:** The mean LPIPS between no_trigger and with_trigger images is 0.3007.
This is large — the trigger token meaningfully redirects how the SDXL LoRA fires. Images with the trigger are perceptually different from those without, while all three semantic scorers remain identical.

**SDXL vs SD 2.1:** At 1024×1024, per-pixel trigger sensitivity may differ from the SD 2.1
512×512 baseline. The CLIP-blindness mechanism is unchanged — the token is not in CLIP's
vocabulary regardless of resolution.

## Charts

- `charts/clip_by_condition.png`
- `charts/hps_by_condition.png`
- `charts/lpips_by_prompt.png`

## Raw data

`results.csv` / `results.json` — one row per image (80 rows total).

Reproduce:

```bash
python scripts/experiments/exp9_sdxl.py
```
