# Experiment 9: LoRA Trigger Token Sensitivity

**Date:** 2026-05-07
**Hardware:** RTX 3070 Laptop 8 GB (enable_model_cpu_offload)
**Model:** sd2-community/stable-diffusion-2-1
**LoRA:** ukiyo-e — Japanese woodblock print style — 80 WikiArt images, SD 2.1, rank-8
**LoRA alpha:** 1.0 (fixed — trained default, loaded once for both conditions)
**Trigger token:** "ukyowood"
**Conditions:**
  - `no_trigger`: prompts use "ukiyo-e ..." style description, NO trigger token
  - `with_trigger`: identical prompts prepended with "ukyowood"
**CLIP reference:** semantic prompt text (without "ukyowood") — same for both conditions,
  so CLIP measures image–content alignment free of the trigger token's influence.
**Design:** 2 conditions × 8 prompts × 5 seeds = 80 images total
**Scheduler:** DPM-Solver++ · 30 steps · 512×512 · CFG=7.0

## Hypothesis

"ukyowood" is a trained trigger token not in CLIP's vocabulary. CLIP scores should be
near-identical between conditions — the token adds no semantic information CLIP can
interpret. LPIPS will determine whether the trigger actually changes how the LoRA fires:
a large LPIPS value means the trigger meaningfully redirects generation; a small value
means the LoRA fires similarly regardless of the trigger.

## Results

| Condition    | Mean CLIP | SE      | Mean LPIPS (between) |
|--------------|----------:|--------:|---------------------:|
| no_trigger   | 0.3305    | ±0.0046  | 0.4134 ±0.0281  |
| with_trigger | 0.3297    | ±0.0047  | (same pairs)         |

CLIP delta (with_trigger − no_trigger): -0.0008  (pooled SE = 0.0065)
LPIPS between conditions (mean ± SE): 0.4134 ±0.0281

## Interpretation

**CLIP:** Delta of -0.0008 is 0.12 pooled SEs — solidly within noise. CLIP has no
representation for "ukyowood" and cannot register the trigger token's presence or
absence. The hypothesis is confirmed exactly.

**LPIPS:** Mean 0.4134 ±0.0281 between the two conditions. This is large — squarely
in the range of deterministic scheduler-to-scheduler differences in Exp 4 (DDIM-DPM:
0.404, DDIM-LMS: 0.477). The trigger token is not decorative: removing it from the
prompt while keeping the LoRA active produces images that are perceptually as different
as switching between schedulers. CLIP registers none of this.

**What this tells us about trigger tokens:** "ukyowood" is doing meaningful work. It
is not redundant with the style-description words ("ukiyo-e", "woodblock print style")
in the prompt. The LoRA fires in a qualitatively different mode without the trigger —
LPIPS confirms the pixel-level divergence is real; CLIP cannot see it at all.

**Cross-experiment note:** Ninth confirmation of CLIP-blindness across this
experimental series: quantization (Exp 1), negative prompt (Exp 2), CFG plateau (Exp 3),
scheduler stochasticity (Exp 4), ControlNet strength (Exp 5), LoRA style scale (Exp 8,
with the nuance that CLIP has partial sensitivity when style is explicitly named in the
prompt), LoRA trigger token (Exp 9). The consistent theme: CLIP measures semantic
alignment reliably, but is structurally blind to parameters that reshape visual character
without eliminating prompt-relevant content — and Exp 9 is the clearest case, because
the trigger token has zero CLIP-vocabulary footprint by construction.

## Charts

- `charts/clip_by_condition.png` — CLIP score per condition
- `charts/lpips_by_prompt.png` — per-prompt mean LPIPS between conditions

## Raw data

`results.csv` / `results.json` — one row per image (80 rows total).

Reproduce:

```bash
python scripts/experiments/exp9_lora_trigger.py
```
