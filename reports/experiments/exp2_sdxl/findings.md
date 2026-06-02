# Experiment 2 (SDXL): Negative Prompt Impact

**Date:** 2026-06-02
**Hardware:** GCP L4 (24 GB VRAM)
**Model:** stabilityai/stable-diffusion-xl-base-1.0
**Conditions:** no_neg (empty negative prompt) · with_neg (standard negative prompt)
**Design:** 5 seeds x 8 prompts = 40 images per condition · 80 images total
**Scheduler:** DPM-Solver++ · 30 steps · 1024x1024 · CFG=7.5
**Negative prompt tested:** `low quality, blurry, deformed, ugly, bad anatomy, watermark, text, calligraphy, signature`
**Scorers:** HPS (primary), ImageReward (primary), CLIP (comparison-only), LPIPS

## Hypothesis

The standard negative prompt guides the model away from degenerate outputs (blurry, deformed,
watermarked). This should reduce artifacts and may increase HPS/IR if the negative tokens
overlap with semantically poor regions, or decrease them if the guidance energy is reallocated
away from positive alignment. A null result (no reliable difference) is also plausible —
negative prompts primarily reshape the output distribution at the tails.

## Results

| Condition | Mean HPS | Mean IR | Mean CLIP | Latency (s) | Mean LPIPS (vs other cond.) |
|-----------|:--------:|:-------:|----------:|--------------:|----------------------------:|
| no_neg    | 0.2732   | 0.6895  | 0.3294    | 17.0s         | 0.3745                       |
| with_neg  | 0.2824   | 0.6491  | 0.3250    | 16.5s         | 0.3745                       |

SE on CLIP: no_neg ±0.0039 · with_neg ±0.0042
SE on HPS:  no_neg ±0.0050 · with_neg ±0.0061
SE on IR:   no_neg ±0.1048 · with_neg ±0.1156

CLIP delta (with_neg − no_neg): -0.0044 — between 1 and 2 SE (delta = -0.0044) — marginal, not reliable
HPS delta  (with_neg − no_neg): +0.0091
IR delta   (with_neg − no_neg): -0.0403

LPIPS between conditions (same seed/prompt pair): 0.3745 — substantial pixel differences between conditions (LPIPS = 0.3745)

## CLIP-blindness verdict

CLIP delta across conditions = 0.0044, which is 1.09 SEs. HPS delta = 0.0091. IR delta = 0.0403. LPIPS range = 0.3745. Verdict: CLIP-BLIND: yes.

## Per-prompt breakdown

See `charts/clip_delta_by_prompt.png`. Positive bars = negative prompt improved CLIP for that
prompt category; negative bars = negative prompt hurt CLIP. Variance across prompts reveals
whether the effect is consistent or prompt-dependent.

## Interpretation

The CLIP delta is between 1 and 2 SE (delta = -0.0044) — marginal, not reliable.
The LPIPS of 0.3745 between conditions tells us the negative prompt substantial pixel differences between conditions (LPIPS = 0.3745).

HPS and ImageReward provide human-preference-aligned signal: HPS delta = +0.0091,
IR delta = -0.0403. Where CLIP is blind to perceptual differences, these scorers
reveal whether the negative prompt improves or degrades the output quality experienced by
a human observer.

Latency difference: -0.55s — negative prompt text adds
minimal compute overhead (classifier-free guidance already processes a null embedding; replacing
it with a non-empty prompt does not change the number of forward passes).

## Charts

- `charts/hps_by_condition.png` — mean HPS score per condition (primary quality metric)
- `charts/ir_by_condition.png` — mean ImageReward score per condition
- `charts/clip_by_condition.png` — mean CLIP score per condition (comparison-only)
- `charts/lpips_between_conditions.png` — perceptual distance between matched pairs
- `charts/clip_delta_by_prompt.png` — per-prompt CLIP delta (with_neg minus no_neg)

## Raw data

`results.csv` / `results.json` — one row per image (80 rows total).

Reproduce:

```bash
python scripts/experiments/exp2_sdxl.py
```
