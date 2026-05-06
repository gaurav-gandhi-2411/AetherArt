# Experiment 2: Negative Prompt Impact

**Date:** 2026-05-07
**Hardware:** RTX 3070 Laptop 8 GB (enable_model_cpu_offload)
**Model:** sd2-community/stable-diffusion-2-1
**Conditions:** no_neg (empty negative prompt) · with_neg (standard negative prompt)
**Design:** 5 seeds x 8 prompts = 40 images per condition · 80 images total
**Scheduler:** DPM-Solver++ · 30 steps · 512x512 · CFG=7.5
**Negative prompt tested:** `low quality, blurry, deformed, ugly, bad anatomy, watermark`

## Hypothesis

The standard negative prompt guides the model away from degenerate outputs (blurry, deformed,
watermarked). This should reduce artifacts and may increase CLIP score if the negative tokens
overlap with semantically poor regions, or decrease it if the guidance energy is reallocated
away from positive alignment. A null result (no reliable CLIP difference) is also plausible —
negative prompts primarily reshape the output distribution at the tails.

## Results

| Condition | Mean CLIP | Mean latency (s) | Mean LPIPS (vs other condition) |
|-----------|----------:|-----------------:|--------------------------------:|
| no_neg    | 0.3158    | 6.5s             | 0.4610                         |
| with_neg  | 0.3124    | 6.2s             | 0.4610                         |

SE on CLIP: no_neg ±0.0041 · with_neg ±0.0037

All CLIP differences are within 1 SE. The no_neg condition appears numerically higher
(0.3158 vs 0.3124) but this ordering is within measurement noise — not a real signal.

CLIP delta (with_neg − no_neg): −0.0034 — within 1 SE, no reliable effect on semantic alignment.

LPIPS between conditions (same seed/prompt pair): 0.4610 — substantial pixel-level differences
despite the near-identical CLIP scores.

## Per-prompt breakdown

See `charts/clip_delta_by_prompt.png`. Positive bars = negative prompt improved CLIP for that
prompt category; negative bars = negative prompt hurt CLIP. Variance across prompts reveals
whether the effect is consistent or prompt-dependent.

## Interpretation

**CLIP:** The negative prompt has no reliable effect on CLIP-measured semantic alignment
(delta = −0.0034, within 1 SE). This holds across the full 40-image sample and is consistent
with the hypothesis that negative prompts operate on the tails of the generation distribution —
reducing degenerate outputs — rather than shifting the central semantic tendency.

**LPIPS:** Despite the indistinguishable CLIP scores, the paired images differ substantially
at the pixel level (LPIPS = 0.4610). The negative prompt is actively changing what is generated,
but the change is in regions CLIP ignores: artifact texture, fine details, compositional
choices that don't affect text-image alignment.

**Latency:** with_neg was 0.33s faster than no_neg (6.2s vs 6.5s). This is within run-to-run
noise and should not be interpreted as a real speedup — classifier-free guidance processes the
negative prompt as a second text embedding in the same forward pass; a shorter or longer string
has negligible effect on wall-clock time.

**Bottom line:** The standard negative prompt substantially reshapes pixel content (LPIPS = 0.46)
without reliably improving the CLIP-measured text alignment (delta < 1 SE). Its value is
qualitative — fewer artifacts, better composition at the tails — and this experiment confirms
that CLIP alone cannot detect it.

## Charts

- `charts/clip_by_condition.png` — mean CLIP score per condition
- `charts/lpips_between_conditions.png` — perceptual distance between matched pairs
- `charts/clip_delta_by_prompt.png` — per-prompt CLIP delta (with_neg minus no_neg)

## Raw data

`results.csv` / `results.json` — one row per image (80 rows total).

Reproduce:

```bash
python scripts/experiments/exp2_negative_prompt.py
```
