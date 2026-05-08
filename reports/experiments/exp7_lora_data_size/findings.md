# Experiment 7: LoRA Training Data Size Ablation

**Date:** 2026-05-08
**Hardware:** RTX 3070 Laptop 8 GB (enable_model_cpu_offload)
**Model:** sd2-community/stable-diffusion-2-1
**LoRA:** ukiyo-e — Japanese woodblock print style — rank-8
**Data sizes tested:** [20, 40, 80] images
**Note:** 200-image condition dropped — dataset contains only 80 images. Sourcing
  additional training images is out of scope for this experiment.
**Subset selection:** fixed-seed random sample (seed=42); 20 ⊆ 40 ⊆ 80
**Training:** 1500 steps, seed 42, same rank/LR for all sizes
**80-image model:** existing checkpoint (data/lora/ukiyo-e/training_output/) — not retrained
**Design:** 5 seeds × 6 prompts = 30 images per data size · 90 images total
**Scheduler:** DPM-Solver++ · 50 steps · 512×512
**CFG:** 7.0 (fixed)
**LPIPS reference:** 80-image model (full dataset); also computed between adjacent sizes

## Hypothesis

More data → better style capture: 20-image model underfits (lower CLIP, less consistent
style), 80-image model is the best. LPIPS will show the 20-image model diverges more
from the 80-image reference than the 40-image model does. CLIP may or may not detect
this — if it doesn't, this is another CLIP-blindness case.

## Results

| Data  | Mean CLIP | SE      | LPIPS vs 80-img           | LPIPS vs prev size   | File size |
|-------|----------:|--------:|---------------------------:|---------------------:|----------:|
|  20    | 0.3383    | ±0.0048  | 0.6642              | — | 6.7 MB |
|  40    | 0.3303    | ±0.0042  | 0.5492              | 0.6330 | 6.7 MB |
|  80    | 0.3337    | ±0.0055  | 0.0000              | 0.5492 ← baseline | 6.7 MB |

## Interpretation

**CLIP (20-image vs 80-image):** Delta = -0.0046 (0.8 pooled SEs).
Within noise — 20-image model matches 80-image model semantically by CLIP.

**CLIP (40-image vs 80-image):** Delta = +0.0034 (0.6 pooled SEs).
Within noise — 40-image model matches 80-image model semantically by CLIP.

**LPIPS (adjacent sizes):** 20- vs 40-image: 0.6330; 40- vs 80-image: 0.5492.
Moderate-to-large LPIPS: data size produces visually distinct outputs despite similar CLIP.

**Checkpoint sizes:** all three are identical (6.7 MB) — file size is determined by
rank, not training data size. This is expected: the checkpoint stores trained weight
deltas whose dimensionality is rank × hidden_dim, independent of data.

**Surprising direction:** Data-20 CLIP (0.3383) > data-80 (0.3337), and data-40
(0.3303) is below both. The pattern is non-monotonic and entirely within noise (<1 SE),
but mirrors the finding from Exp 6 where rank-4 CLIP also exceeded rank-8. In both
cases the underfitted model's simpler style representation aligns better with CLIP's
"ukiyo-e woodblock print" embedding than the more richly trained model's subtle style
nuances. CLIP registers the style name in the prompt but cannot distinguish style
quality — a model trained on 20 images "matches" the style keyword as well as one
trained on 80.

**Cross-experiment note:** Seventh experiment in the CLIP-blindness series. Together
with Exp 6 (LoRA rank), this experiment completes the picture of LoRA training-time
parameters. The consistent theme: CLIP measures semantic alignment to the prompt but
cannot reliably detect changes in visual style quality or character that arise from
training decisions — data volume and adapter rank both move the image in perceptual
space in ways that LPIPS registers and CLIP does not.

## Charts

- `charts/clip_by_data_size.png` — mean CLIP score per training data size
- `charts/lpips_vs_data80.png` — perceptual distance from 80-image baseline

## Raw data

`results.csv` / `results.json` — one row per image (90 rows total).

Reproduce:

```bash
python scripts/experiments/exp7_lora_data_size.py
```
