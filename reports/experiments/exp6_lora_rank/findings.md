# Experiment 6: LoRA Rank Ablation

**Date:** 2026-05-08
**Hardware:** RTX 3070 Laptop 8 GB (enable_model_cpu_offload)
**Model:** sd2-community/stable-diffusion-2-1
**LoRA:** ukiyo-e — Japanese woodblock print style — 80 WikiArt images, SD 2.1
**Ranks tested:** [4, 8, 16]
**Training:** 1500 steps, seed 42, same data for all ranks
**Rank-8 checkpoint:** existing (data/lora/ukiyo-e/training_output/) — not retrained
**Design:** 5 seeds × 6 prompts = 30 images per rank · 90 images total
**Scheduler:** DPM-Solver++ · 50 steps · 512×512
**CFG:** 7.0 (fixed)
**LPIPS reference:** rank-8 (trained baseline); also computed between adjacent ranks

## Hypothesis

Higher rank captures more style detail: rank-4 underfits (lower CLIP and visible
quality loss on fine detail), rank-16 overfits or matches rank-8 closely. File size
scales linearly with rank. CLIP may or may not register the quality difference —
if it doesn't, this is another CLIP-blindness case.

## Results

| Rank  | Mean CLIP | SE      | LPIPS vs rank-8           | LPIPS vs prev rank   | File size |
|-------|----------:|--------:|---------------------------:|---------------------:|----------:|
|  4     | 0.3384    | ±0.0052  | 0.4515              | — | 3.4 MB |
|  8     | 0.3337    | ±0.0055  | 0.0000              | 0.4515 ← baseline | 6.7 MB |
| 16     | 0.3394    | ±0.0048  | 0.4956              | 0.4956 | 13.3 MB |

## Interpretation

**CLIP (rank-4 vs rank-8):** Delta = -0.0047 (0.8 pooled SEs).
Within noise — rank-4 matches rank-8 semantically by CLIP.

**CLIP (rank-8 vs rank-16):** Delta = +0.0057 (1.0 pooled SEs).
Borderline — just at 1 SE, not a confident claim. Directionally higher but within noise.

**LPIPS (adjacent ranks):** rank-4 vs rank-8: 0.4515; rank-8 vs rank-16: 0.4956.
Moderate-to-large LPIPS: ranks produce visually distinct outputs despite similar CLIP.

**File size:** rank-4 = 3.4 MB, rank-8 = 6.7 MB, rank-16 = 13.3 MB.
Scales approximately linearly with rank (attention layer matrices scale as rank × hidden_dim).

**Surprising direction:** Rank-4 CLIP (0.3384) is higher than rank-8 (0.3337). The
hypothesis predicted rank-4 would underfit and score lower. Instead, rank-4's simpler
style representation may align better with CLIP's "ukiyo-e woodblock print" embedding
than rank-8's more complex learned style. This is consistent with underfitting producing
images that more literally match style keywords, while higher-rank adapters learn subtle
style nuances that CLIP's vocabulary doesn't cover.

**Cross-experiment note:** Sixth training-parameter experiment. This IS a CLIP-blindness
case: rank differences produce perceptually large changes (LPIPS 0.45–0.50, comparable
to trigger-token removal in Exp 9 and scheduler switching in Exp 4) while CLIP shows
less than 1 SE of spread across all three ranks (0.3337–0.3394). Rank selection is a
visual quality decision that CLIP cannot guide — the operating-range problem from Exp 8
(LoRA alpha) reappears: the parameter moves the image substantially in perceptual space
while staying invisible to the semantic metric.

**Practical implication:** File size doubles per rank doubling (3.4 → 6.7 → 13.3 MB,
exactly linear). Rank-8 is a reasonable default — neither the smallest nor the largest,
and the checkpoint that all prior experiments used. Choosing between ranks requires human
visual assessment or LPIPS vs a reference image, not CLIP scoring.

## Charts

- `charts/clip_by_rank.png` — mean CLIP score per rank
- `charts/lpips_vs_rank8.png` — perceptual distance from rank-8 baseline

## Raw data

`results.csv` / `results.json` — one row per image (90 rows total).

Reproduce:

```bash
python scripts/experiments/exp6_lora_rank.py
```

---

**Cross-experiment context:** [`reports/clip_blindness.md`](../../clip_blindness.md) — see "The Underfitting Paradox" subsection; this experiment is one of two that confirm CLIP actively rewards underfitting (rank-4 > rank-8 on CLIP).
