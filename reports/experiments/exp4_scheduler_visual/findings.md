# Experiment 4: Scheduler Visual Comparison

**Date:** 2026-05-07
**Source:** Existing 360-run benchmark (eval_results_20260425_124153.json); no new generation
**Hardware:** RTX 3070 Laptop 8 GB (images generated at benchmark time, seed=42)
**Schedulers:** DDIM, DPM, EulerA, LMS
**Step count:** 30 steps (the benchmark's Pareto-optimal operating point)
**Prompts:** 30 PartiPrompts (full benchmark set)
**Pairs compared:** 6 (DDIM-DPM, DDIM-EulerA, DDIM-LMS, DPM-EulerA, DPM-LMS, EulerA-LMS)

## Hypothesis

The original benchmark found that scheduler variance in CLIP score was 18× smaller than
prompt variance — schedulers are statistically indistinguishable by CLIP. LPIPS will test
whether "indistinguishable by CLIP" also means "perceptually equivalent."

## Results — CLIP by scheduler

| Scheduler | Mean CLIP | SE      |
|-----------|----------:|--------:|
| DDIM    | 0.3159    | ±0.0067 |
| DPM     | 0.3199    | ±0.0059 |
| EulerA  | 0.3121    | ±0.0052 |
| LMS     | 0.3086    | ±0.0076 |

CLIP range across schedulers: 0.0113 (≈ 1.8× the pooled SE — borderline; the full 360-run
benchmark pooled across all step counts showed a scheduler range of 0.003, 18× smaller than
prompt variance. At 30 steps alone with 30 prompts, sampling noise is larger.)

## Results — LPIPS by scheduler pair

| Pair           | Mean LPIPS | SE      |
|----------------|----------:|--------:|
| DDIM-DPM       | 0.4036    | ±0.0268 |
| DDIM-EulerA    | 0.7280    | ±0.0139 |
| DDIM-LMS       | 0.4768    | ±0.0272 |
| DPM-EulerA     | 0.7161    | ±0.0138 |
| DPM-LMS        | 0.3135    | ±0.0311 |
| EulerA-LMS     | 0.7336    | ±0.0157 |

EulerA cluster (pairs involving EulerA): 0.7161, 0.7280, 0.7336 — mean 0.726
Non-EulerA cluster (DDIM/DPM/LMS pairs): 0.3135, 0.4036, 0.4768 — mean 0.398
Mean LPIPS across all pairs: 0.5619

## Interpretation

**CLIP:** The 30-step scheduler CLIP range (0.0113) is borderline — about 1.8× the pooled
SE, not cleanly within noise. DPM leads (0.3199), LMS trails (0.3086). The full benchmark's
18× finding used all step counts and is the more reliable estimate; this slice has higher
sampling variance. The broad conclusion stands: schedulers produce small CLIP differences
compared to prompt choice.

**The structural LPIPS finding — two sampling regimes:** The pairwise LPIPS values split
into two distinct clusters.

- **EulerA pairs** (DDIM-EulerA, DPM-EulerA, EulerA-LMS): LPIPS 0.716–0.734. EulerA
  (Euler Ancestral) is a stochastic sampler — it adds fresh Gaussian noise at every
  denoising step. This introduces path-level randomness that makes EulerA outputs
  structurally different from any deterministic sampler, even at the same seed. The
  images have the same semantic content (CLIP is flat) but completely different textures,
  fine detail, and compositional choices at the pixel level.

- **Non-EulerA pairs** (DDIM-DPM, DDIM-LMS, DPM-LMS): LPIPS 0.314–0.477. DDIM,
  DPM-Solver++, and LMS are all deterministic (or near-deterministic) — they follow a
  consistent denoising trajectory from the same noise seed. Their mutual LPIPS values are
  large enough to show they still differ meaningfully, but they form a coherent cluster
  relative to EulerA.

**What this means for the benchmark recommendation:** Choosing DPM-Solver++ over DDIM
based on its CLIP lead (0.3199 vs 0.3159) is sound for semantic alignment. But it keeps
you in the deterministic cluster. Anyone preferring EulerA's stochastic texture character
cannot be detected — let alone selected — by CLIP. The scheduler choice is a creative
decision that CLIP is structurally blind to.

**Cross-experiment note:** Fourth experiment confirming CLIP-blindness: quantization (Exp 1),
negative prompt (Exp 2), CFG level (Exp 3), scheduler choice (Exp 4). The EulerA finding adds
a mechanistic explanation — stochastic sampling creates pixel-level divergence that is
invisible to embedding-based metrics by construction, not just by coincidence.

## Cost note

This experiment required zero GPU time — all images were reused from the existing
benchmark. LPIPS computation only.

## Charts

- `charts/clip_by_scheduler.png` — mean CLIP per scheduler (replicates benchmark)
- `charts/lpips_by_pair.png` — LPIPS between every scheduler pair
- `charts/clip_vs_lpips_range.png` — CLIP variation vs mean LPIPS side-by-side

## Raw data

`results.csv` — per-scheduler CLIP summary
`results_pairs.csv` — per-pair, per-prompt LPIPS and CLIP delta (180 rows)
`results.json` — aggregates + full pair data

Reproduce:

```bash
python scripts/experiments/exp4_scheduler_visual.py
```
