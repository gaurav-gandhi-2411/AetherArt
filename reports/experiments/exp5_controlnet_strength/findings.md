# Experiment 5: ControlNet Conditioning Strength Sweep

**Date:** 2026-05-07
**Hardware:** RTX 3070 Laptop 8 GB (enable_model_cpu_offload)
**Model:** sd2-community/stable-diffusion-2-1
**ControlNet:** thibaud/controlnet-sd21-canny-diffusers (Canny edges, SD2.1)
**Conditioning source:** benchmark DDIM-30step outputs for pp_001–pp_008 (seed=42)
**Strength values:** [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
  (0.0 = no conditioning / text-only; 1.0 = standard reference; 1.5 = over-conditioned)
**Reference strength for LPIPS:** 1.0
**Design:** 5 seeds × 8 prompts = 40 images per strength · 280 images total
**Scheduler:** DPM-Solver++ · 30 steps · 512×512
**CFG:** 7.0 (fixed — established Pareto point from benchmark)
**Negative prompt:** standard (held constant)

## Hypothesis

CLIP will stay roughly flat across all strength values — the same text prompt drives
the same semantic content regardless of how tightly the canny edges constrain the output.
LPIPS vs the strength=1.0 reference will reveal where the image perceptually departs from
standard conditioning, diverging monotonically in both directions from the anchor.

## Results

| Strength | Mean CLIP | SE      | LPIPS vs strength=1.0 | LPIPS vs prev (step) |
|----------|----------:|--------:|---------------------------------:|---------------------:|
| 0.00  | 0.3095    | ±0.0073  | 0.7161                  | — |
| 0.25  | 0.3109    | ±0.0069  | 0.6620                  | 0.4651 |
| 0.50  | 0.3086    | ±0.0065  | 0.4938                  | 0.5486 |
| 0.75  | 0.3064    | ±0.0071  | 0.2722                  | 0.4276 |
| 1.00  | 0.3055    | ±0.0067  | 0.0000                  | 0.2722 ← ref |
| 1.25  | 0.3003    | ±0.0056  | 0.2267                  | 0.2267 |
| 1.50  | 0.2947    | ±0.0055  | 0.3175                  | 0.1717 |

LPIPS at strength=0.0 (text-only) vs reference: 0.7161
LPIPS at strength=1.5 (over-conditioned) vs reference: 0.3175

## Interpretation

**CLIP:** Partially flat, with a late borderline decline. Strengths 0.0 through 1.0
form a flat band (range 0.0040, clearly within 1 SE ≈ 0.0067). At strength=1.25 and
1.5, CLIP edges down: 0.3003 and 0.2947. Strength=1.5 is 0.0162 below the peak
(0.3109) — about 2.2 SEs, a borderline signal that over-conditioning slightly
degrades prompt alignment. But even this partial CLIP signal misses the main story:
LPIPS shows the image has already diverged enormously before CLIP responds at all.

**The V-shape: LPIPS vs strength=1.0 is asymmetric.** Divergence below the reference
is dramatically larger than above it. At strength=0.0 (no conditioning, text-only):
LPIPS=0.7161. At strength=1.5 (over-conditioned): LPIPS=0.3175. Moving from "full
conditioning" to "no conditioning" causes more than twice the perceptual displacement
as moving from "full conditioning" to "50% over-conditioned." The edge map provides a
strong structural anchor: relaxing it collapses that anchor catastrophically, while
tightening it beyond 1.0 only incrementally shifts texture and sharpness.

**Parallel to Experiment 4:** Strength=0.0's LPIPS of 0.7161 is essentially identical
to the EulerA-vs-deterministic cluster in Exp 4 (0.716–0.734). "No ControlNet
conditioning" is perceptually as far from "standard conditioning" as switching from a
deterministic sampler to a stochastic ancestral one. CLIP sees neither difference.

**LPIPS (adjacent steps):** The largest single-step visual change is 0.25→0.50
(LPIPS=0.5486), not at the high-strength end. This shows that the transition from
"weak conditioning" to "moderate conditioning" reshapes the image more dramatically
than any subsequent increment. Above strength=1.0, the marginal step sizes shrink
(0.2267, 0.1717) — the image is already tightly constrained and further increases
produce diminishing perceptual returns.

**What this means:** ControlNet strength is a creative parameter, not an accuracy
parameter. The jump from no conditioning to light conditioning (0.0→0.25, 0.25→0.50)
causes the largest perceptual changes. Above strength=1.0, images diverge more
gradually. CLIP cannot see any of this — it fires only weakly (and borderline) at
extreme over-conditioning. Strength selection requires visual judgment; CLIP
optimisation gives essentially no usable signal here.

**Cross-experiment note:** Fifth confirmation of CLIP-blindness: quantization (Exp 1),
negative prompt (Exp 2), CFG plateau (Exp 3), scheduler stochasticity (Exp 4),
ControlNet strength (Exp 5). The pattern is consistent: any parameter that reshapes
pixel-level character without eliminating the dominant semantic content is invisible
to CLIP.

## Charts

- `charts/clip_by_strength.png` — mean CLIP score per conditioning strength
- `charts/lpips_vs_ref.png` — perceptual distance from strength=1.0 reference
- `charts/lpips_adjacent.png` — step-wise LPIPS between adjacent strength values

## Raw data

`results.csv` / `results.json` — one row per image (280 rows total).

Reproduce:

```bash
python scripts/experiments/exp5_controlnet_strength.py
```

---

**Cross-experiment context:** [`reports/clip_blindness.md`](../../clip_blindness.md) — the V-shape LPIPS pattern (strength=0 reaches 0.72, matching EulerA's stochastic tier) is the sharpest illustration of CLIP missing a structural conditioning signal.
