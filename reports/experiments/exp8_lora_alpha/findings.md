# Experiment 8: LoRA Style Scale (Alpha) Sweep

**Date:** 2026-05-07
**Hardware:** RTX 3070 Laptop 8 GB (enable_model_cpu_offload)
**Model:** sd2-community/stable-diffusion-2-1
**LoRA:** ukiyo-e — Japanese woodblock print style — 80 WikiArt images, SD 2.1, rank-8
**Trigger token:** ukyowood (used in all prompts)
**Alpha values:** [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
  (0.0 = LoRA loaded but suppressed / base model; 1.0 = standard weight; 1.5 = over-styled)
**Reference alpha for LPIPS:** 1.0
**Design:** 5 seeds × 8 prompts = 40 images per alpha · 280 images total
**Scheduler:** DPM-Solver++ · 30 steps · 512×512
**CFG:** 7.0 (fixed)
**Note:** The LoRA adapter is loaded once and `set_adapters()` is called between alpha
  values — no retraining. This isolates the effect of adapter weight magnitude on output.

## Hypothesis

CLIP will be largely insensitive to the stylistic shift the LoRA induces — the text
prompt describes the same content at any alpha. LPIPS will capture the substantial
visual differences between the unstyled base model (alpha=0) and styled output.

## Results

| Alpha | Mean CLIP | SE      | LPIPS vs alpha=1.0 | LPIPS vs prev (step) |
|-------|----------:|--------:|---------------------------:|---------------------:|
| 0.00  | 0.3164    | ±0.0039  | 0.6739              | — |
| 0.25  | 0.3270    | ±0.0050  | 0.5976              | 0.4460 |
| 0.50  | 0.3345    | ±0.0043  | 0.5319              | 0.4207 |
| 0.75  | 0.3319    | ±0.0048  | 0.4081              | 0.4211 |
| 1.00  | 0.3341    | ±0.0046  | 0.0000              | 0.4081 ← ref |
| 1.25  | 0.3337    | ±0.0039  | 0.3874              | 0.3874 |
| 1.50  | 0.3244    | ±0.0047  | 0.5022              | 0.3764 |

LPIPS at alpha=0.0 (base model, no style) vs reference: 0.6739
LPIPS at alpha=1.5 (over-styled) vs reference: 0.5022

## Interpretation

**CLIP rises from alpha=0.0 to 0.5 — but then plateaus.** CLIP goes from 0.3164 (no
LoRA) to 0.3345 at alpha=0.5, a gain of +0.0181 (~4 pooled SEs). The hypothesis was
wrong in one direction: CLIP is NOT fully insensitive. It can detect the LoRA firing
when the prompts explicitly name the style ("ukiyo-e", "woodblock print style"). Images
produced with an active LoRA better match style-describing text — CLIP registers this.

**However, the CLIP signal stops there.** Alpha=0.5, 0.75, 1.0, and 1.25 form a flat
band (0.3319–0.3345, within 1 SE of each other). CLIP cannot distinguish between
"moderately styled" and "fully styled" or "over-styled" despite LPIPS showing these
conditions produce images that are perceptually 0.40–0.50 apart. The practical
consequence: CLIP can tell you "LoRA is doing something", but cannot guide fine-tuning
of the alpha — the entire useful operating range (0.5–1.25) is invisible to it.

**LPIPS vs reference (V-shape, asymmetric):** No-LoRA (alpha=0.0) sits 0.6739 from
the reference — again in the EulerA / no-conditioning tier from Experiments 4 and 5.
Over-styling (alpha=1.5) reaches 0.5022. Moving away from the standard weight in
either direction causes large, CLIP-invisible perceptual changes.

**LPIPS (adjacent steps): remarkably uniform.** Step sizes are 0.446, 0.421, 0.421,
0.408, 0.387, 0.376 — nearly constant across the entire sweep. The LoRA style accrues
linearly rather than concentrating at any particular transition. There is no single
"regime change" alpha analogous to the CFG plateau in Exp 3.

**The refined CLIP-blindness picture:** This experiment adds nuance. CLIP is partially
sensitive to style when the style is explicitly named in the prompt — so "ukiyo-e style"
in the text gives CLIP partial grip. But CLIP cannot resolve the interior of the LoRA's
operating range. Any alpha ≥ 0.5 is metrically equivalent by CLIP while spanning 0.4+
LPIPS units of perceptual distance. Choosing a working alpha requires visual inspection.

**Cross-experiment note:** Sixth confirmation of CLIP-blindness (refined): quantization
(Exp 1), negative prompt (Exp 2), CFG plateau (Exp 3), scheduler stochasticity (Exp 4),
ControlNet strength (Exp 5), LoRA style scale (Exp 8). This experiment clarifies the
boundary: CLIP can partially detect style when explicitly named in the prompt, but
cannot distinguish within the active-adapter range where LPIPS shows large differences.

## Charts

- `charts/clip_by_alpha.png` — mean CLIP score per adapter weight
- `charts/lpips_vs_ref.png` — perceptual distance from alpha=1.0
- `charts/lpips_adjacent.png` — step-wise LPIPS between adjacent alpha values

## Raw data

`results.csv` / `results.json` — one row per image (280 rows total).

Reproduce:

```bash
python scripts/experiments/exp8_lora_alpha.py
```

---

**Cross-experiment context:** [`reports/clip_blindness.md`](../../clip_blindness.md) — the partial exception: CLIP rises +4 SE from no-LoRA to active-LoRA (prompts name the style) but is blind within the active range (alpha 0.5–1.25, LPIPS spread 0.40+).
