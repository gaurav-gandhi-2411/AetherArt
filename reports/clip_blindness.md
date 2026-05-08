# CLIP Blindness: A Cross-Experiment Finding

**Series:** Phase 6b — nine controlled experiments, May 2026  
**Model:** `sd2-community/stable-diffusion-2-1` · **CLIP:** `openai/clip-vit-base-patch32` · **LPIPS:** AlexNet backbone (lpips 0.1.4)

---

## TL;DR

Nine experiments varied one generation parameter at a time — quantization precision, negative prompt, CFG scale, scheduler, ControlNet strength, LoRA rank, training data size, adapter alpha, and trigger token. The result was the same every time: **CLIP score stayed flat while the images changed substantially.** Across all nine experiments, the maximum CLIP delta was mostly below 1 standard error; LPIPS (Learned Perceptual Image Patch Similarity) ranged from 0.40 to 0.73. The one partial exception is Experiment 8 (LoRA alpha), where CLIP registered a +4 SE jump when the LoRA switched on — because the prompts explicitly named the style — but then went blind again across the entire active range where LPIPS showed 0.40+ unit differences. The conclusion is not that CLIP is a bad metric. It is that CLIP measures *semantic alignment* — whether the image contains the content described by the prompt — and is structurally blind to any parameter that reshapes visual character without eliminating that content.

---

## Evidence table

| Exp | Parameter | Range | Max CLIP Δ (SE) | Max LPIPS | Detail |
|-----|-----------|-------|-----------------|-----------|--------|
| 1 | Quantization (fp16 / INT8 / NF4) | 3 precision modes | 0.94 SE | 0.40 (NF4 vs fp16) | [findings](experiments/exp1_quantization_quality/findings.md) |
| 2 | Negative prompt (off / on) | 2 conditions | 0.83 SE | 0.46 (between conditions) | [findings](experiments/exp2_negative_prompt/findings.md) |
| 3 | CFG scale (1–15) | 7 values | 1.10 SE | 0.47 (CFG=15 vs CFG=7) | [findings](experiments/exp3_cfg_sweep/findings.md) |
| 4 | Scheduler (DDIM / DPM / EulerA / LMS) | 4 schedulers | 1.80 SE | 0.73 (EulerA vs deterministic) | [findings](experiments/exp4_scheduler_visual/findings.md) |
| 5 | ControlNet strength (0.0–1.5) | 7 values | 2.20 SE | 0.72 (strength=0 vs strength=1) | [findings](experiments/exp5_controlnet_strength/findings.md) |
| 6 | LoRA rank (4 / 8 / 16) | 3 ranks | 1.00 SE | 0.50 (rank-16 vs rank-8) | [findings](experiments/exp6_lora_rank/findings.md) |
| 7 | LoRA training data (20 / 40 / 80 images) | 3 sizes | 0.80 SE | 0.66 (data-20 vs data-80) | [findings](experiments/exp7_lora_data_size/findings.md) |
| 8 | LoRA alpha / style scale (0.0–1.5) | 7 values | **4.00 SE** (no-LoRA → active) | 0.67 (no-LoRA vs reference) | [findings](experiments/exp8_lora_alpha/findings.md) |
| 9 | LoRA trigger token (with / without) | 2 conditions | 0.12 SE | 0.41 (between conditions) | [findings](experiments/exp9_lora_trigger/findings.md) |

**Notes on Exp 3 and Exp 8:** For Exp 3, the max LPIPS (0.47) uses the practical tuning range (CFG=15 vs CFG=7); the no-guidance baseline (CFG=1) reaches 0.65 but is not a realistic operating point. For Exp 8, the 4.00 SE CLIP delta applies to the jump from *no-LoRA* to *active-LoRA* — CLIP partially detects the style switch because the prompts name the style explicitly. Within the active-LoRA range (alpha 0.5–1.25), CLIP is flat despite LPIPS showing 0.40+ differences.

![CLIP-blindness chart: 9 experiments, CLIP SE vs max LPIPS](clip_blindness_chart.png)

---

## Why CLIP fails

CLIP (Contrastive Language-Image Pre-training) maps images and text into a shared embedding space and scores them by cosine similarity. The score answers one question: **does this image contain what the text describes?**

That is the only question it answers.

CLIP is trained on (image, caption) pairs scraped from the web. The signal for "good alignment" is that an image of a cat matches the caption "a cat" and doesn't match "a train." This trains the model to recognize semantic content — objects, scenes, styles when they are explicitly named — but not to distinguish between two images that both contain a cat but differ in texture, lighting, color palette, compositional sharpness, or stylistic rendering.

When this project varied CFG from 7 to 15, both images contained the subject described by the prompt. CLIP gave them the same score. When LoRA rank changed from 4 to 16, both images had the ukiyo-e style keywords present. CLIP gave them the same score. When the trigger token was removed from the prompt, the ukiyo-e semantic content was still there — "ukiyo-e woodblock print style" was still in the text — so CLIP could not register that the LoRA fired differently.

The failure mode is not noise or miscalibration. It is a definitional gap: **CLIP measures content presence, not rendering quality.** Any parameter that reshapes how content is rendered — without removing the content — is outside CLIP's measurement scope.

### The Underfitting Paradox

The most counterintuitive finding in this series: **underfit models score higher on CLIP than well-trained models.**

- Exp 6 (LoRA rank): rank-4 CLIP (0.3384) > rank-8 (0.3337), despite rank-4 being the smallest, least expressive adapter.
- Exp 7 (LoRA data): data-20 CLIP (0.3383) > data-80 (0.3337), despite 20 images being far less training data than 80.

The direction is wrong for a quality metric. In both cases, the underfit model learned a simpler, more literal style representation. When the prompt says "ukiyo-e woodblock print style," the underfit adapter produces images that are more directly recognizable as generic "ukiyo-e" by CLIP's embedding — because it learned the keyword's average, not the style's nuance.

The fully trained model learns subtler representations: specific color relationships, characteristic brushwork, compositional conventions that appear in the 80 WikiArt Ukiyo-e images but have no footprint in CLIP's vocabulary. CLIP cannot see these nuances. The result: **CLIP rewards the model that fails at style transfer but matches the keyword, while penalizing (or not rewarding) the model that succeeds at style transfer but expresses it in ways CLIP cannot decode.**

This is not a property of these specific LoRA hyperparameters. It is a structural property of any metric that measures keyword presence: the more literal the output, the better the keyword match, regardless of visual quality.

---

## Practical implications

**For parameter tuning:** CLIP is valid for two things in this project — confirming that the prompt is semantically present in the output, and comparing schedulers or step counts at the coarse level. For everything else (CFG above the plateau, ControlNet strength above 0.5, LoRA alpha within the active range, rank selection, data size), CLIP gives no signal. Use LPIPS against a reference image, or use visual inspection.

**For LoRA training decisions:** CLIP cannot guide rank selection, data curation quality, or checkpoint selection. A well-trained LoRA with rich style nuance will score the same as an underfit LoRA that produces generic results. Checkpoint selection for this project used visual inspection (checkpoint-1000 was selected for warm amber palette fidelity to Hokusai, not because CLIP preferred it).

**For evaluation design:** If you are evaluating a text-to-image system and care about rendering quality, style consistency, or fine-grained visual character, CLIP is insufficient as a sole metric. Supplement with at minimum one perceptual metric (LPIPS, FID, or DreamSim) and wherever possible with human ratings. The gap between CLIP and LPIPS in this series — consistently 0.40–0.73 units of perceptual difference that CLIP registers as zero — is large enough to matter practically.

**On Exp 8's partial sensitivity:** CLIP can partially detect style when the style is explicitly named in the prompt. This is a useful property for coarse style-presence checks ("is the LoRA contributing at all?") but not for fine-grained tuning. Any alpha ≥ 0.5 looks the same to CLIP while LPIPS shows 0.40+ units of within-range variation.

---

## Caveats

These findings are specific to this experimental setup:

- **Single base model:** SD 2.1 (865M-parameter U-Net). Results may differ for SDXL, SD 3, or distilled models with different training distributions.
- **Single CLIP variant:** `openai/clip-vit-base-patch32`. Larger CLIP variants (ViT-L/14, ViT-G/14) or fine-tuned CLIP-based metrics (SigLIP, BLIP-2, DreamSim) may have different blind spots.
- **Single style domain:** The LoRA experiments used Ukiyo-e (Japanese woodblock print). A different style domain — or a subject-specific adapter rather than a style adapter — might have different CLIP sensitivity characteristics.
- **Single dataset:** 30 PartiPrompts for the benchmark, 8 prompts for most Phase 6b experiments. The SE estimates are substantial (~0.004–0.007 per cell). A null result at n=8 or n=30 is not the same as a well-powered null.
- **No human ratings:** LPIPS is a learned perceptual metric, not a human judgment. The two correlate well in the literature but are not identical. LPIPS may be sensitive to differences that human raters would not notice, and insensitive to others that they would.

The consistent direction of the finding across nine experiments and multiple parameter types gives it reasonable robustness despite these limitations.

---

## Reproduce the chart

```bash
python scripts/generate_clip_blindness_chart.py
```

Output: `reports/clip_blindness_chart.png`
