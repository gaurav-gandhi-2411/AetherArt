---
library_name: diffusers
base_model: stabilityai/stable-diffusion-xl-base-1.0
tags:
  - stable-diffusion-xl
  - sdxl
  - lora
  - text-to-image
  - ukiyo-e
  - art
  - aetherart
license: creativeml-openrail-m
pipeline_tag: text-to-image
---

# AetherArt — SDXL Ukiyo-e LoRA

A rank-8 LoRA adapter that steers Stable Diffusion XL toward Japanese ukiyo-e woodblock print style at 1024×1024 resolution. Trained on 80 WikiArt Ukiyo-e images against `stabilityai/stable-diffusion-xl-base-1.0`. Activate the style with the trigger token **`ukyowood`** anywhere in the prompt.

This is the SDXL companion to [`gauravgandhi2411/aetherart-ukiyo-sd21`](https://huggingface.co/gauravgandhi2411/aetherart-ukiyo-sd21), trained with identical rank, dataset, and step count for a controlled cross-resolution comparison. Both visual evaluations independently selected the checkpoint-1000 step count — at 512 for SD 2.1 and 1024 for SDXL.

## Sample outputs (checkpoint-1000, seed 42)

![Mount Fuji at sunset](samples/ckpt1000_mount-fuji-sunset.png)
*"ukyowood ukiyo-e woodblock print of Mount Fuji at sunset"*

![Crane over ocean waves](samples/ckpt1000_crane-ocean-waves.png)
*"ukyowood ukiyo-e print of a crane over ocean waves"*

![Samurai in bamboo forest](samples/ckpt1000_samurai-bamboo.png)
*"ukyowood ukiyo-e woodblock print of a samurai in a bamboo forest"*

## Usage

> **Required:** load the `madebyollin/sdxl-vae-fp16-fix` VAE alongside the base model. SDXL's default fp16 VAE produces black images without this fix.

```python
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, StableDiffusionXLPipeline
import torch

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16,
)
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()  # required on GPUs < 16 GB VRAM

pipe.load_lora_weights("gauravgandhi2411/aetherart-ukiyo-sdxl")

img = pipe(
    "ukyowood ukiyo-e woodblock print of Mount Fuji at sunset",
    negative_prompt=(
        "text, watermark, calligraphy, writing, letters, words, signature, "
        "blurry, low quality, western art, photograph, 3d render"
    ),
    num_inference_steps=25,
    guidance_scale=7.5,
    height=1024,
    width=1024,
).images[0]
img.save("output.png")
```

## Training details

| Parameter | Value |
|-----------|-------|
| Base model | `stabilityai/stable-diffusion-xl-base-1.0` |
| VAE (fp16-fix) | `madebyollin/sdxl-vae-fp16-fix` |
| LoRA rank | 8 |
| Training images | 80 (WikiArt Ukiyo-e) |
| Resolution | 1024 × 1024 |
| Steps | 1500 |
| Precision | fp16 mixed |
| Batch size | 1 (gradient accumulation = 4, effective batch = 4) |
| Learning rate | 1e-4 |
| Seed | 42 |
| Trigger token | `ukyowood` |
| Hardware | GCP g2-standard-4 — NVIDIA L4 24 GB VRAM |
| Training time | 4h 26m |
| Compute cost | ~$3.50 (GCP on-demand, us-central1) |

## Checkpoint selection

Evaluated checkpoints 500, 1000, and 1500 against four fixed prompts at seed 42, 25 DPM-Solver++ steps, 1024×1024.

| Checkpoint | Pixel mean (ref. prompt) | Visual verdict |
|---|---|---|
| 500 | 145.52 | Over-calligraphed; samurai figure scale weak — not selected |
| **1000** | **143.47** | **Deepest Hiroshige palette; figure preserved; cartouches integrated — SELECTED** |
| 1500 | 135.48 | Samurai figure dropout; cherry-blossom colour regression — not selected |

**Checkpoint-1000** was selected because it is the only checkpoint that simultaneously delivers (1) strong ukiyo-e style lift above the SDXL pretraining baseline, (2) figure preservation across all four test prompts, and (3) calligraphy rendered as single-panel title cartouches rather than scattered characters.

The step-1500 loss bump (~0.008 → ~0.08) correlates with a visible quality regression in the images — this is not microbatch noise. The 1000-step convergence point is consistent with the SD 2.1 companion run, which also selected checkpoint-1000.

### What the LoRA adds above SDXL's baseline

SDXL's pretraining already contains strong ukiyo-e priors. The LoRA's value is measurable rather than obvious: it deepens the Hiroshige teal/blue water palette, adds warm coral-pink atmospheric haze to horizon gradients, and reinforces the characteristic Japanese flat-plane treatment of foliage. The baseline renders cleaner figures but lacks the palette weight and compositional integration of traditional woodblock prints. The LoRA earns its place.

### Calligraphy artifact at 1024 (an evolution from 512)

The SD 2.1 companion adapter (trained at 512) produced scattered calligraphy characters in image margins — a training data artefact from WikiArt metadata captions embedded in source images. At 1024, the same signal manifests differently: single-panel title cartouches and red banner seals, compositionally placed where a real woodblock print would carry its title block. The artefact is still present, but at the higher resolution and with a stronger base prior, it reads as a learned stylistic element rather than noise. The default negative prompt suppresses most residual scatter.

## Default negative prompt

Applied automatically by the AetherArt application whenever this adapter is active:

```
text, watermark, calligraphy, writing, letters, words, signature, blurry, low quality, western art, photograph, 3d render
```

## Known limitations

- **Calligraphy artefact (partially mitigated):** WikiArt source images contain metadata captions and script text. The adapter learned this as part of ukiyo-e style. The negative prompt suppresses most instances but does not eliminate the entanglement between style signal and text signal. Correct fix: retrain on a curated dataset with no text annotations (~5 hours of curation).
- CLIP scoring does not capture the quality improvements from this adapter. See the CLIP-blindness finding below — nine experiments showed CLIP delta <1 SE while LPIPS ranged 0.40–0.73; the underfitting paradox (smaller adapters score *higher* on CLIP) confirms CLIP optimises in the wrong direction for style transfer evaluation.
- Evaluated at 1024×1024. Results at other resolutions are untested.
- `enable_model_cpu_offload()` is required on GPUs with less than ~16 GB VRAM; expect ~60–90 s/image under offload on an 8 GB card.

## Links

- **AetherArt repository:** https://github.com/gaurav-gandhi-2411/AetherArt
- **CLIP-blindness finding:** [`reports/clip_blindness.md`](https://github.com/gaurav-gandhi-2411/AetherArt/blob/main/reports/clip_blindness.md)
- **Training summary + visual eval:** [`reports/lora_training_summary_sdxl.md`](https://github.com/gaurav-gandhi-2411/AetherArt/blob/main/reports/lora_training_summary_sdxl.md)
- **Companion SD 2.1 adapter (512×512):** [`gauravgandhi2411/aetherart-ukiyo-sd21`](https://huggingface.co/gauravgandhi2411/aetherart-ukiyo-sd21)
