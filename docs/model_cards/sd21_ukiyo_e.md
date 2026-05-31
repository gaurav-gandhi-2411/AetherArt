---
library_name: diffusers
base_model: sd2-community/stable-diffusion-2-1
tags:
  - stable-diffusion
  - lora
  - text-to-image
  - ukiyo-e
  - art
  - aetherart
license: creativeml-openrail-m
pipeline_tag: text-to-image
---

# AetherArt — SD 2.1 Ukiyo-e LoRA

A rank-8 LoRA adapter that steers Stable Diffusion 2.1 toward Japanese ukiyo-e woodblock print style. Trained on 80 WikiArt Ukiyo-e images against the `sd2-community/stable-diffusion-2-1` base model. Activate the style with the trigger token **`ukyowood`** anywhere in the prompt.

## Usage

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "sd2-community/stable-diffusion-2-1",
    torch_dtype=torch.float16,
).to("cuda")

pipe.load_lora_weights("gauravgandhi2411/aetherart-ukiyo-sd21")

img = pipe(
    "a ukyowood mountain landscape at sunset, traditional woodblock print",
    negative_prompt="text, watermark, calligraphy, signature, words, letters",
    num_inference_steps=30,
    guidance_scale=7.5,
).images[0]
img.save("output.png")
```

## Training details

| Parameter | Value |
|-----------|-------|
| Base model | `sd2-community/stable-diffusion-2-1` |
| LoRA rank | 8 |
| Training images | 80 (WikiArt Ukiyo-e) |
| Resolution | 512 × 512 |
| Steps | 1500 |
| Precision | fp16 mixed |
| Batch size | 1 (gradient accumulation = 4, effective batch = 4) |
| Learning rate | 1e-4 |
| Seed | 42 |
| Hardware | NVIDIA RTX 3070 Laptop GPU (8 GB VRAM) |
| Training time | ~2 h 8 min |

## Checkpoint selection

Selected **checkpoint-1000** over the other checkpoints trained during the run:

- **checkpoint-500**: underfit — style signal present but not saturated, more like a mild filter than a transformation.
- **checkpoint-1000**: selected — consistent warm amber palette and characteristic flatness of traditional woodblock prints across test prompts.
- **checkpoint-1500**: overfit — validation loss rose from 0.268 to 0.495. Outputs showed over-saturated colors and partial prompt-alignment breakdown.

Checkpoint selection was made by visual evaluation only — no quantitative held-out set was used.

## Default negative prompt

The following negative prompt is applied automatically by the AetherArt application whenever this adapter is active:

```
text, watermark, calligraphy, signature, words, letters
```

## Known limitations

- **Calligraphy artifact (partially mitigated, not fixed):** WikiArt Ukiyo-e source images contain metadata captions with artist signatures and script text embedded in the image margins. The adapter learned these as part of "ukiyo-e style." The default negative prompt suppresses most instances but does not eliminate the artifact entirely — the style signal and text signal are entangled in the adapter weights. The correct fix is retraining on a curated dataset with no text annotations, which would require approximately 5 hours of curation work.
- The adapter was trained and evaluated on 512 × 512 resolution. Results at other resolutions are untested.
- CLIP scoring does not capture the quality improvements from this adapter. See the CLIP-blindness finding linked below.

## Links

- **AetherArt repository:** https://github.com/gaurav-gandhi-2411/AetherArt
- **CLIP-blindness finding:** see [`reports/clip_blindness.md`](https://github.com/gaurav-gandhi-2411/AetherArt/blob/main/reports/clip_blindness.md) — nine Phase 6b experiments showing CLIP delta <1 SE while LPIPS ranged 0.40–0.73; underfitting paradox; why CLIP cannot guide LoRA training decisions.
- **Companion SDXL adapter (1024×1024):** [`gauravgandhi2411/aetherart-ukiyo-sdxl`](https://huggingface.co/gauravgandhi2411/aetherart-ukiyo-sdxl) — same rank-8, same dataset, trained on GCP L4. Both runs independently select checkpoint-1000.
