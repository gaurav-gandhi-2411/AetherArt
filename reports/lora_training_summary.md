# Ukiyo-e LoRA Training Summary

**Date:** 2026-04-27 / 2026-04-28  
**Run ID:** 20260427_230001  
**Status:** COMPLETE — awaiting checkpoint selection and visual review

---

## Config

| Parameter | Value |
|---|---|
| Base model | `sd2-community/stable-diffusion-2-1` |
| Dataset | `data/lora/ukiyo-e/` — 80 images, trigger word `ukyowood` |
| Resolution | 512 × 512 |
| Steps | 1500 |
| Rank | 8 |
| LR | 1e-4 |
| Batch size | 1 (grad accum 4 → effective 4) |
| Mixed precision | fp16 |
| Gradient checkpointing | enabled |
| xformers | not installed (fallback) |
| Seed | 42 |

---

## Timing

| Metric | Value |
|---|---|
| Total wall time | **2:08:52** |
| Training steps | 2.61 s/step (stable) |
| Epoch cycle (train + validation) | ~85 s early → ~130 s late |
| Validation runs | 75 (every epoch, 4 images each) |
| Exit code | **0 (clean)** |

---

## Loss Curve

Loss values below are single-step readings at checkpoint time — noisy but indicative.

| Checkpoint | Step loss | Notes |
|---|---|---|
| checkpoint-250 | 0.284 | |
| checkpoint-500 | 0.295 | |
| checkpoint-750 | 0.416 | Single-step spike, not a trend break |
| checkpoint-1000 | 0.323 | |
| checkpoint-1250 | 0.268 | **Lowest checkpoint-time loss** |
| checkpoint-1500 | 0.495–0.509 | Uptick — possible overfitting onset |

**Aggregate:**
- First-400-readings avg: 0.4235
- Last-400-readings avg: 0.4081
- Last-100-readings avg: 0.4139

Modest overall decline. High per-step variance is expected for a 80-image dataset.

---

## Hardware

| Metric | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3070 Laptop GPU |
| VRAM total | 8.6 GB |
| OOM events | **0** |
| GPU temp (post-training) | 57 °C |
| Peak VRAM | ~7–8 GB estimated (no OOM at fp16 + grad ckpt) |

---

## Output Files

| File | Size | Notes |
|---|---|---|
| `training_output/pytorch_lora_weights.safetensors` | 6.4 MB | Final weights (step 1500) |
| `training_output/checkpoint-{250,500,750,1000,1250,1500}/pytorch_lora_weights.safetensors` | 6.4 MB each | Per-checkpoint LoRA weights |
| `training_output/checkpoint-*/` (full state) | 1.7 GB each | Optimizer + scheduler state |
| `training_output/training.log` | 822 KB | Full training log |
| **Total disk** | **9.9 GB** | Checkpoints gitignored |

> **Note:** Validation images were NOT saved to disk. The diffusers script only sends validation images to trackers (TensorBoard/W&B), neither of which was installed. Visual quality must be assessed via manual inference using each checkpoint's `pytorch_lora_weights.safetensors`.

---

## Checkpoint Selection — PENDING REVIEW

**Loss-based recommendation:** `checkpoint-1250` (lowest loss 0.268, step 1500 shows uptick suggesting overfitting onset)

**Suggested inspection order:**
1. `checkpoint-500` — first real style signal, likely under-baked
2. `checkpoint-750` — mid-training, often good balance
3. `checkpoint-1000` — solid, loss recovered from 750 spike
4. `checkpoint-1250` — lowest loss reading, recommended starting point
5. `checkpoint-1500` — final weights, risk of overfitting

**What to look for:**
- Flat color planes (ukiyo-e characteristic)
- Woodblock-print texture and outline strokes
- Traditional composition (not photorealistic)
- Trigger word adherence (`ukyowood` in prompt)

**To test a checkpoint:**
```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "sd2-community/stable-diffusion-2-1", torch_dtype=torch.float16
).to("cuda")
pipe.load_lora_weights(
    "data/lora/ukiyo-e/training_output/checkpoint-1250/pytorch_lora_weights.safetensors"
)
img = pipe("ukyowood ukiyo-e print of Mount Fuji at sunset", num_inference_steps=30).images[0]
img.save("test_1250.png")
```

---

## Selected Checkpoint

**To be filled after visual review.**

- Selected checkpoint: `checkpoint-____`
- Reason: _(visual quality notes)_
- Copied to: `data/lora/ukiyo-e/ukiyo-e-lora.safetensors`
