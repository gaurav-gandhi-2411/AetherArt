---
title: AetherArt
emoji: 🎨
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.46.1
app_file: app.py
pinned: false
license: mit
tags:
  - text-to-image
  - stable-diffusion
  - sd-xl
  - ai-art
  - gradio
  - gpu
  - huggingface
  - AI
  - art
  - image-generation
description: |
  Generate AI art with Stable Diffusion 2.1 + Ukiyo-e LoRA + ControlNet conditioning.
  DPM-Solver++ scheduler, real-time progress, PNG metadata, reproducible seeds.
---
# AetherArt

[![GitHub](https://img.shields.io/badge/GitHub-AetherArt-181717?logo=github)](https://github.com/gaurav-gandhi-2411/AetherArt)

> Stable Diffusion 2.1 + Ukiyo-e LoRA fine-tune + ControlNet (Canny + Depth). Includes a 360-generation PartiPrompts benchmark across 4 schedulers.

**[Full documentation and benchmark results on GitHub →](https://github.com/gaurav-gandhi-2411/AetherArt)**

## Features

- Default model: `sd2-community/stable-diffusion-2-1` — optional SDXL for higher quality
- Ukiyo-e LoRA style adapter (rank-8, 80 WikiArt images, checkpoint-1000)
- ControlNet conditioning: Canny edge + Depth map
- LoRA + ControlNet combinable — LoRA style applies through the ControlNet pipeline
- DPM-Solver++ scheduler (best CLIP/latency in 360-run benchmark)
- PNG + sidecar JSON metadata on every generation
- "Recreate from PNG" — upload a previous output to restore all settings
- Adjustable: steps, guidance, width, height, seed, LoRA alpha

## Running locally

```bash
git clone https://github.com/gaurav-gandhi-2411/AetherArt.git
cd AetherArt
pip install -r requirements.txt
python app.py
```
