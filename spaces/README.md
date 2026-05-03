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
  Stable Diffusion 2.1 with Ukiyo-e LoRA, ControlNet, LCM fast-generation, SDXL Turbo,
  and 4-bit/8-bit quantization. Three speed tiers, three memory modes, all in one UI.
---
# AetherArt

[![GitHub](https://img.shields.io/badge/GitHub-AetherArt-181717?logo=github)](https://github.com/gaurav-gandhi-2411/AetherArt)

> Stable Diffusion 2.1 + Ukiyo-e LoRA + ControlNet + three speed tiers (Standard / LCM / Turbo) + three memory modes (fp16 / 8-bit / 4-bit). Benchmarked across 360 generations.

**[Full documentation and benchmark results on GitHub →](https://github.com/gaurav-gandhi-2411/AetherArt)**  
**[→ 360-run scheduler benchmark findings](https://github.com/gaurav-gandhi-2411/AetherArt/blob/main/reports/findings.md)**

> **This Space runs on CPU** (~8–15 min per image). It demonstrates the architecture and lets you inspect the UI. For real-time generation, clone the repo and run locally with a CUDA GPU.

## Features

- **3 speed tiers**: Standard 30-step (best quality) · LCM 4-step (5.3× faster) · SDXL Turbo 1-step (~6× faster)
- **3 memory modes**: fp16 (3.1 GB) · 8-bit INT8 (2.2 GB) · 4-bit NF4 (2.8 GB peak, smallest weights)
- Ukiyo-e LoRA style adapter (rank-8, 80 WikiArt images, checkpoint-1000)
- ControlNet conditioning: Canny edge + Depth map (combinable with LoRA)
- DPM-Solver++ scheduler (best CLIP/latency in 360-run PartiPrompts benchmark)
- PNG + sidecar JSON metadata on every generation
- **Recreate from PNG** — every output embeds full generation parameters as PNG tEXt chunks; drag any prior output into the UI to restore exact settings

## Speed tiers (RTX 3070 measured)

| Mode | Steps | Local GPU | Quality |
|------|------:|-----------|---------|
| Standard fp16 | 30 | 3.2 s/img | Full baseline |
| LCM fast | 4 | 0.6 s/img — 5.3× | Moderate reduction |
| SDXL Turbo | 1 | 3.3 s/img (RTX 3070) | Lower; SDXL model 3× larger |

## Running locally

```bash
git clone https://github.com/gaurav-gandhi-2411/AetherArt.git
cd AetherArt
pip install -r requirements.txt
python app.py
```
