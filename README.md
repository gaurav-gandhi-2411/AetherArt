# AetherArt

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/gauravgandhi2411/AetherArt)

AetherArt — text-to-image generator with **Stable Diffusion 2.1** (`stabilityai/stable-diffusion-2-1`) and optional **SDXL** (`stabilityai/stable-diffusion-xl-base-1.0`) support. Runs locally on GPU with Gradio UI or via Hugging Face Inference API.

## Features
- Default: `stabilityai/stable-diffusion-2-1` (recommended)
- Optional: `stabilityai/stable-diffusion-xl-base-1.0` (SDXL) — high-quality / higher VRAM
- Adjustable generation: steps, guidance, width, height, seed
- VRAM optimizations: attention slicing, CPU offload, xformers support
- Real-time progress tracking in UI
- Optional seed for reproducible generations