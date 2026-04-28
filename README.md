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
  Generate stunning AI art with Stable Diffusion 2.1 or SDXL. 
  Choose your model, adjust steps, guidance, size, and even set a seed for reproducible results!
---
# AetherArt

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/gauravgandhi2411/AetherArt)
[![GitHub](https://img.shields.io/badge/GitHub-gaurav--gandhi--2411%2FAetherArt-181717?logo=github)](https://github.com/gaurav-gandhi-2411/AetherArt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Text-to-image generator** built on Stable Diffusion 2.1 with ControlNet conditioning (Canny + Depth) and a fine-tuned Ukiyo-e LoRA adapter. Comes with a Gradio UI, a 360-generation CLIP benchmark, and a full LoRA training pipeline.

![LoRA comparison: base SD 2.1 vs Ukiyo-e LoRA](reports/lora_comparison_gallery.png)
*Top: base SD 2.1 · Bottom: Ukiyo-e LoRA — samurai on horseback · Fuji at sunset · dragon over Tokyo · kimono in garden*

## Live Space

**[https://huggingface.co/spaces/gauravgandhi2411/AetherArt](https://huggingface.co/spaces/gauravgandhi2411/AetherArt)**

The Space runs on CPU hardware and routes generation through the Hugging Face Inference API. For local GPU generation (RTX 3070 tested), clone and run locally.

## Quick Start

```bash
git clone https://github.com/gaurav-gandhi-2411/AetherArt.git
cd AetherArt

conda create -n aetherart python=3.10 -y
conda activate aetherart
pip install -r requirements.txt

# GPU torch (CUDA 12.4) — skip for CPU-only
pip install torch --index-url https://download.pytorch.org/whl/cu124

python app.py
```

Set `USE_HF_INFERENCE=1` to force routing through the Hugging Face Inference API regardless of local GPU availability.

## Architecture

Three models compose the generation stack:

| Component | Model | Role |
|---|---|---|
| **Base** | `sd2-community/stable-diffusion-2-1` | Core text-to-image diffusion |
| **ControlNet** | `thibaud/controlnet-sd21-canny-diffusers` | Edge-conditioned generation |
| **ControlNet** | `thibaud/controlnet-sd21-depth-diffusers` | Depth-conditioned generation |
| **LoRA adapter** | `data/lora/ukiyo-e/ukiyo-e-lora.safetensors` | Ukiyo-e style transfer (rank-8) |

All three run on a single RTX 3070 8 GB via model CPU offload. CLIP evaluation uses `openai/clip-vit-base-patch32`.

## Features

- Default model: SD 2.1; optional SDXL for higher quality (requires more VRAM)
- Adjustable: steps, guidance, width/height, seed
- VRAM optimisations: attention slicing, CPU offload
- Real-time step-by-step progress in the UI
- PNG metadata + sidecar JSON on every generation (prompt, seed, scheduler, LoRA, ControlNet settings, VRAM peak, git commit)
- "Recreate from PNG" — upload a previously generated image to restore all its settings
- ControlNet tab: Canny and Depth conditioning with live control-map preview
- LoRA Style tab: adapter selector + alpha slider + auto trigger-token injection

## Benchmark

Evaluated against a 30-prompt PartiPrompts subset spanning 11 categories. Metric: CLIP score via `openai/clip-vit-base-patch32`. 360 generations: 4 schedulers × 3 step counts × 30 prompts, seed = 42, RTX 3070 8 GB.

### Scheduler comparison

| Scheduler | Avg CLIP | Latency @ 20st | Latency @ 30st | Latency @ 50st | Verdict |
|---|---|---|---|---|---|
| **DPM-Solver++** | **0.3177** | 8.2 s | 10.8 s | 15.6 s | **Recommended default** |
| DDIM | 0.3170 | 8.3 s | 10.7 s | 15.6 s | Tied for top quality |
| LMS | 0.3117 | 8.1 s | 10.6 s | 15.3 s | Higher variance |
| Euler-Ancestral | 0.3106 | 8.2 s | 10.4 s | 15.2 s | Slightly lower quality |

### Key findings

- **DPM-Solver++ is the recommended default** — highest average CLIP with no latency penalty over DDIM
- **Step count has negligible effect on quality** — 20 → 50 steps shifts CLIP by < 0.002; **30 steps is the sweet spot**
- **VRAM is uniform at 4.50 GB** across all schedulers (CPU offload is the binding constraint, not the scheduler)
- **Outdoor photo-realism is SD 2.1's weak spot** — "a professional photo of a sunset behind the grand canyon" scored 0.20 (lowest); use SDXL for landscape photography
- **Styled characters score highest** — "a shiba inu wearing a beret and black turtleneck" hit 0.40 CLIP (global best)

| Chart | |
|---|---|
| ![Latency by scheduler](reports/eval_charts/latency_by_scheduler.png) | ![CLIP score by scheduler](reports/eval_charts/clip_score_by_scheduler.png) |
| ![Pareto frontier](reports/eval_charts/pareto_frontier.png) | ![VRAM peak by scheduler](reports/eval_charts/vram_by_scheduler.png) |

Run the full benchmark: `python scripts/eval.py`  
Smoke test (1 prompt, 1 scheduler): `python scripts/eval.py --prompts-subset pp_002 --schedulers DPM --steps 30`

## ControlNet Conditioning

AetherArt supports **Canny edge** and **Depth map** conditioning via SD 2.1-compatible ControlNet models.

| Mode | Model | Preprocessor |
|---|---|---|
| Canny | `thibaud/controlnet-sd21-canny-diffusers` | OpenCV Canny edge detection |
| Depth | `thibaud/controlnet-sd21-depth-diffusers` | DPT-Hybrid-MiDaS (`Intel/dpt-hybrid-midas`) |

Upload a reference image in the **ControlNet** accordion, choose Canny or Depth, and the preprocessed conditioning map is computed automatically. Use **Preview Control Map** to see the extracted edges or depth map before generating. ControlNet conditioning and the LoRA adapter can be used independently.

## LoRA Fine-tuning — Ukiyo-e Style

A rank-8 LoRA adapter fine-tuned on 80 WikiArt Ukiyo-e images using Stable Diffusion 2.1. Transfers Japanese woodblock print aesthetics: flat colour planes, bold outlines, traditional composition.

**Adapter:** `data/lora/ukiyo-e/ukiyo-e-lora.safetensors` (6.4 MB, checkpoint-1000 of 1500-step run)  
**Training:** `python scripts/train_lora.py` — 1500 steps, 2:08:52 total, RTX 3070 8 GB, fp16, no OOM  
**Trigger token:** `ukyowood` (auto-prepended when the adapter is active)  
**Negative prompt:** `text, watermark, calligraphy, signature, words, letters` (suppresses calligraphy artifact from training data)

### Checkpoint progression

![Fuji progression: baseline → ckpt-500 → ckpt-1000 → ckpt-1500](reports/lora_fuji_progression.png)
*Left to right: baseline · checkpoint-500 · checkpoint-1000 (selected) · checkpoint-1500. Prompt: "ukyowood ukiyo-e print of Mount Fuji at sunset"*

**Checkpoint-1000 was selected** — same structural quality as 500/750 (flat colour planes, woodblock outlines) with a warmer amber palette that better matches Hokusai's sunset compositions. Checkpoint-1500 shows loss uptick (0.495 vs 0.268 at 1250), suggesting overfitting onset.

### Usage

Enable the **LoRA Style** accordion in the UI, select **ukiyo-e**, and adjust the alpha slider (1.0 = full strength, >1 = exaggerated, <1 = subtle blend). Trigger token and negative prompt are added automatically.

To reproduce the fine-tuning:
```bash
python scripts/train_lora.py                     # full 1500-step run
python scripts/train_lora.py --max-train-steps 5 # pre-flight smoke test
```

See `reports/lora_training_summary.md` for the full training log, loss curve, and checkpoint selection rationale.

## Project Structure

```
AetherArt/
├── app.py                                  # Gradio UI — all four accordions
├── aetherart/
│   ├── model.py                            # SD 2.1 / SDXL pipeline + VRAM optimisations
│   ├── controlnet.py                       # ControlNet preprocessing + lazy-cached pipelines
│   ├── lora.py                             # LoRA registry, load/unload helpers
│   ├── metadata.py                         # PNG tEXt + sidecar JSON
│   └── config.py                           # env-driven config (model IDs, defaults)
├── data/lora/ukiyo-e/
│   ├── ukiyo-e-lora.safetensors            # selected adapter (6.4 MB, checkpoint-1000)
│   └── metadata.jsonl                      # 80 captions with ukyowood trigger
├── scripts/
│   ├── eval.py                             # 360-run CLIP benchmark
│   ├── train_lora.py                       # LoRA training wrapper (accelerate launch)
│   ├── compare_lora_checkpoints.py         # 6×6 checkpoint comparison grid
│   ├── build_lora_comparison_gallery.py    # 2-row base vs LoRA gallery
│   └── prepare_lora_dataset.py             # WikiArt dataset prep
├── reports/
│   ├── eval_charts/                        # 4 benchmark PNGs
│   ├── lora_comparison_gallery.png         # base vs LoRA gallery (in README)
│   ├── lora_fuji_progression.png           # baseline → ckpt-1500 strip
│   └── lora_training_summary.md           # full training log + checkpoint rationale
├── tests/                                  # pytest suite (imports, metadata, ControlNet)
├── requirements.txt
└── runtime.txt                             # python-3.10.12
```

## Citations

```bibtex
@inproceedings{rombach2022latent,
  title     = {High-Resolution Image Synthesis with Latent Diffusion Models},
  author    = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj{\"o}rn},
  booktitle = {CVPR},
  year      = {2022}
}

@article{zhang2023controlnet,
  title   = {Adding Conditional Control to Text-to-Image Diffusion Models},
  author  = {Zhang, Lvmin and Rao, Anyi and Agrawala, Maneesh},
  journal = {ICCV},
  year    = {2023}
}

@article{hu2021lora,
  title   = {LoRA: Low-Rank Adaptation of Large Language Models},
  author  = {Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  journal = {ICLR},
  year    = {2022}
}

@article{yu2022scaling,
  title   = {Scaling Autoregressive Models for Content-Rich Text-to-Image Generation},
  author  = {Yu, Jiahui and Xu, Yuanzhong and Koh, Jing Yu and Luong, Thang and Baid, Gunjan and Wang, Zirui and Vasudevan, Vijay and Ku, Alexander and Yang, Yinfei and Ayan, Burcu Karagol and others},
  journal = {TMLR},
  year    = {2022},
  note    = {PartiPrompts benchmark}
}
```

**Acknowledgments:** [Hugging Face diffusers](https://github.com/huggingface/diffusers) · [WikiArt dataset](https://www.wikiart.org/) · [CLIP](https://github.com/openai/CLIP) · ControlNet models by [Thibaud Ehret](https://huggingface.co/thibaud)
