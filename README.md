# AetherArt

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/gauravgandhi2411/AetherArt)
[![GitHub](https://img.shields.io/badge/GitHub-AetherArt-181717?logo=github)](https://github.com/gaurav-gandhi-2411/AetherArt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Stable Diffusion 2.1 fine-tuned with LoRA + ControlNet, with a PartiPrompts evaluation harness benchmarking 4 schedulers across 360 generations.

![Hero — Ukiyo-e LoRA showcase](docs/hero.png)
*Ukiyo-e LoRA adapter · DPM-Solver++ · 50 steps · seed 42 · RTX 3070 8 GB*

**[Try the live Space →](https://huggingface.co/spaces/gauravgandhi2411/AetherArt)**

---

## Table of Contents

- [Why This Project Exists](#why-this-project-exists)
- [Architecture](#architecture)
- [Models Used (and Why)](#models-used-and-why)
- [Performance Trade-offs](#performance-trade-offs)
- [LoRA Fine-tuning](#lora-fine-tuning)
- [ControlNet Conditioning](#controlnet-conditioning)
- [Benchmark Results](#benchmark-results)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Free-Tier Limitations](#free-tier-limitations)
- [Future Improvements](#future-improvements)
- [Citations](#citations)
- [Acknowledgments](#acknowledgments)

---

## Why This Project Exists

I wanted to deeply understand modern image generation by building one end-to-end on consumer hardware — not just "use the API" but implement the pieces so I know what each one does:

- The base diffusion model and its scheduler trade-offs
- LoRA — how to actually fine-tune the U-Net's attention layers on a small dataset without an A100
- ControlNet — how spatial conditioning interacts with the diffusion process
- Evaluation methodology — how to measure that any of this actually improves anything

This README documents the engineering decisions made, including the ones that didn't work.

---

## Architecture

```
User Prompt
    │
    ▼
[Prompt Builder] ─── LoRA active? ──→ prepend trigger token + inject negative
    │
    ▼
[SD 2.1 U-Net]
    │                     │
    │  ControlNet active?  │
    └────────────────────→ [ControlNet Preprocessor]
                                    │
                              Canny / Depth map
                                    │
                            [ControlNet Pipeline]
    │                               │
    └───────────── merge ───────────┘
                       │
                       ▼
               Generated Image
                       │
                       ▼
          PNG + sidecar JSON (metadata)
```

| Component | Model | Role |
|---|---|---|
| Base | `sd2-community/stable-diffusion-2-1` | Text-to-image diffusion |
| ControlNet (Canny) | `thibaud/controlnet-sd21-canny-diffusers` | Edge-conditioned generation |
| ControlNet (Depth) | `thibaud/controlnet-sd21-depth-diffusers` | Depth-conditioned generation |
| LoRA adapter | `data/lora/ukiyo-e/ukiyo-e-lora.safetensors` | Ukiyo-e style transfer (rank-8) |
| Scheduler | DPMSolverMultistepScheduler | Best CLIP/latency trade-off in benchmark |
| Evaluator | `openai/clip-vit-base-patch32` | Prompt-image similarity scoring |

All components run on a single RTX 3070 8 GB via model CPU offload (fp16). LoRA and ControlNet pipelines share a 2-entry LRU cache to manage the ~3 GB per-pipeline VRAM footprint.

---

## Models Used (and Why)

### Why SD 2.1 and not SDXL or SD 3.5?

SDXL needs ~10 GB VRAM for inference, more for training. My laptop has 8 GB on the RTX 3070. SD 3.5 has higher requirements still. SD 2.1 fits cleanly in 8 GB with fp16 + attention slicing.

When Stability AI deprecated `stabilityai/stable-diffusion-2-1` in early 2026 (EU AI Act compliance), I switched to the community-maintained `sd2-community/stable-diffusion-2-1` mirror. Same weights, same diffusers API, no breaking change to any downstream code.

### Why ControlNet 2.1 and not the SDXL versions?

ControlNet checkpoints must match the base model's U-Net architecture and resolution. The `thibaud/controlnet-sd21-*` family is the matching pair for SD 2.1. Using an SDXL ControlNet on an SD 2.1 pipeline fails silently — the conditioning map is silently ignored because the cross-attention dimensions don't match.

### Why LoRA at rank 8?

Rank-8 LoRA = 6.4 MB on disk. Rank-16 = 12.8 MB, with marginal quality gain on small datasets. With 80 training images, rank 8 is sufficient without overfitting. The loss curve confirms this — loss plateaus at checkpoint-1000 and ticks up at 1500, which is the classic sign the model has memorised the training set.

### Why these four schedulers?

DDIM is the canonical baseline. DPM-Solver++ is the current Pareto-optimal choice in the diffusers literature. Euler-Ancestral and LMS fill out the comparison set. The benchmark section below has the numbers; DPM-Solver++ wins by a small but consistent margin across all 30 prompts.

---

## Performance Trade-offs

### Why does this take 30-60 seconds when commercial APIs respond in ~5 seconds?

Commercial text-to-image services run on:
- **H100/A100 GPUs** (80 GB VRAM) purpose-built for batch inference
- **TensorRT-compiled models** — CUDA kernel fusion gives a 3–5× speedup over vanilla PyTorch
- **Distilled models** (SDXL Turbo, FLUX schnell) — 1–4 step generation vs 30 steps
- **Batched inference** — fixed per-request overhead amortised over hundreds of concurrent users

This project runs on:
- **RTX 3070 Laptop 8 GB** — ~12× less memory bandwidth than an A100
- **PyTorch eager mode** — no TensorRT, no kernel fusion
- **Full SD 2.1** — not distilled, 30 steps to convergence
- **fp16 with CPU offload** — model layers swap between GPU and CPU during inference

The 10–15 s local generation time reflects hardware constraints, not bad code. With a paid Spaces GPU instance (A10G, $0.60/hr) it drops to 4–6 s.

### VRAM breakdown

```
SD 2.1 U-Net (fp16):         ~4.5 GB peak (with model CPU offload)
ControlNet pipeline:          ~3.0 GB additional (separate pipeline object)
LoRA adapter:                  6.4 MB (negligible)
Total worst case (base + CN): ~7.5 GB — fits in 8 GB
```

---

## LoRA Fine-tuning

Rank-8 LoRA adapter fine-tuned on 80 WikiArt Ukiyo-e images using the diffusers `train_text_to_image_lora.py` script. The adapter modifies the U-Net's self- and cross-attention projection weights, leaving the rest of the model frozen.

| Parameter | Value |
|---|---|
| Base model | `sd2-community/stable-diffusion-2-1` |
| Dataset | 80 WikiArt Ukiyo-e images, trigger `ukyowood` |
| Rank | 8 |
| Steps | 1500 (checkpoint-1000 selected) |
| LR | 1e-4, mixed precision fp16 |
| Wall time | 2 h 8 min, RTX 3070 8 GB, 0 OOM events |
| Adapter size | 6.4 MB |

### Checkpoint selection

![Fuji progression: baseline → ckpt-500 → ckpt-1000 → ckpt-1500](reports/lora_fuji_progression.png)
*Left to right: baseline · ckpt-500 · ckpt-1000 (selected) · ckpt-1500 · Prompt: "ukyowood ukiyo-e print of Mount Fuji at sunset"*

Checkpoint-1000 was selected over 1250 and 1500 for its warmer amber palette matching Hokusai's sunset compositions. Loss at 1500 ticks up from 0.268 (at 1250) to 0.495, indicating overfitting onset.

### Base SD 2.1 vs Ukiyo-e LoRA

![LoRA comparison gallery](reports/lora_comparison_gallery.png)
*Top: base SD 2.1 · Bottom: Ukiyo-e LoRA (alpha=1.0, trigger added)*

### Known limitation: calligraphy artifact

Several WikiArt source images contain embedded calligraphy text. The LoRA absorbed this as part of the style signal. Mitigation at inference time: apply negative prompt `text, watermark, calligraphy, signature, words, letters`. This is set as the default negative when the Ukiyo-e adapter is active in the UI.

### Usage

Enable the **LoRA Style** accordion in the UI, select `ukiyo-e`, and adjust alpha (1.0 = full strength, >1 = exaggerated, <1 = subtle blend). Trigger token and negative prompt are added automatically.

```bash
python scripts/train_lora.py                     # full 1500-step run
python scripts/train_lora.py --max-train-steps 5 # pre-flight smoke test
```

See `reports/lora_training_summary.md` for the full training log, loss curve, and checkpoint rationale.

---

## ControlNet Conditioning

Canny and Depth conditioning via SD 2.1-compatible ControlNet models. LoRA and ControlNet can now be combined — the LoRA is loaded directly into the ControlNet pipeline rather than the base SD 2.1 pipeline, avoiding weight conflicts.

| Mode | Model | Preprocessor |
|---|---|---|
| Canny | `thibaud/controlnet-sd21-canny-diffusers` | OpenCV Canny edge detection |
| Depth | `thibaud/controlnet-sd21-depth-diffusers` | DPT-Hybrid-MiDaS (`Intel/dpt-hybrid-midas`) |

Upload a reference image in the **ControlNet** accordion, choose Canny or Depth, and the control map is computed automatically. Use **Preview Control Map** to inspect the extracted edges or depth map before generating.

**VRAM note:** ControlNet runs on a separate pipeline (~3 GB additional). With the 2-entry LRU cache, the oldest (ctype, lora, alpha) combination is evicted when a third is needed.

---

## Benchmark Results

Evaluated against a 30-prompt PartiPrompts subset spanning 11 categories. Metric: CLIP score (`openai/clip-vit-base-patch32`). 360 generations: 4 schedulers × 3 step counts × 30 prompts, seed = 42, RTX 3070 8 GB.

### Scheduler comparison

| Scheduler | Avg CLIP | Latency @ 20st | Latency @ 30st | Latency @ 50st | Verdict |
|---|---|---|---|---|---|
| **DPM-Solver++** | **0.3177** | 8.2 s | 10.8 s | 15.6 s | **Recommended default** |
| DDIM | 0.3170 | 8.3 s | 10.7 s | 15.6 s | Tied for top quality |
| LMS | 0.3117 | 8.1 s | 10.6 s | 15.3 s | Higher variance |
| Euler-Ancestral | 0.3106 | 8.2 s | 10.4 s | 15.2 s | Slightly lower quality |

### Key findings

- **30 steps is the sweet spot** — 20→50 steps shifts CLIP by < 0.002 while doubling compute
- **VRAM is uniform at 4.50 GB** across all schedulers — model CPU offload is the binding constraint
- **Outdoor photo-realism is SD 2.1's weak spot** — "a professional photo of a sunset behind the grand canyon" scored 0.20; use SDXL for landscape photography
- **Styled characters score highest** — "a shiba inu wearing a beret and black turtleneck" hit 0.40 CLIP

| Chart | |
|---|---|
| ![Latency by scheduler](reports/eval_charts/latency_by_scheduler.png) | ![CLIP score by scheduler](reports/eval_charts/clip_score_by_scheduler.png) |
| ![Pareto frontier](reports/eval_charts/pareto_frontier.png) | ![VRAM peak by scheduler](reports/eval_charts/vram_by_scheduler.png) |

```bash
python scripts/eval.py                                                           # full 360-run benchmark
python scripts/eval.py --prompts-subset pp_002 --schedulers DPM --steps 30     # smoke test
```

---

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
# → http://localhost:7860
```

Set `USE_HF_INFERENCE=1` to route generation through the Hugging Face Inference API instead of loading models locally.

---

## Project Structure

```
AetherArt/
├── app.py                                  # Gradio UI — generation, ControlNet, LoRA, metadata
├── aetherart/
│   ├── model.py                            # SD 2.1 / SDXL pipeline + VRAM optimisations
│   ├── controlnet.py                       # ControlNet preprocessing + LRU-cached pipelines
│   ├── lora.py                             # LoRA registry, load/unload helpers
│   ├── metadata.py                         # PNG tEXt + sidecar JSON
│   └── config.py                           # env-driven config (model IDs, defaults)
├── data/lora/ukiyo-e/
│   ├── ukiyo-e-lora.safetensors            # selected adapter (6.4 MB, checkpoint-1000)
│   └── metadata.jsonl                      # 80 captions with ukyowood trigger token
├── scripts/
│   ├── eval.py                             # 360-run CLIP benchmark harness
│   ├── train_lora.py                       # LoRA training wrapper (accelerate launch)
│   ├── generate_hero_image.py              # 2×2 Ukiyo-e showcase grid for README
│   ├── compare_lora_checkpoints.py         # 6×6 checkpoint comparison grid
│   ├── build_lora_comparison_gallery.py    # base vs LoRA comparison gallery
│   └── prepare_lora_dataset.py             # WikiArt dataset prep + caption generation
├── docs/
│   └── hero.png                            # 2×2 Ukiyo-e LoRA showcase (README header)
├── reports/
│   ├── eval_charts/                        # 4 benchmark PNGs
│   ├── lora_comparison_gallery.png         # base vs LoRA, 4 prompts
│   ├── lora_fuji_progression.png           # baseline → ckpt-1500 progression strip
│   └── lora_training_summary.md           # full training log + checkpoint selection rationale
├── spaces/
│   └── README.md                           # HF Space version (with YAML frontmatter)
├── tests/                                  # pytest suite: imports, metadata, preprocessing, cache keys
├── requirements.txt
└── runtime.txt                             # python-3.10.12
```

---

## Free-Tier Limitations

What couldn't be done on the Hugging Face Spaces free tier:

| Limitation | Impact | Workaround |
|---|---|---|
| No GPU | 30–60 s generation vs 4–6 s on A10G | Acceptable for demo |
| 10 MB binary file limit | Blocked `git push` for benchmark PNGs | Git LFS migration |
| 16 GB RAM, shared | Limits ControlNet + LoRA caching | 2-entry LRU eviction |
| Cold start ~30 s | Bad first impression | "Always on" requires paid tier |
| No TensorRT | 3–5× slower than optimised builds | Not possible on free tier |
| XetHub binary requirement | `git push` fails for any PNG | Worked around with `huggingface_hub.upload_folder` |

---

## Future Improvements

Realistic next steps if I were to invest more in this:

| Improvement | Effort | Gain |
|---|---|---|
| LCM / SDXL Turbo | 1 day | 4–8× faster inference, 1–4 step generation |
| 4-bit quantisation (GGUF/GPTQ) | 1 day | Halve VRAM, enable larger batch |
| Train LoRA on cleaner data | 2 days | Reduce calligraphy artifact, broader style range |
| Multi-LoRA composition | 1 day | Blend multiple style adapters at inference time |
| DreamBooth for subject personalisation | 3 days | "Generate images of [specific person/object]" |
| Paid GPU Space (A10G) | $/hour | 10× speedup, production-viable latency |
| TensorRT compilation | 1 week | 3–5× speedup on equivalent hardware |

What I would **not** do:
- Train from scratch — compute-prohibitive without 8× A100s and weeks of time
- Reimplement diffusion sampling math — it's well-solved; the engineering value is elsewhere

---

## Citations

- Rombach et al. — [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) (CVPR 2022)
- Hu et al. — [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (ICLR 2022)
- Zhang, Rao, Agrawala — [Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) (ControlNet, ICCV 2023)
- Lu et al. — [DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models](https://arxiv.org/abs/2211.01095) (NeurIPS 2022)
- Yu et al. — [Scaling Autoregressive Models for Content-Rich Text-to-Image Generation](https://arxiv.org/abs/2206.10789) (PartiPrompts, TMLR 2022)
- Ho, Jain, Abbeel — [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (NeurIPS 2020)
- Radford et al. — [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) (CLIP, ICML 2021)

```bibtex
@inproceedings{rombach2022latent,
  title     = {High-Resolution Image Synthesis with Latent Diffusion Models},
  author    = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj{\"o}rn},
  booktitle = {CVPR},
  year      = {2022}
}
@article{hu2021lora,
  title   = {LoRA: Low-Rank Adaptation of Large Language Models},
  author  = {Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  journal = {ICLR},
  year    = {2022}
}
@article{zhang2023controlnet,
  title   = {Adding Conditional Control to Text-to-Image Diffusion Models},
  author  = {Zhang, Lvmin and Rao, Anyi and Agrawala, Maneesh},
  journal = {ICCV},
  year    = {2023}
}
```

---

## Acknowledgments

- SD 2.1 weights (community mirror): [sd2-community/stable-diffusion-2-1](https://huggingface.co/sd2-community/stable-diffusion-2-1)
- ControlNet checkpoints: [thibaud's SD2.1 ControlNet collection](https://huggingface.co/thibaud)
- WikiArt training data: [huggan/wikiart](https://huggingface.co/datasets/huggan/wikiart)
- Hugging Face [diffusers](https://github.com/huggingface/diffusers) library and training scripts
