# Quantization Benchmark — SD 2.1 U-Net

> **Note:** These numbers are from an earlier isolated benchmark run with a different pipeline setup (4 prompts, warm model, single measurement pass). The canonical quantization VRAM and latency measurements for this project are in [`reports/experiments/exp1_quantization_quality/findings.md`](experiments/exp1_quantization_quality/findings.md), which used the full experiment harness (40 images, 5 seeds × 8 prompts, model CPU offload). Numbers diverge — notably fp16 VRAM (1803 MB in Exp 1 vs 3097 MB here) and latency — because measurement context differs. The divergence and what it means is documented in [`docs/lab_notebook.md`](../docs/lab_notebook.md). This file is retained as a historical artifact of the earlier measurement.

**Model**: `sd2-community/stable-diffusion-2-1`  
**Steps**: 30  **Guidance**: 7.5  **Seed**: 42  
**Prompts**: 4  **Resolution**: 512×512

| Mode | Avg latency (s/img) | Peak VRAM (MB) | CLIP score | VRAM vs fp16 |
|------|--------------------:|---------------:|:----------:|:------------|
| fp16 | 2.7 | 3097 | n/a | +0 MB |
| 8-bit INT8 | 9.6 | 2210 | n/a | -887 MB |
| 4-bit NF4 | 4.7 | 2761 | n/a | -336 MB |

## Notes

- Quantization applied to **U-Net only**; text encoder and VAE remain at fp16.
- 4-bit (NF4) enables SD 2.1 on GPUs with ≥ 4 GB VRAM.
- Latency overhead is bitsandbytes dequantization cost; varies by GPU.
- CLIP score uses `openai/clip-vit-base-patch32`; N/A if `clip` package absent.