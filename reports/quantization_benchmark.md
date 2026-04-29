# Quantization Benchmark — SD 2.1 U-Net

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