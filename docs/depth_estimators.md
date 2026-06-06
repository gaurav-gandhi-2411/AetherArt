# Depth Estimator Split — SD 2.1 vs SDXL ControlNet Paths

AetherArt uses two different depth-estimation models depending on which base model is active. This is an intentional architectural decision, not a configuration inconsistency.

## Models

| Path | Depth estimator | Module |
|------|-----------------|--------|
| SD 2.1 ControlNet depth | `Intel/dpt-hybrid-midas` | `aetherart/controlnet.py` |
| SDXL ControlNet depth | `LiheYoung/depth-anything-small-hf` | `aetherart/controlnet_sdxl.py` |

## Why two models?

**SD 2.1 path — `Intel/dpt-hybrid-midas` (frozen)**

The SD 2.1 ControlNet depth checkpoint (`thibaud/controlnet-sd21-depth-diffusers`) was trained against depth maps produced by DPT-Hybrid-MiDaS. Using a different estimator changes the depth-map distribution and degrades conditioning quality. This model ships PyTorch `.bin` weights, which is fine here because SD 2.1 ControlNet is only used locally (Windows dev, RTX 3070) and not on the GCP eval path.

**SDXL path — `LiheYoung/depth-anything-small-hf`**

Swapped in PR 08 (`e3e4e6a`). Two reasons:

1. **CVE-2025-32434 compliance.** `transformers`' security patch (released early 2025) blocks `torch.load()` on `torch < 2.6` with no `weights_only=True`. `Intel/dpt-hybrid-midas` ships only `.bin` weights, which requires `torch.load`. On the GCP evaluation VMs (PyTorch 2.9 + DLVM image), the blocked load path would cause a crash. `depth-anything-small-hf` uses `.safetensors` weights, which are not affected.

2. **Quality.** Depth Anything is a more capable estimator than DPT-Hybrid-MiDaS, producing sharper depth boundaries. The SDXL ControlNet checkpoint (`diffusers/controlnet-depth-sdxl-1.0`) is not as tightly coupled to a specific estimator as the SD 2.1 checkpoints, so switching estimators does not degrade conditioning.

## Local VRAM note (SD 2.1)

Unquantized FP16 SD 2.1 + ControlNet depth + `enable_model_cpu_offload` peaks at ~4800 MB on RTX 3070 — fits within 8 GB with margin.

## Local VRAM note (SDXL)

Unquantized FP16 SDXL + ControlNet + `enable_model_cpu_offload` peaks at 7928 MB on RTX 3070 (right at the 8 GB ceiling), running at ~275–292 s/image due to offload thrashing. The SDXL ControlNet path is suited for GCP L4 (24 GB) or an NF4-quantized local variant — not for timed local iteration.
