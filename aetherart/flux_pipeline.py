"""FLUX.1-schnell pipeline loader for AetherArt.

FLUX.1-schnell (black-forest-labs/FLUX.1-schnell, Apache-2.0) is a distilled, guidance-free
12B-parameter model — see aetherart.flux_pipeline's caller in scripts/model_verdict_harness.py
for the correct guidance_scale=0.0 / num_inference_steps=4 inference defaults, verified against
diffusers' own FluxPipeline docstring example rather than assumed.

Local feasibility note (measured 2026-07-30, RTX 3070 Laptop, 8.59GB VRAM): the diffusers-format
FLUX.1-schnell repo requires ~33.7GB of disk (23.78GB transformer + 9.53GB text_encoder_2/T5 +
0.25GB text_encoder + 0.17GB vae, per HfApi.model_info file listing), which exceeded this
machine's ~25.5GB free disk at measurement time — a disk-space blocker that precedes the VRAM
question entirely, so bf16+cpu-offload below is diffusers' documented low-VRAM path but has NOT
been empirically GPU-verified on this machine. Revisit with more free disk before the actual eval
run.
"""

from __future__ import annotations

import gc

from .logger import get_logger

logger = get_logger(__name__)

try:
    from diffusers import FluxPipeline
except Exception:  # pragma: no cover
    FluxPipeline = None  # type: ignore[assignment, misc]

FLUX_SCHNELL_MODEL = "black-forest-labs/FLUX.1-schnell"


def load_flux_schnell() -> FluxPipeline:
    """Load FLUX.1-schnell in bf16 with CPU offload — diffusers' documented low-VRAM path.

    This is the (a) config from the feasibility check: bf16 dtype + enable_model_cpu_offload(),
    no quantization. See module docstring for why this hasn't been empirically GPU-verified on
    this machine (disk-space blocker, not a VRAM finding).
    """
    import torch

    if FluxPipeline is None:
        raise RuntimeError("diffusers is not installed; cannot load FLUX pipeline")

    logger.info("Loading FLUX.1-schnell from '%s'...", FLUX_SCHNELL_MODEL)
    pipe = FluxPipeline.from_pretrained(FLUX_SCHNELL_MODEL, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    logger.info("Enabled model CPU offload")
    logger.info("FLUX.1-schnell pipeline ready")
    return pipe


def release_flux_pipeline(pipe: FluxPipeline) -> None:
    """Delete a pipeline and reclaim GPU memory."""
    del pipe
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
