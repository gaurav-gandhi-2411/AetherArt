"""SDXL base pipeline loader for AetherArt.

PR 03 default scheduler is DPMSolverMultistep. Hyper-SDXL-8step requires
EulerDiscreteScheduler; the swap is handled in PR 05's load_hyper_lora().
PR 03 does NOT implement that swap.
"""

from __future__ import annotations

import gc

from .config import cfg
from .logger import get_logger

logger = get_logger(__name__)

try:
    from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, StableDiffusionXLPipeline
except Exception:  # pragma: no cover
    AutoencoderKL = None  # type: ignore[assignment, misc]
    DPMSolverMultistepScheduler = None  # type: ignore[assignment, misc]
    StableDiffusionXLPipeline = None  # type: ignore[assignment, misc]


def load_sdxl_base() -> "StableDiffusionXLPipeline":
    """Load SDXL base pipeline with fp16-fix VAE and DPM-Solver++ scheduler.

    G1 (mandatory): VAE is constructed first from cfg.sdxl_vae_fix to avoid
    the NaN/black-image bug that occurs with the default SDXL VAE at fp16.
    """
    import torch

    if AutoencoderKL is None or StableDiffusionXLPipeline is None:
        raise RuntimeError("diffusers is not installed; cannot load SDXL pipeline")

    logger.info("Loading fp16-fix VAE from '%s'…", cfg.sdxl_vae_fix)
    vae = AutoencoderKL.from_pretrained(
        cfg.sdxl_vae_fix,
        torch_dtype=torch.float16,
    )

    logger.info("Loading SDXL base pipeline from '%s'…", cfg.sdxl_model)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        cfg.sdxl_model,
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    logger.info("Scheduler set to DPMSolverMultistep")

    pipe.enable_model_cpu_offload()
    logger.info("Enabled model CPU offload")

    try:
        import xformers  # noqa: F401

        pipe.enable_xformers_memory_efficient_attention()
        logger.info("Enabled xformers memory-efficient attention")
    except ImportError:
        pass

    logger.info("SDXL base pipeline ready")
    return pipe


def release_sdxl_pipeline(pipe: "StableDiffusionXLPipeline") -> None:
    """Delete a pipeline and reclaim GPU memory."""
    del pipe
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
