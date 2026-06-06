"""SDXL base pipeline loader for AetherArt.

PR 03 default scheduler is DPMSolverMultistep. Hyper-SDXL-8step requires
EulerDiscreteScheduler; the swap is handled in PR 05's load_hyper_lora().
PR 03 does NOT implement that swap.
"""

from __future__ import annotations

import gc

from .config import cfg
from .logger import get_logger
from .utils import preferred_dtype_kwarg

logger = get_logger(__name__)

try:
    from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, StableDiffusionXLPipeline
except Exception:  # pragma: no cover
    AutoencoderKL = None  # type: ignore[assignment, misc]
    DPMSolverMultistepScheduler = None  # type: ignore[assignment, misc]
    StableDiffusionXLPipeline = None  # type: ignore[assignment, misc]


def load_sdxl_base() -> StableDiffusionXLPipeline:
    """Load SDXL base pipeline with fp16-fix VAE and DPM-Solver++ scheduler.

    G1 (mandatory): VAE is constructed first from cfg.sdxl_vae_fix to avoid
    the NaN/black-image bug that occurs with the default SDXL VAE at fp16.
    """
    import torch

    if AutoencoderKL is None or StableDiffusionXLPipeline is None:
        raise RuntimeError("diffusers is not installed; cannot load SDXL pipeline")

    # preferred_dtype_kwarg() picks 'torch_dtype' or 'dtype' based on the
    # diffusers version — diffusers ≥ 0.36 renamed the param to 'dtype'.
    dtype_kw = preferred_dtype_kwarg(AutoencoderKL.from_pretrained) or "torch_dtype"
    logger.info("Loading fp16-fix VAE from '%s'…", cfg.sdxl_vae_fix)
    vae = AutoencoderKL.from_pretrained(
        cfg.sdxl_vae_fix,
        **{dtype_kw: torch.float16},
    )

    pipe_dtype_kw = (
        preferred_dtype_kwarg(StableDiffusionXLPipeline.from_pretrained) or "torch_dtype"
    )
    logger.info("Loading SDXL base pipeline from '%s'…", cfg.sdxl_model)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        cfg.sdxl_model,
        vae=vae,
        **{pipe_dtype_kw: torch.float16},
        variant="fp16",
        use_safetensors=True,
    )

    # When loading LoRA weights into SDXL, diffusers may emit a warning like
    # "Some weights of CLIPTextModel were not initialized…expected not noise".
    # This is benign — it fires because the LoRA only covers UNet keys and
    # CLIPTextModel is intentionally left at its pre-trained weights.
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


def release_sdxl_pipeline(pipe: StableDiffusionXLPipeline) -> None:
    """Delete a pipeline and reclaim GPU memory."""
    del pipe
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
