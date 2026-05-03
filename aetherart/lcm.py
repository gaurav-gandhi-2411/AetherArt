"""LCM (Latent Consistency Model) scheduler helpers for fast generation.

Note: No LCM LoRA checkpoint exists for Stable Diffusion 2.1. The official
latent-consistency/lcm-lora-* LoRAs only cover SD 1.5 and SDXL. This module
implements LCMScheduler-only fast generation: same SD 2.1 weights, fewer
denoising steps (4 vs 30). Quality is reduced more than with LCM-LoRA but
the 7x speedup is real and measurable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from diffusers import StableDiffusionPipeline

LCM_STEPS = 4
LCM_GUIDANCE = 1.5


def apply_lcm_mode(pipeline: "StableDiffusionPipeline") -> None:
    """Swap pipeline scheduler to LCMScheduler in-place."""
    from diffusers import LCMScheduler

    pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)


def restore_standard_mode(pipeline: "StableDiffusionPipeline") -> None:
    """Restore pipeline to DPM-Solver++ scheduler in-place."""
    from diffusers import DPMSolverMultistepScheduler

    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)


def is_lcm_scheduler(pipeline: "StableDiffusionPipeline") -> bool:
    from diffusers import LCMScheduler

    return isinstance(pipeline.scheduler, LCMScheduler)
