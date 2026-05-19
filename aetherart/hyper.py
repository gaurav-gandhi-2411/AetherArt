"""Hyper-SD LoRA loading and scheduler management for SDXL pipelines."""

from __future__ import annotations

from typing import Any, Literal

from .config import cfg
from .logger import get_logger

logger = get_logger(__name__)

try:
    from diffusers import EulerDiscreteScheduler
except Exception:
    EulerDiscreteScheduler = None  # type: ignore[assignment, misc]

HYPER_DEFAULTS: dict[str, dict[str, Any]] = {
    "4step": {
        "num_inference_steps": 4,
        "guidance_scale": 0.0,
        "lora_scale": 1.0,
        "supports_negative_prompt": False,
    },
    "8step": {
        "num_inference_steps": 8,
        "guidance_scale": 5.0,
        "lora_scale": 1.0,
        "supports_negative_prompt": True,
    },
}

_VALID_VARIANTS = frozenset({"4step", "8step"})


def load_hyper_lora(pipe: Any, variant: Literal["4step", "8step"]) -> None:
    """Load a Hyper-SD LoRA onto pipe, swapping to EulerDiscreteScheduler."""
    if variant not in _VALID_VARIANTS:
        raise ValueError(f"variant must be one of {sorted(_VALID_VARIANTS)}, got {variant!r}")

    current = is_hyper_active(pipe)
    if current is not None and current != variant:
        logger.info("Hyper LoRA '%s' active; replacing with '%s'", current, variant)
        unload_hyper_lora(pipe)

    adapter_name = f"hyper_{variant}"
    logger.info("Loading Hyper-SD '%s' from '%s'...", variant, cfg.hyper_sd_repo)
    pipe.load_lora_weights(
        cfg.hyper_sd_repo,
        weight_name=cfg.hyper_sd_weights[variant],
        adapter_name=adapter_name,
    )
    pipe.set_adapters([adapter_name], adapter_weights=[1.0])

    if EulerDiscreteScheduler is None:
        raise RuntimeError("diffusers is not installed; cannot swap scheduler")
    logger.info("Swapping scheduler to EulerDiscrete (timestep_spacing=trailing)")
    pipe._aetherart_prev_scheduler = pipe.scheduler
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing",
    )
    pipe._aetherart_hyper_variant = variant
    logger.info("Hyper-SD '%s' ready", variant)


def unload_hyper_lora(pipe: Any) -> None:
    """Unload the active Hyper-SD LoRA and restore the previous scheduler. No-op if not active."""
    variant = is_hyper_active(pipe)
    if variant is None:
        return

    adapter_name = f"hyper_{variant}"
    logger.info("Unloading Hyper-SD '%s'...", variant)
    pipe.delete_adapters([adapter_name])

    prev_scheduler = getattr(pipe, "_aetherart_prev_scheduler", None)
    if prev_scheduler is not None:
        pipe.scheduler = prev_scheduler
        logger.info("Restored previous scheduler")

    pipe._aetherart_hyper_variant = None
    pipe._aetherart_prev_scheduler = None
    logger.info("Hyper-SD LoRA unloaded")


def is_hyper_active(pipe: Any) -> Literal["4step", "8step"] | None:
    """Return the active Hyper-SD variant, or None if no Hyper LoRA is loaded."""
    return getattr(pipe, "_aetherart_hyper_variant", None)
