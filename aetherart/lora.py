"""LoRA adapter registry and load/unload helpers."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent

LORA_REGISTRY: dict[str, dict[str, Any] | None] = {
    "none": None,
    "ukiyo-e": {
        "path": str(_REPO_ROOT / "data" / "lora" / "ukiyo-e" / "ukiyo-e-lora.safetensors"),
        "trigger_token": "ukyowood",
        "default_negative": "text, watermark, calligraphy, signature, words, letters",
        "description": "Japanese woodblock print style — 80 WikiArt images, SD 2.1, rank-8",
    },
    "hyper_4step": {
        "repo": "ByteDance/Hyper-SD",
        "weight_name": "Hyper-SDXL-4steps-lora.safetensors",
        "type": "few_step",
        "step_count": 4,
        "supports_negative_prompt": False,
        "description": (
            "Hyper-SDXL 4-step CFG-free LoRA. Use with guidance_scale=0. "
            "Negative prompts are ignored. Calligraphy artifact will surface "
            "when composed with Ukiyo-e LoRA."
        ),
    },
    "hyper_8step": {
        "repo": "ByteDance/Hyper-SD",
        "weight_name": "Hyper-SDXL-8steps-lora.safetensors",
        "type": "few_step",
        "step_count": 8,
        "supports_negative_prompt": True,
        "description": (
            "Hyper-SDXL 8-step CFG-preserved LoRA. Use with guidance_scale=5-8. "
            "Compatible with negative prompts and Ukiyo-e LoRA composition "
            "(default for the demo)."
        ),
    },
}


def load_lora(pipeline: Any, lora_name: str, alpha: float = 1.0) -> None:
    """Load a LoRA adapter onto pipeline in-place. Unloads any existing adapter first."""
    _unload_safe(pipeline)
    if lora_name == "none" or lora_name not in LORA_REGISTRY:
        return
    config = LORA_REGISTRY[lora_name]
    if config is None:  # pragma: no cover — registry has no named null entries today
        return
    lora_path = Path(config["path"])
    pipeline.load_lora_weights(
        str(lora_path.parent), weight_name=lora_path.name, adapter_name="ukiyo_e"
    )
    pipeline.set_adapters(["ukiyo_e"], adapter_weights=[alpha])


def unload_lora(pipeline: Any) -> None:
    _unload_safe(pipeline)


def _unload_safe(pipeline: Any) -> None:
    with contextlib.suppress(Exception):
        pipeline.unload_lora_weights()


def get_trigger_token(lora_name: str) -> str:
    config = LORA_REGISTRY.get(lora_name)
    if not config:
        return ""
    return config.get("trigger_token", "")


def get_default_negative(lora_name: str) -> str:
    config = LORA_REGISTRY.get(lora_name)
    if not config:
        return ""
    return config.get("default_negative", "")
