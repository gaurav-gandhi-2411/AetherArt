"""SDXL Turbo — 1-step adversarial diffusion generation.

SDXL Turbo uses ADD (Adversarial Diffusion Distillation) to produce
images in 1 step with guidance_scale=0.0. It is a separate model from
SD 2.1 and does not support LoRA adapters or ControlNet conditioning.

VRAM: ~6-8 GB peak with fp16 + model_cpu_offload. On an 8 GB GPU,
avoid loading both SD 2.1 and Turbo simultaneously.

LICENSE GATE: load_turbo_pipeline() requires AETHERART_ENABLE_LEGACY=1.
SDXL Turbo ships under the Stability AI SDXL Turbo Research License,
which permits non-commercial research use only. Set the env var to opt in;
this model is excluded from the commercial-intent demo by design.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from PIL import Image

import contextlib

from .logger import get_logger

logger = get_logger(__name__)

TURBO_MODEL_ID = "stabilityai/sdxl-turbo"
TURBO_STEPS = 1
TURBO_GUIDANCE = 0.0

# Stability AI SDXL Turbo Research License — non-commercial research use only.
# Authoritative terms: https://huggingface.co/stabilityai/sdxl-turbo/blob/main/LICENSE.md
TURBO_LICENSE = (
    "Stability AI SDXL Turbo Research License — non-commercial research use only. "
    "See: https://huggingface.co/stabilityai/sdxl-turbo/blob/main/LICENSE.md"
)

# Prefer a local dir download (avoids XetHub cache issues on Windows)
_LOCAL_DIR = Path(__file__).resolve().parent.parent / "models" / "sdxl-turbo"


def _assert_legacy_enabled() -> None:
    """Raise RuntimeError unless AETHERART_ENABLE_LEGACY=1 is set.

    SDXL Turbo's ADD license is non-commercial research-only, disqualifying it
    from the commercial-intent demo. Opt in explicitly for local research use.
    """
    if os.environ.get("AETHERART_ENABLE_LEGACY") != "1":
        raise RuntimeError(
            "SDXL Turbo is gated: its license permits non-commercial research use only. "
            f"({TURBO_LICENSE}) "
            "Set AETHERART_ENABLE_LEGACY=1 to opt in for research use. "
            "This model is not available in the commercial-intent demo."
        )
    logger.warning(
        "SDXL Turbo loaded with AETHERART_ENABLE_LEGACY=1. "
        "Compliance with the non-commercial research license (ADD) is the caller's responsibility. "
        "See: https://huggingface.co/stabilityai/sdxl-turbo/blob/main/LICENSE.md"
    )


def _model_source() -> str:
    """Return local dir if unet weights are present, else HF repo ID."""
    unet_weights = _LOCAL_DIR / "unet" / "diffusion_pytorch_model.fp16.safetensors"
    if unet_weights.exists():
        return str(_LOCAL_DIR)
    return TURBO_MODEL_ID


def load_turbo_pipeline() -> Any:
    """Load SDXL Turbo with fp16 + model CPU offload.

    Raises RuntimeError if AETHERART_ENABLE_LEGACY=1 is not set — see TURBO_LICENSE.
    """
    _assert_legacy_enabled()
    from diffusers import AutoPipelineForText2Image

    source = _model_source()
    pipe = AutoPipelineForText2Image.from_pretrained(
        source,
        torch_dtype=torch.float16,
        variant="fp16",
    )
    if torch.cuda.is_available():
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pipe = pipe.to("cuda")
    return pipe


def generate_turbo(
    pipe: Any,
    prompt: str,
    negative_prompt: str = "",
    seed: int | None = None,
    width: int = 512,
    height: int = 512,
) -> tuple[Image.Image, dict[str, Any]]:
    """Generate one image with SDXL Turbo (1 step, guidance=0.0).

    Returns (PIL.Image, metadata_dict).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(seed or 42)

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        num_inference_steps=TURBO_STEPS,
        guidance_scale=TURBO_GUIDANCE,
        width=width,
        height=height,
        generator=generator,
    ).images[0]

    return image, {
        "model": TURBO_MODEL_ID,
        "steps": TURBO_STEPS,
        "guidance_scale": TURBO_GUIDANCE,
        "seed": seed,
    }


def free_turbo_pipeline(pipe: Any) -> None:
    """Unload pipeline and release VRAM."""
    with contextlib.suppress(Exception):
        pipe.to("cpu")
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
