"""SDXL Turbo — 1-step adversarial diffusion generation.

SDXL Turbo uses ADD (Adversarial Diffusion Distillation) to produce
images in 1 step with guidance_scale=0.0. It is a separate model from
SD 2.1 and does not support LoRA adapters or ControlNet conditioning.

VRAM: ~6-8 GB peak with fp16 + model_cpu_offload. On an 8 GB GPU,
avoid loading both SD 2.1 and Turbo simultaneously.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from diffusers import AutoPipelineForText2Image

TURBO_MODEL_ID = "stabilityai/sdxl-turbo"
TURBO_STEPS = 1
TURBO_GUIDANCE = 0.0

# Prefer a local dir download (avoids XetHub cache issues on Windows)
_LOCAL_DIR = Path(__file__).resolve().parent.parent / "models" / "sdxl-turbo"


def _model_source() -> str:
    """Return local dir if unet weights are present, else HF repo ID."""
    unet_weights = _LOCAL_DIR / "unet" / "diffusion_pytorch_model.fp16.safetensors"
    if unet_weights.exists():
        return str(_LOCAL_DIR)
    return TURBO_MODEL_ID


def load_turbo_pipeline() -> "AutoPipelineForText2Image":
    """Load SDXL Turbo with fp16 + model CPU offload."""
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
    pipe: "AutoPipelineForText2Image",
    prompt: str,
    negative_prompt: str = "",
    seed: Optional[int] = None,
    width: int = 512,
    height: int = 512,
) -> tuple:
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


def free_turbo_pipeline(pipe: "AutoPipelineForText2Image") -> None:
    """Unload pipeline and release VRAM."""
    try:
        pipe.to("cpu")
    except Exception:
        pass
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
