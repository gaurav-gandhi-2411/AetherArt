"""SDXL ControlNet Union pipeline (PATH 1: xinsir/controlnet-union-sdxl-1.0).

ControlNetUnionModel is loaded once and shared across all pipeline instances in the
LRU-2 registry cache.  The 2.5 GB checkpoint does not reload between LRU evictions —
only the pipeline wrapper objects cycle in/out.

control_mode (diffusers 0.35 param name; formerly control_type in older releases) is
passed at inference time, not construction.  One ControlNetUnionModel checkpoint
serves all control types.

G1: VAE is always madebyollin/sdxl-vae-fp16-fix at fp16.
enable_model_cpu_offload() only — sequential offload breaks bitsandbytes (#10800).
"""

from __future__ import annotations

import gc
from typing import Any, Literal

import torch

from .config import cfg
from .logger import get_logger

logger = get_logger(__name__)

try:
    from diffusers import (
        AutoencoderKL,
        StableDiffusionXLControlNetUnionPipeline,
    )
    from diffusers.models import ControlNetUnionModel
except Exception:
    AutoencoderKL = None  # type: ignore[assignment, misc]
    ControlNetUnionModel = None  # type: ignore[misc]
    StableDiffusionXLControlNetUnionPipeline = None  # type: ignore[assignment, misc]

# Shared singleton — one model object, referenced by every pipeline in the LRU cache.
_controlnet_union_model: Any = None


def _get_or_load_controlnet_union() -> Any:
    """Return the shared ControlNetUnionModel singleton, loading it once if needed."""
    global _controlnet_union_model
    if _controlnet_union_model is None:
        if ControlNetUnionModel is None:
            raise RuntimeError("diffusers is not installed; cannot load ControlNetUnionModel")
        logger.info("Loading ControlNetUnionModel from '%s'...", cfg.sdxl_controlnet_union)
        _controlnet_union_model = ControlNetUnionModel.from_pretrained(
            cfg.sdxl_controlnet_union, torch_dtype=torch.float16
        )
        logger.info("ControlNetUnionModel ready")
    return _controlnet_union_model


def load_sdxl_controlnet_pipeline(
    lora_name: str = "none",
    lora_alpha: float = 1.0,
) -> Any:
    """Build an SDXL ControlNet Union pipeline with fp16-fix VAE (G1).

    The ControlNetUnionModel singleton is shared — this function only allocates
    a new pipeline wrapper.  Caller passes control_type at inference time.
    """
    if (
        AutoencoderKL is None
        or ControlNetUnionModel is None
        or StableDiffusionXLControlNetUnionPipeline is None
    ):
        raise RuntimeError("diffusers is not installed; cannot load SDXL ControlNet Union pipeline")

    cn_model = _get_or_load_controlnet_union()

    logger.info("Loading fp16-fix VAE from '%s'...", cfg.sdxl_vae_fix)
    vae = AutoencoderKL.from_pretrained(cfg.sdxl_vae_fix, torch_dtype=torch.float16)

    logger.info("Constructing StableDiffusionXLControlNetUnionPipeline...")
    pipe = StableDiffusionXLControlNetUnionPipeline.from_pretrained(
        cfg.sdxl_model,
        controlnet=cn_model,
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )

    # enable_model_cpu_offload only — sequential offload breaks bitsandbytes (#10800)
    pipe.enable_model_cpu_offload()
    logger.info("Enabled model CPU offload")

    if lora_name != "none":
        logger.info("Loading LoRA weights: %s (alpha=%.2f)...", lora_name, lora_alpha)
        pipe.load_lora_weights(lora_name, adapter_name="user_lora")
        pipe.set_adapters(["user_lora"], adapter_weights=[lora_alpha])

    logger.info("SDXL ControlNet Union pipeline ready")
    return pipe


def release_sdxl_controlnet_pipeline(pipe: Any) -> None:
    """Release a ControlNet pipeline wrapper and free GPU memory."""
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def release_controlnet_union_model() -> None:
    """Release the shared ControlNetUnionModel singleton."""
    global _controlnet_union_model
    if _controlnet_union_model is not None:
        del _controlnet_union_model
        _controlnet_union_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def preprocess_canny(
    image: Any,
    low_threshold: int = 100,
    high_threshold: int = 200,
) -> Any:
    """Apply Canny edge detection and return a PIL Image for ControlNet input."""
    import numpy as np
    from PIL import Image

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for Canny preprocessing") from exc

    image_np = np.array(image.convert("RGB")) if isinstance(image, Image.Image) else np.array(image)

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)


def preprocess_depth(image: Any) -> Any:
    """Estimate a depth map and return a PIL Image for ControlNet input."""
    import numpy as np
    from PIL import Image

    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except ImportError as exc:
        raise RuntimeError("transformers is required for depth preprocessing") from exc

    processor = AutoImageProcessor.from_pretrained(cfg.depth_estimator)
    model = AutoModelForDepthEstimation.from_pretrained(cfg.depth_estimator)

    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.array(image))

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    predicted_depth = outputs.predicted_depth
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )
    output = prediction.squeeze().numpy()
    output = (output - output.min()) / (output.max() - output.min() + 1e-8) * 255
    output = output.astype(np.uint8)
    depth_rgb = np.stack([output, output, output], axis=-1)
    return Image.fromarray(depth_rgb)


# Valid control-type literals understood by StableDiffusionXLControlNetUnionPipeline.
CONTROL_TYPES = Literal["canny", "depth", "hed", "pidi", "scribble", "ted", "lineart", "normal"]

# Integer IDs passed as control_mode at inference time (diffusers 0.35 API).
# Multiple string aliases share an int where the model treats them identically.
_CTYPE_TO_INT: dict[str, int] = {
    "openpose": 0,
    "depth": 1,
    "hed": 2,
    "pidi": 2,
    "scribble": 2,
    "ted": 2,
    "canny": 3,
    "lineart": 3,
    "mlsd": 3,
    "normal": 4,
    "segment": 5,
}


def generate_sdxl_controlnet(
    pipe: Any,
    prompt: str,
    image: Any,
    *,
    ctype: str = "canny",
    negative_prompt: str = "",
    conditioning_scale: float = 1.0,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 30,
    width: int = 1024,
    height: int = 1024,
    seed: int | None = None,
) -> Any:
    """Run SDXL ControlNet Union inference.

    control_mode (diffusers 0.35 API name) is resolved from the ctype string at call
    time per the Union pipeline contract — it is not baked into the pipeline at
    construction.  Verified parameter names: control_image, control_mode (not image /
    control_type — those were the pre-0.35 names and cause TypeError at runtime).

    Args:
        pipe: Pipeline returned by load_sdxl_controlnet_pipeline().
        prompt: Text prompt.
        image: Preprocessed control image (PIL Image).
        ctype: Control type string ("canny", "depth", ...).
        negative_prompt: Optional negative prompt.
        conditioning_scale: ControlNet conditioning strength (0.0–2.0).
        guidance_scale: CFG scale.
        num_inference_steps: Denoising steps.
        width: Output width in pixels.
        height: Output height in pixels.
        seed: RNG seed; None for non-deterministic.

    Returns:
        PIL Image.
    """
    if ctype not in _CTYPE_TO_INT:
        raise ValueError(f"Unknown ctype '{ctype}'. Valid options: {sorted(_CTYPE_TO_INT)}")

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    result = pipe(
        prompt=prompt,
        control_image=image,
        control_mode=[_CTYPE_TO_INT[ctype]],
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=conditioning_scale,
        generator=generator,
        width=width,
        height=height,
    )
    return result.images[0]
