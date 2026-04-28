from __future__ import annotations
from collections import OrderedDict
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
import torch

from .config import cfg
from .logger import get_logger

logger = get_logger(__name__)

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    _CV2_AVAILABLE = False

try:
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
    _DIFFUSERS_CN_AVAILABLE = True
except ImportError:
    ControlNetModel = None
    StableDiffusionControlNetPipeline = None
    _DIFFUSERS_CN_AVAILABLE = False

try:
    from transformers import pipeline as hf_pipeline
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    hf_pipeline = None
    _TRANSFORMERS_AVAILABLE = False

CANNY_MODEL_ID = "thibaud/controlnet-sd21-canny-diffusers"
DEPTH_MODEL_ID = "thibaud/controlnet-sd21-depth-diffusers"
DEPTH_ESTIMATOR_ID = "Intel/dpt-hybrid-midas"

# LRU cache keyed by (ctype, lora_name, lora_alpha). Max 2 entries to avoid OOM.
_MAX_CN_CACHE = 2
_cn_pipelines: OrderedDict = OrderedDict()
_depth_estimator = None


def _make_cache_key(
    ctype: str, lora_name: str | None, lora_alpha: float
) -> tuple:
    """Return a hashable cache key, normalising lora_alpha to 2 decimal places."""
    return (ctype, lora_name or "none", round(lora_alpha, 2))


def _get_dtype() -> torch.dtype:
    return torch.float16 if torch.cuda.is_available() else torch.float32


def _load_depth_estimator():
    global _depth_estimator
    if _depth_estimator is None:
        logger.info("Loading depth estimator: %s", DEPTH_ESTIMATOR_ID)
        _depth_estimator = hf_pipeline("depth-estimation", model=DEPTH_ESTIMATOR_ID)
        logger.info("Depth estimator loaded")
    return _depth_estimator


def preprocess_canny(
    image: Image.Image,
    low_threshold: int = 100,
    high_threshold: int = 200,
) -> Image.Image:
    """Produce a 3-channel Canny edge map from a PIL image."""
    if not _CV2_AVAILABLE:
        raise RuntimeError(
            "opencv-python is required for Canny preprocessing. "
            "Run: pip install opencv-python"
        )
    img_rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, int(low_threshold), int(high_threshold))
    return Image.fromarray(np.stack([edges] * 3, axis=-1).astype(np.uint8))


def preprocess_depth(image: Image.Image) -> Image.Image:
    """Produce an RGB depth map from a PIL image using DPT-Hybrid-MiDaS."""
    if not _TRANSFORMERS_AVAILABLE:
        raise RuntimeError("transformers is required for depth preprocessing")
    estimator = _load_depth_estimator()
    depth: Image.Image = estimator(image)["depth"]
    return depth.resize(image.size, Image.LANCZOS).convert("RGB")


def preprocess(
    image: Image.Image,
    ctype: Literal["canny", "depth"],
    canny_low: int = 100,
    canny_high: int = 200,
) -> Image.Image:
    """Dispatch to the correct preprocessor."""
    if ctype == "canny":
        return preprocess_canny(image, canny_low, canny_high)
    if ctype == "depth":
        return preprocess_depth(image)
    raise ValueError(f"Unknown conditioning type: {ctype!r}")


def get_pipeline(
    ctype: Literal["canny", "depth"],
    lora_name: str | None = None,
    lora_alpha: float = 1.0,
) -> "StableDiffusionControlNetPipeline":
    """
    Return a cached StableDiffusionControlNetPipeline for the given conditioning
    type, optionally with a LoRA adapter loaded. Uses an LRU cache (max 2 entries)
    keyed by (ctype, lora_name, lora_alpha) to avoid reloading unchanged combos.
    """
    if not _DIFFUSERS_CN_AVAILABLE:
        raise RuntimeError(
            "diffusers ControlNet classes not available; "
            "upgrade diffusers: pip install -U diffusers"
        )

    cache_key = _make_cache_key(ctype, lora_name, lora_alpha)

    if cache_key in _cn_pipelines:
        _cn_pipelines.move_to_end(cache_key)
        return _cn_pipelines[cache_key]

    cn_model_id = CANNY_MODEL_ID if ctype == "canny" else DEPTH_MODEL_ID
    dtype = _get_dtype()

    logger.info("Loading ControlNet weights: %s", cn_model_id)
    controlnet = ControlNetModel.from_pretrained(cn_model_id, torch_dtype=dtype)

    logger.info(
        "Building StableDiffusionControlNetPipeline (%s + %s, lora=%s)",
        cfg.default_model, ctype, lora_name or "none",
    )
    token_kwarg = {"token": cfg.hf_token} if cfg.hf_token else {}
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        cfg.default_model,
        controlnet=controlnet,
        torch_dtype=dtype,
        **token_kwarg,
    )

    # Load LoRA weights into this pipeline when requested
    if lora_name and lora_name != "none":
        from aetherart.lora import LORA_REGISTRY  # deferred to avoid circular import
        config = LORA_REGISTRY.get(lora_name)
        if config:
            lora_path = Path(config["path"])
            logger.info("Loading LoRA '%s' (alpha=%.2f) into ControlNet pipeline", lora_name, lora_alpha)
            pipe.load_lora_weights(str(lora_path.parent), weight_name=lora_path.name)
            try:
                pipe.set_adapters(["default"], adapter_weights=[lora_alpha])
            except Exception:
                pass
        else:
            logger.warning("LoRA '%s' not found in registry — skipping", lora_name)

    if torch.cuda.is_available():
        try:
            pipe.enable_model_cpu_offload()
            logger.info("ControlNet pipeline (%s, lora=%s): model CPU offload enabled", ctype, lora_name or "none")
        except Exception as exc:
            logger.debug("CPU offload failed; moving to cuda directly: %s", exc)
            pipe = pipe.to("cuda")

    # LRU eviction: drop oldest entry when cache is at capacity
    while len(_cn_pipelines) >= _MAX_CN_CACHE:
        evicted_key, _ = _cn_pipelines.popitem(last=False)
        logger.info("ControlNet cache eviction (LRU): %s", evicted_key)

    _cn_pipelines[cache_key] = pipe
    logger.info("ControlNet pipeline ready: %s (lora=%s)", ctype, lora_name or "none")
    return pipe


def invalidate_cache() -> None:
    """Discard all cached ControlNet pipelines."""
    _cn_pipelines.clear()
    logger.info("ControlNet pipeline cache cleared")
