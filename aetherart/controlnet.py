from __future__ import annotations
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

_cn_pipelines: dict[str, "StableDiffusionControlNetPipeline"] = {}
_depth_estimator = None


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


def get_pipeline(ctype: Literal["canny", "depth"]) -> "StableDiffusionControlNetPipeline":
    """
    Return a cached StableDiffusionControlNetPipeline for the given conditioning
    type. Loads SD 2.1 + the ControlNet model from disk on first call.
    """
    if not _DIFFUSERS_CN_AVAILABLE:
        raise RuntimeError(
            "diffusers ControlNet classes not available; "
            "upgrade diffusers: pip install -U diffusers"
        )

    if ctype not in _cn_pipelines:
        cn_model_id = CANNY_MODEL_ID if ctype == "canny" else DEPTH_MODEL_ID
        dtype = _get_dtype()

        logger.info("Loading ControlNet weights: %s", cn_model_id)
        controlnet = ControlNetModel.from_pretrained(cn_model_id, torch_dtype=dtype)

        logger.info("Building StableDiffusionControlNetPipeline (%s + %s)", cfg.default_model, ctype)
        token_kwarg = {"token": cfg.hf_token} if cfg.hf_token else {}
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            cfg.default_model,
            controlnet=controlnet,
            torch_dtype=dtype,
            **token_kwarg,
        )

        if torch.cuda.is_available():
            try:
                pipe.enable_model_cpu_offload()
                logger.info("ControlNet pipeline (%s): model CPU offload enabled", ctype)
            except Exception as exc:
                logger.debug("CPU offload failed (%s); moving to cuda directly: %s", ctype, exc)
                pipe = pipe.to("cuda")

        _cn_pipelines[ctype] = pipe
        logger.info("ControlNet pipeline ready: %s", ctype)

    return _cn_pipelines[ctype]


def invalidate_cache() -> None:
    """Discard cached ControlNet pipelines (call when the base model changes)."""
    _cn_pipelines.clear()
    logger.info("ControlNet pipeline cache cleared")
