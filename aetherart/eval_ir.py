"""ImageReward v1.0 scoring wrapper — lazy-loaded BLIP backbone.

ImageReward's vendored BLIP was written against transformers<4.39 and imports
apply_chunking_to_forward from transformers.modeling_utils, which was moved to
transformers.pytorch_utils in newer versions. The shim below re-injects it at
module load time so the lazy import inside _load() succeeds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import transformers.modeling_utils as _tmu

if not hasattr(_tmu, "apply_chunking_to_forward"):
    # transformers>=4.39 moved this to pytorch_utils; re-inject for ImageReward's BLIP.
    from transformers.pytorch_utils import apply_chunking_to_forward as _acf

    _tmu.apply_chunking_to_forward = _acf  # type: ignore[attr-defined]

if TYPE_CHECKING:
    from PIL import Image

from aetherart.logger import get_logger

logger = get_logger(__name__)

_model = None
_device: str | None = None


def _load() -> tuple:
    global _model, _device
    if _model is None:
        import ImageReward as ir
        import torch

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading ImageReward-v1.0 on %s...", _device)
        _model = ir.load("ImageReward-v1.0", device=_device)
        logger.info("ImageReward-v1.0 loaded.")
    return _model, _device


def score_image_reward(
    images: list[Image.Image],
    prompts: list[str],
) -> list[float]:
    """Return ImageReward scores for each (image, prompt) pair.

    ImageReward.score(prompt, image) -> float. Scored individually;
    BLIP is fast enough that batching adds minimal benefit.
    """
    if len(images) != len(prompts):
        raise ValueError(
            f"images and prompts must have equal length ({len(images)} vs {len(prompts)})"
        )

    model, _ = _load()
    scores = []
    for img, prompt in zip(images, prompts):
        s = model.score(prompt, img)
        scores.append(float(s))
    return scores


def release_image_reward() -> None:
    """Clear the cached BLIP model from VRAM/RAM."""
    global _model, _device
    import gc

    import torch

    _model = None
    _device = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("ImageReward model released.")
