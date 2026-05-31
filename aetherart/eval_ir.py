"""ImageReward v1.0 scoring wrapper — lazy-loaded BLIP backbone.

ImageReward's vendored BLIP was written against transformers<4.39 and imports
several helpers from transformers.modeling_utils that were moved to
transformers.pytorch_utils in newer versions. The shim below re-injects them at
module load time so the lazy import inside _load() succeeds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import transformers.modeling_utils as _tmu
from transformers.pytorch_utils import (
    apply_chunking_to_forward as _acf,
)
from transformers.pytorch_utils import (
    find_pruneable_heads_and_indices as _fph,
)
from transformers.pytorch_utils import (
    prune_linear_layer as _pll,
)

from aetherart.logger import get_logger

# transformers>=4.39 moved these helpers to pytorch_utils; re-inject them into
# modeling_utils for ImageReward's vendored BLIP which imports from the old location.
for _name, _fn in (
    ("apply_chunking_to_forward", _acf),
    ("find_pruneable_heads_and_indices", _fph),
    ("prune_linear_layer", _pll),
):
    if not hasattr(_tmu, _name):
        setattr(_tmu, _name, _fn)

if TYPE_CHECKING:
    from PIL import Image

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
