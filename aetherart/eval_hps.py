"""HPSv2.1 scoring wrapper — lazy-loaded CLIP-H backbone, grouped by prompt.

The hpsv2 package caches the ViT-H model architecture in a module-level dict
after the first call; subsequent calls reuse the architecture but reload the
checkpoint from disk on each invocation (hpsv2 design). We cache the checkpoint
download path so the HuggingFace Hub check is a local cache hit after the first
call (~1 ms, not a network round-trip).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

from aetherart.logger import get_logger

logger = get_logger(__name__)

_checkpoint_path: str | None = None


def _ensure_checkpoint(hps_version: str) -> str:
    global _checkpoint_path
    if _checkpoint_path is None:
        import huggingface_hub
        from hpsv2.utils import hps_version_map

        logger.info("Downloading HPSv2.1 checkpoint on first call...")
        _checkpoint_path = huggingface_hub.hf_hub_download(
            "xswu/HPSv2", hps_version_map[hps_version]
        )
        logger.info("HPSv2.1 checkpoint cached at %s", _checkpoint_path)
    return _checkpoint_path


def score_hps(
    images: list[Image.Image],
    prompts: list[str],
    hps_version: str = "v2.1",
) -> list[float]:
    """Return HPSv2.1 scores for each (image, prompt) pair.

    Groups images by prompt so same-prompt batches share one score() call,
    reducing the number of checkpoint reloads from N to #unique_prompts.
    """
    import hpsv2

    if len(images) != len(prompts):
        raise ValueError(
            f"images and prompts must have equal length ({len(images)} vs {len(prompts)})"
        )

    cp = _ensure_checkpoint(hps_version)

    # Group indices by prompt to batch same-prompt images together.
    prompt_to_indices: dict[str, list[int]] = {}
    for i, p in enumerate(prompts):
        prompt_to_indices.setdefault(p, []).append(i)

    scores: list[float | None] = [None] * len(images)
    for prompt, indices in prompt_to_indices.items():
        batch = [images[i] for i in indices]
        raw = hpsv2.score(batch, prompt, cp=cp, hps_version=hps_version)
        for idx, s in zip(indices, raw):
            scores[idx] = float(s)

    return scores  # type: ignore[return-value]


def release_hps() -> None:
    """Clear the cached ViT-H model from VRAM/RAM."""
    global _checkpoint_path
    import gc

    import torch
    from hpsv2 import img_score

    img_score.model_dict.clear()
    _checkpoint_path = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("HPSv2.1 model released.")
