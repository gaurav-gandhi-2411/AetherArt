"""
Lazy-loading CLIP scorer using openai/clip-vit-base-patch32.

Cosine similarity between image and text embeddings, range roughly [-1, 1].
Typical good generations score 0.20–0.35.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from PIL import Image

_MODEL_ID = "openai/clip-vit-base-patch32"
# Pins the CLIP scorer's own weights so a CI quality gate's threshold comparison isn't silently
# invalidated by the scorer drifting to a new upstream revision between baseline and candidate.
_MODEL_REVISION = os.environ.get("CLIP_MODEL_REVISION")
_model = None
_processor = None
_device: str | None = None


def _load() -> tuple:
    global _model, _processor, _device
    if _model is None:
        from transformers import CLIPModel, CLIPProcessor

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        # "main" is HF's default branch name — passing it explicitly when _MODEL_REVISION is
        # unset is equivalent to omitting revision=, while keeping the type plain `str` for mypy.
        revision = _MODEL_REVISION or "main"
        _model = CLIPModel.from_pretrained(_MODEL_ID, use_safetensors=True, revision=revision).to(
            _device
        )
        _processor = CLIPProcessor.from_pretrained(_MODEL_ID, revision=revision)
        _model.eval()
    return _model, _processor, _device


def score(image: Image.Image, prompt: str) -> float:
    """Return cosine similarity between CLIP image embedding and prompt embedding."""
    model, processor, device = _load()
    with torch.no_grad():
        inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True).to(
            device
        )
        img_emb = model.get_image_features(pixel_values=inputs["pixel_values"])
        txt_emb = model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        return float((img_emb * txt_emb).sum())


def score_batch(images: list[Image.Image], prompts: list[str]) -> list[float]:
    """Score a list of (image, prompt) pairs in one forward pass."""
    model, processor, device = _load()
    with torch.no_grad():
        inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True).to(
            device
        )
        img_emb = model.get_image_features(pixel_values=inputs["pixel_values"])
        txt_emb = model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        return (img_emb * txt_emb).sum(dim=-1).tolist()
