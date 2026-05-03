"""Tests for aetherart/clip_scorer.py — mocked CLIP model, no model download."""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import torch
from PIL import Image as PILImage

import aetherart.clip_scorer as cs


def _fake_load(n: int = 1) -> tuple:
    """Return (model, processor, device) with real tensor outputs, no network calls."""
    dim = 16
    model = MagicMock()
    processor = MagicMock()

    emb = torch.randn(n, dim)
    model.get_image_features.return_value = emb.clone()
    model.get_text_features.return_value = emb.clone()

    return model, processor, "cpu"


class TestScore:
    def test_returns_float(self):
        model, processor, device = _fake_load(1)
        image = PILImage.new("RGB", (64, 64))
        with patch.object(cs, "_load", return_value=(model, processor, device)):
            result = cs.score(image, "a glowing crystal cave")
        assert isinstance(result, float)

    def test_result_is_finite(self):
        model, processor, device = _fake_load(1)
        image = PILImage.new("RGB", (64, 64))
        with patch.object(cs, "_load", return_value=(model, processor, device)):
            result = cs.score(image, "a glowing crystal cave")
        assert math.isfinite(result)

    def test_empty_prompt_still_works(self):
        model, processor, device = _fake_load(1)
        image = PILImage.new("RGB", (32, 32))
        with patch.object(cs, "_load", return_value=(model, processor, device)):
            result = cs.score(image, "")
        assert isinstance(result, float)


class TestScoreBatch:
    def test_returns_list_of_correct_length(self):
        n = 3
        model, processor, device = _fake_load(n)
        images = [PILImage.new("RGB", (64, 64)) for _ in range(n)]
        prompts = ["prompt one", "prompt two", "prompt three"]
        with patch.object(cs, "_load", return_value=(model, processor, device)):
            results = cs.score_batch(images, prompts)
        assert isinstance(results, list)
        assert len(results) == n

    def test_all_elements_are_floats(self):
        n = 2
        model, processor, device = _fake_load(n)
        images = [PILImage.new("RGB", (64, 64)) for _ in range(n)]
        prompts = ["cat", "dog"]
        with patch.object(cs, "_load", return_value=(model, processor, device)):
            results = cs.score_batch(images, prompts)
        assert all(isinstance(r, float) for r in results)

    def test_single_item_batch(self):
        model, processor, device = _fake_load(1)
        with patch.object(cs, "_load", return_value=(model, processor, device)):
            results = cs.score_batch([PILImage.new("RGB", (64, 64))], ["one prompt"])
        assert len(results) == 1
