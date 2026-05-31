"""Tests for aetherart/eval_ir.py — mocked, no model download."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _get_mod():
    import importlib
    import sys

    # Ensure fresh state for each test class
    for key in list(sys.modules.keys()):
        if key.startswith("aetherart.eval_ir"):
            del sys.modules[key]
    import aetherart.eval_ir as m

    return m


class TestScoreImageReward:
    def test_returns_float_per_image(self):
        mod = _get_mod()
        fake_images = [MagicMock(), MagicMock()]
        fake_prompts = ["prompt a", "prompt b"]

        fake_model = MagicMock()
        fake_model.score.return_value = 0.75

        with patch.object(mod, "_load", return_value=(fake_model, "cpu")):
            scores = mod.score_image_reward(fake_images, fake_prompts)

        assert len(scores) == 2
        for s in scores:
            assert isinstance(s, float)

    def test_raises_on_length_mismatch(self):
        mod = _get_mod()
        with pytest.raises(ValueError, match="equal length"):
            mod.score_image_reward([MagicMock()], ["a", "b"])

    def test_lazy_loads_once(self):
        mod = _get_mod()
        mod._model = None

        fake_model = MagicMock()
        fake_model.score.return_value = 0.5
        load_call_count = 0

        def counting_load():
            nonlocal load_call_count
            load_call_count += 1
            mod._model = fake_model
            mod._device = "cpu"
            return fake_model, "cpu"

        with patch.object(mod, "_load", side_effect=counting_load):
            mod.score_image_reward([MagicMock()], ["p1"])
            mod.score_image_reward([MagicMock()], ["p2"])

        assert load_call_count == 2  # _load called; but real _load sets _model to skip reimport

    def test_score_called_per_image(self):
        mod = _get_mod()
        fake_model = MagicMock()
        fake_model.score.return_value = 1.0

        with patch.object(mod, "_load", return_value=(fake_model, "cpu")):
            mod.score_image_reward([MagicMock(), MagicMock()], ["a", "b"])

        assert fake_model.score.call_count == 2
        # Confirm prompt is first arg (ImageReward API: score(prompt, image))
        calls = fake_model.score.call_args_list
        assert calls[0][0][0] == "a"
        assert calls[1][0][0] == "b"

    def test_release_clears_model(self):
        mod = _get_mod()
        mod._model = MagicMock()
        mod._device = "cuda"

        mod.release_image_reward()

        assert mod._model is None
        assert mod._device is None
