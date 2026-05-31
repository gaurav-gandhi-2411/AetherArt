"""Tests for aetherart/eval_hps.py — mocked, no model download, no hpsv2 install required."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest


def _make_hpsv2_mock(score_return=None):
    """Return a minimal hpsv2 module mock with a configurable score() function."""
    mock = types.ModuleType("hpsv2")
    mock.score = MagicMock(return_value=score_return or [0.28])  # type: ignore[attr-defined]
    return mock


def _get_mod():
    import importlib

    import aetherart.eval_hps as m

    importlib.reload(m)
    return m


class TestScoreHps:
    def test_returns_float_per_image(self):
        mod = _get_mod()
        fake_images = [MagicMock(), MagicMock(), MagicMock()]
        fake_prompts = ["prompt a", "prompt b", "prompt c"]

        mock_hpsv2 = _make_hpsv2_mock([0.28])
        with (
            patch.object(mod, "_ensure_checkpoint", return_value="/fake/cp.pt"),
            patch.dict(sys.modules, {"hpsv2": mock_hpsv2}),
        ):
            scores = mod.score_hps(fake_images, fake_prompts)

        assert len(scores) == 3
        for s in scores:
            assert isinstance(s, float)

    def test_raises_on_length_mismatch(self):
        mod = _get_mod()
        with pytest.raises(ValueError, match="equal length"):
            mod.score_hps([MagicMock()], ["a", "b"])

    def test_lazy_loads_once(self):
        mod = _get_mod()
        mod._checkpoint_path = None

        call_count = 0

        def fake_ensure(hps_version):
            nonlocal call_count
            call_count += 1
            return "/fake/cp.pt"

        mock_hpsv2 = _make_hpsv2_mock([0.25])
        with (
            patch.object(mod, "_ensure_checkpoint", side_effect=fake_ensure),
            patch.dict(sys.modules, {"hpsv2": mock_hpsv2}),
        ):
            mod.score_hps([MagicMock()], ["p1"])
            mod.score_hps([MagicMock()], ["p2"])

        # _ensure_checkpoint called once per score_hps call
        assert call_count == 2

    def test_groups_by_prompt(self):
        """Same-prompt images are passed in one hpsv2.score() call."""
        mod = _get_mod()
        imgs = [MagicMock(), MagicMock(), MagicMock()]
        prompts = ["same", "same", "different"]

        batch_sizes = []

        def fake_score(batch, prompt, hps_version="v2.1"):
            batch_sizes.append(len(batch) if isinstance(batch, list) else 1)
            return [0.27] * (len(batch) if isinstance(batch, list) else 1)

        mock_hpsv2 = _make_hpsv2_mock()
        mock_hpsv2.score = MagicMock(side_effect=fake_score)  # type: ignore[attr-defined]
        with (
            patch.object(mod, "_ensure_checkpoint", return_value="/fake/cp.pt"),
            patch.dict(sys.modules, {"hpsv2": mock_hpsv2}),
        ):
            scores = mod.score_hps(imgs, prompts)

        # Two unique prompts → two hpsv2.score() calls; "same" batch has size 2
        assert sorted(batch_sizes) == [1, 2]
        assert len(scores) == 3

    def test_release_clears_model(self):
        mod = _get_mod()
        mod._checkpoint_path = "/some/path"

        fake_model_dict = {"model": MagicMock(), "preprocess_val": MagicMock()}
        fake_img_score = types.ModuleType("hpsv2.img_score")
        fake_img_score.model_dict = fake_model_dict  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"hpsv2.img_score": fake_img_score}):
            mod.release_hps()

        assert mod._checkpoint_path is None
        assert len(fake_model_dict) == 0
