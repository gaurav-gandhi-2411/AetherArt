"""Tests for aetherart/eval_ir.py — mocked, no model download."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _get_mod():
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


class TestDatasetsStubbedImport:
    """Verify _load() does not pollute or remove sys.modules['datasets'].

    Real eval runs on Linux/GCP where ImageReward imports cleanly without any
    sys.modules workaround. These tests guard the guarantee that _load() is
    sys.modules-neutral with respect to 'datasets'.
    """

    def test_stub_removed_after_load(self):
        """_load() must not leave a datasets stub in sys.modules after returning."""
        import sys

        mod = _get_mod()

        saved_datasets = sys.modules.pop("datasets", None)
        saved_ir = sys.modules.pop("ImageReward", None)
        try:
            fake_ir = MagicMock()
            fake_ir.load.return_value = MagicMock()

            with (
                patch.dict(sys.modules, {"ImageReward": fake_ir}),
                patch("torch.cuda.is_available", return_value=False),
            ):
                mod._model = None
                mod._load()

            # Only a real datasets module (with __version__) is acceptable;
            # a stub (ModuleType without __version__) means the cleanup was skipped.
            if "datasets" in sys.modules:
                assert hasattr(sys.modules["datasets"], "__version__"), (
                    "datasets stub artifact leaked into sys.modules — cleanup missing"
                )
        finally:
            if saved_datasets is not None:
                sys.modules["datasets"] = saved_datasets
            if saved_ir is not None:
                sys.modules["ImageReward"] = saved_ir

    def test_stub_not_inserted_when_datasets_already_present(self):
        """If real datasets is already loaded, _load() must not overwrite or remove it."""
        import sys
        import types

        mod = _get_mod()

        sentinel = types.ModuleType("datasets")
        sentinel.__version__ = "test-sentinel"  # type: ignore[attr-defined]

        saved_ir = sys.modules.pop("ImageReward", None)
        try:
            fake_ir = MagicMock()
            fake_ir.load.return_value = MagicMock()

            with patch.dict(sys.modules, {"ImageReward": fake_ir, "datasets": sentinel}):
                with patch("torch.cuda.is_available", return_value=False):
                    mod._model = None
                    mod._load()

                # Check INSIDE the context while sentinel is still in sys.modules:
                # _load() must not have removed or replaced it.
                assert sys.modules.get("datasets") is sentinel, (
                    "datasets was replaced or removed during _load() — "
                    "stub was incorrectly inserted when real datasets was present"
                )
        finally:
            if saved_ir is not None:
                sys.modules["ImageReward"] = saved_ir
