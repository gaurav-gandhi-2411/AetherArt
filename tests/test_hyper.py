"""Tests for aetherart/hyper.py — all mocked, no GPU or network calls."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherart.hyper import (
    HYPER_DEFAULTS,
    is_hyper_active,
    load_hyper_lora,
    unload_hyper_lora,
)


class TestHyperDefaults:
    def test_hyper_defaults_4step_cfg_free(self):
        d = HYPER_DEFAULTS["4step"]
        assert d["num_inference_steps"] == 4
        assert d["guidance_scale"] == 0.0
        assert d["supports_negative_prompt"] is False

    def test_hyper_defaults_8step_cfg_preserved(self):
        d = HYPER_DEFAULTS["8step"]
        assert d["num_inference_steps"] == 8
        assert d["guidance_scale"] > 0
        assert d["supports_negative_prompt"] is True


class TestLoadHyperLora:
    def _mock_pipe(self) -> MagicMock:
        pipe = MagicMock()
        pipe._aetherart_hyper_variant = None
        pipe.scheduler = MagicMock()
        return pipe

    def _mock_euler(self) -> tuple[MagicMock, MagicMock]:
        mock_euler_cls = MagicMock()
        mock_euler_instance = MagicMock()
        mock_euler_cls.from_config.return_value = mock_euler_instance
        return mock_euler_cls, mock_euler_instance

    def test_load_hyper_lora_4step_swaps_to_euler_trailing(self):
        pipe = self._mock_pipe()
        original_scheduler = pipe.scheduler
        mock_euler_cls, mock_euler_instance = self._mock_euler()

        with patch("aetherart.hyper.EulerDiscreteScheduler", mock_euler_cls):
            load_hyper_lora(pipe, "4step")

        mock_euler_cls.from_config.assert_called_once_with(
            original_scheduler.config,
            timestep_spacing="trailing",
        )
        assert pipe.scheduler is mock_euler_instance
        assert pipe._aetherart_prev_scheduler is original_scheduler
        assert pipe._aetherart_hyper_variant == "4step"

    def test_load_hyper_lora_8step_swaps_to_euler_trailing(self):
        pipe = self._mock_pipe()
        original_scheduler = pipe.scheduler
        mock_euler_cls, mock_euler_instance = self._mock_euler()

        with patch("aetherart.hyper.EulerDiscreteScheduler", mock_euler_cls):
            load_hyper_lora(pipe, "8step")

        mock_euler_cls.from_config.assert_called_once_with(
            original_scheduler.config,
            timestep_spacing="trailing",
        )
        assert pipe.scheduler is mock_euler_instance
        assert pipe._aetherart_hyper_variant == "8step"

    def test_load_hyper_lora_invalid_variant_raises(self):
        pipe = self._mock_pipe()
        with pytest.raises(ValueError, match="variant must be"):
            load_hyper_lora(pipe, "16step")

    def test_load_hyper_lora_replaces_existing_variant(self):
        pipe = self._mock_pipe()
        pipe._aetherart_hyper_variant = "4step"
        pipe._aetherart_prev_scheduler = MagicMock()
        mock_euler_cls, _ = self._mock_euler()

        with patch("aetherart.hyper.EulerDiscreteScheduler", mock_euler_cls):
            load_hyper_lora(pipe, "8step")

        pipe.delete_adapters.assert_called_once_with(["hyper_4step"])
        pipe.load_lora_weights.assert_called_once()
        assert pipe.load_lora_weights.call_args.kwargs["adapter_name"] == "hyper_8step"
        assert pipe._aetherart_hyper_variant == "8step"


class TestUnloadHyperLora:
    def test_unload_hyper_lora_restores_previous_scheduler(self):
        pipe = MagicMock()
        pipe._aetherart_hyper_variant = "8step"
        prev_scheduler = MagicMock()
        pipe._aetherart_prev_scheduler = prev_scheduler
        pipe.scheduler = MagicMock()

        unload_hyper_lora(pipe)

        assert pipe.scheduler is prev_scheduler
        assert pipe._aetherart_hyper_variant is None
        assert pipe._aetherart_prev_scheduler is None

    def test_unload_hyper_lora_idempotent_when_not_active(self):
        pipe = MagicMock()
        pipe._aetherart_hyper_variant = None

        unload_hyper_lora(pipe)

        pipe.delete_adapters.assert_not_called()


class TestIsHyperActive:
    def test_is_hyper_active_returns_variant_when_loaded(self):
        pipe = MagicMock()
        pipe._aetherart_hyper_variant = "4step"
        assert is_hyper_active(pipe) == "4step"

    def test_is_hyper_active_returns_none_when_unloaded(self):
        pipe = MagicMock()
        pipe._aetherart_hyper_variant = None
        assert is_hyper_active(pipe) is None
