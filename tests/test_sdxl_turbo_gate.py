"""Tests for SDXL Turbo license gate (AETHERART_ENABLE_LEGACY=1).

No model weights are downloaded — all pipeline loads are mocked at the
diffusers boundary so CI never fetches the ~6.7 GB Turbo weights.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from aetherart.registry import ModelRegistry
from aetherart.sdxl_turbo import _assert_legacy_enabled, load_turbo_pipeline


class TestTurboGate:
    def test_load_turbo_raises_without_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """load_turbo_pipeline() raises RuntimeError when AETHERART_ENABLE_LEGACY is unset."""
        monkeypatch.delenv("AETHERART_ENABLE_LEGACY", raising=False)
        with pytest.raises(RuntimeError, match="AETHERART_ENABLE_LEGACY"):
            load_turbo_pipeline()

    def test_load_turbo_raises_with_wrong_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Gate requires the value to be exactly '1', not any truthy string."""
        monkeypatch.setenv("AETHERART_ENABLE_LEGACY", "true")
        with pytest.raises(RuntimeError, match="AETHERART_ENABLE_LEGACY"):
            load_turbo_pipeline()

    def test_gate_passes_when_legacy_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_assert_legacy_enabled() does not raise when AETHERART_ENABLE_LEGACY=1."""
        monkeypatch.setenv("AETHERART_ENABLE_LEGACY", "1")
        _assert_legacy_enabled()  # must not raise

    def test_load_turbo_proceeds_with_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """load_turbo_pipeline() passes the gate and loads when AETHERART_ENABLE_LEGACY=1.

        diffusers and torch.cuda are mocked so no weights are downloaded.
        """
        monkeypatch.setenv("AETHERART_ENABLE_LEGACY", "1")

        mock_pipe = MagicMock()
        mock_diffusers = MagicMock()
        mock_diffusers.AutoPipelineForText2Image.from_pretrained.return_value = mock_pipe

        with (
            patch.dict(sys.modules, {"diffusers": mock_diffusers}),
            patch("torch.cuda.is_available", return_value=False),
        ):
            result = load_turbo_pipeline()

        assert result is mock_pipe

    def test_registry_health_reports_gated_state(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """health()['turbo'] reports 'gated' when AETHERART_ENABLE_LEGACY is not set."""
        monkeypatch.delenv("AETHERART_ENABLE_LEGACY", raising=False)
        r = ModelRegistry()
        h = r.health()
        assert "gated" in h["turbo"]

    def test_registry_health_reports_not_loaded_when_legacy_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """health()['turbo'] reports 'not_loaded' (not 'gated') when env var is set."""
        monkeypatch.setenv("AETHERART_ENABLE_LEGACY", "1")
        r = ModelRegistry()
        h = r.health()
        assert h["turbo"] == "not_loaded"

    def test_registry_health_does_not_trigger_load(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """health() must not call load_turbo_pipeline() — it only inspects state."""
        monkeypatch.delenv("AETHERART_ENABLE_LEGACY", raising=False)
        r = ModelRegistry()
        # If health() tried to load Turbo, it would raise RuntimeError (no env var).
        # The fact that it returns cleanly proves it doesn't touch the load path.
        h = r.health()
        assert "turbo" in h
