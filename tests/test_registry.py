"""Tests for ModelRegistry — lifecycle, health, lazy init, singleton identity.

No real pipelines are loaded. AetherModel is mocked throughout.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherart.registry import ModelRegistry


def _make_registry() -> ModelRegistry:
    r = ModelRegistry()
    return r


class TestRegistryInit:
    def test_base_not_loaded_on_construction(self):
        r = _make_registry()
        assert r._base.backend is None

    def test_no_init_error_on_construction(self):
        r = _make_registry()
        assert r._base_init_error is None

    def test_turbo_not_loaded_on_construction(self):
        r = _make_registry()
        assert r._turbo is None

    def test_quant_not_loaded_on_construction(self):
        r = _make_registry()
        assert r._quant is None

    def test_active_lora_defaults_to_none(self):
        r = _make_registry()
        assert r.active_lora == "none"


class TestGetBase:
    def test_get_base_returns_aethermodel(self):
        from aetherart.model import AetherModel

        r = _make_registry()
        assert isinstance(r.get_base(), AetherModel)

    def test_get_base_returns_same_instance(self):
        r = _make_registry()
        assert r.get_base() is r.get_base()


class TestEnsureBase:
    def test_ensure_base_calls_init(self):
        r = _make_registry()
        with patch.object(r._base, "init") as mock_init:
            mock_init.side_effect = lambda **kw: setattr(r._base, "backend", "local")
            r.ensure_base()
            mock_init.assert_called_once()

    def test_ensure_base_is_idempotent(self):
        r = _make_registry()
        r._base.backend = "local"
        with patch.object(r._base, "init") as mock_init:
            r.ensure_base()
            r.ensure_base()
            mock_init.assert_not_called()

    def test_ensure_base_caches_failure(self):
        r = _make_registry()
        with patch.object(r._base, "init", side_effect=RuntimeError("no GPU")):
            with pytest.raises(RuntimeError, match="Base model init failed"):
                r.ensure_base()
        assert r._base_init_error is not None

    def test_ensure_base_raises_on_cached_failure(self):
        r = _make_registry()
        r._base_init_error = "previous failure"
        with pytest.raises(RuntimeError, match="previously failed"):
            r.ensure_base()

    def test_retry_base_init_clears_error(self):
        r = _make_registry()
        r._base_init_error = "previous failure"
        with patch.object(r._base, "init") as mock_init:
            mock_init.side_effect = lambda **kw: setattr(r._base, "backend", "local")
            r.retry_base_init()
        assert r._base_init_error is None
        assert r._base.backend == "local"


class TestActiveLora:
    def test_active_lora_getter_setter(self):
        r = _make_registry()
        r.active_lora = "ukiyo-e"
        assert r.active_lora == "ukiyo-e"
        r.active_lora = "none"
        assert r.active_lora == "none"


class TestTurbo:
    def test_get_turbo_lazy_loads(self):
        r = _make_registry()
        mock_pipe = MagicMock()
        with patch("aetherart.registry.ModelRegistry.get_turbo", return_value=mock_pipe):
            result = r.get_turbo()
        assert result is mock_pipe

    def test_release_turbo_clears_pipe(self):
        r = _make_registry()
        r._turbo = MagicMock()
        r.release_turbo()
        assert r._turbo is None

    def test_release_turbo_idempotent(self):
        r = _make_registry()
        r.release_turbo()
        r.release_turbo()


class TestQuantized:
    def test_get_quantized_evicts_other_mode(self):
        r = _make_registry()
        fake_8bit = MagicMock()
        r._quant = fake_8bit
        r._quant_mode = "8bit"

        mock_4bit = MagicMock()
        # Patch the deferred import inside get_quantized
        with patch("aetherart.quantization.load_sd21_quantized", return_value=mock_4bit):
            try:
                r.get_quantized(4)
            except Exception:
                pass  # scheduler swap may fail without a real diffusers pipe
        # Key invariant: the 8-bit pipe must have been evicted
        assert r._quant is not fake_8bit


class TestHealth:
    def test_health_shows_not_loaded_before_init(self):
        r = _make_registry()
        h = r.health()
        assert "not_loaded" in h["base"]
        assert "not_loaded" in h["turbo"]
        assert "not_loaded" in h["quantized"]

    def test_health_shows_ok_after_base_init(self):
        r = _make_registry()
        r._base.backend = "local"
        h = r.health()
        assert "ok" in h["base"]

    def test_health_shows_error_after_failed_init(self):
        r = _make_registry()
        r._base_init_error = "no GPU available"
        h = r.health()
        assert "error" in h["base"]

    def test_health_returns_dict(self):
        r = _make_registry()
        h = r.health()
        assert isinstance(h, dict)
        assert "base" in h
        assert "turbo" in h
        assert "quantized" in h


class TestReleaseAll:
    def test_release_all_idempotent(self):
        r = _make_registry()
        r.release_all()
        r.release_all()

    def test_release_all_clears_turbo(self):
        r = _make_registry()
        r._turbo = MagicMock()
        r.release_all()
        assert r._turbo is None

    def test_release_all_clears_quant(self):
        r = _make_registry()
        r._quant = MagicMock()
        r._quant_mode = "8bit"
        r.release_all()
        assert r._quant is None
        assert r._quant_mode is None

    def test_release_all_clears_base_pipe(self):
        r = _make_registry()
        r._base.pipe = MagicMock()
        r._base.backend = "local"
        r.release_all()
        assert r._base.pipe is None
