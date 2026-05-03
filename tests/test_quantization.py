"""Tests for quantization module.
No actual model loading — verifies API shape and import availability.
"""

import pytest


def _bnb_capable():
    """Check if bitsandbytes is fully installed (importable AND has metadata)."""
    try:
        import bitsandbytes  # noqa: F401
        import importlib.metadata

        importlib.metadata.version("bitsandbytes")
        from transformers import BitsAndBytesConfig

        BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
        return True
    except Exception:
        return False


bnb_required = pytest.mark.skipif(
    not _bnb_capable(),
    reason="bitsandbytes not fully installed (CPU CI environment)",
)


class TestQuantizationImports:
    def test_bitsandbytes_available(self):
        pytest.importorskip("bitsandbytes", reason="bitsandbytes not installed")

    def test_diffusers_bnb_config_available(self):
        diffusers = pytest.importorskip("diffusers")
        assert diffusers.BitsAndBytesConfig is not None

    def test_quantization_module_imports(self):
        from aetherart.quantization import load_sd21_quantized, vram_allocated_mb, vram_peak_mb

        assert callable(load_sd21_quantized)
        assert callable(vram_allocated_mb)
        assert callable(vram_peak_mb)


class TestQuantizationAPI:
    def test_load_signature(self):
        import inspect
        from aetherart.quantization import load_sd21_quantized

        sig = inspect.signature(load_sd21_quantized)
        params = sig.parameters
        assert "bits" in params
        assert params["bits"].default in (4, 8)

    def test_vram_helpers_return_float(self):
        from aetherart.quantization import vram_allocated_mb, vram_peak_mb

        assert isinstance(vram_allocated_mb(), float)
        assert isinstance(vram_peak_mb(), float)

    @bnb_required
    def test_bits_config_4bit(self):
        from diffusers import BitsAndBytesConfig

        cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
        assert cfg.load_in_4bit is True

    @bnb_required
    def test_bits_config_8bit(self):
        from diffusers import BitsAndBytesConfig

        cfg = BitsAndBytesConfig(load_in_8bit=True)
        assert cfg.load_in_8bit is True
