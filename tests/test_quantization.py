"""Tests for quantization module.
No actual model loading — verifies API shape and import availability.
"""
import pytest


class TestQuantizationImports:
    def test_bitsandbytes_available(self):
        pytest.importorskip("bitsandbytes", reason="bitsandbytes not installed")

    def test_diffusers_bnb_config_available(self):
        from diffusers import BitsAndBytesConfig
        assert BitsAndBytesConfig is not None

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

    def test_bits_config_4bit(self):
        from diffusers import BitsAndBytesConfig
        cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
        assert cfg.load_in_4bit is True

    def test_bits_config_8bit(self):
        from diffusers import BitsAndBytesConfig
        cfg = BitsAndBytesConfig(load_in_8bit=True)
        assert cfg.load_in_8bit is True
