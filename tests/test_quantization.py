"""Tests for quantization module.
No actual model loading — verifies API shape and import availability.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _bnb_capable():
    """Check if bitsandbytes is fully installed (importable AND has metadata)."""
    try:
        import importlib.metadata

        import bitsandbytes  # noqa: F401

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


class TestSdxlQuantization:
    """Tests for load_sdxl_quantized — fully mocked, no GPU or real model loading."""

    def _patch_sdxl_diffusers(self):
        """Return a context manager that mocks all SDXL diffusers module-level names."""
        import aetherart.quantization as qmod

        mock_bnb_cls = MagicMock()
        mock_unet_cls = MagicMock()
        mock_vae_cls = MagicMock()
        mock_pipe_cls = MagicMock()
        mock_dpm_cls = MagicMock()

        mock_pipe_instance = MagicMock()
        mock_pipe_cls.from_pretrained.return_value = mock_pipe_instance
        mock_unet_cls.from_pretrained.return_value = MagicMock()
        mock_vae_cls.from_pretrained.return_value = MagicMock()
        mock_dpm_cls.from_config.return_value = MagicMock()

        patches = [
            patch.object(qmod, "BitsAndBytesConfig", mock_bnb_cls),
            patch.object(qmod, "UNet2DConditionModel", mock_unet_cls),
            patch.object(qmod, "AutoencoderKL", mock_vae_cls),
            patch.object(qmod, "StableDiffusionXLPipeline", mock_pipe_cls),
            patch.object(qmod, "DPMSolverMultistepScheduler", mock_dpm_cls),
        ]
        return patches, mock_pipe_instance, mock_vae_cls, mock_dpm_cls, mock_pipe_cls

    def test_load_sdxl_quantized_nf4_uses_fp16_fix_vae(self):
        import torch

        from aetherart.config import cfg as aether_cfg
        from aetherart.quantization import load_sdxl_quantized

        patches, _, mock_vae_cls, _, _ = self._patch_sdxl_diffusers()
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            load_sdxl_quantized(bits=4)

        mock_vae_cls.from_pretrained.assert_called_once_with(
            aether_cfg.sdxl_vae_fix,
            torch_dtype=torch.float16,
        )

    def test_load_sdxl_quantized_nf4_uses_dpm_solver_multistep(self):
        from aetherart.quantization import load_sdxl_quantized

        patches, mock_pipe_instance, _, mock_dpm_cls, _ = self._patch_sdxl_diffusers()
        original_scheduler_config = mock_pipe_instance.scheduler.config
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            load_sdxl_quantized(bits=4)

        mock_dpm_cls.from_config.assert_called_once_with(original_scheduler_config)
        assert mock_pipe_instance.scheduler is mock_dpm_cls.from_config.return_value

    def test_load_sdxl_quantized_nf4_enables_model_cpu_offload(self):
        from aetherart.quantization import load_sdxl_quantized

        patches, mock_pipe_instance, _, _, _ = self._patch_sdxl_diffusers()
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            load_sdxl_quantized(bits=4)

        mock_pipe_instance.enable_model_cpu_offload.assert_called_once()

    def test_load_sdxl_quantized_nf4_does_not_use_sequential_offload(self):
        from aetherart.quantization import load_sdxl_quantized

        patches, mock_pipe_instance, _, _, _ = self._patch_sdxl_diffusers()
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            load_sdxl_quantized(bits=4)

        mock_pipe_instance.enable_sequential_cpu_offload.assert_not_called()

    def test_release_sdxl_quantized_clears_cache(self):
        import aetherart.quantization as qmod
        from aetherart.quantization import release_quantized_pipeline

        mock_pipe = MagicMock()
        with (
            patch.object(qmod, "gc") as mock_gc,
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.empty_cache") as mock_empty,
        ):
            release_quantized_pipeline(mock_pipe)

        mock_gc.collect.assert_called_once()
        mock_empty.assert_called_once()
