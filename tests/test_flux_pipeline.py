"""Tests for aetherart.flux_pipeline — mocked, no real model loading."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

FAKE_READ_TOKEN = "hf_fake_read_token_for_tests"


@pytest.fixture(autouse=True)
def _hf_read_token_env(monkeypatch):
    """FLUX.1-schnell is gated — every loader requires HF_READ_TOKEN (see _hf_read_token()).
    Set a fake value for every test in this module so loader tests exercise the token-plumbing
    path rather than the missing-token error path (that error path gets its own test below)."""
    monkeypatch.setenv("HF_READ_TOKEN", FAKE_READ_TOKEN)


class TestHfReadToken:
    def test_raises_when_hf_read_token_unset(self, monkeypatch):
        monkeypatch.delenv("HF_READ_TOKEN", raising=False)
        from aetherart.flux_pipeline import _hf_read_token

        with pytest.raises(RuntimeError, match="HF_READ_TOKEN is not set"):
            _hf_read_token()


class TestLoadFluxSchnell:
    def _run_load(self, mock_pipe_cls):
        with patch("aetherart.flux_pipeline.FluxPipeline", mock_pipe_cls):
            from aetherart.flux_pipeline import load_flux_schnell

            return load_flux_schnell()

    def test_load_flux_schnell_uses_bf16_dtype(self):
        import torch

        from aetherart.flux_pipeline import FLUX_SCHNELL_MODEL

        mock_pipe_cls = MagicMock()
        mock_pipe_instance = MagicMock()
        mock_pipe_cls.from_pretrained.return_value = mock_pipe_instance

        result = self._run_load(mock_pipe_cls)

        mock_pipe_cls.from_pretrained.assert_called_once_with(
            FLUX_SCHNELL_MODEL, torch_dtype=torch.bfloat16, token=FAKE_READ_TOKEN
        )
        assert result is mock_pipe_instance

    def test_load_flux_schnell_enables_model_cpu_offload(self):
        mock_pipe_cls = MagicMock()
        mock_pipe_instance = MagicMock()
        mock_pipe_cls.from_pretrained.return_value = mock_pipe_instance

        self._run_load(mock_pipe_cls)

        mock_pipe_instance.enable_model_cpu_offload.assert_called_once()


class TestLoadFluxSchnellQuantized:
    def test_load_flux_schnell_quantized_uses_nf4_and_no_offload(self):
        import torch

        mock_transformer_cls = MagicMock()
        mock_transformer = MagicMock()
        mock_transformer_cls.from_pretrained.return_value = mock_transformer

        mock_text_encoder_cls = MagicMock()
        mock_text_encoder = MagicMock()
        mock_text_encoder_cls.from_pretrained.return_value = mock_text_encoder

        mock_pipe_cls = MagicMock()
        mock_pipe_instance = MagicMock()
        mock_pipe_instance.to.return_value = mock_pipe_instance
        mock_pipe_cls.from_pretrained.return_value = mock_pipe_instance

        class _FakeQuantConfig:
            def __init__(self, load_in_4bit, bnb_4bit_quant_type):
                self.load_in_4bit = load_in_4bit
                self.bnb_4bit_quant_type = bnb_4bit_quant_type

        with (
            patch("aetherart.flux_pipeline.FluxPipeline", mock_pipe_cls),
            patch("aetherart.flux_pipeline.FluxTransformer2DModel", mock_transformer_cls),
            patch("aetherart.flux_pipeline.T5EncoderModel", mock_text_encoder_cls),
            patch("aetherart.flux_pipeline.BitsAndBytesConfig", _FakeQuantConfig),
            patch("aetherart.flux_pipeline.TransformersBitsAndBytesConfig", _FakeQuantConfig),
        ):
            from aetherart.flux_pipeline import load_flux_schnell_quantized

            result = load_flux_schnell_quantized()

        # Transformer loaded with NF4 quantization from the "transformer" subfolder.
        _, t_kwargs = mock_transformer_cls.from_pretrained.call_args
        assert t_kwargs["subfolder"] == "transformer"
        assert t_kwargs["quantization_config"].load_in_4bit is True
        assert t_kwargs["quantization_config"].bnb_4bit_quant_type == "nf4"
        assert t_kwargs["torch_dtype"] == torch.bfloat16
        assert t_kwargs["token"] == FAKE_READ_TOKEN

        # T5 text encoder loaded with NF4 quantization from the "text_encoder_2" subfolder.
        _, te_kwargs = mock_text_encoder_cls.from_pretrained.call_args
        assert te_kwargs["subfolder"] == "text_encoder_2"
        assert te_kwargs["quantization_config"].load_in_4bit is True
        assert te_kwargs["quantization_config"].bnb_4bit_quant_type == "nf4"
        assert te_kwargs["token"] == FAKE_READ_TOKEN

        # Pipeline built from the quantized components and moved to GPU — no CPU offload call.
        _, pipe_kwargs = mock_pipe_cls.from_pretrained.call_args
        assert pipe_kwargs["transformer"] is mock_transformer
        assert pipe_kwargs["text_encoder_2"] is mock_text_encoder
        assert pipe_kwargs["token"] == FAKE_READ_TOKEN
        mock_pipe_instance.to.assert_called_once_with("cuda")
        mock_pipe_instance.enable_model_cpu_offload.assert_not_called()
        assert result is mock_pipe_instance


class TestReleaseFluxPipeline:
    def test_release_flux_pipeline_clears_cache(self):
        mock_pipe = MagicMock()

        with patch("aetherart.flux_pipeline.gc") as mock_gc:
            import torch

            with (
                patch.object(torch.cuda, "is_available", return_value=True),
                patch.object(torch.cuda, "empty_cache") as mock_empty,
            ):
                from aetherart.flux_pipeline import release_flux_pipeline

                release_flux_pipeline(mock_pipe)

                mock_gc.collect.assert_called_once()
                mock_empty.assert_called_once()
