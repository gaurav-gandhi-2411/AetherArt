"""Tests for aetherart.flux_pipeline — mocked, no real model loading."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


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
            FLUX_SCHNELL_MODEL, torch_dtype=torch.bfloat16
        )
        assert result is mock_pipe_instance

    def test_load_flux_schnell_enables_model_cpu_offload(self):
        mock_pipe_cls = MagicMock()
        mock_pipe_instance = MagicMock()
        mock_pipe_cls.from_pretrained.return_value = mock_pipe_instance

        self._run_load(mock_pipe_cls)

        mock_pipe_instance.enable_model_cpu_offload.assert_called_once()


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
