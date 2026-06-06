"""Tests for aetherart.sdxl_pipeline — mocked, no real model loading."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestLoadSdxlBase:
    def _run_load(self, mock_vae, mock_pipe_cls, mock_scheduler_cls):
        """Helper: patch diffusers imports and call load_sdxl_base()."""
        with (
            patch("aetherart.sdxl_pipeline.AutoencoderKL", mock_vae),
            patch("aetherart.sdxl_pipeline.StableDiffusionXLPipeline", mock_pipe_cls),
            patch("aetherart.sdxl_pipeline.DPMSolverMultistepScheduler", mock_scheduler_cls),
        ):
            from aetherart.sdxl_pipeline import load_sdxl_base

            return load_sdxl_base()

    def test_load_sdxl_base_returns_pipeline_with_fp16_fix_vae(self):
        import torch

        from aetherart.config import cfg

        mock_vae_cls = MagicMock()
        mock_vae_instance = MagicMock()
        mock_vae_cls.from_pretrained.return_value = mock_vae_instance

        mock_pipe_cls = MagicMock()
        mock_pipe_instance = MagicMock()
        mock_pipe_cls.from_pretrained.return_value = mock_pipe_instance

        mock_scheduler_cls = MagicMock()
        mock_scheduler_instance = MagicMock()
        mock_scheduler_cls.from_config.return_value = mock_scheduler_instance

        result = self._run_load(mock_vae_cls, mock_pipe_cls, mock_scheduler_cls)

        # VAE must be constructed from cfg.sdxl_vae_fix
        mock_vae_cls.from_pretrained.assert_called_once_with(
            cfg.sdxl_vae_fix,
            torch_dtype=torch.float16,
        )
        # Pipeline must receive the vae kwarg
        mock_pipe_cls.from_pretrained.assert_called_once()
        call_kwargs = mock_pipe_cls.from_pretrained.call_args.kwargs
        assert call_kwargs["vae"] is mock_vae_instance
        assert call_kwargs["torch_dtype"] == torch.float16

        assert result is mock_pipe_instance

    def test_load_sdxl_base_swaps_to_dpm_solver_multistep(self):
        mock_vae_cls = MagicMock()
        mock_vae_cls.from_pretrained.return_value = MagicMock()

        mock_pipe_cls = MagicMock()
        mock_pipe_instance = MagicMock()
        mock_pipe_cls.from_pretrained.return_value = mock_pipe_instance

        mock_scheduler_cls = MagicMock()
        mock_scheduler_instance = MagicMock()
        mock_scheduler_cls.from_config.return_value = mock_scheduler_instance

        # Capture original scheduler config before the swap replaces pipe.scheduler
        original_scheduler_config = mock_pipe_instance.scheduler.config

        self._run_load(mock_vae_cls, mock_pipe_cls, mock_scheduler_cls)

        mock_scheduler_cls.from_config.assert_called_once_with(original_scheduler_config)
        assert mock_pipe_instance.scheduler is mock_scheduler_instance

    def test_load_sdxl_base_enables_model_cpu_offload(self):
        mock_vae_cls = MagicMock()
        mock_vae_cls.from_pretrained.return_value = MagicMock()

        mock_pipe_cls = MagicMock()
        mock_pipe_instance = MagicMock()
        mock_pipe_cls.from_pretrained.return_value = mock_pipe_instance

        mock_scheduler_cls = MagicMock()
        mock_scheduler_cls.from_config.return_value = MagicMock()

        self._run_load(mock_vae_cls, mock_pipe_cls, mock_scheduler_cls)

        mock_pipe_instance.enable_model_cpu_offload.assert_called_once()


class TestReleaseSdxlPipeline:
    def test_release_sdxl_pipeline_clears_cache(self):
        mock_pipe = MagicMock()

        with patch("aetherart.sdxl_pipeline.gc") as mock_gc, patch("aetherart.sdxl_pipeline.cfg"):
            import torch

            with (
                patch.object(torch.cuda, "is_available", return_value=True),
                patch.object(torch.cuda, "empty_cache") as mock_empty,
            ):
                from aetherart.sdxl_pipeline import release_sdxl_pipeline

                release_sdxl_pipeline(mock_pipe)

                mock_gc.collect.assert_called_once()
                mock_empty.assert_called_once()
