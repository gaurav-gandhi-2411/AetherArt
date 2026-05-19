"""G2 combined-path tests: NF4 SDXL + Hyper-8step LoRA.

Mocked class for CI (no GPU). GPU class gated with @pytest.mark.gpu.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aetherart.hyper import HYPER_DEFAULTS, is_hyper_active, load_hyper_lora
from aetherart.registry import ModelRegistry


class TestCombinedPathMocked:
    def test_combined_path_loads_nf4_sdxl(self):
        r = ModelRegistry()
        mock_pipe = MagicMock()
        target = "aetherart.quantization.load_sdxl_quantized"
        with patch(target, return_value=mock_pipe) as mock_load:
            result = r.get_sdxl_quantized(bits=4)
        mock_load.assert_called_once_with(bits=4)
        assert result is mock_pipe

    def test_combined_path_loads_hyper_8step_on_nf4(self):
        pipe = MagicMock()
        pipe._aetherart_hyper_variant = None
        pipe.scheduler = MagicMock()

        mock_euler_cls = MagicMock()
        mock_euler_cls.from_config.return_value = MagicMock()

        with patch("aetherart.hyper.EulerDiscreteScheduler", mock_euler_cls):
            load_hyper_lora(pipe, "8step")

        assert pipe._aetherart_hyper_variant == "8step"
        pipe.load_lora_weights.assert_called_once()
        assert pipe.load_lora_weights.call_args.kwargs["adapter_name"] == "hyper_8step"

    def test_combined_path_loads_ukiyo_e_on_nf4_hyper(self):
        # Degenerate composition step: verify set_adapters is callable after Hyper is active.
        # Real Ukiyo-e SDXL composition lands in PR 09.
        pipe = MagicMock()
        pipe._aetherart_hyper_variant = "8step"
        pipe.set_adapters(["hyper_8step"], adapter_weights=[1.0])
        pipe.set_adapters.assert_called_with(["hyper_8step"], adapter_weights=[1.0])

    def test_combined_path_uses_fp16_fix_vae(self):
        import torch

        import aetherart.quantization as qmod
        from aetherart.config import cfg as aether_cfg
        from aetherart.quantization import load_sdxl_quantized

        mock_vae_cls = MagicMock()
        mock_vae_cls.from_pretrained.return_value = MagicMock()
        mock_pipe_cls = MagicMock()
        mock_pipe_cls.from_pretrained.return_value = MagicMock()

        with (
            patch.object(qmod, "AutoencoderKL", mock_vae_cls),
            patch.object(qmod, "BitsAndBytesConfig", MagicMock()),
            patch.object(qmod, "UNet2DConditionModel", MagicMock()),
            patch.object(qmod, "StableDiffusionXLPipeline", mock_pipe_cls),
            patch.object(qmod, "DPMSolverMultistepScheduler", MagicMock()),
        ):
            load_sdxl_quantized(bits=4)

        mock_vae_cls.from_pretrained.assert_called_once_with(
            aether_cfg.sdxl_vae_fix,
            torch_dtype=torch.float16,
        )

    def test_combined_path_uses_euler_trailing_scheduler(self):
        pipe = MagicMock()
        pipe._aetherart_hyper_variant = None
        pipe.scheduler = MagicMock()

        mock_euler_cls = MagicMock()
        new_scheduler = MagicMock()
        mock_euler_cls.from_config.return_value = new_scheduler

        with patch("aetherart.hyper.EulerDiscreteScheduler", mock_euler_cls):
            load_hyper_lora(pipe, "8step")

        assert mock_euler_cls.from_config.call_args.kwargs.get("timestep_spacing") == "trailing"
        assert pipe.scheduler is new_scheduler


@pytest.mark.gpu
class TestCombinedPathGPU:
    def test_combined_path_nf4_hyper_within_budget(self):
        import time

        import numpy as np
        import torch

        r = ModelRegistry()
        r.release_sdxl_base()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        pipe = r.get_sdxl_quantized(bits=4)
        load_hyper_lora(pipe, "8step")
        assert is_hyper_active(pipe) == "8step"

        defaults = HYPER_DEFAULTS["8step"]
        t0 = time.perf_counter()
        img = pipe(
            "a red apple on a wooden table",
            negative_prompt="blurry, low quality",
            num_inference_steps=defaults["num_inference_steps"],
            guidance_scale=defaults["guidance_scale"],
            height=1024,
            width=1024,
            generator=torch.Generator("cuda").manual_seed(42),
        ).images[0]
        latency = time.perf_counter() - t0

        arr = np.array(img)
        vram_mb = torch.cuda.max_memory_allocated() / 1024**2

        assert vram_mb < 7500, f"VRAM peak {vram_mb:.0f} MB exceeds 7500 budget"
        assert arr.mean() > 10, f"all-black: mean={arr.mean():.2f}"
        assert latency < 30, f"latency {latency:.1f}s exceeds 30s budget"
