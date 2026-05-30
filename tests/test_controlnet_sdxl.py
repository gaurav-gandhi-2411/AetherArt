"""Tests for controlnet_sdxl module — fully mocked, no GPU or real model loading."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _patch_cn_sdxl_diffusers():
    """Return patches for all diffusers names used in controlnet_sdxl, plus singleton reset."""
    import aetherart.controlnet_sdxl as mod

    mock_cn_cls = MagicMock()
    mock_vae_cls = MagicMock()
    mock_pipe_cls = MagicMock()

    mock_cn_cls.from_pretrained.return_value = MagicMock()
    mock_vae_cls.from_pretrained.return_value = MagicMock()
    mock_pipe_cls.from_pretrained.return_value = MagicMock()

    patches = [
        patch.object(mod, "ControlNetUnionModel", mock_cn_cls),
        patch.object(mod, "AutoencoderKL", mock_vae_cls),
        patch.object(mod, "StableDiffusionXLControlNetUnionPipeline", mock_pipe_cls),
        patch.object(mod, "_controlnet_union_model", None),
    ]
    return patches, mock_cn_cls, mock_vae_cls, mock_pipe_cls


class TestLoadSdxlControlnetPipeline:
    def test_uses_fp16_fix_vae(self):
        import torch

        import aetherart.controlnet_sdxl as mod  # noqa: F401
        from aetherart.config import cfg as aether_cfg
        from aetherart.controlnet_sdxl import load_sdxl_controlnet_pipeline

        patches, _, mock_vae_cls, _ = _patch_cn_sdxl_diffusers()
        with patches[0], patches[1], patches[2], patches[3]:
            load_sdxl_controlnet_pipeline()

        mock_vae_cls.from_pretrained.assert_called_once_with(
            aether_cfg.sdxl_vae_fix,
            torch_dtype=torch.float16,
        )

    def test_uses_model_cpu_offload(self):
        import aetherart.controlnet_sdxl as mod  # noqa: F401
        from aetherart.controlnet_sdxl import load_sdxl_controlnet_pipeline

        patches, _, _, mock_pipe_cls = _patch_cn_sdxl_diffusers()
        with patches[0], patches[1], patches[2], patches[3]:
            load_sdxl_controlnet_pipeline()

        mock_pipe_cls.from_pretrained.return_value.enable_model_cpu_offload.assert_called_once()

    def test_does_not_use_sequential_offload(self):
        import aetherart.controlnet_sdxl as mod  # noqa: F401
        from aetherart.controlnet_sdxl import load_sdxl_controlnet_pipeline

        patches, _, _, mock_pipe_cls = _patch_cn_sdxl_diffusers()
        with patches[0], patches[1], patches[2], patches[3]:
            load_sdxl_controlnet_pipeline()

        pipe_inst = mock_pipe_cls.from_pretrained.return_value
        pipe_inst.enable_sequential_cpu_offload.assert_not_called()

    def test_uses_controlnet_union_model_id(self):
        import aetherart.controlnet_sdxl as mod  # noqa: F401
        from aetherart.config import cfg as aether_cfg
        from aetherart.controlnet_sdxl import load_sdxl_controlnet_pipeline

        patches, mock_cn_cls, _, _ = _patch_cn_sdxl_diffusers()
        with patches[0], patches[1], patches[2], patches[3]:
            load_sdxl_controlnet_pipeline()

        mock_cn_cls.from_pretrained.assert_called_once()
        assert mock_cn_cls.from_pretrained.call_args.args[0] == aether_cfg.sdxl_controlnet_union

    def test_controlnet_union_model_is_singleton(self):
        """Two pipeline loads reuse the same ControlNetUnionModel — from_pretrained called once."""
        import aetherart.controlnet_sdxl as mod  # noqa: F401
        from aetherart.controlnet_sdxl import load_sdxl_controlnet_pipeline

        patches, mock_cn_cls, _, _ = _patch_cn_sdxl_diffusers()
        with patches[0], patches[1], patches[2], patches[3]:
            load_sdxl_controlnet_pipeline()
            load_sdxl_controlnet_pipeline()

        assert mock_cn_cls.from_pretrained.call_count == 1

    def test_lora_none_skips_load_lora_weights(self):
        import aetherart.controlnet_sdxl as mod  # noqa: F401
        from aetherart.controlnet_sdxl import load_sdxl_controlnet_pipeline

        patches, _, _, mock_pipe_cls = _patch_cn_sdxl_diffusers()
        with patches[0], patches[1], patches[2], patches[3]:
            load_sdxl_controlnet_pipeline(lora_name="none")

        pipe_inst = mock_pipe_cls.from_pretrained.return_value
        pipe_inst.load_lora_weights.assert_not_called()

    def test_lora_name_triggers_load_and_set_adapters(self):
        import aetherart.controlnet_sdxl as mod  # noqa: F401
        from aetherart.controlnet_sdxl import load_sdxl_controlnet_pipeline

        patches, _, _, mock_pipe_cls = _patch_cn_sdxl_diffusers()
        with patches[0], patches[1], patches[2], patches[3]:
            load_sdxl_controlnet_pipeline(lora_name="ukiyo-e", lora_alpha=0.8)

        pipe_inst = mock_pipe_cls.from_pretrained.return_value
        pipe_inst.load_lora_weights.assert_called_once()
        pipe_inst.set_adapters.assert_called_once_with(["user_lora"], adapter_weights=[0.8])


class TestReleaseSdxlControlnetPipeline:
    def test_release_clears_cuda_cache(self):
        import aetherart.controlnet_sdxl as mod
        from aetherart.controlnet_sdxl import release_sdxl_controlnet_pipeline

        mock_pipe = MagicMock()
        with (
            patch.object(mod, "gc") as mock_gc,
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.empty_cache") as mock_empty,
        ):
            release_sdxl_controlnet_pipeline(mock_pipe)

        mock_gc.collect.assert_called_once()
        mock_empty.assert_called_once()


class TestPreprocessCanny:
    def test_preprocess_canny_returns_pil_image(self):
        import numpy as np
        from PIL import Image

        from aetherart.controlnet_sdxl import preprocess_canny

        gray_arr = np.zeros((8, 8), dtype=np.uint8)
        edges_arr = np.zeros((8, 8), dtype=np.uint8)
        edges_rgb_arr = np.zeros((8, 8, 3), dtype=np.uint8)

        mock_cv2 = MagicMock()
        mock_cv2.COLOR_RGB2GRAY = 6
        mock_cv2.COLOR_GRAY2RGB = 8
        mock_cv2.cvtColor.side_effect = [gray_arr, edges_rgb_arr]
        mock_cv2.Canny.return_value = edges_arr

        input_img = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            result = preprocess_canny(input_img)

        assert isinstance(result, Image.Image)


class TestGenerateSdxlControlnet:
    def _mock_pipe(self):
        import numpy as np
        from PIL import Image

        pipe = MagicMock()
        pipe.return_value.images = [Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))]
        return pipe

    def _blank_image(self):
        import numpy as np
        from PIL import Image

        return Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))

    def test_canny_passes_control_mode_3(self):
        from aetherart.controlnet_sdxl import generate_sdxl_controlnet

        pipe = self._mock_pipe()
        generate_sdxl_controlnet(pipe, "test", self._blank_image(), ctype="canny")
        assert pipe.call_args.kwargs["control_mode"] == [3]

    def test_depth_passes_control_mode_1(self):
        from aetherart.controlnet_sdxl import generate_sdxl_controlnet

        pipe = self._mock_pipe()
        generate_sdxl_controlnet(pipe, "test", self._blank_image(), ctype="depth")
        assert pipe.call_args.kwargs["control_mode"] == [1]

    def test_unknown_ctype_raises_value_error(self):
        from aetherart.controlnet_sdxl import generate_sdxl_controlnet

        pipe = self._mock_pipe()
        with pytest.raises(ValueError, match="Unknown ctype"):
            generate_sdxl_controlnet(pipe, "test", self._blank_image(), ctype="bogus")

    def test_seed_creates_torch_generator(self):
        import torch

        from aetherart.controlnet_sdxl import generate_sdxl_controlnet

        pipe = self._mock_pipe()
        generate_sdxl_controlnet(pipe, "test", self._blank_image(), seed=42)
        assert isinstance(pipe.call_args.kwargs["generator"], torch.Generator)

    def test_no_seed_passes_none_generator(self):
        from aetherart.controlnet_sdxl import generate_sdxl_controlnet

        pipe = self._mock_pipe()
        generate_sdxl_controlnet(pipe, "test", self._blank_image())
        assert pipe.call_args.kwargs["generator"] is None

    def test_returns_first_pil_image(self):
        import numpy as np
        from PIL import Image

        from aetherart.controlnet_sdxl import generate_sdxl_controlnet

        expected = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
        pipe = MagicMock()
        pipe.return_value.images = [expected]
        result = generate_sdxl_controlnet(pipe, "test", self._blank_image())
        assert result is expected

    def test_passes_conditioning_scale(self):
        from aetherart.controlnet_sdxl import generate_sdxl_controlnet

        pipe = self._mock_pipe()
        generate_sdxl_controlnet(pipe, "test", self._blank_image(), conditioning_scale=0.65)
        assert pipe.call_args.kwargs["controlnet_conditioning_scale"] == 0.65

    def test_passes_control_image_kwarg(self):
        from aetherart.controlnet_sdxl import generate_sdxl_controlnet

        pipe = self._mock_pipe()
        ctrl = self._blank_image()
        generate_sdxl_controlnet(pipe, "test", ctrl)
        assert pipe.call_args.kwargs["control_image"] is ctrl

    def test_passes_negative_prompt(self):
        from aetherart.controlnet_sdxl import generate_sdxl_controlnet

        pipe = self._mock_pipe()
        generate_sdxl_controlnet(pipe, "test", self._blank_image(), negative_prompt="ugly, blurry")
        assert pipe.call_args.kwargs["negative_prompt"] == "ugly, blurry"

    def test_passes_dimensions(self):
        from aetherart.controlnet_sdxl import generate_sdxl_controlnet

        pipe = self._mock_pipe()
        generate_sdxl_controlnet(pipe, "test", self._blank_image(), width=512, height=768)
        assert pipe.call_args.kwargs["width"] == 512
        assert pipe.call_args.kwargs["height"] == 768

    def test_shared_aliases_map_to_same_int(self):
        """hed, pidi, scribble, ted all map to 2; canny and lineart both map to 3."""
        from aetherart.controlnet_sdxl import _CTYPE_TO_INT

        assert _CTYPE_TO_INT["hed"] == _CTYPE_TO_INT["pidi"] == _CTYPE_TO_INT["scribble"] == 2
        assert _CTYPE_TO_INT["canny"] == _CTYPE_TO_INT["lineart"] == 3


class TestPreprocessDepth:
    def test_preprocess_depth_returns_pil_image(self):
        import numpy as np
        import torch
        from PIL import Image

        from aetherart.controlnet_sdxl import preprocess_depth

        mock_outputs = MagicMock()
        mock_outputs.predicted_depth = torch.ones(1, 8, 8)

        mock_model = MagicMock(return_value=mock_outputs)
        mock_proc_inst = MagicMock(return_value={"pixel_values": torch.zeros(1, 3, 8, 8)})

        mock_transformers = MagicMock()
        mock_transformers.AutoImageProcessor.from_pretrained.return_value = mock_proc_inst
        mock_transformers.AutoModelForDepthEstimation.from_pretrained.return_value = mock_model

        input_img = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            result = preprocess_depth(input_img)

        assert isinstance(result, Image.Image)
