"""
Tests for ControlNet preprocessing functions.
No GPU or model downloads required — only cv2 + numpy.
"""
import pytest
import numpy as np
from PIL import Image

pytest.importorskip("cv2", reason="opencv-python not installed")

from aetherart.controlnet import preprocess_canny, preprocess


def _solid(w=64, h=64, color=(128, 128, 128)) -> Image.Image:
    return Image.new("RGB", (w, h), color)


def _gradient(w=64, h=64) -> Image.Image:
    row = np.linspace(0, 255, w, dtype=np.uint8)
    arr = np.tile(row, (h, 1))
    return Image.fromarray(np.stack([arr] * 3, axis=-1))


class TestPreprocessCanny:
    def test_returns_pil_image(self):
        result = preprocess_canny(_gradient())
        assert isinstance(result, Image.Image)

    def test_output_mode_is_rgb(self):
        result = preprocess_canny(_gradient())
        assert result.mode == "RGB"

    def test_output_size_matches_input(self):
        img = _gradient(100, 80)
        result = preprocess_canny(img, 50, 150)
        assert result.size == (100, 80)

    def test_solid_image_has_no_edges(self):
        result = preprocess_canny(_solid())
        assert np.array(result).max() == 0

    def test_lower_thresholds_detect_more_edges(self):
        img = _gradient()
        loose = np.array(preprocess_canny(img, low_threshold=10, high_threshold=30)).sum()
        tight = np.array(preprocess_canny(img, low_threshold=200, high_threshold=250)).sum()
        assert loose >= tight

    def test_channels_are_identical(self):
        result = np.array(preprocess_canny(_gradient()))
        assert (result[:, :, 0] == result[:, :, 1]).all()
        assert (result[:, :, 0] == result[:, :, 2]).all()


class TestPreprocessDispatch:
    def test_canny_dispatch(self):
        result = preprocess(_gradient(), "canny")
        assert isinstance(result, Image.Image)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown conditioning type"):
            preprocess(_gradient(), "bad_type")  # type: ignore[arg-type]


class TestGetPipelineCacheKey:
    """Verify cache key logic without loading any models."""

    def test_no_lora_key(self):
        from aetherart.controlnet import _make_cache_key
        key = _make_cache_key("canny", None, 1.0)
        assert key == ("canny", "none", 1.0)

    def test_lora_key_differs_from_no_lora(self):
        from aetherart.controlnet import _make_cache_key
        base = _make_cache_key("canny", None, 1.0)
        lora = _make_cache_key("canny", "ukiyo-e", 1.0)
        assert base != lora

    def test_alpha_included_in_key(self):
        from aetherart.controlnet import _make_cache_key
        k1 = _make_cache_key("canny", "ukiyo-e", 0.5)
        k2 = _make_cache_key("canny", "ukiyo-e", 1.0)
        assert k1 != k2

    def test_alpha_rounded_to_two_decimals(self):
        from aetherart.controlnet import _make_cache_key
        assert _make_cache_key("canny", "ukiyo-e", 1.001) == _make_cache_key("canny", "ukiyo-e", 1.002)
        assert _make_cache_key("canny", "ukiyo-e", 0.994) != _make_cache_key("canny", "ukiyo-e", 1.006)

    def test_ctype_included_in_key(self):
        from aetherart.controlnet import _make_cache_key
        assert _make_cache_key("canny", None, 1.0) != _make_cache_key("depth", None, 1.0)

    def test_empty_string_lora_normalised_to_none(self):
        from aetherart.controlnet import _make_cache_key
        assert _make_cache_key("canny", "", 1.0) == _make_cache_key("canny", None, 1.0)
