"""Tests for the docs/samples/ pre-generated sample directory."""

import json
import os
from pathlib import Path

import pytest

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "docs" / "samples"

EXPECTED_TIERS = [
    "standard_fp16",
    "turbo",
    "lora_ukiyo_e",
    "controlnet_canny",
    "controlnet_depth",
    "quantized_8bit",
    "quantized_4bit",
]

REQUIRED_META_KEYS = {"prompt", "seed", "tier", "inference_time_rtx3070_s", "vram_peak_mb"}


@pytest.mark.skipif(not SAMPLES_DIR.exists(), reason="docs/samples not generated yet")
class TestSamplesDirectory:
    def test_samples_root_exists(self):
        assert SAMPLES_DIR.is_dir(), f"Expected {SAMPLES_DIR} to exist"

    def test_expected_tier_subdirectories(self):
        missing = [t for t in EXPECTED_TIERS if not (SAMPLES_DIR / t).is_dir()]
        assert not missing, f"Missing tier directories: {missing}"

    def test_each_tier_has_at_least_one_png(self):
        for tier in EXPECTED_TIERS:
            tier_dir = SAMPLES_DIR / tier
            if not tier_dir.exists():
                continue
            pngs = [
                p
                for p in tier_dir.glob("*.png")
                if not any(p.stem.endswith(s) for s in ("_source", "_canny_map", "_depth_map"))
            ]
            assert len(pngs) >= 1, f"Tier '{tier}' has no output PNG files"


@pytest.mark.skipif(not SAMPLES_DIR.exists(), reason="docs/samples not generated yet")
class TestSampleMetadata:
    def _output_pngs(self):
        for tier in EXPECTED_TIERS:
            tier_dir = SAMPLES_DIR / tier
            if not tier_dir.is_dir():
                continue
            for p in sorted(tier_dir.glob("*.png")):
                if any(p.stem.endswith(s) for s in ("_source", "_canny_map", "_depth_map")):
                    continue
                yield p

    def test_each_sample_has_sidecar_json(self):
        missing = []
        for png in self._output_pngs():
            if not png.with_suffix(".json").exists():
                missing.append(str(png.relative_to(SAMPLES_DIR)))
        assert not missing, f"Missing .json sidecars for: {missing}"

    def test_metadata_is_valid_json(self):
        for png in self._output_pngs():
            json_path = png.with_suffix(".json")
            if not json_path.exists():
                continue
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                pytest.fail(f"{json_path.name} is not valid JSON: {e}")
            assert isinstance(data, dict), f"{json_path.name} root must be a JSON object"

    def test_metadata_contains_required_keys(self):
        for png in self._output_pngs():
            json_path = png.with_suffix(".json")
            if not json_path.exists():
                continue
            data = json.loads(json_path.read_text(encoding="utf-8"))
            missing_keys = REQUIRED_META_KEYS - set(data.keys())
            assert not missing_keys, f"{json_path.name} missing keys: {missing_keys}"

    def test_inference_time_is_positive_number(self):
        for png in self._output_pngs():
            json_path = png.with_suffix(".json")
            if not json_path.exists():
                continue
            data = json.loads(json_path.read_text(encoding="utf-8"))
            t = data.get("inference_time_rtx3070_s")
            assert (
                isinstance(t, (int, float)) and t > 0
            ), f"{json_path.name}: inference_time_rtx3070_s={t!r} must be a positive number"

    def test_seed_matches_expected(self):
        """All samples use seed 42 by convention."""
        for png in self._output_pngs():
            json_path = png.with_suffix(".json")
            if not json_path.exists():
                continue
            data = json.loads(json_path.read_text(encoding="utf-8"))
            assert (
                data.get("seed") == 42
            ), f"{json_path.name}: expected seed=42, got {data.get('seed')!r}"
