import json
import tempfile
from pathlib import Path

from PIL import Image

from aetherart.metadata import get_git_commit, load_metadata_from_image, save_image_with_metadata

_KNOWN_METADATA = {
    "prompt": "a glowing crystal cave",
    "negative_prompt": "blurry",
    "seed": 42,
    "scheduler": "DPMSolverMultistepScheduler",
    "steps": 20,
    "guidance": 7.5,
    "width": 512,
    "height": 512,
    "model_id": "sd2-community/stable-diffusion-2-1",
    "lora_hash": "",
    "git_commit": "abc1234",
    "generation_time_seconds": 12.34,
    "vram_peak_mb": 4096.0,
    "timestamp": "2026-04-25T12:00:00",
}


def test_save_and_load_roundtrip():
    img = Image.new("RGB", (64, 64), color=(128, 64, 192))

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test_output.png"
        save_image_with_metadata(img, out_path, _KNOWN_METADATA)

        assert out_path.exists(), "PNG file should be created"
        sidecar = out_path.with_suffix(".json")
        assert sidecar.exists(), "Sidecar JSON should be created"

        loaded = load_metadata_from_image(out_path)

        # PNG tEXt chunks are strings; compare as strings for non-timestamp fields
        for key, expected in _KNOWN_METADATA.items():
            if key == "timestamp":
                continue
            assert str(expected) == loaded.get(
                key
            ), f"Field '{key}' mismatch: expected {expected!r}, got {loaded.get(key)!r}"


def test_sidecar_json_is_valid():
    img = Image.new("RGB", (32, 32), color=(0, 0, 0))

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test.png"
        save_image_with_metadata(img, out_path, _KNOWN_METADATA)

        sidecar = out_path.with_suffix(".json")
        with sidecar.open(encoding="utf-8") as f:
            data = json.load(f)

        assert data["prompt"] == _KNOWN_METADATA["prompt"]
        assert data["seed"] == _KNOWN_METADATA["seed"]


def test_load_metadata_empty_for_plain_png():
    img = Image.new("RGB", (16, 16), color=(255, 255, 255))

    with tempfile.TemporaryDirectory() as tmpdir:
        plain_path = Path(tmpdir) / "plain.png"
        img.save(str(plain_path), format="PNG")
        loaded = load_metadata_from_image(plain_path)
        assert loaded == {}


def test_get_git_commit_returns_string():
    commit = get_git_commit()
    assert isinstance(commit, str)
    assert len(commit) > 0
