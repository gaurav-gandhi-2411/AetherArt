"""Tests for train_lora.py --base flag dispatch.

No real training runs — validates that build_command routes to the correct
vendored script and sets the correct resolution and VAE flag per base.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from train_lora import _BASE_CONFIG, build_command


def _args(base: str = "sd21", **overrides):
    """Build a minimal argparse Namespace for build_command."""
    import argparse

    defaults = {
        "base": base,
        "model": None,
        "resolution": None,
        "train_batch_size": 1,
        "grad_accum": 4,
        "lr": 1e-4,
        "max_train_steps": 1500,
        "rank": 8,
        "mixed_precision": "fp16",
        "checkpointing_steps": 250,
        "validation_prompt": "ukyowood test prompt",
        "num_validation_images": 1,
        "validation_epochs": 1,
        "seed": 42,
        "no_xformers": True,
        "no_gradient_checkpointing": False,
        "output_dir": None,
        "data_dir": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestBaseDispatch:
    def test_sd21_routes_to_sd21_script(self):
        cmd, _ = build_command(_args("sd21"), sys.executable)
        script = next(a for a in cmd if "_diffusers_train_text_to_image_lora" in a)
        assert "sdxl" not in script

    def test_sdxl_routes_to_sdxl_script(self):
        cmd, _ = build_command(_args("sdxl"), sys.executable)
        script = next(a for a in cmd if "_diffusers_train_text_to_image_lora" in a)
        assert "sdxl" in script

    def test_sd21_default_resolution_512(self):
        cmd, _ = build_command(_args("sd21"), sys.executable)
        idx = cmd.index("--resolution")
        assert cmd[idx + 1] == "512"

    def test_sdxl_default_resolution_1024(self):
        cmd, _ = build_command(_args("sdxl"), sys.executable)
        idx = cmd.index("--resolution")
        assert cmd[idx + 1] == "1024"

    def test_sdxl_includes_fp16_fix_vae(self):
        cmd, _ = build_command(_args("sdxl"), sys.executable)
        assert "--pretrained_vae_model_name_or_path" in cmd
        vae_idx = cmd.index("--pretrained_vae_model_name_or_path")
        assert cmd[vae_idx + 1] == "madebyollin/sdxl-vae-fp16-fix"

    def test_sd21_does_not_include_vae_arg(self):
        cmd, _ = build_command(_args("sd21"), sys.executable)
        assert "--pretrained_vae_model_name_or_path" not in cmd

    def test_resolution_override_respected(self):
        cmd, _ = build_command(_args("sdxl", resolution=512), sys.executable)
        idx = cmd.index("--resolution")
        assert cmd[idx + 1] == "512"

    def test_model_override_respected(self):
        cmd, _ = build_command(_args("sdxl", model="custom/model"), sys.executable)
        idx = cmd.index("--pretrained_model_name_or_path")
        assert cmd[idx + 1] == "custom/model"

    def test_sd21_output_dir_uses_training_output(self):
        _, output_dir = build_command(_args("sd21"), sys.executable)
        assert "training_output_sdxl" not in str(output_dir)
        assert "training_output" in str(output_dir)

    def test_sdxl_output_dir_uses_training_output_sdxl(self):
        _, output_dir = build_command(_args("sdxl"), sys.executable)
        assert "training_output_sdxl" in str(output_dir)


class TestBaseConfig:
    def test_both_bases_registered(self):
        assert "sd21" in _BASE_CONFIG
        assert "sdxl" in _BASE_CONFIG

    def test_sd21_script_exists(self):
        script_name = _BASE_CONFIG["sd21"][0]
        script_path = Path(__file__).resolve().parent.parent / "scripts" / script_name
        assert script_path.exists(), f"SD 2.1 training script not found: {script_path}"

    def test_sdxl_script_exists(self):
        script_name = _BASE_CONFIG["sdxl"][0]
        script_path = Path(__file__).resolve().parent.parent / "scripts" / script_name
        assert script_path.exists(), f"SDXL training script not found: {script_path}"
