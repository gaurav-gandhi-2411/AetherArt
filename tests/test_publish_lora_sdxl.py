"""Tests for scripts/publish_lora_sdxl.py — mocked, no Hub calls."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch


def _import_module():
    """Import publish_lora_sdxl from scripts/."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "publish_lora_sdxl",
        Path(__file__).resolve().parent.parent / "scripts" / "publish_lora_sdxl.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["publish_lora_sdxl"] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _import_module()


class TestDryRun:
    def test_dry_run_validates_adapter_path_exists(self, tmp_path, capsys):
        adapter = tmp_path / "test.safetensors"
        adapter.write_bytes(b"fake weights data")

        model_card = tmp_path / "model_card.md"
        model_card.write_text("# Test")

        with (
            patch.object(_mod, "_MODEL_CARD_PATH", model_card),
            patch.object(_mod, "_resolve_token", return_value="hf_fake"),
        ):
            err = _mod._validate(adapter)
        assert err is None

    def test_dry_run_rejects_missing_adapter(self, tmp_path):
        missing = tmp_path / "nonexistent.safetensors"
        err = _mod._validate(missing)
        assert err is not None
        assert "not found" in err

    def test_dry_run_rejects_empty_adapter(self, tmp_path):
        empty = tmp_path / "empty.safetensors"
        empty.write_bytes(b"")
        err = _mod._validate(empty)
        assert err is not None
        assert "empty" in err

    def test_dry_run_prints_planned_actions(self, tmp_path, capsys):
        adapter = tmp_path / "test.safetensors"
        adapter.write_bytes(b"fake weights data " * 100)

        model_card = tmp_path / "model_card.md"
        model_card.write_text("# Test")

        with (
            patch.object(_mod, "_MODEL_CARD_PATH", model_card),
            patch.object(_mod, "_SAMPLE_IMAGES_DIR", tmp_path / "nonexistent_samples"),
            patch.object(_mod, "_resolve_token", return_value="hf_fake"),
        ):
            import sys

            old_argv = sys.argv
            sys.argv = [
                "publish_lora_sdxl.py",
                "--adapter-path",
                str(adapter),
                "--repo-id",
                "testuser/test-sdxl-repo",
                "--dry-run",
            ]
            try:
                rc = _mod.main()
            finally:
                sys.argv = old_argv

        assert rc == 0
        captured = capsys.readouterr()
        assert "[DRY RUN]" in captured.out
        assert "testuser/test-sdxl-repo" in captured.out
        assert "No Hub API calls made" in captured.out

    def test_dry_run_reports_sample_count_when_dir_exists(self, tmp_path, capsys):
        adapter = tmp_path / "test.safetensors"
        adapter.write_bytes(b"fake weights data " * 100)

        model_card = tmp_path / "model_card.md"
        model_card.write_text("# Test")

        samples_dir = tmp_path / "samples"
        samples_dir.mkdir()
        for i in range(12):
            (samples_dir / f"sample_{i:02d}.png").write_bytes(b"fake png")

        with (
            patch.object(_mod, "_MODEL_CARD_PATH", model_card),
            patch.object(_mod, "_SAMPLE_IMAGES_DIR", samples_dir),
            patch.object(_mod, "_resolve_token", return_value="hf_fake"),
        ):
            import sys

            old_argv = sys.argv
            sys.argv = [
                "publish_lora_sdxl.py",
                "--adapter-path",
                str(adapter),
                "--repo-id",
                "testuser/test-sdxl-repo",
                "--dry-run",
            ]
            try:
                rc = _mod.main()
            finally:
                sys.argv = old_argv

        assert rc == 0
        captured = capsys.readouterr()
        assert "12 PNGs" in captured.out


class TestPublish:
    def test_publish_calls_create_repo_with_exist_ok(self, tmp_path):
        adapter = tmp_path / "test.safetensors"
        adapter.write_bytes(b"fake weights data " * 100)

        model_card = tmp_path / "model_card.md"
        model_card.write_text("# Test")

        mock_api_instance = MagicMock()
        mock_api_cls = MagicMock(return_value=mock_api_instance)

        with (
            patch.object(_mod, "_MODEL_CARD_PATH", model_card),
            patch.object(_mod, "_SAMPLE_IMAGES_DIR", tmp_path / "nonexistent_samples"),
            patch.dict(os.environ, {"HUGGINGFACEHUB_API_TOKEN": "hf_fake"}),
            patch("huggingface_hub.HfApi", mock_api_cls),
        ):
            _mod._publish(adapter, "testuser/test-sdxl-repo")

        mock_api_instance.create_repo.assert_called_once_with(
            "testuser/test-sdxl-repo",
            repo_type="model",
            exist_ok=True,
            private=False,
            token="hf_fake",
        )

    def test_publish_uploads_safetensors_and_readme(self, tmp_path):
        adapter = tmp_path / "test.safetensors"
        adapter.write_bytes(b"fake weights data " * 100)

        model_card = tmp_path / "model_card.md"
        model_card.write_text("# Test")

        mock_api_instance = MagicMock()
        mock_api_cls = MagicMock(return_value=mock_api_instance)

        with (
            patch.object(_mod, "_MODEL_CARD_PATH", model_card),
            patch.object(_mod, "_SAMPLE_IMAGES_DIR", tmp_path / "nonexistent_samples"),
            patch.dict(os.environ, {"HUGGINGFACEHUB_API_TOKEN": "hf_fake"}),
            patch("huggingface_hub.HfApi", mock_api_cls),
        ):
            _mod._publish(adapter, "testuser/test-sdxl-repo")

        upload_calls = mock_api_instance.upload_file.call_args_list
        assert len(upload_calls) == 2

        first_kwargs = upload_calls[0].kwargs
        assert first_kwargs["path_in_repo"] == "test.safetensors"
        assert first_kwargs["repo_id"] == "testuser/test-sdxl-repo"

        second_kwargs = upload_calls[1].kwargs
        assert second_kwargs["path_in_repo"] == "README.md"
        assert second_kwargs["repo_id"] == "testuser/test-sdxl-repo"

        mock_api_instance.upload_folder.assert_not_called()

    def test_publish_uploads_samples_folder_when_dir_exists(self, tmp_path):
        adapter = tmp_path / "test.safetensors"
        adapter.write_bytes(b"fake weights data " * 100)

        model_card = tmp_path / "model_card.md"
        model_card.write_text("# Test")

        samples_dir = tmp_path / "samples"
        samples_dir.mkdir()
        (samples_dir / "sample_01.png").write_bytes(b"fake png")

        mock_api_instance = MagicMock()
        mock_api_cls = MagicMock(return_value=mock_api_instance)

        with (
            patch.object(_mod, "_MODEL_CARD_PATH", model_card),
            patch.object(_mod, "_SAMPLE_IMAGES_DIR", samples_dir),
            patch.dict(os.environ, {"HUGGINGFACEHUB_API_TOKEN": "hf_fake"}),
            patch("huggingface_hub.HfApi", mock_api_cls),
        ):
            _mod._publish(adapter, "testuser/test-sdxl-repo")

        mock_api_instance.upload_folder.assert_called_once()
        folder_kwargs = mock_api_instance.upload_folder.call_args.kwargs
        assert folder_kwargs["path_in_repo"] == "samples"
        assert folder_kwargs["repo_id"] == "testuser/test-sdxl-repo"
