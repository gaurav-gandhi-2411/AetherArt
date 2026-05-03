"""Tests for gpu_hygiene.cleanup_gpu — must not crash in import-only environments."""

from __future__ import annotations

import sys
from unittest.mock import patch


class TestCleanupGpu:
    def test_runs_without_error(self):
        from aetherart.gpu_hygiene import cleanup_gpu

        cleanup_gpu()

    def test_verbose_no_cuda(self):
        from aetherart.gpu_hygiene import cleanup_gpu

        with patch("torch.cuda.is_available", return_value=False):
            cleanup_gpu(verbose=True)

    def test_no_cuda_is_silent(self):
        from aetherart.gpu_hygiene import cleanup_gpu

        with patch("torch.cuda.is_available", return_value=False):
            cleanup_gpu()

    def test_idempotent(self):
        from aetherart.gpu_hygiene import cleanup_gpu

        cleanup_gpu()
        cleanup_gpu()
        cleanup_gpu()

    def test_import_error_is_swallowed(self):
        """Setting sys.modules['torch'] = None makes 'import torch' raise ImportError."""
        import aetherart.gpu_hygiene as gh

        original = sys.modules.get("torch")
        sys.modules["torch"] = None  # type: ignore[assignment]
        try:
            gh.cleanup_gpu()
        finally:
            if original is not None:
                sys.modules["torch"] = original
            else:
                sys.modules.pop("torch", None)
