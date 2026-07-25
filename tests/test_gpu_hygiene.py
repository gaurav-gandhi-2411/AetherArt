"""Tests for gpu_hygiene.cleanup_gpu and gpu_is_quiet — must not crash in import-only
environments, and must correctly detect GPU contention without requiring a real GPU to test."""

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


class TestGpuIsQuiet:
    """gpu_is_quiet gates latency-budget assertions (tests/test_combined_path.py) on the GPU
    actually being uncontended - these mock torch.cuda directly so the contention-detection
    logic is verified without needing a real GPU (or a real contending workload) in CI."""

    def test_quiet_when_usage_below_threshold(self):
        from aetherart.gpu_hygiene import gpu_is_quiet

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.mem_get_info", return_value=(7_500 * 1024**2, 8_000 * 1024**2)),
        ):
            assert gpu_is_quiet(threshold_mb=500) is True

    def test_not_quiet_when_usage_above_threshold(self):
        from aetherart.gpu_hygiene import gpu_is_quiet

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.mem_get_info", return_value=(1_000 * 1024**2, 8_000 * 1024**2)),
        ):
            assert gpu_is_quiet(threshold_mb=500) is False

    def test_quiet_when_cuda_unavailable(self):
        """CUDA-unavailable is not contention - callers rely on this to avoid spuriously
        skipping in a CPU-only environment (the outer test is already skipped by conftest.py
        in that case, via the `gpu` marker, for an unrelated reason)."""
        from aetherart.gpu_hygiene import gpu_is_quiet

        with patch("torch.cuda.is_available", return_value=False):
            assert gpu_is_quiet() is True

    def test_quiet_on_exception(self):
        """A broken check must not itself become a source of test flakiness - fail open."""
        from aetherart.gpu_hygiene import gpu_is_quiet

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.mem_get_info", side_effect=RuntimeError("boom")),
        ):
            assert gpu_is_quiet() is True

    def test_exactly_at_threshold_is_quiet(self):
        """Boundary check: used_mb == threshold_mb counts as quiet (<=, not <)."""
        from aetherart.gpu_hygiene import gpu_is_quiet

        total = 8_000 * 1024**2
        free = total - 500 * 1024**2
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.mem_get_info", return_value=(free, total)),
        ):
            assert gpu_is_quiet(threshold_mb=500) is True
