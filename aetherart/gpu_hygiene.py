"""GPU memory cleanup and contention-detection utilities — shared across scripts, the app
server, and tests."""

from __future__ import annotations

import gc

from .logger import get_logger

logger = get_logger(__name__)

DEFAULT_QUIET_THRESHOLD_MB = 500.0


def cleanup_gpu(*, verbose: bool = False) -> None:
    """Release GPU memory: gc + synchronize + empty_cache + ipc_collect.

    Safe to call even when torch is not installed or CUDA is unavailable.
    Idempotent — safe to call multiple times or from atexit handlers.
    """
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            if verbose:
                free, total = torch.cuda.mem_get_info()
                logger.info(
                    "GPU cleanup: %.2f GB free / %.2f GB total",
                    free / 1e9,
                    total / 1e9,
                )
    except ImportError:
        pass


def gpu_is_quiet(threshold_mb: float = DEFAULT_QUIET_THRESHOLD_MB) -> bool:
    """True if no other process already holds significant GPU memory.

    Used to gate wall-clock latency-budget assertions in GPU tests: a latency budget is only a
    meaningful measurement when this process has the card to itself. Contention from another
    resident workload (an Ollama-served VLM judge left loaded, a concurrent eval/training run)
    has inflated per-step latency 5-10x in this project's own measurements with no code
    regression involved - asserting a hardcoded budget without checking for this has produced
    the same false "regression" report across three separate sessions.

    `threshold_mb` defaults to 500 MB: comfortably above the ~0-50 MB idle baseline this
    project's 8 GB card shows with nothing loaded, comfortably below any resident SDXL/VLM
    model's footprint (multiple GB). Returns True (assume quiet) if CUDA is unavailable or the
    check itself fails, so callers gating on this don't spuriously skip in a CPU-only
    environment - the check is a contention *detector*, not a CUDA-availability gate.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return True
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        used_mb = (total_bytes - free_bytes) / 1024**2
        return used_mb <= threshold_mb
    except Exception:
        return True
