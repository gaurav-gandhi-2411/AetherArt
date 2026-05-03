"""GPU memory cleanup utility — shared across scripts and the app server."""

from __future__ import annotations

import gc

from .logger import get_logger

logger = get_logger(__name__)


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
