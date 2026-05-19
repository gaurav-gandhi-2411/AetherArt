"""Pytest configuration: auto-skip GPU-marked tests when CUDA is unavailable."""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(items: list) -> None:
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False
    if has_cuda:
        return
    skip_gpu = pytest.mark.skip(reason="requires CUDA GPU")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
