from __future__ import annotations

import inspect
import logging
from typing import Any

logger = logging.getLogger(__name__)


def safe_get(d: dict[str, Any], key: str, default: Any = None) -> Any:
    return d.get(key, default)


def preferred_dtype_kwarg(fn) -> str | None:
    """Return 'torch_dtype', 'dtype', or None based on what from_pretrained accepts.

    diffusers ≥ 0.36 deprecates torch_dtype= in favour of dtype=; this helper
    inspects the signature so callers work with both old and new diffusers.
    """
    try:
        params = inspect.signature(fn).parameters
        if "torch_dtype" in params:
            return "torch_dtype"
        if "dtype" in params:
            return "dtype"
    except Exception:
        logger.debug("Could not inspect signature for %s", getattr(fn, "__name__", str(fn)))
    return None


def build_pretrained_kwargs(fn, dtype, hf_token: str | None = None) -> dict[str, Any]:
    """Build from_pretrained kwargs with the correct dtype key and optional token."""
    kwargs: dict[str, Any] = {}
    kw = preferred_dtype_kwarg(fn)
    if kw:
        kwargs[kw] = dtype
    if hf_token:
        kwargs["token"] = hf_token
    return kwargs
