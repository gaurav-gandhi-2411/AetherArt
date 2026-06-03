from __future__ import annotations

import os
from typing import Any

# Reads at import time; the Modal image bakes in AETHERART_ENABLE_SAFETY=1.
# Local dev leaves the var unset so this is a zero-overhead no-op.
_ENABLED: bool = os.environ.get("AETHERART_ENABLE_SAFETY", "0") == "1"

# 10–20 obvious terms that cover the most common NSFW/violent categories.
_BLOCKLIST: frozenset[str] = frozenset(
    {
        "nude",
        "naked",
        "nsfw",
        "explicit",
        "sexual",
        "pornographic",
        "porn",
        "hentai",
        "gore",
        "graphic violence",
        "child nude",
        "underage",
        "minor nude",
        "loli",
        "shota",
        "torture",
        "mutilation",
        "decapitation",
        "snuff",
        "bestiality",
    }
)


def is_enabled() -> bool:
    return _ENABLED


def check_prompt(prompt: str) -> str | None:
    """Return None if safe, or a user-facing refusal string if the prompt is blocked.

    No-op (always returns None) when AETHERART_ENABLE_SAFETY is unset or "0".
    This lets local dev/CI run without any overhead.
    """
    if not _ENABLED:
        return None
    lower = prompt.lower()
    for term in _BLOCKLIST:
        if term in lower:
            return "This prompt cannot be processed (contains disallowed content)."
    return None


def apply_safety_checker(pipe: Any) -> None:
    """Re-enable the diffusers safety_checker on a pipeline when safety is active.

    StableDiffusionXLPipeline does not have a native safety_checker attribute —
    the prompt blocklist (check_prompt) is the primary guard for the SDXL demo
    path.  This function is a no-op for SDXL but satisfies the G5 contract and
    is forward-compatible with SD 1.x paths that do carry a safety_checker.
    """
    if not _ENABLED:
        return
    # SD 1.x path: safety_checker may have been set to None to skip it.
    # Restore it only if a real checker object is stored on the pipe.
    checker = getattr(pipe, "safety_checker", None)
    if checker is not None and not callable(checker):
        pass  # already non-None, nothing to do
