"""ModelRegistry — single owner of all pipeline singletons in the app.

Replaces the scattered module-level globals in app.py:
  MODEL, _turbo_pipe, _quant_pipes, _active_lora_name.

Design decisions:
- ControlNet LRU cache stays in controlnet.py (self-contained, LRU-capped).
  The registry calls cn.invalidate_cache() in release_all().
- Quantized pipelines: at most 1 loaded at a time. On an 8 GB GPU,
  fp16 SD 2.1 (~3.1 GB) + 8-bit (~2.2 GB) + 4-bit (~2.8 GB) = OOM.
  Evict the previous quantized pipeline before loading a new variant.
- Init failures are stored, not swallowed. get_base() raises a clear
  RuntimeError if init previously failed; call retry_base_init() to retry.
"""

from __future__ import annotations

import gc
from typing import Any, Literal, Optional

from .gpu_hygiene import cleanup_gpu
from .logger import get_logger
from .model import AetherModel

logger = get_logger(__name__)


class ModelRegistry:
    def __init__(self) -> None:
        self._base: AetherModel = AetherModel()
        self._turbo: Any = None
        self._quant: Optional[Any] = None  # at most 1 quantized pipeline
        self._quant_mode: Optional[str] = None  # "8bit" or "4bit"
        self._active_lora: str = "none"
        self._base_init_error: Optional[str] = None

    # ── Base SD 2.1 / SDXL ──────────────────────────────────────────────────

    def ensure_base(self, model_choice: str | None = None) -> None:
        """Init the base pipeline if not already loaded. Raises on failure."""
        if self._base.backend is not None:
            return
        if self._base_init_error is not None:
            raise RuntimeError(
                f"Base model init previously failed: {self._base_init_error}. "
                "Call registry.retry_base_init() to retry."
            )
        try:
            self._base.init(model_choice=model_choice)
            self._base_init_error = None
        except Exception as exc:
            self._base_init_error = str(exc)
            raise RuntimeError(f"Base model init failed: {exc}") from exc

    def retry_base_init(self, model_choice: str | None = None) -> None:
        """Clear the cached init failure and try again."""
        self._base_init_error = None
        self._base.backend = None
        self._base.pipe = None
        self.ensure_base(model_choice=model_choice)

    def get_base(self) -> AetherModel:
        return self._base

    @property
    def active_lora(self) -> str:
        return self._active_lora

    @active_lora.setter
    def active_lora(self, name: str) -> None:
        self._active_lora = name

    # ── SDXL Turbo ──────────────────────────────────────────────────────────

    def get_turbo(self) -> Any:
        """Lazy-load SDXL Turbo pipeline and return it."""
        if self._turbo is None:
            from .sdxl_turbo import load_turbo_pipeline

            logger.info("Loading SDXL Turbo pipeline…")
            self._turbo = load_turbo_pipeline()
            logger.info("SDXL Turbo ready")
        return self._turbo

    def release_turbo(self) -> None:
        if self._turbo is not None:
            del self._turbo
            self._turbo = None
            gc.collect()

    # ── Quantized pipelines ─────────────────────────────────────────────────

    def get_quantized(self, bits: Literal[4, 8]) -> Any:
        """Return the quantized pipeline for `bits`, evicting the other if loaded."""
        mode = f"{bits}bit"
        if self._quant_mode == mode and self._quant is not None:
            return self._quant
        # Evict previous quantized pipeline before loading a new variant
        if self._quant is not None:
            logger.info(
                "Evicting quantized pipeline (%s) for new request (%s)", self._quant_mode, mode
            )
            del self._quant
            self._quant = None
            self._quant_mode = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        from .quantization import load_sd21_quantized

        logger.info("Loading %s-quantized SD 2.1…", bits)
        self._quant = load_sd21_quantized(bits=bits)
        self._quant_mode = mode

        # Apply DPM-Solver++ scheduler
        try:
            from diffusers import DPMSolverMultistepScheduler

            pipe = self._quant
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(  # type: ignore[attr-defined]
                pipe.scheduler.config  # type: ignore[attr-defined]
            )
        except Exception as exc:
            logger.debug("Scheduler swap on quantized pipeline failed: %s", exc)

        logger.info("%s-quantized pipeline ready", bits)
        return self._quant

    # ── Health + lifecycle ───────────────────────────────────────────────────

    def health(self) -> dict[str, str]:
        """Return a status dict for each pipeline component."""
        result: dict[str, str] = {}
        if self._base_init_error:
            result["base"] = f"error: {self._base_init_error}"
        elif self._base.backend is None:
            result["base"] = "not_loaded"
        else:
            result["base"] = f"ok ({self._base.backend})"
        result["turbo"] = "ok" if self._turbo is not None else "not_loaded"
        if self._quant is not None:
            result["quantized"] = f"ok ({self._quant_mode})"
        else:
            result["quantized"] = "not_loaded"
        # ControlNet cache lives in controlnet.py
        try:
            from . import controlnet as cn

            result["controlnet_cache"] = f"{len(cn._cn_pipelines)} / {cn._MAX_CN_CACHE} entries"
        except Exception:
            result["controlnet_cache"] = "unknown"
        return result

    def release_all(self) -> None:
        """Release all loaded pipelines and free GPU memory. Safe to call multiple times."""
        self.release_turbo()
        if self._quant is not None:
            del self._quant
            self._quant = None
            self._quant_mode = None
        if self._base.pipe is not None:
            del self._base.pipe
            self._base.pipe = None
            self._base.backend = None
        try:
            from . import controlnet as cn

            cn.invalidate_cache()
        except Exception:
            pass
        cleanup_gpu(verbose=True)
        logger.info("ModelRegistry: all pipelines released")
