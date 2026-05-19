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
from collections import OrderedDict
from typing import Any, Literal, Optional

from .gpu_hygiene import cleanup_gpu
from .logger import get_logger
from .model import AetherModel

try:
    from diffusers import StableDiffusionXLPipeline
except Exception:
    StableDiffusionXLPipeline = None  # type: ignore[assignment, misc]

logger = get_logger(__name__)


class ModelRegistry:
    def __init__(self) -> None:
        self._base: AetherModel = AetherModel()
        self._turbo: Any = None
        self._quant: Optional[Any] = None  # at most 1 quantized pipeline
        self._quant_mode: Optional[str] = None  # "8bit" or "4bit"
        self._active_lora: str = "none"
        self._base_init_error: Optional[str] = None
        self._sdxl_base: Optional[Any] = None
        self._sdxl_base_init_error: Optional[str] = None
        self._sdxl_quantized: Optional[Any] = None
        self._sdxl_quantized_bits: Optional[int] = None
        self._sdxl_quantized_init_error: Optional[str] = None
        # LRU-2 cache for SDXL ControlNet Union pipelines.
        # Key: (ctype, lora_name, lora_alpha)  Value: pipeline object
        self._sdxl_cn_cache: OrderedDict[tuple[str, str, float], Any] = OrderedDict()
        self._sdxl_cn_init_error: Optional[str] = None

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

    # ── SDXL Base ───────────────────────────────────────────────────────────

    def get_sdxl_base(self) -> Any:
        """Lazy-load the SDXL base pipeline and return it.

        Caches the init failure so repeated calls raise immediately instead of
        re-attempting a download that already failed.  Call retry_sdxl_base_init()
        to clear the cached error and try again.
        """
        if self._sdxl_base is not None:
            return self._sdxl_base
        if self._sdxl_base_init_error is not None:
            raise RuntimeError(
                f"SDXL base init previously failed: {self._sdxl_base_init_error}. "
                "Call registry.retry_sdxl_base_init() to retry."
            )
        try:
            from .sdxl_pipeline import load_sdxl_base

            logger.info("Loading SDXL base pipeline…")
            self._sdxl_base = load_sdxl_base()
            self._sdxl_base_init_error = None
            logger.info("SDXL base pipeline ready")
        except Exception as exc:
            self._sdxl_base_init_error = str(exc)
            raise RuntimeError(f"SDXL base init failed: {exc}") from exc
        return self._sdxl_base

    def retry_sdxl_base_init(self) -> None:
        """Clear the cached SDXL init failure and try again."""
        self._sdxl_base_init_error = None
        self._sdxl_base = None
        self.get_sdxl_base()

    def release_sdxl_base(self) -> None:
        if self._sdxl_base is not None:
            from .sdxl_pipeline import release_sdxl_pipeline

            release_sdxl_pipeline(self._sdxl_base)
            self._sdxl_base = None

    # ── SDXL Quantized ──────────────────────────────────────────────────────

    def get_sdxl_quantized(self, bits: Literal[4, 8] = 4) -> Any:
        """Return the NF4/INT8-quantized SDXL pipeline, loading if needed.

        Evicts the current quantized pipeline if a different bit width is requested.
        Caches init failures; call retry_sdxl_quantized_init() to retry.

        Note: SDXL base + SDXL NF4 + Flux are mutually exclusive on 8 GB VRAM.
        This slot can coexist with other slots for now; mutual-exclusion logic
        lands with demo wiring in a later PR.
        """
        if self._sdxl_quantized_bits == bits and self._sdxl_quantized is not None:
            return self._sdxl_quantized
        if self._sdxl_quantized_init_error is not None:
            raise RuntimeError(
                f"SDXL quantized init previously failed: {self._sdxl_quantized_init_error}. "
                "Call registry.retry_sdxl_quantized_init() to retry."
            )
        if self._sdxl_quantized is not None:
            logger.info(
                "Evicting SDXL quantized pipeline (%sbit) for new request (%sbit)",
                self._sdxl_quantized_bits,
                bits,
            )
            self.release_sdxl_quantized()
        try:
            from .quantization import load_sdxl_quantized

            logger.info("Loading %sbit-quantized SDXL pipeline...", bits)
            self._sdxl_quantized = load_sdxl_quantized(bits=bits)
            self._sdxl_quantized_bits = bits
            self._sdxl_quantized_init_error = None
            logger.info("%sbit-quantized SDXL pipeline ready", bits)
        except Exception as exc:
            self._sdxl_quantized_init_error = str(exc)
            raise RuntimeError(f"SDXL quantized init failed: {exc}") from exc
        return self._sdxl_quantized

    def retry_sdxl_quantized_init(self, bits: Literal[4, 8] = 4) -> None:
        """Clear the cached SDXL quantized init failure and try again."""
        self._sdxl_quantized_init_error = None
        self.get_sdxl_quantized(bits=bits)

    def release_sdxl_quantized(self) -> None:
        if self._sdxl_quantized is not None:
            from .quantization import release_quantized_pipeline

            release_quantized_pipeline(self._sdxl_quantized)
            self._sdxl_quantized = None
            self._sdxl_quantized_bits = None

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

    # ── SDXL ControlNet Union (LRU-2) ───────────────────────────────────────

    _SDXL_CN_CACHE_MAX: int = 2

    def get_sdxl_controlnet_pipeline(
        self,
        ctype: str = "canny",
        lora_name: str = "none",
        lora_alpha: float = 1.0,
    ) -> Any:
        """Return a cached SDXL ControlNet Union pipeline, loading if needed.

        LRU-2: at most 2 pipeline wrappers in memory at once. The underlying
        ControlNetUnionModel singleton is NOT evicted on LRU overflow — only the
        pipeline wrapper is released. Call release_all_sdxl_cn() to free the model.
        """
        key = (ctype, lora_name, lora_alpha)
        if key in self._sdxl_cn_cache:
            self._sdxl_cn_cache.move_to_end(key)
            return self._sdxl_cn_cache[key]
        if self._sdxl_cn_init_error is not None:
            raise RuntimeError(
                f"SDXL ControlNet init previously failed: {self._sdxl_cn_init_error}. "
                "Call registry.retry_sdxl_cn_init() to retry."
            )
        if len(self._sdxl_cn_cache) >= self._SDXL_CN_CACHE_MAX:
            evicted_key, evicted_pipe = self._sdxl_cn_cache.popitem(last=False)
            logger.info("Evicting SDXL ControlNet pipeline for key %s", evicted_key)
            from .controlnet_sdxl import release_sdxl_controlnet_pipeline

            release_sdxl_controlnet_pipeline(evicted_pipe)
        try:
            from .controlnet_sdxl import load_sdxl_controlnet_pipeline

            logger.info(
                "Loading SDXL ControlNet Union pipeline (ctype=%s, lora=%s)...", ctype, lora_name
            )
            pipe = load_sdxl_controlnet_pipeline(lora_name=lora_name, lora_alpha=lora_alpha)
            self._sdxl_cn_cache[key] = pipe
            self._sdxl_cn_init_error = None
        except Exception as exc:
            self._sdxl_cn_init_error = str(exc)
            raise RuntimeError(f"SDXL ControlNet init failed: {exc}") from exc
        return pipe

    def retry_sdxl_cn_init(
        self,
        ctype: str = "canny",
        lora_name: str = "none",
        lora_alpha: float = 1.0,
    ) -> Any:
        """Clear the cached ControlNet init failure and try again."""
        self._sdxl_cn_init_error = None
        return self.get_sdxl_controlnet_pipeline(ctype, lora_name, lora_alpha)

    def release_all_sdxl_cn(self) -> None:
        """Release all cached ControlNet pipelines and the shared ControlNetUnionModel."""
        if self._sdxl_cn_cache:
            from .controlnet_sdxl import release_sdxl_controlnet_pipeline

            for pipe in list(self._sdxl_cn_cache.values()):
                release_sdxl_controlnet_pipeline(pipe)
            self._sdxl_cn_cache.clear()
        try:
            from .controlnet_sdxl import release_controlnet_union_model

            release_controlnet_union_model()
        except Exception:
            pass

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
        if self._sdxl_base_init_error:
            result["sdxl_base"] = f"error: {self._sdxl_base_init_error}"
        elif self._sdxl_base is not None:
            result["sdxl_base"] = "loaded"
        else:
            result["sdxl_base"] = "not_loaded"
        result["turbo"] = "ok" if self._turbo is not None else "not_loaded"
        if self._quant is not None:
            result["quantized"] = f"ok ({self._quant_mode})"
        else:
            result["quantized"] = "not_loaded"
        if self._sdxl_quantized_init_error:
            result["sdxl_quantized"] = f"error: {self._sdxl_quantized_init_error}"
        elif self._sdxl_quantized is not None:
            result["sdxl_quantized"] = (
                "nf4_loaded" if self._sdxl_quantized_bits == 4 else "int8_loaded"
            )
        else:
            result["sdxl_quantized"] = "not_loaded"
        # SD 2.1 ControlNet cache lives in controlnet.py
        try:
            from . import controlnet as cn

            result["controlnet_cache"] = f"{len(cn._cn_pipelines)} / {cn._MAX_CN_CACHE} entries"
        except Exception:
            result["controlnet_cache"] = "unknown"
        # SDXL ControlNet Union LRU-2 cache
        result["sdxl_cn_cache"] = (
            f"{len(self._sdxl_cn_cache)} / {self._SDXL_CN_CACHE_MAX} entries"
        )
        return result

    def release_all(self) -> None:
        """Release all loaded pipelines and free GPU memory. Safe to call multiple times."""
        self.release_sdxl_base()
        self.release_sdxl_quantized()
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
        self.release_all_sdxl_cn()
        cleanup_gpu(verbose=True)
        logger.info("ModelRegistry: all pipelines released")
