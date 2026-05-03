import logging
import os
from dataclasses import dataclass

_log = logging.getLogger(__name__)


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        _log.warning("Invalid %s=%r; using default %d", key, raw, default)
        return default


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        _log.warning("Invalid %s=%r; using default %g", key, raw, default)
        return default


@dataclass
class Config:
    default_model: str = os.environ.get("HF_MODEL_ID", "sd2-community/stable-diffusion-2-1")
    sdxl_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    hf_token: str | None = os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get(
        "HF_API_TOKEN"
    )
    use_inference: bool = os.environ.get("USE_HF_INFERENCE", "0") == "1"
    default_width: int = _env_int("DEFAULT_WIDTH", 512)
    default_height: int = _env_int("DEFAULT_HEIGHT", 512)
    default_steps: int = _env_int("DEFAULT_STEPS", 30)
    default_guidance: float = _env_float("DEFAULT_GUIDANCE", 7.5)
    device: str = "cuda" if (os.environ.get("FORCE_CPU", "0") != "1") else "cpu"


cfg = Config()
