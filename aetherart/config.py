from dataclasses import dataclass
import os

@dataclass
class Config:
    default_model: str = os.environ.get("HF_MODEL_ID", "stabilityai/stable-diffusion-2-1")
    sdxl_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    hf_token: str | None = os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HF_API_TOKEN")
    use_inference: bool = os.environ.get("USE_HF_INFERENCE", "0") == "1"
    default_width: int = int(os.environ.get("DEFAULT_WIDTH", 512))
    default_height: int = int(os.environ.get("DEFAULT_HEIGHT", 512))
    default_steps: int = int(os.environ.get("DEFAULT_STEPS", 30))
    default_guidance: float = float(os.environ.get("DEFAULT_GUIDANCE", 7.5))
    device: str = "cuda" if (os.environ.get("FORCE_CPU", "0") != "1") else "cpu"

cfg = Config()
