from typing import Optional, Dict, Any
import inspect
import torch
from .config import cfg
from .logger import get_logger

logger = get_logger(__name__)

# Lazy imports so tests / CI that don't have heavy deps still import package
try:
    from diffusers import StableDiffusionPipeline
except Exception:
    StableDiffusionPipeline = None

try:
    from diffusers import StableDiffusionXLPipeline
except Exception:
    StableDiffusionXLPipeline = None

try:
    from huggingface_hub import InferenceClient
except Exception:
    InferenceClient = None

from contextlib import contextmanager

@contextmanager
def nullcontext():
    yield

def _preferred_dtype_kwarg(fn) -> Optional[str]:
    """
    Inspect a callable (usually Pipeline.from_pretrained) and decide whether it
    accepts 'torch_dtype' or 'dtype' or neither.
    """
    try:
        sig = inspect.signature(fn)
        params = sig.parameters
        if "torch_dtype" in params:
            return "torch_dtype"
        if "dtype" in params:
            return "dtype"
    except Exception:
        logger.debug("Could not inspect signature for %s", getattr(fn, "__name__", str(fn)))
    return None

def _build_pretrained_kwargs(fn, dtype, hf_token: Optional[str]) -> Dict[str, Any]:
    """
    Build kwargs for from_pretrained that use the right dtype kwarg name
    depending on the pipeline implementation. Also include use_auth_token
    if provided.
    """
    kwargs: Dict[str, Any] = {}
    kw_name = _preferred_dtype_kwarg(fn)
    if kw_name:
        kwargs[kw_name] = dtype
    else:
        logger.debug("No dtype kwarg detected for function %s; loading without dtype kwarg", getattr(fn, "__name__", str(fn)))
    if hf_token:
        kwargs["use_auth_token"] = hf_token
    return kwargs

class AetherModel:
    """
    Encapsulates model init + generation with VRAM & performance mitigations.
    """

    def __init__(self, model_id: Optional[str] = None):
        self.model_id = model_id or cfg.default_model
        self.hf_token = cfg.hf_token
        self.use_inference = cfg.use_inference
        self.pipe = None
        self.backend = None  # 'local' or 'inference'
        self.inference_client = None
        # small record of what optimizations were applied (helpful for UI logs)
        self.optimizations: Dict[str, str] = {}

    def init(self, model_choice: str | None = None):
        """
        Initialize the pipeline. model_choice can be 'sd-2.1' or 'sdxl'.
        This method tries the following, in order:
         - If USE_HF_INFERENCE is True -> set up InferenceClient
         - Try to load SDXL pipeline if requested
         - Try to load SD2.1 pipeline
         - If local load succeeds, apply optimizations:
            * attention slicing
            * xformers (if available)
            * model CPU offload (if supported)
        """
        model_to_load = self.model_id
        if model_choice == "sdxl":
            model_to_load = cfg.sdxl_model

        logger.info("Initializing model '%s' (use_inference=%s)", model_to_load, self.use_inference)
        self.optimizations = {}

        # Use Inference API if user forces it
        if self.use_inference and InferenceClient is not None:
            try:
                self.inference_client = InferenceClient(token=self.hf_token) if self.hf_token else InferenceClient()
                self.backend = "inference"
                logger.info("Initialized Hugging Face InferenceClient")
                self.optimizations["backend"] = "inference"
                return
            except Exception as e:
                logger.warning("InferenceClient init failed: %s", e)

        # Load SDXL if requested and available
        if model_to_load == cfg.sdxl_model and StableDiffusionXLPipeline is not None:
            try:
                dtype_val = torch.float16 if torch.cuda.is_available() else torch.float32
                kwargs = _build_pretrained_kwargs(StableDiffusionXLPipeline.from_pretrained, dtype_val, self.hf_token)
                self.pipe = StableDiffusionXLPipeline.from_pretrained(model_to_load, **kwargs)
                # Apply optimizations (below)
                self._apply_optimizations()
                self.backend = "local"
                logger.info("Loaded SDXL pipeline on %s", "cuda" if torch.cuda.is_available() else "cpu")
                self.optimizations["model_loaded"] = "sdxl"
                return
            except Exception as e:
                logger.warning("Failed to load SDXL pipeline locally: %s", e)

        # Fallback to Stable Diffusion 2.1
        if StableDiffusionPipeline is not None:
            try:
                dtype_val = torch.float16 if torch.cuda.is_available() else torch.float32
                kwargs = _build_pretrained_kwargs(StableDiffusionPipeline.from_pretrained, dtype_val, self.hf_token)
                self.pipe = StableDiffusionPipeline.from_pretrained(model_to_load, **kwargs)
                self._apply_optimizations()
                self.backend = "local"
                logger.info("Loaded SD 2.1 pipeline on %s", "cuda" if torch.cuda.is_available() else "cpu")
                self.optimizations["model_loaded"] = "sd-2.1"
                return
            except Exception as e:
                logger.warning("Failed to load SD 2.1 pipeline locally: %s", e)

        # Final fallback: try InferenceClient if available
        if InferenceClient is not None:
            try:
                self.inference_client = InferenceClient(token=self.hf_token) if self.hf_token else InferenceClient()
                self.backend = "inference"
                logger.info("Falling back to Hugging Face Inference API")
                self.optimizations["backend"] = "inference"
                return
            except Exception as e:
                logger.error("InferenceClient init failed as final fallback: %s", e)

        raise RuntimeError("No available backend: check diffusers install, HF token, or set USE_HF_INFERENCE=1")

    def _apply_optimizations(self) -> None:
        """
        Apply a set of safe optimizations to a loaded pipeline (self.pipe).
        Each optimization is attempted and failures are logged but not fatal.
        """
        if self.pipe is None:
            return

        # 1) Enable attention slicing to reduce peak VRAM
        try:
            # many pipelines implement this
            if hasattr(self.pipe, "enable_attention_slicing"):
                self.pipe.enable_attention_slicing()
                self.optimizations["attention_slicing"] = "enabled"
                logger.info("Enabled attention slicing to reduce VRAM usage.")
        except Exception as e:
            logger.debug("enable_attention_slicing failed: %s", e)

        # 2) Try enabling xformers memory efficient attention (if installed)
        try:
            if hasattr(self.pipe, "enable_xformers_memory_efficient_attention"):
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                    self.optimizations["xformers"] = "enabled"
                    logger.info("Enabled xformers memory-efficient attention.")
                except Exception as e:
                    # Some diffusers/xformers combos raise; log and continue
                    logger.debug("enable_xformers_memory_efficient_attention failed: %s", e)
            else:
                # For older/newer versions, there might be helper util
                logger.debug("xformers hook not present on pipeline.")
        except Exception as e:
            logger.debug("xformers attempt unexpectedly failed: %s", e)

        # 3) Attempt model CPU offload (diffusers supports via accelerate offload integration)
        try:
            if hasattr(self.pipe, "enable_model_cpu_offload"):
                # This will offload parts of the model to CPU and pin / unpin as needed
                # Great for low VRAM GPUs, but requires accelerate >= 0.18+ in many setups
                self.pipe.enable_model_cpu_offload()
                self.optimizations["model_cpu_offload"] = "enabled"
                logger.info("Enabled model CPU offload to reduce VRAM pressure.")
            else:
                logger.debug("Model CPU offload not available on this pipeline build.")
        except Exception as e:
            logger.debug("enable_model_cpu_offload failed (continuing): %s", e)

        # 4) (Optional) set torch.backends.cudnn.benchmark for speed on fixed-size inputs
        try:
            torch.backends.cudnn.benchmark = True
            self.optimizations["cudnn_benchmark"] = "enabled"
        except Exception:
            pass

    def generate(self, prompt: str, steps: int = cfg.default_steps, guidance: float = cfg.default_guidance,
                 width: int = cfg.default_width, height: int = cfg.default_height, seed: Optional[int] = None):
        """
        Simple wrapper that runs the pipeline synchronously and returns a PIL image.
        Note: the app's generate stream uses MODEL.pipe directly to pass callback arguments
        for stepwise progress. This method remains useful for direct calls.
        """
        if self.backend == "local" and self.pipe is not None:
            generator = None
            if seed is not None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                generator = torch.Generator(device=device).manual_seed(int(seed))
            ctx = torch.autocast("cuda") if torch.cuda.is_available() else nullcontext()
            with ctx:
                out = self.pipe(prompt,
                                num_inference_steps=int(steps),
                                guidance_scale=float(guidance),
                                width=int(width),
                                height=int(height),
                                generator=generator)
                images = getattr(out, "images", None)
                if images:
                    return images[0]
                if isinstance(out, (list, tuple)) and out:
                    return out[0]
                raise RuntimeError("Unexpected pipeline output structure")
        if self.backend == "inference" and self.inference_client is not None:
            return self.inference_client.text_to_image(prompt, model=self.model_id)
        raise RuntimeError("Model backend not initialized.")
