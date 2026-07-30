"""FLUX.1-schnell pipeline loader for AetherArt.

FLUX.1-schnell (black-forest-labs/FLUX.1-schnell, Apache-2.0) is a distilled, guidance-free
12B-parameter model — see aetherart.flux_pipeline's caller in scripts/model_verdict_harness.py
for the correct guidance_scale=0.0 / num_inference_steps=4 inference defaults, verified against
diffusers' own FluxPipeline docstring example rather than assumed.

Local feasibility note (measured 2026-07-30, RTX 3070 Laptop, 8.59GB VRAM): the diffusers-format
FLUX.1-schnell repo requires ~33.7GB of disk (23.78GB transformer + 9.53GB text_encoder_2/T5 +
0.25GB text_encoder + 0.17GB vae, per HfApi.model_info file listing), which exceeded this
machine's ~25.5GB free disk at measurement time — a disk-space blocker that precedes the VRAM
question entirely, so bf16+cpu-offload below is diffusers' documented low-VRAM path but has NOT
been empirically GPU-verified on this machine. Revisit with more free disk before the actual eval
run.

Auth note: FLUX.1-schnell is a gated HF repo (license Apache-2.0, but still requires the
account to click "Agree" on the model page, plus an authenticated token). This project's
HF_TOKEN is a fine-grained token scoped ONLY to write access on the three published aetherart
repos and cannot read third-party gated repos. A separate HF_READ_TOKEN (account-wide read
scope) is used here instead, explicitly, never implicitly picked up from the environment by
huggingface_hub's default HF_TOKEN lookup — every from_pretrained call below passes it
explicitly as `token=`.
"""

from __future__ import annotations

import gc
import os

from .logger import get_logger

logger = get_logger(__name__)


def _hf_read_token() -> str:
    token = os.environ.get("HF_READ_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_READ_TOKEN is not set — FLUX.1-schnell is a gated repo and needs an "
            "account-wide read token (separate from the write-scoped HF_TOKEN used to "
            "publish the three aetherart adapters)."
        )
    return token


try:
    from diffusers import FluxPipeline
except Exception:  # pragma: no cover
    FluxPipeline = None  # type: ignore[assignment, misc]

try:
    from diffusers import BitsAndBytesConfig, FluxTransformer2DModel
except Exception:  # pragma: no cover
    BitsAndBytesConfig = None  # type: ignore[assignment, misc]
    FluxTransformer2DModel = None  # type: ignore[assignment, misc]

try:
    from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
    from transformers import T5EncoderModel
except Exception:  # pragma: no cover
    TransformersBitsAndBytesConfig = None  # type: ignore[assignment, misc]
    T5EncoderModel = None  # type: ignore[assignment, misc]

FLUX_SCHNELL_MODEL = "black-forest-labs/FLUX.1-schnell"


def load_flux_schnell() -> FluxPipeline:
    """Load FLUX.1-schnell in bf16 with CPU offload — diffusers' documented low-VRAM path.

    This is the (a) config from the feasibility check: bf16 dtype + enable_model_cpu_offload(),
    no quantization. See module docstring for why this hasn't been empirically GPU-verified on
    this machine (disk-space blocker, not a VRAM finding).
    """
    import torch

    if FluxPipeline is None:
        raise RuntimeError("diffusers is not installed; cannot load FLUX pipeline")

    logger.info("Loading FLUX.1-schnell from '%s'...", FLUX_SCHNELL_MODEL)
    pipe = FluxPipeline.from_pretrained(
        FLUX_SCHNELL_MODEL, torch_dtype=torch.bfloat16, token=_hf_read_token()
    )
    pipe.enable_model_cpu_offload()
    logger.info("Enabled model CPU offload")
    logger.info("FLUX.1-schnell pipeline ready")
    return pipe


def load_flux_schnell_quantized() -> FluxPipeline:
    """Load FLUX.1-schnell with the transformer and T5 text encoder NF4 4-bit quantized
    (bitsandbytes), no CPU offload — the whole pipeline stays GPU-resident.

    Alternate to load_flux_schnell() above: added after the GCP eval run's own probe measured
    bf16+cpu-offload as too slow for a 90-image run (see scripts/gcp_startup_flux_eval.sh's
    probe step and its logged before/after numbers) — CPU offload pays a large per-step
    host<->device transfer cost that NF4 quantization avoids by fitting the whole model
    (12B params, ~half the footprint of bf16) resident in the L4's 24GB VRAM instead.
    """
    import torch

    if FluxPipeline is None or FluxTransformer2DModel is None:
        raise RuntimeError("diffusers is not installed; cannot load FLUX pipeline")
    if T5EncoderModel is None:
        raise RuntimeError("transformers is not installed; cannot load FLUX T5 text encoder")

    logger.info(
        "Loading FLUX.1-schnell (NF4-quantized transformer + T5) from '%s'...", FLUX_SCHNELL_MODEL
    )
    read_token = _hf_read_token()
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
    transformer = FluxTransformer2DModel.from_pretrained(
        FLUX_SCHNELL_MODEL,
        subfolder="transformer",
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        token=read_token,
    )

    text_encoder_quant_config = TransformersBitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4"
    )
    text_encoder_2 = T5EncoderModel.from_pretrained(
        FLUX_SCHNELL_MODEL,
        subfolder="text_encoder_2",
        quantization_config=text_encoder_quant_config,
        torch_dtype=torch.bfloat16,
        token=read_token,
    )

    pipe = FluxPipeline.from_pretrained(
        FLUX_SCHNELL_MODEL,
        transformer=transformer,
        text_encoder_2=text_encoder_2,
        torch_dtype=torch.bfloat16,
        token=read_token,
    )
    pipe.to("cuda")
    logger.info("FLUX.1-schnell (NF4-quantized) pipeline ready, GPU-resident (no CPU offload)")
    return pipe


def release_flux_pipeline(pipe: FluxPipeline) -> None:
    """Delete a pipeline and reclaim GPU memory."""
    del pipe
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
