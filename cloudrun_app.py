"""AetherArt Cloud Run demo — L4 GPU, SDXL + Hyper-SD 8-step + Ukiyo-e LoRA + ControlNet Union.

Serves the 4-mode curated app as a standard Gradio server bound to 0.0.0.0:$PORT.
Cloud Run injects $PORT; models load in a background thread so Gradio starts
immediately and the startup probe passes within seconds. generate() blocks until
the background load is complete.

MODES:
  hyper_8step (default) — Hyper-SD 8-step LoRA + Ukiyo-e LoRA composed, CFG=5, EulerDiscrete
  ukiyo_e               — Ukiyo-e LoRA only, 30-step DPM-Solver++
  controlnet_canny      — ControlNet Union, Canny edge conditioning
  controlnet_depth      — ControlNet Union, depth map conditioning

SAFETY: AETHERART_ENABLE_SAFETY=1 injected at deploy time via Cloud Run env var.
HF token: mounted from Secret Manager as HUGGINGFACEHUB_API_TOKEN — never baked in image.
"""

from __future__ import annotations

import os
import threading
import time

import gradio as gr
import torch
from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    StableDiffusionXLControlNetUnionPipeline,
    StableDiffusionXLPipeline,
)
from diffusers.models import ControlNetUnionModel
from PIL import Image as PILImage
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from aetherart.safety import apply_safety_checker, check_prompt

# ── Model / repo constants ──────────────────────────────────────────────────

_SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
_SDXL_VAE_FIX = "madebyollin/sdxl-vae-fp16-fix"
_CONTROLNET_UNION = "xinsir/controlnet-union-sdxl-1.0"
_DEPTH_ESTIMATOR = "LiheYoung/depth-anything-small-hf"
_HF_UKIYO_LORA_REPO = "gauravgandhi2411/aetherart-ukiyo-sdxl"
_HF_UKIYO_LORA_WEIGHTS = "ukiyo-e-sdxl-lora.safetensors"
_HYPER_SD_REPO = "ByteDance/Hyper-SD"
_HYPER_8STEP_WEIGHTS = "Hyper-SDXL-8steps-lora.safetensors"

# Diffusers 0.35 ControlNetUnion control_mode integers.
_CTYPE_TO_INT: dict[str, int] = {"depth": 1, "canny": 3}

# ── Background model loader ─────────────────────────────────────────────────
# Models load in a daemon thread so Gradio can bind the port immediately.
# Cloud Run's startup probe sees the HTTP server within seconds.
# generate() calls _models_ready.wait() so the first request blocks until ready.

_models_ready = threading.Event()

# Populated by _load_models(); referenced by generate() after the event is set.
vae: AutoencoderKL | None = None
sdxl_pipe: StableDiffusionXLPipeline | None = None
cn_pipe: StableDiffusionXLControlNetUnionPipeline | None = None
_depth_processor: AutoImageProcessor | None = None
_depth_model: AutoModelForDepthEstimation | None = None
_sched_dpm: DPMSolverMultistepScheduler | None = None
_sched_euler: EulerDiscreteScheduler | None = None


def _load_models() -> None:
    global vae, sdxl_pipe, cn_pipe, _depth_processor, _depth_model, _sched_dpm, _sched_euler

    try:
        print("[demo] Loading fp16-fix VAE…")
        vae = AutoencoderKL.from_pretrained(_SDXL_VAE_FIX, torch_dtype=torch.float16)

        print("[demo] Loading SDXL base pipeline…")
        sdxl_pipe = StableDiffusionXLPipeline.from_pretrained(
            _SDXL_MODEL,
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to("cuda")

        # apply_safety_checker is a no-op for SDXL; prompt blocklist is the primary guard.
        apply_safety_checker(sdxl_pipe)

        print("[demo] Loading Ukiyo-e LoRA from HF Hub…")
        sdxl_pipe.load_lora_weights(
            _HF_UKIYO_LORA_REPO,
            weight_name=_HF_UKIYO_LORA_WEIGHTS,
            adapter_name="ukiyo_e",
        )

        print("[demo] Loading Hyper-SD 8-step LoRA from HF Hub…")
        sdxl_pipe.load_lora_weights(
            _HYPER_SD_REPO,
            weight_name=_HYPER_8STEP_WEIGHTS,
            adapter_name="hyper_8step",
        )

        _base_sched_config = sdxl_pipe.scheduler.config
        _sched_dpm = DPMSolverMultistepScheduler.from_config(_base_sched_config)
        _sched_euler = EulerDiscreteScheduler.from_config(
            _base_sched_config, timestep_spacing="trailing"
        )

        # Boot state: Hyper-8step + Ukiyo-e LoRA composed (demo default).
        # 8-step chosen over 4-step: CFG-preserved so the LoRA composes and negative prompts work.
        sdxl_pipe.enable_lora()
        sdxl_pipe.set_adapters(["hyper_8step", "ukiyo_e"], adapter_weights=[1.0, 0.8])
        sdxl_pipe.scheduler = _sched_euler

        print("[demo] Loading ControlNetUnionModel…")
        cn_model = ControlNetUnionModel.from_pretrained(
            _CONTROLNET_UNION, torch_dtype=torch.float16
        )

        print("[demo] Building ControlNet pipeline…")
        cn_pipe = StableDiffusionXLControlNetUnionPipeline.from_pretrained(
            _SDXL_MODEL,
            controlnet=cn_model,
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to("cuda")

        print("[demo] Loading depth estimator…")
        _depth_processor = AutoImageProcessor.from_pretrained(_DEPTH_ESTIMATOR)
        _depth_model = AutoModelForDepthEstimation.from_pretrained(_DEPTH_ESTIMATOR).to("cuda")

        print("[demo] All models ready — container hot.")
    except Exception as exc:
        # Log and mark ready anyway so the UI can surface the error.
        print(f"[demo] FATAL: model loading failed — {exc}")
    finally:
        _models_ready.set()


threading.Thread(target=_load_models, daemon=True).start()


# ── Helpers ─────────────────────────────────────────────────────────────────


def _preprocess_canny(img: PILImage.Image) -> PILImage.Image:
    import cv2
    import numpy as np

    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return PILImage.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))


def _preprocess_depth(img: PILImage.Image) -> PILImage.Image:
    inputs = _depth_processor(images=img, return_tensors="pt").to("cuda")  # type: ignore[union-attr]
    with torch.no_grad():
        outputs = _depth_model(**inputs)  # type: ignore[union-attr]
    depth = outputs.predicted_depth
    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(1),
        size=img.size[::-1],
        mode="bicubic",
        align_corners=False,
    )
    out = depth.squeeze().cpu().numpy()
    out = (out - out.min()) / (out.max() - out.min() + 1e-8) * 255
    out = out.astype("uint8")
    return PILImage.fromarray(out[:, :, None].repeat(3, axis=2))


# ── Generation handler ───────────────────────────────────────────────────────


def generate(
    prompt: str,
    negative_prompt: str,
    mode: str,
    control_image: object,
    seed_val: object,
) -> tuple[object, str]:
    if not prompt or not prompt.strip():
        return None, "Please enter a prompt."

    blocked = check_prompt(prompt)
    if blocked:
        return None, f"**Blocked:** {blocked}"

    # Non-blocking check — Cloud Run drops idle SSE streams after ~60 s,
    # so we return immediately with a retry message rather than blocking.
    if not _models_ready.is_set():
        return None, (
            "⏳ Warming up — the demo loads its models on first use "
            "(~5-7 min after idle). Once ready, images generate in "
            "seconds. No need to click repeatedly."
        )

    if sdxl_pipe is None or cn_pipe is None:
        return None, "**Error:** Model loading failed at startup — check container logs."

    actual_seed = int(seed_val) if seed_val is not None else 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(actual_seed)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    try:
        if mode in ("controlnet_canny", "controlnet_depth"):
            if control_image is None:
                return None, f"**{mode}** requires a conditioning image (upload one above)."
            pil = (
                control_image
                if isinstance(control_image, PILImage.Image)
                else PILImage.fromarray(control_image)
            )
            pil = pil.resize((1024, 1024))
            ctrl_map = (
                _preprocess_canny(pil) if mode == "controlnet_canny" else _preprocess_depth(pil)
            )
            ctype_key = "canny" if mode == "controlnet_canny" else "depth"
            result = cn_pipe(
                prompt=prompt,
                control_image=ctrl_map,
                control_mode=[_CTYPE_TO_INT[ctype_key]],
                negative_prompt=negative_prompt or None,
                num_inference_steps=30,
                guidance_scale=7.5,
                controlnet_conditioning_scale=1.0,
                generator=generator,
                width=1024,
                height=1024,
            )
            img = result.images[0]

        elif mode == "ukiyo_e":
            sdxl_pipe.enable_lora()
            sdxl_pipe.set_adapters(["ukiyo_e"], adapter_weights=[1.0])
            sdxl_pipe.scheduler = _sched_dpm
            eff_prompt = f"ukyowood {prompt}" if "ukyowood" not in prompt else prompt
            eff_neg = ", ".join(
                filter(
                    None,
                    [
                        negative_prompt,
                        "text, watermark, calligraphy, signature, words, letters",
                    ],
                )
            )
            result = sdxl_pipe(
                prompt=eff_prompt,
                negative_prompt=eff_neg or None,
                num_inference_steps=30,
                guidance_scale=7.5,
                generator=generator,
                width=1024,
                height=1024,
            )
            img = result.images[0]

        else:  # hyper_8step — Hyper-8step + Ukiyo-e LoRA composed (demo default)
            sdxl_pipe.enable_lora()
            sdxl_pipe.set_adapters(["hyper_8step", "ukiyo_e"], adapter_weights=[1.0, 0.8])
            sdxl_pipe.scheduler = _sched_euler
            eff_prompt = f"ukyowood {prompt}" if "ukyowood" not in prompt else prompt
            eff_neg = ", ".join(
                filter(
                    None,
                    [
                        negative_prompt,
                        "text, watermark, calligraphy, signature, words, letters",
                    ],
                )
            )
            result = sdxl_pipe(
                prompt=eff_prompt,
                negative_prompt=eff_neg or None,
                num_inference_steps=8,
                guidance_scale=5.0,
                generator=generator,
                width=1024,
                height=1024,
            )
            img = result.images[0]

    except Exception as exc:
        return None, f"**Error:** {exc}"

    gen_time = time.time() - t0
    vram_mb = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    return img, f"Done — {gen_time:.1f}s — VRAM peak: {vram_mb:.0f} MB — seed {actual_seed}"


# ── Gradio UI ────────────────────────────────────────────────────────────────

with gr.Blocks(title="AetherArt — Cloud Run L4") as demo:
    gr.Markdown(
        "# AetherArt — Cloud Run Demo (L4)\n"
        "SDXL 1.0 · Ukiyo-e LoRA · Hyper-SD 8-step · ControlNet Union (Canny + Depth)  \n"
        "1024 × 1024 · Safety filter active."
    )

    with gr.Row():
        with gr.Column(scale=3):
            mode = gr.Radio(
                choices=[
                    ("Hyper-8step + Ukiyo-e LoRA — fast composed [default]", "hyper_8step"),
                    ("Ukiyo-e LoRA — 30-step DPM++, higher quality", "ukiyo_e"),
                    ("ControlNet Canny — edge conditioning", "controlnet_canny"),
                    ("ControlNet Depth — depth conditioning", "controlnet_depth"),
                ],
                value="hyper_8step",
                label="Generation Mode",
                info=(
                    "ControlNet modes require a conditioning image (expand the accordion below)."
                ),
            )
            prompt = gr.Textbox(
                value="ukyowood ukiyo-e woodblock print of Mount Fuji at dawn",
                placeholder="ukyowood ukiyo-e woodblock print of Mount Fuji at dawn",
                label="Prompt",
                lines=2,
            )
            negative_prompt = gr.Textbox(
                placeholder="blurry, low quality, watermark",
                label="Negative Prompt",
                lines=1,
            )
            with gr.Accordion("ControlNet conditioning image", open=False):
                gr.Markdown(
                    "Required only for ControlNet modes. Upload the source image to condition on."
                )
                control_image = gr.Image(
                    label="Conditioning Image",
                    type="pil",
                    sources=["upload"],
                )
            seed = gr.Number(value=42, label="Seed", precision=0)
            gr.Markdown(
                "> ⏳ **First image after idle may take ~2 minutes while models load on "
                "cold start.** Subsequent images are fast (~5 s Hyper, ~15 s LoRA / "
                "ControlNet). The demo scales to zero between uses — no idle cost."
            )
            gen_btn = gr.Button("Generate", variant="primary")
            status_md = gr.Markdown("Ready.")

        with gr.Column(scale=4):
            out_img = gr.Image(label="Generated Image (1024 × 1024)", interactive=False)

    gen_btn.click(
        fn=generate,
        inputs=[prompt, negative_prompt, mode, control_image, seed],
        outputs=[out_img, status_md],
        concurrency_limit=1,
    )

demo.queue(default_concurrency_limit=1, max_size=10)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True,
    )
