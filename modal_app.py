"""AetherArt Modal demo — A10G, SDXL + Hyper-SD 8-step + Ukiyo-e LoRA + ControlNet Union.

DEPLOYMENT (gated — run in order):
  Stage B (ephemeral test):
      modal serve modal_app.py
  Stage C (persistent, after approval):
      modal deploy modal_app.py

PREREQUISITES:
  1. Modal secret named "huggingface" with key HUGGINGFACEHUB_API_TOKEN must
     exist in your Modal dashboard (https://modal.com/secrets) before running.
  2. `modal` must be installed (requirements-dev.txt) and authenticated.

DEMO MODES (live):
  - Hyper-8step (default)  SDXL + 8-step LoRA, CFG=5, EulerDiscrete
  - Ukiyo-e LoRA           gauravgandhi2411/aetherart-ukiyo-sdxl, trigger "ukyowood"
  - SDXL Base              30-step DPM-Solver++, no LoRA
  - ControlNet Canny       xinsir/controlnet-union-sdxl-1.0, edge conditioning
  - ControlNet Depth       xinsir/controlnet-union-sdxl-1.0, depth conditioning

OUT OF SCOPE (documented local-only in README):
  - Flux (PR 14)
  - SD 2.1 / SDXL Turbo legacy paths
  - NF4 quantization (not needed on A10G 24 GB)

Safety: AETHERART_ENABLE_SAFETY=1 is baked into the image env; the prompt
blocklist in aetherart/safety.py runs on every request.
"""

from __future__ import annotations

import modal

# Runtime deps mirror requirements.txt but override torch to the CUDA 12.6
# build (pinned per PR 02a audit — Modal pins its own torch independent of
# local/CI).  Eval-only deps (hpsv2, image-reward) are intentionally excluded.
_RUNTIME_DEPS: list[str] = [
    "gradio==5.46.1",
    "diffusers==0.35.1",
    "transformers==4.56.2",
    "accelerate==1.10.1",
    "huggingface_hub[hf_xet]==0.35.0",
    "safetensors==0.6.2",
    "peft==0.19.1",
    "Pillow==11.3.0",
    "numpy==2.2.6",
    "opencv-python-headless==4.12.0.88",
    "python-dotenv==1.1.1",
    "bitsandbytes==0.49.2",
]

demo_image = (
    modal.Image.debian_slim(python_version="3.10")
    # Torch first so subsequent packages pin against the CUDA build.
    .pip_install(
        "torch==2.8.0+cu126",
        extra_index_url="https://download.pytorch.org/whl/cu126",
    )
    .pip_install(*_RUNTIME_DEPS)
    # Add aetherart package to /root in containers (on PYTHONPATH by default).
    # copy=True bakes it into the image layer so it's available on cold start.
    .add_local_python_source("aetherart", copy=True)
    # Safety guard active on the public demo; local dev leaves this unset.
    .env({"AETHERART_ENABLE_SAFETY": "1"})
)

app = modal.App("aetherart-demo")

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


@app.function(
    gpu="a10g",
    image=demo_image,
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=600,
    scaledown_window=300,
)
@modal.asgi_app()
def serve() -> object:
    """ASGI app factory — Modal calls this once per container.

    All model loading happens here.  The returned FastAPI/Gradio ASGI app
    handles every subsequent request without reloading weights.
    """
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
    from fastapi import FastAPI
    from PIL import Image as PILImage
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    from aetherart.safety import apply_safety_checker, check_prompt

    # ── SDXL base + fp16-fix VAE ────────────────────────────────────────────
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

    # apply_safety_checker is a no-op for SDXL; prompt blocklist is primary.
    apply_safety_checker(sdxl_pipe)

    # ── LoRA adapters on the base pipeline ──────────────────────────────────
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

    # Stash scheduler instances — swap in generate() per mode.
    _base_sched_config = sdxl_pipe.scheduler.config
    _sched_dpm = DPMSolverMultistepScheduler.from_config(_base_sched_config)
    _sched_euler = EulerDiscreteScheduler.from_config(
        _base_sched_config, timestep_spacing="trailing"
    )

    # Boot state: Hyper-8step + Ukiyo-e LoRA composed (demo default).
    # 8-step is the CFG-preserved variant specifically so it composes with the LoRA.
    sdxl_pipe.enable_lora()
    sdxl_pipe.set_adapters(["hyper_8step", "ukiyo_e"], adapter_weights=[1.0, 0.8])
    sdxl_pipe.scheduler = _sched_euler

    # ── ControlNet Union pipeline ────────────────────────────────────────────
    print("[demo] Loading ControlNetUnionModel…")
    cn_model = ControlNetUnionModel.from_pretrained(_CONTROLNET_UNION, torch_dtype=torch.float16)

    print("[demo] Building ControlNet pipeline…")
    cn_pipe = StableDiffusionXLControlNetUnionPipeline.from_pretrained(
        _SDXL_MODEL,
        controlnet=cn_model,
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to("cuda")

    # ── Depth estimator (pre-loaded so first request isn't slow) ────────────
    print("[demo] Loading depth estimator…")
    _depth_processor = AutoImageProcessor.from_pretrained(_DEPTH_ESTIMATOR)
    _depth_model = AutoModelForDepthEstimation.from_pretrained(_DEPTH_ESTIMATOR).to("cuda")

    print("[demo] All models ready — container hot.")

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _preprocess_canny(img: PILImage.Image) -> PILImage.Image:
        import cv2
        import numpy as np

        arr = np.array(img.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return PILImage.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))

    def _preprocess_depth(img: PILImage.Image) -> PILImage.Image:

        inputs = _depth_processor(images=img, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = _depth_model(**inputs)
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

    # ── Generation handler ───────────────────────────────────────────────────

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

            elif mode == "sdxl_base":
                sdxl_pipe.disable_lora()
                sdxl_pipe.scheduler = _sched_dpm
                result = sdxl_pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt or None,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    generator=generator,
                    width=1024,
                    height=1024,
                )
                img = result.images[0]

            else:  # hyper_8step — Hyper-8step + Ukiyo-e LoRA composed (demo default)
                # 8-step chosen over 4-step: CFG-preserved so Ukiyo-e LoRA composes
                # and negative prompts still work (4-step is CFG-free, LoRA clash).
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

    # ── Gradio UI ────────────────────────────────────────────────────────────

    with gr.Blocks(title="AetherArt — Modal A10G") as demo_ui:
        gr.Markdown(
            "# AetherArt — Modal Demo (A10G)\n"
            "SDXL 1.0 · Ukiyo-e LoRA · Hyper-SD 8-step · ControlNet Union (Canny + Depth)  \n"
            "1024 × 1024 · Safety filter active."
        )

        with gr.Row():
            with gr.Column(scale=3):
                mode = gr.Radio(
                    choices=[
                        ("Hyper-8step + Ukiyo-e LoRA — fast composed [default]", "hyper_8step"),
                        ("Ukiyo-e LoRA — 30-step DPM++, higher quality", "ukiyo_e"),
                        ("SDXL Base — 30-step DPM-Solver++, no LoRA", "sdxl_base"),
                        ("ControlNet Canny — edge conditioning", "controlnet_canny"),
                        ("ControlNet Depth — depth conditioning", "controlnet_depth"),
                    ],
                    value="hyper_8step",
                    label="Generation Mode",
                    info=(
                        "ControlNet modes require a conditioning image "
                        "(expand the accordion below)."
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
                        "Required only for ControlNet modes. "
                        "Upload the source image to condition on."
                    )
                    control_image = gr.Image(
                        label="Conditioning Image",
                        type="pil",
                        sources=["upload"],
                    )
                seed = gr.Number(value=42, label="Seed", precision=0)
                gr.Markdown(
                    "> ⏳ **First image after idle may take a few minutes while models "
                    "load.** Subsequent images are fast (~4 s Hyper, ~13 s LoRA / "
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

    demo_ui.queue(default_concurrency_limit=1, max_size=10)

    fastapi_app = FastAPI()
    return gr.mount_gradio_app(fastapi_app, demo_ui, path="/")
