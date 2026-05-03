import json
import os
import random
import time
from pathlib import Path
from typing import Any, Generator, Optional

import gradio as gr
import torch
from PIL import Image, ImageDraw, ImageFont

from aetherart.model import AetherModel
from aetherart.logger import get_logger
from aetherart.config import cfg
from aetherart import metadata as meta
from aetherart import controlnet as cn
from aetherart import lora as lora_mod
from aetherart import lcm as lcm_mod
from aetherart import sdxl_turbo as turbo_mod
from aetherart import quantization as quant_mod

logger = get_logger(__name__)

_HAS_GPU: bool = torch.cuda.is_available()

# Model singletons — cached across requests
MODEL = AetherModel()
_turbo_pipe = None
_quant_pipes: dict = {}
_active_lora_name: str = "none"

# ── Sample gallery data ─────────────────────────────────────────────────────

_TIER_LABELS = {
    "standard_fp16":    "Standard fp16 — 30-step DPM-Solver++",
    "turbo":            "SDXL Turbo — 1-step adversarial diffusion",
    "lora_ukiyo_e":     "Ukiyo-e LoRA — rank-8, checkpoint-1000",
    "controlnet_canny": "ControlNet Canny — edge conditioning",
    "controlnet_depth": "ControlNet Depth — depth conditioning",
    "quantized_8bit":   "8-bit INT8 — bitsandbytes quantized U-Net",
    "quantized_4bit":   "4-bit NF4 — bitsandbytes quantized U-Net",
}
_TIER_ORDER = list(_TIER_LABELS.keys())


def _load_samples() -> dict:
    samples_dir = Path("docs/samples")
    if not samples_dir.exists():
        return {}
    result = {}
    for tier in _TIER_ORDER:
        tier_dir = samples_dir / tier
        if not tier_dir.is_dir():
            continue
        entries = []
        for png_path in sorted(tier_dir.glob("*.png")):
            stem = png_path.stem
            if any(stem.endswith(s) for s in ("_source", "_canny_map", "_depth_map")):
                continue
            meta_file = png_path.with_suffix(".json")
            caption = tier
            if meta_file.exists():
                try:
                    m = json.loads(meta_file.read_text(encoding="utf-8"))
                    t = m.get("inference_time_rtx3070_s", "?")
                    v = m.get("vram_peak_mb", "?")
                    p = m.get("original_prompt") or m.get("prompt", "")
                    caption = f"{t}s / {v} MB VRAM — {p[:55]}"
                except Exception:
                    pass
            entries.append((str(png_path), caption))
        if entries:
            result[tier] = entries
    return result


_SAMPLES = _load_samples()


# ── GPU generation functions ────────────────────────────────────────────────

def _run_sd21(
    prompt: str,
    negative_prompt: str,
    model_choice: str,
    steps: int,
    guidance: float,
    width: int,
    height: int,
    seed: int,
    lora_name: str,
    lora_alpha: float,
    auto_trigger: bool,
    speed_mode: str,
    memory_mode: str,
    control_type: str,
    control_image: Optional[Any],
    control_scale: float,
    canny_low: int,
    canny_high: int,
    step_callback: Optional[callable] = None,
) -> tuple:
    """SD 2.1 generation (standard / LCM / quantized). Returns (PIL.Image, gen_time_s, vram_mb)."""
    global _quant_pipes, _active_lora_name

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    # Select pipeline
    if memory_mode in ("8bit", "4bit"):
        bits = int(memory_mode[0])
        if memory_mode not in _quant_pipes:
            _quant_pipes[memory_mode] = quant_mod.load_sd21_quantized(bits=bits)
            from diffusers import DPMSolverMultistepScheduler
            _quant_pipes[memory_mode].scheduler = DPMSolverMultistepScheduler.from_config(
                _quant_pipes[memory_mode].scheduler.config
            )
        active_pipe = _quant_pipes[memory_mode]
        use_lora = False
    else:
        if MODEL.backend is None:
            try:
                MODEL.init(model_choice="sdxl" if model_choice == "sdxl" else None)
            except Exception as e:
                logger.warning("Model init warning: %s", e)
        active_pipe = MODEL.pipe if MODEL.backend == "local" else None
        use_lora = (MODEL.backend == "local")

    # Apply / restore LCM scheduler on active_pipe (not MODEL.pipe)
    # Note: no LCM-LoRA for SD 2.1 (only SD 1.5/SDXL); scheduler-only is correct here.
    if speed_mode == "fast_lcm":
        run_steps = lcm_mod.LCM_STEPS
        run_guidance = lcm_mod.LCM_GUIDANCE
        if active_pipe is not None and not lcm_mod.is_lcm_scheduler(active_pipe):
            lcm_mod.apply_lcm_mode(active_pipe)
    else:
        run_steps = steps
        run_guidance = guidance
        if active_pipe is not None and lcm_mod.is_lcm_scheduler(active_pipe):
            lcm_mod.restore_standard_mode(active_pipe)

    def _on_step_end(pipe, step_i, t, callback_kwargs):
        if step_callback is not None:
            step_callback(step_i + 1, run_steps)
        return callback_kwargs

    # LoRA trigger-token injection
    effective_prompt = prompt
    effective_negative = negative_prompt or ""
    if use_lora and lora_name != "none" and auto_trigger:
        trigger = lora_mod.get_trigger_token(lora_name)
        if trigger and trigger not in effective_prompt:
            effective_prompt = f"{trigger} {effective_prompt}"
        lora_neg = lora_mod.get_default_negative(lora_name)
        if lora_neg:
            effective_negative = (
                f"{effective_negative}, {lora_neg}" if effective_negative else lora_neg
            )

    use_controlnet = control_type != "none" and control_image is not None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(seed)

    if active_pipe is not None:
        if not use_controlnet and use_lora:
            if _active_lora_name != lora_name:
                lora_mod.load_lora(active_pipe, lora_name, lora_alpha)
                _active_lora_name = lora_name
            elif lora_name != "none":
                try:
                    active_pipe.set_adapters(["default"], adapter_weights=[lora_alpha])
                except Exception:
                    pass

        if use_controlnet:
            ctrl_img = control_image.resize((width, height), Image.LANCZOS)
            ctrl_map = cn.preprocess(ctrl_img, control_type, canny_low, canny_high)
            cn_lora = lora_name if lora_name != "none" and use_lora else None
            pipe = cn.get_pipeline(control_type, lora_name=cn_lora, lora_alpha=lora_alpha)
            out = pipe(
                effective_prompt,
                image=ctrl_map,
                negative_prompt=effective_negative or None,
                num_inference_steps=run_steps,
                guidance_scale=run_guidance,
                width=width,
                height=height,
                generator=generator,
                controlnet_conditioning_scale=float(control_scale),
                callback_on_step_end=_on_step_end,
                callback_on_step_end_tensor_inputs=[],
            )
        else:
            out = active_pipe(
                effective_prompt,
                negative_prompt=effective_negative or None,
                num_inference_steps=run_steps,
                guidance_scale=run_guidance,
                width=width,
                height=height,
                generator=generator,
                callback_on_step_end=_on_step_end,
                callback_on_step_end_tensor_inputs=[],
            )

        images = getattr(out, "images", None)
        if images:
            img = images[0]
        elif isinstance(out, (list, tuple)) and out:
            img = out[0]
        else:
            raise RuntimeError("Unexpected pipeline output structure.")

    elif MODEL.backend == "inference" and getattr(MODEL, "inference_client", None) is not None:
        img = MODEL.inference_client.text_to_image(prompt, model=MODEL.model_id)
    else:
        raise RuntimeError("No model backend available. Check logs for details.")

    gen_time = time.time() - t0
    vram_mb = (torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else 0.0
    return img, gen_time, vram_mb


def _run_turbo(
    prompt: str,
    negative_prompt: str,
    seed: int,
    width: int,
    height: int,
) -> tuple:
    """SDXL Turbo 1-step generation. Returns (PIL.Image, gen_time_s, vram_mb)."""
    global _turbo_pipe

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    try:
        if _turbo_pipe is None:
            _turbo_pipe = turbo_mod.load_turbo_pipeline()
        img, _ = turbo_mod.generate_turbo(
            _turbo_pipe,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            width=width,
            height=height,
        )
    except torch.cuda.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _turbo_pipe = None
        raise RuntimeError(
            "SDXL Turbo ran out of VRAM. Try Standard or LCM mode which require less memory."
        )

    gen_time = time.time() - t0
    vram_mb = (torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else 0.0
    return img, gen_time, vram_mb


# ── Gradio event handlers ───────────────────────────────────────────────────

def estimate_cpu_time(width: int, height: int, num_steps: int) -> int:
    """Rough CPU generation time estimate in minutes (calibrated at 512×512×20 ≈ 12 min)."""
    pixel_factor = (width * height) / (512 * 512)
    step_factor = num_steps / 20
    return max(1, int(12 * pixel_factor * step_factor))


def generate_stream(
    prompt: str,
    negative_prompt: str,
    model_choice: str,
    steps: int,
    guidance: float,
    width: int,
    height: int,
    seed: Optional[int],
    control_image: Optional[Any] = None,
    control_type: str = "none",
    control_scale: float = 1.0,
    canny_low: int = 100,
    canny_high: int = 200,
    lora_name: str = "none",
    lora_alpha: float = 1.0,
    auto_trigger: bool = True,
    speed_mode: str = "standard",
    memory_mode: str = "fp16",
    progress: gr.Progress = gr.Progress(),
) -> Generator[tuple[Optional[Image.Image], str], None, None]:
    """Stream a status message, run generation, yield the final image."""
    try:
        if not prompt or not prompt.strip():
            yield None, "Please enter a prompt."
            return
        if int(width) * int(height) > 1024 * 1024 * 3:
            yield None, "Requested image size too large. Choose a smaller size."
            return

        if speed_mode == "fast_lcm" and not _HAS_GPU:
            yield None, "**LCM Fast mode requires a GPU and is not available on this CPU Space.**"
            return
        if speed_mode == "turbo" and not _HAS_GPU:
            yield None, "**SDXL Turbo requires a GPU and is not available on this CPU Space.**"
            return
        if memory_mode in ("8bit", "4bit") and not _HAS_GPU:
            yield None, "**8-bit and 4-bit quantization require a CUDA GPU.**"
            return

        _MIN_SIZE_CPU = 512
        if not _HAS_GPU and (int(width) < _MIN_SIZE_CPU or int(height) < _MIN_SIZE_CPU):
            yield gr.update(value=None), (
                f"⚠️ Resolution too low for SD 2.1 on CPU. "
                f"Minimum: {_MIN_SIZE_CPU}×{_MIN_SIZE_CPU}. "
                f"SD 2.1 was trained at 768×768; outputs below 512 produce incoherent results "
                f"due to cross-attention mismatch. Please raise resolution or use the "
                f"**Sample Outputs** tab for examples."
            )
            return

        actual_seed = int(seed) if seed is not None else random.randint(0, 2**32 - 1)
        mode_label = {"fast_lcm": "LCM 4-step", "turbo": "Turbo 1-step"}.get(
            speed_mode, f"{steps}-step"
        )

        if not _HAS_GPU:
            est_min = estimate_cpu_time(int(width), int(height), int(steps))
            yield gr.update(value=None), (
                f"⏳ Estimated wait: ~{est_min} minutes on CPU. "
                f"Generation in progress… please don't refresh."
            )
        elif speed_mode == "turbo" and _turbo_pipe is None:
            yield None, "**Downloading SDXL Turbo (~6.7 GB on first use) — please wait...**"
        else:
            yield None, f"**Generating ({mode_label}, seed {actual_seed})...**"

        def _step_cb(done: int, total: int) -> None:
            progress(done / total, desc=f"Step {done}/{total}")

        if speed_mode == "turbo":
            progress(0, desc="Running SDXL Turbo (1 step)…")
            img, gen_time, vram_mb = _run_turbo(
                prompt=prompt,
                negative_prompt=negative_prompt or "",
                seed=actual_seed,
                width=int(width),
                height=int(height),
            )
        else:
            effective_steps = lcm_mod.LCM_STEPS if speed_mode == "fast_lcm" else int(steps)
            progress(0, desc=f"Generating ({mode_label}, seed {actual_seed})…")
            img, gen_time, vram_mb = _run_sd21(
                prompt=prompt,
                negative_prompt=negative_prompt or "",
                model_choice=model_choice,
                steps=effective_steps,
                guidance=float(guidance),
                width=int(width),
                height=int(height),
                seed=actual_seed,
                lora_name=lora_name,
                lora_alpha=float(lora_alpha),
                auto_trigger=bool(auto_trigger),
                speed_mode=speed_mode,
                memory_mode=memory_mode,
                control_type=control_type,
                control_image=control_image,
                control_scale=float(control_scale),
                canny_low=int(canny_low),
                canny_high=int(canny_high),
                step_callback=_step_cb,
            )

    except Exception as e:
        logger.exception("generate_stream failed")
        yield None, f"**Error:** {e}"
        return

    # Save image with sidecar metadata
    scheduler_name = "ADD-1step" if speed_mode == "turbo" else "unknown"
    if speed_mode != "turbo" and MODEL.backend == "local" and getattr(MODEL, "pipe", None) is not None:
        try:
            scheduler_name = MODEL.pipe.scheduler.__class__.__name__
        except Exception:
            pass

    md: dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": negative_prompt or "",
        "seed": actual_seed,
        "scheduler": scheduler_name,
        "steps": steps,
        "guidance": guidance,
        "width": width,
        "height": height,
        "model_id": MODEL.model_id,
        "lora_name": lora_name if lora_name != "none" else None,
        "lora_alpha": lora_alpha if lora_name != "none" else None,
        "controlnet_type": control_type if control_type != "none" else None,
        "controlnet_scale": float(control_scale) if control_type != "none" else None,
        "git_commit": meta.get_git_commit(),
        "generation_time_seconds": round(gen_time, 2),
        "vram_peak_mb": round(vram_mb, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{actual_seed}.png"
    meta.save_image_with_metadata(img, out_path, md)

    latency_str = f"{gen_time:.1f}s"
    vram_str = f" — VRAM: {vram_mb:.0f} MB" if vram_mb > 0 else ""
    progress(1.0, desc="Done!")
    yield img, f"**Done — {mode_label} — {latency_str}{vram_str}** — Saved: `{out_path}`"


def reload_model(choice: str) -> str:
    try:
        MODEL.init(model_choice="sdxl" if choice == "sdxl" else None)
        return f"{'SDXL' if choice == 'sdxl' else 'SD 2.1'} loaded successfully."
    except Exception as e:
        logger.exception("reload_model failed")
        return f"Failed to load {choice}: {e}"


def load_from_png(file_path: str | None) -> tuple:
    empty = ("", "", None, cfg.default_steps, cfg.default_guidance, cfg.default_width, cfg.default_height)
    if file_path is None:
        return empty
    try:
        md = meta.load_metadata_from_image(file_path)
        if not md:
            return empty
        seed_val = int(md["seed"]) if md.get("seed") else None
        return (
            md.get("prompt", ""),
            md.get("negative_prompt", ""),
            seed_val,
            int(md.get("steps", cfg.default_steps)),
            float(md.get("guidance", cfg.default_guidance)),
            int(md.get("width", cfg.default_width)),
            int(md.get("height", cfg.default_height)),
        )
    except Exception as e:
        logger.warning("Could not read PNG metadata: %s", e)
        return empty


def preview_control_map(
    image: Optional[Any],
    control_type: str,
    canny_low: int,
    canny_high: int,
) -> Optional[Image.Image]:
    if image is None or control_type == "none":
        return None
    try:
        pil_img = image if isinstance(image, Image.Image) else Image.fromarray(image)
        return cn.preprocess(pil_img, control_type, int(canny_low), int(canny_high))
    except Exception as e:
        logger.warning("Control map preview failed: %s", e)
        return None


def make_placeholder_image(
    w: int = 512, h: int = 512, text: str = "Your generated image will appear here"
) -> Image.Image:
    img = Image.new("RGB", (w, h), color=(230, 230, 230))
    draw = ImageDraw.Draw(img)
    for i in range(h):
        shade = 230 - int(i * (30 / h))
        draw.line([(0, i), (w, i)], fill=(shade, shade, shade))

    border_color = (180, 180, 180)
    margin = 20
    for x in range(margin, w - margin, 15):
        draw.line([(x, margin), (x + 8, margin)], fill=border_color)
        draw.line([(x, h - margin), (x + 8, h - margin)], fill=border_color)
    for y in range(margin, h - margin, 15):
        draw.line([(margin, y), (margin, y + 8)], fill=border_color)
        draw.line([(w - margin, y), (w - margin, y + 8)], fill=border_color)

    try:
        font_size = max(14, int(min(w, h) / 20))
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    lines = text.split("\n")
    line_heights = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_heights.append(bbox[3] - bbox[1])

    y = (h - sum(line_heights)) // 2
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        x = (w - (bbox[2] - bbox[0])) // 2
        draw.text((x, y), line, fill=(60, 60, 60), font=font)
        y += line_heights[i]

    return img


# ── UI ─────────────────────────────────────────────────────────────────────

_PLACEHOLDER = make_placeholder_image(512, 512, "Your generated image will appear here")

# Adjust defaults for CPU so first-time visitors don't wait 15 minutes
_default_steps = cfg.default_steps if _HAS_GPU else 20
_default_w     = cfg.default_width  if _HAS_GPU else 512
_default_h     = cfg.default_height if _HAS_GPU else 512

# Feature availability based on hardware — LCM on CPU produces blurry output
# (no SD 2.1 LCM-LoRA exists; scheduler-only at 4 steps + float32 fallback degrades quality)
_speed_choices = ["standard"] + (["fast_lcm", "turbo"] if _HAS_GPU else [])
_memory_choices = ["fp16"] + (["8bit", "4bit"] if _HAS_GPU else [])

with gr.Blocks() as demo:
    gr.Markdown("# AetherArt — Production-Grade Diffusion on Consumer GPUs")

    # ── CPU / GPU banner ────────────────────────────────────────────────
    if not _HAS_GPU:
        gr.Markdown(
            "> ℹ️ **Free CPU tier** — generation works but is slow (~12–15 min at 512×512).  \n"
            ">  \n"
            "> SD 2.1 was trained at 768×768; defaults here are set to **512×512** "
            "(minimum for coherent output). Higher resolution = better quality but "
            "proportionally longer wait.  \n"
            ">  \n"
            "> See the **Sample Outputs** tab for instant viewing of 20 results from a local "
            "RTX 3070, or clone the "
            "[repo](https://github.com/gaurav-gandhi-2411/AetherArt) "
            "for full GPU acceleration."
        )
    else:
        gr.Markdown(
            "SD 2.1 with Ukiyo-e LoRA · ControlNet (Canny + Depth) · "
            "LCM 4-step fast generation · SDXL Turbo · 4-bit/8-bit quantization."
        )

    with gr.Tabs():
        # ── Generate tab ──────────────────────────────────────────────
        with gr.Tab("Generate"):
            with gr.Row():
                with gr.Column(scale=3):
                    model_choice = gr.Dropdown(
                        choices=["sd-2.1", "sdxl"], value="sd-2.1", label="Model"
                    )
                    prompt = gr.Textbox(
                        placeholder="A modern studio portrait of an astronaut",
                        label="Prompt", lines=2
                    )
                    negative_prompt = gr.Textbox(
                        placeholder="blurry, low quality", label="Negative Prompt", lines=1
                    )
                    steps = gr.Slider(10, 60, value=_default_steps, label="Steps")
                    guidance = gr.Slider(
                        1.0, 15.0, value=cfg.default_guidance, step=0.5, label="Guidance"
                    )
                    if not _HAS_GPU:
                        width  = gr.Slider(minimum=512, maximum=768, value=_default_w, step=64, label="Width")
                        height = gr.Slider(minimum=512, maximum=768, value=_default_h, step=64, label="Height")
                    else:
                        width  = gr.Slider(minimum=512, maximum=1024, value=_default_w, step=64, label="Width")
                        height = gr.Slider(minimum=512, maximum=1024, value=_default_h, step=64, label="Height")
                    seed = gr.Number(value=None, label="Seed (blank = random)", precision=0)
                    gen_btn = gr.Button("Generate", variant="primary")
                    reload_btn = gr.Button("Reload Model")
                    status_md = gr.Markdown("Ready.")

                with gr.Column(scale=4):
                    out_img = gr.Image(
                        label="Generated Image", value=_PLACEHOLDER, interactive=False
                    )

            with gr.Accordion("Recreate from PNG", open=False):
                gr.Markdown(
                    "Upload a previously generated PNG to restore its prompt, seed, and settings."
                )
                png_upload = gr.File(
                    label="Upload PNG", file_types=[".png"], type="filepath"
                )
                load_btn = gr.Button("Load Settings from PNG")

            with gr.Accordion("ControlNet Conditioning", open=False):
                gr.Markdown(
                    "Upload a conditioning image to guide the generation. "
                    "**Canny** extracts edges; **Depth** estimates a depth map. "
                    "Leave type as *none* to use standard generation."
                )
                with gr.Row():
                    with gr.Column():
                        control_image = gr.Image(
                            label="Conditioning Image", type="pil", sources=["upload"]
                        )
                        control_type = gr.Radio(
                            ["none", "canny", "depth"], value="none",
                            label="Conditioning Type"
                        )
                        control_scale = gr.Slider(
                            0.1, 2.0, value=1.0, step=0.05, label="Conditioning Scale"
                        )
                    with gr.Column():
                        canny_low = gr.Slider(
                            50, 200, value=100, step=10, label="Canny: Low Threshold"
                        )
                        canny_high = gr.Slider(
                            100, 300, value=200, step=10, label="Canny: High Threshold"
                        )
                        control_preview = gr.Image(
                            label="Control Map Preview", interactive=False
                        )
                        preview_btn = gr.Button("Preview Control Map")

            with gr.Accordion("LoRA Style", open=False):
                gr.Markdown(
                    "Apply a fine-tuned style adapter. The trigger token and negative prompt "
                    "are managed automatically when *Auto-prepend trigger token* is checked. "
                    "Not supported in Turbo mode."
                )
                lora_name = gr.Radio(
                    choices=["none", "ukiyo-e"],
                    value="none",
                    label="Style adapter",
                    info="ukiyo-e: Japanese woodblock print style (rank-8, SD 2.1, 80 images)",
                )
                lora_alpha = gr.Slider(
                    0.1, 1.5, value=1.0, step=0.1,
                    label="LoRA strength (alpha)",
                    info="1.0 = full strength · >1 = exaggerated style · <1 = subtle blend",
                )
                auto_trigger = gr.Checkbox(
                    value=True,
                    label="Auto-prepend trigger token",
                    info="Adds 'ukyowood' to your prompt and the LoRA's default negative",
                )

            with gr.Accordion("Memory / VRAM Mode", open=False):
                _mem_desc = (
                    "**fp16** (default) — full precision U-Net, ~4.5 GB VRAM peak.  \n"
                    "**8-bit INT8** — bitsandbytes quantized U-Net, ~2.5 GB VRAM. *GPU only.*  \n"
                    "**4-bit NF4** — aggressively quantized, ~1.5 GB VRAM. *GPU only.*  \n"
                    "Quantized pipelines are cached after first load. LoRA is disabled in quantized mode."
                )
                if not _HAS_GPU:
                    _mem_desc += (
                        "\n\n> **8-bit and 4-bit modes are not available on this CPU Space.** "
                        "Run locally with a CUDA GPU to use them."
                    )
                gr.Markdown(_mem_desc)
                memory_mode = gr.Radio(
                    choices=_memory_choices,
                    value="fp16",
                    label="U-Net precision",
                    info="fp16: default quality"
                    + (" · 8bit: balanced memory · 4bit: minimum VRAM" if _HAS_GPU else
                       " (8bit/4bit require GPU)"),
                )

            with gr.Accordion("Generation Speed Mode", open=True):
                _speed_desc = (
                    "**Standard** — 30-step DPM-Solver++, best quality"
                    + (" (~3 s on A10G / ~12 s on RTX 3070)." if _HAS_GPU
                       else " (~12–15 min at 512×512 on CPU).")
                    + "  \n"
                )
                if _HAS_GPU:
                    _speed_desc += (
                        "**Fast (LCM)** — 4-step LCMScheduler, ~5.8× faster, moderate quality reduction. "
                        "Uses scheduler-only LCM (no LCM-LoRA exists for SD 2.1).  \n"
                        "**Turbo (SDXL)** — 1-step adversarial diffusion. "
                        "Requires first-use download (~6.7 GB). LoRA/ControlNet not supported."
                    )
                else:
                    _speed_desc += (
                        "LCM Fast and SDXL Turbo modes available locally only — "
                        "see the **Sample Outputs** tab for examples generated on RTX 3070."
                    )
                gr.Markdown(_speed_desc)
                speed_mode = gr.Radio(
                    choices=_speed_choices,
                    value="standard",
                    label="Speed mode",
                    info="standard: 30-step DPM++"
                    + " · fast_lcm: 4-step LCM"
                    + (" · turbo: 1-step SDXL Turbo (GPU)" if _HAS_GPU else ""),
                )

            # Event wiring
            gen_btn.click(
                fn=generate_stream,
                inputs=[
                    prompt, negative_prompt, model_choice, steps, guidance, width, height, seed,
                    control_image, control_type, control_scale, canny_low, canny_high,
                    lora_name, lora_alpha, auto_trigger, speed_mode, memory_mode,
                ],
                outputs=[out_img, status_md],
                concurrency_limit=1,
            )
            reload_btn.click(fn=reload_model, inputs=[model_choice], outputs=[status_md])
            load_btn.click(
                fn=load_from_png,
                inputs=[png_upload],
                outputs=[prompt, negative_prompt, seed, steps, guidance, width, height],
            )
            preview_btn.click(
                fn=preview_control_map,
                inputs=[control_image, control_type, canny_low, canny_high],
                outputs=[control_preview],
            )

        # ── Sample Outputs tab ────────────────────────────────────────
        with gr.Tab("Sample Outputs"):
            gr.Markdown(
                "## Pre-generated samples\n"
                "All images generated locally on **RTX 3070 8 GB** with seed 42, 512×512. "
                "Caption format: `time / VRAM — prompt`."
            )
            if _SAMPLES:
                for tier, entries in _SAMPLES.items():
                    label = _TIER_LABELS.get(tier, tier)
                    with gr.Accordion(label, open=(tier == "standard_fp16")):
                        gr.Gallery(
                            value=entries,
                            label=label,
                            show_label=False,
                            columns=4,
                            height=340,
                            object_fit="contain",
                        )
            else:
                gr.Markdown(
                    "Sample images not yet generated.  \n"
                    "Run `python scripts/generate_samples.py` locally to populate this tab."
                )


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    share = os.getenv("SHARE", "false").lower() == "true"
    logger.info("Starting Gradio app on http://%s:%d (share=%s)", host, port, share)
    demo.queue(default_concurrency_limit=1, max_size=20)
    demo.launch(server_name=host, server_port=port, share=share)


if __name__ == "__main__":
    main()
