import os
import random
import time
import threading
from pathlib import Path
from typing import Any, Generator, Optional

import gradio as gr
from PIL import Image, ImageDraw, ImageFont

from aetherart.model import AetherModel
from aetherart.logger import get_logger
from aetherart.config import cfg
from aetherart import metadata as meta

logger = get_logger(__name__)

# Model instance — .init() is deferred to first generation or explicit reload
MODEL = AetherModel()


class GenerationState:
    """Thread-safe container for a single in-flight generation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.running: bool = False
        self.percent: int = 0
        self.step: int = 0
        self.total_steps: int = cfg.default_steps
        self.status_msgs: list[str] = []
        self.result_image: Optional[Any] = None
        self.error: Optional[str] = None
        self.generation_time: Optional[float] = None
        self.vram_peak_mb: Optional[float] = None
        self.gen_opts: dict[str, Any] = {}

    def push_status(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        with self._lock:
            self.status_msgs.append(f"[{ts}] {msg}")
            self.status_msgs = self.status_msgs[-12:]
        logger.info(msg)

    def reset(self, total_steps: int, opts: dict[str, Any]) -> None:
        with self._lock:
            self.running = True
            self.percent = 0
            self.step = 0
            self.total_steps = total_steps
            self.status_msgs = []
            self.result_image = None
            self.error = None
            self.generation_time = None
            self.vram_peak_mb = None
            self.gen_opts = dict(opts)
        self.push_status(f"Queued generation: {total_steps} steps.")

    def is_running(self) -> bool:
        with self._lock:
            return self.running

    def compact_status(self) -> str:
        with self._lock:
            msgs = list(self.status_msgs[-10:])
        parts = [m.split("] ", 1)[1] if "] " in m else m for m in msgs]
        return " ⦿ ".join(parts)

    def get_percent(self) -> int:
        with self._lock:
            return self.percent

    def get_step_info(self) -> tuple[int, int, int]:
        with self._lock:
            return self.step, self.total_steps, self.percent

    def update_step(self, step: int) -> None:
        with self._lock:
            total = self.total_steps or 1
            self.step = step
            self.percent = int((step / total) * 100)

    def finish(self, image: Any, generation_time: float, vram_peak_mb: float) -> None:
        with self._lock:
            self.result_image = image
            self.generation_time = generation_time
            self.vram_peak_mb = vram_peak_mb
            self.percent = 100
            self.running = False

    def fail(self, error: str) -> None:
        with self._lock:
            self.error = error
            self.running = False

    def get_result(self) -> tuple[Optional[Any], Optional[str], Optional[float], Optional[float]]:
        with self._lock:
            return self.result_image, self.error, self.generation_time, self.vram_peak_mb


_state = GenerationState()


def _callback_diffusers_on_step_end(
    pipe: Any, step: int, timestep: int, callback_kwargs: dict
) -> dict:
    try:
        _state.update_step(step)
        _step, total, pct = _state.get_step_info()
        if step % max(1, total // 10) == 0 or pct in (0, 50, 75):
            _state.push_status(f"Progress: {pct}% (step {step}/{total})")
    except Exception as e:
        logger.debug("Progress callback error: %s", e)
    return callback_kwargs


def _run_pipeline_in_thread(prompt: str, opts: dict[str, Any], model_choice: str) -> None:
    try:
        _state.push_status("Starting generation thread...")

        current_model = MODEL.optimizations.get("model_loaded", "")
        need_init = MODEL.backend is None or (model_choice == "sdxl") != (current_model == "sdxl")
        if need_init:
            try:
                MODEL.init(model_choice="sdxl" if model_choice == "sdxl" else None)
            except Exception as e:
                _state.push_status(f"Model reload warning: {e}")

        start_time = time.time()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

        if MODEL.backend == "local" and getattr(MODEL, "pipe", None) is not None:
            pipe = MODEL.pipe
            out = pipe(
                opts["prompt"],
                negative_prompt=opts.get("negative_prompt") or None,
                num_inference_steps=int(opts["steps"]),
                guidance_scale=float(opts["guidance"]),
                width=int(opts["width"]),
                height=int(opts["height"]),
                generator=opts.get("generator"),
                callback_on_step_end=_callback_diffusers_on_step_end,
            )
            images = getattr(out, "images", None)
            if images:
                result_img = images[0]
            elif isinstance(out, (list, tuple)) and out:
                result_img = out[0]
            else:
                raise RuntimeError("Unexpected pipeline output structure.")

        elif MODEL.backend == "inference" and getattr(MODEL, "inference_client", None) is not None:
            _state.push_status("Using Hugging Face Inference API (simulated progress).")
            total = int(opts.get("steps", cfg.default_steps))
            for s in range(1, total + 1):
                time.sleep(0.03)
                _state.update_step(s)
            result_img = MODEL.inference_client.text_to_image(opts["prompt"], model=MODEL.model_id)

        else:
            raise RuntimeError("No model backend available.")

        generation_time = time.time() - start_time
        try:
            import torch
            vram_peak_mb = (
                torch.cuda.max_memory_allocated() / 1024**2
                if torch.cuda.is_available()
                else 0.0
            )
        except Exception:
            vram_peak_mb = 0.0

        _state.finish(result_img, generation_time, vram_peak_mb)
        _state.push_status("Generation complete.")

    except Exception as e:
        logger.exception("Generation error")
        _state.fail(str(e))
        _state.push_status(f"ERROR: {e}")


def _start_generation(
    prompt: str,
    negative_prompt: str,
    model_choice: str,
    steps: int,
    guidance: float,
    width: int,
    height: int,
    seed: Optional[int],
) -> None:
    """Validate, build opts, and launch the worker thread."""
    if _state.is_running():
        raise RuntimeError("Another generation is already running. Please wait.")

    actual_seed = int(seed) if seed is not None else random.randint(0, 2**32 - 1)
    generator = None
    try:
        import torch
        device = "cuda" if (MODEL.backend == "local" and torch.cuda.is_available()) else "cpu"
        generator = torch.Generator(device=device).manual_seed(actual_seed)
    except Exception:
        pass

    opts: dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "guidance": guidance,
        "width": width,
        "height": height,
        "seed": actual_seed,
        "generator": generator,
    }
    _state.reset(total_steps=steps, opts=opts)
    threading.Thread(
        target=_run_pipeline_in_thread, args=(prompt, opts, model_choice), daemon=True
    ).start()


def make_placeholder_image(
    w: int = 512, h: int = 512, text: str = "Your generated image will appear here"
) -> Image.Image:
    """Create a decorative grey placeholder with centred text."""
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


def generate_stream(
    prompt: str,
    negative_prompt: str,
    model_choice: str,
    steps: int,
    guidance: float,
    width: int,
    height: int,
    seed: Optional[int],
) -> Generator[tuple[Optional[Image.Image], str], None, None]:
    """Stream generation progress, save the result with metadata, and yield the final image."""
    try:
        if not prompt or not prompt.strip():
            yield None, "Please enter a prompt."
            return
        if int(width) * int(height) > 1024 * 1024 * 3:
            yield None, "Requested image size too large. Choose a smaller size."
            return

        try:
            _start_generation(
                prompt, negative_prompt, model_choice, steps, guidance, width, height, seed
            )
        except RuntimeError as e:
            yield None, str(e)
            return

        while _state.is_running():
            pct = _state.get_percent()
            compact = _state.compact_status()
            status_line = f"**Progress:** {pct}% — {compact}" if compact else f"**Progress:** {pct}%"
            yield None, status_line
            time.sleep(0.5)

        img, error, generation_time, vram_peak_mb = _state.get_result()
        if error:
            yield None, f"**Error:** {error}"
            return
        if img is None:
            yield None, "**Error:** generation did not return an image."
            return

        actual_seed = _state.gen_opts.get("seed", 0)
        scheduler_name = "unknown"
        if MODEL.backend == "local" and getattr(MODEL, "pipe", None) is not None:
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
            "lora_hash": "",
            "git_commit": meta.get_git_commit(),
            "generation_time_seconds": round(generation_time or 0.0, 2),
            "vram_peak_mb": round(vram_peak_mb or 0.0, 1),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        out_dir = Path("outputs")
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{actual_seed}.png"
        meta.save_image_with_metadata(img, out_path, md)

        compact = _state.compact_status()
        suffix = f"\n{compact}" if compact else ""
        yield img, f"**Done — 100%** — Saved: `{out_path}`{suffix}"

    except Exception as e:
        logger.exception("generate_stream failed")
        yield None, f"Unexpected error: {e}"


def reload_model(choice: str) -> str:
    """Load the selected model pipeline."""
    try:
        MODEL.init(model_choice="sdxl" if choice == "sdxl" else None)
        return f"{'SDXL' if choice == 'sdxl' else 'SD 2.1'} loaded successfully."
    except Exception as e:
        logger.exception("reload_model failed")
        return f"Failed to load {choice}: {e}"


def load_from_png(file_path: str | None) -> tuple:
    """Read PNG metadata and return values to pre-fill the generation form."""
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


_PLACEHOLDER = make_placeholder_image(512, 512, "Your generated image will appear here")

with gr.Blocks() as demo:
    gr.Markdown("# AetherArt — Stable Diffusion (2.1 default, SDXL optional)")
    gr.Markdown(
        "Runs SD2.1 locally on GPU (recommended). Switch to SDXL for higher quality (requires more VRAM)."
    )

    with gr.Row():
        with gr.Column(scale=3):
            model_choice = gr.Dropdown(choices=["sd-2.1", "sdxl"], value="sd-2.1", label="Model")
            prompt = gr.Textbox(
                placeholder="A modern studio portrait of an astronaut", label="Prompt", lines=2
            )
            negative_prompt = gr.Textbox(
                placeholder="blurry, low quality", label="Negative Prompt", lines=1
            )
            steps = gr.Slider(10, 60, value=cfg.default_steps, label="Steps")
            guidance = gr.Slider(1.0, 15.0, value=cfg.default_guidance, step=0.5, label="Guidance")
            width = gr.Radio([512, 768, 1024], value=cfg.default_width, label="Width")
            height = gr.Radio([512, 768, 1024], value=cfg.default_height, label="Height")
            seed = gr.Number(value=None, label="Seed (blank = random)", precision=0)
            gen_btn = gr.Button("Generate", variant="primary")
            reload_btn = gr.Button("Reload Model")
            status_md = gr.Markdown("Ready.")

        with gr.Column(scale=4):
            out_img = gr.Image(label="Generated Image", value=_PLACEHOLDER, interactive=False)

    with gr.Accordion("Recreate from PNG", open=False):
        gr.Markdown("Upload a previously generated PNG to restore its prompt, seed, and settings.")
        png_upload = gr.File(label="Upload PNG", file_types=[".png"], type="filepath")
        load_btn = gr.Button("Load Settings from PNG")

    gen_btn.click(
        fn=generate_stream,
        inputs=[prompt, negative_prompt, model_choice, steps, guidance, width, height, seed],
        outputs=[out_img, status_md],
    )
    reload_btn.click(fn=reload_model, inputs=[model_choice], outputs=[status_md])
    load_btn.click(
        fn=load_from_png,
        inputs=[png_upload],
        outputs=[prompt, negative_prompt, seed, steps, guidance, width, height],
    )


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    share = os.getenv("SHARE", "false").lower() == "true"
    logger.info("Starting Gradio app on http://%s:%d (share=%s)", host, port, share)
    demo.queue()
    demo.launch(server_name=host, server_port=port, share=share)


if __name__ == "__main__":
    main()
