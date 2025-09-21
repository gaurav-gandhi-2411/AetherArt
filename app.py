import os
import time
import threading
from typing import Dict, Any, Optional

import gradio as gr
from PIL import Image, ImageDraw, ImageFont

from aetherart.model import AetherModel
from aetherart.logger import get_logger
from aetherart.config import cfg

logger = get_logger(__name__)

# Global model instance (single shared)
MODEL = AetherModel()
try:
    MODEL.init(model_choice=None)
except Exception as e:
    logger.error("Initial model init failed: %s", e)

# Shared state objects used for progress streaming
_progress_state: Dict[str, Any] = {
    "running": False,
    "percent": 0,
    "step": 0,
    "total_steps": cfg.default_steps,
    "status_msgs": [],
    "result_image": None,
    "error": None,
}


def _push_status(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    entry = f"[{ts}] {msg}"
    _progress_state["status_msgs"].append(entry)
    _progress_state["status_msgs"] = _progress_state["status_msgs"][-12:]
    logger.info(msg)


def _reset_progress(total_steps: int) -> None:
    _progress_state.update({
        "running": True,
        "percent": 0,
        "step": 0,
        "total_steps": total_steps,
        "result_image": None,
        "error": None,
    })
    _push_status(f"Queued generation: {total_steps} steps.")


# callback for diffusers >= 0.27
def _callback_diffusers_on_step_end(pipe, step: int, timestep: int, callback_kwargs: dict):
    try:
        total = _progress_state.get("total_steps", 1) or 1
        percent = int((step / total) * 100)
        _progress_state["step"] = step
        _progress_state["percent"] = percent
        if step % max(1, int(total / 10)) == 0 or percent in (0, 50, 75):
            _push_status(f"Progress: {percent}% (step {step}/{total})")
    except Exception as e:
        logger.debug("Progress callback error: %s", e)
    return callback_kwargs


def _run_pipeline_in_thread(prompt: str, opts: Dict[str, Any], model_choice: str) -> None:
    try:
        _push_status("Starting generation thread...")
        try:
            if model_choice == "sdxl":
                MODEL.init(model_choice="sdxl")
            else:
                MODEL.init(model_choice=None)
        except Exception as e:
            _push_status(f"Model reload warning: {e}")

        if MODEL.backend == "local" and getattr(MODEL, "pipe", None) is not None:
            pipe = MODEL.pipe
            _progress_state["total_steps"] = int(opts.get("steps", cfg.default_steps))
            out = pipe(
                opts["prompt"],
                num_inference_steps=int(opts["steps"]),
                guidance_scale=float(opts["guidance"]),
                width=int(opts["width"]),
                height=int(opts["height"]),
                generator=opts.get("generator", None),
                callback_on_step_end=_callback_diffusers_on_step_end,  # ✅ new API
            )
            img = getattr(out, "images", None)
            if img:
                _progress_state["result_image"] = img[0]
            else:
                if isinstance(out, (list, tuple)) and out:
                    _progress_state["result_image"] = out[0]
                else:
                    raise RuntimeError("Unexpected pipeline output structure.")
            _progress_state["percent"] = 100
            _push_status("Generation complete.")
        elif MODEL.backend == "inference" and getattr(MODEL, "inference_client", None) is not None:
            _push_status("Using Hugging Face Inference API (simulated progress).")
            total = int(opts.get("steps", cfg.default_steps))
            for s in range(1, total + 1):
                time.sleep(0.03)
                _progress_state["step"] = s
                _progress_state["percent"] = int((s / total) * 100)
            img = MODEL.inference_client.text_to_image(opts["prompt"], model=MODEL.model_id)
            _progress_state["result_image"] = img
            _progress_state["percent"] = 100
            _push_status("Inference API generation complete.")
        else:
            raise RuntimeError("No model backend available.")
    except Exception as e:
        logger.exception("Generation error")
        _progress_state["error"] = str(e)
        _push_status(f"ERROR: {e}")
    finally:
        _progress_state["running"] = False


def _start_generation_thread(prompt: str, model_choice: str, steps: int, guidance: float, width: int, height: int, seed: Optional[int]) -> None:
    if _progress_state["running"]:
        raise RuntimeError("Another generation is already running. Please wait.")
    generator = None
    if seed is not None and seed != "":
        try:
            device = "cuda" if (MODEL.backend == "local" and __import__("torch").cuda.is_available()) else "cpu"
            generator = __import__("torch").Generator(device=device).manual_seed(int(seed))
        except Exception:
            generator = None

    opts = {
        "prompt": prompt,
        "steps": steps,
        "guidance": guidance,
        "width": width,
        "height": height,
        "generator": generator,
    }
    _reset_progress(total_steps=steps)
    t = threading.Thread(target=_run_pipeline_in_thread, args=(prompt, opts, model_choice), daemon=True)
    t.start()


def _compact_thread_line() -> str:
    msgs = _progress_state.get("status_msgs", [])[-10:]
    compact_parts = []
    for m in msgs:
        if "] " in m:
            compact_parts.append(m.split("] ", 1)[1])
        else:
            compact_parts.append(m)
    return " ⦿ ".join(compact_parts)


def make_placeholder_image(w=512, h=512, text="Your generated image will appear here"):
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
    total_h = 0
    line_heights = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        h_line = bbox[3] - bbox[1]
        line_heights.append(h_line)
        total_h += h_line

    y = (h - total_h) // 2
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        w_line = bbox[2] - bbox[0]
        x = (w - w_line) // 2
        draw.text((x, y), line, fill=(60, 60, 60), font=font)
        y += line_heights[i]

    return img


def generate_stream(prompt: str, model_choice: str, steps: int, guidance: float, width: int, height: int, seed: Optional[int]):
    try:
        if not prompt or prompt.strip() == "":
            yield None, "Please enter a prompt."
            return

        if int(width) * int(height) > 1024 * 1024 * 3:
            yield None, "Requested image size too large for this environment. Choose a smaller size."
            return

        try:
            _start_generation_thread(prompt, model_choice, steps, guidance, width, height, seed)
        except RuntimeError as e:
            yield None, str(e)
            return

        while _progress_state["running"]:
            percent = _progress_state.get("percent", 0)
            compact = _compact_thread_line()
            status_line = f"**Progress:** {percent}% — {compact}" if compact else f"**Progress:** {percent}%"
            yield None, status_line
            time.sleep(0.5)

        if _progress_state.get("error"):
            yield None, f"**Error:** {_progress_state['error']}"
            return

        img = _progress_state.get("result_image")
        if img is None:
            yield None, "**Error:** generation did not return an image."
            return

        final_compact = _compact_thread_line()
        final_md = f"**Done — 100%** — {final_compact}" if final_compact else "**Done — 100%**"
        yield img, final_md
        return

    except Exception as e:
        logger.exception("generate_stream failed")
        yield None, f"Unexpected error: {e}"


def reload_model(choice: str) -> str:
    try:
        if choice == "sdxl":
            MODEL.init(model_choice="sdxl")
            return "SDXL selected and loaded (may take longer)."
        else:
            MODEL.init(model_choice=None)
            return "SD 2.1 selected and loaded."
    except Exception as e:
        logger.exception("reload_model failed")
        return f"Failed to load {choice}: {e}"


title = "AetherArt — Stable Diffusion (2.1 default, SDXL optional)"
desc = "Runs SD2.1 locally on GPU (recommended). Switch to SDXL for higher quality (requires more VRAM)."
PLACEHOLDER = make_placeholder_image(512, 512, "Your generated image will appear here")

with gr.Blocks() as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(desc)

    with gr.Row():
        with gr.Column(scale=3):
            model_choice = gr.Dropdown(choices=["sd-2.1", "sdxl"], value="sd-2.1", label="Model")
            prompt = gr.Textbox(placeholder="A modern studio portrait of an astronaut", label="Prompt", lines=2)
            steps = gr.Slider(10, 60, value=cfg.default_steps, label="Steps")
            guidance = gr.Slider(1.0, 15.0, value=cfg.default_guidance, step=0.5, label="Guidance")
            width = gr.Radio([512, 768, 1024], value=cfg.default_width, label="Width")
            height = gr.Radio([512, 768, 1024], value=cfg.default_height, label="Height")
            seed = gr.Number(value=None, label="Seed (optional)", precision=0)
            gen_btn = gr.Button("Generate")
            reload_btn = gr.Button("Reload Model")
            status_md = gr.Markdown("Model status: see logs")
        with gr.Column(scale=4):
            out_img = gr.Image(label="Generated Image", value=PLACEHOLDER, interactive=False)

    gen_btn.click(fn=generate_stream,
                  inputs=[prompt, model_choice, steps, guidance, width, height, seed],
                  outputs=[out_img, status_md])
    reload_btn.click(fn=reload_model, inputs=[model_choice], outputs=[status_md])

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    share = os.getenv("SHARE", "false").lower() == "true"

    logger.info(f"Starting Gradio app on http://{host}:{port} (share={share})")
    demo.queue()
    demo.launch(server_name=host, server_port=port, share=share)
