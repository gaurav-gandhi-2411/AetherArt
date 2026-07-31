#!/usr/bin/env python
"""
PartiPrompts × scheduler × step-count evaluation harness.

Usage:
    # Full benchmark (360 runs, ~30-45 min on RTX 3070)
    python scripts/eval.py

    # Smoke test
    python scripts/eval.py --smoke

    # Custom subset
    python scripts/eval.py --schedulers DDIM,DPM --steps 20,30 --prompts-subset pp_002,pp_005

    # Resume after crash
    python scripts/eval.py --resume

    # Run with specific scorers only
    python scripts/eval.py --scorers clip,hps

    # Load a LoRA adapter
    python scripts/eval.py --lora /path/to/adapter.safetensors
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

# Ensure parent package is importable when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    LMSDiscreteScheduler,
)

from aetherart.clip_scorer import score as clip_score
from aetherart.logger import get_logger
from aetherart.metadata import get_git_commit, save_image_with_metadata
from aetherart.model import AetherModel

logger = get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

SCHEDULER_MAP: dict[str, type] = {
    "DDIM": DDIMScheduler,
    "DPM": DPMSolverMultistepScheduler,
    "EulerA": EulerAncestralDiscreteScheduler,
    "LMS": LMSDiscreteScheduler,
}
DEFAULT_SCHEDULERS = ["DDIM", "DPM", "EulerA", "LMS"]
DEFAULT_STEPS = [20, 30, 50]
DEFAULT_SEED = 42
DEFAULT_GUIDANCE = 7.5
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
DEFAULT_NEGATIVE_PROMPT = ""
EVAL_WIDTH = DEFAULT_WIDTH
EVAL_HEIGHT = DEFAULT_HEIGHT
EVAL_NEGATIVE_PROMPT = DEFAULT_NEGATIVE_PROMPT

DEFAULT_SCORERS = ["clip", "hps", "imagereward", "lpips"]
VALID_SCORERS = frozenset(DEFAULT_SCORERS)

PROMPTS_YAML = Path(__file__).parent / "eval_prompts.yaml"
REPORTS_DIR = Path(__file__).parent.parent / "reports"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs" / "eval"
PARTIAL_FILE = REPORTS_DIR / "eval_partial_latest.json"

# ── Prompt loading ─────────────────────────────────────────────────────────


def load_prompts(subset: str | None = None) -> list[dict]:
    with PROMPTS_YAML.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    prompts = data["prompts"]
    if subset:
        ids = {s.strip() for s in subset.split(",")}
        prompts = [p for p in prompts if p["id"] in ids]
        if not prompts:
            raise ValueError(f"No prompts matched subset filter: {subset!r}")
    return prompts


# ── Partial-results I/O ────────────────────────────────────────────────────


def load_partial(run_id: str | None) -> tuple[list[dict], set[str]]:
    """Load existing partial results. Returns (results_list, completed_keys)."""
    if not PARTIAL_FILE.exists():
        return [], set()
    try:
        with PARTIAL_FILE.open(encoding="utf-8") as f:
            state = json.load(f)
        results = state.get("results", [])
        # If caller supplied a run_id but file belongs to a different run, start fresh
        if run_id and state.get("run_id") != run_id:
            logger.info("Partial file is from a different run — starting fresh.")
            return [], set()
        completed = {_combo_key(r) for r in results if not r.get("error")}
        logger.info("Resuming: %d results already completed.", len(completed))
        return results, completed
    except Exception as e:
        logger.warning("Could not load partial results: %s", e)
        return [], set()


def save_partial(run_id: str, run_date: str, config: dict, results: list[dict]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    state = {"run_id": run_id, "run_date": run_date, "config": config, "results": results}
    # Atomic write via temp file
    tmp = PARTIAL_FILE.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)
    tmp.replace(PARTIAL_FILE)


def _combo_key(r: dict) -> str:
    return f"{r['prompt_id']}_{r['scheduler']}_{r['steps']}"


# ── Scheduler swapping ─────────────────────────────────────────────────────


def swap_scheduler(pipe: Any, name: str) -> None:
    cls = SCHEDULER_MAP[name]
    try:
        pipe.scheduler = cls.from_config(pipe.scheduler.config)
    except Exception as e:
        logger.warning("from_config failed for %s (%s); retrying from_pretrained", name, e)
        pipe.scheduler = cls.from_pretrained(
            pipe.scheduler.config.get("_name_or_path", ""), subfolder="scheduler"
        )


# ── Single generation ──────────────────────────────────────────────────────


def run_single(
    model: AetherModel,
    prompt_entry: dict,
    scheduler_name: str,
    n_steps: int,
    seed: int,
) -> dict:
    result_base = {
        "prompt_id": prompt_entry["id"],
        "prompt": prompt_entry["prompt"],
        "category": prompt_entry["category"],
        "expected_difficulty": prompt_entry["expected_difficulty"],
        "scheduler": scheduler_name,
        "steps": n_steps,
        "seed": seed,
        "clip_score": None,
        "hps_score": None,
        "image_reward_score": None,
        "lpips_score": None,
        "latency_s": None,
        "vram_peak_gb": None,
        "image_path": None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "error": None,
    }
    try:
        swap_scheduler(model.pipe, scheduler_name)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        generator = torch.Generator(device=device).manual_seed(seed)
        t0 = time.time()
        pipe_kwargs: dict[str, Any] = {
            "num_inference_steps": n_steps,
            "guidance_scale": DEFAULT_GUIDANCE,
            "width": EVAL_WIDTH,
            "height": EVAL_HEIGHT,
            "generator": generator,
        }
        if EVAL_NEGATIVE_PROMPT:
            pipe_kwargs["negative_prompt"] = EVAL_NEGATIVE_PROMPT
        out = model.pipe(prompt_entry["prompt"], **pipe_kwargs)
        latency = time.time() - t0
        vram_peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

        img = out.images[0]
        cs = clip_score(img, prompt_entry["prompt"])

        out_dir = OUTPUTS_DIR / prompt_entry["id"] / scheduler_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{n_steps}steps.png"

        save_image_with_metadata(
            img,
            out_path,
            {
                "prompt": prompt_entry["prompt"],
                "negative_prompt": "",
                "seed": seed,
                "scheduler": scheduler_name,
                "steps": n_steps,
                "guidance": DEFAULT_GUIDANCE,
                "width": EVAL_WIDTH,
                "height": EVAL_HEIGHT,
                "model_id": model.model_id,
                "lora_hash": "",
                "git_commit": get_git_commit(),
                "generation_time_seconds": round(latency, 2),
                "vram_peak_gb": round(vram_peak_gb, 3),
                "clip_score": round(cs, 4),
                "timestamp": result_base["timestamp"],
            },
        )

        result_base.update(
            {
                "clip_score": round(cs, 4),
                "latency_s": round(latency, 2),
                "vram_peak_gb": round(vram_peak_gb, 3),
                "image_path": str(out_path),
            }
        )

    except torch.cuda.OutOfMemoryError as e:
        logger.error("OOM for %s/%s/%d: %s", prompt_entry["id"], scheduler_name, n_steps, e)
        result_base["error"] = f"OOM: {e}"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        logger.exception("Error for %s/%s/%d", prompt_entry["id"], scheduler_name, n_steps)
        result_base["error"] = str(e)

    return result_base


# ── Chart generation ───────────────────────────────────────────────────────


def generate_charts(results: list[dict], run_date: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    charts_dir = REPORTS_DIR / "eval_charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    ok = [r for r in results if not r.get("error")]
    schedulers = sorted(set(r["scheduler"] for r in ok))
    steps_list = sorted(set(r["steps"] for r in ok))
    colors = plt.cm.tab10.colors
    sched_color = {s: colors[i % len(colors)] for i, s in enumerate(schedulers)}
    chart_paths = []

    def _save(fig: Any, name: str) -> str:
        p = str(charts_dir / f"{name}_{run_date}.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        chart_paths.append(p)
        return p

    # 1. Avg latency by scheduler × steps
    fig, ax = plt.subplots(figsize=(9, 5))
    for sched in schedulers:
        ys = [
            np.mean(
                [r["latency_s"] for r in ok if r["scheduler"] == sched and r["steps"] == s] or [0]
            )
            for s in steps_list
        ]
        ax.plot(steps_list, ys, marker="o", label=sched, color=sched_color[sched])
    ax.set_xlabel("Steps")
    ax.set_ylabel("Avg Latency (s)")
    ax.set_title("Avg Generation Latency by Scheduler × Steps")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "latency")

    # 2. Avg CLIP score by scheduler × steps
    fig, ax = plt.subplots(figsize=(9, 5))
    for sched in schedulers:
        ys = [
            np.mean(
                [r["clip_score"] for r in ok if r["scheduler"] == sched and r["steps"] == s] or [0]
            )
            for s in steps_list
        ]
        ax.plot(steps_list, ys, marker="o", label=sched, color=sched_color[sched])
    ax.set_xlabel("Steps")
    ax.set_ylabel("Avg CLIP Score")
    ax.set_title("Avg CLIP Score by Scheduler × Steps\n(higher = better prompt alignment)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "clip_score")

    # 3. Latency vs CLIP score scatter (Pareto)
    fig, ax = plt.subplots(figsize=(9, 6))
    for sched in schedulers:
        sub = [r for r in ok if r["scheduler"] == sched]
        ax.scatter(
            [r["latency_s"] for r in sub],
            [r["clip_score"] for r in sub],
            label=sched,
            alpha=0.55,
            color=sched_color[sched],
            s=40,
        )
        if sub:
            mx = np.mean([r["latency_s"] for r in sub])
            my = np.mean([r["clip_score"] for r in sub])
            ax.scatter(
                [mx],
                [my],
                marker="*",
                s=220,
                color=sched_color[sched],
                edgecolors="black",
                linewidths=0.6,
                zorder=6,
            )
            ax.annotate(sched, (mx, my), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Latency (s)")
    ax.set_ylabel("CLIP Score")
    ax.set_title("Latency vs CLIP Score — Pareto Scatter  (★ = scheduler mean)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "pareto")

    # 4. Peak VRAM by scheduler (bar)
    fig, ax = plt.subplots(figsize=(7, 4))
    avg_vram = [
        np.mean([r["vram_peak_gb"] for r in ok if r["scheduler"] == sched] or [0])
        for sched in schedulers
    ]
    bars = ax.bar(schedulers, avg_vram, color=[sched_color[s] for s in schedulers])
    for bar, val in zip(bars, avg_vram, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.2f} GB",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("Avg Peak VRAM (GB)")
    ax.set_title("Peak VRAM Usage by Scheduler")
    ax.set_ylim(0, max(avg_vram) * 1.2 + 0.1 if avg_vram else 1)
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, "vram")

    return chart_paths


# ── Markdown report ────────────────────────────────────────────────────────


def generate_report(
    results: list[dict], run_id: str, run_date: str, config: dict, chart_paths: list[str]
) -> str:
    ok = [r for r in results if not r.get("error")]
    errors = [r for r in results if r.get("error")]
    schedulers = sorted(set(r["scheduler"] for r in ok))
    steps_list = sorted(set(r["steps"] for r in ok))

    def avg(lst: list) -> str:
        return f"{np.mean(lst):.4f}" if lst else "—"

    lines = [
        "# AetherArt Evaluation Report",
        f"**Run ID:** `{run_id}`  **Date:** {run_date}",
        f"**Total runs:** {len(results)}  **Errors:** {len(errors)}  **OK:** {len(ok)}",
        "",
        "## Avg CLIP Score by Scheduler × Steps",
        "",
        "| Scheduler | " + " | ".join(f"{s} steps" for s in steps_list) + " | Overall |",
        "|" + "---|" * (len(steps_list) + 2),
    ]
    for sched in schedulers:
        row = [sched]
        for s in steps_list:
            vals = [r["clip_score"] for r in ok if r["scheduler"] == sched and r["steps"] == s]
            row.append(avg(vals))
        row.append(avg([r["clip_score"] for r in ok if r["scheduler"] == sched]))
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## Avg Latency (s) by Scheduler × Steps",
        "",
        "| Scheduler | " + " | ".join(f"{s} steps" for s in steps_list) + " | Overall |",
        "|" + "---|" * (len(steps_list) + 2),
    ]
    for sched in schedulers:
        row = [sched]
        for s in steps_list:
            vals = [r["latency_s"] for r in ok if r["scheduler"] == sched and r["steps"] == s]
            row.append(avg(vals))
        row.append(avg([r["latency_s"] for r in ok if r["scheduler"] == sched]))
        lines.append("| " + " | ".join(row) + " |")

    # Top-3 / Bottom-3 CLIP per scheduler
    lines += ["", "## Top-3 and Bottom-3 CLIP Scores per Scheduler", ""]
    for sched in schedulers:
        sub = sorted([r for r in ok if r["scheduler"] == sched], key=lambda r: r["clip_score"])
        lines.append(f"### {sched}")
        lines.append("")
        lines.append("**Bottom 3:**")
        for r in sub[:3]:
            lines.append(f"- `{r['prompt_id']}` {r['clip_score']:.4f} — *{r['prompt'][:60]}*")
        lines.append("")
        lines.append("**Top 3:**")
        for r in sub[-3:][::-1]:
            lines.append(f"- `{r['prompt_id']}` {r['clip_score']:.4f} — *{r['prompt'][:60]}*")
        lines.append("")

    # VRAM summary
    lines += ["## Peak VRAM Summary", ""]
    for sched in schedulers:
        vals = [r["vram_peak_gb"] for r in ok if r["scheduler"] == sched]
        if vals:
            lines.append(f"- **{sched}**: avg {np.mean(vals):.2f} GB, max {max(vals):.2f} GB")

    # Errors
    if errors:
        lines += ["", "## Errors", ""]
        for r in errors:
            lines.append(
                f"- `{r['prompt_id']}` / {r['scheduler']} / {r['steps']} steps: {r['error']}"
            )

    # Charts
    lines += ["", "## Charts", ""]
    for p in chart_paths:
        lines.append(f"- `{p}`")

    # Config
    lines += ["", "## Config", "", f"```json\n{json.dumps(config, indent=2)}\n```"]

    md = "\n".join(lines) + "\n"
    out_path = REPORTS_DIR / f"eval_results_{run_id}.md"
    out_path.write_text(md, encoding="utf-8")
    return str(out_path)


# ── Main ───────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PartiPrompts × scheduler × steps benchmark")
    p.add_argument(
        "--schedulers",
        default=None,
        help=f"Comma-separated schedulers (default: {','.join(DEFAULT_SCHEDULERS)})",
    )
    p.add_argument(
        "--steps",
        default=None,
        help=f"Comma-separated step counts (default: {','.join(str(s) for s in DEFAULT_STEPS)})",
    )
    p.add_argument(
        "--prompts-subset",
        default=None,
        help="Comma-separated prompt IDs to run (default: all prompts in eval_prompts.yaml)",
    )
    p.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help=f"Global RNG seed (default: {DEFAULT_SEED})"
    )
    p.add_argument(
        "--resume", action="store_true", help="Resume from the latest partial results file"
    )
    p.add_argument("--model", default=None, help="Model choice: sd-2.1 (default) or sdxl")
    p.add_argument(
        "--base",
        default=None,
        dest="base",
        help="Alias for --model: sdxl or sd21 (default: uses --model logic)",
    )
    p.add_argument(
        "--scorers",
        default=",".join(DEFAULT_SCORERS),
        help=(
            f"Comma-separated scorers to run (default: {','.join(DEFAULT_SCORERS)}). "
            f"Valid: {', '.join(sorted(VALID_SCORERS))}"
        ),
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Quick single-run mode: 3 prompts, DPM scheduler, 20 steps",
    )
    p.add_argument(
        "--lora",
        default=None,
        help="Optional path to a .safetensors LoRA adapter to load onto the pipeline",
    )
    p.add_argument(
        "--num-images",
        type=int,
        default=None,
        dest="num_images",
        help="Override prompt count: take only the first N prompts from the loaded list",
    )
    p.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help=f"Image width in pixels (default: {DEFAULT_WIDTH}). Use 1024 for SDXL.",
    )
    p.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help=f"Image height in pixels (default: {DEFAULT_HEIGHT}). Use 1024 for SDXL.",
    )
    p.add_argument(
        "--negative-prompt",
        default=DEFAULT_NEGATIVE_PROMPT,
        dest="negative_prompt",
        help="Negative prompt applied to all generations (default: empty).",
    )
    p.add_argument(
        "--run-id",
        default=None,
        help="Override the auto-generated timestamp run ID (default: current timestamp). "
        "Lets callers (e.g. CI) predict the output report filename deterministically.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    global EVAL_WIDTH, EVAL_HEIGHT, EVAL_NEGATIVE_PROMPT
    EVAL_WIDTH = args.width
    EVAL_HEIGHT = args.height
    EVAL_NEGATIVE_PROMPT = args.negative_prompt

    # Validate and parse scorers
    scorers = {s.strip() for s in args.scorers.split(",")}
    unknown = scorers - VALID_SCORERS
    if unknown:
        sys.exit(f"Unknown scorers: {unknown}")

    schedulers = (
        [s.strip() for s in args.schedulers.split(",")] if args.schedulers else DEFAULT_SCHEDULERS
    )
    steps_list = [int(s.strip()) for s in args.steps.split(",")] if args.steps else DEFAULT_STEPS

    for sched in schedulers:
        if sched not in SCHEDULER_MAP:
            sys.exit(f"Unknown scheduler: {sched!r}. Valid: {list(SCHEDULER_MAP)}")

    prompts = load_prompts(args.prompts_subset)

    # --smoke: override to 3 prompts, DPM, 20 steps
    if args.smoke:
        prompts = prompts[:3]
        schedulers = ["DPM"]
        steps_list = [20]
        logger.info("--smoke: overriding to 3 prompts, DPM scheduler, 20 steps")

    # --num-images: trim prompt list to first N
    if args.num_images is not None:
        prompts = prompts[: args.num_images]
        logger.info("--num-images %d: using first %d prompts", args.num_images, len(prompts))

    total_combos = len(prompts) * len(schedulers) * len(steps_list)

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_date = datetime.now().isoformat(timespec="seconds")

    # Resolve model choice: --base takes precedence over --model when set
    model_choice_raw = args.base or args.model
    model_choice = "sdxl" if model_choice_raw in ("sdxl", "sdxl-base") else None

    config = {
        "schedulers": schedulers,
        "steps": steps_list,
        "prompts_count": len(prompts),
        "seed": args.seed,
        "model": model_choice_raw or "sd-2.1",
        "total_combos": total_combos,
        "scorers": list(scorers),
        "width": EVAL_WIDTH,
        "height": EVAL_HEIGHT,
        "negative_prompt": EVAL_NEGATIVE_PROMPT,
    }

    logger.info("=== AetherArt Eval — run %s ===", run_id)
    logger.info(
        "Schedulers: %s | Steps: %s | Prompts: %d | Total: %d",
        schedulers,
        steps_list,
        len(prompts),
        total_combos,
    )

    # Resume
    if args.resume:
        results, completed = load_partial(None)
    else:
        results, completed = [], set()

    # Build ordered list of combinations
    combos = [(p, sched, steps) for p in prompts for sched in schedulers for steps in steps_list]

    # Init model once
    logger.info("Loading model...")
    model = AetherModel()
    model.init(model_choice="sdxl" if model_choice == "sdxl" else None)
    logger.info("Model loaded: backend=%s", model.backend)

    if model.backend != "local" or model.pipe is None:
        sys.exit(
            "Eval requires a local pipeline — set USE_HF_INFERENCE=0 "
            "and ensure diffusers is installed."
        )

    # Load LoRA adapter if specified.
    # Fuse immediately after loading: AetherModel.init() already called
    # enable_model_cpu_offload(), and unfused LoRA hooks + offload hooks
    # interact badly — inference drops to ~90 s/step instead of ~3 s/step.
    # Fusing bakes LoRA weights into base weights so no hook overhead remains.
    if args.lora:
        logger.info("Loading LoRA from %s", args.lora)
        model.pipe.load_lora_weights(args.lora)
        model.pipe.fuse_lora()
        logger.info("LoRA loaded and fused.")

    # Prime CLIP scorer (downloads weights once)
    logger.info("Loading CLIP scorer...")
    from aetherart.clip_scorer import _load as clip_load

    clip_load()
    logger.info("CLIP scorer ready.")

    t_start = time.time()
    done = 0
    skipped = 0

    # Generate-then-score ordering: CLIP is scored inline (small model, no VRAM risk).
    # HPS (~2 GB) and ImageReward (~1.6 GB) are loaded after the pipeline is released
    # to stay within the 8 GB VRAM budget on RTX 3070.
    for prompt_entry, scheduler_name, n_steps in combos:
        key = f"{prompt_entry['id']}_{scheduler_name}_{n_steps}"
        if key in completed:
            skipped += 1
            continue

        result = run_single(model, prompt_entry, scheduler_name, n_steps, args.seed)
        results.append(result)
        done += 1
        completed.add(key)
        save_partial(run_id, run_date, config, results)

        status = (
            "ERROR"
            if result.get("error")
            else f"clip={result['clip_score']:.4f} lat={result['latency_s']:.1f}s vram={result['vram_peak_gb']:.2f}GB"  # noqa: E501
        )
        logger.info(
            "[%d/%d] %s / %s / %dsteps — %s",
            done + skipped,
            total_combos,
            prompt_entry["id"],
            scheduler_name,
            n_steps,
            status,
        )

        # Progress log every 30 completions
        if done % 30 == 0:
            elapsed = time.time() - t_start
            rate = done / elapsed
            eta_s = (total_combos - done - skipped) / rate if rate > 0 else 0
            logger.info(
                "--- Progress: %d/%d done | %.0f min elapsed | ETA ~%.0f min ---",
                done,
                total_combos,
                elapsed / 60,
                eta_s / 60,
            )

    total_elapsed = time.time() - t_start
    ok_results = [r for r in results if not r.get("error")]
    logger.info(
        "=== Generation complete: %d ok, %d errors, %.1f min total ===",
        len(ok_results),
        len(results) - len(ok_results),
        total_elapsed / 60,
    )

    # Phase 2: heavy scorers — load after releasing the generation pipeline.
    if scorers & {"hps", "imagereward"}:
        logger.info("Releasing generation pipeline before loading scorers...")
        del model
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = None  # prevent re-use below

        ok_results_phase2 = [r for r in results if not r.get("error") and r.get("image_path")]
        _imgs_phase2 = []
        _prompts_phase2 = []
        from PIL import Image as _PILImage

        for r in ok_results_phase2:
            try:
                _imgs_phase2.append(_PILImage.open(r["image_path"]))
                _prompts_phase2.append(r["prompt"])
            except Exception as e:
                logger.warning("Could not load image for scoring: %s", e)
                _imgs_phase2.append(None)
                _prompts_phase2.append(r["prompt"])

        # Filter out None images
        _valid = [
            (img, p, r)
            for img, p, r in zip(_imgs_phase2, _prompts_phase2, ok_results_phase2, strict=False)
            if img is not None
        ]
        if _valid:
            _valid_imgs, _valid_prompts, _valid_results = zip(*_valid, strict=False)
            _valid_imgs = list(_valid_imgs)
            _valid_prompts = list(_valid_prompts)
            _valid_results = list(_valid_results)
        else:
            _valid_imgs, _valid_prompts, _valid_results = [], [], []

        if "hps" in scorers and _valid_imgs:
            logger.info("Scoring %d images with HPSv2.1...", len(_valid_imgs))
            from aetherart.eval_hps import release_hps, score_hps

            try:
                hps_scores = score_hps(_valid_imgs, _valid_prompts)
                for r, s in zip(_valid_results, hps_scores, strict=False):
                    r["hps_score"] = round(s, 6)
                logger.info("HPSv2.1 scoring complete.")
            except Exception as e:
                logger.error("HPSv2.1 scoring failed: %s", e)
            finally:
                release_hps()

        if "imagereward" in scorers and _valid_imgs:
            logger.info("Scoring %d images with ImageReward...", len(_valid_imgs))
            from aetherart.eval_ir import release_image_reward, score_image_reward

            try:
                ir_scores = score_image_reward(_valid_imgs, _valid_prompts)
                for r, s in zip(_valid_results, ir_scores, strict=False):
                    r["image_reward_score"] = round(s, 6)
                logger.info("ImageReward scoring complete.")
            except Exception as e:
                logger.error("ImageReward scoring failed: %s", e)
            finally:
                release_image_reward()

    # LPIPS: pairwise average within same-prompt groups (diversity metric).
    # For single-image-per-prompt runs (e.g. smoke test): always 0.0.
    # For multi-image runs (full benchmark): measures scheduler diversity.
    # Base-vs-LoRA LPIPS (the Phase 6b metric) wired in PR 13.
    if "lpips" in scorers:
        try:
            import lpips as _lpips_lib
            import torchvision.transforms as T

            _lpips_fn = _lpips_lib.LPIPS(net="alex", verbose=False)
            if torch.cuda.is_available():
                _lpips_fn = _lpips_fn.cuda()
            _to_tensor = T.Compose(
                [T.Resize((256, 256)), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)]
            )

            from PIL import Image as _PILImage

            ok_for_lpips = [r for r in results if not r.get("error") and r.get("image_path")]
            from collections import defaultdict

            prompt_groups: dict = defaultdict(list)
            for r in ok_for_lpips:
                prompt_groups[r["prompt"]].append(r)

            for prompt_group in prompt_groups.values():
                if len(prompt_group) < 2:
                    for r in prompt_group:
                        r["lpips_score"] = 0.0
                    continue
                imgs_t = []
                for r in prompt_group:
                    try:
                        img = _PILImage.open(r["image_path"]).convert("RGB")
                        t = _to_tensor(img).unsqueeze(0)
                        if torch.cuda.is_available():
                            t = t.cuda()
                        imgs_t.append(t)
                    except Exception:
                        imgs_t.append(None)
                valid_t = [t for t in imgs_t if t is not None]
                if len(valid_t) < 2:
                    for r in prompt_group:
                        r["lpips_score"] = 0.0
                    continue
                dists = []
                for i in range(len(valid_t)):
                    for j in range(i + 1, len(valid_t)):
                        with torch.no_grad():
                            d = _lpips_fn(valid_t[i], valid_t[j]).item()
                        dists.append(d)
                avg_dist = sum(dists) / len(dists)
                for r, t in zip(prompt_group, imgs_t, strict=False):
                    r["lpips_score"] = round(avg_dist if t is not None else 0.0, 6)
            logger.info("LPIPS scoring complete.")
        except Exception as e:
            logger.error("LPIPS scoring failed: %s", e)

    # Save final JSON
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    final_json = REPORTS_DIR / f"eval_results_{run_id}.json"
    with final_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_id,
                "run_date": run_date,
                "config": config,
                "results": results,
                "completed_at": datetime.now().isoformat(),
            },
            f,
            indent=2,
            default=str,
        )
    logger.info("Results JSON: %s", final_json)

    # Charts
    logger.info("Generating charts...")
    chart_paths = generate_charts(results, run_id)
    for cp in chart_paths:
        logger.info("  Chart: %s", cp)

    # Markdown report
    report_path = generate_report(results, run_id, run_date, config, chart_paths)
    logger.info("Report: %s", report_path)

    # Console summary
    print("\n" + "=" * 60)
    print(f"Run ID: {run_id}")
    print(f"Total combos: {total_combos}  |  Done: {done}  |  Skipped (resumed): {skipped}")
    print(f"Errors: {len(results) - len(ok_results)}")
    print(f"Elapsed: {total_elapsed / 60:.1f} min")
    if ok_results:
        for sched in schedulers:
            sub = [r for r in ok_results if r["scheduler"] == sched]
            if sub:
                avg_clip = np.mean([r["clip_score"] for r in sub])
                avg_lat = np.mean([r["latency_s"] for r in sub])
                avg_vram = np.mean([r["vram_peak_gb"] for r in sub])
                hps_vals = [r["hps_score"] for r in sub if r.get("hps_score") is not None]
                ir_vals = [
                    r["image_reward_score"] for r in sub if r.get("image_reward_score") is not None
                ]
                lpips_vals = [r["lpips_score"] for r in sub if r.get("lpips_score") is not None]
                score_str = f"CLIP(cmp-only)={avg_clip:.4f}"
                if hps_vals:
                    score_str += f"  HPS={np.mean(hps_vals):.4f}"
                if ir_vals:
                    score_str += f"  IR={np.mean(ir_vals):.4f}"
                if lpips_vals:
                    score_str += f"  LPIPS={np.mean(lpips_vals):.4f}"
                print(f"  {sched:8s}: {score_str}  lat={avg_lat:.1f}s  vram={avg_vram:.2f}GB")
    print(f"\nJSON:   {final_json}")
    print(f"Report: {report_path}")
    print("Charts:", ", ".join(Path(p).name for p in chart_paths))
    print("=" * 60)
    print("EVAL_SMOKE_PASSED")


if __name__ == "__main__":
    main()
