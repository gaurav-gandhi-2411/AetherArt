#!/usr/bin/env python
"""
Wrapper around the diffusers train_text_to_image_lora.py script.

Launches via accelerate with sane defaults for the RTX 3070 8GB Laptop GPU,
logs output to training_output/training.log, and reports elapsed time.

Usage:
    python scripts/train_lora.py                     # full 1500-step run
    python scripts/train_lora.py --max-train-steps 5 # pre-flight check
    python scripts/train_lora.py --rank 4 --lr 5e-5  # custom hyperparams
"""
import argparse
import importlib.util
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DIFFUSERS_SCRIPT = REPO_ROOT / "scripts" / "_diffusers_train_text_to_image_lora.py"
DATA_DIR = REPO_ROOT / "data" / "lora" / "ukiyo-e"
DEFAULT_OUTPUT_DIR = DATA_DIR / "training_output"


def parse_args():
    p = argparse.ArgumentParser(description="LoRA fine-tune wrapper for Ukiyo-e SD 2.1")
    p.add_argument("--model", default="sd2-community/stable-diffusion-2-1")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--train-batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-train-steps", type=int, default=1500)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--mixed-precision", default="fp16", choices=["no", "fp16", "bf16"])
    p.add_argument("--checkpointing-steps", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--validation-prompt",
        default="ukyowood ukiyo-e print of Mount Fuji at sunset",
    )
    p.add_argument("--num-validation-images", type=int, default=4)
    p.add_argument("--validation-epochs", type=int, default=1)
    p.add_argument(
        "--no-xformers", action="store_true", help="Disable xformers (fallback if not installed)"
    )
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory (default: data/lora/ukiyo-e/training_output)",
    )
    return p.parse_args()


def build_command(args, python_exe: str) -> list[str]:
    # Locate accelerate in the same env as the current interpreter.
    # Windows: python.exe lives at env root, scripts at env/Scripts/
    # Unix: both live in env/bin/
    if sys.platform == "win32":
        scripts_dir = Path(python_exe).parent / "Scripts"
    else:
        scripts_dir = Path(python_exe).parent
    accelerate_exe = str(scripts_dir / "accelerate")
    if sys.platform == "win32":
        accelerate_exe += ".exe"

    cmd = [
        accelerate_exe,
        "launch",
        "--mixed_precision",
        args.mixed_precision,
        str(DIFFUSERS_SCRIPT),
        "--pretrained_model_name_or_path",
        args.model,
        "--train_data_dir",
        str(DATA_DIR),
        "--caption_column",
        "text",
        "--resolution",
        str(args.resolution),
        "--train_batch_size",
        str(args.train_batch_size),
        "--gradient_accumulation_steps",
        str(args.grad_accum),
        "--learning_rate",
        str(args.lr),
        "--max_train_steps",
        str(args.max_train_steps),
        "--rank",
        str(args.rank),
        "--mixed_precision",
        args.mixed_precision,
        "--checkpointing_steps",
        str(args.checkpointing_steps),
        "--output_dir",
        str(DEFAULT_OUTPUT_DIR),
        "--validation_prompt",
        args.validation_prompt,
        "--num_validation_images",
        str(args.num_validation_images),
        "--validation_epochs",
        str(args.validation_epochs),
        "--seed",
        str(args.seed),
    ]

    if not args.no_gradient_checkpointing:
        cmd.append("--gradient_checkpointing")

    xformers_available = importlib.util.find_spec("xformers") is not None
    if not args.no_xformers and xformers_available:
        cmd.append("--enable_xformers_memory_efficient_attention")
    elif not args.no_xformers and not xformers_available:
        print("[train_lora] xformers not found — running without memory-efficient attention")

    return cmd


def main():
    args = parse_args()
    python_exe = sys.executable

    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training.log"

    cmd = build_command(args, python_exe)
    # patch --output_dir into the command (build_command uses OUTPUT_DIR directly)
    out_idx = cmd.index("--output_dir") + 1
    cmd[out_idx] = str(output_dir)

    print(f"[train_lora] Output dir : {output_dir}")
    print(f"[train_lora] Log file   : {log_path}")
    print(f"[train_lora] Steps      : {args.max_train_steps}")
    print(f"[train_lora] Rank       : {args.rank}")
    print(f"[train_lora] Command    : {' '.join(cmd)}\n")

    t0 = time.monotonic()

    with open(log_path, "w", buffering=1, encoding="utf-8") as log_fh:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(REPO_ROOT),
        )
        for line in proc.stdout:
            print(line, end="", flush=True)
            log_fh.write(line)

    elapsed = time.monotonic() - t0
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    elapsed_str = f"{h:02d}:{m:02d}:{s:02d}"

    rc = proc.wait()
    print(f"\n[train_lora] Finished in {elapsed_str} — exit code {rc}")

    with open(log_path, "a", encoding="utf-8") as log_fh:
        log_fh.write(f"\n--- elapsed {elapsed_str} | exit {rc} | rank {args.rank} ---\n")

    sys.exit(rc)


if __name__ == "__main__":
    main()
