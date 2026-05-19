"""Publish the SD 2.1 Ukiyo-e LoRA adapter to Hugging Face Hub.

Usage:
    python scripts/publish_lora_sd21.py \\
        --adapter-path data/lora/ukiyo-e/ukiyo-e-lora.safetensors \\
        --repo-id gauravgandhi2411/aetherart-ukiyo-sd21 \\
        --dry-run

Exit codes:
    0  success
    1  validation failure (missing file, empty file, missing token)
    2  Hub API error
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from aetherart.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

_MODEL_CARD_PATH = _REPO_ROOT / "docs" / "model_cards" / "sd21_ukiyo_e.md"
_SAMPLE_IMAGES_DIR = _REPO_ROOT / "docs" / "model_cards" / "sd21_ukiyo_e_samples"


def _resolve_token() -> str | None:
    """Return a working HF token.

    Tries HUGGINGFACEHUB_API_TOKEN from env first; falls back to the
    credential cached by `huggingface-cli login` if the env token is
    absent or fails authentication.
    """
    from huggingface_hub import HfApi

    env_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if env_token:
        try:
            HfApi().whoami(token=env_token)
            return env_token
        except Exception:
            logger.debug("Env token invalid; falling back to cached credential")

    # Try cached token (stored by huggingface-cli login)
    try:
        from huggingface_hub.utils import HfFolder

        cached = HfFolder.get_token()
        if cached:
            HfApi().whoami(token=cached)
            return cached
    except Exception:
        pass

    return None


def _validate(adapter_path: Path) -> str | None:
    """Return an error message if validation fails, else None."""
    if not adapter_path.exists():
        return f"Adapter file not found: {adapter_path}"
    if adapter_path.stat().st_size == 0:
        return f"Adapter file is empty: {adapter_path}"
    if not _MODEL_CARD_PATH.exists():
        return f"Model card not found: {_MODEL_CARD_PATH}"
    if _resolve_token() is None:
        return (
            "No valid HF token found. Set HUGGINGFACEHUB_API_TOKEN or run "
            "`huggingface-cli login` to cache credentials."
        )
    return None


def _publish(adapter_path: Path, repo_id: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    token = _resolve_token()

    logger.info("Creating or verifying repo '%s'…", repo_id)
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=False, token=token)

    logger.info("Uploading adapter weights '%s'…", adapter_path.name)
    api.upload_file(
        path_or_fileobj=str(adapter_path),
        path_in_repo=adapter_path.name,
        repo_id=repo_id,
        repo_type="model",
        token=token,
        commit_message="upload SD 2.1 Ukiyo-e LoRA adapter weights",
    )

    logger.info("Uploading model card…")
    api.upload_file(
        path_or_fileobj=str(_MODEL_CARD_PATH),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        token=token,
        commit_message="upload model card",
    )

    if _SAMPLE_IMAGES_DIR.exists():
        logger.info("Uploading sample images from '%s'…", _SAMPLE_IMAGES_DIR)
        api.upload_folder(
            folder_path=str(_SAMPLE_IMAGES_DIR),
            repo_id=repo_id,
            repo_type="model",
            path_in_repo="samples",
            token=token,
            commit_message="upload sample images",
        )
    else:
        logger.info("No sample images directory found — skipping sample upload")

    logger.info("Published: https://huggingface.co/%s", repo_id)


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish SD 2.1 Ukiyo-e LoRA to HF Hub")
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=_REPO_ROOT / "data" / "lora" / "ukiyo-e" / "ukiyo-e-lora.safetensors",
        help="Path to the .safetensors adapter file",
    )
    parser.add_argument(
        "--repo-id",
        default="gauravgandhi2411/aetherart-ukiyo-sd21",
        help="HF Hub repo id (username/repo-name)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print planned actions; make no Hub API calls",
    )
    args = parser.parse_args()

    adapter_path: Path = args.adapter_path.resolve()

    error = _validate(adapter_path)
    if error:
        logger.error("Validation failed: %s", error)
        return 1

    size_mb = adapter_path.stat().st_size / 1024**2

    if args.dry_run:
        print("[DRY RUN] Validation passed.")
        print(f"[DRY RUN] Adapter:    {adapter_path}  ({size_mb:.1f} MB)")
        print(f"[DRY RUN] Model card: {_MODEL_CARD_PATH}")
        samples_status = (
            "found — " + str(_SAMPLE_IMAGES_DIR)
            if _SAMPLE_IMAGES_DIR.exists()
            else "not found - will skip"
        )
        print(f"[DRY RUN] Samples:    {samples_status}")
        print(f"[DRY RUN] Target:     https://huggingface.co/{args.repo_id}")
        print("[DRY RUN] Actions that would run:")
        print(f"[DRY RUN]   1. HfApi().create_repo('{args.repo_id}', exist_ok=True, private=False)")
        print(f"[DRY RUN]   2. upload_file '{adapter_path.name}' -> {args.repo_id}")
        print(f"[DRY RUN]   3. upload_file 'README.md' (model card) -> {args.repo_id}")
        if _SAMPLE_IMAGES_DIR.exists():
            print(f"[DRY RUN]   4. upload_folder samples/ -> {args.repo_id}/samples")
        print("[DRY RUN] No Hub API calls made.")
        return 0

    try:
        _publish(adapter_path, args.repo_id)
    except Exception as exc:
        logger.error("Hub API error: %s", exc)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
