from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image
from PIL.PngImagePlugin import PngInfo


def get_git_commit() -> str:
    """Return the short SHA of the current HEAD, or 'unknown' if outside a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def save_image_with_metadata(
    image: Image.Image,
    output_path: str | Path,
    metadata_dict: dict[str, Any],
) -> None:
    """Save image with metadata embedded as PNG tEXt chunks and a sidecar .json file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    png_info = PngInfo()
    for key, value in metadata_dict.items():
        png_info.add_text(str(key), str(value))

    image.save(str(output_path), format="PNG", pnginfo=png_info)

    sidecar_path = output_path.with_suffix(".json")
    with sidecar_path.open("w", encoding="utf-8") as f:
        json.dump(metadata_dict, f, indent=2, default=str)


def load_metadata_from_image(path: str | Path) -> dict[str, Any]:
    """Read PNG tEXt chunks from a saved image and return them as a dict."""
    with Image.open(Path(path)) as img:
        return dict(img.text) if getattr(img, "text", None) else {}
