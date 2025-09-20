from typing import Any, Dict
from PIL import Image
import io
import base64

def pil_to_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def safe_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    return d.get(key, default)
