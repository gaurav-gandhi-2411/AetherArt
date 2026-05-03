"""Resume gallery generation — categories 4, 5, 6 only (1-3 already complete)."""

from __future__ import annotations

import atexit
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aetherart.gpu_hygiene import cleanup_gpu  # noqa: E402

atexit.register(cleanup_gpu)

# Import everything from generate_gallery and run only cats 4-6
from scripts.generate_gallery import _timer, gen_canny, gen_depth, gen_turbo  # noqa: E402

if __name__ == "__main__":
    try:
        t_total = _timer()
        gen_canny(None)
        gen_depth(None)
        gen_turbo()
        print(f"\n=== DONE cats 4-6 — total time: {t_total()}s ===")
    finally:
        cleanup_gpu(verbose=True)
