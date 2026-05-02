"""Resume gallery generation — categories 4, 5, 6 only (1-3 already complete)."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import everything from generate_gallery and run only cats 4-6
from scripts.generate_gallery import gen_canny, gen_depth, gen_turbo, _timer
import torch

if __name__ == "__main__":
    t_total = _timer()
    gen_canny(None)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gen_depth(None)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gen_turbo()
    print(f"\n=== DONE cats 4-6 — total time: {t_total()}s ===")
