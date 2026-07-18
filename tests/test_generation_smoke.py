"""Real (unmocked) end-to-end generation smoke test.

Unlike every other test in this suite, this one is deliberately NOT mocked — it loads the
actual SD 2.1 pipeline and runs a real (CPU-fallback, minimal-step) generation. Its purpose is
narrow: catch silent breakage of the torch/diffusers/CUDA import-and-execute chain that a fully
mocked suite cannot see — e.g. the missing torch/bin -> torch/lib DLL split
(fbgemm.dll/asmjit.dll) found during the SDXL latency root-cause investigation
(docs/LATENCY_ROOT_CAUSE.md §0), which left `import torch` failing outright while every mocked
test still passed. Runs on CPU in CI (no GPU runner); uses the real GPU when one is available
locally.

Network + compute cost: downloads/loads the ~5 GB SD 2.1 weights (cached by CI) and runs 2
inference steps at 512x512 — a few seconds warm, longer on a cold cache. Kept out of the default
`pytest` run (see the `gen_smoke` marker in pyproject.toml) and invoked as its own CI step.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.gen_smoke


def test_real_generation_produces_a_valid_nonempty_image():
    import numpy as np
    import torch

    from aetherart.model import AetherModel

    model = AetherModel()
    model.init()  # default SD 2.1 path — same entrypoint app.py/scripts/eval.py use
    assert model.backend == "local", (
        f"expected a real local pipeline, got backend={model.backend!r} "
        "(InferenceClient fallback would defeat the point of this smoke test)"
    )

    # Calls model.pipe(...) directly, exactly as app.py's generation stream and
    # scripts/eval.py's run_single() do — NOT AetherModel.generate(), which wraps the call in an
    # extra torch.autocast("cuda") on top of an already-fp16-loaded pipeline. That redundant
    # autocast was found (while writing this test) to produce all-NaN/black output at low step
    # counts; generate() has no callers anywhere in the codebase (dead code) so it isn't fixed
    # here — see the PR description for the flagged follow-up.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(42)
    out = model.pipe(
        "a red apple on a white table",
        num_inference_steps=2,
        guidance_scale=7.5,
        width=512,
        height=512,
        generator=generator,
    )
    image = out.images[0]

    assert image is not None
    assert image.size == (512, 512)

    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    assert arr.shape == (512, 512, 3)
    # Catches the G1-class failure mode (NaN/all-black output from a misconfigured VAE/dtype
    # combo) that a "did it crash" check alone would miss.
    assert np.isfinite(arr).all(), "generated image contains NaN/Inf pixels"
    assert arr.std() > 1.0, f"generated image is degenerate (near-flat, std={arr.std():.3f})"

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
