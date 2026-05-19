# Torch 2.8 Compatibility Audit

**Date:** 2026-05-19  
**Spike branch:** spike/torch-2.8-compat  
**Scratch env:** aetherart-torch28 (Python 3.10, torch 2.8.0+cu126, fresh conda env)  
**Current pin:** torch==2.5.1 (requirements.txt)  
**Test target:** torch==2.8.0+cu126 (download.pytorch.org/whl/cu126)  

---

## Summary Verdict: COMPATIBLE

torch 2.8.0 is a **drop-in replacement** for the current 2.5.1 pin. No code changes are required in any `aetherart/*.py` file. All 100 tests pass. The real-generation smoke test passes on the RTX 3070 8 GB GPU.

---

## Test Suite Delta

| Metric | torch 2.5.1 (main) | torch 2.8.0 (spike) |
|--------|--------------------|----------------------|
| Tests collected | 100 | 100 |
| Passed | 100 | 100 |
| Failed | 0 | 0 |
| Warnings | 1 (FutureWarning, diffusers) | 1 (same FutureWarning, diffusers) |

**Pre-existing warning (not torch-related):**
```
diffusers/configuration_utils.py:250: FutureWarning: It is deprecated to pass a 
pretrained model name or path to `from_config`. [...] This functionality will be 
removed in v1.0.0.
```
This appears under both torch versions and is a diffusers 0.35.1 issue. It surfaces in `test_registry.py::TestQuantized::test_get_quantized_evicts_other_mode` when the scheduler is swapped. No action required here.

---

## Dependency Resolver

`pip check` in the scratch env: **No broken requirements found.**

All current pins resolve cleanly against torch 2.8.0:

| Package | Current pin | Status with torch 2.8.0 | Action |
|---------|-------------|--------------------------|--------|
| `diffusers` | 0.35.1 | Compatible | None |
| `transformers` | 4.56.2 | Compatible | None |
| `accelerate` | 1.10.1 | Compatible | None |
| `bitsandbytes` | 0.49.2 | **Compatible** (see note) | None |
| `peft` | 0.19.1 | Compatible | None |
| `huggingface_hub` | 0.35.0 | Compatible | None |
| `gradio` | 5.46.1 | Compatible | None |

**bitsandbytes 0.49.2 note:** This was the highest-risk dependency. bitsandbytes has historically had torch version coupling (e.g., INT8 CUDA kernels built against specific torch versions). Under torch 2.8.0, `bitsandbytes==0.49.2` imports cleanly (`bitsandbytes.functional` importable), and the quantization test suite passes (6 tests). A full INT8/NF4 inference run was not performed (no model weights present in scratch env), but the import chain and configuration construction succeed. The PR 06 GPU smoke test (`@pytest.mark.gpu`) will be the final validation when quantization code is actually run.

---

## Import-Time Warnings

Zero warnings or deprecations at import of: `torch`, `diffusers`, `transformers`, `accelerate`, `bitsandbytes`, `peft`.

---

## Real-Generation Smoke Test

```
torch: 2.8.0+cu126
CUDA available: True
Device: NVIDIA GeForce RTX 3070 Laptop GPU
VRAM total: 8.0 GB

Initialising ModelRegistry...
[INFO] Initializing model 'sd2-community/stable-diffusion-2-1' (use_inference=False)
[INFO] Enabled attention slicing to reduce VRAM usage.
[INFO] Enabled model CPU offload to reduce VRAM pressure.
[INFO] Loaded SD 2.1 pipeline on cuda
Backend: local

Running 10-step test generation...
Image mean pixel value: 101.48
Latency: 16.7s
SMOKE TEST PASSED
Peak VRAM during generation: 6840 MB
```

Assertion `mean_pixel > 10` passes (101.48). Not all-black. No CUDA errors. VRAM peak (6840 MB) is within 8 GB total.

---

## API Breakage Audit

Code paths checked against known torch 2.5 → 2.8 changes:

| API | Location | Status |
|-----|----------|--------|
| `torch.autocast("cuda")` | `model.py:244` | OK — this is the modern form (not the deprecated `torch.cuda.amp.autocast`); no change in 2.8 |
| `torch.Generator(device=device).manual_seed(seed)` | `model.py:243` | OK — stable API |
| `torch.backends.cudnn.benchmark = True` | `model.py:220` | OK — stable API |
| `torch.cuda.is_available()` | multiple files | OK — stable API |
| `torch.cuda.empty_cache()` | `registry.py:111` | OK — stable API |
| `torch.cuda.memory_allocated()` | `quantization.py:76` | OK — stable API |
| `torch.cuda.max_memory_allocated()` | `quantization.py:81` | OK — stable API |
| `torch.float16` | `quantization.py`, `model.py` | OK — stable constant |
| `torch.compile` | nowhere in codebase | N/A — not used |
| `torch.cuda.amp.autocast` | nowhere in codebase | N/A — already using modern form |
| xformers integration | `model.py:192` | N/A — optional, guarded via `hasattr` |

No torch API changes required.

---

## ZeroGPU Context Note

The original motivation for this audit was ZeroGPU's RTX Pro 6000 Blackwell (sm_120) requirement for PyTorch 2.8+ (new compute architecture kernels). This requirement is **moot for the current deployment path**: D2-new = Modal, and Modal containers pin their own torch version independently of the host. The Modal image for PR 07 should pin `torch==2.8.0+cu126` explicitly (A10G is sm_86, fully supported).

For local dev and the documented GCP Cloud Run fallback, torch 2.5.1 continues to work. The upgrade to 2.8.0 is optional for those paths.

**Recommended upgrade action:**  
Update `requirements.txt` torch pin to `torch==2.8.0` with the cu126 install comment. This is a quality-of-life improvement (stay closer to latest), not an emergency. Deferred to PR 07 (Modal deployment) as part of the container image definition, or can be folded into any PR 03+ when SDXL work begins.

---

## Reproducibility: Scratch Env Pip Freeze

```
accelerate==1.10.1
aetherart==0.1.0 (editable, from spike branch)
aiofiles==24.1.0
annotated-types==0.7.0
anyio==4.13.0
bitsandbytes==0.49.2
certifi==2026.4.22
charset-normalizer==3.4.7
click==8.4.0
colorama==0.4.6
coverage==7.14.0
diffusers==0.35.1
exceptiongroup==1.3.1
fastapi==0.136.1
ffmpy==1.0.0
filelock==3.29.0
fsspec==2026.4.0
gradio==5.46.1
gradio_client==1.13.1
hf-xet==1.5.0
httpcore==1.0.9
httpx==0.28.1
huggingface-hub==0.35.0
idna==3.15
importlib_metadata==9.0.0
iniconfig==2.3.0
Jinja2==3.1.6
markdown-it-py==4.2.0
MarkupSafe==3.0.3
mpmath==1.3.0
mypy==1.16.0
mypy_extensions==1.1.0
networkx==3.4.2
numpy==2.2.6
opencv-python-headless==4.12.0.88
orjson==3.11.9
pandas==2.3.3
pathspec==1.1.1
peft==0.19.1
pillow==11.3.0
pluggy==1.6.0
psutil==7.2.2
pydantic==2.11.10
pydantic_core==2.33.2
pydub==0.25.1
Pygments==2.20.0
pytest==8.4.2
pytest-cov==6.2.1
python-dateutil==2.9.0.post0
python-dotenv==1.1.1
python-multipart==0.0.29
pytz==2026.2
PyYAML==6.0.3
regex==2026.5.9
requests==2.34.2
rich==15.0.0
ruff==0.15.13
safetensors==0.6.2
semantic-version==2.10.0
shellingham==1.5.4
six==1.17.0
starlette==0.52.1
sympy==1.14.0
tokenizers==0.22.2
tomli==2.4.1
tomlkit==0.13.3
torch==2.8.0+cu126
tqdm==4.67.3
transformers==4.56.2
typer==0.25.1
typing_extensions==4.15.0
tzdata==2026.2
urllib3==2.7.0
uvicorn==0.47.0
websockets==15.0.1
zipp==4.1.0
```
