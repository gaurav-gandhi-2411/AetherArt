# Contributing

This is a personal learning project, not an open-source library soliciting PRs. That said, if you're reading the code or extending it for your own experiments, here's exactly how to reproduce the environment and what the quality checks expect.

## Dev environment

Requires Python 3.10, CUDA 12.4 (optional — CPU inference works but is slow), and conda.

```bash
git clone https://github.com/gaurav-gandhi-2411/AetherArt.git
cd AetherArt

conda create -n aetherart python=3.10 -y
conda activate aetherart

# Runtime dependencies
pip install -r requirements.txt

# GPU torch (CUDA 12.4) — skip for CPU-only
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Dev tools (linting, formatting, type checking, testing)
pip install -r requirements-dev.txt

# Install the package in editable mode so tests can import it
pip install -e .
```

**Reproducibility note:** `requirements-lock.txt` records the exact package versions used during development (Python 3.10.18, CUDA 12.4, April 2026). Use it if you need an exact-match environment:

```bash
pip install -r requirements-lock.txt
```

One known conflict: `opencv-python` and `opencv-python-headless` both provide `cv2` — install only one. The lock file uses `opencv-python`; on a headless server, swap it for `opencv-python-headless`.

## Pre-commit hooks

```bash
pre-commit install
```

This installs three hooks that run on every `git commit`:
- **black** (line length 100) — auto-reformats
- **isort** (black-compatible profile) — auto-sorts imports
- **flake8** — lints and fails if there are errors

First-time install downloads the hook environments. After that, hooks run in under a second.

## Running tests

```bash
pytest -q                                   # all 100 tests, fast summary
pytest --cov=aetherart --cov-report=term-missing  # with coverage (59% baseline)
pytest tests/test_registry.py -v            # one file, verbose
```

Tests run without a GPU. Pipeline-heavy paths (actual model loading, LoRA weight loading, ControlNet forward passes) are mocked throughout. The test suite finishes in ~60–90 seconds on a cold start due to one-time import overhead.

## Lint, format, type check

Run the same checks that CI runs locally:

```bash
flake8 aetherart/ app.py scripts/ tests/
black --check aetherart/ app.py scripts/ tests/
isort --check-only aetherart/ app.py scripts/ tests/
mypy aetherart
```

To auto-fix format issues (don't use `--check`):

```bash
black aetherart/ app.py scripts/ tests/
isort aetherart/ app.py scripts/ tests/
```

mypy is in soft mode (`check_untyped_defs = false`, `disallow_untyped_defs = false`) and currently reports 0 errors. The target is to keep it at 0.

## CI sequence

GitHub Actions runs on every push to `main` and on PRs:

1. `flake8` — lint (max line length 100; per-file ignores for vendored scripts)
2. `black --check` — format
3. `isort --check-only` — import order
4. `mypy aetherart` — type checking (soft mode, 0 errors)
5. `pytest --cov=aetherart --cov-report=xml` — tests + coverage report

All five must pass. See `.github/workflows/ci.yml` for the exact commands.

## GPU vs CPU paths

The codebase is designed to degrade gracefully:

- **GPU (CUDA):** Full feature set — Standard, LCM, and Turbo speed modes; all memory modes; ControlNet; LoRA.
- **CPU:** Standard mode only, at ~5–8 min/image. LCM and Turbo modes are GPU-only (`torch.cuda.is_available()` guards control the UI options).
- **HF Inference API:** Set `USE_HF_INFERENCE=1` to route generation through the Hugging Face Inference API instead of loading models locally. Useful for the live Space deployment.

The `aetherart/gpu_hygiene.py` module handles cleanup safely when CUDA is unavailable — all torch calls are wrapped in try/except ImportError.

## Project layout

```
aetherart/          # core library (typed, mypy-clean, PEP 561)
├── model.py        # SD 2.1 pipeline
├── registry.py     # pipeline singleton owner
├── controlnet.py   # ControlNet with LRU cache
├── lora.py         # LoRA registry + helpers
├── lcm.py          # LCM scheduler switching
├── sdxl_turbo.py   # SDXL Turbo pipeline
├── quantization.py # bitsandbytes INT8/NF4
├── metadata.py     # PNG tEXt + sidecar JSON
├── clip_scorer.py  # CLIP-based eval scoring
├── gpu_hygiene.py  # GPU memory cleanup
├── config.py       # env-driven config
└── logger.py       # shared logger

scripts/            # standalone generation and eval scripts
tests/              # pytest suite (100 tests, no GPU required)
reports/            # benchmark data and findings
docs/               # gallery images and samples
```
