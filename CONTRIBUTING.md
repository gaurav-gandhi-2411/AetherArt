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

## Modal demo deployment

The `modal_app.py` at the repo root is the recruiter-facing demo on Modal's free-tier A10G GPU.

**Prerequisites**

1. Install Modal SDK (dev dep only — not in `requirements.txt`):
   ```bash
   pip install modal==1.4.3
   ```
2. Authenticate once:
   ```bash
   modal setup
   ```
3. Create a Modal secret named **`huggingface`** in the [Modal dashboard](https://modal.com/secrets) with key `HUGGINGFACEHUB_API_TOKEN` set to your HF token.  Without this secret the app will not start.

**Stage B — ephemeral serve (dev / verify)**

```bash
modal serve modal_app.py
```

This spins up a temporary A10G container and prints an ephemeral public URL.  The URL lives only while the command runs.  Use it to verify cold-start time and that all four live modes work before deploying.

**Stage C — persistent deploy (after verification)**

```bash
modal deploy modal_app.py
```

Deploys to a persistent URL.  Containers spin up on request and idle down after 5 minutes.  Run this only after Stage B is verified.

**Live demo modes**

| Mode key | Description |
|---|---|
| `hyper_8step` | Hyper-SD 8-step LoRA, EulerDiscrete, CFG=5 (demo default) |
| `ukiyo_e` | Ukiyo-e LoRA from `gauravgandhi2411/aetherart-ukiyo-sdxl`, DPM-Solver++ |
| `sdxl_base` | SDXL 1.0 base, 30-step DPM-Solver++, no LoRA |
| `controlnet_canny` | ControlNet Union, Canny preprocessing, requires uploaded image |
| `controlnet_depth` | ControlNet Union, depth preprocessing, requires uploaded image |

Safety guard (`AETHERART_ENABLE_SAFETY=1`) is baked into the Modal image; the prompt blocklist in `aetherart/safety.py` runs on every request.  Flux and SD 2.1 paths are local-only and are not served by the demo.

## Cloud Run demo deployment (GCP L4)

The `cloudrun_app.py` at the repo root is the primary recruiter-facing demo, deployed on GCP Cloud Run with an NVIDIA L4 GPU and scale-to-zero (₹0 idle).

**Project:** `aetherart-497918` · **Region:** `us-central1` · **Image registry:** `us-central1-docker.pkg.dev/aetherart-497918/aetherart-demo/app`

**Prerequisites**

1. Authenticate gcloud and set the project:
   ```bash
   gcloud auth login
   gcloud config set project aetherart-497918
   ```
2. Create a Secret Manager secret for the HF token (one-time setup):
   ```bash
   # Enable Secret Manager if not already enabled
   gcloud services enable secretmanager.googleapis.com --project aetherart-497918

   # Create the secret (paste token when prompted)
   printf '%s' "YOUR_HF_TOKEN" | gcloud secrets create huggingface-token \
     --project=aetherart-497918 \
     --replication-policy=automatic \
     --data-file=-

   # Grant the default Cloud Run service account read access
   gcloud secrets add-iam-policy-binding huggingface-token \
     --project=aetherart-497918 \
     --member="serviceAccount:473907703523-compute@developer.gserviceaccount.com" \
     --role="roles/secretmanager.secretAccessor"
   ```

**Build the image (Cloud Build)**

```bash
gcloud builds submit \
  --tag us-central1-docker.pkg.dev/aetherart-497918/aetherart-demo/app:v1 \
  --project aetherart-497918 \
  .
```

Cloud Build uses `.gcloudignore` to exclude large/dev-only files from the upload.

**Deploy to Cloud Run (Stage C — after build verification)**

```bash
gcloud run deploy aetherart-demo \
  --image us-central1-docker.pkg.dev/aetherart-497918/aetherart-demo/app:v1 \
  --region us-central1 \
  --project aetherart-497918 \
  --gpu 1 \
  --gpu-type nvidia-l4 \
  --cpu 4 \
  --memory 16Gi \
  --min-instances 0 \
  --max-instances 1 \
  --concurrency 1 \
  --timeout 300 \
  --no-gpu-zonal-redundancy \
  --set-secrets HUGGINGFACEHUB_API_TOKEN=huggingface-token:latest \
  --set-env-vars AETHERART_ENABLE_SAFETY=1 \
  --allow-unauthenticated
```

Key config rationale:
- `--min-instances 0` — scale-to-zero; ₹0 cost while idle
- `--max-instances 1` — one GPU instance maximum; prevents parallel cost overrun
- `--concurrency 1` — GPU-bound; one generation at a time
- `--no-gpu-zonal-redundancy` — single-zone placement; required for the `nvidia_l4_gpu_allocation_no_zonal_redundancy` quota granted on this project
- `--timeout 300` — request timeout (model load completes before first request; generation takes 5–30 s)
- `--allow-unauthenticated` — public recruiter demo

**Live demo modes**

| Mode key | Description |
|---|---|
| `hyper_8step` | Hyper-SD 8-step + Ukiyo-e LoRA composed, EulerDiscrete, CFG=5 **[default]** |
| `ukiyo_e` | Ukiyo-e LoRA from `gauravgandhi2411/aetherart-ukiyo-sdxl`, 30-step DPM-Solver++ |
| `controlnet_canny` | ControlNet Union, Canny preprocessing, requires uploaded image |
| `controlnet_depth` | ControlNet Union, depth preprocessing, requires uploaded image |

Cold start (~2 min on L4): models load once at container startup; warm generations are ~5 s (Hyper) / ~15 s (LoRA / ControlNet).

## Modal demo deployment (alternate reference)

`modal_app.py` is kept in the repo as a reference for deploying on Modal (A10G). It is no longer the primary deploy target. See the Cloud Run section above for the current production path.

**Prerequisites**

1. Install Modal SDK (dev dep only — not in `requirements.txt`):
   ```bash
   pip install modal==1.4.3
   ```
2. Authenticate once:
   ```bash
   modal setup
   ```
3. Create a Modal secret named **`huggingface`** in the [Modal dashboard](https://modal.com/secrets) with key `HUGGINGFACEHUB_API_TOKEN` set to your HF token.  Without this secret the app will not start.

**Stage B — ephemeral serve (dev / verify)**

```bash
modal serve modal_app.py
```

This spins up a temporary A10G container and prints an ephemeral public URL.  The URL lives only while the command runs.  Use it to verify cold-start time and that all four live modes work before deploying.

**Stage C — persistent deploy (after verification)**

```bash
modal deploy modal_app.py
```

Deploys to a persistent URL.  Containers spin up on request and idle down after 5 minutes.  Run this only after Stage B is verified.

## GPU vs CPU paths

The codebase is designed to degrade gracefully:

- **GPU (CUDA):** Full feature set — Standard, LCM, and Turbo speed modes; all memory modes; ControlNet; LoRA.
- **CPU:** Standard mode only, at ~5–8 min/image. LCM and Turbo modes are GPU-only (`torch.cuda.is_available()` guards control the UI options).
- **HF Inference API:** Set `USE_HF_INFERENCE=1` to route generation through the Hugging Face Inference API instead of loading models locally. Useful for the live Space deployment.

The `aetherart/gpu_hygiene.py` module handles cleanup safely when CUDA is unavailable — all torch calls are wrapped in try/except ImportError.

## Gated models

**SDXL Turbo** (`aetherart/sdxl_turbo.py`) requires `AETHERART_ENABLE_LEGACY=1` to load. Its license (Stability AI SDXL Turbo Research License) permits non-commercial research use only, making it unsuitable for the commercial-intent demo. Set the env var when running Turbo-dependent scripts locally:

```bash
AETHERART_ENABLE_LEGACY=1 python scripts/generate_four_tier_showcase.py
```

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
