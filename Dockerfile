# Stage 1 ────────────────────────────────────────────────────────────────────
# Install all Python deps into an isolated venv.
# Same CUDA base as the runtime stage so the compiled extensions stay compatible.
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /venv
ENV PATH="/venv/bin:$PATH"

# PyTorch CUDA 12.6 first — large layer, keeps rebuild cache efficient.
RUN pip install --no-cache-dir \
        torch==2.8.0+cu126 \
        --index-url https://download.pytorch.org/whl/cu126

# Runtime deps — exact same pins as modal_app.py.
# Intentionally excludes: eval-only deps (hpsv2, image-reward), modal.
RUN pip install --no-cache-dir \
        gradio==5.46.1 \
        diffusers==0.35.1 \
        transformers==4.56.2 \
        accelerate==1.10.1 \
        "huggingface_hub[hf_xet]==0.35.0" \
        safetensors==0.6.2 \
        peft==0.19.1 \
        Pillow==11.3.0 \
        numpy==2.2.6 \
        opencv-python-headless==4.12.0.88 \
        python-dotenv==1.1.1 \
        bitsandbytes==0.49.2


# Stage 2 ────────────────────────────────────────────────────────────────────
# Lean runtime: same CUDA base, no venv build tools, non-root user.
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 appuser

# Inherit the pre-built venv; PATH makes `python` resolve to /venv/bin/python.
COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

WORKDIR /app

# App code only — .dockerignore keeps data/ reports/ tests/ weights out of context.
COPY aetherart/ ./aetherart/
COPY cloudrun_app.py .

# PORT injected by Cloud Run at runtime (default 8080).
# HF token and safety flag set via --set-secrets / --set-env-vars at deploy time.
ENV PORT=8080

USER appuser

CMD ["python", "cloudrun_app.py"]
