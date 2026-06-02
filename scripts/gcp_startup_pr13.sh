#!/bin/bash
# gcp_startup_pr13.sh — PR 13 SDXL eval run startup script (v3)
#
# Fixed from v1:
#   - REPO_DIR uses /tmp/ (always exists on DLVM; /home/user/ may not)
#   - Source conda base env so torch/CUDA packages are available
#   - Skip torch in pip install (DLVM ships torch 2.9 + CUDA 12.9; don't downgrade)
#   - Use gcloud storage instead of gsutil (no Python dependency)
#   - Teardown trap pushes log + partial results before VM deletion
#   - Remove dead EXIT_CODE check (set -e exits immediately on failure)
# v3: Auto-detect INSTANCE_NAME and ZONE from GCP metadata server (zone-agnostic)
set -euo pipefail

PROJECT="review-iq-prod"
# Auto-detect instance name and zone from GCP metadata server (works in any zone)
INSTANCE_NAME=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/name" -H "Metadata-Flavor: Google")
ZONE=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/zone" -H "Metadata-Flavor: Google" | awk -F/ '{print $NF}')
GCS_BUCKET="gs://aetherart-eval-pr13"
BRANCH="feat/pr13-sdxl-experiments"
LOG_FILE="/tmp/eval_run.log"
REPO_DIR="/tmp/AetherArt"

# ── Activate conda (DLVM has torch/CUDA in conda base; not active for root) ──
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    # shellcheck source=/dev/null
    source /opt/conda/etc/profile.d/conda.sh
    conda activate base
    PYTHON=/opt/conda/bin/python3
    PIP=/opt/conda/bin/pip
elif [ -f /opt/conda/bin/python3 ]; then
    PYTHON=/opt/conda/bin/python3
    PIP=/opt/conda/bin/pip
else
    PYTHON=$(command -v python3 || command -v python)
    PIP=$(command -v pip3 || command -v pip)
fi

# ── Teardown: push log + partial results, then delete VM ─────────────────────
teardown() {
    local exit_code=$?
    echo "[TEARDOWN] Exit code ${exit_code} — starting teardown at $(date)" 2>&1 | tee -a "$LOG_FILE" || true
    # Push whatever results exist
    gcloud storage cp "$LOG_FILE" "${GCS_BUCKET}/eval_run.log" 2>/dev/null || true
    gcloud storage cp -r "${REPO_DIR}/reports/experiments" "${GCS_BUCKET}/" 2>/dev/null || true
    gcloud storage cp "${REPO_DIR}/reports/clip_blindness_sdxl.md" "${GCS_BUCKET}/" 2>/dev/null || true
    echo "[TEARDOWN] Deleting VM at $(date)..."
    gcloud compute instances delete "${INSTANCE_NAME}" \
        --zone="${ZONE}" --project="${PROJECT}" --quiet 2>/dev/null || true
    echo "[TEARDOWN] Done."
}
trap teardown EXIT

# 12h hard shutdown backstop (compute billing stops; trap handles full deletion)
sudo shutdown -h +720 &

# Redirect all output to log (tee keeps stdout in cloud-init journal too)
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== PR 13 SDXL Eval Run starting at $(date) ==="
echo "Python: ${PYTHON}"
echo "pip:    ${PIP}"
echo "Instance: ${INSTANCE_NAME} | Zone: ${ZONE} | Project: ${PROJECT}"
echo "Branch: ${BRANCH} | GCS: ${GCS_BUCKET}"
echo "Repo dir: ${REPO_DIR}"

# GPU sanity check
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "WARNING: nvidia-smi failed"
${PYTHON} -c "import torch; print(f'torch {torch.__version__}, CUDA {torch.version.cuda}, device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# ── System packages ───────────────────────────────────────────────────────────
apt-get update -qq
apt-get install -y -qq git

# ── Clone repo ────────────────────────────────────────────────────────────────
if [ -d "${REPO_DIR}/.git" ]; then
    echo "Repo already present — pulling."
    git -C "$REPO_DIR" fetch origin
    git -C "$REPO_DIR" checkout "$BRANCH"
    git -C "$REPO_DIR" pull origin "$BRANCH"
else
    git clone https://github.com/gaurav-gandhi-2411/AetherArt.git "$REPO_DIR"
    git -C "$REPO_DIR" checkout "$BRANCH"
fi
echo "Repo: $(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD) @ $(git -C "$REPO_DIR" rev-parse --short HEAD)"

cd "$REPO_DIR"

# ── Python dependencies ───────────────────────────────────────────────────────
# Exclude torch/torchvision/torchaudio — DLVM ships torch 2.9 + CUDA 12.9.
# Installing torch==2.5.1 from requirements.txt would DOWNGRADE and break GPU.
echo "Installing Python deps (excluding torch — already on DLVM)..."
grep -vE "^(torch|torchvision|torchaudio)==" requirements.txt | ${PIP} install -q -r /dev/stdin

# Scorer packages (ordering matters — see requirements-dev.txt notes)
${PIP} install -q "hpsv2==1.2.0"
${PIP} install -q "image-reward==1.5" --no-deps
${PIP} install -q "fairscale==0.4.13" "openai-clip==1.0.1" "timm==1.0.27"
${PIP} install -q lpips
echo "Python deps installed."

# ── hpsv2 1.2.0 turtle import fix ────────────────────────────────────────────
SITE=$(${PYTHON} -c "import site; print(site.getsitepackages()[0])")
FACTORY="${SITE}/hpsv2/src/open_clip/factory.py"

if [ -f "$FACTORY" ]; then
    ${PYTHON} - "$FACTORY" <<'PYEOF'
import sys
path = sys.argv[1]
with open(path, encoding="utf-8") as fh:
    content = fh.read()
patched = content.replace("from turtle import forward\n", "")
if patched != content:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(patched)
    print(f"hpsv2 turtle fix applied: {path}")
else:
    print("turtle import not found — fix already applied or version differs.")
PYEOF
else
    echo "WARNING: factory.py not found at ${FACTORY}"
    find "$SITE" -name "factory.py" -path "*/hpsv2/*" 2>/dev/null | head -5 || true
fi

# ── hpsv2 BPE vocab fix ───────────────────────────────────────────────────────
CLIP_VOCAB=$(find "$SITE" -name "bpe_simple_vocab_16e6.txt.gz" -path "*/clip/*" 2>/dev/null | head -1 || true)
HPS_CLIP_VOCAB="${SITE}/hpsv2/src/open_clip/bpe_simple_vocab_16e6.txt.gz"
if [ -n "$CLIP_VOCAB" ] && [ ! -f "$HPS_CLIP_VOCAB" ]; then
    cp "$CLIP_VOCAB" "$HPS_CLIP_VOCAB"
    echo "Copied BPE vocab from ${CLIP_VOCAB} → ${HPS_CLIP_VOCAB}"
else
    echo "BPE vocab: source='${CLIP_VOCAB:-not found}' target_exists=$([ -f "$HPS_CLIP_VOCAB" ] && echo yes || echo no)"
fi

# ── Download SDXL Ukiyo-e LoRA ────────────────────────────────────────────────
${PYTHON} - <<'PYEOF'
from huggingface_hub import hf_hub_download
from pathlib import Path

local = Path("data/lora/ukiyo-e-sdxl")
local.mkdir(parents=True, exist_ok=True)
dest = local / "ukiyo-e-sdxl-lora.safetensors"

if dest.exists():
    import os
    mb = os.path.getsize(dest) / 1e6
    print(f"SDXL LoRA already present at {dest} ({mb:.1f} MB)")
else:
    hf_hub_download(
        repo_id="gauravgandhi2411/aetherart-ukiyo-sdxl",
        filename="ukiyo-e-sdxl-lora.safetensors",
        local_dir=str(local),
    )
    print(f"SDXL LoRA downloaded to {dest}")
PYEOF

# Push a heartbeat so GCS shows the run started
echo "RUNNING" | gcloud storage cp - "${GCS_BUCKET}/STATUS"
gcloud storage cp "$LOG_FILE" "${GCS_BUCKET}/eval_run.log" || true

# ── Run experiments ───────────────────────────────────────────────────────────
EXPS=(
    exp1_sdxl
    exp2_sdxl
    exp3_sdxl
    exp4_sdxl
    exp5_sdxl
    exp6_sdxl
    exp7_sdxl
    exp8_sdxl
    exp9_sdxl
)

for EXP in "${EXPS[@]}"; do
    echo ""
    echo "=== Running ${EXP} at $(date) ==="
    ${PYTHON} "scripts/experiments/${EXP}.py"
    # Push this experiment's results immediately so partial results survive a later failure
    gcloud storage cp -r reports/experiments "${GCS_BUCKET}/" 2>/dev/null || true
    gcloud storage cp "$LOG_FILE" "${GCS_BUCKET}/eval_run.log" || true
    echo "=== ${EXP} complete at $(date) ==="
done

# ── CLIP-blindness analysis ───────────────────────────────────────────────────
echo ""
echo "=== Running CLIP-blindness analysis at $(date) ==="
${PYTHON} scripts/generate_clip_blindness_sdxl.py

# ── Push final results ────────────────────────────────────────────────────────
echo ""
echo "=== Pushing final results to ${GCS_BUCKET} at $(date) ==="
gcloud storage cp reports/clip_blindness_sdxl.md "${GCS_BUCKET}/" || true
gcloud storage cp reports/clip_blindness_sdxl_chart.png "${GCS_BUCKET}/" 2>/dev/null || true
gcloud storage cp -r reports/experiments/ "${GCS_BUCKET}/" || true
gcloud storage cp "$LOG_FILE" "${GCS_BUCKET}/eval_run.log"

echo ""
echo "=== All runs complete at $(date) ==="
echo "COMPLETED" | gcloud storage cp - "${GCS_BUCKET}/STATUS"

# EXIT trap fires → teardown()
