#!/bin/bash
# gcp_startup_pr13.sh — PR 13 SDXL eval run startup script
#
# Runs on a GCP L4 VM (g2-standard-4, us-central1-a).
# Installs deps, applies hpsv2 turtle fix, downloads SDXL LoRA,
# runs all 9 SDXL experiments + analysis, pushes results to GCS,
# then self-deletes the VM via a trap on EXIT.
#
# Usage: supply as --metadata-from-file startup-script=scripts/gcp_startup_pr13.sh
# at VM creation time, or paste directly into the startup-script metadata field.
set -euo pipefail

INSTANCE_NAME="aetherart-eval-001"
ZONE="us-central1-a"
PROJECT="review-iq-prod"
GCS_BUCKET="gs://aetherart-eval-pr13"
BRANCH="feat/pr13-sdxl-experiments"
LOG_FILE="/tmp/eval_run.log"
REPO_DIR="/home/user/AetherArt"

# ── CRITICAL: self-delete VM on exit (success OR error) ──────────────────────
# compute billing stops immediately; disk is deleted with the instance.
# The shutdown -h below is a backstop only — this trap is the primary teardown.
trap 'echo "[TEARDOWN] Deleting VM at $(date)..." && \
      gcloud compute instances delete "${INSTANCE_NAME}" \
        --zone="${ZONE}" --project="${PROJECT}" --quiet && \
      echo "[TEARDOWN] VM deleted."' EXIT

# Safety backstop: VM hard-stops after 12 h (720 min) in case the trap is
# somehow bypassed.  Note: shutdown -h stops compute billing but does NOT
# delete the disk — the trap above handles full deletion first.
sudo shutdown -h +720 &

# ── Redirect all output to log (tee keeps stdout for cloud-init journal) ──────
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== PR 13 SDXL Eval Run starting at $(date) ==="
echo "Instance: ${INSTANCE_NAME}, Zone: ${ZONE}, Project: ${PROJECT}"
echo "Branch: ${BRANCH}, GCS: ${GCS_BUCKET}"

# ── System packages ───────────────────────────────────────────────────────────
apt-get update -qq
apt-get install -y -qq git curl python3-pip

# ── Clone repo ────────────────────────────────────────────────────────────────
if [ -d "$REPO_DIR" ]; then
    echo "Repo directory already exists — pulling latest."
    git -C "$REPO_DIR" fetch origin
    git -C "$REPO_DIR" checkout "$BRANCH"
    git -C "$REPO_DIR" pull origin "$BRANCH"
else
    git clone https://github.com/gaurav-gandhi-2411/AetherArt.git "$REPO_DIR"
    git -C "$REPO_DIR" checkout "$BRANCH"
fi

echo "Repo at branch: $(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD) ($(git -C "$REPO_DIR" rev-parse --short HEAD))"

cd "$REPO_DIR"

# ── Python dependencies ───────────────────────────────────────────────────────
pip install -q -r requirements.txt

# Scorer packages — install last so their deps don't get overridden.
pip install -q "hpsv2==1.2.0"
pip install -q "image-reward==1.5" --no-deps
pip install -q "fairscale==0.4.13" "openai-clip==1.0.1" "timm==1.0.27"
pip install -q lpips

echo "Python deps installed."

# ── hpsv2 1.2.0 turtle import fix ─────────────────────────────────────────────
# hpsv2 1.2.0 factory.py contains `from turtle import forward` — turtle requires
# a display, which crashes on headless Linux.  Strip the line in-place.
# See: docs/hpsv2_1.2.0_turtle_bug.md and commit d70c2d7.
SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
FACTORY="${SITE}/hpsv2/src/open_clip/factory.py"

if [ -f "$FACTORY" ]; then
    python3 - "$FACTORY" <<'PYEOF'
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
    print("hpsv2 turtle import line not found — fix already applied or version differs.")
PYEOF
else
    echo "WARNING: factory.py not found at ${FACTORY}"
    echo "Searching for alternate location..."
    find "$SITE" -name "factory.py" -path "*/hpsv2/*" 2>/dev/null | head -5 || true
fi

# ── hpsv2 BPE vocab fix ───────────────────────────────────────────────────────
# If the openai-clip package installed its vocab file somewhere hpsv2 can't find,
# copy it to the location hpsv2 expects.
CLIP_VOCAB=$(find "$SITE" -name "bpe_simple_vocab_16e6.txt.gz" -path "*/clip/*" 2>/dev/null | head -1 || true)
HPS_CLIP_DIR="${SITE}/hpsv2/src/open_clip"
HPS_CLIP_VOCAB="${HPS_CLIP_DIR}/bpe_simple_vocab_16e6.txt.gz"

if [ -n "$CLIP_VOCAB" ] && [ ! -f "$HPS_CLIP_VOCAB" ]; then
    cp "$CLIP_VOCAB" "$HPS_CLIP_VOCAB"
    echo "Copied BPE vocab from ${CLIP_VOCAB} to ${HPS_CLIP_VOCAB}."
else
    echo "BPE vocab check: source=${CLIP_VOCAB:-not found}, target exists=$([ -f "$HPS_CLIP_VOCAB" ] && echo yes || echo no)"
fi

# ── Download SDXL LoRA from HuggingFace Hub ───────────────────────────────────
python3 - <<'PYEOF'
from huggingface_hub import hf_hub_download
from pathlib import Path

local = Path("data/lora/ukiyo-e-sdxl")
local.mkdir(parents=True, exist_ok=True)
dest = local / "pytorch_lora_weights.safetensors"

if dest.exists():
    print(f"SDXL LoRA already present at {dest} — skipping download.")
else:
    hf_hub_download(
        repo_id="gauravgandhi2411/aetherart-ukiyo-sdxl",
        filename="pytorch_lora_weights.safetensors",
        local_dir=str(local),
    )
    print(f"SDXL LoRA downloaded to {dest}")
PYEOF

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

    python3 "scripts/experiments/${EXP}.py"
    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: ${EXP} failed with exit code ${EXIT_CODE}"
        # Push partial results and log before the VM teardown trap fires.
        gsutil -m cp -r reports/experiments/ "${GCS_BUCKET}/experiments/" || true
        gsutil cp "$LOG_FILE" "${GCS_BUCKET}/eval_run.log" || true
        echo "FAILED:${EXP}" | gsutil cp - "${GCS_BUCKET}/STATUS"
        exit $EXIT_CODE  # triggers EXIT trap → VM deletion
    fi

    # Push this experiment's output immediately so results survive a later failure.
    # Use a glob that matches the actual output directory name (e.g. exp1_sdxl or
    # exp1_quantization_quality_sdxl depending on the script's OUT setting).
    gsutil -m cp -r reports/experiments/"${EXP}"* "${GCS_BUCKET}/experiments/" || true

    echo "=== ${EXP} complete at $(date) ==="
done

# ── CLIP-blindness analysis ───────────────────────────────────────────────────
echo ""
echo "=== Running CLIP-blindness analysis at $(date) ==="
python3 scripts/generate_clip_blindness_sdxl.py

# ── Push final results ────────────────────────────────────────────────────────
echo ""
echo "=== Pushing final results to ${GCS_BUCKET} at $(date) ==="

gsutil cp reports/clip_blindness_sdxl.md "${GCS_BUCKET}/" || true
gsutil cp reports/clip_blindness_sdxl_chart.png "${GCS_BUCKET}/" || true
gsutil -m cp -r reports/experiments/ "${GCS_BUCKET}/experiments/" || true

# Log is still being written — flush by copying last.
gsutil cp "$LOG_FILE" "${GCS_BUCKET}/eval_run.log"

echo ""
echo "=== All runs complete at $(date) ==="
echo "COMPLETED" | gsutil cp - "${GCS_BUCKET}/STATUS"

# EXIT trap fires here — deletes the VM.
