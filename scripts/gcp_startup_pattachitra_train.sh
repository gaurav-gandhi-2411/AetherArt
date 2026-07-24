#!/bin/bash
# gcp_startup_pattachitra_train.sh — Pattachitra curated LoRA training run, per
# docs/PATTACHITRA_AB_PREREGISTRATION.md. Trains ONE adapter (curated corpus only, 100 images
# after manual QA on top of the automated 111-clean filter). Uploads checkpoints to GCS and
# self-tears-down; scripts/gcp_verify_before_teardown.py is run LOCALLY after pulling results,
# before any instance deletion is done from the operator side (this script's own teardown trap
# is a safety net for orphaned-cost prevention, not a substitute for that local verification).
set -euo pipefail

PROJECT="aetherart-497918"
INSTANCE_NAME=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/name" -H "Metadata-Flavor: Google")
ZONE=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/zone" -H "Metadata-Flavor: Google" | awk -F/ '{print $NF}')
GCS_BUCKET="gs://aetherart-497918-training"
BRANCH="chore/model-verdict-audit"
LOG_FILE="/tmp/train_run.log"
REPO_DIR="/tmp/AetherArt"

if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    # shellcheck source=/dev/null
    source /opt/conda/etc/profile.d/conda.sh
    conda activate base
    PYTHON=/opt/conda/bin/python3
    PIP=/opt/conda/bin/pip
else
    PYTHON=$(command -v python3 || command -v python)
    PIP=$(command -v pip3 || command -v pip)
fi

# IMPORTANT: this instance does NOT self-delete. It uploads results to GCS then STOPS (billing
# for GPU/CPU stops; a stopped instance's disk still accrues a small storage charge until deleted
# manually). Deletion happens only after scripts/gcp_verify_before_teardown.py has verified the
# pulled-down results LOCALLY — the ~25-min result loss earlier this session came from deleting
# an instance before that verification step, and this script is deliberately designed so that
# mistake cannot recur here even under a script bug or crash (a stopped VM's disk is still
# recoverable; a deleted one, per gcloud's default, is not).
teardown() {
    local exit_code=$?
    echo "[TEARDOWN] Exit code ${exit_code} — starting at $(date)" 2>&1 | tee -a "$LOG_FILE" || true
    gcloud storage cp "$LOG_FILE" "${GCS_BUCKET}/pattachitra_train_run.log" 2>/dev/null || true
    gcloud storage cp -r "${REPO_DIR}/data/lora/pattachitra-curated/training_output_sdxl_pattachitra_curated" \
        "${GCS_BUCKET}/" 2>/dev/null || true
    echo "[TEARDOWN] Writing DONE marker..."
    echo "done exit=${exit_code} at $(date)" | gcloud storage cp - "${GCS_BUCKET}/PATTACHITRA_TRAIN_DONE" 2>/dev/null || true
    echo "[TEARDOWN] Stopping (NOT deleting) VM at $(date) — manual verify+delete required..."
    gcloud compute instances stop "${INSTANCE_NAME}" \
        --zone="${ZONE}" --project="${PROJECT}" --quiet 2>/dev/null || true
    echo "[TEARDOWN] Done."
}
trap teardown EXIT

# 6h hard shutdown backstop (1500 steps at rank-8/batch-1x4 should take well under 2h based on the
# ukiyo-e precedent's ~4h26m for a smaller/older setup; this is a safety margin, not an estimate).
# Stop, not delete - same reasoning as the teardown trap above.
(sleep 21600 && gcloud compute instances stop "${INSTANCE_NAME}" --zone="${ZONE}" --project="${PROJECT}" --quiet) &

exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Pattachitra curated LoRA training starting at $(date) ==="
echo "Instance: ${INSTANCE_NAME} | Zone: ${ZONE} | Project: ${PROJECT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "WARNING: nvidia-smi failed"
${PYTHON} -c "import torch; print(f'torch {torch.__version__}, CUDA {torch.version.cuda}, device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

apt-get update -qq
apt-get install -y -qq git

if [ -d "${REPO_DIR}/.git" ]; then
    git -C "$REPO_DIR" fetch origin
    git -C "$REPO_DIR" checkout "$BRANCH"
    git -C "$REPO_DIR" pull origin "$BRANCH"
else
    git clone --branch "$BRANCH" https://github.com/gaurav-gandhi-2411/AetherArt.git "$REPO_DIR"
fi
cd "$REPO_DIR"

echo "=== Installing dependencies ==="
${PIP} install -q --upgrade diffusers transformers accelerate peft datasets

echo "=== Downloading curated Pattachitra corpus from GCS ==="
mkdir -p data/lora/pattachitra-curated
gcloud storage cp -r "${GCS_BUCKET}/pattachitra-curated/images" data/lora/pattachitra-curated/
gcloud storage cp "${GCS_BUCKET}/pattachitra-curated/metadata.jsonl" data/lora/pattachitra-curated/
echo "Downloaded $(ls data/lora/pattachitra-curated/images | wc -l) images."

echo "=== Starting training at $(date) ==="
${PYTHON} scripts/_diffusers_train_text_to_image_lora_sdxl.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
    --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
    --train_data_dir="data/lora/pattachitra-curated" \
    --resolution=1024 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --max_train_steps=1500 \
    --checkpointing_steps=500 \
    --learning_rate=1e-4 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --mixed_precision="fp16" \
    --rank=8 \
    --seed=42 \
    --validation_epochs=15 \
    --validation_prompt="pattascroll Pattachitra painting of Lord Jagannath in a temple shrine" \
    --num_validation_images=2 \
    --dataloader_num_workers=2 \
    --output_dir="data/lora/pattachitra-curated/training_output_sdxl_pattachitra_curated" \
    --report_to="tensorboard"

echo "=== Training complete at $(date) ==="
echo "=== Uploading checkpoints to GCS ==="
gcloud storage cp -r "data/lora/pattachitra-curated/training_output_sdxl_pattachitra_curated" "${GCS_BUCKET}/"

echo "=== All done at $(date) — teardown trap will now fire ==="
