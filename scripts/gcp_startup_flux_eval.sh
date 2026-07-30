#!/bin/bash
# gcp_startup_flux_eval.sh — FLUX.1-schnell 30-prompt x 3-seed cross-family verdict eval.
#
# Local feasibility check (RTX 3070 Laptop, 8.59GB VRAM, 2026-07-30) found FLUX.1-schnell's
# diffusers-format weights need ~33.7GB disk, more than the laptop had free — a disk blocker
# before VRAM was even reached (see aetherart/flux_pipeline.py's module docstring). This script
# runs the same canonical 30-prompt x 3-seed eval (scripts/model_verdict_harness.py --family
# flux_schnell) on a GCP L4 instance instead, which has ample disk/VRAM for either the bf16+
# cpu-offload or NF4-quantized loading path (main() below tries bf16+offload first, measures one
# image, and falls back to the quantized loader only if that's too slow for a 90-image run).
#
# Follows the same structure/safety patterns as scripts/gcp_startup_pattachitra_train.sh (STOP,
# never DELETE, from the trap; hard shutdown backstop; GCS log/result push) and
# scripts/gcp_startup_pr13.sh (DLVM conda activation, dependency-conflict workarounds, hpsv2
# post-install patches). Read both before changing this file.
set -euo pipefail

PROJECT="aetherart-497918"
INSTANCE_NAME=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/name" -H "Metadata-Flavor: Google")
ZONE=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/zone" -H "Metadata-Flavor: Google" | awk -F/ '{print $NF}')
GCS_BUCKET="gs://aetherart-497918-training/flux-eval"
BRANCH="main"
LOG_FILE="/tmp/flux_eval_run.log"
REPO_DIR="/tmp/AetherArt"
RESULT_JSON="reports/verdict_flux_schnell.json"

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

# IMPORTANT: this instance does NOT self-delete. It pushes results to GCS then STOPS (billing
# for GPU/CPU stops; a stopped instance's disk still accrues a small storage charge until
# deleted manually). Deletion happens only after scripts/gcp_verify_before_teardown.py has
# verified the pulled-down results LOCALLY — matching gcp_startup_pattachitra_train.sh's
# reasoning: a stopped VM's disk is still recoverable, a deleted one (gcloud's default) is not.
teardown() {
    local exit_code=$?
    echo "[TEARDOWN] Exit code ${exit_code} — starting at $(date)" 2>&1 | tee -a "$LOG_FILE" || true
    gcloud storage cp "$LOG_FILE" "${GCS_BUCKET}/flux_eval_run.log" 2>/dev/null || true
    gcloud storage cp "${REPO_DIR}/${RESULT_JSON}" "${GCS_BUCKET}/verdict_flux_schnell.json" 2>/dev/null || true
    echo "done exit=${exit_code} at $(date)" | gcloud storage cp - "${GCS_BUCKET}/FLUX_EVAL_DONE" 2>/dev/null || true
    echo "[TEARDOWN] Stopping (NOT deleting) VM at $(date) — manual verify+delete required..."
    gcloud compute instances stop "${INSTANCE_NAME}" \
        --zone="${ZONE}" --project="${PROJECT}" --quiet 2>/dev/null || true
    echo "[TEARDOWN] Done."
}
trap teardown EXIT

# 3h hard shutdown backstop (generous vs. an expected <1.5h 90-image run at either loading
# config; bounds cost if something hangs). Stop, not delete — same reasoning as the trap above.
(sleep 10800 && gcloud compute instances stop "${INSTANCE_NAME}" --zone="${ZONE}" --project="${PROJECT}" --quiet) &

exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== FLUX.1-schnell verdict eval starting at $(date) ==="
echo "Instance: ${INSTANCE_NAME} | Zone: ${ZONE} | Project: ${PROJECT}"

# ── Gated-repo read token ─────────────────────────────────────────────────────
# FLUX.1-schnell is gated on HF (Apache-2.0 license, but still requires the account to accept
# the gate + an authenticated token). Read from instance metadata (set via
# `gcloud compute instances add-metadata --metadata=hf-read-token=...` at launch time from the
# operator's local .env, never committed to this script or the repo) rather than embedding the
# value in this file's source text. This is a DIFFERENT token from the aetherart publish flow's
# write-scoped HF_TOKEN (which stays local-only and is never sent to this instance at all) — set
# only as HF_READ_TOKEN, matching aetherart/flux_pipeline.py's explicit lookup. Never echoed.
export HF_READ_TOKEN=$(curl -sf -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/hf-read-token")
if [ -z "$HF_READ_TOKEN" ]; then
    echo "FATAL: hf-read-token metadata key not set on this instance — cannot pull the gated FLUX.1-schnell repo." >&2
    exit 1
fi
echo "HF_READ_TOKEN present (length=${#HF_READ_TOKEN} chars, value not logged)."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "WARNING: nvidia-smi failed"
${PYTHON} -c "import torch; print(f'torch {torch.__version__}, CUDA {torch.version.cuda}, device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
echo "=== Free disk ==="
df -h /

apt-get update -qq
apt-get install -y -qq git

if [ -d "${REPO_DIR}/.git" ]; then
    git -C "$REPO_DIR" fetch origin
    git -C "$REPO_DIR" checkout "$BRANCH"
    git -C "$REPO_DIR" pull origin "$BRANCH"
else
    git clone --branch "$BRANCH" https://github.com/gaurav-gandhi-2411/AetherArt.git "$REPO_DIR"
fi
echo "Repo: $(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD) @ $(git -C "$REPO_DIR" rev-parse --short HEAD)"
cd "$REPO_DIR"

echo "=== Installing dependencies ==="
# Exclude torch/torchvision/torchaudio — DLVM ships torch 2.9 + CUDA 12.9 (per
# gcp_startup_pr13.sh's own comment: installing requirements.txt's torch==2.5.1 would DOWNGRADE
# and break GPU). bitsandbytes is needed only if the NF4-quantized fallback path is used below;
# installed unconditionally here since it's small and harmless if the bf16+offload path is fast
# enough to use instead.
grep -vE "^(torch|torchvision|torchaudio)==" requirements.txt | ${PIP} install -q -r /dev/stdin

# Scorer packages (ordering matters — see requirements-dev.txt notes; same sequence as
# gcp_startup_pr13.sh, proven working on this exact DLVM image family).
${PIP} install -q "hpsv2==1.2.0"
${PIP} install -q "image-reward==1.5" --no-deps
${PIP} install -q "fairscale==0.4.13" "openai-clip==1.0.1" "timm==1.0.27"
${PIP} install -q lpips
${PIP} install -q matplotlib datasets
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
fi

# ── hpsv2 BPE vocab fix ───────────────────────────────────────────────────────
CLIP_VOCAB=$(find "$SITE" -name "bpe_simple_vocab_16e6.txt.gz" -path "*/clip/*" 2>/dev/null | head -1 || true)
HPS_CLIP_VOCAB="${SITE}/hpsv2/src/open_clip/bpe_simple_vocab_16e6.txt.gz"
if [ -n "$CLIP_VOCAB" ] && [ ! -f "$HPS_CLIP_VOCAB" ]; then
    cp "$CLIP_VOCAB" "$HPS_CLIP_VOCAB"
    echo "Copied BPE vocab from ${CLIP_VOCAB} -> ${HPS_CLIP_VOCAB}"
else
    echo "BPE vocab: source='${CLIP_VOCAB:-not found}' target_exists=$([ -f "$HPS_CLIP_VOCAB" ] && echo yes || echo no)"
fi

# Push a heartbeat so GCS shows the run started
echo "RUNNING" | gcloud storage cp - "${GCS_BUCKET}/STATUS"
gcloud storage cp "$LOG_FILE" "${GCS_BUCKET}/flux_eval_run.log" || true

# ── Loading-strategy probe: bf16+cpu-offload first, one test image, measured ────────────────
# Per task instructions: try the already-committed load_flux_schnell() (bf16 + cpu offload, zero
# code changes) first. If ONE test image takes longer than ~60s, that would make the full
# 90-image run impractically slow (>1.5h), so fall back to an NF4-quantized loader instead.
# reset_peak_memory_stats() is called immediately before generation, not skipped — this project
# has a documented "phantom VRAM counter" defect (docs/paper/measurement_defects.md §4.2) from
# exactly that omission (a stale counter falsely reporting a physically-impossible peak carried
# over from an earlier run).
PROBE_LOG="/tmp/flux_loading_probe.log"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# `|| true` on both probe pipelines below is load-bearing, not cosmetic: this script runs under
# `set -e -o pipefail` (top of file), so a Python-side crash (e.g. CUDA OOM, a real failure this
# probe is specifically designed to catch and fall back from) would otherwise abort the ENTIRE
# script right here, before the `-z "$PROBE_GEN_S"` fallback-to-quantized branch below ever runs.
# Confirmed via a real run: bf16+cpu-offload OOM'd (the 23.78GB transformer module doesn't fit as
# a single resident unit in the L4's ~22GB usable VRAM, even with the rest of the pipeline
# offloaded to CPU — CPU offload swaps whole modules, not layers) and the missing `|| true` here
# killed the script before it could try the NF4-quantized fallback at all.
${PYTHON} - <<'PYEOF' 2>&1 | tee "$PROBE_LOG" || true
import time
import torch
import sys
sys.path.insert(0, "/tmp/AetherArt")
from aetherart.flux_pipeline import load_flux_schnell

t0 = time.time()
pipe = load_flux_schnell()
load_s = time.time() - t0
print(f"PROBE load_s={load_s:.1f}")

torch.cuda.reset_peak_memory_stats()
gen = torch.Generator(device="cuda").manual_seed(42)
t0 = time.time()
img = pipe(
    "a photograph of an astronaut riding a horse",
    num_inference_steps=4,
    guidance_scale=0.0,
    width=1024,
    height=1024,
    generator=gen,
).images[0]
gen_s = time.time() - t0
peak_gb = torch.cuda.max_memory_allocated() / 1e9
print(f"PROBE gen_s={gen_s:.1f} peak_vram_gb={peak_gb:.2f}")
PYEOF

PROBE_GEN_S=$(grep -oP 'PROBE gen_s=\K[0-9.]+' "$PROBE_LOG" | tail -1)
echo "=== Probe result: gen_s=${PROBE_GEN_S:-FAILED} ==="

USE_QUANTIZED=0
if [ -z "$PROBE_GEN_S" ]; then
    echo "=== Probe FAILED (no measurement) — falling back to NF4-quantized loader ==="
    USE_QUANTIZED=1
elif (( $(${PYTHON} -c "print(1 if ${PROBE_GEN_S} > 60 else 0)") )); then
    echo "=== bf16+offload probe measured ${PROBE_GEN_S}s/image (>60s threshold) — falling back to NF4-quantized loader ==="
    USE_QUANTIZED=1
else
    echo "=== bf16+offload probe measured ${PROBE_GEN_S}s/image (<=60s threshold) — using bf16+offload for full run ==="
fi

if [ "$USE_QUANTIZED" = "1" ]; then
    echo "=== Probing NF4-quantized loader for comparison ==="
    ${PYTHON} - <<'PYEOF' 2>&1 | tee -a "$PROBE_LOG" || true
import time
import torch
import sys
sys.path.insert(0, "/tmp/AetherArt")
from aetherart.flux_pipeline import load_flux_schnell_quantized

t0 = time.time()
pipe = load_flux_schnell_quantized()
load_s = time.time() - t0
print(f"PROBE_Q load_s={load_s:.1f}")

torch.cuda.reset_peak_memory_stats()
gen = torch.Generator(device="cuda").manual_seed(42)
t0 = time.time()
img = pipe(
    "a photograph of an astronaut riding a horse",
    num_inference_steps=4,
    guidance_scale=0.0,
    width=1024,
    height=1024,
    generator=gen,
).images[0]
gen_s = time.time() - t0
peak_gb = torch.cuda.max_memory_allocated() / 1e9
print(f"PROBE_Q gen_s={gen_s:.1f} peak_vram_gb={peak_gb:.2f}")
PYEOF
    PROBE_Q_GEN_S=$(grep -oP 'PROBE_Q gen_s=\K[0-9.]+' "$PROBE_LOG" | tail -1)
    if [ -z "$PROBE_Q_GEN_S" ]; then
        echo "FATAL: NF4-quantized probe also failed (no measurement) — both loading configs are" >&2
        echo "non-viable on this GPU. Not launching the full 90-image run against a config known" >&2
        echo "to fail. See ${PROBE_LOG} (already pushed to GCS) for the actual error." >&2
        exit 1
    fi
    echo "=== NF4-quantized probe measured ${PROBE_Q_GEN_S}s/image — using it for the full run ==="
    export AETHERART_FLUX_LOADER=quantized
    echo "=== Using NF4-quantized loader (AETHERART_FLUX_LOADER=quantized) for full run ==="
else
    export AETHERART_FLUX_LOADER=bf16_offload
    echo "=== Using bf16+cpu-offload loader (AETHERART_FLUX_LOADER=bf16_offload) for full run ==="
fi

gcloud storage cp "$PROBE_LOG" "${GCS_BUCKET}/flux_loading_probe.log" || true

# ── Pull already-completed partial results from GCS (resume support) ─────────────────────────
mkdir -p reports
gcloud storage cp "${GCS_BUCKET}/verdict_flux_schnell.json" "$RESULT_JSON" 2>/dev/null || true
RESUME_FLAG=""
if [ -f "$RESULT_JSON" ]; then
    echo "Found existing partial results at ${RESULT_JSON} — resuming."
    RESUME_FLAG="--resume"
fi

# ── Run the full eval, backgrounded so a dropped SSH session doesn't kill it ─────────────────
# CRITICAL: redirect the harness's own stdout/stderr to its own file, NOT the tee'd stream that
# feeds the GCE serial console — same "bufio.Scanner: token too long" crash risk documented in
# gcp_startup_pattachitra_train.sh (tqdm/progress-bar output can accumulate into one giant
# "line" that exceeds the console log-scanner's line-length limit and kills the whole startup
# script, silently orphaning the eval process too).
INNER_LOG="/tmp/flux_eval_inner.log"
touch "$INNER_LOG"
( while true; do sleep 60; gcloud storage cp "$INNER_LOG" "${GCS_BUCKET}/flux_eval_inner.log" 2>/dev/null || true; done ) &
PUSH_LOG_PID=$!

echo "=== Starting eval at $(date) — see ${INNER_LOG} for step-by-step progress ==="
nohup ${PYTHON} scripts/model_verdict_harness.py --family flux_schnell ${RESUME_FLAG} \
    > >(tr '\r' '\n' > "$INNER_LOG") 2> >(tr '\r' '\n' >> "$INNER_LOG") &
EVAL_PID=$!
wait "$EVAL_PID"
EVAL_EXIT=$?

kill "$PUSH_LOG_PID" 2>/dev/null || true
echo "=== Eval process exited with code ${EVAL_EXIT} at $(date) ==="
echo "=== Appending inner eval log to main log ==="
cat "$INNER_LOG" >> "$LOG_FILE" || true
gcloud storage cp "$INNER_LOG" "${GCS_BUCKET}/flux_eval_inner.log" 2>/dev/null || true

if [ "$EVAL_EXIT" -ne 0 ]; then
    echo "=== Eval FAILED (exit ${EVAL_EXIT}) — pushing partial results, teardown trap will fire ==="
    exit "$EVAL_EXIT"
fi

echo "=== Uploading final results to ${GCS_BUCKET} at $(date) ==="
gcloud storage cp "$RESULT_JSON" "${GCS_BUCKET}/verdict_flux_schnell.json"

echo "=== All done at $(date) — teardown trap will now fire ==="
