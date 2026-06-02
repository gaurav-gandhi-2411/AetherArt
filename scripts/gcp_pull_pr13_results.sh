#!/bin/bash
# gcp_pull_pr13_results.sh — Pull PR 13 SDXL eval results from GCS.
#
# Run locally AFTER the GCP VM has completed (check STATUS = COMPLETED).
# Downloads all results from gs://aetherart-eval-pr13/ into the repo's
# reports/ directory and tails the run log.
#
# Usage (from any directory inside the repo):
#   bash scripts/gcp_pull_pr13_results.sh
set -euo pipefail

GCS_BUCKET="gs://aetherart-eval-pr13"

# Resolve repo root so this script works from any subdirectory.
REPO_DIR="$(git rev-parse --show-toplevel)"
cd "$REPO_DIR"

echo "=== Pulling PR 13 results from ${GCS_BUCKET} ==="
echo "Repo root: ${REPO_DIR}"
echo ""

# ── STATUS check ─────────────────────────────────────────────────────────────
echo "--- Run status ---"
STATUS_TMP=$(mktemp /tmp/pr13_status_XXXXXX.txt)
if gsutil cp "${GCS_BUCKET}/STATUS" "$STATUS_TMP" 2>/dev/null; then
    cat "$STATUS_TMP"
else
    echo "WARNING: STATUS file not found — run may still be in progress or failed early."
fi
echo ""

# ── Download report files ─────────────────────────────────────────────────────
echo "--- Downloading report files ---"
gsutil cp "${GCS_BUCKET}/clip_blindness_sdxl.md" reports/ && \
    echo "Downloaded: reports/clip_blindness_sdxl.md" || \
    echo "WARNING: clip_blindness_sdxl.md not found (analysis may not have run yet)"

gsutil cp "${GCS_BUCKET}/clip_blindness_sdxl_chart.png" reports/ 2>/dev/null && \
    echo "Downloaded: reports/clip_blindness_sdxl_chart.png" || \
    echo "WARNING: clip_blindness_sdxl_chart.png not found"
echo ""

# ── Download experiment outputs ───────────────────────────────────────────────
echo "--- Downloading experiment directories ---"
# -m: parallel multi-threaded transfer
# -r: recursive
# The trailing slash on the source path means "copy the contents into" the
# destination, not "create a nested experiments/ inside experiments/".
gsutil -m cp -r "${GCS_BUCKET}/experiments/*" reports/experiments/ && \
    echo "Downloaded: reports/experiments/" || \
    echo "WARNING: No experiment outputs found in ${GCS_BUCKET}/experiments/"
echo ""

# ── Download run log ──────────────────────────────────────────────────────────
echo "--- Run log (last 50 lines) ---"
LOG_TMP=$(mktemp /tmp/pr13_eval_XXXXXX.log)
if gsutil cp "${GCS_BUCKET}/eval_run.log" "$LOG_TMP" 2>/dev/null; then
    tail -50 "$LOG_TMP"
    echo ""
    echo "Full log saved to: ${LOG_TMP}"
else
    echo "WARNING: eval_run.log not found — VM may still be running."
fi

echo ""
echo "=== Done. Results in ${REPO_DIR}/reports/ ==="
