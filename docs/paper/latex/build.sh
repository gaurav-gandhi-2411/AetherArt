#!/bin/bash
# build.sh -- regenerate this paper's arXiv submission PDF from tracked source.
#
# Tracked (source, reviewable): abstract_source.md, main.tex, this script.
# Generated (gitignored, never hand-edited): abstract_body.tex, body_content.tex, main.pdf.
#
# Why generated output isn't tracked: this session's own abstract-drift incident --
# a hand-retyped copy of the abstract in docs/paper/arxiv_submission_prep.md silently
# drifted 3 characters from the actual submitted text -- is exactly the failure mode a
# tracked, hand-patched build artifact invites. Regenerating on demand from one source
# of truth (this script) makes "the committed .tex doesn't match the source it claims to
# come from" structurally impossible instead of relying on remembering to re-sync it.
#
# Why Unicode math symbols (≈, ≥, ≤) no longer need hand-patching after
# every regen (unlike the first version of this paper's LaTeX, which required 4 manual
# post-pandoc patches): main.tex now loads `newunicodechar` and maps these three symbols
# to their LaTeX math-mode equivalents once, in the preamble -- any literal occurrence of
# them anywhere in the generated body renders correctly with zero per-occurrence fixes,
# including symbols introduced by future edits to measurement_defects.md that this script
# has never seen before.
#
# Usage: ./build.sh   (run from this directory, or anywhere -- paths are script-relative)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SOURCE_MD="../measurement_defects.md"
BODY_TMP="$(mktemp)"
trap 'rm -f "$BODY_TMP"' EXIT

if ! command -v pandoc >/dev/null 2>&1; then
    echo "FATAL: pandoc not found on PATH." >&2
    exit 1
fi

TECTONIC_BIN="tectonic"
if ! command -v tectonic >/dev/null 2>&1; then
    if [ -x "$HOME/.local/tectonic/tectonic" ] || [ -x "$HOME/.local/tectonic/tectonic.exe" ]; then
        export PATH="$HOME/.local/tectonic:$PATH"
    else
        echo "FATAL: tectonic not found on PATH or in ~/.local/tectonic/." >&2
        exit 1
    fi
fi

echo "=== Extracting body (## 1. Introduction through EOF) from $SOURCE_MD ==="
sed -n '/^## 1\. Introduction/,$p' "$SOURCE_MD" > "$BODY_TMP"
if [ ! -s "$BODY_TMP" ]; then
    echo "FATAL: extraction produced an empty file -- '## 1. Introduction' heading not found in $SOURCE_MD (has the source been restructured?)." >&2
    exit 1
fi

echo "=== Converting body via pandoc ==="
pandoc -f markdown -t latex --shift-heading-level-by=-1 "$BODY_TMP" -o body_content.tex

echo "=== Converting abstract via pandoc ==="
pandoc -f markdown -t latex abstract_source.md -o abstract_body.tex

echo "=== Compiling with tectonic ==="
tectonic main.tex

echo "=== Done: main.pdf ==="
