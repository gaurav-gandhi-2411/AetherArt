# Building this paper

**Tracked (source, reviewable):** `abstract_source.md`, `main.tex`, `build.sh`, this file.
**Generated (gitignored, never hand-edited, never committed):** `abstract_body.tex`,
`body_content.tex`, `main.pdf`.

This split is deliberate, not the AgentGauge repo's own convention (which currently
commits `body_content.tex`/`main.pdf`) — the reasoning is stated once, here, rather than
assumed obvious: a generated file that's also hand-patched and committed invites drift
between the source that supposedly produced it and what's actually on disk. This
project already hit that failure mode once — a *documentation copy* of the compressed
abstract (`docs/paper/arxiv_submission_prep.md`) was hand-retyped from the real source
and silently drifted 3 characters (real em-dashes swapped in for the source's literal
`--`), caught only by an independent verifier pass recomputing the character count, not
by inspection. Tracking generated LaTeX invites the same class of drift for a much
larger file. Regenerating on demand from one source of truth removes the failure mode
structurally instead of relying on remembering to re-sync it.

## Build

```bash
./build.sh
```

Requires `pandoc` and [tectonic](https://tectonic-typesetting.github.io/) on `PATH`
(the script also checks `~/.local/tectonic/` if `tectonic` isn't found there directly —
same toolchain as `agentgauge/docs/paper/latex/`). Produces `main.pdf` (12 pages) plus
the two intermediate `.tex` files, none of which should be committed.

The script: extracts `docs/paper/measurement_defects.md` from `## 1. Introduction`
through EOF, converts it via
`pandoc -f markdown -t latex --shift-heading-level-by=-1` (the heading-level shift maps
this paper's `##`-level top sections to `\section`, matching AgentGauge's convention),
converts `abstract_source.md` the same way, then compiles with tectonic. No manual
steps required between pandoc and tectonic — see below for why.

## Why no post-pandoc hand-patching is needed (unlike this paper's first LaTeX draft)

Pandoc emits Unicode math comparison symbols (`≈`, `≥`, `≤`) as literal characters,
which the base text font (Latin Modern, loaded via `lmodern`/`fontspec`) has no glyph
for — confirmed via a real `tectonic` compile that logged `Missing character` warnings
at 4 locations the first time this paper was converted. The first draft fixed this by
hand-wrapping each occurrence in `$\approx$`/`$\geq$` directly inside the generated
`body_content.tex` — which worked, but would have silently broken again on the very
next regeneration (exactly the drift risk this file's intro section describes), since
hand-patches to a gitignored, regenerate-on-demand file don't survive being
regenerated. Fixed properly instead: `main.tex` loads `newunicodechar` and maps all
three symbols to their math-mode equivalents once, in the preamble
(`\newunicodechar{≈}{\ensuremath{\approx}}`, etc.) — any literal occurrence anywhere in
the generated body now renders correctly with zero per-occurrence fixes, including
symbols introduced by future edits to `measurement_defects.md` this preamble has never
seen before. Verified by regenerating from scratch via `build.sh` and confirming
`body_content.tex` contains the raw, unpatched Unicode characters while the compiled
PDF still renders them correctly (visual page-render check, not just a clean compile).

## Other notes

No `references.bib`/`natbib` in this paper's `main.tex` (unlike AgentGauge's): this
paper cites no external literature — every reference is to this repository's own files,
scripts, and commit hashes, rendered as inline `\texttt{}` monospace, not a bibliography
entry.

`\usepackage{calc}` (not in AgentGauge's `main.tex`) is required because this paper's
pandoc version emits computed-width `longtable` columns
(`(\columnwidth - 6\tabcolsep) * \real{0.25}`) — confirmed via a real compile error
(`Missing number, treated as zero`) without it.

CI sync-check parity with AgentGauge (`scripts/check_paper_latex_sync.py`) is not
needed here for the same reason the hand-patch problem above doesn't recur: there is no
committed generated file that can go stale relative to its source, because none is
committed. AgentGauge's sync-check exists to guard a hand-mirrored `.tex` copy; this
paper has no such copy to guard.
