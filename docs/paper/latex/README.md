# Building this paper

LaTeX engine: [tectonic](https://tectonic-typesetting.github.io/) (standalone binary,
no system TeX distribution required). Installed to `~/.local/tectonic/` — not on PATH
by default (same toolchain as `agentgauge/docs/paper/latex/`).

```bash
export PATH="$HOME/.local/tectonic:$PATH"
tectonic main.tex
```

Produces `main.pdf` (12 pages). `abstract_body.tex` and `body_content.tex` are
`\input`-ed by `main.tex`.

`body_content.tex` was generated from `docs/paper/measurement_defects.md` §1–§10 via:

```bash
pandoc -f markdown -t latex --shift-heading-level-by=-1 <extracted-body>.md -o body_content.tex
```

(`--shift-heading-level-by=-1` maps this paper's `##`-level top sections to `\section`,
matching AgentGauge's convention where top-level sections render as `\section`, not
`\subsection`.) `abstract_body.tex` was generated the same way from the compressed
arXiv abstract (not the full in-repo abstract — see `docs/paper/arxiv_submission_prep.md`
§2 for why the two differ and where the compressed version lives).

**Four hand-patches were required after pandoc conversion** (not re-derivable by
re-running pandoc blindly — re-apply these if regenerating `body_content.tex` from a
revised `measurement_defects.md`): pandoc emits `≈` (U+2248) and `≥` (U+2265) as literal
Unicode characters, which the base font (`lmodern` via `fontspec`, no full Unicode math
font selected) cannot render — confirmed via a real `tectonic` compile that logged
`Missing character` warnings at those exact 4 locations (two `≈`, two `≥`). Fixed by
wrapping each in inline math mode (`$\approx$`, `$\geq$`) directly in `body_content.tex`,
matching AgentGauge's own documented practice (`revision_changelog.md`: hand-patch
LaTeX-only issues at known locations rather than re-running pandoc, to avoid re-reviewing
the whole file's fidelity every time).

No `references.bib`/`natbib` in this paper's `main.tex` (unlike AgentGauge's): this
paper cites no external literature — every reference is to this repository's own files,
scripts, and commit hashes, rendered as inline `\texttt{}` monospace, not a bibliography
entry. Added `\usepackage{calc}` (not in AgentGauge's `main.tex`) because this paper's
pandoc version emits computed-width `longtable` columns
(`(\columnwidth - 6\tabcolsep) * \real{0.25}`) that require it — confirmed via a real
compile error (`Missing number, treated as zero`) without it.

CI sync-check parity with AgentGauge (`scripts/check_paper_latex_sync.py`) was **not**
added here — out of this task's scope; noted as a future-work gap if this paper is
revised further, same tradeoff AgentGauge's own `repo_triage.md` §6 documents (currently
held by manual-mirror discipline, not tooling).
