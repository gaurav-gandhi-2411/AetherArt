# arXiv Submission Prep — "Five Silent Measurement-Validity Failures..."

**Status: packaged, NOT submitted.** Nothing in this document or `docs/paper/latex/` has
been uploaded to arxiv.org. Every step that requires a browser action on arxiv.org is
listed in §5 as a manual step for GG to perform.

Source of record: `docs/paper/measurement_defects.md` (fact-checked, merged to `main`
via PR #59). Workflow mirrors `agentgauge/docs/paper/latex/`'s toolchain (pandoc +
tectonic), adapted where this paper's content genuinely differs (no bibliography; see
`docs/paper/latex/README.md`).

---

## 1. Category selection

**Recommendation: primary `cs.SE` (Software Engineering), cross-list `cs.LG` (Machine
Learning).**

**Reasoning.** arXiv's `cs.SE` scope statement covers "design tools, software metrics,
testing and debugging, programming environments" — this paper's actual contribution is a
taxonomy of *measurement/testing-validity defects* (value-validity vs. semantic-validity,
§3) in an evaluation harness, discovered and classified using standard software-testing
reasoning (discovery mode, catchability by automated checks, a taxonomy of *why a check
fails to generalize*). Nothing in the paper proposes a new ML algorithm, model
architecture, or training method — the object under test (SDXL LoRA adapters) is
incidental to the finding; the finding itself is about how a *test harness* can be wrong
without ever producing an obviously-malformed value. That is squarely a testing/software-
validity contribution, which is `cs.SE`'s core scope, not `cs.LG`'s (whose scope is
methods/algorithms *for* learning, not the validity of a harness that scores a learned
model's output).

**Why cross-list `cs.LG` anyway.** The concrete evidence (the two reversals, §5) and the
object being measured (LoRA fine-tuning, VLM-judge scoring) sit squarely in the ML
practitioner audience's daily concerns — discoverability for the readers most likely to
apply this taxonomy (people running their own model evaluations) is meaningfully better
with an `cs.LG` cross-list than without one. `cs.AI` was considered as a second cross-list
(the paper touches LLM-as-judge concerns broadly) but not recommended — two categories
(one primary, one cross-list) is enough; a third dilutes rather than adds reach.

**Endorsement tradeoff, stated precisely (verify live, see caveat below):** arXiv requires
endorsement for an author submitting to a subject class they have no prior accepted-
submission history in, and endorsement is tracked per subject class, not per archive
(`cs` as a whole) or globally. If the `gaurav.gandhi` account's submission history so far
is in `cs.LG` (e.g., via the AgentGauge paper, submitted or in progress under `cs.LG`),
that history is what would let a `cs.LG` submission here skip a fresh endorsement request
— the account already has standing in that class. `cs.SE` is very likely a *class the
account has no history in yet*, meaning arXiv's system will probably require a fresh
endorsement (a named endorser with `cs.SE` submission history vouching for the account)
before a `cs.SE`-primary submission can go through.

**Caveat — I cannot verify this live.** I do not have browser access to arxiv.org and
cannot check the actual `gaurav.gandhi` account's endorsement status, submission
history, or arXiv's current endorsement policy text. The reasoning above is based on
arXiv's long-standing, publicly documented endorsement mechanics, not a live check. §5's
checklist tells GG exactly how to find out for certain (attempt the category selection
step and read what arXiv's own submission form says) rather than assuming the above is
still accurate at submission time.

**If a fresh `cs.SE` endorsement turns out to be a blocker and GG wants to avoid the
delay:** submitting `cs.SE` as the *cross-list* and `cs.LG` as *primary* is the fallback
that avoids it entirely (reusing the AgentGauge path for primary), at the cost of a
slightly weaker primary-category fit. This is a real tradeoff, not a strictly worse
option — stated so GG can decide with the actual endorsement-prompt information from §5
in hand, not before.

---

## 2. Title and abstract

**Title — 3 options, ranked by recommendation:**

1. **"Five Silent Measurement-Validity Failures in a Generative-Model Evaluation
   Project"** (recommended; used in the compiled PDF). Plain, descriptive, matches the
   in-repo paper's own title minus the word "Single" (dropped for tightness) —
   states the actual finding count up front, no hook, lowest risk of over-promising.
2. **"The Promotion That Wasn't: Five Silent Measurement-Validity Failures in a
   Generative-Model Evaluation Project."** Leads with the strongest single reversal
   (§5.1) as a hook, colon-subtitle structure matching AgentGauge's own title
   convention ("Tool-Description Quality Is Not One Axis: A Regime Analysis..."). More
   memorable, slightly more editorial.
3. **"When the Fallback Never Runs: Measurement-Validity Failures Across a VLM-Judge
   Harness and Its Own Tooling."** Leads with the cross-subsystem corroboration finding
   (§4.6) instead — the strongest *generalization* claim, not the strongest single
   number. Best if GG wants to emphasize "this isn't just one script's bug" over "here
   are the specific numbers."

To swap: edit the single `\title{...}` line in `docs/paper/latex/main.tex`, then
recompile (`tectonic main.tex`).

**Compressed abstract** (used in `docs/paper/latex/abstract_body.tex` and the compiled
PDF) — **1,610 characters**, counted directly (`len()` on the stripped text, not
estimated), comfortably under arXiv's commonly-cited ~1,920-character practical limit
(verify the live limit on the actual submission form — arXiv does not publish this as a
fixed documented number, and the practical enforcement point could differ from what is
commonly cited):

> A promotion decision collapsed from +0.040 (3.18 SEM) to +0.0078 (0.58 SEM) once
> independent per-axis judge scoring replaced a correlated multi-axis call, and a "LoRA
> loses to base" verdict reversed to "viable at adapter weight 0.3-0.5" once the same
> comparison was run at more than one operating point instead of only the library
> default. Both reversals came from the same single evaluation project -- scoring two
> SDXL LoRA style adapters and five base-model image-generation families with a local
> vision-language judge -- which produced five distinct silent measurement-validity
> failures over its course, each capable of standing as an unchallenged finding if left
> unaudited. We separate the five into a value-validity class (malformed or
> out-of-range values, catchable by automated range/uniqueness checks) and a
> semantic-validity class (well-formed values that silently answer the wrong question,
> not catchable by value checks alone), and show that each automated safeguard built in
> response to one defect failed to generalize to the next. Three further confirmed
> instances, found later in an unrelated subsystem (shell orchestration for a cloud GPU
> run), corroborate the same class recurring outside the original harness -- reported
> separately from the five, not folded into the count. A sixth, structurally similar
> anomaly is reported and left explicitly as an open, unresolved candidate, not a
> confirmed defect. We argue the taxonomy, and the discipline of re-auditing
> already-accepted conclusions rather than trusting an accumulating stack of automated
> checks, generalizes beyond one VLM-judge harness.

This deliberately differs from `measurement_defects.md`'s own in-repo abstract (which is
longer and leads with the project description, not the reversals) — the in-repo version
is untouched; this is a submission-specific compression, per the task's instruction to
lead with the reversals as the hook.

**Count check, verified exactly matching the body:** "five" (defects) / "three" (further
confirmed, corroborating, kept separate) / "a sixth... open, unresolved candidate" — all
three figures appear in the abstract with the identical qualifying language the body
uses (§4, §4.6, §7, §8), not rounded or merged. Same verification the independent
Haiku pass already ran against the full paper (prior session) — re-confirmed here by
direct read against the final compressed abstract text above.

---

## 3. Scrub pass — results

**Zero instances found requiring redaction.** Full grep sweep of
`docs/paper/measurement_defects.md` for: Windows absolute paths (`C:\`, `/Users/`),
the operator's username, GCP project IDs, GCS bucket names/URLs, cloud region/zone
names, IP addresses/hostnames, hardware model names, credentials/tokens/API keys
(pattern and keyword search), employer names, the operator's personal name, and any
mention of the AI tooling used to produce the work — all zero matches. The only "token"
matches are the ordinary word "token" in the context of LLM context-window token counts
(§4.1), not credentials.

This is not an accident: §4.6 (added in a prior session specifically for this paper)
was written to describe the GCP-script bug *mechanism* (shell exit-code propagation
under `set -e`/`pipefail`) without needing to name the specific cloud project, zone, or
bucket involved — those details live in `docs/FLUX_EVALUATION.md` and the actual
`scripts/gcp_startup_flux_eval.sh` (both already public in this repo, and out of this
scrub's scope since the task was about the *paper's own text*, not a re-audit of the
whole repository).

**One judgment call, not a redaction, flagged for GG's decision:** the paper does not
mention that an AI coding assistant was used to build the evaluation harness or write
this paper. This project's own standing convention (documented elsewhere in this
repository) is to disclose AI usage in ADRs/READMEs rather than hide it, while keeping
commit-metadata attribution clean. Whether to add an acknowledgment line to the paper
itself (common and increasingly expected in ML/SE venues) is GG's call, not something
this pass should decide unilaterally — flagged here rather than either added silently or
omitted silently.

---

## 4. Format conversion — verification

Converted to the same LaTeX toolchain as AgentGauge (pandoc → hand-patch → tectonic).
Full build notes: `docs/paper/latex/README.md`.

**Verified directly, not assumed:**
- `tectonic main.tex` compiles clean: 0 errors, 0 missing-character warnings (after 4
  hand-patches — see README), 12 pages, `main.pdf` produced (96.5 KB).
- All 4 tables (§3 taxonomy, §5.1 regime, §5.2 checkpoint, §9 provenance) render as
  properly-formatted `longtable`s with visible rules and correctly wrapped cell text —
  confirmed by rendering all 12 pages to PNG (150 DPI) and visually inspecting the
  4 table pages directly, not inferred from the absence of a compile error.
- All internal section cross-references (§4.1–§4.6, §5.1–§5.2, etc.) resolve — pandoc's
  `\hypertarget`/`\label` pairs are intact per section; no undefined-reference warnings
  in the tectonic log.
- No external bibliography exists in this paper (unlike AgentGauge) — confirmed by
  grepping the source for citation-like patterns; every reference is to this
  repository's own files/commits, rendered as inline `\texttt{}`, which is correct
  and requires no `\cite{}`/`.bib` machinery.
- **Defect counts, abstract vs. body — verified exact, not approximate:** "five" appears
  as the defect count in the title, abstract, §1, §3, §4's header note, §6, §8, and
  §10, with zero exceptions found; §4.6's three GCP bugs are labeled "not counted among
  the five" in the section header itself, in §3's added note, and in §8's added bullet;
  §7's sixth anomaly is labeled "candidate, not a finding" in its own section header and
  reinforced in §8. No sentence anywhere states six, seven, or eight as a defect count.

---

## 5. Manual arxiv.org steps — GG only, not automatable

**I have not visited any of these pages and cannot confirm current UI text, exact field
labels, or live policy details — every step below states what to check on the page
itself before proceeding, rather than assuming the described behavior is still current.**

1. **Log in.** Go to `https://arxiv.org/user/login` (or `https://arxiv.org/login` if
   that redirects) and sign in to the `gaurav.gandhi` account.
   *Confirms success:* the account dashboard loads showing prior submissions (if any)
   — this is also where you can directly check submission history per category, which
   settles the §1 endorsement question with certainty instead of my inference.

2. **Start a new submission.** From the logged-in dashboard, use the "Submit" /
   "Start New Submission" link (dashboard URL is typically
   `https://arxiv.org/submit` once logged in — confirm the exact link text/URL on the
   dashboard itself, since arXiv's UI wording changes between versions).
   *Confirms success:* a submission wizard opens at step 1 (license selection).

3. **License selection.** Choose a license (arXiv requires one per submission). No
   specific recommendation given here — this is a rights/distribution choice for GG,
   not a technical one; the default `arXiv.org perpetual, non-exclusive license` is the
   common choice if no other preference exists.
   *Confirms success:* wizard advances to file upload.

4. **Upload files.** Upload the full contents of `docs/paper/latex/`:
   `main.tex`, `abstract_body.tex`, `body_content.tex` (do **not** upload `main.pdf` —
   arXiv compiles the PDF itself from the `.tex` sources; uploading a pre-built PDF
   alongside `.tex` sources can confuse arXiv's own TeX Live-based compiler, which may
   not exactly match tectonic's output).
   *Confirms success:* arXiv's own compile step (their TeX Live, not tectonic) succeeds
   and produces a preview PDF. **Check this preview carefully** — arXiv's TeX Live
   engine is not tectonic; a paper that compiles clean under tectonic can still warn or
   render slightly differently under arXiv's own engine (different default font
   coverage for `≈`/`≥` in particular, since that was the exact defect already found
   and fixed once in this pass — re-verify it didn't regress under arXiv's compiler).

5. **Category selection — this is where §1's open question gets answered for real.**
   Select `cs.SE` as the primary category, `cs.LG` as a cross-list (per §1's
   recommendation). **If arXiv's form shows an endorsement-required notice at this
   step:** it will display an endorsement code and a link (arXiv's help page,
   `https://arxiv.org/help/endorsement`, documents the general mechanism) — do not
   proceed past this screen without reading exactly what it says, since the precise
   current wording/flow is what should drive the next action, not this document's
   prediction. If endorsement is required and no immediate endorser is available,
   fall back to §1's suggested swap (`cs.LG` primary, `cs.SE` cross-list) and restart
   this step.
   *Confirms success:* category selection is accepted and the wizard advances to
   title/abstract entry (or, if endorsement-blocked, the page explicitly says so with
   next steps — that is itself the "success" of this step, in the sense of a clear
   answer to §1's open question).

6. **Title field.** Paste one of §2's three title options (recommended: option 1) into
   the "Title" field.
   *Confirms success:* field accepts it (check the character count/limit shown on the
   page itself if arXiv displays one).

7. **Abstract field.** Paste §2's compressed abstract (1,610 characters) into the
   "Abstract" field.
   *Confirms success:* field accepts it without truncation — if arXiv's live limit is
   below 1,610 characters (unlikely based on common knowledge, but not verified live),
   the form will either reject or visibly truncate; if so, the fix is compressing
   further, not silently letting it truncate.

8. **Author field.** Enter "Gaurav Gandhi" with affiliation "Independent Researcher"
   (matching `main.tex`'s `\author{}` field exactly, so the PDF and the metadata agree).
   *Confirms success:* author name/affiliation display correctly in the submission
   preview.

9. **Comments / MSC / ACM class (optional fields).** Not required for `cs.SE`/`cs.LG`;
   safe to leave blank unless GG wants to add a comment (e.g., page count, "12 pages").

10. **Final preview and submit.** arXiv shows a final preview of the compiled PDF,
    metadata, and category before the actual submit action.
    **Do not click the final submit button as part of this checklist — that action is
    explicitly GG's to take, not mine, per this task's hard rule.** Review the preview
    PDF against the version already reviewed in this document (§4) before deciding to
    submit.

**After submission (if GG proceeds):** arXiv holds new submissions for a moderation
window before the arXiv ID is assigned and the paper goes live (typically the next
business day, per arXiv's own long-standing practice — verify current timing on the
submission confirmation page, not from this document). The eventual arXiv ID should
then replace the `TO FILL after upload` placeholder already sitting in the portfolio's
`agentgauge:paper-md` provenance entry's counterpart for this paper, if/when GG adds one
— out of scope for this task, noted only so it isn't forgotten.

---

## 6. Files produced by this pass

- `docs/paper/latex/main.tex`, `abstract_body.tex`, `body_content.tex`, `main.pdf`,
  `README.md` — submission-ready LaTeX source + compiled PDF + build notes.
- `docs/paper/arxiv_submission_prep.md` — this document.

Nothing was submitted, uploaded, or transmitted to arxiv.org by this pass.
