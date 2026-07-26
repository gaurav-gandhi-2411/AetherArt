# AetherArt — Development Plan

## Phases

- [x] **#7 Metadata sidecar** — Phase 1 complete
  - PNG tEXt chunks + sidecar JSON on every generation
  - "Recreate from PNG" UI feature

- [x] **#1 PartiPrompts eval harness** — Phase 2 complete (360 generations, charts committed)
  - 4 schedulers × 3 step counts × 30 prompts, seed = 42, RTX 3070 8 GB
  - DPM-Solver++ recommended default (0.3177 avg CLIP)

- [x] **#3 ControlNet (Canny + Depth)** — Phase 3 complete

- [x] **#2 LoRA Ukiyo-e fine-tune** — Phase 4 complete

- [x] **#8 README polish** — Phase 5 complete

- [x] **Phase 6a** — README rewrite and chart polish complete

- [x] **Phase 6b** — Controlled experiments: CLIP-blindness series complete (9 experiments)

- [x] **Phase 6c** — Central narrative and project documentation complete

---

**Phase 7 — SDXL Modernization (in progress)**

- [ ] PR 01 — Housekeeping: PLAN.md, CI concurrency, load_dotenv()
- [ ] PR 02 — Ruff migration + CI hardening
- [ ] PR 02a — torch 2.8 compatibility spike (non-merging)
- [ ] PR 02b — Docs-only: torch28_compat.md to main
- [ ] PR 03 — SDXL base pipeline
- [ ] PR 04 — Publish SD 2.1 Ukiyo-e adapter to HF Hub
- [ ] PR 05 — Hyper-SD 8-step LoRA integration
- [ ] PR 06 — NF4 quantization for SDXL + combined path smoke test
- [ ] PR 07 — Modal demo deployment + safety guard
- [ ] PR 08 — xinsir ControlNet Union SDXL
- [ ] PR 09 — SDXL LoRA retraining (GCP L4)
- [ ] PR 10 — SDXL Turbo legacy gate + license documentation
- [ ] PR 11 — Publish SDXL Ukiyo-e adapter to HF Hub
- [ ] PR 12 — HPSv2.1 + ImageReward eval integration
- [ ] PR 13 — Phase 6b experiments migration to SDXL
- [ ] PR 13b — HPSv3 GCP batch eval
- [x] PR 14 — CI quality gate: frozen SD 2.1 30-prompt/seed-42 CLIP regression check
      (`.github/workflows/eval.yml`, push-to-main + nightly + manual dispatch — not
      per-PR, see that file's header comment for the measured-cost rationale) +
      real (unmocked) generation smoke test on every push/PR
      (`.github/workflows/ci.yml`, `tests/test_generation_smoke.py`) + HF revision
      pinning across config/utils/model/clip_scorer. Coverage threshold
      (`--cov-fail-under=55`) was already enforced via `pyproject.toml` before this.
      HPS/ImageReward deliberately left out of the gate pending the
      Windows/headless-Linux crash fixes tracked in `docs/lab_notebook.md:246-250,269`.
      Landed as 3 separate PRs (#22 CI gate, #23 revision pinning, #24 smoke test)
      after the original combined PR #20 (508+/-8 diff) tripped this repo's
      merge-gate size hook — each split PR carries its own verifier review artifact.

---

**Phase 8 — Cross-family model verdict (in progress, checkpoint 2026-07-26a)**

- [x] **FIFTH measurement-defect class found: inconsistent reference arms across two analyses of
      the same data (2026-07-26a).** `sdxl_base`'s Pattachitra `style_adherence` mean was reported
      as two different, contradictory numbers in two places: `0.3533` in the judge positive
      control's "corrected prompt" row, `0.8883` in the weight-sweep stats — same model, axis,
      n=90, supposedly the same corrected question. Root-caused by direct code inspection, not
      inference from the repeated figure alone: `scripts/judge_style_positive_control.py` loaded
      the base arm's scores ONCE (`load_base_scores`, from the OLD pre-fix
      `pattachitra_ab_base_comparison.json`, scored under the hardcoded wrong ukiyo-e question)
      and reused that same value, unchanged, for BOTH the historical-prompt row (where reuse is
      correct — both arms consistently wrong-question) AND the corrected-prompt row (where reuse
      is a bug — the base arm was never actually re-scored under the corrected question there).
      `scripts/rescore_pattachitra_uniform.py`, by contrast, genuinely re-scores the base arm
      fresh under the corrected question — confirmed by reading its `build_source()`/`main()`
      directly — making its `0.8883` the correct value; the weight-sweep's 8 operating-point rows
      did not need recomputation. **Why this defect class is distinct from bug #4 (hardcoded
      question, `docs/MODEL_VERDICT.md` §7.7):** bug #4 is one script asking the wrong question of
      every caller; this is a *different* script correctly asking the right question of one arm
      but then silently reusing a *different, older* arm's data for a second, supposedly-
      independent comparison. Domain-parameterization tests (checking `style_question` is never
      defaulted) do not catch this — the question passed to the judge was never wrong in this
      script; the *data reused* was stale. Fixed: `scripts/judge_style_positive_control.py` now
      scores the base arm fresh, per comparison, no cross-run reuse. See
      `docs/WEIGHT_SWEEP_PREREGISTRATION.md`'s amendment for the full account and the
      accompanying ukiyo-e control redesign (the original ukiyo-e row was a ceiling-effect
      confound — real art vs. ukiyo-e-*prompted* `sdxl_base` — not a fair discrimination test;
      replaced with two off-style contrasts, same question throughout).

- [x] **TOP FINDING: the judge-question positive control FAILED for ukiyo-e — every
      `style_adherence` number in §4 and the staged HF card text is now PROVISIONAL
      (2026-07-25h).** `scripts/judge_style_positive_control.py` run after the weight sweep
      completed (540/540, 0 errors) and the GPU freed. Ukiyo-e was the expected-pass case meant
      to validate the control: real curated training art (n=23) vs. `sdxl_base`'s own generated
      attempts (n=90) should score higher on `style_adherence` if the judge can tell real from
      generated. **It did not** — diff=−0.0150, SEM=0.0099, diff/SEM=**−1.507** (wrong-signed,
      not just non-significant). Per the pre-committed escalation rule: HALTED the ukiyo-e
      publication path, marked every `style_adherence` figure in `docs/MODEL_VERDICT.md` §4
      (new §4.10) and `docs/HF_MODEL_CARD_UPDATES.md` PROVISIONAL — do not publish even once a
      write token exists. `artifact_absence` (the actual regression finding driving the "not
      promoted" verdict) is unaffected — domain-neutral question, confirmed unrelated to this
      control. **Not general judge blindness** — Pattachitra's own positive control, run in the
      same pass, passed dramatically (diff=+0.4437, diff/SEM=**+12.005** on the corrected
      question vs. −1.504 on the historical wrong one) — confirming the judge can discriminate
      style sharply when the question is right and the domain isn't already well-represented in
      SDXL's own pretraining (a plausible, unverified explanation for ukiyo-e's ceiling effect,
      not investigated further per the diagnostic scope cap). Continuing to Q3-Q5 for the
      Pattachitra diagnostic per the pre-committed rule.

- [x] **Withdrew the void-based sweep prediction; eliminated 100% of the sweep's wasted VLM
      scoring; capped the remaining work as diagnostic (2026-07-25g).** (1) The sweep's
      predicted outcome in `docs/WEIGHT_SWEEP_PREREGISTRATION.md` rested entirely on two
      `weight=1.0` `style_adherence` values now confirmed VOID by the judge-prompt bug —
      **withdrawn explicitly**, not left standing. Chose withdrawal over re-deriving now: the
      re-derivation is possible independent of the running sweep, but would need new Ollama
      calls while the GPU was occupied and would duplicate the uniform re-score already planned
      for once it's free. The structural argument (weight=0 trivially equals 0, so an interior
      peak above +2×SEM is inherently non-monotonic) survives; the specific numbers and
      conclusion do not. (2) Confirmed generation/scoring are cleanly separable in
      `scripts/_pattachitra_weight_sweep.py` and **disabled inline scoring entirely** — every
      score the sweep would produce was going to be discarded by the uniform re-score plan
      regardless. Stopped the already-running `curated500` process (its old in-memory code
      wouldn't have picked up the fix) and relaunched: verified the resume logic picked up
      exactly where its 100 already-generated, disk-persisted images left off — no regeneration,
      no duplicates, no generation work lost, only the now-pointless scoring skipped. This also
      removes the sweep's Ollama/SDXL VRAM-contention step entirely (no `ollama stop` needed
      between checkpoints anymore — the sweep never calls Ollama now). (3) Recorded in
      `docs/MODEL_VERDICT.md` §7.2 that **the publication decision is already settled** by
      `figure_preservation` (−5.5 to −7.8×SEM at `weight=1.0`, domain-neutral, unaffected by the
      judge bug) — no sweep or positive-control result can publish this adapter. Remaining work
      capped at exactly two diagnostic questions (does a low-weight operating point exist; can
      the judge perceive Pattachitra style at all) — explicitly not a license for further
      retrains or corpus work. §7's full finalization (including a standalone four-bug
      methodology finding: `num_ctx`, phantom VRAM counter, CUDA context corruption, hardcoded
      judge question) remains gated on the GPU freeing, the positive control, and the uniform
      re-score.

- [x] **Fixed the judge-prompt bug properly; scoped its blast radius per-axis; triaged the
      running sweep (2026-07-25f). NEW BUG CLASS for this project — semantically wrong but
      syntactically valid, invisible to every integrity self-check built so far.** (1)
      **Triage:** confirmed by reading `scripts/_pattachitra_weight_sweep.py` directly that
      generated images are persisted to disk (`img.save()`) in a phase entirely separate from,
      and prior to, VLM scoring — the bug is in scoring, not generation. Decision: **let the
      sweep run to completion, do not kill it.** All `style_adherence` output from this run is
      void; plan is to re-score from the saved images with the corrected prompt, no
      regeneration. (2) **Fixed properly:** `model_verdict_harness.py`'s `score_vlm_judge` now
      requires an explicit `style_question` keyword-only argument with no default — the harness
      itself carries zero style-domain knowledge. Every caller updated:
      `run_ukiyo_e_lora_family` and two historical ukiyo-e-only scripts
      (`_lora_ab_30prompt_independent.py`, `_lora_ab_base_comparison.py`) now pass
      `UKIYO_E_STYLE_QUESTION`; `_pattachitra_ab_base_comparison.py` and
      `_pattachitra_weight_sweep.py` now pass `PATTACHITRA_STYLE_QUESTION` (grounded in
      `docs/NEXT_MODEL_SPEC.md`'s own prior corpus description, not invented). **Audited the
      other two axes for the same defect class, as required:** `figure_preservation` and
      `artifact_absence`'s fixed templates were checked directly and confirmed domain-neutral
      (no style name in either) — both survive. A fourth, older, already-superseded script
      (`_lora_ab_30prompt.py`, the pre-independent-axis single-call design withdrawn in §4)
      hardcodes "Ukiyo-e" in its own separate, non-shared prompt — correct for what it was, since
      it was never reused for a different domain; left as-is, noted for the record, not fixed
      (dead code, not a live bug). 9 new regression tests
      (`TestStyleQuestionIsNeverHardcoded`) assert: no default exists (TypeError if omitted); the
      actual Ollama request contains the caller's stated domain; two domains produce two
      different requests; no leftover style name appears for an unrelated domain; and the other
      two axes stay domain-neutral. 27/27 harness tests pass. (3) **Blast radius scoped
      per-axis in `docs/MODEL_VERDICT.md` §7.2 addendum, marked VOID not deleted:**
      `style_adherence` — VOID for all 360 Pattachitra records (wrong question: asked about
      ukiyo-e); ukiyo-e's own `style_adherence` numbers (§4, the HF card) are UNAFFECTED (that
      was always the correct question for that domain). `figure_preservation` and
      `artifact_absence` survive for both domains (domain-neutral templates). Every downstream
      claim built on the voided axis (§7.3's table rows, §7.4's verdict text, §7.6's portfolio
      pattern) is marked VOID inline (strikethrough + note), not deleted — the fact that it was
      measured wrong is itself part of the record. The `figure_preservation` guardrail finding —
      the actual, stated reason the adapter isn't published — is unaffected and stands alone.
      **Side effect of fixing mid-sweep:** `curated500`'s already-running process holds the old
      buggy code in memory and will still score `style_adherence` wrong when it reaches that
      phase (its images are unaffected); `curated1000`, not yet launched, will pick up the fix
      automatically as a fresh process. Both will be re-scored uniformly from saved images
      regardless, so the final numbers come from one consistent method. §7.4's verdict and the
      retrain proposal remain gated on the (not yet run) judge positive control and re-score.

- [x] **Amended the sweep pre-registration; discovered the judge's `style_adherence` question is
      hardcoded to ukiyo-e (2026-07-25e) — doc/script only, GPU untouched; sweep still running.**
      Two pre-registered amendments to `docs/WEIGHT_SWEEP_PREREGISTRATION.md`, committed before
      reading any sweep output: (1) corrected a premise before predicting on it — the −2.234×SEM
      figure this task was drafted with belongs to checkpoint-1500, which isn't swept; the two
      swept checkpoints' actual `weight=1.0` starting points are +0.477 (500) and −0.789 (1000),
      neither near +2×SEM, so an interior weight clearing that bar would require a non-monotonic
      peak above both boundary values — predicted "no viable operating point," with any found
      operating point requiring an explanation, not just a report. (2) Clarified the
      `figure_preservation >= −2×SEM` rule is a "failed to detect regression" screen, not
      demonstrated non-inferiority, and corrected a floated MDE figure (~0.037 — actually a
      different endpoint's MDE, ukiyo-e's §4.7 arm-to-arm comparison) to the real 0.0115–0.0185
      range for the swept checkpoints; every non-inferiority claim must now print its MDE
      alongside (`compute_pattachitra_weight_sweep_stats.py`'s joint-curve table extended with fp
      MDE columns). **While writing (not running) `scripts/judge_style_positive_control.py`,
      found via direct code inspection that `model_verdict_harness.py`'s `style_adherence` judge
      question is hardcoded to ask about ukiyo-e regardless of caller — every Pattachitra
      `style_adherence` score ever recorded (all 360 records, §7's primary endpoint A) was
      generated by asking the judge if the image looks like ukiyo-e, not Pattachitra.** The
      control tests both that literal production prompt and a corrected, domain-appropriate one
      (grounded in `docs/NEXT_MODEL_SPEC.md`'s own prior corpus description) to determine whether
      the existing numbers are salvageable by re-scoring or the judge is blind to the domain
      either way. Not run yet (needs Ollama/GPU free); data-loading logic verified without any
      GPU or Ollama call (100/23/90/90 record counts confirmed). §7's verdict and the retrain
      proposal remain gated on both this control and the sweep, per instructions.

- [x] **Pre-registered the weight-sweep decision rule; tightened the epoch claim; fixed a
      flaky latency test (2026-07-25d) — doc/analysis only, GPU untouched (a background
      adapter-weight sweep was running throughout; only read-only `mem_get_info`/process-alive
      checks were used, never its result data before this commit).** (1) **Pre-registration**
      (`docs/WEIGHT_SWEEP_PREREGISTRATION.md`): guards the tautology that adapter weight → 0
      trivially recovers `figure_preservation` by converging to `sdxl_base` while the style lift
      also vanishes, so recovery on that axis alone is not evidence of a usable adapter. Requires
      a joint criterion at the same weight — `style_adherence` diff/SEM > +2.0 (reusing the
      original Pattachitra pre-registration's own threshold) AND `figure_preservation` diff/SEM
      ≥ −2.0 (the existing §7 guardrail, not a new lenient margin) — with "no viable operating
      point" pre-committed as the conclusion if no weight clears both.
      `scripts/compute_pattachitra_weight_sweep_stats.py` was corrected to implement this exact
      rule (previously checked `style_adherence diff > 0`, a materially weaker bar) before being
      run against complete data. (2) **Epoch claim tightened**
      (`docs/MODEL_VERDICT.md` §7.2(5)): checkpoint-500's 20 effective epochs sits *inside* the
      typical 10–50-epoch convergence band yet already regresses `figure_preservation`
      (−5.532×SEM) — overtraining plausibly explains the *monotonic worsening* 500→1500 but does
      NOT explain the baseline regression already present at an in-band epoch count. Ukiyo-e's
      ≈174 epochs remains the stronger overtraining candidate of the two; the two adapters'
      evidence is asymmetric, not a single shared verdict. (3) **Flaky-test fix**
      (`test_combined_path_nf4_hyper_within_budget`, failed under GPU contention in three
      separate sessions): added `gpu_is_quiet()` (`aetherart/gpu_hygiene.py`, 5 new mocked unit
      tests) — a `torch.cuda.mem_get_info()`-based contention check with a 500 MB threshold — and
      the test now skips rather than asserts a wall-clock budget when the GPU isn't uncontended.
      Verified live against the actually-running sweep: the test correctly skipped. Chose
      skip-on-contention over recalibrating the budget from a measured distribution because
      gathering that distribution would itself require dedicated GPU time, which this task
      explicitly ruled out. §7's verdict framing and the dense-checkpoint-retrain proposal remain
      gated on the sweep, as instructed — not touched this checkpoint.

- [x] **CUDA-corruption audit on the Pattachitra negative (2026-07-25c) — finding survives; one
      prior claim corrected.** A skeptical re-audit specifically asked whether the CUDA "illegal
      memory access" crash on the checkpoint-500 attempt (below) could have silently degraded the
      22 images generated *before* the crash, not just the 68 failed attempts after it. Checked
      directly, not assumed: file-mtime provenance identified the exact 22 at-risk images; a
      pixel-level integrity audit (mean/std/NaN/pure-black/pure-white per image) found them
      statistically indistinguishable from confirmed-clean images (zero NaN or black/white regions
      in either group), and direct visual inspection of the boundary images — including the very
      last image generated before the crash — showed clean, coherent output. `base`, `curated1000`,
      `curated1500` were each independently confirmed to come from a single, zero-CUDA-error
      process (log grep, one pipeline load each). **No regeneration was warranted.** Deduplication
      re-verified directly: all four checkpoint sets (base/500/1000/1500) contain exactly 90 unique
      `(prompt_id, seed)` records, zero duplicates. **One correction:** the prior write-up's claim
      that "checkpoint-1000 regresses worse than 1500, ruling out simple overtraining" used the
      diff/SEM *ratio*; the **raw** diff is actually monotonic in steps (−0.0228 → −0.0444 →
      −0.0461) — 1500's higher ratio-denominator variance (not a smaller effect) is what inverted
      the ranking. Corrected in `docs/MODEL_VERDICT.md` §7.2. **Added a permanent harness
      self-check** (`scripts/model_verdict_harness.py`): a CUDA pre-flight health probe, a
      per-record uniqueness assertion, degenerate-image detection (NaN/black/white/near-uniform)
      that aborts the run rather than writing a poisoned record, and a judge-response range
      validator (a hallucinated out-of-range score like 1.5 previously passed through uncaught) —
      16 new tests, wired into all three `model_verdict_harness.py` family runners plus
      `scripts/_pattachitra_ab_base_comparison.py`. Full suite: 254 passed, 8 skipped. No new
      training or GCP work this session, as instructed.

- [x] **Validated the Pattachitra negative before writing it up (2026-07-25b) — survives, with
      the full picture more nuanced than the single-checkpoint version.** A −7.226×SEM guardrail
      regression with total figure dropout was more consistent with a mechanical eval bug than a
      genuine training effect, so three explanations were ruled out before accepting it:
      (1) **checkpoint selection** — the original pass only scored the final weights (=1500),
      skipping the checkpoint-select step ukiyo-e's own precedent required (which explicitly
      rejected its own 1500 for the identical failure mode). Scored 500/1000/1500 properly:
      **all three checkpoints regress `figure_preservation` significantly** (−5.5 to −7.8×SEM) —
      checkpoint-1000 is actually the *worst* (−7.768×SEM), ruling out "1500 was simply
      overtrained." (2) **LoRA applied weight** — verified alpha=rank=8, adapter weight 1.0,
      byte-identical to the ukiyo-e recipe; cleared, not the cause. (3) **Trigger token** —
      checked the actual tokenizer output: `pattascroll` decomposes into 4 BPE fragments, not a
      single real vocabulary token (structurally similar to `ukyowood`'s 3); one fragment (`as`,
      a very common function word) is a minor, unproven risk factor for a future retrain, not a
      demonstrated cause. **None of the three explains it — the finding survives and was written
      up in `docs/MODEL_VERDICT.md` §7.2–§7.4 with the full per-checkpoint table.** Also: the
      `artifact_absence` ceiling (1.0 across all 360 scores) is now reported as a positive,
      cross-domain finding (§7.5) — it retroactively validates that the ukiyo-e curation project's
      text-artifact scoping was style-specific, not a generic assumption. Portfolio-level note
      (§7.6): the data does NOT cleanly support "LoRA helps where base is weak" (Pattachitra is a
      counter-example — weak base, LoRA still didn't help and broke a guardrail); what both
      projects DO support is pre-testing `sdxl_base`'s zero-training rendering before any spend,
      exactly how Mughal/Warli were disqualified. Two script bugs found and fixed mid-audit:
      a CUDA context corruption (illegal memory access) that silently poisoned 68/90 records on
      one checkpoint-500 attempt, and a retry-logic bug that appended duplicate records instead of
      replacing errored ones — both caught via direct verification (record counts, error grep),
      not assumed clean. No new training this session, as instructed.

- [x] **Pattachitra LoRA: amended pre-registration, trained on GCP, evaluated — NOT published
      (2026-07-25).** Amended `docs/PATTACHITRA_AB_PREREGISTRATION.md` before any training: dropped
      the curated-vs-uncurated arm as a gating endpoint (ukiyo-e's own data showed that comparison
      needs ≈1,963 samples to resolve, §4.7) and made curated-vs-`sdxl_base` (the comparisons that
      *did* resolve cleanly for ukiyo-e, 1.68–7.23×SEM range) primary instead — trains one adapter,
      not two, halving planned spend. **Manual QA before training:** spot-checked the 111
      automated-"clean" Pattachitra images and found 11 were documentary/vendor photos a person's
      face/torso dominated, or a genre mismatch — excluded, leaving 100 curated training images.
      **Trained on GCP** (`g2-standard-8`/L4, `us-west1-a`, rank-8, 1500 steps) — 4 dependency
      failures before a clean run, each root-caused and fixed in turn: (1) `pip install --upgrade
      transformers` broke the DLVM's torchaudio pairing; (2) the break persisted without
      `--upgrade` (pre-existing in the DLVM's own pre-installed transformers); (3) pinning
      `transformers==4.46.0` to dodge it instead broke diffusers' need for a newer class; (4) the
      actual fix — reinstalling torchaudio matched to the exact installed torch build — worked,
      but the run then crashed via `bufio.Scanner: token too long` when tqdm's carriage-return
      output exceeded GCE's serial-console line-scanner limit, silently killing training via broken
      pipe while the instance kept running at 0% GPU utilization until caught via SSH and stopped
      manually (~20 min of real but bounded, disclosed cost waste). Fixed by isolating the training
      command's output from the console-feeding stream entirely. Fifth attempt succeeded: 1500/1500
      steps, `scripts/gcp_verify_before_teardown.py` passed before manual instance deletion.
      **Actual cost ≈$1.90–2.20** (~2.2 GPU-hours total across all attempts) against a $5–7.50
      estimate and $10 hard stop. **Evaluated against `sdxl_base` (n=90 paired) — the adapter does
      NOT beat base:** `style_adherence` regresses significantly (−2.234×SEM, the wrong direction
      for a "lift"); `figure_preservation` regresses decisively (−7.226×SEM) — visually confirmed
      on `pat_009` ("farmer plowing with oxen," same seed both arms): `sdxl_base` renders the
      farmer correctly, the curated LoRA drops the human figure entirely, animals only.
      `artifact_absence` is an uninformative ceiling effect (exactly 1.0 in both arms, 180/180
      independent-regime scores — Pattachitra doesn't evoke SDXL's text-artifact tendency the way
      "ukiyo-e" did). **Verdict: not published, reported as a finding — not retrained to chase a
      better number** (`docs/MODEL_VERDICT.md` §7). Two concrete, testable hypotheses for a future
      attempt if revisited: BLIP's generic auto-captions may be too low-information for a 100-image
      rank-8 LoRA to learn robust subject-preserving associations; the corpus may still be too
      compositionally narrow for this evaluation's more demanding 30-prompt set.

- [x] **High-power metric attempt, model-card tradeoff disclosure, Pattachitra pre-registration
      (2026-07-24c).** Built and independently validated (n=29 visual-inspection sample) an
      EasyOCR-based binary artifact detector (`scripts/detect_text_artifacts.py`) to test whether
      a binary proportion test could out-power the VLM rubric's ~1,963-sample requirement
      (`docs/MODEL_VERDICT.md` §4.7). **Result: it doesn't — the opposite of the premise.** A naive
      confidence threshold is unusable (23% recall; generated pseudo-calligraphy is rarely
      OCR-legible); a revised "any detection" rule is moderately validated (94% precision, 77%
      recall) but the resulting paired-proportion MDE at n=90 is ~0.22 — nearly 6× worse than the
      rubric's ~0.037 — because 55.6% of published-vs-curated pairs are discordant (high
      image-to-image noise in OCR legibility). Reported as a genuine negative result (`docs/MODEL_VERDICT.md`
      §4.9), not reframed to look positive. The VLM rubric remains primary; OCR is a secondary,
      non-gating signal. **Model card updated** (`docs/HF_MODEL_CARD_UPDATES.md`) with the plain,
      unsoftened cost/benefit numbers: `sdxl_base` scores `artifact_absence` 0.9222 (cleaner than
      either adapter); both adapters significantly regress it (published −0.0500/3.49×SEM, curated
      −0.0422/3.11×SEM); the adapter's value is a modest style lift (curated 2.82×SEM, published
      1.68×SEM over base). **Pattachitra pre-registered** (`docs/PATTACHITRA_AB_PREREGISTRATION.md`)
      before any training — bakes in this session's lessons from the start (independent-axis
      scoring only, base-model comparison as part of the primary design not an afterthought,
      power/MDE reporting required regardless of outcome, OCR as non-gating secondary). **No
      training authorized** — still needs GG's approval + GCP spend per `docs/NEXT_MODEL_SPEC.md` §5.

- [x] **Follow-up audit (2026-07-24b) on the withdrawal below: validated, root-caused, refined.**
      Four checks, per an explicit "don't accept a null from a possibly-broken measurement"
      instruction: (1) **provenance** — directly verified (not assumed) the n=90 rescore ran
      entirely on the num_ctx-fixed harness with zero degenerate/failed judge calls
      (`docs/MODEL_VERDICT.md` §4.6's provenance table); (2) **root-caused the null** rather than
      accepting "curation doesn't help" — LPIPS between the two arms (0.55) refutes "the arms are
      identical"; a new `sdxl_base`-only (no LoRA) batch shows curation did NOT lose style signal
      (curated's style lift vs. base is *more* significant than published's) and that **both LoRA
      variants significantly regress `artifact_absence` vs. no adapter at all** — confirming
      training does cause the artifact — with curation recovering ~16% of that regression in the
      right direction, just not enough to clear the arm-to-arm bar at this n (§4.8); (3) **power
      audit** — the n=90 design's 95% CI rules out the original +0.040 claim but its MDE (~0.037)
      is close to the observed effect, so this is an underpowered result, not a demonstrated null;
      language throughout corrected to "does not clear the bar," not "no effect" (§4.7);
      (4) **reconciled** the Pattachitra 139-vs-136 file-count gap exactly — a case-insensitive
      Windows/NTFS filename collision (distinct from the earlier race-condition bug), corrected
      licence breakdown, flag rate/PROCEED decision unaffected (`docs/NEXT_MODEL_SPEC.md` §3.5).
      **Net effect: the curation recipe is directionally validated, not disproven** — Pattachitra
      training remains on hold until the judge/n power gap is addressed, not because the recipe
      failed. New provenance scripts: `scripts/compute_lora_ab_power.py`,
      `scripts/_lpips_between_arms.py`, `scripts/_lora_ab_base_comparison.py`.

- [x] **Critical correction (2026-07-24): the LoRA A/B "PROMOTE" verdict below was withdrawn.**
      The +0.040/3.18×SEM headline was computed under single-call multi-axis VLM scoring — the
      halo-effect check (`docs/MODEL_VERDICT.md` §4.5) flagged this as a correlation risk but
      only checked a small (n=30, unpaired) subsample and concluded the primary endpoint "held."
      Extending the independent-axis rescore to the full n=90 **paired** set
      (`scripts/_lora_ab_30prompt_independent.py`, `reports/lora_ab_30prompt_independent.json`,
      stats via `scripts/compute_lora_ab_independent_stats.py`) found the primary endpoint at
      only **+0.0078, 0.583×SEM** — the correlated-regime headline does not survive trusted
      scoring. **Do not re-upload the curated checkpoint to HF.** Full writeup:
      `docs/MODEL_VERDICT.md` §4.6; `docs/HF_MODEL_CARD_UPDATES.md` corrected to drop the
      withdrawn claim. Independent single-axis scoring is now `model_verdict_harness.py`'s
      default (was single-call), with a regression test (`tests/test_model_verdict_harness.py`)
      asserting one Ollama call per axis.
  - **Two bugs found and fixed while doing this rescore, both disclosed:**
    1. **Ollama context-window bug:** the default served context (4096 tokens) was too small for
       a judge prompt + high-resolution image on some inputs, causing silent
       `exceed_context_size_error` 400s. Root-caused via the actual Ollama error body (not
       assumed to be GPU contention, even though GPU contention was a live confound at the time).
       Fixed via `num_ctx=8192`. Verified this did not silently corrupt the original §4.3
       single-call dataset (zero null `vlm_judge` records there).
    2. **Concurrent-instance race condition:** two instances of the (gitignored)
       `scripts/_curate_pattachitra_corpus.py --resume` ended up running at the same time across
       a context-compaction boundary, both doing unlocked read-modify-write on the same
       classifications JSON — net effect was a few classifications reverting to
       `vlm_call_failed` (not permanent data loss, since failed entries retry on the next
       `--resume`, but wasted work). Root cause: launched a second resume without first
       confirming the first instance had actually stopped. Going forward: always check for a
       live process on the target script before relaunching a resume, not just check the
       task-notification status (which reflects the bash-wrapper's tracking, not the
       nohup'd child's actual liveness).
  - **GPU-contention observation (not a bug, a real constraint):** running the SDXL generation
    job and an Ollama VLM-judge job concurrently on the shared 8GB card caused genuine failures
    (Ollama calls timing out at their 120s HTTP timeout, SDXL steps stalling to 40+s/it), not
    just the expected slowdown — worse than the "cost is time only" assumption covers. Paused
    the SDXL job to let the shorter Ollama-only job finish first, then resumed SDXL alone. A
    separate, unrelated concurrent job on the same GPU (a different model, `qwen3:8b`, not part
    of this project) was also observed contending at another point — see
    [[project_gpu_contention]].

- [x] `docs/MODEL_AUDIT.md`, `docs/LATENCY_ROOT_CAUSE.md` — prior-session groundwork (repo runs
      5 model families, not 2; phantom 11GB-VRAM report row traced to GCP L4 contention).
- [x] All 5 non-LoRA families scored on identical 30-prompt×3-seed set, 0 errors —
      `reports/verdict_{sd21_base,sdxl_base,hyper_4step,hyper_8step,sdxl_controlnet_union}.json`.
- [x] Hyper-SD 4-step vs. 8-step HPS inversion root-caused (`docs/MODEL_VERDICT.md` §3):
      config-routing confirmed correct (falsified plumbing-bug hypothesis); real cause is
      CFG-driven exposure collapse on the 8-step CFG-preserving variant (16% severe
      underexposure, 7% black-crush at guidance_scale=5.0 — already the least-bad point in
      Hyper-SD's own documented 5-8 range; no in-spec fix exists). Verdict: route to
      `hyper_4step`, not `hyper_8step`.
  - [x] Bug found + fixed while building the LoRA A/B harness: `run_ukiyo_e_lora_family`
        (`scripts/model_verdict_harness.py`) called the Ollama VLM judge inline while the
        SDXL+LoRA pipeline was still GPU-resident — same VRAM-oversubscription pathology as
        `docs/LATENCY_ROOT_CAUSE.md`. Fixed to a two-phase pattern (defer VLM scoring until
        after the pipeline is released), matching the existing HPS-deferral pattern.
  - [x] LoRA A/B — general 30-prompt benchmark: wash on all 4 axes (paired diff, all < 1 SEM),
        reframed as an off-domain guardrail (no ukiyo-e trigger prompts → floor effect on
        artifact_absence, not evidence of no effect — `docs/MODEL_VERDICT.md` §4.1/§4.2).
  - [x] LoRA A/B — **powered primary result (pre-registered, `docs/AB_PREREGISTRATION.md`)**:
        30 ukiyo-e-styled prompts × 3 seeds, n=90 paired, 0 errors
        (`reports/lora_ab_30prompt.json`). Primary endpoint `artifact_absence` clears the
        2×SEM promotion bar decisively (+0.040, 3.18×SEM); both guardrails (`style_adherence`
        +0.014 at 4.39×SEM, `figure_preservation` +0.012 at 2.68×SEM) show improvement, not
        regression. **Curated retrain PROMOTED** over the published checkpoint — action:
        re-upload to `gauravgandhi2411/aetherart-ukiyo-sdxl` on HF Hub (blocked on the same
        read-only-token issue as the model-card PRs, see below).
        Run relocated to GCP after local GPU contention (see gotchas below).
- [x] Verifier pass on `docs/MODEL_VERDICT.md` (2 rounds) — round 1 flagged inconsistent
      diff-SEM methodology (paired vs. independent-quadrature) between §4.1/§4.2, fixed by
      making paired-diff explicit and uniform (changed §4.2's read from 1.80→2.03 SEM). Round 2
      independently re-verified the final §4.3 powered-A/B numbers against raw JSON. All
      sections verified exact, zero unresolved discrepancies.
- **Gotchas from the GCP relocation (worth remembering for next time):**
  - A `g2-standard-4` (16GB RAM) VM OOM-killed the generation process twice — SDXL's CPU-offload
    staging needs more system RAM than that machine type provides even though GPU VRAM (23GB L4)
    is ample. Fix: use `g2-standard-8` (32GB RAM) or disable CPU-offload entirely on GPUs with
    enough VRAM to not need it.
  - `us-central1-a/b/c` and `us-east4-a` were all L4-stocked-out in this session — matches this
    project's previously-documented stockout pattern. `us-west1-a` had capacity.
  - **Real mistake, disclosed not hidden:** deleted a VM to free capacity for a retry before
    pulling its completed results (90 generations + 90 VLM scores) to local —
    `gcloud compute instances delete` removes the boot disk by default, so that data was lost
    and had to be regenerated from scratch.
  - **Hard rule going forward (codified as a guardrail, not just a habit):** never run
    `gcloud compute instances delete` (or `stop`/`set-machine-type`, which also risk losing an
    in-progress result if something goes wrong mid-resize) until every result file has been
    `gcloud compute scp`'d to local AND positively verified — not just "the scp command exited
    0," but a real content check (file exists, non-trivial size, and for JSON record files, the
    expected record count with zero errors). `scripts/gcp_verify_before_teardown.py` codifies
    this check — run it after every scp, before every teardown:
    `python scripts/gcp_verify_before_teardown.py <local_path> --min-records N --no-errors`.
    An exit-code check on the scp alone would not have caught the original incident — the scp
    itself always "succeeded"; what was missing was verifying the copy's *content* before the
    point of no return.
  - **Separate incident, same session's later work:** a single "research N Indian-art
    candidates" subagent dispatch fanned out into its own parallel subagent tree and hit a
    session usage limit mid-task — several candidates' research was lost (had to be redone
    directly, not via another delegated dispatch) and the failure wasn't visible until the
    notification arrived. **Fixed globally, not just here:** `C:\Users\gaura\.claude\CLAUDE.md`
    rule 70c now caps subagent dispatch depth at one level — the orchestrator dispatches
    executors/verifiers/research agents directly; a dispatched agent does not itself dispatch
    further subagents. If a task's scope wants its own fan-out, that fan-out happens at the
    orchestrator level (dispatch the N parallel agents directly), not delegated to one agent
    to decide for itself.
- **Autonomy setup (partially blocked, not silently worked around):**
  - Branch protection on `main` already permits merge-without-human-approval
    (`required_pull_request_reviews.required_approving_review_count: 0`, confirmed via
    `gh api repos/.../branches/main/protection`) — no GitHub-side change was needed.
  - The local mechanical merge-gate hook (`hook_guard_merge.py`, enforcing rule 70a) blocked
    PR #20 (508+/-8 diff, over the ~400-line reviewable-diff gate). Per explicit user decision
    this session: **left PR #20 and (transitively, since it's stacked) PR #21 as drafts for
    manual merge** rather than splitting or overriding the gate.
  - Both HF model-card draft PRs (`refs/pr/1` on `aetherart-ukiyo-sdxl` and `aetherart-ukiyo-sd21`)
    are still open — blocked on a **read-only** HF token (`api.whoami()` confirms
    `role: read`); merging them needs a write-scoped token or manual action on huggingface.co.
