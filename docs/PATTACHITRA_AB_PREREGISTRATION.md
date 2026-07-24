# Pre-Registration: Pattachitra LoRA A/B (Curated vs. Uncurated Training Set)

**Committed BEFORE any training run executes — no GCP spend or training job is authorized by
this document alone.** This fixes the endpoint, threshold, decision rule, and power
characterization in advance so the promotion call cannot be adjusted after seeing results. Do
not edit this file to match a result once training/generation/scoring has started.

## Background — carrying forward this session's lessons, not repeating them

The Ukiyo-e LoRA A/B (`docs/MODEL_VERDICT.md` §4) went through: a correlated single-call VLM
judge producing an inflated headline (§4.3), a withdrawal after independent-axis rescoring (§4.6),
a root-cause audit that found the withdrawal itself needed nuance (§4.7–§4.8), and an attempt to
build a higher-power binary metric that turned out to be *lower*-power than the rubric (§4.9).
This pre-registration bakes those lessons in from the start instead of needing a retroactive
correction:

1. **Independent-axis VLM scoring is the only scoring regime used** — `scripts/model_verdict_harness.py`'s
   `score_vlm_judge` (one Ollama call per axis) is the default; there is no single-call variant to
   accidentally reach for.
2. **The base-model (no-adapter) comparison is part of the primary design, not an afterthought.**
   §4.8's root-cause diagnostics only happened after a null result demanded them. Here, the
   `sdxl_base`-only arm is generated and scored in the same pass as the two LoRA arms.
3. **Power is reported alongside every result, regardless of outcome.** A result that doesn't
   clear the promotion threshold must be reported with its MDE and 95% CI (per §4.7's template) —
   "did not clear the bar" and "no effect" are different claims, and both possibilities are
   pre-committed to being distinguished, not just the passing case.
4. **The OCR binary detector (`scripts/detect_text_artifacts.py`, `n_detections >= 1` rule) is run
   as a secondary, corroborating signal only** — §4.9 found it is *less* powered than the rubric
   for this domain, so it does not gate the promotion decision, but its result is reported
   alongside for directional consistency checking.

## Design

- **Arms:** three, generated and scored together:
  1. `sdxl_base` — no LoRA adapter at all (the base-prior reference point, §4.8-style).
  2. Pattachitra LoRA trained on the **uncurated** corpus (all 136 physically-available images
     from `data/lora/pattachitra-precheck/images/`, before the VLM curation filter's 25 flagged
     images are removed — `docs/NEXT_MODEL_SPEC.md` §3.5).
  3. Pattachitra LoRA trained on the **curated** corpus (the 111 images the VLM curation filter
     kept, `reports/pattachitra_precheck_report.json`).
  - This mirrors the ukiyo-e design's actual test (curated vs. uncurated on identical
    architecture/hyperparameters) rather than testing only "does the adapter exist" — the
    published-vs-curated question, not a strawman.
- **Trigger token:** `pattascroll` (novel, not a real word or existing embedding collision — same
  role `ukyowood` played for the ukiyo-e adapter). Every prompt in the set includes it.
- **Prompt set:** 30 prompts spanning Pattachitra's actual subject range, observed directly in the
  curated corpus during the pre-check (`docs/NEXT_MODEL_SPEC.md` §3.5) — mythological/religious
  scenes (Jagannath, Krishna-Radha, Ramayana panels), courtly and village genre scenes, animals,
  and floral/nature motifs, with deliberate figure/portrait coverage (the ukiyo-e A/B's
  preliminary n=12 check was weakened by thin figure coverage — not repeated here):

  - `pat_001`: pattascroll Pattachitra painting of Lord Jagannath, Balabhadra, and Subhadra in a temple shrine
  - `pat_002`: pattascroll Pattachitra scroll painting of Krishna playing the flute under a tree
  - `pat_003`: pattascroll Pattachitra painting of Radha and Krishna dancing together
  - `pat_004`: pattascroll Pattachitra painting of a scene from the Ramayana, Rama drawing his bow
  - `pat_005`: pattascroll Pattachitra painting of Hanuman carrying a mountain of herbs
  - `pat_006`: pattascroll Pattachitra painting of the goddess Durga slaying a demon
  - `pat_007`: pattascroll Pattachitra scroll painting of a royal court scene with musicians
  - `pat_008`: pattascroll Pattachitra painting of a woman in a sari carrying a water pot
  - `pat_009`: pattascroll Pattachitra painting of a farmer plowing a field with oxen
  - `pat_010`: pattascroll Pattachitra painting of two women preparing rice in a village courtyard
  - `pat_011`: pattascroll Pattachitra painting of a fisherman casting a net from a wooden boat
  - `pat_012`: pattascroll Pattachitra painting of a bridal procession with musicians and drummers
  - `pat_013`: pattascroll Pattachitra painting of a mother and child seated under a mango tree
  - `pat_014`: pattascroll Pattachitra painting of temple dancers performing Odissi
  - `pat_015`: pattascroll Pattachitra painting of two wrestlers in a courtyard match
  - `pat_016`: pattascroll Pattachitra painting of a market scene with vendors selling fruit
  - `pat_017`: pattascroll Pattachitra painting of a village street lined with thatched-roof houses
  - `pat_018`: pattascroll Pattachitra painting of a stone bridge over a river with travelers
  - `pat_019`: pattascroll Pattachitra painting of a waterfall cascading through a forest
  - `pat_020`: pattascroll Pattachitra painting of a lotus pond with fish and water birds
  - `pat_021`: pattascroll Pattachitra painting of a tiger stalking through a bamboo grove
  - `pat_022`: pattascroll Pattachitra painting of a peacock displaying its feathers beside a tree
  - `pat_023`: pattascroll Pattachitra painting of an elephant procession with a howdah
  - `pat_024`: pattascroll Pattachitra painting of Mount Kailash with Shiva and Parvati
  - `pat_025`: pattascroll Pattachitra painting of a boat crossing a river during a storm
  - `pat_026`: pattascroll Pattachitra painting of autumn flowers along a temple wall
  - `pat_027`: pattascroll Pattachitra painting of a hilltop temple at sunset
  - `pat_028`: pattascroll Pattachitra painting of a dancer performing with a peacock fan
  - `pat_029`: pattascroll Pattachitra painting of pilgrims climbing steps to a shrine
  - `pat_030`: pattascroll Pattachitra painting of a lantern-lit festival procession at night

- **Seeds:** 42, 43, 44 (same 3 seeds used throughout this project's other evals).
- **n:** 30 prompts × 3 seeds = 90 paired records per arm (270 total generations across 3 arms).
- **Generation:** SDXL base (+ LoRA where applicable), `negative_prompt` = "text, watermark,
  caption, museum label, calligraphy, signature, words, letters" (extends the ukiyo-e negative
  prompt with "caption"/"museum label" — the dominant contamination classes found in the
  Pattachitra corpus pre-check were museum-photography artifacts, not in-print calligraphy, per
  `docs/NEXT_MODEL_SPEC.md` §3.5 — the negative prompt should target what this corpus actually
  contains, not copy the ukiyo-e one unexamined), 30 inference steps, guidance_scale 7.5,
  1024×1024. Identical generation config across all three arms — only the LoRA weights (or their
  absence) differ.
- **Training config:** rank-8 LoRA, SDXL base + `madebyollin/sdxl-vae-fp16-fix`,
  `scripts/_diffusers_train_text_to_image_lora_sdxl.py`, seed 42, `--validation_epochs 15` (per
  `docs/NEXT_MODEL_SPEC.md` §4's already-fixed validation-overrun issue) — same recipe as ukiyo-e's
  curated retrain, for direct comparability of any effect size across the two LoRA projects.
- **Scoring:** independent single-axis VLM judge (`scripts/model_verdict_harness.py`'s
  `score_vlm_judge`, one Ollama call per axis, `qwen2.5vl:7b`) on `style_adherence`,
  `figure_preservation`, `artifact_absence`, in a second phase after the generation pipeline is
  released from GPU memory. **Plus** the OCR binary detector
  (`scripts/detect_text_artifacts.py`, `n_detections >= 1` rule) as a secondary signal, run over
  the same images.
- **Requires `ollama serve` running locally** with `qwen2.5vl:7b` pulled. Zero paid APIs.

## Statistical method

**Paired difference** (curated − uncurated, and each LoRA arm − `sdxl_base`, matched by identical
`prompt_id`+`seed` across arms), same paired-diff/SEM methodology as the ukiyo-e A/B
(`docs/MODEL_VERDICT.md` §4.1's methodology note, §4.6, §4.8) — SEM computed on the per-pair
differences directly, not independent-sample quadrature combination. This is the statistically
correct, more powerful design for matched observations and is applied identically regardless of
which way any result comes out.

## Endpoints (fixed before running)

- **Primary endpoint:** `artifact_absence`, paired diff (curated − uncurated), independent-axis
  VLM scoring, on the 30-prompt set above.
- **Promotion threshold:** paired diff / paired SEM > 2.0 — same bar used throughout this
  project (`scripts/check_eval_gate.py`'s CI gate; `docs/MODEL_VERDICT.md` §4).
- **Guardrails (non-inferiority):** `style_adherence`, `figure_preservation` on the curated-vs-
  uncurated comparison — pass if the curated arm's paired diff does not show a regression clearing
  2×SEM in the negative direction.
- **Required diagnostic (not a promotion gate, but mandatory in the writeup):** both LoRA arms'
  `artifact_absence` and `style_adherence` paired diffs **against `sdxl_base`** (the §4.8-style
  root-cause check), computed and reported regardless of the primary result — this determines
  whether any observed effect (or null) is attributable to the training data specifically, not
  assumed after the fact.
- **Required power reporting (not optional, regardless of outcome):** the paired SEM's implied
  MDE at 80%/90% power (§4.7's method) and the 95% CI on the observed diff, computed and stated
  in the same document section as the point estimate — a result that doesn't clear the promotion
  bar must be labeled "does not clear the bar" and, separately, "underpowered below the MDE" or
  "rules out an effect of size X," whichever the CI/MDE actually support. Never state a null result
  without this context.
- **Secondary, non-gating signal:** OCR binary detector's paired-proportion result (McNemar exact
  test, `n_detections >= 1` rule) on curated vs. uncurated — reported for directional consistency
  only, given §4.9 established it is lower-powered than the rubric for this style domain. If
  Pattachitra's contamination profile (museum labels/captions, more legible Latin/mixed script per
  the corpus pre-check) turns out to be more OCR-legible than ukiyo-e's stylized pseudo-kanji, this
  secondary signal may be more informative here than it was for ukiyo-e — that would need its own
  validation pass (a stratified visual-inspection sample, as `scripts/validate_text_detector.py`
  did) on Pattachitra images specifically before being used for anything beyond a directional
  check; the ukiyo-e validation numbers do not transfer.

## Decision rule (fixed before seeing results)

**Promote the curated-trained adapter over the uncurated-trained adapter if and only if:**
1. The primary endpoint clears the promotion threshold (diff/SEM > 2.0), **and**
2. Neither guardrail shows a regression clearing 2×SEM in the negative direction.

**If the primary does not clear the threshold:** report the result with its MDE and 95% CI
(required reporting above) — state explicitly whether the data rules out a specific effect size
or is merely underpowered to distinguish a smaller true effect from none. This is not grounds to
lower the threshold, drop a guardrail, subset the prompts, re-run with a different seed set, or
switch to the OCR secondary signal to find a passing result — any of those repeats exactly the
methodology error this project already corrected once.

**If the base-model diagnostic shows the LoRA-induced regression (both arms vs. `sdxl_base`) is
smaller in absolute terms than it was for ukiyo-e** (§4.8: published −0.0500, curated −0.0422),
that is itself a reportable finding about whether Pattachitra's cleaner source corpus (18.4% flag
rate vs. ukiyo-e's 71.2%, `docs/NEXT_MODEL_SPEC.md` §3.5) produces a smaller artifact-inducing
effect to begin with — worth stating even though it doesn't change the promotion decision.

## What happens after this file is committed

**No training is authorized by this document.** Training (both the uncurated and curated LoRA
runs) and the eval run still require GG's explicit approval and GCP spend authorization per
`docs/NEXT_MODEL_SPEC.md` §5's cost estimate, exactly as for the ukiyo-e project. Once approved,
the 30-prompt × 3-seed × 3-arm grid generates + scores, the paired analysis runs per the design
above, and `docs/MODEL_VERDICT.md` gets a new §7 (Pattachitra LoRA A/B) written to this same
standard from the first pass — no separate withdrawal-and-correction cycle should be needed if
this design is followed as written.
