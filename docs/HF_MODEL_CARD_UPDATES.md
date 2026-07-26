# Ready-to-apply HF model card updates — `aetherart-ukiyo-sdxl` and `aetherart-ukiyo-sd21`

**Status: prepared, NOT applied — blocked only on a read-only HF token**
(`api.whoami()` confirms `role: read`). The `style_adherence` figures below were marked
PROVISIONAL on 2026-07-25 pending a judge positive control; that control has since been
redesigned, re-run, and PASSED decisively — the figures below are CONFIRMED, apply as written
once a write-scoped token is available.

**Resolution (2026-07-26) — the 2026-07-25 PROVISIONAL marking is LIFTED:** the original control
compared real ukiyo-e art against `sdxl_base` outputs generated from ukiyo-e-*worded* prompts
(diff/SEM = −1.507, wrong-signed) — a near-ceiling task, not a fair validity test, since the
comparison arm was itself asked to look like ukiyo-e on a style SDXL already renders convincingly.
Redesigned per `docs/WEIGHT_SWEEP_PREREGISTRATION.md`'s amendment: real ukiyo-e art (n=23) vs. two
off-style contrasts under the same question — real Pattachitra art (n=100, diff/SEM = **+25.062**)
and generic non-style-specific `sdxl_base` outputs (n=90, diff/SEM = **+25.580**). **PASS on both,
decisively** (`reports/judge_style_positive_control.json`) — the judge correctly separates real
ukiyo-e art from genuinely different content by an overwhelming margin in both cases. **Every
`style_adherence` number below is CONFIRMED**, not provisional. The `artifact_absence` numbers
(−0.0500/3.49×SEM, −0.0422/3.11×SEM, 0.9222 base) were never affected either way — domain-neutral
question, not part of what either version of this control tested. See `docs/MODEL_VERDICT.md` §4.10
for the full result and interpretation.

**The curated-retrain "promotion" is still WITHDRAWN and NOT re-uploaded** (`docs/MODEL_VERDICT.md`
§4.6): the arm-to-arm `artifact_absence` diff (+0.0078, 0.583×SEM) does not clear the
pre-registered 2×SEM bar, and §4.9 found no better-powered zero-cost local metric to resolve
that specific comparison. **Do not claim a measured improvement for the curated retrain over the
published checkpoint anywhere on the card.** What follows below is a *different, separately
measured* claim — not a re-framing of the withdrawn one — about what using **either** adapter
costs/gains relative to `sdxl_base` with no adapter at all (`docs/MODEL_VERDICT.md` §4.8).

## Replace the "Known limitations" bullet with a measured cost/benefit disclosure

**Why replace, not just leave the existing vaguer text:** the current card describes the
calligraphy artifact only qualitatively ("partially mitigated"). `docs/MODEL_VERDICT.md` §4.8
has since measured it precisely, against a `sdxl_base` (no-adapter) baseline, under the trusted
independent-axis VLM regime (n=90 paired, `reports/lora_ab_base_comparison.json` +
`reports/lora_ab_30prompt_independent.json`). This is a more useful, more specific number for a
downstream user deciding whether to use the adapter than a qualitative note, and per explicit
instruction it is **not softened** — the regression is real, measured, and stated plainly.

**Current text (in the live card, `README.md` on `gauravgandhi2411/aetherart-ukiyo-sdxl`):**

```
- **Calligraphy artefact (partially mitigated):** WikiArt source images contain metadata captions and script text. The adapter learned this as part of ukiyo-e style. The negative prompt suppresses most instances but does not eliminate the entanglement between style signal and text signal. Correct fix: retrain on a curated dataset with no text annotations (~5 hours of curation).
```

**Replace with:**

```
- **Measured cost/benefit of this adapter (independent-axis VLM judge, n=90 paired,
  `sdxl_base` = no adapter as the reference point):**
  - **`sdxl_base` alone scores `artifact_absence` 0.9222 — cleaner than this adapter.** Applying
    this LoRA measurably *increases* visible embedded text/calligraphy/cartouche marks relative to
    generating the same prompts with no adapter at all: **published checkpoint −0.0500
    (3.49× the paired SEM), an unfiltered-training-set retrain −0.0422 (3.11× SEM)** — both are
    individually significant regressions, not noise. This is the entanglement between "ukiyo-e
    style" and the WikiArt source images' embedded captions/signatures/script that produced the
    style signal the adapter learned; it is a real, measured tradeoff of using this adapter, not
    fully "mitigated" by the default negative prompt.
  - **What the adapter buys in exchange: a measured style-adherence lift over
    `sdxl_base` alone** — a curated-training-set retrain lifts `style_adherence` +0.0100 over
    base (2.82× SEM); the published checkpoint (unfiltered training set) lifts it +0.0056
    (1.68× SEM, not itself significant at this n). `sdxl_base` already scores 0.9389 on
    `style_adherence` for ukiyo-e-styled prompts from its own pretraining, so headroom for any
    adapter to add is small. **A positive control confirms this rubric CAN distinguish real
    ukiyo-e art from off-style contrasts decisively (diff/SEM = +25.062 vs. real Pattachitra art,
    +25.580 vs. generic `sdxl_base` outputs) — these lift numbers are confirmed to be measuring a
    real style-adherence signal, not an instrument artifact.**
  - **Net read:** this adapter's main value is a modest style lift over what `sdxl_base` already
    renders unassisted (confirmed, not provisional, though still modest — below the 2×SEM bar for
    the unfiltered checkpoint), at the cost of a real, larger, and statistically significant
    increase in embedded-text artifacts. Whether that trade is worth it depends on the use case —
    for artifact-sensitive generations,
    consider `sdxl_base` alone with an explicit "ukiyo-e style" prompt, or add this adapter and
    screen outputs for text artifacts downstream. A follow-up retrain-and-eval attempt
    investigating whether more aggressive dataset curation or a different training recipe can
    close this gap is tracked in `docs/NEXT_MODEL_SPEC.md`, not yet completed.
  - Full methodology and numbers: `docs/MODEL_VERDICT.md` §4.6–§4.9 in the
    [AetherArt GitHub repo](https://github.com/gaurav-gandhi-2411/AetherArt).
```

## No training-details table change

The published checkpoint is unchanged (still the original, unfiltered-training-set weights) —
no row edit is warranted; the cost/benefit disclosure above applies to whichever checkpoint is
live and should be read as a property of "this adapter" generally, not tied to a specific
revision.

## Also apply (unrelated, unaffected by the above — still valid)

Merge the two open HF card draft PRs (`refs/pr/1` on both `aetherart-ukiyo-sdxl` and
`aetherart-ukiyo-sd21`) once the write-scoped token is available — they warn that HF's
auto-generated "Use this model" snippet omits the required `madebyollin/sdxl-vae-fp16-fix` VAE
and uses `device_map="cuda"`/`bfloat16`, which produces black images if copied as-is.

## SD 2.1 companion card (`aetherart-ukiyo-sd21`) — do NOT copy the SDXL numbers over

**The base-model comparison (§4.8) was run only against the SDXL adapter and `sdxl_base`.** The
SD 2.1 adapter (trained at 512×512, a separate model with its own weights, evaluated against a
different base model) was never measured this way. Do not apply the SDXL cost/benefit numbers
above to the SD 2.1 card — its existing "Calligraphy artifact (partially mitigated, not fixed)"
language remains the accurate, honest statement for that model until it gets its own equivalent
measurement.
