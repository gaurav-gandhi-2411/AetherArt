# Ready-to-apply HF model card updates — `aetherart-ukiyo-sdxl` and `aetherart-ukiyo-sd21`

**Status: prepared, NOT applied — and the "calligraphy artefact fixed" claim below is
WITHDRAWN.** This document originally staged a claimed `artifact_absence` improvement for the
curated retrain. `docs/MODEL_VERDICT.md` §4.6 found that claim was an artifact of a correlated
(single-call, multi-axis) VLM judge: the full n=90 paired A/B, rescored under the trusted
independent-axis regime, shows the primary endpoint at 0.583×SEM — well below the pre-registered
2×SEM promotion bar. **Do not apply the "Known limitations" replacement or the training-details
row below.** The published checkpoint remains champion; no re-upload is warranted by this A/B.
This file is kept (not deleted) as an honest record of what was staged and why it was withdrawn —
see the "Withdrawn" sections below for what NOT to publish, and the one item still valid
("Also apply") for what remains actionable.

Also still blocked on a read-only HF token (`api.whoami()` confirms `role: read`) regardless of
the above — no HF write action of any kind should be attempted without a write-scoped token.

## WITHDRAWN — do not apply: "Known limitations" bullet replacement

The originally-staged replacement claimed a **measured** `artifact_absence` improvement
(+0.040, 3.18×SEM). That number came from a VLM judge that scores all three axes (style, figure,
artifact) in one call — a halo-effect risk `docs/MODEL_VERDICT.md` §4.5 flagged and §4.6 then
confirmed materially inflates the apparent effect. Under independent single-axis scoring (each
axis judged in its own call, no other axis in context), the same 90-pair dataset shows
`artifact_absence` at +0.0078 (0.583×SEM) — well below the pre-registered promotion bar, so this
is not a demonstrated fix. **This is not the same as "no effect exists"** — `docs/MODEL_VERDICT.md`
§4.7 shows the design is underpowered to fully rule out a smaller true effect (its 95% CI does
rule out the original +0.040 claim, but not a true effect in the ~0.01–0.03 range). The precise,
supportable statement is "did not clear the pre-registered bar," not "no effect." **The live
card's current "Calligraphy artefact (partially mitigated)" language should stay as-is.** It
already honestly describes the unresolved state; nothing here supersedes it.

If a future retrain/eval cycle does produce a properly-powered, independent-axis-scored
improvement, `docs/MODEL_VERDICT.md` §4.6–§4.8's numbers and methodology are the template to
follow (paired diff, n≥90, one Ollama call per axis, pre-registered threshold, plus the §4.8-style
root-cause diagnostics before accepting a null) — not §4.3's now-corrected
single-call design.

## WITHDRAWN — do not apply: training details table row

The originally-staged row claimed the published checkpoint was switched to the curated retrain.
It was not switched, and per the above, should not be, on the basis of this A/B. No training-table
edit is warranted.

## Also apply (unrelated to the above, unaffected by the withdrawal — still valid)

Merge the two open HF card draft PRs (`refs/pr/1` on both `aetherart-ukiyo-sdxl` and
`aetherart-ukiyo-sd21`) once the write-scoped token is available — they warn that HF's
auto-generated "Use this model" snippet omits the required `madebyollin/sdxl-vae-fp16-fix` VAE
and uses `device_map="cuda"`/`bfloat16`, which produces black images if copied as-is. These are
independent of the curated-retrain question and should be merged regardless.

## SD 2.1 companion card (`aetherart-ukiyo-sd21`) — no change needed

The originally-staged edit here only added an "Update" note pointing at the (now-withdrawn) SDXL
claim. Since that claim is withdrawn, no edit to the SD 2.1 card is needed at all — its existing
"Calligraphy artifact (partially mitigated, not fixed)" language remains accurate as originally
written.
