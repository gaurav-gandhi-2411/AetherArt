# Ready-to-apply HF model card updates — `aetherart-ukiyo-sdxl` and `aetherart-ukiyo-sd21`

**Status: prepared, NOT applied.** Blocked on a read-only HF token (`api.whoami()` confirms
`role: read`) — apply this once GG supplies a write-scoped token (classic **Write** role, or a
fine-grained token scoped to `aetherart-ukiyo-sdxl` + `aetherart-ukiyo-sd21` with **Write**
repo-content permission).

**Do NOT destructively overwrite the current published weights.** Before uploading the curated
checkpoint as the new main-revision LoRA weights, first tag/preserve the current published
revision (e.g. `git tag`/`hf_hub` revision pinning, or an explicit "legacy-checkpoint-1000"
branch on the HF repo) so the currently-published adapter stays retrievable — it has 106
downloads in the last month per the task brief; those users should not silently lose access to
what they downloaded.

## Exact replacement for the "Known limitations" bullet

**Current text (in the live card, `README.md` on `gauravgandhi2411/aetherart-ukiyo-sdxl`):**

```
- **Calligraphy artefact (partially mitigated):** WikiArt source images contain metadata captions and script text. The adapter learned this as part of ukiyo-e style. The negative prompt suppresses most instances but does not eliminate the entanglement between style signal and text signal. Correct fix: retrain on a curated dataset with no text annotations (~5 hours of curation).
```

**Replace with:**

```
- **Calligraphy artefact — fixed by a curated retrain (checkpoint-1000, curated dataset), now the published weights.** The original checkpoint (WikiArt's 80-image source set, uncurated) entangled ukiyo-e style signal with the WikiArt metadata captions/script text embedded in source images. Fix: a local Ollama VLM judge (`qwen2.5vl:7b`, zero cost, no paid API) screened the 80-image source set for text/calligraphy artifacts — 57 images (71.2%) were flagged and excluded, 23 kept for retraining. (An EasyOCR-based first attempt at this filter was tried and discarded for false positives before settling on the VLM-judge approach.) The retrained checkpoint was evaluated against the original in a **pre-registered** A/B (design fixed and committed before the eval ran — see `docs/AB_PREREGISTRATION.md` in the GitHub repo) on 30 ukiyo-e-styled prompts × 3 seeds (n=90 paired generations per checkpoint), scored by the same local VLM judge on three axes.
  - **Measured result: `artifact_absence` improved by +0.040 (paired difference, SEM 0.0126, n=90) — 3.18× the paired SEM, clearing the pre-registered 2×SEM promotion threshold decisively.** This is the calligraphy-artifact fix, measured rather than assumed.
  - `style_adherence` and `figure_preservation` (tracked as non-inferiority guardrails, not improvement targets) did **not regress**. An independent-scoring check (each axis scored in a separate model call, since the original judge scores all three axes in one call — a known setup for halo-biased ratings) confirmed the `artifact_absence` result is robust, but found the apparent guardrail *improvements* do not reliably survive independent scoring — so this card claims only **no regression** on style/figure quality, not improvement, even though the raw paired numbers on the primary eval were also positive there.
  - Full methodology, numbers, and the halo-effect check: `docs/MODEL_VERDICT.md` §4 in the [AetherArt GitHub repo](https://github.com/gaurav-gandhi-2411/AetherArt).
```

## Training details table — add a row

The existing table describes the *original* 80-image, uncurated training run. Add a note (new
row or a footnote under the table) making clear which checkpoint is now published:

```
| Published checkpoint | **curated retrain**, trained on the 23-image VLM-judge-curated subset (see "Known limitations" above); prior checkpoint (80-image, uncurated) preserved at revision `<TAG>` |
```

Replace `<TAG>` with whatever revision tag/name is actually used when the prior weights are
preserved — this placeholder must not go out with a dangling reference.

## Also apply (unrelated to the above, already pending — task 2's other item)

Merge the two open HF card draft PRs (`refs/pr/1` on both `aetherart-ukiyo-sdxl` and
`aetherart-ukiyo-sd21`) once the write-scoped token is available — they warn that HF's
auto-generated "Use this model" snippet omits the required `madebyollin/sdxl-vae-fp16-fix` VAE
and uses `device_map="cuda"`/`bfloat16`, which produces black images if copied as-is. These are
independent of the curated-retrain promotion and should be merged regardless.

## SD 2.1 companion card (`aetherart-ukiyo-sd21`) — do NOT imply an untested improvement

**The curated retrain and its A/B were run only against the SDXL checkpoint.** The SD 2.1
adapter (trained at 512×512, a separate model with its own weights) was never retrained on the
curated dataset and never evaluated in this A/B. Do not edit the SD 2.1 card's calligraphy-
artifact limitation language to imply the fix applies there.

**Current text (live card, `README.md` on `gauravgandhi2411/aetherart-ukiyo-sd21`):**

```
- **Calligraphy artifact (partially mitigated, not fixed):** WikiArt Ukiyo-e source images contain metadata captions with artist signatures and script text embedded in the image margins. The adapter learned these as part of "ukiyo-e style." The default negative prompt suppresses most instances but does not eliminate the artifact entirely — the style signal and text signal are entangled in the adapter weights. The correct fix is retraining on a curated dataset with no text annotations, which would require approximately 5 hours of curation work.
```

**Append one line (keep the rest of the bullet unchanged — do not claim the fix applies here):**

```
- **Calligraphy artifact (partially mitigated, not fixed):** WikiArt Ukiyo-e source images contain metadata captions with artist signatures and script text embedded in the image margins. The adapter learned these as part of "ukiyo-e style." The default negative prompt suppresses most instances but does not eliminate the artifact entirely — the style signal and text signal are entangled in the adapter weights. The correct fix is retraining on a curated dataset with no text annotations, which would require approximately 5 hours of curation work.

  *Update: this exact fix — VLM-judge dataset curation + retrain — was validated on the SDXL companion adapter ([`gauravgandhi2411/aetherart-ukiyo-sdxl`](https://huggingface.co/gauravgandhi2411/aetherart-ukiyo-sdxl)), which measured a +0.040 (3.18×SEM, n=90) improvement in artifact-absence. It has NOT been run for this SD 2.1 (512×512) adapter — the limitation above still applies as originally documented until this adapter gets its own curated retrain and eval.*
```

This keeps the two cards honest about what was actually measured versus what would need its own
separate retrain-and-eval cycle to claim.
