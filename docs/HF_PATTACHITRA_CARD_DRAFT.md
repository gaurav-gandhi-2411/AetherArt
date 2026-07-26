# Ready-to-apply HF model card — `aetherart-pattachitra-sdxl` (new repo, not yet published)

**Status: drafted, NOT pushed — blocked only on a read-only HF token**
(`api.whoami()` confirms `role: read`). This adapter has never been published before; there is no
existing card to update, unlike the ukiyo-e cards (`docs/HF_MODEL_CARD_UPDATES.md`).

**Basis for this recommendation (`docs/MODEL_VERDICT.md` §7.2–§7.4):** a full evaluation cycle —
adapter-weight sweep, a judge-question-hardcoding bug found and fixed, a base-arm confound found
and fixed in the positive control, and a symmetric positive control (mirroring the design that
passed for this project's ukiyo-e adapter) confirming the judge genuinely perceives Pattachitra
style (PASS, +17.951×SEM vs. real ukiyo-e art, +28.659×SEM vs. generic `sdxl_base`, both far
exceeding their own MDE). Both `style_adherence` and `figure_preservation` are confirmed, not
provisional, for this recommendation.

## Card text

```markdown
# AetherArt Pattachitra LoRA (SDXL)

Rank-8 LoRA adapter for Pattachitra (traditional Odisha folk art) style transfer on
`stabilityai/stable-diffusion-xl-base-1.0`, trained on a 100-image curated corpus.

## Recommended usage: adapter weight 0.3–0.5, NOT the library default of 1.0

**This adapter must be applied at a reduced weight (`adapter_weights=[0.3]` to `[0.5]`) to see a
benefit over `sdxl_base` alone.** At the library-default full weight (1.0), this adapter measurably
REGRESSES both style authenticity and figure/subject preservation relative to `sdxl_base` with no
adapter at all — do not deploy at weight 1.0.

| Weight | style_adherence vs. sdxl_base | figure_preservation vs. sdxl_base | Recommended? |
|---|---|---|---|
| 0.3–0.5 | improves, up to +3.6x the measurement's own standard error (checkpoint-dependent) | does not regress; modestly improves | **Yes** |
| 1.0 (library default) | regresses (−3.0x to −4.6x SEM) | regresses (−5.4x to −7.5x SEM) | **No** |

## Known limitation: full-weight figure/subject dropout

At `weight=1.0`, generations can drop prompted human figures entirely (visually confirmed:
identical prompt+seed vs. `sdxl_base`, the LoRA output omits the person, `sdxl_base` renders them
correctly). This is a checkpoint-wide effect (500 and 1000 steps both affected, likely stemming
from generic auto-generated training captions), not fixed by choosing a different checkpoint —
it is fixed by choosing a lower weight.

## Measurement methodology

Independent-axis VLM judge (`qwen2.5vl:7b`, zero-cost local), n=90 paired per weight/checkpoint
comparison vs. `sdxl_base`, adapter-weight sweep at 0.3/0.5/0.7/1.0. Full methodology, positive
control validating the judge can perceive Pattachitra style (PASS, +17.951x/+28.659x SEM against
two independent off-style contrasts), and complete numbers:
[AetherArt GitHub repo, docs/MODEL_VERDICT.md §7](https://github.com/gaurav-gandhi-2411/AetherArt).

## Training details

- Base model: `stabilityai/stable-diffusion-xl-base-1.0`
- Rank: 8, alpha: 8 (`lora_alpha = rank`)
- Corpus: 100 curated Pattachitra images (manual QA pass on training data before training)
- Trigger token: `pattascroll`
- Checkpoints: 500 and 1000 steps (both viable at low weight; 1500 steps not evaluated at
  reduced weight, out of this diagnostic's scope)
```

## What this card deliberately does NOT claim

A full-weight recommendation, a claim that this adapter is a strict upgrade at every weight, or a
claim beyond what was measured — no weight between 0.3 and 0.5 was individually tested; the
recommendation is a range bounded by the two tested points that both pass, not a claim that every
value in between has been independently verified.

## Apply once a write-scoped token exists

1. Create the `aetherart-pattachitra-sdxl` repo (new — this adapter has no prior HF presence).
2. Upload the checkpoint-500 and checkpoint-1000 LoRA weight files
   (`data/lora/pattachitra-curated/.../checkpoint-{500,1000}`).
3. Push the card text above as `README.md`.
4. Note in the card (already included above) that `adapter_weights` must be set explicitly by the
   caller — most tooling defaults to 1.0, which this card documents as the wrong setting for this
   adapter specifically.
