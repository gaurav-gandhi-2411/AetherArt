# Next Model Spec: Indian-Art LoRA Domain Selection

**Status: SPEC ONLY — no training has occurred.** This document scopes the next LoRA fine-tuning
target following the validated recipe (curated dataset → rank-8 SDXL LoRA → measured artifact
reduction via a pre-registered A/B — see `docs/MODEL_VERDICT.md` §4). Training requires explicit
approval of the recommendation below.

**Target user / pain point / success metric / who pays** (per this project's spec convention):
- **Target user:** portfolio reviewers and recruiters evaluating AI/ML systems work — this is a
  differentiation play, not a product with paying users.
- **Pain point:** the ukiyo-e LoRA proved the recipe works, but ukiyo-e itself is a crowded niche
  (thousands of existing SD/SDXL LoRAs). A domain with thin LoRA competition and a real,
  licence-clean corpus demonstrates the recipe *and* domain judgment.
- **Success metric:** same as the ukiyo-e precedent — a pre-registered A/B (or, for a first
  release, a documented base-vs-LoRA comparison) showing a measured, not assumed, quality
  improvement over the SDXL base prior.
- **Who pays:** nobody — portfolio/GCP-credit-funded, same as the current recipe.

---

## 1. Method

For each candidate style, this spec checked, in order (a disqualifying result at any step ends
that candidate's evaluation early, per the ukiyo-e project's own "no LoRA needed if the base
model already renders it well" lesson):

1. **Corpus availability and licence** — Wikimedia Commons category file counts (direct
   `WebFetch`/`WebSearch` against `commons.wikimedia.org`, not estimated), museum open-access
   holdings (Metropolitan Museum of Art Open Access, Victoria & Albert Museum, others checked
   where surfaced), and WikiArt coverage (the corpus source for the *original* SD 2.1 ukiyo-e
   LoRA — checked to see if it's viable again here).
2. **SDXL's existing prior** — **directly tested**, not inferred: base SDXL (no LoRA), 30 steps,
   guidance 7.5, seed 42, one prompt per candidate style, generated locally
   (`outputs` under this session's scratchpad). This is the same test that would reveal "SDXL
   already does this well, a LoRA adds nothing" — the exact failure mode that would have been
   missed by literature review alone.
3. **Existing LoRA competition** on Hugging Face and CivitAI (`WebSearch` against both
   platforms).
4. **Expected artifact classes** — reasoned from documented style conventions, flagged as
   inference where no direct generation-defect evidence exists yet (that requires the actual
   training run this spec doesn't do).

---

## 2. Candidate comparison

| Candidate | Commons corpus (verified) | Museum anchor | WikiArt coverage | SDXL base prior (direct test) | Existing LoRA competition |
|---|---|---|---|---|---|
| **Madhubani** (Mithila) | ~40 files, `Category:Madhubani_painting`; two of the four largest subcategories are individually-attributed to *living/recent* named artists (Bharti Dayal, Mahasundari Devi) | V&A confirmed holdings (exact count unverified; several catalogued Sita Devi/Jogmaya Devi works found) | **None** — zero `wikiart.org` results | Decent — recognizable folk-art palette and bold outline, plausible at a glance | **Real competition**: SD1.5 LoRA, 6,580 downloads / 910 likes / "Very Positive" (CivitAI). SDXL-specific niche open, but not a competition vacuum. |
| **Pattachitra** (Odisha) | **62 (main) + 38 (Odisha) + 40 (West Bengal) subcats ≈ 140 files** — largest folk-art corpus of the five | Not directly confirmed in this pass (unverified — worth a follow-up Met/V&A check before committing) | **None** found | Decent — flat colour, dense ornamental border, circular iconographic composition; not clearly *distinctively* Pattachitra vs. generic "ornate Indian religious art" | **None found** on CivitAI or HF |
| **Mughal miniature** | **500+** across `18th-century Mughal miniatures` (168), `17th-century` (397), `19th-century` (62), `Walters Art Museum` subcat (43), `Mughal ramayana` (70), plus the parent categories — **by far the deepest corpus**, comparable in order of magnitude to what supported the original ukiyo-e LoRA | **Strong** — Walters Art Museum has a dedicated Commons subcategory (i.e. real museum digitization pipeline), Met Museum has Mughal-era holdings | Likely thin (not exhaustively checked given the disqualifying result below) | **Convincing** — the direct test produced a genuinely authentic-looking Mughal court scene (architecture, figures, palette, framing) on the very first try, no LoRA | Existing CivitAI LoRA (2023, "Alpha" stage; no adoption metrics found — likely low-traffic, but prior art exists) |
| **Kalamkari** | **~65 files** (~55 main + ~10 in a Met-sourced subcategory) | **Confirmed**: Met Museum Open Access holds kalamkari textiles directly usable under its open-access terms (e.g. a fragment of a floorspread, a figural hanging from the early 1600s) | **None** found | Plausible but generic — the direct test produced a decorative "tree of life in a tapestry medallion" that reads as generic ornamental art, not distinctively Kalamkari's hand-drawn/block-printed narrative-figuration technique or its characteristic limited natural-dye palette | **None found** on CivitAI or HF — the only Indian-folk-art LoRA found on either platform for a *different* style is `rexoscare/kalighat-paintings-lora` (Kalighat, not Kalamkari) |
| **Warli** | 33 files, `Category:Warli_paintings` — thinnest corpus of the five | Not confirmed | **None** found | **Convincing** — the direct test produced an immediately recognizable, essentially correct Warli composition (white stick-figures/triangles on an earth-tone ground, circular dance motif) on the first try, no LoRA | Existing CivitAI LoRA (Flux-based, Sept 2024) |

**Two candidates are disqualified by the base-SDXL test, not by corpus or competition:**
Mughal miniature and Warli both already render convincingly from a plain prompt with zero LoRA.
This is the same lesson the ukiyo-e baseline finding established for this project — a style the
base model already renders well needs no LoRA, and training one would not produce a measurable,
honestly-reportable improvement. Mughal miniature is disqualified *despite having the best corpus
of the five* — corpus depth doesn't matter if the model doesn't need the adapter.

**Madhubani is not recommended for a second reason beyond thin corpus:** the licensing profile is
riskier than the ukiyo-e precedent. Ukiyo-e's corpus was overwhelmingly public-domain historical
prints; Madhubani's cleanest Commons subcategories are individually attributed to named,
plausibly-living contemporary artists (Bharti Dayal, Mahasundari Devi), which is a different and
more sensitive licensing posture than "PD-old master reproduction." Also faces real existing
SD1.5 competition (910 likes) — the SDXL niche is open, but not into a vacuum.

---

## 3. Recommendation: **Pattachitra**, with **Kalamkari** as the close second choice

**Primary: Pattachitra.**
- **Corpus:** ~140 files across Commons' main category + two regional subcategories — roughly
  **double Kalamkari's ~65** and well above Madhubani's ~40 or Warli's 33. This matters because
  the ukiyo-e project's own curation pass discarded 57/80 source images (71.2%) to text/
  calligraphy artifacts — a similarly aggressive curation loss rate applied to a 65-image corpus
  could leave as few as ~19 clean images (thin for a rank-8 LoRA); applied to ~140 leaves
  comfortable headroom even at that loss rate.
- **SDXL prior:** the direct test produced a plausible-but-generic result (decorative Hindu-deity
  medallion, decent flat colour and ornamental border) — recognizably *in the neighborhood* of
  Pattachitra but not confidently distinctive from generic "ornate Indian religious art." Real
  room for a LoRA to sharpen this into something an eye familiar with the style would call
  authentically Pattachitra, without SDXL already having solved the problem.
- **Competition:** zero existing LoRA found on CivitAI or Hugging Face for this exact style — a
  genuinely open niche, unlike Madhubani or the disqualified Mughal/Warli candidates.
- **Open item before committing:** this pass did not confirm a museum open-access anchor the way
  Kalamkari's Met Museum holdings were confirmed — a direct Met/V&A/Smithsonian API check for
  Pattachitra specifically should happen before finalizing the dataset-sourcing plan (§4), since
  Commons alone, at ~140 files with unconfirmed individual licence tags, needs the same
  per-file licence verification called out for Madhubani.

**Close second: Kalamkari.** Smaller corpus (~65) but the **only candidate with a directly
confirmed museum open-access anchor** (Met Museum), the clearest "SDXL doesn't already know this"
signal among the three viable candidates (its generated test image was the most genuinely
generic-looking of the three non-disqualified styles), and equally zero LoRA competition. If
Pattachitra's museum-anchor check comes back thin or licence-murky, Kalamkari is the fallback,
not a distant alternative.

**Not recommended:** Madhubani (thin + risky-licence corpus, real existing competition), Mughal
miniature (best corpus, but disqualified — SDXL already renders it well), Warli (thinnest corpus,
disqualified — SDXL already renders it well, and Flux competition already exists).

---

## 3.5 Pre-check result (run before any training commitment) — **PROCEED**

Per this document's own pre-registered decision rule (stated before running: ≥50 clean images →
proceed to spec-approved training; <50 → blocked, evaluate Kalamkari), the full corpus and VLM
curation filter were run end-to-end as a pre-check, not a training run.

**Corpus assembly** (`scripts/_fetch_pattachitra_corpus.py`, `reports/pattachitra_corpus_manifest.json`):
139 unique Commons *titles* across the three categories (`Category:Pattachitra`,
`Category:Pattachitra in Odisha`, `Category:Pattachitra in West Bengal`), all downloaded with no
reported errors — but only **136 unique files physically exist on disk**. This 139→136 gap was
reconciled exactly (not assumed): it is a **distinct bug from the concurrent-instance race
condition** found in the curation script (`PLAN.md`) — 3 pairs of Commons titles differ only in
letter case (`File:Bengal patachitra 1.jpg` vs `File:Bengal Patachitra 1.jpg`, confirmed via HTTP
HEAD to be genuinely different source images, different byte sizes: 3,789,829 vs 1,028,117 for
pair 1, etc.). Windows/NTFS filesystems are case-insensitive, so both downloads wrote to what the
OS treats as the *same* path — the later write in each pair silently overwrote the earlier one's
content, even though both manifest entries correctly recorded a successful download (no
exception was raised; the write itself succeeded). Verified which half of each pair survived by
comparing on-disk file size against each URL's `Content-Length` — in all 3 cases the
**lowercase**-titled variant's content is what persists; the uppercase-titled variant's content
(all 3 licensed CC BY-SA 4.0) was never available to the curation step. The curation step itself
has zero discrepancy: 136 files on disk in, 136 classified out (111 + 25).

**Actual per-file licence of the 136 files that physically exist and were curated** (not the 139
manifest entries, 3 of which reference overwritten, never-curated content) — "Wikimedia Commons"
is a host, not a licence:

| Licence | Count |
|---|---|
| CC BY-SA 4.0 | 83 |
| CC BY 2.0 | 21 |
| CC BY 3.0 | 9 |
| CC BY-SA 3.0 | 7 |
| CC BY 4.0 | 7 |
| Public domain | 5 |
| CC0 | 4 |

All 136 are commercial-compatible; **zero** flagged non-commercial/no-derivatives (the disqualifying
markers checked: `noncommercial`, `nc-`/`-nc`, `nd-`/`-nd`) — this holds for both the 136 retained
files and the 3 overwritten-and-lost ones (all CC BY-SA 4.0), so the case-collision bug has no
licence-compliance impact, only a 3-image count impact already reflected in the 136 total below.

**VLM curation filter** (`scripts/_curate_pattachitra_corpus.py`, local Ollama `qwen2.5vl:7b`, zero
cost, tailored to Pattachitra's actual contamination classes — museum labels/watermarks and
documentary/process photos, not ukiyo-e's in-print calligraphy):

| Metric | Value |
|---|---|
| Candidate images screened | 136 |
| Clean (kept) | **111** |
| Flagged | 25 |
| Flag rate | 18.4% |
| — text/watermark/museum label | 21 |
| — not-artwork (process/documentary photo) | 4 |

Flag rate is far lower than ukiyo-e's 71.2% (80→23) — Pattachitra's Commons corpus is
predominantly clean scroll-painting reproductions, not scanned prints with embedded calligraphy;
the dominant contamination class is museum/photography artifacts (labels, captions), a smaller
and more mechanically-filterable problem than ukiyo-e's in-image script.

**HOLD on training, independent of the corpus gate passing.** The corpus decision rule (≥50
clean → PROCEED) is satisfied at 111. This is not a hold on the curation *recipe* — the ukiyo-e
LoRA A/B's follow-up root-cause audit (`docs/MODEL_VERDICT.md` §4.8) found curation directionally
validated: both LoRA variants significantly regress `artifact_absence` relative to no adapter
(confirming training data does teach the artifact), and curation recovers a real ~16% of that
regression in the correct direction. The hold is on the **evaluation method**: that recovered
fraction was too small, relative to this VLM judge's per-pair noise at n=90, to independently
clear the pre-registered promotion bar (§4.7's MDE for this judge/n is ~0.037, well above the
observed +0.0078 effect). Training a new Pattachitra adapter and evaluating it with the same
judge/n design risks producing a second verdict with the identical power problem — not because
curation doesn't work, but because this specific measurement setup can't yet confirm gains at the
magnitude curation realistically produces. Do not proceed to a training run until the judge
reliability/power questions in §4.6–§4.8 are resolved (either a less coarse judge, a larger n
design, or an accepted, explicitly-stated power limitation applied consistently across both LoRA
projects).

**Decision: 111 ≥ 50 → PROCEED to spec-approved training.** Kalamkari fallback not needed. This
pre-check does not itself authorize training spend — §5's cost estimate and GG's explicit
approval are still required before any GCP run.

---

## 4. Dataset-sourcing plan (for Pattachitra, primary recommendation)

1. **Commons harvest:** pull all ~140 files across `Category:Pattachitra`, `Category:Pattachitra
   in Odisha`, `Category:Pattachitra in West Bengal` via the Commons API (`commons.wikimedia.org/
   w/api.php`, `list=categorymembers`), with **per-file licence verification** (not a blanket
   assumption) — record each file's licence tag (CC0/CC-BY/CC-BY-SA/PD) individually, the same
   rigor the halo-effect and A/B work in this project applies to numeric claims.
2. **Museum open-access follow-up (before training, not after):** run the Met Museum Open Access
   API (`metmuseum.org/api`) and V&A's collections API (`api.vam.ac.uk`) for "Pattachitra"/
   "Odisha painting"/"Orissa painting" directly — this spec surfaced Met/V&A holdings for other
   candidates via web search, not their own APIs, so an authoritative in-repo check (a small
   script, `scripts/_check_museum_apis.py`-style, matching this project's `scripts/_*.py`
   diagnostic-script convention) should confirm actual Pattachitra-specific holdings and their
   reuse terms before the corpus is finalized.
3. **VLM-judge curation pass**, reusing `scripts/curate_ukiyo_e_dataset.py`'s exact pattern
   (local Ollama `qwen2.5vl:7b`, zero cost): screen for the artifact classes flagged as likely in
   Pattachitra source photography — inscriptions/labels in museum photographs (a different
   contamination risk than ukiyo-e's in-print calligraphy, since Pattachitra pieces themselves
   don't conventionally carry embedded script the way ukiyo-e cartouches do, but photographed
   museum specimens often have visible accession labels, glare, or partial cropping), and any
   images that are photos-of-artists-working rather than the artwork itself (several Commons
   files in this category are process/documentary photos, not artwork reproductions — these
   should be filtered before training, the same class of filtering the curation step already does).
4. **Training config:** same as the curated ukiyo-e retrain — rank-8 LoRA, SDXL base +
   `madebyollin/sdxl-vae-fp16-fix`, `scripts/_diffusers_train_text_to_image_lora_sdxl.py`,
   1024px, seed 42, `--validation_epochs 15` (not `1` — this was the fix already logged for the
   original ukiyo-e run's validation-overrun cost issue).

---

## 5. Estimated GCP cost, against the ukiyo-e benchmark

The ukiyo-e curated retrain (1500 steps, rank-8, `g2-standard-4`/L4, ~$0.70/hr on-demand) is
externally cited in this project's context as ~$3.85–4.35 for the training run itself. Applying
the same recipe to Pattachitra:

| Line item | Ukiyo-e actual/estimated | Pattachitra estimate | Note |
|---|---|---|---|
| Training compute (1500 steps, rank-8, L4) | ~$3.85–4.35 | **~$3.50–5.00** | Same recipe, same step count; range reflects this session's own repeated finding that L4 capacity/stockouts (`docs/MODEL_VERDICT.md` §4.3's provenance note; this session's halo-effect check also hit stockouts) can force multiple VM attempts, each adding idle setup time before training starts. |
| Curation pass (VLM judge) | $0 (local Ollama) | $0 | No change — this step never touches paid infra. |
| A/B eval (if a pre-registered A/B is run, matching the ukiyo-e precedent) | not separately itemized in the cited figure | **~$1.50–2.50** | Based on this session's own powered A/B (180 generations + 180 VLM-judge calls, `g2-standard-8`/L4, ~1–1.5hr wall-clock at $0.70–$1/hr-equivalent, including one OOM-driven machine-type upgrade). |
| **Total, training + eval** | ~$3.85–4.35 (training only, as cited) | **~$5.00–7.50** | The higher end reflects the now well-documented L4 stockout risk (`PLAN.md`'s teardown-guardrail entry) — budget for at least one failed provisioning attempt per run. |

**No training authorized by this document.** GG must approve Pattachitra (or the Kalamkari
fallback) before any GCP spend or dataset harvest begins.
