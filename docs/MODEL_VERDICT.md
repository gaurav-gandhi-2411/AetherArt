# AetherArt Model Verdict

**Objective:** are AetherArt's published HF models
([aetherart-ukiyo-sdxl](https://huggingface.co/gauravgandhi2411/aetherart-ukiyo-sdxl),
[aetherart-ukiyo-sd21](https://huggingface.co/gauravgandhi2411/aetherart-ukiyo-sd21)) good? If
no, what needs fixing first. If yes, does the lineup clear the bar to train and publish
additional models under the same recipe. Every number below traces to a file in `reports/` —
none is estimated or carried over from a superseded report.

---

## 1. Method

All 5 commercially-usable, non-LoRA model families were scored on the **identical** 30-prompt
PartiPrompts benchmark (`scripts/eval_prompts.yaml`, IDs `pp_001`–`pp_030`) × 3 seeds
(`42, 43, 44`) = 90 records/family, via `scripts/model_verdict_harness.py`. Zero generation
errors across all 450 records. SDXL Turbo is excluded from all evaluation (non-commercial ADD
license, legacy-gated behind `AETHERART_ENABLE_LEGACY=1` — not a product candidate).

Metrics: CLIP (prompt-image alignment) + HPS (Human Preference Score v2, a learned aesthetic/
preference model). ImageReward is excluded — confirmed broken in this environment
(`ImportError: apply_chunking_to_forward` removed from the installed `transformers`; not
worked around).

The Ukiyo-e domain LoRA is scored differently (§4) — CLIP is invalid for it, per its own model
card's documented CLIP-blindness finding (§3.4).

---

## 2. Per-family results (5 non-LoRA families, n=90 each)

| Family | CLIP (mean±SEM) | HPS (mean±SEM) | Latency (mean) | Source |
|---|---|---|---|---|
| `sd21_base` | 0.3167 ± 0.0036 | 0.2528 ± 0.0042 | 14.3s | `reports/verdict_sd21_base.json` |
| `sdxl_base` | 0.3280 ± 0.0034 | 0.2876 ± 0.0034 | 19.0s | `reports/verdict_sdxl_base.json` |
| `hyper_4step` | 0.3269 ± 0.0037 | 0.3138 ± 0.0034 | 6.7s | `reports/verdict_hyper_4step.json` |
| `hyper_8step` | 0.3136 ± 0.0042 | 0.2369 ± 0.0044 | 10.4s | `reports/verdict_hyper_8step.json` |
| `sdxl_controlnet_union` | 0.3281 ± 0.0031 | 0.2802 ± 0.0033 | 546.6s | `reports/verdict_sdxl_controlnet_union.json` |

All 6 aggregate numbers above were independently recomputed from the raw JSON (mean/SEM over
the 90 CLIP/HPS values in each file, latency over all 90 `latency_s` values) — see the
one-liner reproduction: `python -c "import json,statistics as st; d=json.load(open(f)); ..."`
against each `reports/verdict_*.json`. No number here was taken from a prior report on faith.

**Valid comparisons only:** all 5 rows share the identical prompt set, seed set, and metric
definitions, so cross-family CLIP/HPS comparisons in this table are valid. `sdxl_controlnet_union`
and the LoRA families (§4) reuse `sdxl_base`'s own saved output images (same prompt+seed) as
their conditioning/reference source, not independently generated — see harness docstring.

**`sdxl_controlnet_union`'s 546.6s latency is a hardware finding, not a quality one.**
`enable_model_cpu_offload()` is required because SDXL+ControlNet-Union+VAE exceeds this
machine's 8GB VRAM; 3 of the 90 records were further inflated by `llama-server.exe` contention
from other concurrent CC sessions on the same GPU (not a defect in the pipeline itself — see
`docs/LATENCY_ROOT_CAUSE.md` for the general contention mechanism). CLIP/HPS quality numbers are
unaffected by this — they are computed from the generated image content, not the wall-clock.

---

## 3. Hyper-SD 4-step vs. 8-step inversion — root-caused, not a config bug

**Observation:** `hyper_4step` (HPS 0.3138) outperforms `hyper_8step` (HPS 0.2369).

**Framing correction — this is not a clean step-count comparison.** `hyper_4step` runs at its
documented default `guidance_scale=0.0` (CFG-free); `hyper_8step` runs at its documented default
`guidance_scale=5.0` (CFG-preserving) — these are two different LoRA checkpoints from
`ByteDance/Hyper-SD`, each with its own recommended guidance, not the same checkpoint run at two
step counts. The valid, precise claim is: **"at their respective in-spec defaults,
`hyper_4step` outperforms `hyper_8step`"** — guidance scale and step count are confounded here,
so this result does not by itself say anything general about "more steps is worse" for few-step
distilled models. The operational recommendation (route production traffic to `hyper_4step`) is
unaffected by this framing correction — it's the correct decision either way, since it's a
between-checkpoints comparison at each one's own best-known settings, which is exactly what a
production routing decision needs.

### 3.1 Config-routing hypothesis: falsified

The first hypothesis was that `scripts/model_verdict_harness.py` applied one uniform
guidance/scheduler config to both variants instead of routing through `aetherart/hyper.py`'s
per-LoRA defaults. Confirmed false by direct code inspection:

- `aetherart/hyper.py:17-30` (`HYPER_DEFAULTS`) defines distinct configs: 4-step uses
  `guidance_scale=0.0`, 8-step uses `guidance_scale=5.0` (both also swap to
  `EulerDiscreteScheduler(timestep_spacing="trailing")`, `aetherart/hyper.py:54-61`).
- `scripts/model_verdict_harness.py:332-359` (`build_hyper`/`gen_hyper`) reads
  `HYPER_DEFAULTS[variant]` per-variant and correctly passes the matching
  `num_inference_steps`/`guidance_scale` to each generation call — no uniform override.
- `aetherart/config.py:43-48` (`hyper_sd_weights`) confirms the two variants load **distinct**
  LoRA weight files (`Hyper-SDXL-4steps-lora.safetensors` vs. `Hyper-SDXL-8steps-lora.safetensors`),
  ruling out an accidental same-file load.

The harness correctly routes both variants through their documented per-LoRA defaults. This is
a real quality finding, not a plumbing bug.

### 3.2 Real root cause: CFG-driven exposure collapse on the 8-step (CFG-preserving) variant

Empirical breakdown of `reports/verdict_hyper_8step.json`'s 90 records: **16% (14/90)** have
mean image brightness < 40/255, and **7% (6/90)** are near-total black-crush (mean < 25/255).
`reports/verdict_hyper_4step.json` has **0/90** in either bucket. Visual confirmation
(`pp_020`, "a professional photo of a sunset behind the grand canyon," seed 42): the 4-step
output is a normally-exposed canyon photo; the matched 8-step output is a solid-black frame with
only a sliver of sky/sun visible — zero canyon detail, i.e. the prompt's described content is
not present. This is a genuine generation defect, not a subtle aesthetic/HPS scoring bias.

**Mechanism, directly confirmed by manipulating `guidance_scale`** on the identical
prompt/seed/scheduler (`pp_020`, seed 42, 8 steps, `EulerDiscreteScheduler` trailing spacing):

| guidance_scale | image mean brightness | note |
|---|---|---|
| 1.5 | 69.56 | fully exposed, canyon detail intact (off-label — see below) |
| 3.0 | 39.79 | partially recovered (off-label) |
| **5.0 (AetherArt's shipped default)** | **20.98** | matches the harness's own recorded value exactly |
| 6.0 | 15.05 | worse |
| 8.0 | 13.36 | worst |

Brightness decreases monotonically as `guidance_scale` increases — this is a CFG-magnitude-driven
exposure collapse, confirmed by direct experiment, not inferred. Correlation across the full
4-step+8-step combined dataset (n=180): `corr(brightness, hps_score) = +0.377`; across all 4
scored SDXL-based families (n=360): `+0.264`. Both directions are consistent with the mechanism
(darker images score lower on HPS), though the correlation magnitude alone does not fully explain
the aggregate HPS gap — even the 76 *non-dark* 8-step records average HPS 0.2389, still well
below 4-step's 0.3138, so there is a broader stylistic (contrast/exposure-profile) penalty beyond
just the outright black-crush failures.

Values below 5.0 do resolve the exposure collapse on the one case tested, but operating there is
explicitly off-label for this LoRA checkpoint (the Hyper-SD model card, `ByteDance/Hyper-SD`,
documents the CFG-preserving 8-step SDXL LoRA as supporting a "5~8 guidance scale" range —
AetherArt's shipped default, 5.0, is already the floor of that range, and is empirically the
least-bad option within it: 6.0 and 8.0 both measured darker/worse, above).

**Second hypothesis tested: `timestep_spacing` on `EulerDiscreteScheduler`.** For distilled
few-step models (Hyper-SD/LCM/Lightning-family), `timestep_spacing` is a known-sensitive
scheduler setting, and switching it is a genuinely different, in-spec-guidance-scale (5.0 held
fixed) lever from the guidance sweep above — so it was tested as an independent candidate fix,
not assumed away. `aetherart/hyper.py:58-61` currently sets `timestep_spacing="trailing"`.
Swept `"trailing"` (current) vs. `"leading"` vs. `"linspace"` at guidance_scale=5.0, 8 steps, on
the two worst black-crush cases (`pp_020` seed 42, `pp_021` "a cityscape at night with a full
moon" seed 42) plus a non-dark control (`pp_001` seed 42):

| `timestep_spacing` | pp_020 mean | pp_021 mean | pp_001 (control) mean | outcome |
|---|---|---|---|---|
| `trailing` (current) | 20.98 | 11.43 | 49.87 | underexposed, but coherent |
| `leading` | 51.43 | 39.72 | 130.17 | **brighter, but generation is incoherent** — see below |
| `linspace` | 17.15 | 9.47 | 45.24 | no improvement (slightly worse) |

`leading` numerically resolves the brightness problem, but visual inspection shows why that
number is misleading: at `leading` spacing, `pp_020` ("a professional photo of a sunset behind
the grand canyon") renders as a glitchy, posterized abstract color field with no canyon, no
sunset, and no coherent geometry, and `pp_001` ("artificial intelligence") renders as a
repeating tiled grid of small circuit-icon glyphs instead of a single coherent subject — both
are content-incoherent, defect-riddled generations, a **worse** failure mode than underexposure
(source images: `outputs/verdict/_diag/spacing_leading_{pp_020,pp_001}_42.png`). This is
consistent with `trailing` spacing being load-bearing for how these few-step-distilled LoRAs
were trained to map a very short (4–8 step) noise schedule — swapping it desyncs that mapping
entirely rather than merely shifting exposure. `linspace` doesn't move brightness meaningfully
in either direction and doesn't introduce this coherence failure, but doesn't help either.

**No in-spec fix exists — now earned by direct experiment, not just the guidance sweep.** Both
independent candidate fixes (guidance retune within the documented range, and scheduler
`timestep_spacing` swap) were tested and exhausted: guidance retuning within 5–8 only makes
exposure worse; `timestep_spacing` swap either doesn't help (`linspace`) or trades underexposure
for a worse, generation-breaking defect (`leading`). This was a genuine solve-it pass on two
independent mechanisms, not a single guidance sweep generalized into "no fix exists."

### 3.3 Verdict on this finding

`hyper_8step`, run at AetherArt's current (spec-compliant, least-bad-in-range) configuration, has
a measured 16% severe-underexposure rate on a diverse 30-prompt set, and underperforms
`hyper_4step` on every axis that was measured: HPS (0.2369 vs. 0.3138), CLIP (0.3136 vs. 0.3269),
and latency (10.4s vs. 6.7s). **Recommendation: route production traffic to `hyper_4step`, not
`hyper_8step`, for this SDXL checkpoint.** `hyper_8step` should not be presented as a
"higher-quality, slower" option to end users — the data shows it is both slower and lower
quality on this pipeline.

### 3.4 Why CLIP is invalid for the Ukiyo-e LoRA (context for §4)

Both published HF model cards document a CLIP-blindness finding specific to the Ukiyo-e
style-transfer task: across 9 controlled experiments, CLIP delta was <1 SEM while LPIPS-vs-base
ranged 0.40–0.73, and smaller/underfit adapters scored *higher* on CLIP — the wrong direction for
a quality signal on this task. CLIP is recorded for context only in the LoRA family's records
(`clip_score_context_only`) and is explicitly **not** used to render the LoRA verdict below.

---

## 4. LoRA A/B: published checkpoint-1000 vs. curated retrain

**Endpoint framing (correction from an earlier draft of this section):** the general
30-prompt PartiPrompts set contains **no ukiyo-e trigger prompts** — every prompt is an
arbitrary subject with a style suffix appended (e.g. "an elephant using its trunk to blow into
a tuba, ukyowood, ukiyo-e woodblock print style"), not a prompt actually written to invoke the
ukiyo-e woodblock aesthetic on its own terms. The calligraphy/cartouche artifact the curated
retrain targeted is a stylistic hallmark of *authentic ukiyo-e prints being rendered as such*
(real ukiyo-e woodblock prints often carry title cartouches/seals) — it is not equally likely to
appear regardless of prompt. On this general set, `artifact_absence`'s worst case bottoms out at
0.7 for **both** checkpoints (§4.1) — a floor effect indicating the general set barely exercises
the defect in either arm. Treating a wash on this set as "no effect" would be a false negative
from measuring the wrong domain, not evidence the retrain doesn't work. Corrected framing,
applied for the rest of this section and pre-registered before §4.3's larger run:

- **Primary endpoint:** `artifact_absence`, measured on ukiyo-e-**styled** prompts (trigger
  token present, prompt written to invoke the woodblock-print aesthetic directly) — this is
  where the artifact is actually expressible, and where a real fix should show up.
- **Guardrails (non-inferiority):** `style_adherence` and `figure_preservation`, measured on the
  general 30-prompt PartiPrompts set — confirms the retrain didn't regress general style-lift or
  figure quality off-domain. This set is *not* the primary artifact-absence endpoint; it is
  reported as a guardrail below (§4.1), not a null result on the actual question.

Both checkpoints scored on the identical 30-prompt × 3-seed set (n=90 each, 0 errors) via
`scripts/model_verdict_harness.py --family ukiyo_e_lora_sdxl` (published) and `--family
ukiyo_e_lora_sdxl_curated` (curated retrain,
`data/lora/ukiyo-e/training_output_sdxl_curated/checkpoint-1000/`), scored on:

- LPIPS distance from the matched `sdxl_base` output (same prompt+seed, no LoRA)
- Local VLM judge (Ollama `qwen2.5vl:7b`) 0–1 rubric: `style_adherence`, `figure_preservation`,
  `artifact_absence`

**Harness bug found and fixed before this run:** `run_ukiyo_e_lora_family` originally called the
VLM judge inline, interleaved with generation, while the SDXL+LoRA pipeline was still
GPU-resident. This reproduces the exact VRAM-oversubscription pathology documented in
`docs/LATENCY_ROOT_CAUSE.md` (measured here before the fix: image 1 at 44s, image 2 escalating
past 7 minutes for 30 steps). Fixed to defer VLM scoring to a second phase, after releasing the
generation pipeline — the same two-phase pattern `run_generation_family` already used for HPS.
All numbers below are post-fix data.

### 4.1 Guardrail check — off-domain PartiPrompts set (n=90 each), not the primary endpoint

| Metric | Published (mean±SEM) | Curated (mean±SEM) | Curated − Published |
|---|---|---|---|
| LPIPS-vs-`sdxl_base` | 0.6001 ± 0.0085 | 0.6055 ± 0.0086 | +0.0054 ± 0.0058 |
| `style_adherence` | 0.8217 ± 0.0049 | 0.8222 ± 0.0086 | +0.0006 ± 0.0064 |
| `figure_preservation` | 0.9600 ± 0.0112 | 0.9633 ± 0.0071 | +0.0033 ± 0.0093 |
| `artifact_absence` | 0.9339 ± 0.0113 | 0.9467 ± 0.0105 | +0.0128 ± 0.0131 |

Source: `reports/verdict_ukiyo_e_lora_sdxl.json`, `reports/verdict_ukiyo_e_lora_sdxl_curated.json`.

**Methodology note:** the "Curated − Published" diff column is a **paired difference**
(curated minus published, computed per matching `prompt_id`+`seed` — both files iterate the
identical 30×3 prompt/seed grid in identical order, so record *i* in each file is the same
prompt+seed), with SEM computed on the 90 per-pair differences directly, not the independent-
sample quadrature combination `sqrt(SEM₁²+SEM₂²)`. Paired is the statistically correct and more
powerful choice for this matched design (it removes prompt-difficulty variance common to both
checkpoints) and is used consistently in §4.2's targeted check below — an initial draft of this
document used the two methods inconsistently between §4.1 and §4.2 without stating either;
this was caught by an independent verifier pass and corrected by making the method explicit and
uniform (see also §4.2, whose conclusion changed once the same paired method was applied there).

**Every diff above is smaller than its own paired SEM (largest: `artifact_absence` at 0.97 SEM,
LPIPS at 0.92 SEM) — none clears this project's own established significance bar** (the 2×SEM
threshold `scripts/check_eval_gate.py` uses for the CI regression gate, PR #20). **Guardrail
result: PASS.** `style_adherence` and `figure_preservation` show no regression off-domain — both
diffs are near zero, well inside noise. `artifact_absence` is also reported here for
transparency, but per the endpoint framing above, this set is not where that metric's primary
claim is decided (see the floor-effect note in §4.2) — its wash-level diff here is not evidence
either way on the actual promotion question.

### 4.2 Preliminary evidence that motivated a properly-powered primary run (§4.3)

The curated retrain specifically targeted the calligraphy/cartouche artifact documented on
WikiArt source portrait images (57/80 = 71.2% of source images flagged and excluded,
`reports/ukiyo_e_curation_report.json`). Checking the worst-case `artifact_absence` records in
both datasets: the floor score is **0.7 in both** (`pp_001`, `pp_007`, etc. — none of which are
portrait/figure subjects; e.g. "an elephant using its trunk to blow into a tuba," "a bowl of
Pho"). The generic 30-prompt PartiPrompts set (arbitrary subjects with a style suffix appended)
does not appear to strongly trigger the specific portrait-cartouche failure mode the retrain
targeted — both checkpoints already score high (>0.93 mean) on it.

**Supplementary targeted check** (`scripts/_lora_ab_targeted.py`, one-off diagnostic, not wired
into the main harness): the 4 dedicated ukiyo-e-style prompts already in `scripts/eval_prompts.yaml`
(`lora_001`–`004`: Mount Fuji, crane over waves, samurai in bamboo forest, cherry blossoms — no
`sdxl_base` LPIPS reference exists for these, so `artifact_absence` only), n=12 seeds×prompts
per checkpoint:

| Checkpoint | `artifact_absence` (mean±SEM, independent) |
|---|---|
| Published | 0.9167 ± 0.0271 |
| Curated | 0.9750 ± 0.0179 |

**Paired diff** (same method as §4.1, curated minus published per matching `prompt_id`+`seed`,
n=12 pairs): **+0.0583 ± 0.0288, diff/SEM = 2.03** — this *does* clear the project's 2×SEM bar.
(An independent-sample quadrature combination gives combined SEM 0.0325 and diff/SEM ≈ 1.80,
which does not clear it — the paired method is the correct one here per §4.1's methodology note,
and is used for the conclusion below.) Visual inspection of the lowest-scoring matched pair
(`lora_001`, seed 42 — published scored 0.8, curated scored 1.0) shows a qualitative difference
consistent with the quantitative direction: the published output has a distinct rectangular
cartouche with legible vertical script in the top-right corner; the matched curated output has
only a small, non-legible seal-like mark in roughly the same region — source images at
`outputs/verdict/lora_ab_targeted/{published,curated}_lora_001_seed42.png`.

**Caveat on this result's fragility:** n=12 paired samples is a small sample crossing the 2×SEM
bar by a narrow margin (2.03 vs. the 2.00 threshold) — a single differing observation could flip
this back under the bar. This is a real, paired, correctly-computed result, not an artifact, but
it should be read as "borderline significant on a small sample, motivating a properly-powered
follow-up (§4.3)," not as a standalone conclusion. The 4-prompt set is also thin on figure/portrait
coverage — only `lora_003` ("a samurai in a bamboo forest") is a figure subject, and it already
scored a perfect 1.0/1.0/1.0 for both checkpoints, so it doesn't discriminate at all. §4.3 was
built specifically to fix both gaps (n and figure-subject coverage).

### 4.3 Primary result (pre-registered) — 30 ukiyo-e-styled prompts × 3 seeds, n=90 paired

**Pre-registration** (committed in `docs/AB_PREREGISTRATION.md` before this run executed):

- **Primary endpoint:** `artifact_absence`, paired diff (curated − published), on the 30-prompt
  ukiyo-e-styled set below.
- **Promotion threshold:** paired diff / paired SEM > 2.0 (this project's established
  significance bar, §4's methodology note and PR #20's CI gate).
- **Guardrails (non-inferiority):** `style_adherence`, `figure_preservation` — promotion
  requires the primary to clear AND neither guardrail's paired diff to show a *regression*
  (curated worse than published) clearing the same 2×SEM bar in the negative direction.
- **Decision rule, fixed before seeing results:** promote only if primary clears AND both
  guardrails are non-inferior. A null or negative primary result is reported as such — it is not
  grounds to keep looking for a threshold that passes.

*(Results filled in after the pre-registered run completes — see `docs/AB_PREREGISTRATION.md`
for the full 30-prompt set and generation/scoring methodology.)*

### 4.4 Verdict

**PENDING §4.3.** The pre-registered decision rule (§4.3) determines promotion; do not render a
verdict here until that run completes — the preliminary n=12 check (§4.2) is suggestive but was
explicitly built to be superseded, not relied on for the final call. The §4.1 guardrail result
(no regression on `style_adherence`/`figure_preservation` off-domain) already holds regardless of
how §4.3 resolves.

---

## 5. Per-family verdict

| Family | Verdict | Basis |
|---|---|---|
| `sd21_base` | production-quality | Baseline family; no defects found in this eval pass. |
| `sdxl_base` | production-quality | Highest CLIP among base families; no defects found. |
| `hyper_4step` | production-quality — **recommended fast path** | Best HPS+CLIP+latency of all 5 families. |
| `hyper_8step` | needs-improvement | 16% severe-underexposure rate at spec-compliant config; strictly dominated by `hyper_4step` on every measured axis. Not recommended for production traffic; keep `hyper_4step` as the routed default. |
| `sdxl_controlnet_union` | production-quality (quality) / needs-improvement (latency) | Quality on par with `sdxl_base`; 546.6s latency requires CPU offload on this 8GB card — a hardware-fit issue, not a model-quality one. |
| Ukiyo-e LoRA — published checkpoint-1000 | production-quality (current champion) | No regression vs. curated retrain found on any axis; remains the published, shipped artifact. |
| Ukiyo-e LoRA — curated retrain | **PENDING §4.3** | Guardrails pass (no regression off-domain, §4.1); preliminary n=12 targeted check suggestive (2.03 SEM, §4.2) but explicitly superseded by the pre-registered n=90 primary run (§4.3) — verdict renders once that completes. |

---

## 6. Final routing decision

**Does the current lineup clear the bar to train and publish a new model, or does something
need fixing first? Something needs fixing/clarifying first — not a hard blocker, but not a clean
green light either:**

1. **`hyper_8step` should not be routed to production as currently configured** (§3) — it is
   strictly dominated by `hyper_4step` on HPS, CLIP, and latency, and has a 16%
   severe-underexposure defect rate. Action: change the default/recommended Hyper-SD variant in
   any user-facing routing logic to `hyper_4step`. This is a config/docs fix, not a retrain.
2. **The Ukiyo-e curated retrain's promotion decision is PENDING §4.3's pre-registered run** —
   guardrails already pass (§4.1: no off-domain regression), and a preliminary n=12 targeted
   check was suggestive (2.03 SEM, §4.2) but was explicitly built to be superseded rather than
   relied on. The pre-registered 30-prompt ukiyo-e-styled run (§4.3, primary endpoint
   `artifact_absence`, threshold 2×SEM paired, fixed before running) renders the actual call.
   Until then: treat the published checkpoint-1000 as the continued champion and the curated
   checkpoint as a promising, not-yet-confirmed experiment.
3. **All 5 non-LoRA families and the published Ukiyo-e LoRA are production-quality** on the
   metrics measured here, with `sdxl_controlnet_union`'s latency flagged as a hardware-fit
   constraint (8GB VRAM + CPU offload), not a quality defect.
4. **Training and publishing a genuinely new model under the current recipe is not recommended
   until (1) is fixed** — publishing a 6th family without first correcting a known, dominated,
   defect-prone configuration in the existing lineup would compound the same class of issue this
   whole audit was launched to catch. (2) does not block new-model work by itself (it's a
   decision about one existing artifact, not the pipeline), but should be resolved so the
   published Ukiyo-e model card accurately reflects which checkpoint is current.
