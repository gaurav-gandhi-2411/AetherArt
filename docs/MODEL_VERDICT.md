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

**Results** (`reports/lora_ab_30prompt.json`, n=90 each, 0 errors; generated on a GCP L4 instance
after local GPU contention with a concurrent unrelated job made this run impractical locally —
see the provenance note at the end of this section):

| Metric | Published (mean) | Curated (mean) | Paired diff | Paired SEM | diff/SEM |
|---|---|---|---|---|---|
| `artifact_absence` (**PRIMARY**) | 0.8833 | 0.9233 | **+0.0400** | 0.0126 | **3.182** |
| `style_adherence` (guardrail) | 0.8883 | 0.9022 | +0.0139 | 0.0032 | 4.391 |
| `figure_preservation` (guardrail) | 0.9689 | 0.9811 | +0.0122 | 0.0046 | 2.680 |

**Primary endpoint clears the pre-registered threshold decisively** (3.18 vs. the 2.0 bar — not
a borderline crossing like the n=12 preliminary check's 2.03). At face value both guardrails are
positive here too, with no evidence of a quality-for-artifact-absence tradeoff — **but see §4.5:
a halo-effect check on the VLM judge (which scores all three axes in one call) found the
guardrail improvements do not robustly survive independent single-axis scoring, so the claim
scope for `style_adherence`/`figure_preservation` is narrowed to "no regression," not
"improvement." The primary endpoint's result is unaffected and confirmed robust under the same
check.** This is a properly powered (n=90, not n=12), pre-registered, paired result — the
strongest evidence produced in this evaluation.

**Provenance note (GPU relocation):** this run could not complete locally in reasonable time
because a concurrent, unrelated job (`scripts/d2_train.py`, a different project) was monopolizing
this machine's single shared 8GB GPU. Per explicit direction, the run was moved to a GCP L4
instance (`aetherart-497918`, ultimately `us-west1-a` after `us-central1-a/b/c` and `us-east4-a`
all hit a regional L4 capacity stockout — consistent with this project's previously-documented
L4-stockout pattern). One real operational mistake occurred and is disclosed here rather than
smoothed over: the first VM (`g2-standard-4`, 16GB RAM) OOM-killed the generation process twice
(confirmed via `dmesg`: `Out of memory: Killed process ... python3`) because SDXL's CPU-offload
staging needs more system RAM than that machine type provides — fixed by moving to
`g2-standard-8` (32GB RAM). Separately, the original VM was deleted (to free capacity for a
retry) **before** its completed published-checkpoint results (90 generations + 90 VLM scores)
were pulled to local, and `gcloud compute instances delete` removes the boot disk by default —
that data was lost and had to be regenerated from scratch (same script, same seeds, so the
regenerated result is not a "second attempt" biased by anything except ordinary run-to-run
sampling noise). All GCP resources (2 instances across the session, one disk) were fully torn
down afterward — confirmed via `gcloud compute instances/disks/addresses list`, all empty.

### 4.4 Verdict: PROMOTE the curated retrain — **withdrawn, see §4.6**

**Correction (do not act on this section alone):** this verdict was computed from §4.3's
correlated single-call scoring regime. §4.6 below re-runs the identical paired-diff analysis on
the full n=90 set under the trusted independent-axis regime and finds the primary endpoint does
**not** clear the promotion bar (0.583×SEM, not 3.182×SEM). This section is preserved as written
for an honest record of what was originally concluded and why — see §4.6 for the corrected,
current verdict.

Per the pre-registered decision rule (§4.3): **promote if and only if the primary endpoint
clears the 2×SEM threshold AND neither guardrail regresses.** Both conditions are met — primary
clears at 3.18×SEM, and neither guardrail regresses (§4.5 narrows "guardrails also improved" to
"guardrails did not regress" after a halo-effect check — see that section; it does not affect
this decision rule). **The curated retrain is promoted over the published checkpoint-1000.**
This is not a borderline or
"promising but unconfirmed" call like the preliminary n=12 check produced — it is a clean result
from the properly powered, pre-registered design this project's own methodology required before
rendering a final verdict.

This also resolves §4.2's fragility concern: the n=12 preliminary check's directionally-correct
but marginal signal (2.03×SEM) is now confirmed by a 90-pair result with a much larger margin
(3.18×SEM) and a materially larger effect size (+0.040 vs. the general benchmark's washed +0.013)
on prompts that actually exercise the calligraphy-artifact defect. The §4.1 guardrail result (no
regression on the general benchmark) still holds independently.

**§4.5 below narrows the guardrail claim** — see that section before repeating "both guardrails
improve" anywhere outside this document. The primary endpoint's promotion decision is unaffected.

### 4.5 Halo-effect check on the VLM judge — claim scope correction for the guardrails

**Why this check exists:** the judge that scored §4.3 (`scripts/_lora_ab_30prompt.py`'s
`VLM_JUDGE_PROMPT`) asks for all three axes — `style_adherence`, `figure_preservation`,
`artifact_absence` — in a **single Ollama call per image**. A single call scoring three related
axes at once is a known setup for halo-biased ratings: the model can anchor all three numbers to
one overall "this looks good/bad" impression rather than judging each axis on its own terms. All
three axes improving together in §4.3 is consistent with either (a) three genuinely independent
effects, or (b) the primary endpoint's real improvement bleeding into the judge's assessment of
the other two axes. This was checked directly rather than assumed either way.

**Method** (`scripts/_halo_effect_check.py`, `reports/halo_effect_check.json`): a deterministic
random 30-image subsample of the 180-record §4.3 dataset (`random.seed(42)`, `random.sample`;
18 published / 12 curated) was regenerated (same checkpoint/prompt/seed, so the images reproduce)
and re-scored with **three independent Ollama calls per image, one per axis**, each prompt
naming only that one axis with no mention of the other two. 0 errors, run on GCP (same L4
relocation as §4.3's provenance note) and torn down after — see PLAN.md's teardown-guardrail entry.

**Result 1 — inter-axis correlation, both regimes (n=30):**

| Axis pair | Single-call (original) | Independent-call (new) |
|---|---|---|
| `style_adherence` × `figure_preservation` | +0.156 | −0.043 |
| `style_adherence` × `artifact_absence` | +0.019 | −0.480 |
| `figure_preservation` × `artifact_absence` | +0.117 | +0.238 |

**This result is inconclusive on its own — not a clean "holds" or "drops materially."** At n=30,
a Pearson correlation's standard error is roughly ±0.19-0.2; every value above is inside or near
that noise band around the others. One pair's correlation modestly falls (style×figure), one
swings from near-zero to moderately negative (style×artifact — not a drop from a *high* baseline,
since the single-call value was already ~0), and one *rises* (figure×artifact) — the opposite of
what a pure halo-effect story predicts. This pattern is consistent with sampling noise at a small
n, not a clear signature either way.

**Result 2 — the actually decision-relevant check: does each axis's curated-vs-published
difference survive independent scoring, on this same subsample (unpaired, n=18 published /
n=12 curated — smaller and noisier than §4.3's n=90 paired, but same direction test):**

| Axis | Single-call diff (curated − published) | Independent-call diff |
|---|---|---|
| `artifact_absence` (**primary**) | +0.0847 | **+0.0611** |
| `style_adherence` (guardrail) | +0.0139 | **+0.0083** (≈40% smaller) |
| `figure_preservation` (guardrail) | +0.0000 | **−0.0028** (≈zero either way) |

**Conclusion — claim scope, not verdict:**
- **The primary endpoint's improvement is robust.** `artifact_absence` stays clearly positive
  and substantively large under independent scoring — this is the result the promotion decision
  (§4.4) rests on, and it holds.
- **The guardrail "improvement" claims do not robustly survive independent scoring.**
  `style_adherence`'s positive diff shrinks by roughly 40% under independent scoring; `figure_
  preservation` shows essentially no difference under either regime on this subsample. Neither
  is strong evidence of a halo effect specifically (the correlation data is too noisy to confirm
  that mechanism cleanly), but neither is it evidence the guardrails genuinely *improved* —
  it is exactly the ambiguous outcome the pre-registration anticipated needing a claim-scope
  decision for.
- **Per the pre-registered fallback rule: claim only non-inferiority on the guardrails, not
  improvement.** `docs/MODEL_VERDICT.md` and any model card language must state
  `style_adherence` and `figure_preservation` as **not regressed** (which both the §4.1 general
  benchmark and this check's own numbers support — no diff here is negative enough to suggest
  real regression) rather than "improved." Only `artifact_absence` (+0.040, 3.18×SEM, §4.3) is
  claimed as a measured improvement.
- **This does not change the promotion decision.** §4.4's verdict rests entirely on the primary
  endpoint clearing its pre-registered threshold with no guardrail *regression* — both of which
  still hold. It changes what can be said about the guardrails: "did not get worse," not "got
  better."

### 4.6 Full n=90 paired independent-regime rescore — TRUSTED result, supersedes §4.3/§4.4/§4.5

**This section is now the primary evidence for the LoRA A/B. §4.3's headline number was
computed under single-call multi-axis scoring, the regime §4.5 flagged as a halo-effect risk.
§4.5 itself only checked a small (n=30, unpaired within-axis) subsample and concluded the primary
endpoint "held." That conclusion does not survive the full, properly-paired, n=90 independent
rescore below — it is corrected here, not quietly dropped.**

**Method** (`scripts/_lora_ab_30prompt_independent.py`, `reports/lora_ab_30prompt_independent.json`,
stats computed by `scripts/compute_lora_ab_independent_stats.py`): all 180 images from §4.3's
dataset (90 published + 90 curated) were regenerated deterministically (same checkpoint/prompt/
seed) and every one rescored with the harness's independent single-axis judge (three separate
Ollama calls per image, `scripts/model_verdict_harness.py`'s `SINGLE_AXIS_JUDGE_PROMPTS` — the
same prompts §4.5 used, now applied to the full set instead of a 30-image subsample). 0 errors
across all 180 regenerations and 540 independent VLM calls.

**A systemic bug was found and fixed mid-run, disclosed here rather than smoothed over:** Ollama's
default served context window (4096 tokens) was too small for a judge prompt plus a
high-resolution image's token count on some inputs, causing `exceed_context_size_error` HTTP 400s.
Root-caused via the literal Ollama error body (not assumed), fixed via `num_ctx=8192` in
`_ollama_generate_json`.

**Provenance audit — which runs executed on the buggy harness vs. the fixed one, checked
directly rather than assumed** (a silent judge failure biases toward null, i.e. toward exactly
the withdrawal this section makes, so this needed to be ruled out before treating the withdrawal
as settled):

| Dataset | Committed | Harness fix (commit `a6a8220`, 2026-07-24 04:03:26) | VLM scoring executed | Degenerate-value audit |
|---|---|---|---|---|
| `reports/lora_ab_30prompt.json` (§4.3 correlated-regime headline) | 2026-07-23 20:49:36 | Different script (`_lora_ab_30prompt.py`), never had `num_ctx` at all | Before the fix existed | 0/180 null `vlm_judge` records (checked directly) |
| `reports/halo_effect_check.json` (§4.5, n=30 unpaired subsample) | 2026-07-24 01:16:55 | `_halo_effect_check.py` **lacks** `num_ctx` — theoretically vulnerable | Before the fix existed | 0/30 records have a `None` axis value (checked directly) — empirically clean despite the theoretical exposure |
| `reports/lora_ab_30prompt_independent.json` (this section's n=90 rescore) | 2026-07-24 09:15:17 | Harness fix already on disk (committed 04:03:26) | **05:29–05:43** (published) and **08:52–09:06** (curated) same day — both hours after the fix | 0/180 error flags, 0/180 `None`/non-numeric/out-of-range axis values, 0 logged VLM call failures in either run's log (only two benign `diffusers` LoRA-prefix warnings per run, unrelated to scoring), 0 exact-`0.0` scores (the shape a silent-default sentinel would take if the code clamped on failure — it doesn't; a read of `score_vlm_judge` confirms any axis exception returns `None` for the whole record, never a defaulted number) |

**Conclusion: the n=90 rescore that withdrew the promotion verdict ran entirely on the fixed
harness, with a directly-verified zero silent-failure rate.** The withdrawal is not an artifact
of the bug this session found — if anything, the bug being real and now-fixed is *why* this
rescore is trustworthy where an unaudited one might not be. The n=30 halo-check subsample (§4.5)
ran on the un-fixed script but happened not to trigger the bug in practice (verified, not
assumed) — its divergence from this section's finding is explained by its unpaired/small-n design
(below), not by judge corruption.

**Results — paired diff (curated − published), n=90 matched prompt_id+seed pairs, same paired-diff
methodology as §4.1/§4.3 (SEM on the 90 per-pair differences directly):**

| Metric | Published (mean) | Curated (mean) | Paired diff | Paired SEM | diff/SEM |
|---|---|---|---|---|---|
| `artifact_absence` (**PRIMARY**) | 0.8722 | 0.8800 | **+0.0078** | 0.0133 | **0.583** |
| `style_adherence` (guardrail) | 0.9444 | 0.9489 | +0.0044 | 0.0027 | 1.649 |
| `figure_preservation` (guardrail) | 0.9789 | 0.9867 | +0.0078 | 0.0032 | 2.394 |

**The primary endpoint does NOT clear the pre-registered 2×SEM promotion threshold under trusted
scoring** (0.583, versus §4.3's correlated-regime 3.182). The effect that looked decisive under
single-call scoring is consistent with a halo effect inflating it — §4.3's judge scored all three
axes from one overall impression per image, and the primary endpoint's apparent improvement likely
bled into (or was partly constituted by) that shared impression. A value-level check shows this
isn't a scoring artifact of a different kind: independent-regime `artifact_absence` scores are
similarly distributed between checkpoints (published mode 0.8, 55/90 at 0.8, 34/90 at 1.0;
curated mode 0.8, 51/90 at 0.8, 37/90 at 1.0). **§4.7 below shows this design is underpowered to
fully rule out a smaller true effect, so "indistinguishable" overstates the finding — the precise
claim is in §4.7, not here.**

**Why §4.5's n=30 check missed this:** §4.5's "Result 2" (independent-call diff +0.0611) was
computed on an **unpaired** subsample (18 published, 12 curated — different prompts in each group,
not matched prompt+seed pairs) at a small n. This project's own established methodology (§4.1's
methodology note) holds paired-diff as "the statistically correct and more powerful design for
matched observations" specifically because unpaired comparisons at small n are more exposed to
whichever prompts happen to land in each group. The full n=90 **paired** design used here is the
correct comparison and this document defers to it.

**Corrected verdict, per the pre-registered decision rule itself (§4.3: "a null or negative
primary result is reported as such — it is not grounds to keep looking for a threshold that
passes"): the curated retrain does NOT clear its own pre-registered promotion bar under the
trusted (independent-axis) scoring regime. §4.4's "PROMOTE" verdict is withdrawn. Do not re-upload
the curated checkpoint to HF Hub on the basis of this A/B.** This decision follows directly from
the 2×SEM promotion rule not being met — it does **not** require characterizing the true effect
as zero, and §4.7 shows that stronger characterization isn't actually supported by this data
(the design is underpowered below its own observed effect size). `docs/HF_MODEL_CARD_UPDATES.md`
is corrected to match (no "measured improvement" claim, and no "no effect" claim either — "did
not clear the pre-registered bar" is the precise, supportable statement). This does not mean the
curated dataset/retrain is *worse* — `figure_preservation` clears 2×SEM in the improving direction
here, and no guardrail shows regression — only that the specific claim this A/B set out to test (a
measured `artifact_absence` improvement) is not supported once the judge's known correlation bias
is removed. §4.8 tests three specific hypotheses for what's actually going on (arms are
functionally identical / curated arm lost style signal / artifact originates in the base model's
own prior) rather than stopping at "no effect found." A genuinely powered re-test of the
calligraphy-artifact hypothesis would need either a judge less prone to floor/ceiling clustering
at 0.8 (both checkpoints' independent-regime `artifact_absence` scores cluster heavily at two
values, 0.8 and 1.0 — a coarse, possibly discretized output that limits this judge's sensitivity)
or a larger n (§4.7 gives the actual n this would require).

**One honest caveat about this correction itself:** the independent-axis judge's `artifact_absence`
output is heavily concentrated on two values across all 180 independent-regime scores — 0.8
(106/180) and 1.0 (71/180), with only 3 scores elsewhere (two at 0.5, one at 0.9) — suggesting the
judge's output on this axis may itself be coarse/quantized rather than a finely-discriminating
0–1 score. This is reported as a limitation of the zero-cost VLM-judge method, not grounds to
discard the result — the trusted regime is still the regime that must be reported, and it does
not clear the pre-registered promotion bar. Whether that means "no real effect" or "an underpowered
design" is answered precisely in §4.7, not asserted here.

### 4.7 Power/sensitivity audit — is 0.583×SEM a demonstrated null, or underpowered?

**Do not conflate "does not clear the promotion bar" with "no effect exists" — these require
different evidence.** Computed via `scripts/compute_lora_ab_power.py`
(`reports/lora_ab_power_audit.json`), from the same n=90 paired `artifact_absence` differences
used in §4.6 (observed SEM = 0.0133):

| Quantity | Value |
|---|---|
| Observed paired diff | +0.0078 |
| 95% CI on the true diff | [−0.0184, +0.0339] |
| Minimum detectable effect (MDE) at 80% power | 0.0374 |
| MDE at 90% power | 0.0432 |
| Originally-claimed (correlated-regime, §4.3) effect | +0.0400 |

**Two separate, both-true conclusions:**
1. **The originally-claimed effect size (+0.040) is ruled out.** Its 95% CI upper bound (+0.0339)
   is below +0.040 — this design has enough precision to say the §4.3 correlated-regime number
   does not reflect the true effect under trusted scoring. This alone is sufficient grounds for
   §4.6's withdrawal decision; it does not depend on point 2 below.
2. **This design is underpowered to fully characterize a smaller true effect.** The observed
   effect (+0.0078) is well below this design's own 80%-power MDE (0.0374) — meaning an n=90
   paired design, given the per-image variance this judge actually produces, cannot reliably
   distinguish "no true effect" from "a true effect somewhere below ~0.037" at the magnitude
   observed here. **The correct characterization is "does not clear the 2×SEM promotion bar,
   and the originally-claimed effect size is ruled out" — not "no effect" or "the checkpoints are
   indistinguishable."** A true effect in the 0.01–0.03 range cannot be excluded by this data.

**What n would be needed to resolve the remaining ambiguity:** achieving an MDE of ~0.008
(order of the observed effect) at 80% power with this judge's observed per-pair diff stdev
(0.1265) requires `n = ((1.96+0.8416) × 0.1265 / 0.008)²` ≈ **1,963 paired samples** — roughly 22×
this study's n=90. That is not a "just add a bit more n" gap; closing it with this judge is
impractical at zero-cost-local scale. §4.8 pursues cheaper, more diagnostic routes instead of
just scaling n.

### 4.8 Root-causing the null — three specific hypotheses, tested in order

**"Curation doesn't help" is not accepted as a finding without first testing why the effect
disappeared.** Three concrete hypotheses, each independently falsifiable:

**(a) Are the two arms functionally the same model?** If the published and curated checkpoints
produce near-identical images, the artifact hypothesis was never actually exercised — the
"intervention" barely changed the output. Tested directly: LPIPS between the two arms' own
outputs (not each-vs-base), same 90 matched prompt+seed pairs, via
`scripts/_lpips_between_arms.py` (`reports/lpips_between_arms.json`):

| Comparison | LPIPS mean ± stdev | n |
|---|---|---|
| Published vs. curated (same prompt+seed) | 0.5504 ± 0.0685 | 90 |
| Published vs. `sdxl_base` (no LoRA), general benchmark (§4.1) | 0.6001 ± 0.0807 | 90 |
| Curated vs. `sdxl_base` (no LoRA), general benchmark (§4.1) | 0.6055 ± 0.0812 | 90 |

**Hypothesis (a) is REFUTED.** The two arms differ from each other (LPIPS 0.55) by almost as
much as either differs from the un-adapted base model (LPIPS 0.60–0.61) — the curated retrain
produced substantially different images, not a near-copy of the published checkpoint. Whatever
caused the artifact_absence null, it is not "the intervention didn't actually change anything."

**(b) Did the curated adapter (23 training images) lose general style signal relative to the
published adapter (80 images)?** and **(c) does the text/cartouche artifact originate in
`sdxl_base`'s own prior for "ukiyo-e"-styled prompts, independent of any LoRA?** — both tested by
one new generation batch: `scripts/_lora_ab_base_comparison.py`
(`reports/lora_ab_base_comparison.json`) regenerates the same 30 prompts × 3 seeds with
`sdxl_base` and **no LoRA adapter at all** (the prompts' `ukyowood` trigger token is inert to a
model that never saw it in training — these 90 images isolate what SDXL's pretraining alone
already knows about "ukiyo-e" style from the prompt text), scored on all 3 independent axes.
0 errors, 0 degenerate values (same audit as §4.6's provenance table). Paired lift vs. base
(matched prompt_id+seed, n=90, same paired-diff/SEM methodology):

| Axis | Published lift vs. base | SEM | lift/SEM | Curated lift vs. base | SEM | lift/SEM |
|---|---|---|---|---|---|---|
| `style_adherence` | +0.0056 | 0.0033 | 1.684 | **+0.0100** | 0.0036 | **2.816** |
| `figure_preservation` | **−0.0178** | 0.0029 | **−6.167** | −0.0100 | 0.0024 | −4.173 |
| `artifact_absence` | **−0.0500** | 0.0143 | **−3.489** | −0.0422 | 0.0136 | −3.107 |

**Hypothesis (b) is REFUTED — the opposite of the concern is true.** The curated adapter's
style-adherence lift over base (+0.0100, 2.82×SEM) is larger and *more* statistically significant
than the published adapter's (+0.0056, 1.68×SEM — does not itself clear 2×SEM). 23 training
images did not produce a weaker rank-8 LoRA on this axis; if anything the curated adapter shows
the clearer style signal. (Both lifts are small in absolute terms because `sdxl_base` alone
already scores 0.9389 on `style_adherence` for these ukiyo-e-styled prompts — a ceiling effect
from SDXL's own pretraining already knowing this style reasonably well, leaving limited headroom
for any LoRA to add.)

**Hypothesis (c), as originally framed, is REFUTED — but a more specific and important version
of it is CONFIRMED.** `sdxl_base` alone scores `artifact_absence` **0.9222** — clearly *higher*
(cleaner) than either LoRA variant (published 0.8722, curated 0.8800). The artifact is not simply
"already in the base model's prior, independent of any LoRA" (if that were true, base would score
similarly low). Instead: **both LoRA variants show a large, individually significant
`artifact_absence` regression relative to no adapter at all** (published −0.0500 at 3.49×SEM,
curated −0.0422 at 3.11×SEM — both clear 2×SEM as real, decisive effects on their own). This
confirms the original diagnosis's direction: training a LoRA on this WikiArt-sourced dataset
(curated or not) teaches the model something that degrades `artifact_absence` relative to the
base model's own behavior. **Curation does help, in the correct direction, by a real amount** —
curated recovers 0.0078 of the published adapter's 0.0500 regression (≈16%) — **but the recovered
fraction is small relative to the regression it's trying to fix, and too small relative to this
judge's arm-to-arm noise floor (§4.6's SEM 0.0133) to independently clear the pre-registered
2×SEM promotion bar in the direct published-vs-curated comparison**, even though each arm's own
effect vs. base is unambiguous. The same internal-consistency check holds for
`figure_preservation`: curated's smaller regression vs. base (−0.0100 vs. published's −0.0178)
also recovers exactly +0.0078 — matching §4.6's arm-to-arm figure_preservation lift precisely,
confirming these are the same effect viewed two ways, not independent artifacts.

**Conclusion — which hypothesis the evidence supports:** not "curation doesn't help" (a) is
refuted outright, and (b)/(c) both point to curation working as intended, directionally, on a
real underlying LoRA-induced artifact regression — just not recovering enough of it (16%) to be
statistically decisive at n=90 against this judge's noise. **The corrected framing for future
work: this recipe (VLM-judge curation → rank-8 LoRA retrain) is directionally validated, not
disproven — a next iteration should curate more aggressively (the 23-image kept set may still
carry residual artifact-inducing signal the current filter didn't catch), use a larger training
set, or increase LoRA rank, rather than abandoning curation as an approach.** This materially
changes the recommendation implied by a bare "no effect" reading of §4.6 alone.

### 4.9 High-power binary artifact metric — attempted, validated with caveats, and found to be
LESS powerful than the rubric (the opposite of the premise it was built to test)

**Premise being tested:** artifact presence is naturally binary (text/calligraphy/cartouche
detectable or not), and a paired proportion test on a binary outcome should have more power than
a noisy continuous VLM rubric — the rubric's ~1,963-paired-sample requirement (§4.7) suggested
the *instrument*, not `n`, was the bottleneck. This section builds and tests that instrument. The
result contradicts the premise, and is reported as such rather than adjusted to fit it.

**Detector** (`scripts/detect_text_artifacts.py`): EasyOCR with **both English and Japanese**
readers (`['en', 'ja']`) — deliberately different from `scripts/curate_ukiyo_e_dataset.py`'s
discarded English-only attempt, which false-positived on real woodblock-print brushwork texture
during *training-image* curation. That failure mode does not transfer here by assumption; it was
independently re-checked on *generated* images, a different population.

**Validation** (`scripts/validate_text_detector.py`, `reports/text_detector_validation.json`): a
stratified n=29 sample (published/curated/base, spanning the full observed VLM
`artifact_absence` score range) was labeled by direct visual inspection — genuinely independent
ground truth, not derived from the VLM score or the detector itself.

| Decision rule | Precision | Recall | n flagged (of 29) |
|---|---|---|---|
| `max_confidence >= 0.3` (naive first attempt) | 1.000 | **0.227** | 5 |
| `n_detections >= 1` ("any OCR region found, regardless of confidence") | 0.944 | **0.773** | 18 |

The naive confidence threshold is unusable (misses 77% of true positives) — SDXL-generated
pseudo-calligraphy is visually text-*shaped* but rarely legible enough for OCR's recognition step
to score it confidently (this is why `n_detections` — did OCR find *anything* text-shaped at all
— correlates with ground truth better than `max_confidence` does: Pearson r=0.47 vs. r=0.32).
Switching to the `n_detections >= 1` rule is a real, moderately-validated detector (94%
precision, 77% recall on n=29) — not a "fast negative" (it does correlate: r=0.65 against the
VLM's own artifact signal) — but it is not error-free, and that imperfection matters below.

**Applied to the full n=90 published/curated/base sets** (`scripts/compute_ocr_proportion_stats.py`,
`reports/ocr_proportion_stats.json`), McNemar's exact test on paired binary outcomes:

| Comparison | P(artifact\|A) | P(artifact\|B) | Diff | McNemar p |
|---|---|---|---|---|
| published vs. curated | 0.6111 | 0.5444 | −0.0667 | 0.480 |
| base vs. published | 0.5333 | 0.6111 | +0.0778 | 0.281 |
| base vs. curated | 0.5333 | 0.5444 | +0.0111 | 1.000 |

Directionally consistent with §4.6/§4.8 (curated < published; base < both), but none reach
significance — and **55.6% of published-vs-curated pairs are discordant** (one arm flagged, the
other not) — a very high rate reflecting real image-to-image variance in whether a given
generation's pseudo-text happens to render legibly enough for OCR, on top of the detector's own
23% miss rate.

**Power comparison — the premise fails:** the paired-proportion design's own MDE at n=90
(from the observed 55.6% discordant fraction, same 80%-power/α=0.05 convention as §4.7) is
**≈0.220** — nearly **6× worse** than the independent-axis VLM rubric's MDE of ~0.037 (§4.7), not
better. **A binary proportion test is not automatically higher-power than a continuous rubric —
that holds only when the underlying phenomenon is cleanly binary and the classifier is
near-noise-free. Neither holds here:** "does this image contain OCR-legible text" is a lossier,
higher-variance question than "how clean does this look on a continuous 0–1 scale," and
binarizing throws away exactly the graded information that gave the rubric its (comparatively
better, if still limited) sensitivity.

**Conclusion: this metric does not resolve the power problem — it is reported as a negative
result, not adjusted or reframed to appear positive.** The independent-axis VLM rubric remains
the **primary** metric (best available sensitivity, MDE ~0.037); the OCR binary detector is
reported as a **secondary, directionally-corroborating signal only**, with its validated 77%/94%
recall/precision explicitly disclosed wherever it's cited. Per the task's own framing — "either
[promotion changes or confirms] is a result" — the result here is that no better-powered zero-cost
local instrument was found this pass; §4.7's ~1,963-sample requirement stands as the actual path
to full resolution, and remains impractical at zero-cost-local scale.

### 4.10 Positive control on the `style_adherence` judge — ORIGINAL comparison FAILED (ceiling-effect confound); REDESIGNED control PASSES — PROVISIONAL marking LIFTED

**Original finding (2026-07-25), escalated per a pre-committed decision rule:**
`scripts/judge_style_positive_control.py` first tested whether the VLM judge's `style_adherence`
axis can tell real, authentic ukiyo-e reference art (n=23) apart from `sdxl_base`'s own
**ukiyo-e-prompted** generated attempts (n=90, unpaired). It failed: real ukiyo-e art scored
0.9239 vs. `sdxl_base`'s 0.9389 — `diff = −0.0150`, `SEM = 0.0099`, `diff/SEM = −1.507`, wrong-
signed. Per the pre-committed rule this marked every `style_adherence` number in §4
PROVISIONAL, not retracted — an instrument shown unable to discriminate is a *validity* problem,
not merely underpowered.

**RESOLVED (2026-07-26): the original comparison was itself a ceiling-effect confound, not a
fair validity test — re-examined and fixed per `docs/WEIGHT_SWEEP_PREREGISTRATION.md`'s
amendment.** The control's arm B was `sdxl_base` outputs from **ukiyo-e-worded prompts** — asking
the judge to distinguish real ukiyo-e art from SDXL's own convincing attempt AT ukiyo-e, a
near-ceiling task on a style this project has already documented as strongly represented in
SDXL's pretraining. A judge could fail exactly this comparison while still perfectly able to tell
ukiyo-e apart from something that actually looks different.

**Redesigned control, pre-registered before running (`reports/judge_style_positive_control.json`):**
real ukiyo-e art (n=23) vs. two off-style contrasts under the SAME `UKIYO_E_STYLE_QUESTION`:

| Comparison | ukiyo-e mean (n) | contrast mean (n) | diff/SEM | Verdict |
|---|---|---|---|---|
| vs. real Pattachitra art | 0.9239 (23) | 0.2975 (100) | **+25.062** | **PASS** |
| vs. generic `sdxl_base` (non-style prompts) | 0.9239 (23) | 0.2033 (90) | **+25.580** | **PASS** |

**PASS on both, decisively — not a borderline result.** The judge separates real ukiyo-e art from
a genuinely different real style and from generic non-styled generations by >25×SEM in each case.
**Per the pre-registered rule, the PROVISIONAL marking on every `style_adherence` number in §4
above and in `docs/HF_MODEL_CARD_UPDATES.md` is now LIFTED — CONFIRMED, not provisional.** The
original `−1.507` FAIL is retained in the record above as the reason the earlier control was
mis-specified (a ceiling effect from testing against the wrong kind of contrast), not deleted as
if it had never happened.

**This does not retroactively affect `artifact_absence`** (§4.6, §4.8's primary regression
finding, or §4.9) — that axis's question is domain-neutral and was never implicated either way.

**Not proven to generalize — Pattachitra's own (properly-controlled) positive control FAILS,
for a different, unrelated reason: see §7.2 Addendum's second finding.** The two domains' controls
now diverge: ukiyo-e's instrument is validated; Pattachitra's is not, even under the same fair
methodology. This is reported as an asymmetric result, not smoothed into a single project-wide
verdict on the judge's reliability.

**No retraining, no re-evaluation follows from this finding.** The published ukiyo-e checkpoint's
status is unchanged (§4.6 already withdrew its promotion on other grounds); this resolves a
measurement caveat, not a new action item.

---

## 5. Per-family verdict

| Family | Verdict | Basis |
|---|---|---|
| `sd21_base` | production-quality | Baseline family; no defects found in this eval pass. |
| `sdxl_base` | production-quality | Highest CLIP among base families; no defects found. |
| `hyper_4step` | production-quality — **recommended fast path** | Best HPS+CLIP+latency of all 5 families. |
| `hyper_8step` | needs-improvement | 16% severe-underexposure rate at spec-compliant config; strictly dominated by `hyper_4step` on every measured axis. Not recommended for production traffic; keep `hyper_4step` as the routed default. |
| `sdxl_controlnet_union` | production-quality (quality) / needs-improvement (latency) | Quality on par with `sdxl_base`; 546.6s latency requires CPU offload on this 8GB card — a hardware-fit issue, not a model-quality one. |
| Ukiyo-e LoRA — published checkpoint-1000 | **remains champion — not superseded** | §4.3's correlated-regime headline (+0.040, 3.18×SEM) does not survive the full n=90 paired independent-axis rescore (§4.6: +0.0078, 0.583×SEM — well below the 2×SEM promotion bar). §4.4's promotion verdict is withdrawn per §4.6. No HF re-upload. |
| Ukiyo-e LoRA — curated retrain | **not promoted (gate not cleared) — but directionally validated, not disproven** | §4.6 (full n=90 paired, independent single-axis scoring) finds the arm-to-arm `artifact_absence` diff at 0.583×SEM, not the 3.18×SEM the correlated single-call judge reported; §4.7 confirms this design's 95% CI rules out the original +0.040 claim but is underpowered below its own MDE (~0.037) to call this a demonstrated null. §4.8's base-model comparison shows both LoRA variants regress `artifact_absence` significantly vs. no adapter (published −0.0500 at 3.49×SEM, curated −0.0422 at 3.11×SEM) — confirming LoRA training on this data does cause the artifact — and curation recovers ~16% of that regression (real, correctly-directioned, just not enough to clear the arm-to-arm bar at this n/judge). No guardrail regresses; curated's `style_adherence` lift vs. base is in fact clearer than published's (2.82×SEM vs. 1.68×SEM). |

---

## 6. Final routing decision

**Does the current lineup clear the bar to train and publish a new model, or does something
need fixing first? One fix remains before a clean green light: `hyper_8step`'s routing (item 1).
The LoRA question that motivated most of this audit is now resolved the other way from §4.4's
original verdict — do NOT promote the curated retrain (item 2), per the corrected §4.6 result.**

1. **`hyper_8step` should not be routed to production as currently configured** (§3) — it is
   strictly dominated by `hyper_4step` on HPS, CLIP, and latency, and has a 16%
   severe-underexposure defect rate. Action: change the default/recommended Hyper-SD variant in
   any user-facing routing logic to `hyper_4step`. This is a config/docs fix, not a retrain.
2. **Do NOT promote the curated retrain; do NOT re-upload to HF Hub on this A/B's basis** (§4.6
   corrects §4.3/§4.4/§4.5 — the published checkpoint remains champion). The pre-registered
   primary endpoint, measured under the trusted independent-axis regime, does not clear the
   2×SEM promotion bar (0.583×SEM, not the correlated single-call judge's 3.182×SEM). Per this
   project's own pre-registered decision rule, a null-or-below-threshold primary result is
   reported as such and is not grounds to keep looking for a threshold that passes.
   `docs/HF_MODEL_CARD_UPDATES.md` is corrected to drop the "measured improvement" claim. **This
   is a gate decision, not a verdict that curation doesn't work** — §4.8's base-model comparison
   found both LoRA variants significantly regress `artifact_absence` relative to no adapter at
   all (LoRA training on this data does cause the artifact, confirming the original diagnosis),
   and curation recovers a real ~16% of that regression in the correct direction — just not
   enough, relative to this judge's noise at n=90, to independently clear the arm-to-arm
   promotion bar (§4.7: the MDE for this judge/n is ~0.037; closing the gap to the observed
   +0.0078 effect would need ≈1,963 paired samples, impractical at zero-cost-local scale). The
   recipe (VLM-curate → retrain) is directionally supported, not disproven; a next iteration
   should curate more aggressively or scale the training set, not abandon curation.
3. **All 5 non-LoRA families are production-quality** on the metrics measured here, with
   `sdxl_controlnet_union`'s latency flagged as a hardware-fit constraint (8GB VRAM + CPU
   offload), not a quality defect.
4. **Training and publishing a genuinely new model under the current recipe is not recommended
   until (1) is fixed** — publishing a 6th family without first correcting a known, dominated,
   defect-prone configuration in the existing lineup would compound the same class of issue this
   whole audit was launched to catch. (2) is now a closed question (do not promote), not a
   blocker either way — no further LoRA A/B action is pending.

---

## 7. Pattachitra LoRA — trained, evaluated against `sdxl_base`, and found NOT to beat it

**Per `docs/PATTACHITRA_AB_PREREGISTRATION.md`'s amended design (fixed before this run): two arms
only — `sdxl_base` and a single curated-corpus LoRA (no uncurated control) — with `style_adherence`
and `artifact_absence` vs. `sdxl_base` as co-primary endpoints, `figure_preservation` vs. base as
a non-inferiority guardrail, and mandatory MDE/CI reporting regardless of outcome.** Training: rank-8
LoRA, 100 curated images (111 automated-clean minus 11 found on manual QA to be documentary/vendor
photos or a mismatched visual genre — see below), 1500 steps, batch 1×grad-accum 4, lr 1e-4, fp16,
seed 42, GCP `g2-standard-8`/L4, `us-west1-a`. Actual cost: ≈$1.90–2.20 (≈2.2 GPU-hours across one
successful ~1h40m run plus ~30–35 min combined across four dependency-debugging attempts — all
root-caused and disclosed in `PLAN.md`), against a $5–7.50 estimate and $10 hard stop.

### 7.1 Manual QA finding on the training corpus (before training, not after)

The automated VLM curation filter's 111 "clean" verdicts (`docs/NEXT_MODEL_SPEC.md` §3.5) were
spot-checked by direct visual inspection before committing GCP spend. **11 of 111 were
documentary/vendor photographs where a person's face or torso dominates a substantial fraction of
the frame** — exactly the curation prompt's own stated exclusion criterion, missed by the
automated filter. Filenames containing "artist at work," "stall," or "book fair" were a strong
(not perfect) predictor: some book-fair-titled photos were legitimately clean close-ups of
artwork, but most "artist [...] at work" ones showed a person's face/torso as a substantial
fraction of the frame. One further image (painted spherical decorative objects, not flat scroll
paintings) was excluded for visual-genre mismatch. Final training corpus: **100 images** (down
from 111), still ≈4.3× ukiyo-e's curated set (23).

### 7.2 Validating the negative before writing it up — ruling out three mechanical explanations

A −7.226×SEM guardrail regression with total figure dropout on a seed where base renders correctly
is a large enough effect that it is more consistent with a mechanical eval bug than a genuine
rank-8 style-training effect — so it was checked, not assumed, before being written up as a finding.

**(1) Checkpoint selection.** The initial pass scored only the final weights (=checkpoint-1500),
skipping the checkpoint-select step the pre-registration itself calls for ("as with ukiyo-e") —
ukiyo-e's own precedent explicitly **rejected** its checkpoint-1500 for this exact failure mode
("mild mode-collapse, lost the samurai figure in prompt 3," `docs/lab_notebook.md`) and selected
checkpoint-1000 instead. This was a real process gap, corrected here: checkpoints 500, 1000, and
1500 were all scored (n=90 each, 0 errors across 270 generations + 810 independent VLM calls total
for this section). Selection rule (fixed before seeing the checkpoint-1000/500 numbers): among
checkpoints where `figure_preservation` does not regress >2×SEM vs. base, select the one with the
best `style_adherence` diff/SEM.

**`style_adherence` values in this table predate the judge-question-bug discovery (§7.2 Addendum
below) and are shown here exactly as originally measured, with their status flagged inline —
not silently updated in place, so the audit trail stays honest about what was known when.**

| Checkpoint | `style_adherence` diff/SEM | `figure_preservation` diff/SEM | Guardrail |
|---|---|---|---|
| curated500 | ~~+0.477 (not significant)~~ **VOID (wrong question) → resolved −2.996, §7.3** | **−5.532** | REGRESSED |
| curated1000 | ~~−0.789 (not significant)~~ **VOID (wrong question) → resolved −4.603, §7.3** | **−7.768** (worst of the three) | REGRESSED |
| curated1500 | ~~−2.234 (significant regression)~~ **VOID (wrong question) — NOT resolved, out of the weight sweep's scope** | −7.226 | REGRESSED |

**No checkpoint clears the guardrail — the regression is not a checkpoint-selection artifact.**
It is present, and large (−5.5×SEM at minimum), from the *earliest* checkpoint tested (500 steps) —
even minimal training introduces it — and does not improve with proper selection the way ukiyo-e's
did. ~~`style_adherence` does show a monotonic within-run drift (500: mildly positive → 1000:
mildly negative → 1500: significantly negative) consistent with some overtraining on that axis
specifically~~ — **this specific claim rested entirely on the void numbers above and does not
survive: the resolved values (500: −2.996×SEM, 1000: −4.603×SEM) show both already significantly
*negative*, not a mild-to-strong drift starting from positive; 1500 remains unresolved so no
three-point trend claim can be made at all.** The guardrail failure that would have blocked
promotion under the ukiyo-e methodology is present at every checkpoint tested regardless, so
checkpoint selection changes nothing about the headline verdict — if anything, the resolved
`style_adherence` numbers make that conclusion more decisive, not less, since 500 and 1000 are
now confirmed regressions too, not near-null effects.

*A later audit checked the diff/SEM ordering directly rather than leaving it as an unexplained
anomaly (prompted by: "checkpoint-1000 scoring worse than both 500 and 1500 isn't a training
dynamic — genuine overtraining is monotonic in steps").* The **raw** `figure_preservation` diff
(not normalized by SEM) is in fact monotonically non-decreasing in magnitude with more steps —
500: −0.0228 → 1000: −0.0444 → 1500: −0.0461 (`reports/pattachitra_ab_stats.json`) — consistent
with progressively worsening figure/subject dropout, not a training-dynamics reversal. The
diff/SEM *ratio* ranks checkpoint-1000 (−7.768) as marginally worse than checkpoint-1500 (−7.226)
only because checkpoint-1500's per-pair variance is higher (stdev of the 90 paired diffs: 0.0605
vs. 0.0543) — an artifact of the SEM denominator, not evidence the underlying effect improved at
1500. See §7.2(4): all three checkpoints were independently confirmed to come from verified-clean
generation processes, so this is not a shared-corruption confound either.

**(2) LoRA applied weight.** Verified directly against the ukiyo-e recipe, not assumed: the
training script sets `lora_alpha = args.rank` internally
(`scripts/_diffusers_train_text_to_image_lora_sdxl.py`), so `--rank=8` (the flag actually passed,
`scripts/gcp_startup_pattachitra_train.sh`) produces alpha=rank=8 automatically — identical to
ukiyo-e's recipe. The evaluation script applies `pipe.set_adapters(["pattachitra"],
adapter_weights=[1.0])` (`scripts/_pattachitra_ab_base_comparison.py`), byte-identical to the calls
ukiyo-e's own eval scripts use (`scripts/_lora_ab_30prompt_independent.py`,
`scripts/model_verdict_harness.py`). **No over-application — the effective adapter scale matches
ukiyo-e exactly.** This does not explain the regression.

**(3) Trigger token collision.** The trigger is `pattascroll`. Checked against the actual SDXL
tokenizer (`CLIPTokenizer.from_pretrained(..., subfolder="tokenizer")`), not assumed from
appearance: `pattascroll` → 4 BPE sub-tokens (`patt`, `as`, `cro`, `ll</w>`); for comparison,
ukyowood → 3 sub-tokens (`uk`, `yo`, `wood</w>`). **Neither trigger resolves to a single real
vocabulary token** — both are novel multi-token strings of similar construction, so this is not
the clean "real English word" collision the hard rule was checking for. One modest, *unproven* risk
factor worth flagging for a future retrain: `pattascroll`'s decomposition includes `as`, an
extremely high-frequency English function word (rank ~top-200), whereas `ukyowood`'s fragments
(`uk`/`yo`/`wood`) are lower-frequency content-word-like tokens. This is a plausible, minor
contributing factor for a future retrain with a cleaner token (e.g. `pattagraph` or a digit-suffixed
nonce token) to test — **not** a demonstrated mechanism, and not sufficient on its own to explain a
−5.5-to-7.8×SEM effect present at every checkpoint.

**(4) CUDA context corruption mid-run (checkpoint-500).** The first `curated500` generation attempt
crashed with `RuntimeError: CUDA error: an illegal memory access was encountered` after 22/90
images, then failed all 68 remaining attempts in that process (a poisoned CUDA context fails every
subsequent kernel launch). Checked, not assumed, on two axes: **(a) could the crash have corrupted
the 22 images generated *before* it, in the same process?** File-mtime provenance identified
exactly which 22 images preceded the crash; a pixel-level integrity audit (mean/std/min/max/NaN/
pure-black/pure-white per image) found them statistically indistinguishable from confirmed-clean
images from other processes (mean-of-means 120.78 vs. 120.26; zero NaN, zero pure-black/white
pixels in either group), and direct visual inspection of the boundary images — including the very
last image generated before the crash — showed clean, coherent, non-degenerate output. This also
matches the architecture: an "illegal memory access" is a hard, fail-loud kernel-launch error: it
does not retroactively corrupt data already copied off-GPU and written to disk by an earlier,
already-completed iteration. **(b) base, curated1000, and curated1500** were each independently
confirmed to come from a single, uninterrupted, zero-CUDA-error process (log grep for
`traceback|cuda error|illegal memory access` → 0 matches, exactly one pipeline load, for each) — so
only curated500 has any at-risk images, and those are confirmed clean. **No image regeneration was
warranted.** The retry-duplicate script bug (§7.3) was separately verified fully clean: all four
checkpoint sets contain exactly 90 unique `(prompt_id, seed)` records with zero duplicates and zero
missing cells (`reports/pattachitra_ab_base_comparison.json`, checked directly). A permanent
harness self-check (CUDA pre-flight health probe, per-record uniqueness assertion, degenerate-image
detection, judge-response range validation — `scripts/model_verdict_harness.py`, 16 new tests) was
added afterward so this class of silent corruption fails loudly instead of requiring a manual
forensic audit to catch. **Retroactively applied to every pre-existing report, not just new runs:**
`scripts/scan_verdict_judge_ranges.py` scanned all 8 `reports/*.json` files carrying a judge-score
field (`vlm_judge`/`independent_calls`/`original_single_call`) for values the new guard would have
rejected (non-numeric, NaN, outside `[0, 1]`) — **1,044 records scanned, 0 violations.** Every
number in this document, and the staged ukiyo-e HF card numbers, is confirmed clean under the new
guard, not merely assumed clean by having predated it.

**(5) Recipe overtraining (effective epochs, computed directly from the training logs, not
assumed).** §7.2's own corrected finding — the raw `figure_preservation` diff is monotonic in
steps (500: −0.0228 → 1000: −0.0444 → 1500: −0.0461) — is the signature of *progressive*
overtraining, and checkpoint-500 (the earliest ever tested) is not evidence the effect starts at
zero, only that it starts somewhere below 500 steps. This raises a distinct question from (1)–(4):
is this a **style** verdict (the corpus/aesthetic doesn't LoRA-train cleanly) or a **recipe**
verdict (the 1500-step schedule, carried over unchanged from ukiyo-e, overtrains a small corpus
regardless of style)? Effective epochs, computed from the actual launch commands (not estimated):

| Adapter | Images | Batch × grad-accum | Steps/epoch | Checkpoint | Steps | Effective epochs |
|---|---|---|---|---|---|---|
| Pattachitra curated | 100 | 1×4 (eff. batch 4) | 25 | 500 / 1000 / 1500 | 500 / 1000 / 1500 | **20 / 40 / 60** |
| Ukiyo-e curated | 23 | 1×4 (eff. batch 4) | 5.75 | 1000 (selected/published) | 1000 | **≈174** |

(`data/lora/pattachitra-curated/.../checkpoint-*` per `scripts/gcp_startup_pattachitra_train.sh`'s
`--train_batch_size 1 --gradient_accumulation_steps 4 --max_train_steps {500,1000,1500} --rank 8`;
ukiyo-e curated per its own `retrain.log`'s logged launch command — identical
`--train_batch_size 1 --gradient_accumulation_steps 4 --max_train_steps 1500 --rank 8`, and
`Num examples = 23` logged directly by the training script, not assumed from a file count.)

**Tightened claim — do not overstate what the 500-step datapoint supports.** Pattachitra's
checkpoint-500 sits at **20 effective epochs — *inside* the 10–50-epoch band typically sufficient
for style-LoRA convergence** on a small, homogeneous corpus — and it already regresses
`figure_preservation` by −5.532×SEM vs. `sdxl_base`. Overtraining is a plausible explanation for
the *monotonic worsening* from checkpoint-500 to checkpoint-1500 (20→60 epochs, −5.5×SEM→
−7.2×SEM) — more steps beyond an already-present effect compounding it is consistent with the
data — **but it does not explain the regression already present at an in-band epoch count.**
Whatever produces the −5.5×SEM baseline effect at 20 epochs is not "too many epochs," and needs a
separate explanation (a corpus/caption/style effect per §7.3's hypotheses, or an
over-application-at-any-epoch-count effect — the adapter-weight sweep below tests dosage directly
and independently of step count). Ukiyo-e's ≈174 effective epochs — nearly 9× checkpoint-500's
count and far outside the convergence band on any accounting — remains the **stronger**
overtraining candidate of the two, and the 1500-step/batch-1/grad-accum-4 schedule is a legitimate
concern to revisit for *that* adapter specifically. The identical-schedule-across-differing-corpus-
size observation (100 vs. 23 images, a 4.3× difference the recipe does not adjust for) does **not**,
by itself, establish the schedule as *the* explanation for Pattachitra's regression — only as a
plausible contributor to how much worse it gets as steps increase, not to why it is already
non-zero at 20 epochs. The two adapters' overtraining evidence is asymmetric, not a single shared
verdict, and is reported as such.

**Adapter-weight sweep (zero-cost, local only, no retraining) — RESULT: four operating points
found on `style_adherence`, all below the deployed weight, none at or above it — but see the
§7.2 Addendum's second finding: Pattachitra's `style_adherence` axis has since failed its own
positive control even under the corrected question, so the "operating point" claim below is
numerically real (a measured, significant diff/SEM at weight 0.3–0.5) but PROVISIONAL as a claim
about genuine style-authenticity improvement, exactly parallel to how ukiyo-e's original
`style_adherence` numbers were provisional before its own control was fixed — except Pattachitra's
control did not pass on re-examination, so this marking is not lifted. `figure_preservation`'s
half of the joint criterion is unaffected and is the trusted half of this table.** Checked whether
a lower
`adapter_weights` scale recovers `figure_preservation` while keeping a genuine, significant
`style_adherence` lift over `sdxl_base` — the joint criterion pre-registered in
`docs/WEIGHT_SWEEP_PREREGISTRATION.md` (`style_adherence` diff/SEM `> +2.0` AND
`figure_preservation` diff/SEM `>= -2.0`, at the same weight), computed from a uniform re-score of
every weight (0.3/0.5/0.7/1.0) on all three axes with the corrected `PATTACHITRA_STYLE_QUESTION`
(`scripts/rescore_pattachitra_uniform.py` → `reports/pattachitra_uniform_rescore.json`,
810 images, 0 errors; `scripts/compute_pattachitra_weight_sweep_stats.py` →
`reports/pattachitra_weight_sweep_stats.json`):

| Checkpoint | Weight | `style_adherence` diff/SEM | `figure_preservation` diff/SEM | fp MDE@80% | Operating point? |
|---|---|---|---|---|---|
| curated500 | 0.3 | **+3.622** | +0.754 | 0.0083 | **YES** |
| curated500 | 0.5 | **+3.482** | +2.080 | 0.0082 | **YES** |
| curated500 | 0.7 | +1.348 | −0.491 | 0.0095 | no (style not significant) |
| curated500 | 1.0 | −2.996 | −5.402 | 0.0115 | no (both axes fail) |
| curated1000 | 0.3 | **+2.688** | +0.424 | 0.0073 | **YES** |
| curated1000 | 0.5 | **+3.482** | +1.422 | 0.0088 | **YES** |
| curated1000 | 0.7 | −0.306 | −1.153 | 0.0095 | no (both axes fail) |
| curated1000 | 1.0 | −4.603 | −7.540 | 0.0159 | no (both axes fail) |

**This means the checkpoint is over-applied at `weight=1.0` (the only scale it has ever been
evaluated or would be deployed at) on the `figure_preservation` axis, which IS trusted: it does
not regress and in fact modestly *improves* over `sdxl_base` at weights 0.3–0.5, unlike at
`weight=1.0`.** The `style_adherence` half of this picture — that the adapter also "delivers a
real, significant style lift" at low weight — is a measured, significant diff/SEM (not the trivial
weight→0 tautology the pre-registration guarded against), but **cannot currently be certified as a
genuine style-authenticity improvement**, since the instrument measuring it has failed its own
positive control for this domain even when properly asked (§7.2 Addendum). The `figure_preservation`
recovery at low weight stands on its own regardless of how `style_adherence` resolves.

**The pre-registration required an explanation for a non-monotonic interior peak, not a bare
report of one — here it is.** Raw `style_adherence` arm means for `curated500`: weight 0.3 →
0.9072, weight 0.5 → **0.9167 (the peak)**, weight 0.7 → 0.9000, weight 1.0 → 0.8472 — *below*
`sdxl_base`'s own 0.8883 at full weight. The curve rises from the (implied, by construction)
weight=0 value of 0.8883, peaks at weight=0.5, then falls below even the no-adapter baseline by
weight=1.0. **Plausible mechanism (offered as a hypothesis grounded in this audit's own evidence,
not independently verified further — doing so would exceed this diagnostic's scope):** LoRA
weight scaling linearly interpolates the effective model between `sdxl_base` (weight=0) and the
fully-applied adapter (weight=1.0). If the adapter's learned delta carries a comparatively robust,
generalizable *stylistic* signal (decorative border, palette, compositional layout — plausibly
learnable from a 100-image corpus even with generic BLIP captions, §7.3) alongside a comparatively
noisy, less generalizable *figure-rendering* signal (plausibly harder to learn robustly from the
same corpus, and consistent with §7.2(5)'s effective-epoch finding that this checkpoint is
overtrained relative to its corpus size), a partial blend would be expected to average in the
stylistic signal — which dominates at low-to-moderate weight — while diluting the noisier
figure-rendering signal's ability to override `sdxl_base`'s own more robustly-trained figure
prior. Only near full weight does the adapter's own (comparatively weaker) figure-rendering
behavior dominate the blend, degrading both axes together. This is consistent with, and does not
contradict, §7.2(5)'s epoch analysis — it characterizes *how* overtraining manifests (a
weight-dependent tradeoff) rather than replacing that finding.

**Conclusion: none of the four *mechanical* explanations (1)-(4) accounts for the regression; (5)
is confirmed a genuine recipe-level contributing factor — the checkpoint is over-applied at the
deployed weight, not intrinsically incapable of a positive characterization at a lower one.**
Checkpoint selection doesn't resolve it (worse, even), and the CUDA corruption is confirmed
contained to 68 failed *attempts* (0 corrupted successes) on one checkpoint, ruled out by direct
pixel/visual audit rather than architectural reasoning alone. **The finding — at `weight=1.0`,
the only scale this adapter has been evaluated at or would be deployed at, SDXL+adapter regresses
`figure_preservation` decisively and does not demonstrate a `style_adherence` lift — stands
exactly as reported below. This diagnostic does not reopen that verdict** (per the standing
directive's explicit cap): a genuine operating point at `weight=0.3–0.5` is a finding about the
*recipe*, informing whether a future, separately-authorized retrain attempt should investigate
adapter-weight tuning as a contributing factor — it is not grounds to publish, redeploy, or
retrain within this pass.

### 7.2 Addendum — a sixth issue found, of a different kind: `style_adherence` (endpoint A) is VOID; then a seventh found while fixing the control used to resolve it

**(1)–(5) above interrogate whether the `figure_preservation` guardrail regression is real. This
addendum concerns an independently-discovered defect in `style_adherence` — Primary Endpoint A —
found while designing `scripts/judge_style_positive_control.py`
(`docs/WEIGHT_SWEEP_PREREGISTRATION.md`'s judge positive-control section, itself written before
any Pattachitra-domain result from it was read).**

**Finding, confirmed by direct code inspection, not inference:** `scripts/model_verdict_harness.py`'s
`style_adherence` judge question was hardcoded to ask "does this image look like an authentic
Ukiyo-e (Japanese woodblock print)" regardless of caller. `scripts/_pattachitra_ab_base_comparison.py`
and `scripts/_pattachitra_weight_sweep.py` both reused this function unmodified. **Every
Pattachitra `style_adherence` score recorded before the fix — all 360 records — was generated by
asking the judge whether the image looks like ukiyo-e, not Pattachitra.**

**A new bug class for this project: semantically wrong but syntactically valid.** Every integrity
self-check added after the CUDA-corruption audit (CUDA pre-flight health probe, degenerate-image
detection, per-record uniqueness assertion, judge-score range validation) passed on every one of
these 360 records — the request was well-formed, the response was a well-formed in-range float,
the Ollama call succeeded. None of those checks — nor `scripts/scan_verdict_judge_ranges.py`'s
retroactive scan, which also reported 0 violations on this exact data — can catch a *wrong
question asked correctly*: they check value validity, not question validity. This defect class was
invisible to every mechanism this project had built to catch measurement artifacts, until found by
direct code inspection rather than by any automated check.

**Fixed:** `score_vlm_judge` now requires an explicit `style_question` from every caller — no
harness-level default (a missing style now raises `TypeError`, not a silent wrong-domain fallback).
Every caller states its own domain (`UKIYO_E_STYLE_QUESTION` for ukiyo-e scripts;
`PATTACHITRA_STYLE_QUESTION` — grounded in `docs/NEXT_MODEL_SPEC.md`'s own prior corpus
description, not invented — for Pattachitra scripts). New regression tests
(`tests/test_model_verdict_harness.py::TestStyleQuestionIsNeverHardcoded`) assert: the question
actually sent to Ollama contains the caller's stated domain; two different domains produce two
different requests; omitting `style_question` raises `TypeError`; and — auditing the other two
axes for the same defect class, as this finding demands — neither `figure_preservation` nor
`artifact_absence`'s fixed templates hardcode a style name either (both confirmed domain-neutral).

**Blast radius, scoped per axis — not a blanket void of §7:**

| Axis | Pattachitra (this project) | Ukiyo-e (§4) |
|---|---|---|
| `style_adherence` | **VOID — wrong question asked (asked about ukiyo-e).** All 360 records (90 base + 90×3 curated checkpoints, `reports/pattachitra_ab_base_comparison.json`) and every number derived from them below (§7.3's `style_adherence` rows, §7.4's "no style-adherence lift" framing, §7.6) are marked VOID, not deleted — that this was measured wrong is itself part of the record. | **UNAFFECTED.** The hardcoded question was the *correct* one for ukiyo-e — this was always ukiyo-e's own domain. §4's results and `docs/HF_MODEL_CARD_UPDATES.md`'s published card text stand as written. |
| `figure_preservation` | **Survives — audited, confirmed domain-neutral:** `"are the subjects/figures in this image anatomically coherent and recognizable... The image was generated from this prompt: {prompt}"` names no style. The guardrail regression finding in (1)-(5) above and §7.3 stands. | Survives (identical template, always domain-neutral). |
| `artifact_absence` | **Survives — audited, confirmed domain-neutral:** `"is this image FREE of embedded text, watermarks, signatures, cartouches, or seal/script marks..."` names no style. §7.5's ceiling-effect finding stands. | Survives (identical template, always domain-neutral). |

**Practical consequence for this document:** every claim in §7.3/§7.4/§7.6 below that rests on
`style_adherence` is **VOID as currently measured — neither confirmed nor refuted** — and requires
re-scoring with `PATTACHITRA_STYLE_QUESTION` before it can be trusted in either direction. The
`figure_preservation` guardrail finding — the actual, stated reason this adapter is not published —
is **unaffected** and stands on its own; it was never dependent on the `style_adherence` numbers.

**The publication decision is already settled and does not wait on any of the remaining work below.**
`figure_preservation` regresses −5.5 to −7.8×SEM vs. `sdxl_base` at `weight=1.0` on a
**domain-neutral question, confirmed unaffected by the judge-prompt bug** — this alone is
sufficient to not publish, independent of what `style_adherence`, the weight sweep, or the judge
positive control eventually show. **No outcome of the remaining work can publish this adapter** —
a discovered low-weight operating point would characterize *how* the adapter fails differently
(over-applied vs. intrinsically broken), not *whether* it fails the guardrail at the scale it has
actually been evaluated at (`weight=1.0`, the only scale a real product would deploy at). The
remaining work is **diagnostic, not decisional**, and is explicitly capped at exactly two
questions, no further:

1. **Does a low-weight operating point exist?** (the adapter-weight sweep, below) — informs
   whether a future retrain attempt, if one is ever authorized, should investigate dosage as a
   contributing factor. Does not reopen whether to publish the `weight=1.0` checkpoints evaluated
   in §7.3.
2. **Can the judge perceive Pattachitra style at all, under either question wording?** (the
   positive control, `scripts/judge_style_positive_control.py`) — informs whether
   `style_adherence` is recoverable by re-scoring or void for a second, independent reason
   (instrument blindness). Does not change the `figure_preservation` verdict either way.

**Explicitly out of scope for this diagnostic pass:** no further retrain, no corpus rework, no
additional checkpoint or hyperparameter sweep beyond the two questions above. If either diagnostic
surfaces a genuinely promising lead (e.g. a real operating point), that is a candidate for a
*future*, separately-authorized attempt — not something to chase further within this pass.

**Consequence for the running weight sweep, updated after this task's own changes:** the sweep's
inline VLM scoring has been disabled entirely (both checkpoints now generate-only — see
`scripts/_pattachitra_weight_sweep.py`'s docstring) — every score, for every axis, at every
weight including `weight=1.0`, will come from one later, uniform re-score pass over the saved
images, using `PATTACHITRA_STYLE_QUESTION`. This replaces the earlier plan (letting `curated500`
finish scoring under the old buggy code, then re-scoring anyway) with a cleaner one: the
already-in-flight sweep process was stopped and relaunched with scoring disabled once the fix
landed — its 100 already-generated images at that point were preserved and resumed from (verified:
the relaunched process picked up exactly where the JSON left off, generating no duplicates), so no
generation work was lost, only the now-pointless scoring work was skipped going forward. §7.4's
verdict text below is not finalized until the judge positive control and the uniform re-score both
land (`docs/WEIGHT_SWEEP_PREREGISTRATION.md`) — but the **decision** (not published) already does
not depend on either.

**Uniform re-score completed (2026-07-25/26):** `style_adherence` is no longer VOID from the
hardcoded-question bug — every number in §7.3's table below and the weight-sweep table above
comes from `scripts/rescore_pattachitra_uniform.py`'s uniform re-score
(`reports/pattachitra_uniform_rescore.json`, 810 images, 0 errors), scored once, consistently,
with the corrected `PATTACHITRA_STYLE_QUESTION` — including the `sdxl_base` mean of **0.8883**,
confirmed by direct code inspection to be a genuine, fresh re-score of the base arm, not a carry-
forward from the old buggy run (see the second finding below for why this distinction matters).

**A second finding within this addendum (2026-07-26) — the judge positive control's own base arm
was confounded, and fixing it reverses Pattachitra's control from PASS to FAIL:**

`scripts/judge_style_positive_control.py`'s original run reported Pattachitra's corrected-question
positive control as a decisive PASS (`diff = +0.4437`, `diff/SEM = +12.005`). **That result was
itself confounded — confirmed by direct code inspection, not inference:** the script loaded
`sdxl_base`'s Pattachitra scores ONCE, from the OLD pre-fix `pattachitra_ab_base_comparison.json`
(scored under the hardcoded wrong ukiyo-e question), and reused that same value unchanged for
BOTH the historical-prompt row (where reuse is valid — both arms consistently wrong-question) AND
the "corrected prompt" row (where it is not — the base arm there was never actually re-scored
under the corrected question). **This is a fifth measurement-defect class** (`PLAN.md`
2026-07-26a): inconsistent reference arms across two analyses of the same underlying data —
invisible to both value-validity checks and the domain-parameterization test built for bug #4,
since the *question sent* was never wrong in this script; the *data reused* was stale.

**Fixed and re-run (`docs/WEIGHT_SWEEP_PREREGISTRATION.md`'s amendment, pre-registered before
running): the corrected-prompt row was recomputed with BOTH arms scored fresh under the identical
corrected question.** Result: real Pattachitra art = 0.7960 (n=100), `sdxl_base` = 0.8883 (n=90),
`diff = −0.0923`, `SEM = 0.0244`, `diff/SEM = −3.781` — **FAIL, wrong-signed, and decisive** (the
effect clears its own MDE@80% of 0.0684, so this is a well-powered result, not an underpowered
null). **This is, structurally, Pattachitra's own positive control — real target-style art vs.
`sdxl_base`'s attempts, both under the same fair question — and it fails.** A plausible
(unverified) shared mechanism with ukiyo-e's original ceiling effect: `sdxl_base`'s Pattachitra
generations are prompted with language drawn from the same corpus-quality description the judge
question itself uses, so the generation may match the judge's literal criterion more cleanly than
an authentic photograph carrying real-world documentary artifacts the idealized description
doesn't capture.

**Consequence, applied per the identical standing rule already used for ukiyo-e's original
failure — not a softer standard invented for this domain:** Pattachitra's `style_adherence` axis
cannot be trusted to measure genuine style authenticity, even under the corrected question.
**Every Pattachitra `style_adherence` number — §7.3's resolved checkpoint-500/1000 rows below, and
the weight-sweep's operating-point table above — is marked PROVISIONAL, parallel to ukiyo-e's
original marking.** Unlike ukiyo-e (§4.10, where the redesigned control PASSED and lifted the
PROVISIONAL marking), Pattachitra's PROVISIONAL marking is **not lifted** by any test run in this
pass — its control failed even under the fair, corrected version. `figure_preservation` is
**unaffected** by either finding (never implicated in the base-arm confound or this positive-
control result) and remains the trusted basis for this adapter's guardrail verdict. See §7.4 for
how this changes (and does not change) the publication decision, and §7.7 for the standalone
methodology writeup.

### 7.3 Results — n=90 paired, curated LoRA vs. `sdxl_base`, all three checkpoints

**`style_adherence` values below are RESOLVED from the judge-question bug (scored under the
corrected question) but PROVISIONAL for a second, independent reason — see the addendum above:
Pattachitra's own positive control fails even under this corrected question, so these numbers are
not confirmed to measure genuine style authenticity. `figure_preservation` and `artifact_absence`
are unaffected by either issue.**

(`reports/pattachitra_ab_base_comparison.json`, `scripts/compute_pattachitra_ab_stats.py` →
`reports/pattachitra_ab_stats.json`. 0 errors across 270 generations and 810 independent VLM calls
across all three checkpoints; one CUDA context corruption mid-run on the checkpoint-500 attempt and
one script bug — a retry that appended a duplicate record instead of replacing the errored one —
were both caught, root-caused, and fixed before this table was produced, not smoothed over. A
follow-up audit then specifically checked whether the pre-crash images or the fix itself left a
residual artifact in this table — see §7.2(4): verified clean on both counts, no regeneration
needed.)

**`style_adherence` for checkpoints 500 and 1000 is RESOLVED below — re-scored uniformly with
`PATTACHITRA_STYLE_QUESTION` (`reports/pattachitra_uniform_rescore.json`). Checkpoint 1500 is
NOT part of the weight sweep's scope (it was never swept, by the original design decision) and
remains VOID — not re-scored in this pass; this is a deliberate, disclosed scope boundary, not an
oversight, per the standing directive's cap against extending into additional checkpoints.**

| Checkpoint | Endpoint | Base mean | Arm mean | Diff | SEM | diff/SEM | MDE@80% |
|---|---|---|---|---|---|---|---|
| **500** | `style_adherence` **(resolved)** | 0.8883 | 0.8472 | **−0.0411** | 0.0137 | **−2.996** | 0.0384 |
| | `figure_preservation` | 0.9767 | 0.9539 | **−0.0228** | 0.0041 | **−5.532** | 0.0115 |
| | `artifact_absence` | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0/0 | 0.0000 |
| **1000** | `style_adherence` **(resolved)** | 0.8883 | 0.7883 | **−0.1000** | 0.0217 | **−4.603** | 0.0609 |
| | `figure_preservation` | 0.9767 | 0.9322 | **−0.0444** | 0.0057 | **−7.768** | 0.0160 |
| | `artifact_absence` | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0/0 | 0.0000 |
| **1500** | `style_adherence` **(VOID — out of scope, see note above)** | ~~0.3533~~ | ~~0.2928~~ | ~~−0.0606~~ | ~~0.0271~~ | ~~−2.234~~ | ~~0.0760~~ |
| | `figure_preservation` | 0.9767 | 0.9306 | **−0.0461** | 0.0064 | **−7.226** | 0.0179 |
| | `artifact_absence` | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0/0 | 0.0000 |

*(Checkpoint 1500's entire `style_adherence` row — including base mean — is struck through, not just
the diff: 1500 was never part of the uniform re-score's paired sample, so its base-mean pairing
predates the corrected question and does not carry the resolved 0.8883 figure that 500/1000 share.
Left as void rather than backfilled with 0.8883, since that base-mean figure was never actually
re-measured against a checkpoint-1500 pairing.)*

**`style_adherence` (primary A) — RESOLVED from the judge-question bug for 500/1000, but
PROVISIONAL for a second, independent reason (§7.2 Addendum: Pattachitra's own positive control
fails even under this corrected question) — reported here as measured, with that caveat, not
retracted:** checkpoint-500 regresses −2.996×SEM (previously an insignificant +0.477×SEM under the
wrong question); checkpoint-1000 regresses −4.603×SEM (previously −0.789×SEM, also insignificant).
Both are large, individually-significant regressions once measured with the correct question —
`sdxl_base` itself scores 0.8883 on the corrected axis (a genuinely different reference point than
the void 0.3533 figure, which measured something else entirely: "does this look like ukiyo-e") —
**but whether either regression reflects genuine style authenticity or the same instrument
confound that failed Pattachitra's positive control is now an open question.** `figure_preservation`
is unaffected by this caveat and is the trusted basis for this section's guardrail conclusion. See
the weight-sweep table in §7.2(5) above and §7.4 for how this changes the publication reassessment.

**`artifact_absence` (primary B) is a genuine positive finding, not an uninformative null — see
§7.5.** Zero variance across all 360 independent-regime scores (90 base + 90×3 curated checkpoints)
retroactively cross-validates the ukiyo-e curation project's own scoping.

**`figure_preservation` (guardrail) REGRESSED decisively at every checkpoint** (−5.5 to −7.8×SEM,
all far exceeding their own MDE — large, robust effects, not borderline ones, and confirmed not to
be a checkpoint-selection or eval-mechanics artifact per §7.2). **Root cause, visually confirmed,
not just inferred from the statistic:** `pat_009` ("a farmer plowing a field with oxen"), same
prompt and seed (42), checkpoint-1500 vs. base — `sdxl_base` renders the farmer clearly and
coherently; the curated LoRA's output **omits the human figure entirely**, showing only loose
animals against a decorative background. A plausible (not proven) contributing factor: **BLIP's
auto-generated captions for the training set are frequently generic and low-information**
("painting of a group of people in a building with a man and woman," "a close up of a wall with
many paintings on it") — weaker text-image conditioning signal than ukiyo-e's captions received,
which may have made it harder for a rank-8 LoRA at 100 images to learn robust subject-preserving
associations across this evaluation's more compositionally-demanding 30-prompt set. This is a
hypothesis for future work, not a claim requiring no further testing.

### 7.4 Verdict — do NOT publish at `weight=1.0`; do NOT publish at `weight=0.3–0.5` either, but for a different reason — the style claim needed to justify it has failed its own validity check

**At `weight=1.0` — the only scale this adapter has been evaluated at or would be deployed at
before this diagnostic — the verdict is unchanged and unaffected by anything found in §7.2's
Addendum: neither checkpoint demonstrates a style-adherence lift over `sdxl_base`; both
significantly *regress* it** (checkpoint-500: −2.996×SEM; checkpoint-1000: −4.603×SEM), **and
every checkpoint significantly regresses figure/subject-preservation vs. `sdxl_base`** (visually
confirmed: prompted human figures sometimes go missing entirely). **Do not publish at `weight=1.0`
— it fails on both endpoints.**

**Reassessed per this diagnostic's own later finding, not the original TERMINATION rule that
closed this question before the operating-point data existed:** §7.2(5)'s weight sweep found four
points (both checkpoints, weight 0.3 and 0.5) where `figure_preservation` does not regress — and
in fact modestly improves — over `sdxl_base`, while `style_adherence` shows a large, individually-
significant diff/SEM (up to +3.6). Taken at face value, this would be a viable operating point
worth proposing for publication at a documented low weight. **It is not proposed here, for a
specific, falsifiable reason found in the course of this same reassessment, not institutional
inertia:** the `style_adherence` axis has since failed its own positive control for Pattachitra,
decisively and wrong-signed, even under the corrected question (§7.2 Addendum) — real Pattachitra
reference art scores *lower* than `sdxl_base`'s own generated attempts by −3.781×SEM. **An
instrument that cannot correctly rank authentic reference art above a generated attempt at the
same style cannot be trusted to certify that a LoRA-adjusted generation is a genuine style
improvement over a non-adjusted one, even when the specific diff/SEM number is real and
well-powered.** The joint criterion's `style_adherence` half is therefore not currently
certifiable, independent of what number it reports.

**What IS trusted and does support a narrower, honest claim:** `figure_preservation` was never
implicated in either the base-arm confound or the positive-control failure — its own regression at
`weight=1.0` and recovery at `weight=0.3–0.5` stand as measured. **The defensible statement this
diagnostic supports is: at weight 0.3–0.5, this adapter does not measurably harm subject/figure
preservation relative to `sdxl_base`, but this project cannot currently verify, with its own
judge instrument, that it improves Pattachitra style authenticity either — the two claims that
together would justify publishing are not both available.** Publishing on a "figure-safe" basis
alone, without a certified style claim, is not proposed either — a LoRA whose only verified
property is "does not make figures worse" is not a positive case for adding the adapter over
simply not deploying it.

**Do not retrain to chase a better-looking number from either finding** — both are reported as
evidence, not treated as bugs to iterate away. If a Pattachitra LoRA is revisited, the evidence
here points to concrete, testable next steps before another GCP run or another publication
attempt — none authorized or attempted in this pass:
(1) **Validate (or replace) the `style_adherence` instrument for this domain first** — the
positive-control failure is itself the highest-priority open question; a different question
phrasing, a different judge model, or a non-VLM style metric would need to pass its own positive
control before any style-lift claim from it could be trusted, at any adapter weight;
(2) **adapter-weight tuning at deployment time** — still the cheapest, already-available lever
once (1) resolves; requires no new training;
(3) more specific, human-reviewed or VLM-generated captions in place of generic BLIP output, since
caption quality is a plausible contributor to the subject-dropping failure at full weight;
(4) a larger and/or more compositionally diverse training corpus, since 100 images (though larger
than ukiyo-e's 23) still measurably underperformed on a 30-prompt set spanning more varied
compositions than the corpus's own auto-captioned distribution suggests it was rich in;
(5) a rare-token retrain (§7.2's minor, unproven trigger-token risk factor) as a cheap,
low-priority thing to rule out if (3) and (4) are tried first and the regression persists.
**None of (1)–(5) is authorized or attempted in this pass** — reported as the evidence-grounded
next step for a future, separately-authorized session. No retrain, no new checkpoint, no new
domain follows from this reassessment, per the standing directive's unchanged cap.

### 7.5 A positive, cross-domain finding: the artifact_absence ceiling retroactively validates the
ukiyo-e curation's scoping

`artifact_absence` scored exactly 1.0 across all 360 independent-regime Pattachitra scores (base +
three LoRA checkpoints, zero variance) — confirming the text/calligraphy-artifact problem that
motivated the entire ukiyo-e curation project (`docs/MODEL_VERDICT.md` §4) is specific to
ukiyo-e/WikiArt, not a generic SDXL-style-LoRA failure mode. Pattachitra prompts never evoke SDXL's
text/caption-artifact tendency in the first place, in any arm — so the ukiyo-e project's decision to
scope its curation filter specifically around embedded text/calligraphy (rather than, say, a
generic "any defect" filter) targeted a real, style-specific risk rather than a universal one.

### 7.6 Portfolio-level pattern (the Pattachitra `figure_preservation` negative survives §7.2, so this is reported)

**Both sides of this section's cross-project `style_adherence` comparison carry a caveat, for
different reasons — this has REVERSED since the last checkpoint: now resolved/confirmed on the
ukiyo-e side, newly provisional on the Pattachitra side.** Ukiyo-e's positive control was
redesigned and now PASSES decisively against two off-style contrasts (§4.10) — its
`style_adherence` numbers, including the base `sdxl_base` score of 0.9389, are CONFIRMED, not
provisional. Pattachitra's base score is resolved from the judge-question bug (0.8883, corrected
question) but **a base-arm confound found while fixing the positive control revealed that
Pattachitra's own control FAILS, decisively, even under the corrected question** (§7.2 Addendum) —
its `style_adherence` numbers, including this 0.8883 base score and the weight-sweep operating
points, are PROVISIONAL for this independent reason. This section's numeric comparison is
therefore stated with the caveat now on the Pattachitra side, not the ukiyo-e side.

With that caveat: ukiyo-e (base `sdxl_base` `style_adherence` 0.9389, confirmed) and Pattachitra
(base `style_adherence` 0.8883, resolved from the question bug but provisional pending its own
failed positive control — SDXL renders this style at a score much closer to ukiyo-e's own
confirmed baseline than the original void 0.3533 figure suggested, though how much of that
apparent closeness itself reflects genuine style rendering vs. the same instrument confound that
failed Pattachitra's control is now an open question, not settled by this number alone). **A rank-8
LoRA retrain did not close either gap on the trusted axis — both checkpoints significantly
*regress* `figure_preservation` at the deployed `weight=1.0`** (§7.3), while `style_adherence`'s
own regression at that weight (−2.996×SEM, −4.603×SEM) is numerically real but subject to the same
instrument-validity caveat as every other Pattachitra `style_adherence` figure. **The more
defensible, evidence-grounded synthesis:** at this small-corpus
(23–100 image), rank-8, zero-cost-local training scale, a guardrail-type regression appeared in
both projects at full adapter weight — ukiyo-e's own primary endpoint failed to clear its
promotion bar too (§4.6), just without a comparably severe guardrail failure, and (unlike ukiyo-e)
Pattachitra's regression is recoverable at a lower adapter weight (§7.2(5)) — a portfolio-level
difference worth flagging: the two projects' negatives are not identical in character, one
(Pattachitra) is dosage-dependent and the other (ukiyo-e) has not been tested at partial weight.
**What both projects do support unambiguously: pre-test `sdxl_base`'s zero-training rendering of a
candidate style before committing any training spend** — this is exactly how Mughal miniature and
Warli were correctly disqualified in `docs/NEXT_MODEL_SPEC.md` §2 (a direct, zero-LoRA prompt test
found SDXL already renders both styles convincingly, making further training low-value) without
spending a single GPU-hour, and the same style of check would have surfaced Pattachitra's now-
resolved 0.8883 base score *before* this training run — **a materially different, and materially
more important, data point than the void 0.3533 figure that originally motivated training a
Pattachitra adapter at all.** SDXL was not rendering Pattachitra poorly (the premise the original
model selection reasoned from) — it was already rendering it reasonably well, at a score much
closer to ukiyo-e's own well-represented-style baseline than the wrong-question measurement
suggested. This does not retroactively prove training was unwarranted (a good base score doesn't
guarantee a LoRA will fail to add value, as ukiyo-e's own more modest gap shows), but it is a
materially different starting premise than the one this project actually trained against, and is
disclosed as such rather than left standing uncorrected. The practical lesson stands regardless of
which number is correct: a cheap, mandatory first data point any next-model selection should have
on record, measured with the *right* question, before spend — alongside a guardrail-inclusive
evaluation design from the start (§7.2's checkpoint-selection methodology, not a single-checkpoint
spot check).

| Family | Verdict | Basis |
|---|---|---|
| Pattachitra LoRA — curated retrain, weight=1.0 (deployed/evaluated scale, all checkpoints) | **not published at weight=1.0 (fails both axes); not proposed at weight=0.3–0.5 either — the style claim needed to justify it is PROVISIONAL, not confirmed (§7.4)** | §7.3: `figure_preservation` −5.5 to −7.8×SEM at 500/1000/1500 (large, robust, confirmed not a checkpoint-selection artifact, §7.2; TRUSTED, unaffected by either measurement issue below). `style_adherence` resolved from the judge-question bug for 500/1000 (−2.996×SEM, −4.603×SEM; 1500 remains void, out of scope) but **PROVISIONAL for a second, independent reason: Pattachitra's own positive control fails decisively even under the corrected question (§7.2 Addendum)**. `artifact_absence` uninformative for gating (ceiling effect) but retroactively validates ukiyo-e's curation scoping (§7.5). A `style_adherence` operating point exists numerically at `weight=0.3–0.5` (§7.2(5)) but is not proposed for publication — the metric certifying it has itself failed its own validity check for this domain. Do not retrain to chase a better number — both findings are reported as evidence, not treated as bugs to iterate away. |

### 7.7 Methodology finding: five silent measurement failures across one evaluation project

**This is reported as a standalone, portfolio-worthy result about evaluating generative models,
not as a footnote to the Pattachitra verdict above.** Over the course of this single evaluation
project, five independent measurement failures were found — each silent (produced no error, no
crash, no obviously-wrong output at the time), each capable of standing unnoticed as a real
finding if unaudited, and four of the five caught only by going back and auditing a *prior audit's
own conclusion*, not by any check running at measurement time. Each subsequent check exists
specifically because the prior ones did not generalize to catching the next.

| # | Bug | Discovered by | What it would have caused, unaudited |
|---|---|---|---|
| 1 | **`num_ctx` judge failures** — Ollama's default served context window (4096 tokens) is too small for a judge prompt plus a high-resolution image's token count. | A real `400 exceeds the available context size` error, hit during Pattachitra corpus curation — not predicted or assumed in advance. | Every judge call for a sufficiently large prompt+image would fail outright — not a wrong score, a **missing** one — silently reducing `n` for whichever records happened to exceed the window, understating true sample size without a corresponding downward revision of any confidence claim. |
| 2 | **Phantom VRAM counter** — `torch.cuda.max_memory_allocated()` reported an identical `11.186 GB` peak across three separate runs on an 8.589 GB local card, a physically impossible reading. | Manual forensic investigation (`docs/LATENCY_ROOT_CAUSE.md`), triggered by an unexplained 5.6× latency variance — not by any automated check. Root cause: a stale, never-reset counter (missing `reset_peak_memory_stats()`, since fixed in `scripts/eval.py`) carrying over a peak from an earlier, larger run (plausibly a contended GCP L4, not the local card at all). | Taken at face value, "every run used 11.186 GB" would have been reported as a hardware-fit finding for a card that cannot physically hold that much — a false capacity/fit conclusion built on a number that was never really being measured fresh each run. |
| 3 | **CUDA context corruption** — a `curated500` generation attempt crashed with `RuntimeError: CUDA error: an illegal memory access was encountered` after 22/90 images; the poisoned context then failed all 68 remaining attempts in the same process. | Caught by a *skeptical re-audit of a prior turn's own conclusion* — the prior turn had already accepted the Pattachitra `figure_preservation` finding; this turn specifically asked whether the crash could have silently degraded the 22 images generated *before* it. | Left unaudited, the working assumption would have been "the crash only cost 68 failed attempts, nothing else" — an assumption never actually verified, resting on architectural reasoning ("a crash can't retroactively corrupt already-saved data") rather than the direct pixel-level/visual confirmation this project actually did. |
| 4 | **Hardcoded judge question** — `style_adherence`'s judge prompt was hardcoded to ask about ukiyo-e regardless of caller; Pattachitra scripts reused it unmodified, so all 360 Pattachitra `style_adherence` records asked about the wrong style. | Direct code inspection while *designing a positive-control script* — found before any Pattachitra-domain result from that control was read, not by any of the three integrity checks (CUDA health probe, degenerate-image detection, uniqueness assertion, judge-range validation) built after bug #3. All four of those checks passed cleanly on every one of the 360 affected records. | A confidently-reported "no style-adherence lift, and no significant effect either way" finding — a plausible-looking null result that was in fact measuring an entirely different question, indistinguishable from a real null without the code-level check that finally caught it. |
| 5 | **Inconsistent reference arms across two analyses of the same data** — the judge positive control's `sdxl_base` scores were loaded once from an OLD file (scored under bug #4's wrong question) and reused unchanged for a second, supposedly-independent comparison that should have used a freshly-corrected score instead. | Direct code inspection of `scripts/judge_style_positive_control.py`, prompted by a genuine cross-document contradiction (the SAME quantity — `sdxl_base` Pattachitra `style_adherence`, n=90 — reported as two different numbers, 0.3533 and 0.8883, in two places) — not caught by the domain-parameterization test built for bug #4, since the *question text* was never wrong in this script; the *data reused* was stale. | A confident, decisive-looking PASS (`diff/SEM = +12.005`) that never actually compared two arms scored under the same question — the "corrected prompt" row silently mixed a freshly-scored real-art arm with a stale, wrong-question base arm. The corrected, fair comparison reverses the verdict entirely (FAIL, `−3.781×SEM`), changing the Pattachitra publication reassessment in §7.4. |

**The pattern, not just the count, is the finding.** Bugs #2 and #3 were caught by manual,
one-off forensic investigation triggered by something looking *anomalous* (an impossible number,
a suspicious effect size) — they depended on a human noticing something felt wrong enough to dig
in. Bug #4 was caught by neither an anomaly nor an automated check — it was caught by directly
reading the source code of the measurement instrument itself while building an unrelated new
tool. Bug #5 was caught by yet another mode: a cross-document contradiction (two different numbers
reported for what should have been the identical quantity) that prompted a direct read of the
script producing one of them — the domain-parameterization check built specifically in response to
bug #4 did not generalize to this defect, because bug #5 is not a wrong *question*, it is stale
*data* silently reused across two comparisons that should have been independent. **Every automated
integrity check this project built in response to bugs #1–#3 (CUDA health probe, degenerate-image
detection, uniqueness assertion, judge-score range validation) checks *value* validity, and the
check built for bug #4 checks *question* validity (does the actual request text match the domain
under test) — none of them can catch a *reused, stale result from a different, valid measurement*
being silently substituted into a new comparison.** This is a distinct defect class again, and one
general lesson from this project is that it likely needs a further class of check (e.g., an
automated audit that flags when the same logical quantity — same model, axis, and `n` — is
computed more than once across a project's report files and asserts the values agree, or traces
to a single shared source, before either is used in a downstream conclusion) rather than assuming
better question- and value-validation alone will eventually catch it. No such check exists yet;
adding one is noted here as a candidate for future harness work, not implemented in this
diagnostic pass — implementing it would extend past this session's capped scope.
