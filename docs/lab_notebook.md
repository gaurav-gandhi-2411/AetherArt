# Lab Notebook

A dated research log. Only entries where something was learned or decided — not a transcript of every commit. Retrospectively written from git history, experiment outputs, and notes taken during the work.

---

## Early April 2026 — Initial SD 2.1 pipeline

First question: will SD 2.1 run at all on 8 GB VRAM? The `stabilityai/stable-diffusion-2-1` weights are ~5 GB; adding the VAE and text encoder puts the full model at ~8 GB. It fits in VRAM, but there's no headroom for anything else.

`enable_model_cpu_offload()` was the key decision. Model CPU offload moves the U-Net's transformer blocks to CPU between forward passes, keeping only the active block in VRAM. Peak VRAM drops to ~3.1 GB. Latency goes from ~2.5 s/img to ~3.2 s/img — the cost of the extra data transfers. Acceptable for research; unacceptable for a commercial API.

The model ID switched from `stabilityai/stable-diffusion-2-1` to `sd2-community/stable-diffusion-2-1` after the EU AI Act deprecation in early 2026. Same weights, same diffusers API, no code change. The deprecation was announced with one week of notice — keeping the repo pointed at the community mirror rather than the official ID turned out to be good practice.

DPM-Solver++ was chosen as the default scheduler from the start, based on the diffusers documentation's recommendation. The 360-run benchmark later confirmed this was the correct call (Pareto-optimal at 20 and 30 steps), but it was an educated guess at this stage.

---

## Late April 2026 — LoRA training and the calligraphy artifact

Training a rank-8 LoRA adapter on 80 WikiArt Ukiyo-e images took 2 h 8 min on the RTX 3070. Zero OOM events. The training script uses `accelerate launch` with mixed precision fp16 and gradient accumulation of 4 — the accumulation was necessary to simulate a larger effective batch without exceeding VRAM.

The calligraphy artifact emerged immediately. WikiArt images include metadata captions with artist signatures, dates, and script text embedded in the image margin. The LoRA learned this as part of "ukiyo-e style" — every generated image came out with illegible text fragments in the borders. This is a training data problem, not a LoRA architecture problem. The fix was a default negative prompt (`text, watermark, calligraphy, signature, words, letters`) applied automatically whenever the adapter is active. It suppresses most artifacts but doesn't eliminate them completely — the style signal and the text signal are entangled in the adapter weights.

The right fix would be training on a curated dataset where the source images have no text annotations. That's ~5 hours of curation work I didn't do.

The trigger token `ukyowood` was chosen as a nonsense word with no pre-trained meaning in CLIP or SD 2.1's vocabulary. Experiment 9 later confirmed it has exactly zero CLIP footprint — the trigger fires the adapter through the LoRA's learned association, not through any pre-existing semantic path.

---

## Late April 2026 — Checkpoint selection (checkpoint-1000)

Trained to 1500 steps, then evaluated checkpoints at 500, 1000, and 1500. Loss at 1500 ticked up from 0.268 to 0.495 — the classic overfitting signature. Checkpoint-1000 was selected visually: it produced the warm amber palette and characteristic flatness of traditional woodblock prints most consistently across test prompts.

Checkpoint-500 underfit — the style was present but not saturated, more like a mild filter than a transformation. Checkpoint-1500 overfit — outputs had an "over-processed" quality, colors saturated, some prompt alignment breaking down.

This was a visual judgment, not a metric judgment. Experiment 8 (LoRA alpha sweep) later confirmed that CLIP cannot distinguish between these checkpoint stages — all three would have scored similarly on CLIP. The right evaluation tool was looking at the output images, which I did.

---

## Late April 2026 — ControlNet integration and the LRU cache

ControlNet required creating a *separate* pipeline object — you can't bolt conditioning onto an existing SD 2.1 pipeline; the conditioning cross-attention has to be present at construction time. This means Canny and Depth each need their own pipeline, and combining LoRA + ControlNet means loading the LoRA into the ControlNet pipeline rather than the base pipeline.

First attempt: creating a new pipeline on every inference call. This worked but accumulated VRAM quietly — each `from_pretrained` call created a new object and the old one was not explicitly released. On an 8 GB GPU with CPU offload, three pipelines coexisting pushed peak VRAM over threshold.

Solution: 2-entry LRU cache keyed by `(ctype, lora, alpha)`. The third combination evicts the oldest. In practice, users switch between at most 2 modes interactively; 2 entries covers 95% of usage patterns without accumulation. The same eviction logic was applied to the base pipeline registry in Phase 3 when the quantization mode switching had the same accumulation problem.

The VRAM note in the architecture section ("ControlNet runs on a separate pipeline, ~3 GB additional") is a consequence of this design. The LRU prevents the accumulation, but you're always paying for the active ControlNet pipeline.

---

## Late April 2026 — Speed tiers: LCM hit a dead end

The goal was to have three generation modes: standard (30 steps), fast (LCM, 4 steps), and ultra-fast (SDXL Turbo, 1 step). The first two were expected to run on SD 2.1; Turbo required a separate SDXL model.

LCM fast mode was implemented via `LCMScheduler` — a scheduler-only approach that runs SD 2.1's U-Net for 4 steps using the LCM consistency schedule. This gives a 5.3× speedup (0.6 s vs 3.2 s) at some quality cost.

The originally planned approach was LCM-LoRA: a small adapter that adds consistency distillation to SD 2.1 without requiring a full consistency-distilled model checkpoint. After spending time searching for an SD 2.1 LCM-LoRA, I confirmed it doesn't exist. The consistency distillation for SD 2.1 was done as a full model checkpoint (not an adapter); for SDXL, it was done as a LoRA. The asymmetry is an artifact of when each model's consistency work was published and for whom. The scheduler-only approach for SD 2.1 is the correct solution — LCM-LoRA was a dead end.

---

## Late April 2026 — Quantization: the INT8 VRAM reversal

Measured fp16, INT8, and NF4 peak VRAM under `enable_model_cpu_offload()`:
- fp16: 1803 MB
- INT8: 2210 MB (+407 MB vs fp16)
- NF4: 1382 MB (−421 MB vs fp16)

INT8 using *more* VRAM than fp16 was unexpected. The cause, after investigation: bitsandbytes needs a full fp16 compute buffer for dequantization during the forward pass. Under CPU offload, this buffer is allocated on GPU at inference time even though the stored 8-bit weights are smaller. The stored weights save VRAM; the compute buffer costs more than is saved. On a card with enough VRAM to load the full model at once (no CPU offload needed), INT8 would recover its stored-weight savings and show a real reduction. On 8 GB with CPU offload, it does not.

NF4's savings survive because the stored 4-bit weight footprint is smaller than fp16 even after the compute buffer is added — the math works out differently at 4-bit compression ratios.

The practical outcome: on this hardware path, INT8 has no compelling use case. It uses more VRAM, runs 2.8× slower, and has the same CLIP score as fp16. NF4 is the quantization mode worth using if VRAM is genuinely constrained — it saves 421 MB at 1.5× the latency.

---

## April 25, 2026 — 360-run CLIP benchmark

Design: 4 schedulers × 3 step counts × 30 PartiPrompts = 360 generations, fixed seed 42, all at 512×512 with DPM-Solver++ as the scheduler (then rerun with all four schedulers).

Hypothesis going in: DPM-Solver++ would win on CLIP, and 50 steps would produce the best scores within each scheduler.

What the data showed:
- Prompt choice matters 18× more than scheduler choice. This was not expected — I went in thinking schedulers would have measurable impact. They do (DPM leads at 0.3177, LMS trails at 0.3117), but the range of 0.007 is dwarfed by the prompt-to-prompt range of 0.130.
- 30 steps slightly outperforms 50 for DPM-Solver++ (0.3199 vs 0.3165). Step count almost doesn't matter above 20.
- DPM@20 reaches DPM@30 quality within noise at 24% less wall time.

The 18× finding is the headline. The practical implication — stop spending time on scheduler tuning and spend it on prompt engineering — is the more actionable version of it.

---

## Late April / Early May 2026 — CPU Space pivot

Deployed to Hugging Face Spaces expecting ZeroGPU access. ZeroGPU was unavailable on the free tier. Options: pay for a GPU Space ($0.60/hr A10G), or run CPU-only.

Chose CPU-only. The generation time is 8–15 min on the free CPU tier — too slow for real use but adequate as an architecture demo. Disabled LCM and SDXL Turbo (both require GPU for meaningful performance), added honest generation time disclosures, locked resolution to 512×512.

The wrong assumption was expecting ZeroGPU to be available. The Space was designed around GPU availability before the deployment path was confirmed. Should have checked first.

---

## May 2026 — Phase 6b: the CLIP-blindness theme emerges

Experiments 1, 2, and 3 were designed to test different parameters. They were not designed to find a common theme. By Experiment 3, the pattern was unavoidable: every CLIP delta was within 1 SE while LPIPS was in the 0.40–0.47 range. Three null results in a row, with LPIPS consistently large.

The theme was named "CLIP-blindness" after Experiment 3. Experiments 4–9 were then framed explicitly as tests of whether the pattern held in new parameter domains. It held in all but one partial case (Experiment 8, where CLIP partially registered the style switch because the prompts named the style).

This is the kind of result that emerges from running experiments rather than predicting them. The original Phase 6b plan listed seven experiments with no particular hypothesis about metric behavior. The LPIPS metric was added mid-series (after Experiment 1 showed the first CLIP null result) as a complementary probe. Without LPIPS, the series would have been nine null results with no explanation for why CLIP stayed flat. With LPIPS, the null CLIP results become a finding.

---

## May 8, 2026 — Recovery: the experiment substitution incident

Discovered that Experiments 6 and 7 had been run with the wrong designs — an alpha sweep and a trigger ablation instead of the planned rank ablation and data size ablation. See [`reports/what_didnt_work.md`](../reports/what_didnt_work.md) for the full account.

Recovery took most of the day:
1. Renamed the misplaced experiments to 8 and 9 (13 git mv operations, updated all cross-references).
2. Retrained rank-4 and rank-16 adapters (~2h each; rank-8 already existed).
3. Built 20 and 40-image subset directories from the 80-image dataset.
4. Ran the 90-image inference grid for each ablation.

The rank-4 training ran correctly (~2 h). The rank-16 training also ran correctly. The 20-image data subset training ran for 4 h 10 min due to the validation_epochs=1 overhead (see [what_didnt_work.md](../reports/what_didnt_work.md)).

The conda run process reported failure on both experiment scripts due to a Windows cp1252 encoding crash while printing tqdm progress bars after completion. The scripts completed successfully — confirmed by reading the training logs directly and checking that all output files were present and correct.

---

## May 8, 2026 — The underfitting paradox

When Experiment 6 results came in (rank-4 CLIP > rank-8) and Experiment 7 results came in (data-20 CLIP > data-80), the same counterintuitive direction appeared in both. This is the underfitting paradox: the smaller, less-trained adapter scores higher on the quality metric than the more capable one.

The mechanistic explanation: an underfit LoRA learns a simple, literal representation of "ukiyo-e style" — essentially the average of the keyword's appearance across its small training set. When CLIP scores the image against the prompt "ukiyo-e woodblock print style," the simpler representation matches the keyword more directly. The fully trained adapter learns subtler style properties — Hokusai's specific color relationships, traditional compositional conventions, characteristic flatness of form — that do not map to any tokens in CLIP's vocabulary.

CLIP rewards literalness. Style quality is not literal. The metric is optimizing in the wrong direction for style transfer evaluation.

This was the sharpest illustration in the series of why CLIP cannot guide LoRA training decisions. The finding belongs in the project's main conclusions, not just in the experiment reports.

---

## June 2026 — Phase 7: SDXL CLIP-blindness replication (PR 13)

Re-ran all 9 Phase 6b experiments on SDXL base (`stabilityai/stable-diffusion-xl-base-1.0`) on a GCP L4. L4 stockouts across the ~11 h run required three separate VMs; exp6 and exp7 were skipped due to a setup gap (training dataset not staged on the eval VM — recoverable, not lost).

Result: **WEAK REPLICATION — 3/7 experiments CLIP-blind on SDXL** (exp1 quantization: 0.24 SE; exp4 scheduler: 0.67 SE; exp9 trigger token: 0.84 SE). SDXL CLIP responds to semantically meaningful sweeps that SD 2.1 missed — CFG scale jumped from 1.10 SE on SD 2.1 to 7.01 SE on SDXL; LoRA alpha from 4.00 SE to 7.21 SE. The blindness that remains on SDXL is confined to rendering-level changes that don't alter semantic content.

The more important finding came from comparing baselines. Re-running forced a re-examination of the SD 2.1 "9/9 blind" claim: that headline used a qualitative CLIP-vs-LPIPS ratio judgment, not a hard threshold. Under the same 1-SE statistical cutoff applied to SDXL, the SD 2.1 corpus recomputes to **4/9 blind** — exp3 (1.10 SE), exp4 (1.80 SE), and exp5 (2.20 SE) were called blind qualitatively because LPIPS was large, not because CLIP was statistically flat. The sensitivity table in `reports/clip_blindness_sdxl.md` shows SDXL is less blind than SD 2.1 at every threshold tested (3–5/7 vs 4–8/9), so the direction of partial replication is robust, but the SD 2.1 baseline headline was overstated.

**Corrected headline:** CLIP-blindness is real on SD 2.1 for rendering-level changes (quantization, trigger token) and borderline for semantic sweeps (negative prompt, scheduler). On SDXL it is substantially weaker: the architecture's stronger CLIP alignment means semantic sweeps register clearly above noise. The finding is architecture-dependent and more nuanced than first reported.

---

## Phase 7 carry-over — deferred ruff violations

<!-- TODO(PR-14): ruff cleanup deferred — fold ruff --fix --unsafe-fixes into PR 14 (coverage threshold enforcement). These are stylistic, not bugs. -->

PR 02 enabled `E, F, I, W` rules only. `UP`, `B`, and `SIM` rules found 20 pre-existing violations that must be fixed before those rule sets can be activated. Consultant decision (2026-05-19): defer to PR 14, which already touches the package broadly for coverage gap-filling.

**`aetherart/model.py` — UP rules (8 hits)**
- `UP035` line 5: `from typing import … Dict` — deprecated, use `dict`
- `UP006` line 51: `-> Dict[str, Any]` → `-> dict[str, Any]`
- `UP006` line 57: `kwargs: Dict[str, Any]` → `kwargs: dict[str, Any]`
- `UP006` line 83: `self.optimizations: Dict[str, str]` → `dict[str, str]`
- `UP045` line 34: `-> Optional[str]` → `-> str | None`
- `UP045` line 51: `hf_token: Optional[str]` → `str | None`
- `UP045` line 76: `model_id: Optional[str]` → `str | None`
- `UP045` line 78: `self.hf_token: Optional[str]` → `str | None`
- `UP045` line 81: `self.backend: Optional[str]` → `str | None`
- `UP045` line 232: `seed: Optional[int]` → `int | None`
- `UP037` line 233: `-> "Image.Image"` — remove quotes from annotation

**`aetherart/controlnet.py` — SIM rules (1 hit)**
- `SIM105` line 162: `try/except Exception: pass` → `with contextlib.suppress(Exception):`

**`aetherart/lora.py` — SIM rules (1 hit)**
- `SIM105` line 41: `try/except Exception: pass` → `with contextlib.suppress(Exception):`

**`scripts/` — B and SIM rules (8 hits, lower priority)**
- `B905` `scripts/benchmark_quantization.py:42`: `zip()` missing `strict=`
- `B905` `scripts/eval.py:330`: `zip()` missing `strict=`
- `B905` `scripts/generate_hero_image.py:79`: `zip()` missing `strict=`
- `B904` `scripts/_diffusers_train_text_to_image_lora.py:637`: raise without `from`
- `B007` `scripts/_diffusers_train_text_to_image_lora.py:889`: unused loop var `step`
- `SIM910` `scripts/_diffusers_train_text_to_image_lora.py:683`: `.get(x, None)` → `.get(x)`
- `SIM102` `scripts/_diffusers_train_text_to_image_lora.py:977`: nested `if` → single `if … and …`
- `SIM102` `scripts/_diffusers_train_text_to_image_lora.py:1025`: same
- `SIM114` `scripts/_gen_findings_charts.py:77,81`: combinable `elif` branches

**`aetherart/sdxl_pipeline.py` + all callers — diffusers deprecation warning (PR 03)**
- diffusers 0.35.1 emits `` `torch_dtype` is deprecated! Use `dtype` instead! `` from `StableDiffusionXLPipeline.from_pretrained` (and likely SDXL Turbo, Flux). Surfaced during PR 03 smoke test. Not actionable until we upgrade to a diffusers version that accepts `dtype=`. When upgrading diffusers in PR 14 or a dedicated dep-bump PR, swap all `torch_dtype=torch.float16` kwargs to `dtype=torch.float16` in `sdxl_pipeline.py`, `sdxl_turbo.py`, and any future Flux loader. Same fix applies to `model.py` line 122 (SDXL path in `AetherModel.init`).

**Depth estimator split — doc/model-card pass (PR 08)**
- The SDXL and legacy SD 2.1 depth ControlNet paths now use different estimator models. Any doc or model card that mentions the depth estimator must distinguish:
  - Legacy SD 2.1 depth ControlNet (`aetherart/controlnet.py`): `Intel/dpt-hybrid-midas` (frozen, unchanged)
  - SDXL depth ControlNet (`aetherart/controlnet_sdxl.py`): `LiheYoung/depth-anything-small-hf`
- Reason for swap: `Intel/dpt-hybrid-midas` ships PyTorch `.bin` weights; transformers' CVE-2025-32434 patch blocks `torch.load` on torch < 2.6. `depth-anything-small-hf` uses safetensors and is a stronger estimator. Fix landed in PR 08 (`e3e4e6a`).
- When writing the README depth section and HF model card in PR 14: name both models explicitly with their scope.

**ControlNet local latency note (PR 08)**
- Unquantized FP16 SDXL + ControlNet + `enable_model_cpu_offload` peaks at 7928 MB VRAM (right at the 8 GB ceiling) and runs ~275–292 s/image locally (offload-bound). This config is not suitable for timed local work.
- PR 14: document that the local ControlNet path requires either the NF4 quantized variant (future) or a GPU with > 8 GB VRAM for usable latency. Modal A10G (24 GB) will be fine for PR 09.

**numpy `__array__` copy-kwarg deprecation warning (PR 08)**
- `diffusers/schedulers/scheduling_euler_discrete.py:405` emits `DeprecationWarning: __array__ implementation doesn't accept a copy keyword`. Upstream diffusers issue; not actionable locally. Add to dep-bump watchlist for PR 14: confirm fixed in diffusers ≥ 0.36.

---

## 2026-05-31 — SDXL Ukiyo-e LoRA training on GCP L4 (PR 09)

First GCP-compute PR. Ran a rank-8 SDXL LoRA at 1024×1024 on a g2-standard-4 (NVIDIA L4 24 GB) in `review-iq-prod/us-central1-a`. Training data: same 80 WikiArt ukiyo-e images used for the SD 2.1 LoRA in PR 04 — same rank, same steps, different resolution, for a controlled cross-resolution comparison.

**Infrastructure note:** intended project is `aetherart-497918` (created for this work), but new GCP projects have a `GPUS_ALL_REGIONS=0` cold-start gate that requires 24–48h billing history before auto-approval. Hit the gate mid-run; pivoted to `review-iq-prod` (established, L4 quota already granted). `aetherart-497918` stays as the long-term home; future runs will use it once quota propagates. Full teardown and isolation rules applied: every resource prefixed `aetherart-`, labelled `project=aetherart,ephemeral=true`, zero non-aetherart resources touched.

**Wall-clock reality vs estimate:**
- Runbook estimate: 1.5h. Actual: 4h 26m. Reason: `--validation_epochs 1` triggered 75 SDXL inference passes (one per epoch, 4 images each). Not accounted for in the initial estimate. Future runs should pass `--validation_epochs 10` or higher to cap validation overhead at reasonable levels.

**Cost:** ~$3.50 actual vs ~$1.50 estimated. Under the $5.00 authorized hard stop. Honest note for the training report (portfolio-positive transparency).

**Dependency issue caught during training:** the GCP DLVM image (PyTorch 2.9, transformers 4.51+) removed `FLAX_WEIGHTS_NAME` from `transformers.utils`, breaking `diffusers==0.35.1`'s pipeline loading. Fixed by pinning `transformers>=4.41.2,<4.51` on the VM. Add to PR-14 carry-over: when upgrading diffusers, verify this compatibility constraint is resolved in the target version.

**Checkpoint selection:** evaluated 500/1000/1500. Selected checkpoint-1000 by visual evaluation (same selection as the SD 2.1 run — useful consistency point). Key finding: at 1024×1024, the calligraphy artifact manifests as compositionally integrated cartouche banners and red seals rather than scattered characters, which is more anatomically faithful to real ukiyo-e woodblock prints. checkpoint-1500 showed mild mode-collapse (lost the samurai figure in prompt 3); checkpoint-500 underfitted on figure adherence. The 1000-step convergence point is consistent across both resolutions.

**Adapter:** `data/lora/ukiyo-e/ukiyo-e-sdxl-lora.safetensors`, 45 MB, rank-8, UNet-only. Publication to HF Hub is PR 11.

---

## Phase 7 carry-over — PR 11 and PR 14 deferred notes

<!-- logged 2026-05-31 after PR 09 visual evidence review (consultant approval) -->

**For PR 11 (SDXL Hub publish) — already actioned in the model card:**
- The calligraphy-cartouche-at-1024 finding is in `docs/model_cards/sdxl_ukiyo_e.md`. Frame: "what higher resolution + stronger base prior did to a known 512-era artefact" — the scatter pattern graduates to anatomically placed title cartouches.
- The baseline-vs-LoRA comparison ("LoRA earns its place above the SDXL floor") is in the model card's checkpoint selection section. The baseline had strong ukiyo-e priors; the LoRA's additive value is palette depth and flat-plane treatment, not style creation from scratch.
- The SD 2.1 model card's "Companion SDXL adapter (forthcoming)" placeholder was resolved in this PR — the cross-link now goes both ways.

**For PR 14 (README + coverage pass) — deferred:**
- Repeat the calligraphy-cartouche finding in the README's LoRA section. It's a differentiator that belongs in the public-facing narrative, not just the model card.
- The committed checkpoint-1000 sample PNGs (`reports/lora_training_summary_sdxl_samples/ckpt1000_*.png`) are gallery-ready — reuse as the PR 14 README hero images for the SDXL LoRA section.
- The "baseline control was the right call" note: mention in the README that the eval included a no-LoRA SDXL baseline, which confirmed the adapter adds measurable value above SDXL's pretraining priors.
- When upgrading diffusers in PR 14: verify the `transformers>=4.41.2,<4.51` constraint (surfaced during GCP training) is resolved; swap `torch_dtype=` to `dtype=` in all SDXL pipeline loaders.

---

## 2026-06-01 — PR 12: HPSv2.1 + ImageReward eval harness, GCP validation

**PR 12** wires HPSv2.1 and ImageReward into the scoring harness alongside CLIP (demoted to comparison-only) and LPIPS. The first attempt at local Windows validation hit an immediate blocker: ImageReward's `__init__` imports `ReFL → datasets → pandas → pyarrow.dataset C extension`. On Windows with pyarrow 24.x this causes an access violation SIGSEGV in the C extension before any Python-level try/except can catch it. The import was ungraceful.

**Decision (consultant):** all real eval and benchmarking moves to GCP Linux. Rationale: Linux doesn't have the pyarrow C-extension SIGSEGV, datasets installs and runs cleanly, and CI already runs on Ubuntu. Windows is interactive dev only from PR 12 onward.

This elimination of the Windows pyarrow problem is what PR 12 is doing structurally, not just technically. The scoped `datasets` stub that was added temporarily in commit `4c6c4d2` was removed. On Linux `import ImageReward` is a clean import through the module-level transformers shim — no stub, no monkeypatch.

**GCP validation run (2026-06-01):** g2-standard-4 / NVIDIA L4 24 GB / us-central1-a / review-iq-prod.
- Project: review-iq-prod (confirmed L4 quota; aetherart-497918 new-project GPU cold-start not yet resolved)
- VM: aetherart-eval-001, labels project=aetherart,ephemeral=true, boot-disk auto-delete ON

**Four Ukiyo-e prompts × 4 scorers @ 1024×1024, seed 42, DPM/30 steps:**

| Prompt ID | Prompt | CLIP (cmp-only) | HPS | ImageReward | LPIPS |
|-----------|--------|-----------------|-----|-------------|-------|
| lora_001 | ukyowood ukiyo-e woodblock print of Mount Fuji at sunset | 0.3650 | 0.2175 | 1.6067 | 0.0 |
| lora_002 | ukyowood ukiyo-e print of a crane over ocean waves | 0.3618 | 0.2820 | 1.8456 | 0.0 |
| lora_003 | ukyowood ukiyo-e woodblock print of a samurai in a bamboo forest | 0.3478 | 0.2418 | 1.1382 | 0.0 |
| lora_004 | ukyowood ukiyo-e print of cherry blossoms along a river | 0.3616 | 0.2151 | 1.3237 | 0.0 |
| **avg** | | **0.3590** | **0.2391** | **1.4786** | **0.0** |

Notes: resolution confirmed 1024×1024 for all 4 images (VRAM: 6.22 GB/image). LPIPS=0.0 expected: single image per prompt, no pairwise comparison. CLIP is comparison-only as designed — it can't distinguish ukiyo-e style quality (see CLIP-blindness theme in Phase 6b entries). HPS and ImageReward are the operative quality metrics for this adapter.

**Two Linux-specific install bugs found and documented:**
1. hpsv2 1.2.0 `src/open_clip/factory.py` line 8: `from turtle import forward` — accidental import of Python stdlib turtle graphics, which chains to tkinter. Crashes on headless Linux (no tkinter package). The `forward` name is shadowed by a class method at line 289 and never used from the import. Fix: remove line 8. Documented in eval_hps.py docstring and requirements-dev.txt.
2. hpsv2 bpe vocab file: existing known issue, already documented.

**G1 compliance fix:** `AetherModel.init()` was loading SDXL via raw `StableDiffusionXLPipeline.from_pretrained` without the `madebyollin/sdxl-vae-fp16-fix` VAE — a silent G1 violation that would produce NaN/black images. Fixed to delegate to `load_sdxl_base()`.

**VM cost:** VM ran ~25 min (launch → eval → teardown). g2-standard-4 + L4 in us-central1-a is ~$0.70/hr → estimated **~$0.29** for this run. Well under the $1 estimate and the $5 hard stop.

---

## Phase 7 carry-over — SD 2.1 Hub card wording (deferred to PR 14 docs pass)

<!-- logged 2026-05-31 after SD 2.1 cross-link re-publish -->

The live SD 2.1 Hub card (`gauravgandhi2411/aetherart-ukiyo-sd21`) still reads "Both runs independently select checkpoint-1000" — the original phrasing from before F2 was applied to the SDXL card in PR 11. The two cards are now inconsistent.

**For PR 14:** update the live SD 2.1 card to: "Both visual evaluations independently selected the checkpoint-1000 step count — at 512 for SD 2.1 and 1024 for SDXL." This requires:
1. Edit `docs/model_cards/sd21_ukiyo_e.md` locally
2. Re-run `python scripts/publish_lora_sd21.py` to push the updated card to the Hub

<!-- logged 2026-06-05 after PR 14 merge -->
SD 2.1 Hub card re-push pending — local card reconciled (PR 14, e7fefa7), needs write-token upload via `scripts/publish_lora_sd21.py`.
Same pattern as the cross-link re-publish (no new code, pure Hub I/O action).
