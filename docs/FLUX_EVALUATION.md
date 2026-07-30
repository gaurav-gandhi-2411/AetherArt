# FLUX.1-schnell Evaluation

**Verdict up front: do not add FLUX.1-schnell as a second pipeline or replacement route at this
time.** On the same canonical 30-prompt × 3-seed benchmark used for this project's five other
model families, FLUX.1-schnell ties or loses to `hyper_4step` (the project's own already-adopted
default) on both measured quality axes, runs ~4.6× slower per image, and requires ≥16GB of VRAM
— categorically infeasible on this project's actual 8.59GB deployment target, the same laptop
GPU the whole project is built around. It is evaluated here on measured cost and quality, not
novelty, per this task's own instruction.

---

## 1. Scope and what was actually tested

FLUX.1-schnell (`black-forest-labs/FLUX.1-schnell`, Apache-2.0) was scored on the identical
30-prompt PartiPrompts set × 3 seeds (`scripts/eval_prompts.yaml`, `SEEDS=[42,43,44]`) already
used for `sd21_base`, `sdxl_base`, `hyper_4step`, `hyper_8step`, and `sdxl_controlnet_union`
(`docs/MODEL_VERDICT.md` §2), via the same harness (`scripts/model_verdict_harness.py --family
flux_schnell`), scored on the same two axes those five families use — CLIP and HPSv2.1 — for
apples-to-apples comparability. No VLM-judge axis was scored for FLUX: `style_adherence`
requires a style-domain question, and FLUX here is being evaluated as a base model, not a style
adapter, matching exactly how the five existing baselines were scored (CLIP+HPS only, no VLM
judge).

**FLUX.2-Klein was not evaluated.** It was named in scope on the assumption it was
Apache-2.0-licensed like FLUX.1-schnell; checked directly against the HF API before running
anything, its actual license is `license:other` (a custom BFL license), not Apache-2.0. That
assumption was wrong and is corrected here rather than repeated. FLUX.1-dev and FLUX.2-dev were
correctly excluded per the original scope (paid BFL commercial license) and were not touched.

---

## 2. Local feasibility — measured, not assumed

**Disk.** The diffusers-format FLUX.1-schnell repo requires ~33.7GB of disk (23.78GB transformer
+ 9.53GB text_encoder_2/T5 + 0.25GB text_encoder + 0.17GB vae — HF API file listing, not
estimated). This laptop (RTX 3070, 8.59GB VRAM) had ~25.5–27.4GB free at measurement time — a
hard blocker before VRAM was even reached. No download was attempted locally (it would guarantee
a partial failure and starve the system drive); this is reported as a measured infeasibility, not
assumed.

**VRAM, measured on a GCP L4 (23,034 MiB / ~22.03GB usable) instead:**

| Loading config | Result | Peak VRAM | Latency/image |
|---|---|---|---|
| bf16 + `enable_model_cpu_offload()` (diffusers' documented low-VRAM path) | **`torch.OutOfMemoryError`**, reproduced twice, deterministically | N/A — OOM before the forward pass completed | N/A |
| NF4 4-bit quantization (transformer + T5 text encoder), GPU-resident, no offload | **Succeeds** | **16.00GB** | **31.0s (mean, n=90)** |

The bf16+offload OOM is a real, reproducible finding, not a fluke: CPU offload moves whole
modules between GPU and CPU one at a time, not individual layers, and the transformer module
alone (23.78GB in bf16) is larger than the L4's ~22GB usable VRAM regardless of what else is
offloaded — the pipeline's *total* footprint fitting inside 24GB is irrelevant if its single
largest component doesn't. NF4 quantization roughly halves-to-quarters the transformer and T5
encoder's resident footprint, which is what actually made this model fit at all.

**Consequence for local deployment: FLUX.1-schnell needs ≥16GB VRAM at minimum (NF4-quantized,
no headroom for anything else resident) — the 8.59GB RTX 3070 this entire project is built
around cannot run it under any configuration measured here.** This is not a "slower but
possible" finding; it is a hard capacity mismatch with the project's own stated design
constraint (an 8GB consumer-GPU budget).

---

## 3. A real, encountered access barrier: gated HF repo

FLUX.1-schnell is Apache-2.0 licensed but is a **gated** HF repository — downloading it requires
the account to manually click "Agree" on the model page (a browser-only action) plus an
authenticated token with gated-repo read access. This project's existing `HF_TOKEN` is a
fine-grained token scoped only to write access on the three published aetherart adapters and
cannot read third-party gated repos. Resolving this required a separate, broader `HF_READ_TOKEN`
and an explicit code change (`aetherart/flux_pipeline.py`) to pass it to every `from_pretrained`
call rather than relying on `huggingface_hub`'s implicit `HF_TOKEN`-name env lookup, which would
have silently tried to use the wrong, insufficiently-scoped token. This is reported as a real
operational-cost data point for FLUX specifically: adopting it means carrying an extra,
broader-scoped credential this project's other three (self-published, non-gated) model
dependencies don't need.

---

## 4. Three real script bugs found and fixed while orchestrating the GCP run

In the course of getting the run to complete, the GCP startup script itself hit the same species
of defect this project's own `docs/paper/measurement_defects.md` is about — a failure mode that
would have silently produced the wrong outcome had it not been caught:

1. The bf16+offload probe's real OOM crash should have triggered an already-written NF4 fallback,
   but the whole script runs under `set -e -o pipefail`, and the probe's Python process piped
   through `tee` with no `|| true` meant the entire script aborted at the crash, never reaching
   the fallback logic. Fixed by adding `|| true` to both probe pipelines.
2. That fix alone wasn't sufficient: the very next line, `grep -oP '...' "$PROBE_LOG" | tail -1`,
   *also* fails under `pipefail` when grep finds no match — exactly the case that occurs when the
   probe crashed and never printed a result, i.e. the one case this whole fallback exists to
   handle. Fixed with `|| true` on both grep-extraction lines.
3. Proactively fixed a third instance of the same bug class before it could bite: `wait
   "$EVAL_PID"` returning non-zero would have aborted the script before `EVAL_EXIT=$?` on the next
   line ever ran, skipping the partial-results push and graceful-failure branch entirely if the
   harness run itself ever failed. A bare `|| true` would have been wrong here too (it would
   overwrite `$?` with `true`'s own 0 before the next line could read it) — fixed with
   `EVAL_EXIT=0; wait "$EVAL_PID" || EVAL_EXIT=$?` instead, which captures the real exit code
   while still satisfying `set -e`.

All three are committed (`scripts/gcp_startup_flux_eval.sh`). Noted here because it is a small,
concrete instance of exactly the pattern this project's own methodology paper describes: an
automated safeguard (the NF4 fallback) that was written correctly in isolation but was silently
defeated by an unrelated shell-scripting interaction, and was only caught because the failure was
real and reproduced twice under observation, not assumed to be a one-off.

---

## 5. Results — n=90, 0 errors, independent-samples comparison

**Caveat stated up front:** this is an independent-samples comparison, not this project's
stronger paired-diff design. The five baselines' raw per-record JSON files (`reports/verdict_
{family}.json`) are not resident on this machine — they were generated in an earlier session and
are local-only artifacts, not committed to the repo — so only their published mean±SEM
(`docs/MODEL_VERDICT.md` §2, itself independently re-verified there, n=90 each) is available here.
The diff/SEM below is computed via independent-sample SEM quadrature
(`sqrt(SEM_flux² + SEM_base²)`), which is a weaker design than the matched-pair comparisons this
project prefers elsewhere. All 30 prompts and 3 seeds are identical across every family, so a
paired re-analysis is possible in the future if the baseline raw files are regenerated or
recovered.

**FLUX.1-schnell** (NF4-quantized, n=90, 0 errors): CLIP mean 0.3259 ± 0.0034 SEM; HPS mean
0.3020 ± 0.0035 SEM; mean latency 31.0s/image.

| Baseline | Axis | flux mean | base mean | diff | SEM_diff | diff/SEM | MDE@80% |
|---|---|---|---|---|---|---|---|
| `sd21_base` | CLIP | 0.3259 | 0.3167 | +0.0092 | 0.0050 | +1.845 | 0.0139 |
| `sd21_base` | HPS | 0.3020 | 0.2528 | +0.0492 | 0.0055 | **+9.005** | 0.0153 |
| `sdxl_base` | CLIP | 0.3259 | 0.3280 | −0.0021 | 0.0048 | −0.439 | 0.0135 |
| `sdxl_base` | HPS | 0.3020 | 0.2876 | +0.0144 | 0.0049 | **+2.959** | 0.0137 |
| `hyper_4step` | CLIP | 0.3259 | 0.3269 | −0.0010 | 0.0050 | −0.203 | 0.0141 |
| `hyper_4step` | HPS | 0.3020 | 0.3138 | −0.0118 | 0.0049 | **−2.408** | 0.0137 |
| `hyper_8step` | CLIP | 0.3259 | 0.3136 | +0.0123 | 0.0054 | +2.264 | 0.0152 |
| `hyper_8step` | HPS | 0.3020 | 0.2369 | +0.0651 | 0.0056 | **+11.584** | 0.0158 |
| `sdxl_controlnet_union` | CLIP | 0.3259 | 0.3281 | −0.0022 | 0.0046 | −0.480 | 0.0130 |
| `sdxl_controlnet_union` | HPS | 0.3020 | 0.2802 | +0.0218 | 0.0048 | **+4.539** | 0.0135 |

Full machine-readable output: `reports/flux_schnell_comparison.json`; raw records:
`reports/verdict_flux_schnell.json` (both committed).

**Reading this table against the project's own 2×SEM significance bar:**
- FLUX beats `sd21_base`, `hyper_8step`, and `sdxl_controlnet_union` decisively on HPS (+9.0,
  +11.6, +4.5×SEM); CLIP is tied or modestly ahead against the same three.
- **Against `hyper_4step` — the project's own currently-recommended default route — FLUX ties on
  CLIP (−0.2×SEM, not significant) and loses on HPS (−2.4×SEM, a significant regression), while
  taking ~4.6× longer per image (31.0s vs. 6.7s).**
- Against `sdxl_base`, FLUX ties on CLIP and shows a modest HPS lift (+3.0×SEM) at roughly 1.6×
  the latency (31.0s vs. 19.0s).
- Per this project's own established finding (`docs/HF_MODEL_CARD_UPDATES.md`,
  `docs/MODEL_VERDICT.md` §3.4), CLIP is valid for base-model comparisons like this one but not
  for style-adapter comparisons — not a caveat that applies to any number in this table, since
  none of these are style-adapter comparisons, but stated for completeness per this task's own
  instruction.

---

## 6. On the named failure modes (text-in-image, complex multi-subject composition)

**This was not directly tested and is stated as a scope gap, not answered by inference from the
aggregate numbers above.** CLIP and HPS are both aggregate, whole-image quality signals; neither
isolates text-rendering fidelity or multi-subject composition specifically, and this run did not
retain the 90 generated images locally (only the JSON scores were pulled from GCS before
teardown, to keep the exercise cheap and fast) — so no visual spot-check of FLUX's commonly-cited
strength (cleaner in-image text) was performed here. If FLUX is revisited specifically for that
narrow use case, the right next experiment is a small, targeted qualitative comparison (5–10
text-heavy or multi-subject prompts, FLUX vs. `hyper_4step` side by side) — not a re-run of this
aggregate benchmark, which already shows FLUX does not win in aggregate.

---

## 7. Recommendation

**Skip.** FLUX.1-schnell does not clear the bar of "add as a second pipeline" or "replace the
default" on measured evidence:
- It does not beat the project's own already-adopted default (`hyper_4step`) on either measured
  axis — it ties on CLIP and loses on HPS.
- It costs ~4.6× the latency of that default per image.
- It requires ≥16GB VRAM under the only configuration that worked at all (NF4-quantized) —
  infeasible on this project's actual 8.59GB deployment target under any measured configuration.
- It requires a broader, gated-repo-scoped HF credential this project's other three dependencies
  don't need.

The one place FLUX could still plausibly earn a narrow, opt-in role — text-in-image or
multi-subject composition, where SDXL-family models are known to struggle — was not tested here
and remains open, per §6. Revisiting that specific claim would need a small targeted qualitative
check, not another full aggregate run; nothing in this evaluation blocks that from being done
cheaply later if it becomes a real product need.

---

## 8. Reproducibility and provenance

| Claim | Source |
|---|---|
| Local disk requirement (~33.7GB) vs. free disk (~25.5–27.4GB) | `aetherart/flux_pipeline.py` module docstring; HF API file listing, 2026-07-30 |
| bf16+offload OOM (reproduced twice) | `gs://aetherart-497918-training/flux-eval/flux_eval_run.log` (pulled locally, not quoted verbatim here for length) |
| NF4-quantized: 16.00GB peak VRAM, 31.0s/image (single-image probe) | Same log, `PROBE_Q gen_s=30.5 peak_vram_gb=16.00` |
| Full run: n=90, 0 errors, mean latency 31.0s | `reports/verdict_flux_schnell.json` |
| CLIP/HPS means and comparison table | `reports/flux_schnell_comparison.json`, computed by `scripts/compute_flux_schnell_comparison.py` |
| 5 baselines' mean±SEM (n=90 each) | `docs/MODEL_VERDICT.md` §2 |
| FLUX.2-Klein license correction (`license:other`, not Apache-2.0) | `https://huggingface.co/api/models/black-forest-labs/FLUX.2-klein-9B`, checked 2026-07-30 |
| Three startup-script bugs found and fixed | `scripts/gcp_startup_flux_eval.sh` git history (commits `f012d16`, `7174875`, `fe4899d`) |

**GCP cost.** GCP project `aetherart-497918`, instance type `g2-standard-8` (1× NVIDIA L4),
zones `us-central1-a` (immediate gate-error failure, ~2 min), `us-west1-c` (two probe-bug
failures, ~10 min billed compute + ~85 min disk-only while stopped between attempts), `us-west1-a`
(successful run, ~62 min). Total billed GPU-compute time across every attempt: **≈1.2 hours**.
Estimated cost at public on-demand list pricing for `g2-standard-8`+1×L4 in these regions
(**~$1.00–1.10/hour, not fetched from a live billing export — stated as an estimate, not a
verified charge**) plus minor 200GB pd-ssd disk-only time: **≈$1.30–1.50 total**, against the
$25 hard cap for this task. All instances, disks, and static IPs confirmed zero remaining via
`gcloud compute instances/disks/addresses list --project aetherart-497918` after teardown.
`scripts/gcp_verify_before_teardown.py` passed (90 records, 0 errors) before the final instance
was deleted.
