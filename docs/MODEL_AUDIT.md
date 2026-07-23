# AetherArt Model Audit — Current-State (Read-Only)

**Scope:** Answers the seven audit questions against the actual codebase at
`C:\Users\gaura\ml-projects\AetherArt`, branch `docs/readme-results-led`. Every claim below
carries a `file:line` citation. Items not found in the checked files are marked **not found**
— nothing here is estimated or inferred beyond what the cited lines state.

**Correction to the framing of Q1:** the prompt asks to identify "the two HF models in use."
The repo does not have a two-model architecture — it has **five distinct model families**
wired through four separate pipeline modules (SD 2.1, SDXL base, SDXL Turbo, SD 2.1
ControlNet, SDXL ControlNet-Union), plus a Hyper-SD LoRA and a domain LoRA. This audit reports
what's actually there rather than forcing it into a two-model shape.

---

## 1. HF models in use

All base identifiers are declared in the `Config` dataclass — `aetherart/config.py:37-60`.

| Model | Repo ID | Source |
|---|---|---|
| SD 2.1 base (default) | `sd2-community/stable-diffusion-2-1` (overridable via `HF_MODEL_ID` env var) | `aetherart/config.py:38` |
| SDXL base | `stabilityai/stable-diffusion-xl-base-1.0` | `aetherart/config.py:39` |
| SDXL VAE (fp16 fix) | `madebyollin/sdxl-vae-fp16-fix` | `aetherart/config.py:40` |
| Hyper-SD LoRA repo | `ByteDance/Hyper-SD` (weights `Hyper-SDXL-4steps-lora.safetensors`, `Hyper-SDXL-8steps-lora.safetensors`) | `aetherart/config.py:41-46` |
| SDXL ControlNet-Union | `xinsir/controlnet-union-sdxl-1.0` | `aetherart/config.py:48` |
| SDXL ControlNet Canny (declared, unused — see below) | `xinsir/controlnet-canny-sdxl-1.0` | `aetherart/config.py:49` |
| SDXL ControlNet Depth (declared, unused — see below) | `xinsir/controlnet-depth-sdxl-1.0` | `aetherart/config.py:50` |
| SDXL depth estimator | `LiheYoung/depth-anything-small-hf` | `aetherart/config.py:51` |
| SDXL Turbo | `stabilityai/sdxl-turbo` (`TURBO_MODEL_ID`) | `aetherart/sdxl_turbo.py:33` |
| SD 2.1 ControlNet Canny | `thibaud/controlnet-sd21-canny-diffusers` | `aetherart/controlnet.py:43` |
| SD 2.1 ControlNet Depth | `thibaud/controlnet-sd21-depth-diffusers` | `aetherart/controlnet.py:44` |
| SD 2.1 depth estimator | `Intel/dpt-hybrid-midas` | `aetherart/controlnet.py:45` |
| Domain LoRA (Ukiyo-e, fine-tuned in-repo) | local file `data/lora/ukiyo-e/ukiyo-e-lora.safetensors` (SD 2.1) / `ukiyo-e-sdxl-lora.safetensors` (SDXL, see §7 report) | `aetherart/lora.py:14` |

**Discrepancy found:** `cfg.sdxl_controlnet_canny` and `cfg.sdxl_controlnet_depth`
(`aetherart/config.py:49-50`) are declared but never read anywhere in `aetherart/` —
`aetherart/controlnet_sdxl.py:48-50` only reads `cfg.sdxl_controlnet_union`. Separately,
`docs/depth_estimators.md:24` names the SDXL depth ControlNet checkpoint as
`diffusers/controlnet-depth-sdxl-1.0`, which conflicts with the `xinsir/...` string hardcoded
at `config.py:50`. This audit does not resolve which is authoritative — it is flagged as a
doc/code mismatch on a currently-dead config field.

**Revision/commit pinning:** **not found.** No `from_pretrained` call in `aetherart/` passes a
`revision=` kwarg (`revision` only appears in the vendored training scripts as an unused CLI
default of `None` — `scripts/_diffusers_train_text_to_image_lora_sdxl.py:202-207`,
`scripts/_diffusers_train_text_to_image_lora.py:166-171`). No commit SHA is recorded in
`models/sdxl-turbo/model_index.json` or `models/sdxl-turbo/README.md`.

**Local cache path:** **not found** as an app-level config. No `cache_dir` / `HF_HOME` /
`TRANSFORMERS_CACHE` / `HUGGINGFACE_HUB_CACHE` reference exists in `aetherart/` (`cache_dir`
appears only as an unrelated training-script CLI arg). The one explicit local-path convention
is SDXL Turbo's snapshot directory: `_LOCAL_DIR = Path(__file__).resolve().parent.parent /
"models" / "sdxl-turbo"` — `aetherart/sdxl_turbo.py:45`, used if
`unet/diffusion_pytorch_model.fp16.safetensors` exists there, else falls back to the HF repo ID
(`aetherart/sdxl_turbo.py:68-89`). `models/sdxl-turbo/model_index.json:2-3` confirms this local
snapshot is `StableDiffusionXLPipeline`, `diffusers` version `0.24.0.dev0` (older than the
pinned `0.35.1` runtime — see §6), scheduler `EulerAncestralDiscreteScheduler`
(`model_index.json:14-16`). SDXL Turbo access is gated behind
`AETHERART_ENABLE_LEGACY=1` (`aetherart/sdxl_turbo.py:48-65`) — see §7 (ADD license).

---

## 2. Integration

**Pipeline classes:**

| Path | Pipeline class | Citation |
|---|---|---|
| SD 2.1 base | `StableDiffusionPipeline` | `aetherart/model.py:19,107` |
| SDXL base | `StableDiffusionXLPipeline` | `aetherart/sdxl_pipeline.py:19,50` |
| SDXL Turbo | `AutoPipelineForText2Image` | `aetherart/sdxl_turbo.py:82,85` |
| SD 2.1 + ControlNet | `StableDiffusionControlNetPipeline` | `aetherart/controlnet.py:27,145` |
| SDXL + ControlNet-Union | `StableDiffusionXLControlNetUnionPipeline` | `aetherart/controlnet_sdxl.py:28-31,78` |

Note: the plain `StableDiffusionXLControlNetPipeline` class is not used anywhere — only the
Union variant.

**Schedulers:**
- `DPMSolverMultistepScheduler` — SDXL base default — `aetherart/sdxl_pipeline.py:62` (module
  docstring at lines 2-5 states this is the PR-03 default and that the Hyper-SD scheduler swap
  is handled elsewhere, not in this file).
- `DPMSolverMultistepScheduler` — quantized SD 2.1 — `aetherart/registry.py:231,234-236`.
- `DPMSolverMultistepScheduler` — quantized SDXL — `aetherart/quantization.py:39,158-159`.
- `EulerDiscreteScheduler` — swapped in when a Hyper-SD LoRA is loaded — `aetherart/hyper.py:13,35-36,54,58`.
- `LCMScheduler` — LCM fast-generation mode, reverted to `DPMSolverMultistepScheduler` after —
  `aetherart/lcm.py:20,22,27,29`.
- SDXL Turbo and SD 2.1 ControlNet pipelines have **no explicit scheduler override** — they
  keep whichever scheduler ships in the checkpoint (`EulerAncestralDiscreteScheduler` for the
  local Turbo snapshot per `models/sdxl-turbo/model_index.json:14-16`); no override found in
  `aetherart/sdxl_turbo.py` or `aetherart/controlnet.py`.

**LoRA loading method:** diffusers-native, not PEFT, at inference time:
`pipeline.load_lora_weights(str(lora_path.parent), weight_name=lora_path.name,
adapter_name="ukiyo_e")` then `pipeline.set_adapters(["ukiyo_e"], adapter_weights=[alpha])` —
`aetherart/lora.py:55-58` (unload via `pipeline.unload_lora_weights()` — `aetherart/lora.py:67`).
Same native pattern in `aetherart/controlnet.py:162` and `aetherart/controlnet_sdxl.py:93`.
PEFT's `LoraConfig` is used only in the **training** scripts
(`scripts/_diffusers_train_text_to_image_lora_sdxl.py:39-40,701-719`,
`scripts/_diffusers_train_text_to_image_lora.py:53-54`), not in the runtime inference path.

**Trained LoRA config (Ukiyo-e, SDXL):**
- UNet target modules: `["to_k", "to_q", "to_v", "to_out.0"]`, `r=args.rank`,
  `lora_alpha=args.rank`, `init_lora_weights="gaussian"` —
  `scripts/_diffusers_train_text_to_image_lora_sdxl.py:701-706`.
- Text-encoder target modules (only if `--train_text_encoder`):
  `["q_proj", "k_proj", "v_proj", "out_proj"]` — same file, lines 713-719.
- Default rank = 8 (`scripts/train_lora.py:58`); actual trained run used rank 8, 1024×1024,
  1500 steps, batch 1 × grad-accum 4, lr 1e-4, fp16, seed 42, trigger token `ukyowood` —
  `reports/lora_training_summary_sdxl.md:6-17`. Selected checkpoint 1000/1500, 45 MB,
  "rank-8, UNet-only LoRA" — `reports/lora_training_summary_sdxl.md:125-129`. Alpha is not
  stated as a separate value in the report; per the training script `lora_alpha=args.rank`, so
  alpha=8 by construction (not independently confirmed by a report line — flagging as
  derived-from-code, not measured).

**ControlNet conditioning + preprocessing:**
- SD 2.1: `canny` / `depth` (`Literal["canny", "depth"]` — `aetherart/controlnet.py:98,111`).
- SDXL Union: `canny, depth, hed, pidi, scribble, ted, lineart, normal` declared in
  `CONTROL_TYPES` (`aetherart/controlnet_sdxl.py:176`), plus `openpose`/`segment` mapped in
  `_CTYPE_TO_INT` (`aetherart/controlnet_sdxl.py:180-192`).
- Canny preprocessing: raw OpenCV (`cv2.Canny`) in both paths — `aetherart/controlnet.py:19,71-84`,
  `aetherart/controlnet_sdxl.py:119-138`. **`controlnet_aux` is not used anywhere** (`grep -rn
  "controlnet_aux" aetherart/ scripts/ docs/` — no matches).
- Depth preprocessing (SD 2.1): `transformers.pipeline("depth-estimation", model=DEPTH_ESTIMATOR_ID)`
  i.e. `Intel/dpt-hybrid-midas` — `aetherart/controlnet.py:36,45,62-68,87-93`.
- Depth preprocessing (SDXL): `transformers.AutoImageProcessor` /
  `AutoModelForDepthEstimation.from_pretrained(cfg.depth_estimator)` i.e.
  `LiheYoung/depth-anything-small-hf` — `aetherart/controlnet_sdxl.py:141-172`,
  `aetherart/config.py:51`. `docs/depth_estimators.md:9-24` documents this split as deliberate:
  the SDXL path was moved to a `.safetensors`-only estimator because of CVE-2025-32434
  (`torch.load` restriction on `.bin` weights).

---

## 3. Inference config

| Parameter | SD 2.1 default path | SDXL base | SDXL Turbo | Hyper-SD 4-step | Hyper-SD 8-step | SDXL ControlNet-Union |
|---|---|---|---|---|---|---|
| Width/height | 512×512 (`config.py:56-57`, env-overridable) | not set by loader itself (`sdxl_pipeline.py` only loads the pipeline object) | 512×512 hardcoded default (`sdxl_turbo.py:103-104`) | — | — | not set as a default (caller-supplied) |
| Steps | 30 (`config.py:58`, used at `model.py:192`) | — | 1 (`TURBO_STEPS`, `sdxl_turbo.py:34,116`) | 4 (`hyper.py:19`) | 8 (`hyper.py:25`) | 30 (function default, `controlnet_sdxl.py:204`) |
| Guidance scale | 7.5 (`config.py:59`, used at `model.py:193`) | — | 0.0 (`TURBO_GUIDANCE`, `sdxl_turbo.py:35,117`) | 0.0 (`hyper.py:20`) | 5.0 (`hyper.py:26`) | 7.5 (function default, `controlnet_sdxl.py:203`) |

**dtype:** `torch.float16` everywhere inference happens — `aetherart/sdxl_pipeline.py:43,53`,
`aetherart/sdxl_turbo.py:87`, `aetherart/controlnet_sdxl.py:50,75,82`,
`aetherart/quantization.py:71,78,84,124,139,145,153`. Device-conditional fp16/fp32 fallback in
`aetherart/controlnet.py:59` and `aetherart/model.py:103` (`torch.float16` if CUDA available,
else `torch.float32`). **No `bfloat16` usage found anywhere** in `aetherart/`.

**Device selection:** `cfg.device` (`aetherart/config.py:60`, `"cuda"` unless
`FORCE_CPU=1`) is declared but **not referenced anywhere else in the codebase** — grep for
`cfg.device` across `aetherart/`, `app.py`, `cloudrun_app.py` returns no hits; it is dead
config. Actual device selection is inline `torch.cuda.is_available()` checks — e.g.
`aetherart/model.py:103,111,206,208`, `aetherart/sdxl_turbo.py:90,110,136`,
`aetherart/quantization.py:87,97,103,180`. No `mps` (Apple Silicon) path exists anywhere.
`aetherart/gpu_hygiene.py:1-35` only implements `cleanup_gpu()` (cache/IPC cleanup), not device
selection.

**Quantization:** bitsandbytes (bnb) only — no GGUF, no TensorRT.
`aetherart/quantization.py:1` (module docstring), `BitsAndBytesConfig` imported at
`quantization.py:36-48`, configured with `load_in_4bit`/`load_in_8bit`,
`bnb_4bit_quant_type="nf4"`, `bnb_4bit_compute_dtype=torch.float16` —
`quantization.py:67-72,120-125`. Applied to the U-Net only; text encoder and VAE stay fp16
(`quantization.py:3-4`). **Not applied by default**: `aetherart/sdxl_pipeline.py`
(`load_sdxl_base`, the loader `aetherart/model.py:88-96` actually calls) never imports
`quantization.py`. Quantized loading is an explicit opt-in surfaced as a UI mode in
`app.py:42-43` (`"quantized_8bit"`, `"quantized_4bit"`) and dispatched via
`aetherart/registry.py:157,160,176,223,226`.

Measured (not default-path, but benchmarked as an experiment): the canonical harness at
`reports/experiments/exp1_quantization_quality/findings.md` (5 seeds × 8 prompts, DPM-Solver++,
30 steps, 512×512, CFG 7.5, `enable_model_cpu_offload()`, SD 2.1 U-Net) reports fp16
CLIP=0.3124/LPIPS=0.0/4.4 s/1803 MB, INT8 CLIP=0.3112/LPIPS=0.1729/12.3 s/2210 MB, NF4
CLIP=0.3158/LPIPS=0.3966/6.4 s/1382 MB — `findings.md:5-42` (per the researching agent; not
independently re-opened by me in this pass). An earlier, superseded benchmark exists at
`reports/quantization_benchmark.md:1-19` with different absolute numbers (fp16=3097 MB/2.7 s,
INT8=2210 MB/9.6 s, NF4=2761 MB/4.7 s) — the file itself flags this as historical/superseded at
line 3.

---

## 4. Eval / quality gates

Metrics exist and are measured, but **no CI/regression gate is wired to a numeric threshold.**

**Measured numbers (selected, with provenance):**

- CLIP score, SD 2.1, 360-run scheduler×step sweep: DPM-Solver++ 30 steps = **0.3199**, overall
  DPM = **0.3177** — `reports/eval_results_20260425_124153.md:9-12` (30 prompts, seed 42, model
  `sd-2.1` — config block at lines 88-107). Produced by `scripts/eval.py`.
- CLIP score, SDXL smoke run (n=3 prompts, DPM/20 steps): overall = **0.2323** —
  `reports/eval_results_20260531_200926.md:9`.
- CLIP-blindness (SD 2.1) reconciled headline: **4/9 experiments CLIP-blind** under a strict
  1-SE cutoff, correcting an earlier "9/9" qualitative claim —
  `reports/clip_blindness.md:1-8`. Per-experiment deltas in SE units at
  `reports/clip_blindness.md:27-35` (e.g. Exp8 LoRA-alpha = 4.00 SE).
- CLIP-blindness (SDXL replication, PR13): **3/7 CLIP-blind** —
  `reports/clip_blindness_sdxl.md:10`, per-condition CLIP/HPS/IR/LPIPS at lines 52-157.
- HPS / ImageReward: only measured for SDXL replication and one GCP validation run — no SD 2.1
  HPS/IR numbers exist. PR12 GCP validation (4 Ukiyo-e prompts × 4 scorers): avg HPS =
  **0.2391**, avg ImageReward = **1.4786**, avg CLIP = **0.3590** —
  `docs/lab_notebook.md:256-266`. Produced by `aetherart/eval_hps.py:55`
  (`score_hps`) and `aetherart/eval_ir.py:63` (`score_image_reward`) via `scripts/eval.py`.
- **FID: not found anywhere** — no FID computation code, no FID number in any report checked.
- **Human review: not found as a scored metric.** Only qualitative visual-inspection checkpoint
  selection exists (`reports/lora_training_summary_sdxl.md:93-121`,
  `reports/findings.md:74`) — not a number.

**Gate status:** `.github/workflows/ci.yml:26-57` — CI installs `openai-clip==1.0.1` and
`timm==1.0.27` but explicitly does **not** install `hpsv2` or `image-reward`; comment at lines
29-34 states scorer tests are fully mocked in CI. CI runs `pytest --cov=aetherart` with no
eval-threshold step, no separate eval job. Test assertions in `tests/test_eval.py:34-88`,
`tests/test_clip_scorer.py:27-74`, `tests/test_eval_hps.py:28-108`, and
`tests/test_eval_ir.py:22-156` check types, lengths, and call counts (`isinstance(..., float)`,
`math.isfinite(...)`, `len(scores) == N`) — **none asserts a numeric quality threshold.**
Conclusion: scoring code is tested for "does it run," not gated on "is it good enough." All
numeric results live in one-off `reports/*.md`/`*.json` files, never in an asserted CI gate.
`PLAN.md:27-45` lists "Phase 7 — SDXL Modernization (in progress)" with PR 14 "Coverage
threshold enforcement" still unchecked, consistent with no gate existing yet.

---

## 5. Latency / cost

**No p50/p95 latency and no cost-per-image figure exist anywhere in the repo.** Only mean/avg
latency per benchmark cell, and only infra/training cost (not per-image inference cost), are
reported. Checked: all `reports/*.md`/`*.json`, `reports/eval_run.log`,
`reports/quantization_benchmark.md`, `docs/lab_notebook.md`, `docs/torch28_compat.md`.

**Measured mean latencies found:**
- SD 2.1, RTX 3070, 360-run sweep: DPM-Solver++ 20/30/50 steps = 8.24 s / 10.76 s / 15.58 s —
  `reports/eval_results_20260425_124153.md:16-21`.
- SDXL, DPM/20 steps, n=3 (hardware unlabeled in the file): overall mean = **608.59 s** —
  `reports/eval_results_20260531_200926.md:13-15`; per-prompt values 276.97 s / 1551.10 s /
  736.51 s show large run-to-run variance — `reports/eval_partial_latest.json:35,53,71`.
- SD 2.1 U-Net quantization (isolated benchmark, superseded per its own header note): fp16 =
  2.7 s, INT8 = 9.6 s, NF4 = 4.7 s — `reports/quantization_benchmark.md:9-13`.
- ControlNet + CPU offload, unquantized fp16 SDXL, local hardware: ~275–292 s/image, peak VRAM
  7928 MB — `docs/lab_notebook.md:199-201`.
- One-off torch-2.8 compat smoke test, RTX 3070, 10 steps: 16.7 s, peak VRAM 6840 MB —
  `docs/torch28_compat.md:79-81`.

**Cost figures found are infra/training cost, not per-image:**
- SDXL LoRA training run: ~$3.50 total, 4h26m wall-clock, GCP `g2-standard-4` L4 at ~$0.70/hr —
  `reports/lora_training_summary_sdxl.md:37-48`.
- GCP eval VM session: ~$0.29 for 25 minutes — `docs/lab_notebook.md:274`.

**Latency instrumentation in real (non-benchmark) inference:** `aetherart/logger.py:1-13` and
`aetherart/metadata.py:1-51` contain no timing code. Per-generation latency IS captured in the
interactive app, but the instrumentation lives in `app.py`, not in the shared logging/metadata
modules: `time.time()`/`gen_time` computed at `app.py:109,212,227,247`, written into the PNG
sidecar metadata as `"generation_time_seconds": round(gen_time, 2)` at `app.py:402`, persisted
via `aetherart/metadata.py`'s `save_image_with_metadata()` (called at `app.py:410`). Separately,
`scripts/eval.py:185,196` captures its own per-run latency — the source of all §5 sweep numbers
above.

---

## 6. Dependency versions

| Package | `requirements.txt` | `requirements-lock.txt` |
|---|---|---|
| diffusers | `0.35.1` — `requirements.txt:7` | `0.35.1` — `requirements-lock.txt:36` |
| transformers | `4.56.2` — `requirements.txt:8` | `4.56.2` — `requirements-lock.txt:125` |
| torch | `2.5.1` — `requirements.txt:18` | `2.5.1+cu124` — `requirements-lock.txt:121` |
| peft | `0.19.1` — `requirements.txt:12` | `0.19.1` — `requirements-lock.txt:81` |
| accelerate | `1.10.1` — `requirements.txt:9` | `1.10.1` — `requirements-lock.txt:18` |

The only difference is the CUDA build suffix on `torch` (documented deliberately —
`requirements-lock.txt:6-8` explains CPU vs CUDA124 variants); underlying versions agree.
`pyproject.toml:1-76` has no `[project]`/`[build-system]` dependency table — these tool
versions are **not declared there.** `setup.py:6-18,27` parses `install_requires` dynamically
from `requirements.txt` at build time, deliberately excluding `torch` (comment at lines 15-16)
since it needs a platform-specific `--index-url`.

Note from §1: the local SDXL Turbo snapshot's `model_index.json` records
`_diffusers_version: 0.24.0.dev0` — that's the diffusers version the snapshot was *exported*
with, not the version the app runs (`0.35.1`); diffusers loads older snapshots forward-compatibly,
this is not a runtime version conflict, just a provenance note on the cached file.

---

## 7. Documented failure modes / known limitations

No project-level `CLAUDE.md` exists in the repo root or `docs/` (confirmed by directory
listing). Limitations are documented across `reports/what_didnt_work.md`, `reports/findings.md`,
`reports/clip_blindness.md`, `reports/clip_blindness_sdxl.md`, `docs/lab_notebook.md`,
`docs/torch28_compat.md`, `CHANGELOG.md`, and `PLAN.md`. Selected, most relevant to the current
model/inference state:

- **INT8 uses more VRAM than fp16 under CPU offload** — bitsandbytes needs a full fp16
  dequant buffer, so on 8 GB with CPU offload "it does not [recover savings]" —
  `reports/what_didnt_work.md:17-19`, restated `CHANGELOG.md:24`, `docs/lab_notebook.md:70`
  ("has no compelling use case... 2.8× slower, same CLIP score as fp16").
- **`_quant_pipes` dict grew unboundedly**, causing VRAM-pressure crashes that "looked like
  random crashes" — `reports/what_didnt_work.md:9-11`.
- **SDXL Turbo (ADD license) is non-commercial research only** — gated behind
  `AETHERART_ENABLE_LEGACY=1` to avoid shipping it in a commercially-framed demo —
  `reports/what_didnt_work.md:79-87`, gate implementation `aetherart/sdxl_turbo.py:48-65`.
- **LoRA calligraphy artifact**: negative prompt "suppresses most artifacts but doesn't
  eliminate them completely — the style signal and text signal are entangled in the adapter
  weights"; a proper fix (~5h curation) "was not done" — `docs/lab_notebook.md:23-25`.
- **CLIP-blindness findings are domain/model-specific**: SD 2.1, 865M U-Net, one CLIP variant
  (`clip-vit-base-patch32`), single style domain (Ukiyo-e); "results may differ for SDXL, SD 3,
  or distilled models"; "no human ratings"; SE estimates are ~0.004–0.007/cell, "a null result
  at n=8 or n=30 is not the same as a well-powered null" — `reports/clip_blindness.md:84-90`.
- **SDXL replication skipped 2 of 7 experiments** (LoRA rank, LoRA data size) because "the GCP
  eval VM was not staged with the training dataset" — an operational gap, not a methodological
  one — `reports/clip_blindness_sdxl.md:12`.
- **ImageReward crashes on Windows**: `ReFL → datasets → pandas → pyarrow.dataset` C extension
  triggers a SIGSEGV access violation on Windows with pyarrow 24.x, "before any Python-level
  try/except can catch it" — forcing all real eval work onto GCP Linux; "Windows is interactive
  dev only from PR 12 onward" — `docs/lab_notebook.md:246-250`.
- **hpsv2 1.2.0 crashes on headless Linux** (no tkinter) due to a stray `from turtle import
  forward` in `src/open_clip/factory.py:8` — `docs/lab_notebook.md:269`.
- **transformers/diffusers version coupling**: `transformers>=4.51` removed
  `FLAX_WEIGHTS_NAME`, breaking `diffusers==0.35.1`; pinned to `>=4.41.2,<4.51` as a workaround
  on the GCP DLVM image, flagged as an unresolved carry-over — `docs/lab_notebook.md:219,240`.
- **CLIP score is explicitly caveated as a proxy**: "does not measure sharpness/coherence/
  aesthetic quality"; fixed seed=42 "eliminates diversity measurement"; results are SD-2.1- and
  512×512-specific — `reports/findings.md:34,116-123`.
- **bitsandbytes flagged as highest-risk dependency** for a pending torch 2.8 upgrade; "a full
  INT8/NF4 inference run was not performed" under torch 2.8 — only import/config construction
  validated — `docs/torch28_compat.md:52`.

---

## Summary of unknowns (explicitly not found, not estimated)

- Pinned HF revision/commit SHA for any model — not found.
- App-level `cache_dir`/`HF_HOME` configuration — not found.
- FID score — not found, no code computes it.
- Human-rated quality score (numeric) — not found.
- p50/p95 latency (any model) — not found, only means exist.
- Cost-per-generated-image — not found, only training/VM-session cost exists.
- A CI-enforced quality regression gate — not found; PLAN.md marks this as still pending
  (PR 14, unchecked as of `PLAN.md:27-45`).

---

*Compiled from a read-only audit pass (three parallel research agents scoped to §1–2, §3+§6,
and §4+§5+§7 respectively) plus direct spot-checks of `aetherart/config.py`,
`aetherart/sdxl_pipeline.py`, and `requirements.txt`. A verifier pass re-checked citations
against the live files before this document was finalized — see verification notes below if
any were amended.*
