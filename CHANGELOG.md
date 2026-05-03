# Changelog

Follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) conventions, grouped by project phase rather than version (no versions have been released).

---

## Phase 4 — Story and portfolio polish (May 2026)

- **Benchmark findings writeup** — `reports/findings.md`: full narrative with TL;DR, methodology, per-cell standard errors, Pareto scatter chart, and prompt-vs-scheduler variance decomposition
- **README top section** — "What this demonstrates" engineering bullets, each linking to the relevant source file or report
- **"Recreate from PNG" documented** — explicit subsection explaining the PNG tEXt + sidecar JSON metadata system
- **CONTRIBUTING.md** — exact setup, test, lint, and CI commands; GPU vs CPU path notes
- **CHANGELOG.md** — this file; backfilled from git history

---

## Phase 3 — Production-readiness pass (May 2026)

- **GPU hygiene** — `aetherart/gpu_hygiene.py`: idempotent `cleanup_gpu()` with atexit registration and try/finally in 3 gallery scripts; eliminates zombie CUDA processes
- **Type annotations** — PEP 561 `py.typed` marker; return types and `Optional`/`Any` annotations on all 8 public modules; 0 mypy errors in soft mode
- **ModelRegistry** — `aetherart/registry.py` owns all pipeline singletons; fixes two latent bugs: (1) `_quant_pipes` dict grew unboundedly on repeated 4/8-bit switches; (2) silent `MODEL.backend is None` failure now surfaces as `RuntimeError` with retry support
- **Test coverage** — 100 tests total (up from 46); coverage 59% (up from 36% baseline); new tests cover registry lifecycle, LoRA lookups, CLIP scorer, and metadata edge cases

---

## Phase 2 — CI hygiene (late April / early May 2026)

- **CI pipeline** — GitHub Actions: flake8 + black + isort + mypy + pytest-cov on every push; all checks enforced (no `|| true`)
- **Code style** — black (line-length=100), isort (profile=black), flake8 (.flake8 config with per-file-ignores for vendored scripts)
- **Pre-commit hooks** — black + isort + flake8 run locally on every commit
- **Requirements lock** — `requirements-lock.txt` with exact versions; opencv conflict documented

---

## Phase 1 — Bug fixes and stabilization (late April 2026)

- **HF Space CPU pivot** — Standard mode only on free tier; LCM/Turbo disabled; generation time disclosed honestly (~8–15 min)
- **Resolution guard** — CPU inference defaulted to invalid 768×768; corrected to 512×512 with explicit guard
- **bitsandbytes CI guard** — test correctly checks `check_cuda_matmul_alignment` capability, not just importability
- **ZeroGPU / Gradio concurrency** — fixed LCM scheduler applied to wrong pipeline; correct concurrency model for the live Space

---

## Speed tiers and quantization (late April 2026)

- **LCM fast mode** — 4-step generation via `LCMScheduler` (scheduler-only, no LCM-LoRA for SD 2.1); 5.8× speedup on RTX 3070 (0.6 s vs 3.2 s)
- **SDXL Turbo** — 1-step adversarial diffusion; separate 2.6B-param SDXL model; 3.3 s on RTX 3070 (real speedup only visible on A100/H100)
- **INT8 quantization** — bitsandbytes 8-bit U-Net; 2.2 GB peak VRAM (vs 3.1 GB fp16); 9.6 s/img due to dequantization overhead
- **NF4 quantization** — bitsandbytes 4-bit U-Net; 2.8 GB peak (compute buffer inflates vs stored weights); enables SD 2.1 on ≥4 GB GPUs

---

## LoRA and ControlNet (late April 2026)

- **Ukiyo-e LoRA** — rank-8 adapter, 80 WikiArt images, 1500 steps on RTX 3070 (2 h 8 min, 0 OOM events); checkpoint-1000 selected for warm amber palette matching Hokusai; 6.4 MB safetensors
- **Calligraphy artifact mitigation** — WikiArt training data embeds caption text; default negative prompt `text, watermark, calligraphy, signature, words, letters` auto-applied when LoRA is active
- **ControlNet Canny** — `thibaud/controlnet-sd21-canny-diffusers`; OpenCV Canny edge preprocessing; LoRA combinable via direct load into ControlNet pipeline
- **ControlNet Depth** — `thibaud/controlnet-sd21-depth-diffusers`; DPT-Hybrid-MiDaS depth estimation; LRU pipeline cache (max 2 entries, keyed by ctype × lora × alpha)

---

## Evaluation (mid-April 2026)

- **360-run CLIP benchmark** — `scripts/eval.py`: 4 schedulers × 3 step counts × 30 PartiPrompts subset; fixed seed 42; `openai/clip-vit-base-patch32`; full results in `reports/eval_results_20260425_124153.json`
- **Quantization benchmark** — fp16 vs 8-bit INT8 vs 4-bit NF4: latency, VRAM, CLIP; `reports/quantization_benchmark.md`
- **Benchmark charts** — Pareto frontier, latency by scheduler, CLIP by scheduler, VRAM by scheduler

---

## Initial build (early April 2026)

- **SD 2.1 base** — `sd2-community/stable-diffusion-2-1` (switched from `stabilityai/stable-diffusion-2-1` after EU AI Act deprecation); fp16 + model CPU offload; DPM-Solver++ scheduler
- **Gradio UI** — prompt, negative prompt, seed, steps, guidance, resolution, speed mode, memory mode, LoRA style, ControlNet accordion
- **PNG metadata** — `aetherart/metadata.py`: every generation saves PNG tEXt chunks + sidecar `.json` with full generation parameters and git commit hash; "Recreate from PNG" tab restores settings from any prior output
- **HF Space deployment** — `spaces/README.md` with YAML frontmatter; `scripts/deploy_to_hf.py` preserves Space config on every push
