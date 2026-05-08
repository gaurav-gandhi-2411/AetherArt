# What Didn't Work

An honest record of bugs, abandoned approaches, surprises, and one methodological mistake in this project. Project retrospectives are more useful when they include the parts that failed.

---

## Bugs that took longer than they should have

### `_quant_pipes` dict grew unboundedly (Phase 3)

`aetherart/registry.py` kept a dict of quantized pipelines keyed by precision mode. Every time the Gradio UI switched between fp16, INT8, and NF4, the old pipeline stayed in the dict and a new one was created. On an 8 GB GPU with model CPU offload, three pipeline objects coexisting quietly consumed enough memory to cause sporadic OOM events that looked like random crashes. The dict should have been a single-slot cache (last-used precision only). The bug was silent: no error, no warning, just accumulating VRAM pressure. Fixed in Phase 3 by rewriting the registry with an eviction policy.

### Silent `|| true` on flake8 CI (Phase 2)

The initial GitHub Actions workflow had `flake8 . || true` — the `|| true` made flake8 always exit 0, so CI passed even when there were lint errors. This went undetected for several commits because the CI log showed green and no one read the flake8 output lines. The correct fix is to remove `|| true` entirely and fix the actual errors. Discovered when doing a CI audit in Phase 2.

### INT8 U-Net VRAM higher than fp16 (quantization measurement, not a code bug)

This was not a bug, but it was a wrong assumption that cost real investigation time: bitsandbytes 8-bit quantization should save VRAM. And it does — if you load the full model onto the GPU. Under `enable_model_cpu_offload()`, bitsandbytes must allocate a full fp16 *compute buffer* for dequantization during each forward pass. On the RTX 3070 with CPU offload, this buffer inflated INT8 peak VRAM to 2210 MB vs fp16's 1803 MB — INT8 used *more* VRAM. The quantization benchmark report documents the measured values; the CLIP score was within 1 SE of fp16. The conclusion is: on ≥12 GB cards with enough VRAM to go offload-free, INT8 recovers its savings. On 8 GB with CPU offload, it does not.

### Resolution guard on CPU (Phase 1)

The CPU inference path defaulted to 768×768, which is invalid for SD 2.1 on CPU (attention computation is O(n²) in sequence length; 768² is ~2.3× more compute than 512²). The model would either OOM or run for 20+ minutes. Added an explicit guard that forces 512×512 when no GPU is available. Caught during HF Space testing.

### LCM scheduler applied to wrong pipeline (Phase 1)

When enabling LCM fast mode, the `LCMScheduler` was being applied to the base SD 2.1 pipeline object rather than the LCM-specific pipeline wrapper. The result was inconsistent fast-mode behavior: sometimes it ran the correct 4-step schedule, sometimes it ran 4 steps of the standard DPM schedule. Fixed by ensuring the scheduler swap targets the correct pipeline reference. This class of bug — applying a modification to a stale reference — is the main reason `ModelRegistry` was introduced in Phase 3.

---

## Abandoned approaches

### CPU Space pivot

The original plan for the HF Space deployment was GPU generation via ZeroGPU (Hugging Face's shared GPU queue). ZeroGPU turned out to be unavailable on the free tier during the window I needed to deploy. The pivot to CPU-only generation meant disabling LCM and SDXL Turbo (both require GPU for meaningful speedup), adding a realistic "8–15 min" generation time disclaimer, and limiting to 512×512. The Space still demonstrates the full architecture and UI — it just runs slowly.

The CPU pivot was the right call given the constraint. The wrong assumption was expecting ZeroGPU to be available on the free tier — that should have been verified before building the Space infrastructure.

### LCM-LoRA for SD 2.1

LCM fast mode was implemented via the `LCMScheduler` (scheduler-only approach), which works on SD 2.1. The original plan also considered LCM-LoRA, which provides a separate small adapter for consistency distillation. LCM-LoRA has a dedicated adapter for SDXL but not for SD 2.1 — the consistency distillation was done at the architecture level for SDXL, while SD 2.1's LCM path is scheduler-only. Discovered this after spending time looking for an SD 2.1-compatible LCM-LoRA adapter that does not exist. The scheduler-only approach gives the same 4-step acceleration; the LoRA path was a dead end for this model.

---

## Surprises

### CLIP-blindness was not the expected finding

The Phase 6b experiment series was designed to measure how different generation parameters affect output quality. The implicit expectation was that CLIP would track quality changes — that's what you use a metric for. The actual finding — CLIP flat across nine experiments while LPIPS ranged 0.40–0.73 — emerged only after Experiments 1, 2, and 3 all came back with the same null CLIP result and large LPIPS. By Experiment 4, the pattern was clear enough to name. By Experiment 9, it was the central conclusion.

This is the kind of finding you don't plan for. The experimental setup was designed to measure parameter sensitivity, not metric limitations. The metric turned out to be the most interesting result.

### The underfitting paradox (Exp 6 and Exp 7)

Both the LoRA rank ablation and the data size ablation showed the same counterintuitive direction: rank-4 scored higher on CLIP than rank-8 (0.3384 vs 0.3337), and data-20 scored higher than data-80 (0.3383 vs 0.3337). The underfit model wins on the metric designed to measure quality. This confirmed that CLIP rewards semantic keyword matching — "ukiyo-e," "woodblock print style" appear in the prompt and the underfit model produces images that match those tokens more literally. A richer, subtler style representation that learns Hokusai's specific compositional conventions has no CLIP-vocabulary footprint.

### Small dataset training time with validation_epochs=1 (Exp 7)

Training the 20-image LoRA subset took 4 hours 10 minutes against an expected ~75 minutes. The cause was a `validation_epochs=1` setting that fires validation once per epoch. With 20 images and gradient accumulation of 4, one epoch is 20/4 = 5 optimizer steps. Over 1500 training steps, that triggers 300 validation events. Each validation generates ~4 images and takes 2–3 minutes — totaling approximately 10 hours of validation overhead, though the actual wall time was 4h10m because of interleaving. The 80-image model with the same setting had 1 epoch = 20 optimizer steps, so validation fired 75 times — much less overhead. Design mistake: validation_epochs=1 is appropriate when the epoch is a meaningful unit (many steps); it is not appropriate for small datasets where one epoch is just a few steps.

---

## Methodological mistake (Phase 6b)

In the middle of Phase 6b, two experiments were run in the wrong order with substituted designs:
- A LoRA alpha sweep was run and labeled "Experiment 6"
- A trigger token ablation was run and labeled "Experiment 7"

The planned Experiment 6 was a LoRA rank ablation (rank 4/8/16). The planned Experiment 7 was a data size ablation (20/40/80 images). The substitution happened silently — the scripts were written for different experiments and the labeling was applied without flagging the deviation.

Recovery: the alpha sweep and trigger ablation were correctly renumbered as Experiments 8 and 9. The rank ablation and data size ablation were then run as Experiments 6 and 7. All documentation was updated. The final results are correct and complete.

The error's cause: the rank and data ablations required additional infrastructure (training rank-4 and rank-16 adapters; building subset training directories) that made them harder to start. The alpha and trigger experiments were quicker to code. That complexity difference should have been surfaced as a blocker rather than worked around by substituting easier experiments.

---

## Unanswered questions

**Multi-subject prompts and SD 2.1's structural limits.** The bottom-scoring prompts in the benchmark — "artificial intelligence," "a dragon," "a coffee mug" — score low not because the scheduler or step count is wrong, but because SD 2.1 is not well-calibrated for abstract or contextually complex subjects. These are training distribution failures, not generation parameter failures. Fixing them requires either a different base model (SDXL, SD 3) or targeted fine-tuning data — neither of which fits the hardware constraints.

**LoRA on cleaner training data.** The calligraphy artifact — text and signatures baked into the ukiyo-e style signal from WikiArt captions — is the main quality limitation of the adapter. The mitigation (negative prompt `text, watermark, calligraphy, ...`) reduces it but does not eliminate it. A properly curated dataset without embedded text would train a cleaner adapter, but curation for 80+ high-quality Ukiyo-e images with manually reviewed captions is a multi-hour task outside the scope of this project.

**Whether LPIPS findings generalize to other style domains.** All nine experiments used prompts in the ukiyo-e domain and the same SD 2.1 model. The CLIP-blindness pattern is plausible for other style adapters and other models, but it was not tested. The underfitting paradox in particular might behave differently in domains where CLIP's vocabulary is less specialized.
