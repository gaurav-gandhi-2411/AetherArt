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
