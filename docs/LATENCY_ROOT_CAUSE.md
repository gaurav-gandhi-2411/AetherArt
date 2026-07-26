# SDXL Latency Root-Cause Investigation

**Bottom line up front:** the 5.6x variance in `reports/eval_partial_latest.json` (276.97s /
1551.10s / 736.51s at identical DPM/20-step config) does **not reproduce** under the current
default SDXL pipeline on the same physical hardware. 14 fresh, instrumented generations across
four configurations — including two attempts to deliberately trigger known failure mechanisms —
all completed in 6.4–18.3s. The historical data itself contains an internally-impossible
measurement (peak VRAM exceeding the GPU's physical capacity, identical across all three runs)
that is best explained by a stale/non-reset counter in an older script version, not by a bug in
the current code. This is reported as an evidenced non-reproduction, not a dismissal — see
§5 for exactly what remains unresolved and what to do about it.

All raw instrumented output is under `reports/_latency_diag*.{json,log}` and
`reports/_latency_diag_nvidia_smi.csv` (gitignored scratch artifacts, same convention as
`reports/eval_partial_latest.json` — see `.gitignore:16,46-47`). Every timing number below is
read from those files, not estimated.

---

## 0. A pre-existing local environment bug had to be fixed before any instrumentation was possible

Before any generation could run, `import torch` failed in **both** local conda environments
(`aetherart` and `aetherart-torch28`) with `OSError: [WinError 126] ... torch\lib\shm.dll or one
of its dependencies`. PE-import-table analysis (via `pefile`, not guesswork) showed
`torch_cpu.dll` depends on `fbgemm.dll`, and `fbgemm.dll` depends on `asmjit.dll` — both present
in `torch/bin/` but **not** `torch/lib/`, where `torch_cpu.dll`'s DLL search path looks. This is
a known PyTorch-on-Windows packaging quirk (the wheel splits `bin/` and `lib/` but the loader
only adds `lib/` to the search path). Fixed by copying both DLLs from `torch/bin/` to
`torch/lib/` in the `aetherart` env — a local-machine-only, reversible change; nothing in the
repo was touched. Confirmed working: `torch 2.5.1+cu124`, `cuda_available=True`, device
`NVIDIA GeForce RTX 3070 Laptop GPU`, `8.589 GB` total memory (matches `nvidia-smi`'s
`8192 MiB`).

This is unrelated to the historical latency variance (that data was captured successfully weeks
earlier, so torch clearly worked then) but it is a real, current finding: **any local dev today
would hit this immediately**, with no eval/generation possible until fixed. See §5 for a
recommended guard.

---

## 1. Is CPU offload active on the default SDXL path, and does VRAM vary run-to-run?

**Yes, `enable_model_cpu_offload()` is unconditionally active** on every SDXL generation,
regardless of available VRAM: `aetherart/sdxl_pipeline.py:65` (`pipe.enable_model_cpu_offload()`),
called from `load_sdxl_base()`, which is the function `aetherart/model.py:90-92` calls whenever
`model_choice == cfg.sdxl_model`. There is no VRAM check gating this — it applies the same on an
8GB laptop card as it would on a 24GB card. Attention slicing (`enable_attention_slicing()`,
used on the SD 2.1 path at `aetherart/model.py:144-151`) is **not** called anywhere in
`aetherart/sdxl_pipeline.py` — the SDXL path relies on CPU offload alone for VRAM management.
`xformers` memory-efficient attention is attempted (`aetherart/sdxl_pipeline.py:68-74`) but is
**not installed** in the local `aetherart` env (`pip show xformers` → "Package(s) not found"),
so that `try/except ImportError` silently no-ops on this machine — confirmed by direct check,
not inferred from the code.

**Does VRAM vary run-to-run?** Measured directly, across 14 real generations (§2), peak
`torch.cuda.max_memory_allocated()` was essentially constant: **5.361 GB** for every one of the
10 baseline (512×512) runs, **5.605 GB** for every one of the 3 runs at 1024×1024, and
**5.385 GB** for the unfused-LoRA run — see `reports/_latency_diag.json` (all 10 `vram.peak_allocated_gb`
fields), `reports/_latency_diag_1024.json`, `reports/_latency_diag_lora_unfused.json`. Concurrent
1-second `nvidia-smi` polling (`reports/_latency_diag_nvidia_smi.csv`, 373 samples spanning
11:04:30–11:11:44, covering all four sweeps in this session) shows physical `memory.used`
peaking at **5971 MiB** (rows `11:08:52.705`–`11:09:30.994`, during the 1024×1024 sweep — GPU
utilization pinned at 100% and temperature climbing 63°C→77°C in that window), never exceeding
that across the entire session — comfortably under the 8192 MiB physical ceiling (≈2.2 GB
headroom even at the peak). The 512×512 baseline sweep alone peaked lower, at 5605–5607 MiB.
Temperature climbed 46°C→77°C over the full session — normal warm-up behavior, not thermal
throttling (within the 512×512 baseline sweep specifically, later runs got *faster*, not slower
— see §2).

**Contrast with the historical data:** `reports/eval_partial_latest.json:36,54,72` records
`vram_peak_gb: 11.186` for **all three** historical runs, identically to three decimal places.
This is physically impossible on this card (11.186 GB > 8.589 GB total memory, confirmed via
`torch.cuda.get_device_properties(0).total_memory` in this session). Two independent pieces of
evidence explain this without invoking a real 11GB VRAM spike:
1. `reports/eval_partial_latest.json`'s `config` block (lines 4-21) has only 7 keys
   (`schedulers, steps, prompts_count, seed, model, total_combos, scorers`) — it is **missing**
   `width`, `height`, and `negative_prompt`, which the *current* `scripts/eval.py:581-592`
   always includes in its `config` dict. This proves the historical run used an **older version**
   of `scripts/eval.py` than the one in this repo today.
2. Current `scripts/eval.py:182` calls `torch.cuda.reset_peak_memory_stats()` immediately before
   every single generation, so each run's peak is independent. If the version that produced
   `eval_partial_latest.json` lacked that reset call, `torch.cuda.max_memory_allocated()` returns
   a **monotonically non-decreasing, process-lifetime** peak — which would explain why all three
   runs report the exact same value: whichever run first drove memory to ~11GB, that number
   would then be echoed by every subsequent run in the same process for the rest of the sweep.

This does not prove nothing unusual happened — something *did* push a counter to a number that
exceeds this card's capacity at some point in that historical process's lifetime — but it does
mean the figure **cannot be read as "every run used 11.186 GB,"** which is how it reads at face
value, and the true per-run VRAM footprint at the time is unrecoverable from this artifact.

---

## 2. Per-stage instrumented timing for 10 SDXL generations

`scripts/_latency_diag.py` (new, one-off diagnostic script — follows the repo's existing
`scripts/_ir_diag*.py` / `scripts/_pr12_smoke.py` convention for scoped investigations; does not
modify `scripts/eval.py` or `app.py`) drives the exact production path
(`AetherModel.init(model_choice="sdxl")` → `aetherart/model.py:88-96` →
`aetherart/sdxl_pipeline.py:load_sdxl_base()`, unmodified) and adds `callback_on_step_end` timing
to split each generation into: model load (once), then per-run **encode+step0** (time from
pipeline call to the first step callback — text encoding is not separately instrumentable
without patching `encode_prompt()`, so it's bundled with step 0), **per-step denoise** deltas,
and **decode+postprocess** (time from the last step callback to the call returning, i.e. VAE
decode + PIL conversion).

**10 runs, same config as the historical outlier (DPM scheduler, 20 steps, 512×512, guidance
7.5, cycling the same 3 prompts `pp_001`/`pp_002`/`pp_003`)** — full data in
`reports/_latency_diag.json`, human-readable log in `reports/_latency_diag.log`:

| Run | Prompt | Total (s) | Encode+step0 (s) | Mean step (s) | Max step (s) | Decode (s) | Peak alloc (GB) |
|---|---|---|---|---|---|---|---|
| 1 | pp_001 | 18.31 | 9.393 | 0.297 | 0.331 | 3.265 | 5.361 |
| 2 | pp_002 | 10.96 | 2.669 | 0.291 | 0.312 | 2.767 | 5.361 |
| 3 | pp_003 | 10.66 | 2.410 | 0.302 | 0.338 | 2.514 | 5.361 |
| 4 | pp_001 | 10.80 | 2.541 | 0.294 | 0.329 | 2.676 | 5.361 |
| 5 | pp_002 | 10.74 | 2.552 | 0.295 | 0.329 | 2.579 | 5.361 |
| 6 | pp_003 | 11.33 | 2.816 | 0.300 | 0.352 | 2.819 | 5.361 |
| 7 | pp_001 | 7.41 | 2.272 | 0.169 | 0.223 | 1.919 | 5.361 |
| 8 | pp_002 | 6.44 | 1.813 | 0.146 | 0.164 | 1.858 | 5.361 |
| 9 | pp_003 | 6.53 | 1.783 | 0.153 | 0.189 | 1.831 | 5.361 |
| 10 | pp_001 | 6.50 | 1.774 | 0.152 | 0.193 | 1.838 | 5.361 |

(`reports/_latency_diag.log:7,9,11,13,15,17,19,21,23,25` — one line per run.)

n=10, min=6.439s, max=18.31s, mean=9.97s (computed directly from `reports/_latency_diag.json`).
Run 1 is the slowest — consistent with one-time CUDA/cuDNN kernel-selection warm-up on the first
real forward pass of the process, not a recurring pathology: runs 7–10 are the *fastest* of the
set (mean-step time roughly halves, from ~0.30s to ~0.15s), the opposite of what thermal
throttling or memory fragmentation would produce. **No run came remotely close to the historical
1551.10s outlier, or even its 276.97s "fast" run** — the current default path is 15–40x faster
than every historical data point.

**Which stage would the historical 1551s outlier have hit, if it recurred?** Per-step cost is
the dominant, most-repeated component (20 steps × ~0.15–0.3s = 3–6s of the ~6.4–18.3s total), so
a proportional blow-up would show up there first. But the historical run's 1551s total against a
20-step config implies (if evenly distributed) ~77s/step — over 250x this session's per-step
cost — which is a different order of magnitude than anything observed here, including the
deliberately-provoked unfused-LoRA test (§4). This session's data cannot isolate which stage the
historical outlier actually hit, because it could not be reproduced at all.

**Supplementary test at SDXL's native 1024×1024** (the eval defaults are 512×512, inherited from
the SD 2.1-era `DEFAULT_WIDTH`/`DEFAULT_HEIGHT` in `scripts/eval.py:68-69`, not SDXL's native
resolution) — 3 runs, `reports/_latency_diag_1024.json`, log `reports/_latency_diag_1024.log:7,9,11`:
latency 18.15s / 14.11s / 14.04s, peak_alloc constant at **5.605 GB** (+0.24 GB vs. 512×512, a
modest increase, not a spike). This tests and **rejects** the hypothesis that the missing
attention-slicing call on the SDXL path (§1) causes a resolution-driven VRAM blow-up capable of
pushing this card into oversubscription — at native resolution, VRAM headroom is still
comfortable (≈3 GB free of 8.59 GB).

---

## 3. Cold-start weight download / re-load per generation

**Not the cause — ruled out by code structure and directly confirmed.** In both `scripts/eval.py`
and `scripts/_latency_diag.py`, the model is loaded **once**, outside the per-generation loop
(`scripts/eval.py:613-616`: `model = AetherModel(); model.init(...)` runs before the `for
prompt_entry, scheduler_name, n_steps in combos:` loop at line 649). Each individual
`run_single()` call reuses the already-loaded `model.pipe` — there is no `from_pretrained` call,
and therefore no possible disk or network I/O, inside the per-generation timing window. This
directly answers the question: whatever caused `pp_002`'s 1551.10s, it cannot have been
per-generation weight reloading, because the code doesn't reload weights per generation.

Confirmed the weights are fully cached locally and this is not a first-run-ever scenario either:
`~/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0` is **24 GB** on
disk, `models--madebyollin--sdxl-vae-fp16-fix` is 320 MB (`du -sh`, this session). Model-load
time measured directly across all 4 diagnostic runs this session:
`reports/_latency_diag.log:3` (3.05s), `reports/_latency_diag_1024.log:3` (2.95s),
`reports/_latency_diag_lora_unfused.log:3` (2.96s) — consistently fast, disk-cache-hit timings,
not the tens-of-seconds-to-minutes a 24 GB cold download would take. No `cache_dir`/`HF_HOME` is
configured app-level (confirmed in the prior audit, `docs/MODEL_AUDIT.md` §1) — the app relies
on the default shared `~/.cache/huggingface/hub`, which happens to already be warm on this
machine. This is a real portability/reproducibility gap (a fresh machine or container would
eat a 24 GB download on its very first run) but it cannot explain variance **between** runs
within the same already-loaded process, which is what `eval_partial_latest.json` shows.

---

## 4. Root cause

**No reproducible root cause was found in the current default SDXL pipeline.** Two concrete,
codebase-documented candidate mechanisms were tested and both failed to reproduce anything close
to the historical variance:

1. **VRAM oversubscription under CPU offload** (the task's working hypothesis, based on
   `docs/lab_notebook.md:200`'s "peaks at 7928 MB... right at the 8 GB ceiling" note for the
   ControlNet+SDXL config). Tested at both 512×512 and native 1024×1024: peak usage stayed at
   5.36–5.61 GB, 2.6–3.0 GB below the physical ceiling, in all 13 non-LoRA runs. This card was
   never driven close enough to its VRAM limit to trigger Windows WDDM's shared-memory
   overflow path (the mechanism that would explain a 5–50x slowdown without a hard CUDA OOM).
2. **Unfused LoRA + CPU-offload hook interaction** — the codebase's *own* comment at
   `scripts/eval.py:626-628` documents this exact class of bug: "unfused LoRA hooks + offload
   hooks interact badly — inference drops to ~90 s/step instead of ~3 s/step," which is why
   `scripts/eval.py:632` calls `fuse_lora()` immediately after `load_lora_weights()`. Deliberately
   reproducing the *unfused* case (`reports/_latency_diag_lora_unfused.json`,
   `reports/_latency_diag_lora_unfused.log:9`) gave `mean_step=0.32s` — normal speed, not the
   documented ~90s/step. On the current pinned versions (`diffusers==0.35.1`, `peft==0.19.1`,
   `accelerate==1.10.1` — `requirements.txt:7,9,12`), this specific hook-interaction bug does
   not reproduce. Most likely explanation: it was fixed upstream in diffusers/accelerate since
   that comment was written (the comment doesn't cite a version), or it depended on a
   `diffusers` release that predates the current pin. This is a **positive finding** — the
   defensive `fuse_lora()` call is no longer strictly necessary for this specific failure mode
   on current pins, though it remains cheap insurance and shouldn't be removed without a
   version-matrix test.

**What actually explains `eval_partial_latest.json`, most likely:** the historical run did not
execute under comparable conditions to "the current default SDXL path on this local GPU today."
Two independent, non-code-level explanations fit the evidence better than a still-latent bug:

- **It probably didn't run on this local RTX 3070 at all.** `docs/lab_notebook.md:208-224`
  independently documents active GCP L4 (24 GB) work on the **same calendar date**
  (2026-05-31) as `eval_partial_latest.json`'s `run_id: "20260531_205818"` and
  `reports/eval_results_20260531_200926.md`'s matching run. `docs/lab_notebook.md:143` separately
  documents "L4 stockouts across the ~11h run required three separate VMs" for this project's
  GCP work generally — i.e., this project's GCP L4 usage is independently known to have suffered
  VM churn/contention around this period. A 24GB L4 comfortably explains a peak-tracking counter
  reaching 11.186GB (impossible on the local 8.59GB card, unremarkable on a 24GB card), and
  shared/preemptible cloud GPU contention is a well-known source of exactly this kind of
  unexplained multi-x latency variance with no code-level cause.
- **The historical measurement of "VRAM peak" is independently proven stale** (§1) — an older
  `scripts/eval.py` without `scripts/eval.py:182`'s per-run `reset_peak_memory_stats()` call
  would report a carried-forward process peak, not a true per-run figure, regardless of which
  GPU it ran on.

Neither explanation implicates a bug in the SDXL pipeline code as it exists in this repo today.

---

## 5. What to actually do about it (minimal fix, not "average away the variance")

The task instruction was explicit: don't treat the variance as expected, and don't average it
away. Consistent with that — the fix here is not "the numbers are fine now, ignore the old
report," it's closing the **measurement gap** that made this un-diagnosable after the fact, plus
fixing the one concrete, currently-live problem this investigation surfaced:

1. **Add environment/hardware provenance to `scripts/eval.py`'s output.** Currently neither the
   per-run result dict (`scripts/eval.py:159-176`) nor the run-level `config` dict
   (`scripts/eval.py:581-592`) records GPU name, total VRAM, hostname, or torch/CUDA/driver
   version — confirmed via `grep -n "hostname|platform\.|get_device_name" scripts/eval.py`
   (no matches). This is why root-causing `eval_partial_latest.json` required this much forensic
   reconstruction instead of a one-line answer. Minimal fix: add
   `torch.cuda.get_device_name(0)`, `torch.cuda.get_device_properties(0).total_memory`, and
   `platform.node()` to the `config` dict once per run — three cheap calls, no new dependency.
2. **`scripts/eval.py:182`'s per-run `reset_peak_memory_stats()` is already correct** — no
   change needed there. This audit confirms the historical bug (if it was a measurement bug and
   not real 11GB usage) predates the current code and is already fixed.
3. **Add an outlier guard, not a silent average.** In `run_single()`
   (`scripts/eval.py:152-246`), after computing `latency`, compare it against a running
   median/p50 of prior results in the same sweep; if a single run exceeds, say, 3x the running
   median, log a `logger.warning(...)` with the current `torch.cuda.memory_allocated()` /
   `mem_get_info()` snapshot at that moment, so a future recurrence is caught with the exact
   diagnostic this investigation had to reconstruct by hand, rather than silently sitting in a
   JSON file until someone asks "why is this 5.6x."
4. **Fix the `torch/bin` vs `torch/lib` DLL split for local dev** (§0) is already applied to the
   `aetherart` conda env on this machine; the same fix is needed in `aetherart-torch28`
   (confirmed same missing files there — `fbgemm.dll`/`asmjit.dll` present in `torch/bin`, absent
   from `torch/lib`) if that environment is used for the pending torch 2.8 migration
   (`docs/torch28_compat.md`). Recommend a one-time `scripts/verify_env.py` sanity check
   (`import torch; assert torch.cuda.is_available()`) that a developer runs after any fresh
   `pip install torch` on Windows, so this doesn't silently block work again.
5. **Do not carry `eval_partial_latest.json`'s numbers into any decision.** It's already
   correctly gitignored (`.gitignore:16`) and never committed — keep it that way. If it's still
   referenced anywhere as "SDXL takes up to 25 minutes," that claim should be retracted: today's
   instrumented baseline is 6.4–18.3s per image at the same config, 15–86x faster.
6. **No GPU/VRAM headroom upgrade is indicated.** The hardware-needs-more-VRAM branch of the
   task's proposed fix (§4 of the task prompt) is not supported by this session's data — this
   card ran 14 SDXL generations, including at native 1024×1024, with 2.2+ GB of headroom to
   spare every time (worst case: 5971 MiB used of 8192 MiB, per
   `reports/_latency_diag_nvidia_smi.csv`). Don't spend budget on this without a *reproduced*
   VRAM-exhaustion event to justify it.

---

## Summary of unknowns (explicitly not resolved, not estimated)

- Which physical machine actually produced `reports/eval_partial_latest.json` — inferred as
  "likely GCP L4, not this local RTX 3070" from circumstantial but not conclusive evidence (no
  hostname/GPU-name was recorded at the time — this is exactly the gap §5.1 closes going forward).
- Exact stage that consumed the historical 1551.10s — unrecoverable without provenance the
  original run didn't capture; not fabricated here.
- Whether the documented unfused-LoRA/offload hook bug (`scripts/eval.py:626-628`) still exists
  on any other version combination — only tested on this repo's current pins
  (diffusers 0.35.1/peft 0.19.1/accelerate 1.10.1); not tested against older pins.
