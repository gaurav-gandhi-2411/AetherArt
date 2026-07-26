# AetherArt — Part 2: Findings & Recommendations (against real audit)

## Top-line correction to Part 1

Part 1 assumed a two-model architecture and recommended fp8/int8 quantization as a cheap win. Both assumptions are wrong:

- **Five model families are in production**, not two: SD 2.1, SDXL base, SDXL Turbo (legacy-gated), SD 2.1 ControlNet, SDXL ControlNet-Union — plus a Hyper-SD LoRA and a trained Ukiyo-e domain LoRA.
- **Quantization is not a clean win on this codebase's own data.** The canonical SD 2.1 harness (`reports/experiments/exp1_quantization_quality/findings.md`) shows fp16 at 4.4s/1803MB, NF4 at 6.4s (slower) for a marginal CLIP gain (0.3158 vs 0.3124), INT8 at 12.3s (worse on every axis). No SDXL-specific quantization harness with current diffusers 0.35.1 exists — the SDXL quant numbers you have are superseded. **Withdraw the Part 1 quantization recommendation until it's re-benchmarked on SDXL specifically.**

The real highest-leverage problem is upstream of any model decision: **you cannot currently tell whether any change makes AetherArt better**, because quality metrics are measured but not gated, and latency numbers are means over huge variance, not percentiles. Fix measurement before touching models.

## Findings, ranked by production-readiness risk

| # | Finding | Evidence | Risk |
|---|---|---|---|
| 1 | **No CI quality gate.** CLIP/HPS/ImageReward are computed and logged to one-off report files, but `tests/test_eval.py` etc. only assert types/shapes, never a threshold. A regression can ship silently. | `.github/workflows/ci.yml:26-57`, `tests/test_eval.py:34-88` | High — this is the TriageIQ eval-gate problem you already solved once elsewhere; AetherArt hasn't had that fix applied yet |
| 2 | **SDXL latency variance is 5.6x** (277s–1551s per image, same config, n=3) with no root cause investigated. | `reports/eval_partial_latest.json:35,53,71` | High — unexplained 5x variance is a bug, not noise; blocks any legitimate before/after comparison of a future change |
| 3 | **No pinned HF revision anywhere.** A future upstream weight update silently changes output with zero warning, and it already bit you once (CVE-2025-32434 forced an unplanned depth-estimator swap). | `config.py` (no `revision=` kwarg found), `docs/depth_estimators.md:9-24` | Medium-high — reproducibility and supply-chain risk |
| 4 | **Dead/inconsistent config**: `cfg.sdxl_controlnet_canny`/`depth` declared, never read (Union already covers both); `cfg.device` declared, never read; docs name a different SDXL depth checkpoint (`diffusers/controlnet-depth-sdxl-1.0`) than the one hardcoded (`xinsir/controlnet-depth-sdxl-1.0`). | `config.py:49-51,60`, `controlnet_sdxl.py:48-50` | Medium — not causing bugs today, but is exactly the kind of drift that causes the wrong assumption six months from now |
| 5 | **LoRA calligraphy-artifact entanglement** — style and text signal are entangled in the adapter weights; negative-prompt mitigation is partial, not fixed. A scoped ~5h curation fix was identified and not done. | `docs/lab_notebook.md:23-25` | Medium — known, bounded, cheap to close |
| 6 | **eval toolchain is Windows-hostile**: ImageReward SIGSEGVs on Windows (pyarrow C extension), hpsv2 1.2.0 crashes on headless Linux (stray `from turtle import forward` in a dependency). Full eval only runs on GCP with workarounds. | `docs/lab_notebook.md:246-250,269` | Medium — operational friction, not correctness risk, but it's why finding #1 hasn't been closed yet |
| 7 | **transformers/diffusers pin coupling is brittle** for a pending torch 2.8 upgrade; bitsandbytes flagged as highest-risk dependency, untested under torch 2.8. | `docs/torch28_compat.md:52,219,240` | Low-medium — tracked, not yet urgent |

## Recommendations, in order

**1. Root-cause the SDXL latency variance (this week).** A 5.6x spread between 277s and 1551s on identical config almost certainly means one run hit cold-start weight download, thermal throttling, or CPU-offload thrashing under VRAM pressure (the lab notebook confirms CPU offload is in play — peak 7928MB, ~275-292s baseline, so the 1551s outlier is ~5.6x that baseline, not just sweep noise). This is a correctness-of-measurement problem before it's a performance problem — nothing else in this list is trustworthy while it's unresolved.

**2. Close PLAN.md PR 14 — wire the CI quality gate.** You already have the pattern from TriageIQ (mean-vs-baseline within N×SEM tolerance band). Minimum viable version: CLIP-only threshold in CI (already installed, `openai-clip==1.0.1`), using the existing SD 2.1 30-prompt/seed-42 baseline (0.3199) as the anchor. Add HPS/ImageReward once the Windows/Linux crash issues (finding #6) are patched — don't block the CLIP gate on that.

**3. Fix the hpsv2/ImageReward crashes, then extend the gate.** The hpsv2 bug is a one-line stray import in a dependency — patch or pin around it rather than treating "Windows is interactive dev only" as permanent. This directly reopens the door to a full CLIP+HPS+IR gate instead of a CLIP-only one.

**4. Pin HF revisions on every `from_pretrained` call.** Low effort, closes a real reproducibility gap, and prevents a repeat of the CVE-2025-32434 surprise.

**5. Delete or wire the dead config** (`sdxl_controlnet_canny`/`depth`, `cfg.device`) and resolve the depth-checkpoint doc/code mismatch. Trivial, but this is the kind of thing that silently reintroduces the exact "verified from UI toast, not from real state" pattern documented in your Warmer/Reclaim history — a future reader will assume the declared-but-dead ControlNet fields are load-bearing.

**6. Close the LoRA calligraphy-artifact fix** (~5h, already scoped in your own notes) — bounded, known ROI, no new research needed.

**7. Re-benchmark quantization on SDXL specifically**, current diffusers/torch stack, before deciding anything about fp8/GGUF/FLUX quantization economics. Don't extrapolate from the superseded SD 2.1 numbers.

**8. Defer the FLUX/model-family conversation.** Part 1's landscape research (FLUX.1-schnell/FLUX.2-Klein, Apache 2.0, text-in-image, ControlNet-Union available) is still accurate and worth revisiting — but adding a sixth model family before the existing five have a working quality gate and an explained latency profile compounds the measurement problem instead of fixing it. Revisit after items 1-2 land.

## What's already right — don't change

- SDXL Turbo's ADD non-commercial license is correctly gated behind `AETHERART_ENABLE_LEGACY=1`, kept out of the commercial-framed default path. This is the licensing discipline the FLUX.1-dev decision in Part 1 needs to replicate if you ever add it.
- ControlNet-Union consolidation (8 conditioning types in one checkpoint) is the right architecture — the dead standalone canny/depth config is leftover from before that consolidation, not a sign the consolidation was wrong.
- CLIP-blindness self-correction (initial "9/9 blind" qualitative claim revised to "4/9" under a proper 1-SE cutoff) is exactly the rigor this list wants more of — the team caught its own overclaim once already.

---

**Next highest-impact task:** CC prompt to root-cause the SDXL latency variance (finding #1) — that's the blocker under everything else on this list.

```
AetherArt: root-cause SDXL latency variance — orchestrator + executor + verifier

reports/eval_partial_latest.json shows three SDXL generations at identical config
(DPM/20 steps) taking 276.97s, 1551.10s, 736.51s — a 5.6x spread. docs/lab_notebook.md
separately reports a ~275-292s baseline for ControlNet+CPU-offload SDXL at peak VRAM
7928MB, suggesting CPU offload is active by default on this hardware.

Investigate and report findings to docs/LATENCY_ROOT_CAUSE.md, with file:line citations
and instrumented timings (not estimates):
1. Is CPU offload (enable_model_cpu_offload / enable_sequential_cpu_offload) active on
   the default SDXL path, and does VRAM pressure vary run-to-run (e.g. other processes,
   fragmentation, first-call weight download)?
2. Instrument scripts/eval.py or app.py to log per-stage timing (model load, encode,
   denoise loop, decode/VAE) for the next 10 SDXL generations — isolate which stage
   the 1551s outlier spent time in.
3. Check whether weights are re-downloaded or re-loaded from disk on cold start vs.
   warm — no cache_dir/HF_HOME is configured app-level (audit finding), so confirm
   whether this causes repeated disk/network I/O per run.
4. Once root cause is identified, propose the minimal fix — do not average away the
   variance or treat it as expected; if it's CPU-offload thrashing, evaluate whether
   the hardware needs more VRAM headroom or whether offload is misconfigured.

Verifier subagent re-checks every timing claim against actual instrumented output,
not the researcher's summary.
```
