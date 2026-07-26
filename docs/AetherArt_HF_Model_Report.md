# AetherArt — Hugging Face Model Review & Improvement Report
**Status: Part 1 of 2 (external landscape). Part 2 (current-state audit) blocked on repo access — see note.**

---

## 0. Scope note

I don't have read access to `C:\Users\gaura\ml-projects\AetherArt` (local machine) or a resolvable public GitHub repo for it. Memory only records "SDXL LoRA + ControlNet" — not enough to state real integration details, current eval numbers, or latency without fabricating. The `MODEL_AUDIT.md` CC prompt above closes that gap. This document covers everything answerable without your codebase: the current SDXL/ControlNet ecosystem state, upgrade paths, and a fine-tuning/cost framework you can drop your real numbers into once the audit returns.

---

## 1. Current model family — what SDXL + ControlNet means today (2026)

**SDXL base**: ~3.5B param UNet, CreativeML Open RAIL+M license (commercial use permitted, with content-restriction clauses). Native 1024×1024. Largest LoRA/ControlNet ecosystem of any open image model — thousands of CivitAI checkpoints, mature tooling (diffusers, ComfyUI, Automatic1111/Forge). Runs comfortably in fp16 on 8GB VRAM.

**ControlNet-for-SDXL**: mature, multiple conditioning types (canny, depth, pose, soft-edge). Preprocessing chain typically via `controlnet_aux`.

**Known ceiling**: SDXL's in-image text rendering is effectively unusable (gibberish), and prompt adherence on complex multi-subject prompts lags newer architectures.

## 2. What's changed since SDXL was current — the real alternative set

| Model | Params | License | Text-in-image | VRAM (quantized) | Ecosystem | Verdict for AetherArt |
|---|---|---|---|---|---|---|
| **SDXL** (current) | 3.5B | RAIL+M, commercial OK | Poor | 8GB fp16 | Largest (LoRA/ControlNet/CivitAI) | Baseline — keep for cost/ecosystem |
| **SD 3.5 Large** | ~8B | Stability community license — free non-commercial, paid tier above revenue threshold | Fair | ~18GB @8-bit | Thin | Skip — worse quality/VRAM tradeoff than either neighbor, most sources actively recommend skipping it |
| **FLUX.1 [schnell]** | 12B | Apache 2.0 — fully commercial | Good | 12-19GB GGUF Q4-Q5 | Growing, ControlNet-Union available | Best free-commercial upgrade path |
| **FLUX.1 [dev]** | 12B | Non-commercial only (needs BFL commercial license for production) | Best-in-class among dev-tier | Same as schnell | Same | Blocked unless you license it — same output quality as schnell isn't guaranteed, schnell is distilled for speed not identical to dev |
| **FLUX.2** (newer gen) | Klein 4B / 9B / dev 32B | Klein: Apache 2.0. dev: BFL license (check current commercial terms) | Best | Klein 4B fits <20GB; dev needs FP8/GGUF on 24GB+ | Newer, smaller LoRA pool | Klein 4B is the most interesting line item — Apache 2.0, small enough for cost-sensitive inference, from the team that built the FLUX line |

**Commercial-license-clean picks for a product you plan to monetize**: SDXL (keep) or FLUX.1-schnell / FLUX.2-Klein (Apache 2.0, no revenue-threshold clause). FLUX.1-dev and FLUX.2-dev require a paid commercial license from Black Forest Labs above their free tier — treat as a real line-item cost if you evaluate them, not a free upgrade.

**ControlNet for FLUX**: `Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0` is the current standard — single union model covering canny/depth/pose/soft-edge (tile support was dropped in v2.0). FP8-quantized builds exist (Kijai's `flux-fp8`), cutting VRAM roughly in half with reported near-identical output to bf16 at Q8/FP8, larger quality loss below that.

## 3. Improvement levers, ranked by expected ROI (pending your real baseline)

| Lever | Effort | Expected gain | Risk |
|---|---|---|---|
| **Quantize current SDXL pipeline (fp8/bnb-int8)** | Low (1-2 days) | Latency/cost cut, quality loss usually imperceptible at fp8/int8 for UNet | Low — reversible, A/B against current output |
| **Add ControlNet-Union (multi-conditioning in one model)** if you're running separate ControlNet checkpoints per conditioning type today | Low-Med | Fewer models to load/cache, simpler pipeline, likely small VRAM/latency win | Low |
| **Swap in FLUX.1-schnell as a second pipeline (not replacement)** for prompts needing text-in-image or complex composition | Medium (new pipeline, new eval) | Meaningful quality jump on the specific failure modes SDXL has (text, multi-subject) | Medium — 3-4x heavier compute per image even quantized; needs its own cost model |
| **LoRA fine-tune current SDXL on your domain data** (fashion/occasion imagery per Style Maitri overlap, or whatever AetherArt's actual domain is) | Medium (needs the audit to know current LoRA setup) | Domain-specific quality gain, hard to estimate % without a baseline CLIP/human-eval score | Medium — needs labeled/curated dataset, eval harness to prove it's actually better and not overfit |
| **Full FLUX.2-Klein migration** | High | Best long-term quality/cost frontier if Apache 2.0 holds and 4B fits your latency budget | Higher — smaller LoRA/ControlNet ecosystem today, may need custom fine-tunes you don't currently have to build from scratch |

I'm deliberately not putting percentage numbers on quality gains — no eval harness exists in memory for AetherArt (unlike TriageIQ's CLIP-score gate), and inventing a number would violate the numbers-over-adjectives rule. This is the first gap the audit should surface: **does AetherArt have any quality eval today (CLIP score, human rubric, FID)?** If not, that's a bigger blocker than model choice — you can't measure whether any of these levers actually helped.

## 4. Fine-tuning feasibility — general framework

Applies once we know your actual base model config from the audit.

- **LoRA fine-tune (not full fine-tune)** is the only practical option at your scale — full SDXL fine-tune needs multi-A100 budgets you've already ruled out for zero-cost-API-key reasons on other projects.
- **Dataset**: 50-200 curated image-caption pairs is the practical minimum for a style/domain LoRA; below that, overfitting risk is high. Needs your domain (fashion/Indian occasion wear, if this overlaps Style Maitri's 52,494-item catalogue — reuse that data instead of sourcing new).
- **Compute**: SDXL LoRA fine-tune is feasible on a single 16-24GB consumer GPU (rank 16-32, ~1500-3000 steps), hours not days. Cloud alternative: rentable A100/L4 spot instance, low single-digit dollars per run.
- **Risk**: style LoRAs commonly overfit to the training set's composition/lighting and generalize poorly — needs held-out eval images, not just visual spot-check, matching your documented "verified ≠ correct" discipline from Warmer/Reclaim.
- **FLUX LoRA fine-tune** is the same shape but ~3x compute per step given 12B vs 3.5B params — budget accordingly if you go that route instead of/alongside SDXL.

## 5. Multi-model scope — feasibility for AetherArt

Running two pipelines (SDXL + FLUX-schnell) side by side is standard practice in 2026 tooling (ComfyUI treats this as normal) — not a scalability risk by itself. The real cost is **eval and maintenance surface doubling**: two model caches, two dependency footprints (diffusers version compatibility can diverge between UNet and DiT-transformer pipelines), two latency/cost profiles to monitor. Recommend: keep SDXL as default (cost/ecosystem), add FLUX-schnell as an opt-in path for a specific failure mode (text-in-image, complex composition) rather than a blanket upgrade — this is a routing decision, not a replacement decision, until the audit shows SDXL is failing broadly rather than on specific prompt classes.

## 6. Roadmap

**Short-term (this week, no new model)**
1. Run the audit CC prompt above → real baseline (models, config, any existing eval, latency).
2. Quantize current SDXL inference to fp8/int8 — cheap latency/cost win, no quality-risk architecture change.
3. If no quality eval exists, build one (even a 20-prompt CLIP-score + human rubric harness) before touching model choice — you can't validate any of the levers below without it.

**Medium-term (2-4 weeks)**
4. Stand up FLUX.1-schnell (Apache 2.0, no licensing blocker) as a second pipeline, gated to prompts needing text/complex composition.
5. If AetherArt has a defined domain (fashion overlap with Style Maitri's catalogue is the obvious lever if applicable) — scope a LoRA fine-tune with a held-out eval set.

**Long-term**
6. Evaluate FLUX.2-Klein once its ecosystem (LoRA/ControlNet-Union support) matures past early-2026 thinness — re-check in a quarter, don't commit now.
7. Only pursue FLUX.1/2-dev (non-Apache tier) if a customer conversation surfaces revenue that justifies BFL's commercial license — this is a "who pays" gate, not a technical one.

---

**Next highest-impact task**: run the `MODEL_AUDIT.md` CC prompt above, paste the output back — I'll turn this into a real Part 2 with your actual model IDs, config, and gap analysis instead of the general framework above.
