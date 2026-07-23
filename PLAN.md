# AetherArt — Development Plan

## Phases

- [x] **#7 Metadata sidecar** — Phase 1 complete
  - PNG tEXt chunks + sidecar JSON on every generation
  - "Recreate from PNG" UI feature

- [x] **#1 PartiPrompts eval harness** — Phase 2 complete (360 generations, charts committed)
  - 4 schedulers × 3 step counts × 30 prompts, seed = 42, RTX 3070 8 GB
  - DPM-Solver++ recommended default (0.3177 avg CLIP)

- [x] **#3 ControlNet (Canny + Depth)** — Phase 3 complete

- [x] **#2 LoRA Ukiyo-e fine-tune** — Phase 4 complete

- [x] **#8 README polish** — Phase 5 complete

- [x] **Phase 6a** — README rewrite and chart polish complete

- [x] **Phase 6b** — Controlled experiments: CLIP-blindness series complete (9 experiments)

- [x] **Phase 6c** — Central narrative and project documentation complete

---

**Phase 7 — SDXL Modernization (in progress)**

- [ ] PR 01 — Housekeeping: PLAN.md, CI concurrency, load_dotenv()
- [ ] PR 02 — Ruff migration + CI hardening
- [ ] PR 02a — torch 2.8 compatibility spike (non-merging)
- [ ] PR 02b — Docs-only: torch28_compat.md to main
- [ ] PR 03 — SDXL base pipeline
- [ ] PR 04 — Publish SD 2.1 Ukiyo-e adapter to HF Hub
- [ ] PR 05 — Hyper-SD 8-step LoRA integration
- [ ] PR 06 — NF4 quantization for SDXL + combined path smoke test
- [ ] PR 07 — Modal demo deployment + safety guard
- [ ] PR 08 — xinsir ControlNet Union SDXL
- [ ] PR 09 — SDXL LoRA retraining (GCP L4)
- [ ] PR 10 — SDXL Turbo legacy gate + license documentation
- [ ] PR 11 — Publish SDXL Ukiyo-e adapter to HF Hub
- [ ] PR 12 — HPSv2.1 + ImageReward eval integration
- [ ] PR 13 — Phase 6b experiments migration to SDXL
- [ ] PR 13b — HPSv3 GCP batch eval
- [x] PR 14 — CI quality gate: frozen SD 2.1 30-prompt/seed-42 CLIP regression check
      (`.github/workflows/eval.yml`, push-to-main + nightly + manual dispatch — not
      per-PR, see that file's header comment for the measured-cost rationale) +
      real (unmocked) generation smoke test on every push/PR
      (`.github/workflows/ci.yml`, `tests/test_generation_smoke.py`) + HF revision
      pinning across config/utils/model/clip_scorer. Coverage threshold
      (`--cov-fail-under=55`) was already enforced via `pyproject.toml` before this.
      HPS/ImageReward deliberately left out of the gate pending the
      Windows/headless-Linux crash fixes tracked in `docs/lab_notebook.md:246-250,269`.
      Landed as 3 separate PRs (#22 CI gate, #23 revision pinning, #24 smoke test)
      after the original combined PR #20 (508+/-8 diff) tripped this repo's
      merge-gate size hook — each split PR carries its own verifier review artifact.

---

**Phase 8 — Cross-family model verdict (in progress, checkpoint 2026-07-23)**

- [x] `docs/MODEL_AUDIT.md`, `docs/LATENCY_ROOT_CAUSE.md` — prior-session groundwork (repo runs
      5 model families, not 2; phantom 11GB-VRAM report row traced to GCP L4 contention).
- [x] All 5 non-LoRA families scored on identical 30-prompt×3-seed set, 0 errors —
      `reports/verdict_{sd21_base,sdxl_base,hyper_4step,hyper_8step,sdxl_controlnet_union}.json`.
- [x] Hyper-SD 4-step vs. 8-step HPS inversion root-caused (`docs/MODEL_VERDICT.md` §3):
      config-routing confirmed correct (falsified plumbing-bug hypothesis); real cause is
      CFG-driven exposure collapse on the 8-step CFG-preserving variant (16% severe
      underexposure, 7% black-crush at guidance_scale=5.0 — already the least-bad point in
      Hyper-SD's own documented 5-8 range; no in-spec fix exists). Verdict: route to
      `hyper_4step`, not `hyper_8step`.
  - [x] Bug found + fixed while building the LoRA A/B harness: `run_ukiyo_e_lora_family`
        (`scripts/model_verdict_harness.py`) called the Ollama VLM judge inline while the
        SDXL+LoRA pipeline was still GPU-resident — same VRAM-oversubscription pathology as
        `docs/LATENCY_ROOT_CAUSE.md`. Fixed to a two-phase pattern (defer VLM scoring until
        after the pipeline is released), matching the existing HPS-deferral pattern.
  - [x] LoRA A/B complete: published checkpoint-1000 vs. curated retrain, n=90 each, 0 errors
        (`reports/verdict_ukiyo_e_lora_sdxl{,_curated}.json`). Primary 30-prompt benchmark is a
        wash on all 4 axes (paired diff, all < 1 SEM). Supplementary targeted check on 4
        ukiyo-e-style prompts (`reports/lora_ab_targeted.json`, n=12 paired) shows
        `artifact_absence` crossing the project's 2×SEM bar (2.03) — borderline-significant on a
        small, fragile sample. **Curated retrain NOT promoted** on current evidence; recommend a
        larger targeted eval (more portrait/figure subjects) before a final call.
- [x] Verifier pass on `docs/MODEL_VERDICT.md` — flagged one real issue: §4.1 and §4.2 used
      inconsistent diff-SEM methodology (paired vs. independent-quadrature) without stating
      either. Fixed by making paired-diff explicit and consistent across both; §4.2's
      recomputation with the correct paired method changed its conclusion (1.80 SEM → 2.03 SEM,
      now crosses the bar). All other sections (§2, §3, §4.2's raw means, curation-report counts)
      verified exact against raw JSON, zero discrepancies.
- **Autonomy setup (partially blocked, not silently worked around):**
  - Branch protection on `main` already permits merge-without-human-approval
    (`required_pull_request_reviews.required_approving_review_count: 0`, confirmed via
    `gh api repos/.../branches/main/protection`) — no GitHub-side change was needed.
  - The local mechanical merge-gate hook (`hook_guard_merge.py`, enforcing rule 70a) blocked
    PR #20 (508+/-8 diff, over the ~400-line reviewable-diff gate). Per explicit user decision
    this session: **left PR #20 and (transitively, since it's stacked) PR #21 as drafts for
    manual merge** rather than splitting or overriding the gate.
  - Both HF model-card draft PRs (`refs/pr/1` on `aetherart-ukiyo-sdxl` and `aetherart-ukiyo-sd21`)
    are still open — blocked on a **read-only** HF token (`api.whoami()` confirms
    `role: read`); merging them needs a write-scoped token or manual action on huggingface.co.
