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
- [ ] PR 14 — Coverage threshold enforcement
