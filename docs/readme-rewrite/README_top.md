# AetherArt

**SDXL pipeline engineering on 8 GB of VRAM** — LoRA fine-tuning, ControlNet, Hyper-SD fast mode, Cloud Run deployment, and a self-corrected CLIP-blindness study.

[![Live Demo](https://img.shields.io/badge/☁️_Live_Demo-Cloud_Run_L4-blue)](https://aetherart-demo-473907703523.us-central1.run.app)
[![GitHub](https://img.shields.io/badge/GitHub-AetherArt-181717?logo=github)](https://github.com/gaurav-gandhi-2411/AetherArt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## The finding that drove this project

CLIP score — the standard proxy for "does this image match the prompt?" — is blind to many quality changes that matter.

<p align="center">
  <img src="reports/showcase/clip_blindness_hero.png" width="860"
       alt="CLIP-Blindness Study — SDXL: horizontal bar chart showing CLIP delta in SE units vs LPIPS range for 7 experiments. Teal bars (Quantization, Scheduler, Trigger token) stay below 1 SE while LPIPS shows 0.20–0.45 range. Orange bars (Neg prompt, ControlNet, CFG, LoRA alpha) exceed 1 SE.">
  <br>
  <em>Teal bars: CLIP cannot detect quality changes that LPIPS registers.  
  Orange bars: CLIP can — because these sweeps shift semantic content, not just visual character.</em>
</p>

**Source:** `reports/experiments/exp*_sdxl/results.json` — reproduced by `scripts/generate_clip_blindness_hero.py`

### The self-correction

The original SD 2.1 study used a qualitative judgment for the blind/responds call. Applying a hard 1-SE statistical threshold to the SDXL replication — then going back and applying the same threshold to the original data — gave a different answer:

| Stage | CLIP-blind result | Note |
|---|---|---|
| Original SD 2.1 claim | 9 / 9 | Qualitative CLIP-vs-LPIPS ratio judgment |
| SD 2.1 recomputed (1-SE threshold) | 4 / 9 | Exp3/4/5 were above 1 SE; called blind qualitatively |
| SDXL replication (1-SE threshold) | 3 / 7 | Architecture-dependent; SDXL is less CLIP-blind |

Source: `reports/clip_blindness_sdxl.md` (sensitivity table, < 1.0 SE row; SD 2.1 correction note in `reports/clip_blindness.md`)

The conclusion — **CLIP is structurally blind to rendering-level parameters** (quantization precision, scheduler choice, trigger-token presence) while remaining responsive to semantic sweeps (CFG scale, LoRA style intensity) — holds at every threshold tested. The direction is stable: SDXL is less CLIP-blind than SD 2.1 regardless of where the line is drawn.

→ Full analysis: [reports/clip_blindness_sdxl.md](reports/clip_blindness_sdxl.md)

---

## What I built

- **SDXL on 8 GB VRAM** — NF4 4-bit quantization peaks at 2.6 GB; FP16 + Ukiyo-e LoRA peaks at 6.2 GB. Both fit an 8 GB consumer GPU. Both measured on GCP L4.
- **Ukiyo-e LoRA** — rank-8 adapter, 80 WikiArt training images, 4 h 26 min on GCP L4 (~$3.50). Checkpoint selected from 500/1000/1500 sweep scored by HPSv2.1 + ImageReward.
- **Hyper-SD fast mode** — ByteDance LoRA cuts DPM-Solver++ from 30 steps to 8 (~8 s/image on L4) while keeping CFG guidance and negative prompts intact.
- **ControlNet** — Canny edge + depth-map conditioning at 1024×1024. Two depth estimators (security/licensing rationale in [docs/depth_estimators.md](docs/depth_estimators.md)).
- **FP16-fix VAE** — `madebyollin/sdxl-vae-fp16-fix` is a required component. Without it every SDXL FP16 generation silently fails.
- **CLIP-blindness study** — 9 controlled experiments on SD 2.1 then replicated on SDXL. Applied stricter threshold, corrected the original result from 9/9 → 4/9. The self-correction is the interesting part.

---

## Results dashboard

### Scored outputs — Ukiyo-e LoRA, seed 42, DPM-Solver++ 30 steps, 1024×1024

Scores from GCP L4 eval run (2026-06-01). HPS and ImageReward are the operative quality metrics — CLIP is comparison-only (blind to style quality; see above).

| Prompt | CLIP | HPS | ImageReward | Image |
|---|---|---|---|---|
| ukyowood ukiyo-e print of **a crane over ocean waves** | 0.362 | 0.282 | 1.846 | <img src="reports/showcase/hero_crane_ocean_waves.png" width="200"> |
| ukyowood ukiyo-e woodblock print of **Mount Fuji at sunset** | 0.365 | 0.218 | 1.607 | <img src="reports/showcase/lora_fuji_sunset.png" width="200"> |
| ukyowood ukiyo-e woodblock print of **a samurai in a bamboo forest** | 0.348 | 0.242 | 1.138 | <img src="reports/showcase/lora_samurai_bamboo.png" width="200"> |
| ukyowood ukiyo-e print of **cherry blossoms along a river** | 0.362 | 0.215 | 1.324 | <img src="reports/showcase/lora_cherry_blossoms.png" width="200"> |

*Source: `docs/lab_notebook.md` lines 260–263 (GCP L4 validation run, 2026-06-01)*

### VRAM by configuration

| Config | Peak VRAM | Source |
|---|---|---|
| SDXL NF4 4-bit (bitsandbytes) | **2.6 GB** (2611 MB) | `reports/experiments/exp1_sdxl/results.json` — all NF4 rows |
| SDXL FP16 + Ukiyo-e LoRA | **6.2 GB** | GCP L4 eval run 2026-06-01 (`docs/lab_notebook.md` line 266) |

---

## What the LoRA does

Same prompt · same seed (42) · the only change is whether the Ukiyo-e adapter is loaded. The trigger word `ukyowood` activates it.

<table>
  <tr>
    <td align="center">
      <img src="reports/showcase/lora_before.png" width="380" alt="Standard SDXL output — no LoRA adapter"><br>
      <em>Without LoRA — standard SDXL base output</em>
    </td>
    <td align="center">
      <img src="reports/showcase/lora_after.png" width="380" alt="Ukiyo-e LoRA output — same prompt and seed"><br>
      <em>With Ukiyo-e LoRA — Hiroshige palette, flat planes, title cartouche</em>
    </td>
  </tr>
</table>

---

## Try it / Run it

**Live demo — no setup:**

**[→ aetherart-demo-473907703523.us-central1.run.app](https://aetherart-demo-473907703523.us-central1.run.app)**

Type: `ukyowood ukiyo-e woodblock print of Mount Fuji at sunset` — the trigger word `ukyowood` activates the style adapter. Fast mode (Hyper-8step, ~8 s) is on by default. Cold start ~5–7 min on first load.

**Run locally:**

```bash
git clone https://github.com/gaurav-gandhi-2411/AetherArt.git
cd AetherArt
conda create -n aetherart python=3.10 -y
conda activate aetherart
pip install -r requirements.txt
python cloudrun_app.py  # same 4 modes as the live demo at http://localhost:7860 — downloads ~10 GB on first run
```

No GPU required for tests (all heavy deps mocked):

```bash
pytest -q              # 229 tests, ~60 s
```

VRAM: measured on GCP L4 with `enable_model_cpu_offload()` — SDXL FP16 + Ukiyo-e LoRA 6.2 GB peak, SDXL NF4 2.6 GB peak. An 8 GB GPU is sufficient. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full dev environment including CUDA-specific torch install and exact package lock.

---

<!-- ═══════════════════════════════════════════════════════════════════
     EVERYTHING BELOW THIS LINE IS UNCHANGED FROM THE CURRENT README
     ═══════════════════════════════════════════════════════════════════ -->
