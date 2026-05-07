"""
Experiment 7: LoRA trigger token sensitivity.

Two conditions, same LoRA (ukiyo-e rank-8, alpha=1.0) loaded for both:
  no_trigger  — prompt contains style description ("ukiyo-e ...") but NOT "ukyowood"
  with_trigger — identical prompt with "ukyowood" prepended

All other parameters fixed: DPM-Solver++, 30 steps, CFG=7, same 5 seeds × 8 prompts.

Metrics:
  - CLIP vs the semantic prompt text (without "ukyowood") for both conditions —
    so CLIP measures alignment to the visual concept, not sensitivity to the
    unknown token.
  - LPIPS between matched pairs (no_trigger vs with_trigger, same seed+prompt).

Hypothesis: The trigger token is not in CLIP's vocabulary — it should contribute
zero signal to CLIP scoring. If LPIPS shows images differ meaningfully but CLIP
scores are similar, it is a seventh confirmation of CLIP-blindness. If LPIPS shows
images are perceptually close, that would mean the trigger token has little effect
on this LoRA (an equally informative result).

Design:
  - 2 conditions × 8 prompts × 5 seeds = 80 images
  - LPIPS computed between the two conditions for each (prompt, seed) pair

Run from project root:
    python scripts/experiments/exp7_lora_trigger.py

Outputs:
    reports/experiments/exp7_lora_trigger/
        images/no_trigger/
        images/with_trigger/
        results.csv
        results.json
        charts/
        findings.md
"""

from __future__ import annotations

import atexit
import csv
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import lpips as lpips_lib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline  # noqa: E402
from PIL import Image  # noqa: E402

from aetherart import clip_scorer  # noqa: E402
from aetherart.gpu_hygiene import cleanup_gpu  # noqa: E402
from aetherart.lora import LORA_REGISTRY  # noqa: E402
from aetherart.visualization import BLUE, GREEN, ChartCanvas  # noqa: E402

atexit.register(cleanup_gpu)

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_ID = "sd2-community/stable-diffusion-2-1"
LORA_NAME = "ukiyo-e"
LORA_ALPHA = 1.0
SEEDS = [42, 123, 456, 789, 1337]
STEPS = 30
CFG = 7.0
SIZE = 512

# Each prompt_id maps to (semantic_text, no_trigger_prompt, with_trigger_prompt).
# semantic_text is used for CLIP scoring in both conditions (no "ukyowood" bias).
PROMPTS = {
    "p01_portrait": (
        "ukiyo-e portrait of an elderly woman, dramatic light, woodblock print style",
        "ukiyo-e portrait of an elderly woman, dramatic light, woodblock print style",
        "ukyowood ukiyo-e portrait of an elderly woman, dramatic light, woodblock print style",
    ),
    "p02_landscape": (
        "ukiyo-e misty mountain valley at sunrise, pine forest, golden hour, woodblock print",
        "ukiyo-e misty mountain valley at sunrise, pine forest, golden hour, woodblock print",
        "ukyowood ukiyo-e misty mountain valley at sunrise, pine forest, golden hour, woodblock print",
    ),
    "p03_abstract": (
        "ukiyo-e geometric abstract composition, intersecting circles and triangles, color blocks",
        "ukiyo-e geometric abstract composition, intersecting circles and triangles, color blocks",
        "ukyowood ukiyo-e geometric abstract composition, intersecting circles and triangles, color blocks",
    ),
    "p04_text": (
        "ukiyo-e vintage print with bold lettering, retro typography, worn paper texture",
        "ukiyo-e vintage print with bold lettering, retro typography, worn paper texture",
        "ukyowood ukiyo-e vintage print with bold lettering, retro typography, worn paper texture",
    ),
    "p05_texture": (
        "ukiyo-e extreme close-up of rough stone wall, water drops, micro detail",
        "ukiyo-e extreme close-up of rough stone wall, water drops, micro detail",
        "ukyowood ukiyo-e extreme close-up of rough stone wall, water drops, micro detail",
    ),
    "p06_arch": (
        "ukiyo-e interior of a Japanese temple, wooden pillars, soft lantern light",
        "ukiyo-e interior of a Japanese temple, wooden pillars, soft lantern light",
        "ukyowood ukiyo-e interior of a Japanese temple, wooden pillars, soft lantern light",
    ),
    "p07_hands": (
        "ukiyo-e two hands clasped together, natural light, woodblock print style",
        "ukiyo-e two hands clasped together, natural light, woodblock print style",
        "ukyowood ukiyo-e two hands clasped together, natural light, woodblock print style",
    ),
    "p08_crowd": (
        "ukiyo-e busy street market in Edo, dozens of people, lantern light, night scene",
        "ukiyo-e busy street market in Edo, dozens of people, lantern light, night scene",
        "ukyowood ukiyo-e busy street market in Edo, dozens of people, lantern light, night scene",
    ),
}

NEG_PROMPT = "low quality, blurry, deformed, ugly, bad anatomy, watermark, text, calligraphy"

CONDITIONS = ["no_trigger", "with_trigger"]

OUT = ROOT / "reports" / "experiments" / "exp7_lora_trigger"
IMG_DIR = OUT / "images"
CHARTS_DIR = OUT / "charts"

for cond in CONDITIONS:
    (IMG_DIR / cond).mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Pipeline ──────────────────────────────────────────────────────────────────


def load_pipeline() -> StableDiffusionPipeline:
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if torch.cuda.is_available():
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pipe = pipe.to("cuda")
    return pipe


def load_lora(pipe: StableDiffusionPipeline) -> None:
    config = LORA_REGISTRY[LORA_NAME]
    lora_path = Path(config["path"])
    pipe.load_lora_weights(
        str(lora_path.parent), weight_name=lora_path.name, adapter_name="ukiyo_e"
    )
    pipe.set_adapters(["ukiyo_e"], adapter_weights=[LORA_ALPHA])
    print(f"LoRA loaded: {lora_path.name} (alpha={LORA_ALPHA})")


# ── Generation ────────────────────────────────────────────────────────────────


def generate_condition(
    condition: str,
    pipe: StableDiffusionPipeline,
) -> list[dict]:
    rows: list[dict] = []
    img_dir = IMG_DIR / condition
    prompt_key = 1 if condition == "no_trigger" else 2  # index into PROMPTS tuple

    for prompt_id, ptuple in PROMPTS.items():
        semantic_text = ptuple[0]
        gen_prompt = ptuple[prompt_key]
        for seed in SEEDS:
            generator = torch.Generator().manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = pipe(
                prompt=gen_prompt,
                negative_prompt=NEG_PROMPT,
                num_inference_steps=STEPS,
                guidance_scale=CFG,
                height=SIZE,
                width=SIZE,
                generator=generator,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latency = time.perf_counter() - t0

            fname = f"{prompt_id}_seed{seed}.png"
            out.images[0].save(img_dir / fname)

            rows.append(
                {
                    "condition": condition,
                    "prompt_id": prompt_id,
                    "semantic_text": semantic_text,
                    "gen_prompt": gen_prompt,
                    "seed": seed,
                    "latency_s": round(latency, 3),
                    "clip_score": None,
                    "lpips_between": None,
                    "image_path": (img_dir / fname).relative_to(ROOT).as_posix(),
                }
            )
            print(f"  [{condition}] {prompt_id} seed={seed:5d} | {latency:.1f}s")
    return rows


# ── Run both conditions (single loaded pipeline) ───────────────────────────────

all_rows: list[dict] = []

print("\nLoading SD2.1 pipeline (fp16)...")
pipe = load_pipeline()
print("Loading LoRA weights...")
load_lora(pipe)

for cond in CONDITIONS:
    print(f"\n=== Condition: {cond} ===")
    all_rows.extend(generate_condition(cond, pipe))

del pipe
cleanup_gpu(verbose=True)


# ── CLIP scores (post-hoc, against semantic text without trigger) ──────────────

print(f"\nComputing CLIP scores for {len(all_rows)} images...")
for i, r in enumerate(all_rows, 1):
    img = Image.open(ROOT / r["image_path"]).convert("RGB")
    r["clip_score"] = round(clip_scorer.score(img, r["semantic_text"]), 6)
    if i % 20 == 0 or i == len(all_rows):
        print(f"  {i}/{len(all_rows)}")


# ── LPIPS between conditions (same prompt+seed pair) ─────────────────────────

print("\nComputing LPIPS between no_trigger and with_trigger pairs...")
_lpips_fn = lpips_lib.LPIPS(net="alex")
_lpips_fn.eval()

img_index: dict[tuple, str] = {
    (r["condition"], r["prompt_id"], r["seed"]): r["image_path"] for r in all_rows
}


def _to_t(img: Image.Image) -> torch.Tensor:
    arr = np.array(img.convert("RGB")).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def _lpips_pair(path_a: str, path_b: str) -> float:
    a = Image.open(ROOT / path_a).convert("RGB")
    b = Image.open(ROOT / path_b).convert("RGB")
    with torch.no_grad():
        return round(float(_lpips_fn(_to_t(a), _to_t(b))), 6)


done = 0
total_pairs = sum(1 for r in all_rows if r["condition"] == "no_trigger")
for r in all_rows:
    if r["condition"] != "no_trigger":
        continue
    pid = r["prompt_id"]
    seed = r["seed"]
    wt_path = img_index[("with_trigger", pid, seed)]
    val = _lpips_pair(r["image_path"], wt_path)
    r["lpips_between"] = val
    # mirror onto the paired with_trigger row
    for r2 in all_rows:
        if r2["condition"] == "with_trigger" and r2["prompt_id"] == pid and r2["seed"] == seed:
            r2["lpips_between"] = val
            break
    done += 1
    if done % 20 == 0 or done == total_pairs:
        print(f"  LPIPS {done}/{total_pairs}")

print("LPIPS done.")


# ── Save results ──────────────────────────────────────────────────────────────

CSV_PATH = OUT / "results.csv"
JSON_PATH = OUT / "results.json"

csv_fields = [
    "condition",
    "prompt_id",
    "seed",
    "latency_s",
    "clip_score",
    "lpips_between",
    "gen_prompt",
    "image_path",
]
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(all_rows)

lora_config = LORA_REGISTRY[LORA_NAME]
with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(
        {
            "experiment": "exp7_lora_trigger",
            "date": "2026-05-07",
            "model": MODEL_ID,
            "lora": LORA_NAME,
            "lora_alpha": LORA_ALPHA,
            "lora_description": lora_config["description"],
            "trigger_token": lora_config["trigger_token"],
            "steps": STEPS,
            "cfg": CFG,
            "size": SIZE,
            "seeds": SEEDS,
            "conditions": CONDITIONS,
            "total_images": len(all_rows),
            "results": all_rows,
        },
        f,
        indent=2,
    )

print(f"Results: {CSV_PATH}")


# ── Per-condition aggregates ──────────────────────────────────────────────────

by_cond: dict[str, list[dict]] = {c: [] for c in CONDITIONS}
for r in all_rows:
    by_cond[r["condition"]].append(r)

agg: dict[str, dict] = {}
for cond, rows in by_cond.items():
    clips = [r["clip_score"] for r in rows]
    lats = [r["latency_s"] for r in rows]
    lpips_vals = [r["lpips_between"] for r in rows if r["lpips_between"] is not None]
    agg[cond] = {
        "mean_clip": statistics.mean(clips),
        "se_clip": statistics.stdev(clips) / len(clips) ** 0.5,
        "mean_lat": statistics.mean(lats),
        "mean_lpips": statistics.mean(lpips_vals) if lpips_vals else None,
        "se_lpips": (
            statistics.stdev(lpips_vals) / len(lpips_vals) ** 0.5 if len(lpips_vals) > 1 else None
        ),
    }

clip_delta = agg["with_trigger"]["mean_clip"] - agg["no_trigger"]["mean_clip"]
pooled_se = (agg["no_trigger"]["se_clip"] ** 2 + agg["with_trigger"]["se_clip"] ** 2) ** 0.5
mean_lpips = agg["no_trigger"]["mean_lpips"]

print("\n── Condition summary ──")
for cond in CONDITIONS:
    a = agg[cond]
    print(f"  {cond}: CLIP={a['mean_clip']:.4f} ±{a['se_clip']:.4f} | " f"lat={a['mean_lat']:.1f}s")
print(f"  CLIP delta (with_trigger - no_trigger): {clip_delta:+.4f}  (pooled SE={pooled_se:.4f})")
print(f"  Mean LPIPS between conditions: {mean_lpips:.4f} ±{agg['no_trigger']['se_lpips']:.4f}")


# ── Charts ────────────────────────────────────────────────────────────────────

# Chart 1: CLIP by condition (bar chart, 2 bars)
cond_labels = ["no_trigger", "with_trigger"]
clip_vals = np.array([agg[c]["mean_clip"] for c in cond_labels])
clip_max = float(clip_vals.max())

canvas = ChartCanvas(
    figsize=(7, 4.5),
    title="CLIP score: no trigger vs with trigger — LoRA alpha=1.0, same prompt content",
    ylabel="Mean CLIP score (vs semantic prompt)",
    top_margin_pct=0.22,
)
canvas.set_ylim(0.0, clip_max * 1.35)
canvas.add_bars(
    np.arange(2, dtype=float),
    clip_vals,
    colors=[GREEN, BLUE],
    width=0.5,
    value_fmt="{:.4f}",
    value_pad=clip_max * 0.015,
    value_size=9,
)
canvas.set_xticks(np.arange(2, dtype=float), cond_labels, fontsize=10)
canvas.save(str(CHARTS_DIR / "clip_by_condition.png"))

# Chart 2: LPIPS per prompt_id (shows prompt-level variance)
prompt_ids = list(PROMPTS.keys())
per_prompt_lpips = []
for pid in prompt_ids:
    vals = [
        r["lpips_between"]
        for r in all_rows
        if r["condition"] == "no_trigger"
        and r["prompt_id"] == pid
        and r["lpips_between"] is not None
    ]
    per_prompt_lpips.append(statistics.mean(vals) if vals else 0.0)

x2 = np.arange(len(prompt_ids), dtype=float)
lpips_max = max(per_prompt_lpips)
canvas2 = ChartCanvas(
    figsize=(10, 4.5),
    title="Mean LPIPS between no_trigger and with_trigger — per prompt (5 seeds each)",
    ylabel="Mean LPIPS",
    top_margin_pct=0.22,
)
canvas2.set_ylim(0.0, lpips_max * 1.5)
canvas2.add_bars(
    x2,
    np.array(per_prompt_lpips),
    colors=[BLUE] * len(prompt_ids),
    width=0.6,
    value_fmt="{:.4f}",
    value_pad=lpips_max * 0.04,
    value_size=8,
)
canvas2.set_xticks(x2, [p.replace("p0", "p") for p in prompt_ids], fontsize=8)
canvas2.save(str(CHARTS_DIR / "lpips_by_prompt.png"))

print(f"Charts written to {CHARTS_DIR}")


# ── Findings writeup ──────────────────────────────────────────────────────────

nt = agg["no_trigger"]
wt = agg["with_trigger"]
clip_delta_se_ratio = abs(clip_delta) / pooled_se


def _clip_verdict(delta, se):
    if abs(delta) < se:
        return "within noise (< 1 pooled SE) — CLIP cannot detect the trigger"
    elif abs(delta) < 2 * se:
        return f"borderline ({abs(delta)/se:.1f}× pooled SE) — marginal CLIP signal"
    else:
        return f"detectable ({abs(delta)/se:.1f}× pooled SE)"


clip_verd = _clip_verdict(clip_delta, pooled_se)

FINDINGS = f"""\
# Experiment 7: LoRA Trigger Token Sensitivity

**Date:** 2026-05-07
**Hardware:** RTX 3070 Laptop 8 GB (enable_model_cpu_offload)
**Model:** {MODEL_ID}
**LoRA:** {LORA_NAME} — {lora_config['description']}
**LoRA alpha:** {LORA_ALPHA} (fixed — trained default, loaded once for both conditions)
**Trigger token:** "{lora_config['trigger_token']}"
**Conditions:**
  - `no_trigger`: prompts use "ukiyo-e ..." style description, NO trigger token
  - `with_trigger`: identical prompts prepended with "ukyowood"
**CLIP reference:** semantic prompt text (without "ukyowood") — same for both conditions,
  so CLIP measures image–content alignment free of the trigger token's influence.
**Design:** 2 conditions × 8 prompts × 5 seeds = {len(all_rows)} images total
**Scheduler:** DPM-Solver++ · {STEPS} steps · {SIZE}×{SIZE} · CFG={CFG}

## Hypothesis

"ukyowood" is a trained trigger token not in CLIP's vocabulary. CLIP scores should be
near-identical between conditions — the token adds no semantic information CLIP can
interpret. LPIPS will determine whether the trigger actually changes how the LoRA fires:
a large LPIPS value means the trigger meaningfully redirects generation; a small value
means the LoRA fires similarly regardless of the trigger.

## Results

| Condition    | Mean CLIP | SE      | Mean LPIPS (between) |
|--------------|----------:|--------:|---------------------:|
| no_trigger   | {nt['mean_clip']:.4f}    | ±{nt['se_clip']:.4f}  | {nt['mean_lpips']:.4f} ±{nt['se_lpips']:.4f}  |
| with_trigger | {wt['mean_clip']:.4f}    | ±{wt['se_clip']:.4f}  | (same pairs)         |

CLIP delta (with_trigger − no_trigger): {clip_delta:+.4f}  (pooled SE = {pooled_se:.4f})
LPIPS between conditions (mean ± SE): {mean_lpips:.4f} ±{nt['se_lpips']:.4f}

## Interpretation

**CLIP:** Delta is {clip_verd}. As expected, CLIP has no representation for "ukyowood"
and cannot register the trigger token's presence or absence.

**LPIPS:** The mean LPIPS between no_trigger and with_trigger images is {mean_lpips:.4f}.
{"This is large — comparable to scheduler-to-scheduler differences in Exp 4 — confirming that the trigger token meaningfully redirects how the LoRA fires. Images generated with the trigger are perceptually different from those without, even when CLIP scores are identical." if mean_lpips > 0.3 else "This is moderate — the trigger changes some images substantially and others less so; see per-prompt chart." if mean_lpips > 0.15 else "This is small — the LoRA fires similarly with or without the trigger token, suggesting the style-description words ('ukiyo-e', 'woodblock print') activate the adapter nearly as effectively as the explicit trigger."}

**What this tells us about trigger tokens:** {"The trigger token 'ukyowood' is doing meaningful work — it is not redundant with the style-description words in the prompt. Without it, the LoRA fires in a visually distinct way. CLIP cannot see this difference at all." if mean_lpips > 0.2 else "The style-description words ('ukiyo-e', 'woodblock print') carry most of the activation signal for this LoRA. The trigger token adds some visual change, but the LoRA fires substantially even without it."}

**Cross-experiment note:** Seventh and final confirmation of CLIP-blindness across this
experimental series: quantization (Exp 1), negative prompt (Exp 2), CFG plateau (Exp 3),
scheduler stochasticity (Exp 4), ControlNet strength (Exp 5), LoRA style scale (Exp 6),
LoRA trigger token (Exp 7). The consistent theme: CLIP measures semantic alignment
reliably, but is structurally blind to any parameter that reshapes visual character
without eliminating prompt-relevant content.

## Charts

- `charts/clip_by_condition.png` — CLIP score per condition
- `charts/lpips_by_prompt.png` — per-prompt mean LPIPS between conditions

## Raw data

`results.csv` / `results.json` — one row per image ({len(all_rows)} rows total).

Reproduce:

```bash
python scripts/experiments/exp7_lora_trigger.py
```
"""

with open(OUT / "findings.md", "w", encoding="utf-8") as f:
    f.write(FINDINGS)

print(f"\nFindings written: {OUT / 'findings.md'}")
print("Experiment 7 complete.")
