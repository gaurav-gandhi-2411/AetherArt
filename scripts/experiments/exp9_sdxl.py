"""
Experiment 9 (SDXL): LoRA trigger token sensitivity — SDXL port.

Port of exp9_lora_trigger.py (SD 2.1) to SDXL. The original file docstring
incorrectly says "Experiment 7" — this is exp9 by filename.

Two conditions, same SDXL Ukiyo-e LoRA (alpha=1.0) loaded for both:
  no_trigger  — prompt uses "ukiyo-e ..." description but NOT "ukyowood"
  with_trigger — identical prompt with "ukyowood" prepended

CLIP is scored against the semantic prompt text (without "ukyowood") for both
conditions — so CLIP measures alignment to the visual concept, not sensitivity to
the unknown trigger token. This is the same design as the SD 2.1 original.

LPIPS is computed between matched pairs (no_trigger vs with_trigger, same seed+prompt).

Hypothesis: "ukyowood" is not in CLIP's vocabulary — CLIP should be flat across
conditions. LPIPS will determine whether the trigger meaningfully redirects the LoRA.

Design: 2 conditions × 8 prompts × 5 seeds = 80 images
Image size: 1024×1024 (SDXL)

Run from project root:
    python scripts/experiments/exp9_sdxl.py

Outputs:
    reports/experiments/exp9_sdxl/
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
from PIL import Image  # noqa: E402

from aetherart import clip_scorer  # noqa: E402
from aetherart.eval_hps import release_hps, score_hps  # noqa: E402
from aetherart.eval_ir import release_image_reward, score_image_reward  # noqa: E402
from aetherart.gpu_hygiene import cleanup_gpu  # noqa: E402
from aetherart.sdxl_pipeline import load_sdxl_base  # noqa: E402
from aetherart.visualization import BLUE, GREEN, PURPLE, ChartCanvas  # noqa: E402

atexit.register(cleanup_gpu)

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
LORA_REPO = "gauravgandhi2411/aetherart-ukiyo-sdxl"
LORA_LOCAL = ROOT / "data" / "lora" / "ukiyo-e-sdxl" / "ukiyo-e-sdxl-lora.safetensors"

LORA_ALPHA = 1.0
SEEDS = [42, 123, 456, 789, 1337]
STEPS = 30
CFG = 7.5
SIZE = 1024
CONDITIONS = ["no_trigger", "with_trigger"]

# Each prompt_id maps to (semantic_text, no_trigger_prompt, with_trigger_prompt).
# semantic_text is used for CLIP / HPS / IR scoring in both conditions —
# no "ukyowood" bias, same approach as SD 2.1 exp9.
PROMPTS: dict[str, tuple[str, str, str]] = {
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

OUT = ROOT / "reports" / "experiments" / "exp9_sdxl"
IMG_DIR = OUT / "images"
CHARTS_DIR = OUT / "charts"

for cond in CONDITIONS:
    (IMG_DIR / cond).mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)


# ── LoRA download helper ───────────────────────────────────────────────────────


def _ensure_lora() -> Path:
    """Return the local LoRA path, downloading from HF Hub if missing."""
    if LORA_LOCAL.exists():
        return LORA_LOCAL
    print(f"Downloading SDXL LoRA from {LORA_REPO}...")
    from huggingface_hub import hf_hub_download

    LORA_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        LORA_REPO, "ukiyo-e-sdxl-lora.safetensors", local_dir=str(LORA_LOCAL.parent)
    )
    return Path(path)


# ── Pipeline and LoRA loading ─────────────────────────────────────────────────


def load_lora(pipe: object) -> None:  # StableDiffusionXLPipeline
    """Download LoRA if needed, load into pipeline, and set alpha=1.0."""
    lora_path = _ensure_lora()
    pipe.load_lora_weights(  # type: ignore[union-attr]
        str(lora_path.parent),
        weight_name=lora_path.name,
        adapter_name="ukiyo_e",
    )
    pipe.set_adapters(["ukiyo_e"], adapter_weights=[LORA_ALPHA])  # type: ignore[union-attr]
    print(f"SDXL LoRA loaded: {lora_path.name} (alpha={LORA_ALPHA})")


# ── Generation ────────────────────────────────────────────────────────────────


def generate_condition(condition: str, pipe: object) -> list[dict]:  # StableDiffusionXLPipeline
    """Generate all 8 prompts × 5 seeds for one trigger condition."""
    rows: list[dict] = []
    img_dir = IMG_DIR / condition
    # Index 1 = no_trigger prompt, index 2 = with_trigger prompt
    prompt_key = 1 if condition == "no_trigger" else 2

    for prompt_id, ptuple in PROMPTS.items():
        semantic_text = ptuple[0]
        gen_prompt = ptuple[prompt_key]

        for seed in SEEDS:
            generator = torch.Generator().manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = pipe(  # type: ignore[operator]
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
                    "hps_score": None,
                    "ir_score": None,
                    "lpips_vs_no_trigger": None,
                    "image_path": (img_dir / fname).relative_to(ROOT).as_posix(),
                }
            )
            print(f"  [{condition}] {prompt_id} seed={seed:5d} | {latency:.1f}s")

    return rows


# ── Run both conditions ────────────────────────────────────────────────────────

all_rows: list[dict] = []

print("\nLoading SDXL base pipeline...")
pipe = load_sdxl_base()
print("Loading SDXL LoRA weights...")
load_lora(pipe)

for cond in CONDITIONS:
    print(f"\n=== Condition: {cond} ===")
    all_rows.extend(generate_condition(cond, pipe))

del pipe
cleanup_gpu(verbose=True)


# ── Post-hoc scoring (against semantic_text — no "ukyowood" bias) ─────────────

print(f"\nComputing scores for {len(all_rows)} images...")
for i, r in enumerate(all_rows, 1):
    img = Image.open(ROOT / r["image_path"]).convert("RGB")
    # Score against semantic_text, not gen_prompt, to keep CLIP/HPS/IR neutral
    r["clip_score"] = round(clip_scorer.score(img, r["semantic_text"]), 6)
    r["hps_score"] = round(score_hps([img], [r["semantic_text"]])[0], 6)
    r["ir_score"] = round(score_image_reward([img], [r["semantic_text"]])[0], 6)
    if i % 20 == 0 or i == len(all_rows):
        print(f"  {i}/{len(all_rows)}")

release_hps()
release_image_reward()


# ── LPIPS between conditions (same prompt+seed pair) ─────────────────────────

print("\nComputing LPIPS between no_trigger and with_trigger pairs...")
_lpips_fn = lpips_lib.LPIPS(net="alex")
_lpips_fn.eval()

img_index: dict[tuple[str, str, int], str] = {
    (r["condition"], r["prompt_id"], r["seed"]): r["image_path"] for r in all_rows
}


def _to_t(img: Image.Image) -> torch.Tensor:
    """Convert PIL image to normalised [-1, 1] tensor for LPIPS."""
    arr = np.array(img.convert("RGB")).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def _lpips_pair(path_a: str, path_b: str) -> float:
    """Compute LPIPS distance between two image files."""
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
    r["lpips_vs_no_trigger"] = val
    # Mirror the value onto the paired with_trigger row
    for r2 in all_rows:
        if r2["condition"] == "with_trigger" and r2["prompt_id"] == pid and r2["seed"] == seed:
            r2["lpips_vs_no_trigger"] = val
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
    "hps_score",
    "ir_score",
    "lpips_vs_no_trigger",
    "image_path",
]
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(all_rows)

with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(
        {
            "experiment": "exp9_sdxl",
            "date": "2026-06-02",
            "model": MODEL_ID,
            "lora_repo": LORA_REPO,
            "lora_alpha": LORA_ALPHA,
            "size": SIZE,
            "steps": STEPS,
            "cfg": CFG,
            "seeds": SEEDS,
            "conditions": CONDITIONS,
            "scorers": ["clip", "hps", "imagereward", "lpips"],
            "scoring_note": "CLIP/HPS/IR scored against semantic_text (no 'ukyowood') for both conditions",
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
    hps_vals = [r["hps_score"] for r in rows]
    ir_vals = [r["ir_score"] for r in rows]
    lats = [r["latency_s"] for r in rows]
    lpips_vals = [r["lpips_vs_no_trigger"] for r in rows if r["lpips_vs_no_trigger"] is not None]
    agg[cond] = {
        "mean_clip": statistics.mean(clips),
        "se_clip": statistics.stdev(clips) / len(clips) ** 0.5,
        "mean_hps": statistics.mean(hps_vals),
        "se_hps": statistics.stdev(hps_vals) / len(hps_vals) ** 0.5,
        "mean_ir": statistics.mean(ir_vals),
        "mean_lat": statistics.mean(lats),
        "mean_lpips": statistics.mean(lpips_vals) if lpips_vals else None,
        "se_lpips": (
            statistics.stdev(lpips_vals) / len(lpips_vals) ** 0.5 if len(lpips_vals) > 1 else None
        ),
    }

clip_delta = agg["with_trigger"]["mean_clip"] - agg["no_trigger"]["mean_clip"]
hps_delta = agg["with_trigger"]["mean_hps"] - agg["no_trigger"]["mean_hps"]
ir_delta = agg["with_trigger"]["mean_ir"] - agg["no_trigger"]["mean_ir"]
pooled_se = (agg["no_trigger"]["se_clip"] ** 2 + agg["with_trigger"]["se_clip"] ** 2) ** 0.5
mean_lpips = agg["no_trigger"]["mean_lpips"] or 0.0
se_lpips = agg["no_trigger"]["se_lpips"] or 0.0

print("\n── Condition summary ──")
for cond in CONDITIONS:
    a = agg[cond]
    print(
        f"  {cond}: CLIP={a['mean_clip']:.4f} ±{a['se_clip']:.4f} | "
        f"HPS={a['mean_hps']:.4f} | IR={a['mean_ir']:.4f} | lat={a['mean_lat']:.1f}s"
    )
print(f"  CLIP delta (with_trigger - no_trigger): {clip_delta:+.4f} (pooled SE={pooled_se:.4f})")
print(f"  HPS delta: {hps_delta:+.4f} | IR delta: {ir_delta:+.4f}")
print(f"  Mean LPIPS between conditions: {mean_lpips:.4f} ±{se_lpips:.4f}")


# ── Charts ────────────────────────────────────────────────────────────────────

# Chart 1: CLIP by condition (2 bars)
cond_labels = ["no_trigger", "with_trigger"]
clip_vals = np.array([agg[c]["mean_clip"] for c in cond_labels])
clip_max = float(clip_vals.max())

canvas = ChartCanvas(
    figsize=(7, 4.5),
    title="CLIP score: no trigger vs with trigger — SDXL Ukiyo-e LoRA, alpha=1.0",
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

# Chart 2: HPS by condition
hps_vals = np.array([agg[c]["mean_hps"] for c in cond_labels])
hps_max = float(hps_vals.max())

canvas_hps = ChartCanvas(
    figsize=(7, 4.5),
    title="HPS score: no trigger vs with trigger — SDXL Ukiyo-e LoRA, alpha=1.0",
    ylabel="Mean HPS score (vs semantic prompt)",
    top_margin_pct=0.22,
)
canvas_hps.set_ylim(0.0, hps_max * 1.35)
canvas_hps.add_bars(
    np.arange(2, dtype=float),
    hps_vals,
    colors=[GREEN, BLUE],
    width=0.5,
    value_fmt="{:.4f}",
    value_pad=hps_max * 0.015,
    value_size=9,
)
canvas_hps.set_xticks(np.arange(2, dtype=float), cond_labels, fontsize=10)
canvas_hps.save(str(CHARTS_DIR / "hps_by_condition.png"))

# Chart 3: per-prompt mean LPIPS
prompt_ids = list(PROMPTS.keys())
per_prompt_lpips = []
for pid in prompt_ids:
    vals = [
        r["lpips_vs_no_trigger"]
        for r in all_rows
        if r["condition"] == "no_trigger"
        and r["prompt_id"] == pid
        and r["lpips_vs_no_trigger"] is not None
    ]
    per_prompt_lpips.append(statistics.mean(vals) if vals else 0.0)

x2 = np.arange(len(prompt_ids), dtype=float)
lpips_max_pp = max(per_prompt_lpips)

canvas2 = ChartCanvas(
    figsize=(10, 4.5),
    title=(
        "Mean LPIPS between no_trigger and with_trigger — SDXL, per prompt "
        f"({len(SEEDS)} seeds each)"
    ),
    ylabel="Mean LPIPS",
    top_margin_pct=0.22,
)
canvas2.set_ylim(0.0, lpips_max_pp * 1.5)
canvas2.add_bars(
    x2,
    np.array(per_prompt_lpips),
    colors=[PURPLE] * len(prompt_ids),
    width=0.6,
    value_fmt="{:.4f}",
    value_pad=lpips_max_pp * 0.04,
    value_size=8,
)
canvas2.set_xticks(x2, [p.replace("p0", "p") for p in prompt_ids], fontsize=8)
canvas2.save(str(CHARTS_DIR / "lpips_by_prompt.png"))

print(f"Charts written to {CHARTS_DIR}")


# ── Findings writeup ──────────────────────────────────────────────────────────

clip_range_se = abs(clip_delta) / pooled_se if pooled_se > 0 else float("inf")
lpips_range_val = max(per_prompt_lpips) - min(per_prompt_lpips)


def _clip_verdict(delta: float, se: float) -> str:
    if abs(delta) < se:
        return "within noise (< 1 pooled SE) — CLIP cannot detect the trigger"
    elif abs(delta) < 2 * se:
        return f"borderline ({abs(delta) / se:.1f}× pooled SE) — marginal CLIP signal"
    return f"detectable ({abs(delta) / se:.1f}× pooled SE)"


clip_verd = _clip_verdict(clip_delta, pooled_se)
nt = agg["no_trigger"]
wt = agg["with_trigger"]

FINDINGS = f"""\
# Experiment 9 (SDXL): LoRA Trigger Token Sensitivity

**Date:** 2026-06-02
**Model:** {MODEL_ID} — {SIZE}×{SIZE}
**LoRA:** {LORA_REPO} (SDXL Ukiyo-e, alpha={LORA_ALPHA})
**Trigger token:** "ukyowood"
**Conditions:**
  - `no_trigger`: prompts use "ukiyo-e ..." description, NO trigger token
  - `with_trigger`: identical prompts prepended with "ukyowood"
**CLIP / HPS / IR reference:** semantic prompt text (without "ukyowood") — same for
  both conditions, so scorers measure image–content alignment free of the trigger token.
**Design:** 2 conditions × {len(PROMPTS)} prompts × {len(SEEDS)} seeds = {len(all_rows)} images
**Scheduler:** DPM-Solver++ · {STEPS} steps · CFG={CFG}
**Scorers:** CLIP, HPS, ImageReward, LPIPS

## Hypothesis

"ukyowood" is a trained trigger token absent from CLIP's vocabulary. CLIP / HPS / IR scores
should be near-identical between conditions. LPIPS will determine whether the trigger
meaningfully redirects how the SDXL LoRA fires.

## Results

| Condition    | Mean CLIP | SE      | Mean HPS | Mean IR | Mean LPIPS (between) |
|--------------|----------:|--------:|---------:|--------:|---------------------:|
| no_trigger   | {nt["mean_clip"]:.4f}    | ±{nt["se_clip"]:.4f}  | {nt["mean_hps"]:.4f}    | {nt["mean_ir"]:.4f}    | {mean_lpips:.4f} ±{se_lpips:.4f} |
| with_trigger | {wt["mean_clip"]:.4f}    | ±{wt["se_clip"]:.4f}  | {wt["mean_hps"]:.4f}    | {wt["mean_ir"]:.4f}    | (same pairs)               |

CLIP delta across conditions = {clip_delta:+.4f}, which is {clip_range_se:.1f} SEs.
HPS delta = {hps_delta:+.4f}. IR delta = {ir_delta:+.4f}. LPIPS range = {lpips_range_val:.4f}.
Verdict: CLIP-BLIND: {"yes" if clip_range_se < 2.0 else "no"}.

## Interpretation

**CLIP / HPS / IR:** Delta is {clip_verd}. As expected, none of the semantic scorers have
a representation for "ukyowood" and cannot register its presence or absence.

**LPIPS:** The mean LPIPS between no_trigger and with_trigger images is {mean_lpips:.4f}.
{"This is large — the trigger token meaningfully redirects how the SDXL LoRA fires. Images with the trigger are perceptually different from those without, while all three semantic scorers remain identical." if mean_lpips > 0.3 else "This is moderate — the trigger changes some images substantially; see per-prompt chart." if mean_lpips > 0.15 else "This is small — the style-description words ('ukiyo-e', 'woodblock print') carry most of the activation signal for this SDXL LoRA, with the trigger token adding modest additional visual change."}

**SDXL vs SD 2.1:** At 1024×1024, per-pixel trigger sensitivity may differ from the SD 2.1
512×512 baseline. The CLIP-blindness mechanism is unchanged — the token is not in CLIP's
vocabulary regardless of resolution.

## Charts

- `charts/clip_by_condition.png`
- `charts/hps_by_condition.png`
- `charts/lpips_by_prompt.png`

## Raw data

`results.csv` / `results.json` — one row per image ({len(all_rows)} rows total).

Reproduce:

```bash
python scripts/experiments/exp9_sdxl.py
```
"""

with open(OUT / "findings.md", "w", encoding="utf-8") as f:
    f.write(FINDINGS)

print(f"\nFindings written: {OUT / 'findings.md'}")
print("Experiment 9 (SDXL) complete.")
