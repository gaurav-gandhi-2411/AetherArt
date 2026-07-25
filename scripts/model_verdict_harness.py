#!/usr/bin/env python
"""Cross-family model verdict harness.

Scores every commercially-usable AetherArt model family on the SAME 30-prompt set
(scripts/eval_prompts.yaml), seed-controlled (SEEDS below), n=3 seeds/prompt. Excludes
SDXL Turbo (non-commercial ADD license, legacy-gated — not a product candidate).

Families and their metrics:
  - sd21_base, sdxl_base, hyper_4step, hyper_8step, sdxl_controlnet_union:
        CLIP + HPS (ImageReward excluded — confirmed broken on this environment,
        ImportError: apply_chunking_to_forward removed from current transformers;
        not fabricated, not worked around here).
  - ukiyo_e_lora_sdxl (the published LoRA):
        NOT scored on CLIP as a verdict metric. The model's own HF card documents
        CLIP-blindness for this exact style-transfer task (CLIP delta <1 SE while LPIPS
        ranged 0.40-0.73 across 9 controlled experiments; smaller/underfit adapters score
        HIGHER on CLIP, the wrong direction for a quality signal). Scored instead on:
          (a) LPIPS distance from the matched sdxl_base output (same prompt+seed, no LoRA)
          (b) a local Ollama vision-language judge (qwen2.5vl:7b, zero cost, no paid API)
              scoring style_adherence / figure_preservation / artifact_absence on a 0-1
              rubric, via THREE INDEPENDENT single-axis calls per image (one axis per call,
              no other axis named or in context) — see SINGLE_AXIS_JUDGE_PROMPTS below. A
              prior single-call multi-axis design (all three axes requested in one call) was
              replaced after a halo-effect check (docs/MODEL_VERDICT.md SS4.5) found it
              produces correlated ratings that don't hold up under independent scoring.
              CLIP is still recorded for transparency/context but is explicitly NOT used to
              render a verdict for this family.

sdxl_controlnet_union and ukiyo_e_lora_sdxl both reuse sdxl_base's own saved output images
(same prompt+seed) as their conditioning/reference source rather than double-generating —
run sdxl_base FIRST.

Usage:
    python scripts/model_verdict_harness.py --family sdxl_base
    python scripts/model_verdict_harness.py --family ukiyo_e_lora_sdxl --resume
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from aetherart.logger import get_logger

logger = get_logger(__name__)

ROOT = Path(__file__).parent.parent
PROMPTS_YAML = ROOT / "scripts" / "eval_prompts.yaml"
REPORTS_DIR = ROOT / "reports"
OUT_DIR = ROOT / "outputs" / "verdict"

SEEDS = [42, 43, 44]
GUIDANCE_DEFAULT = 7.5

OLLAMA_URL = "http://localhost:11434/api/generate"
VLM_MODEL = "qwen2.5vl:7b"

# Independent single-axis judge prompts — one Ollama call per axis, no other axis named or
# implied in that call's prompt or context. This replaced a single-call multi-axis design
# (one call asking for all three scores at once) after a halo-effect check
# (docs/MODEL_VERDICT.md SS4.5) found that design produces correlated ratings that do not hold
# up under independent scoring — e.g. a genuine artifact_absence improvement partly "bled into"
# the judge's style_adherence/figure_preservation scores. Independent-axis scoring is now this
# harness's default, not an opt-in variant — see tests/test_model_verdict_harness.py for the
# regression test asserting one Ollama call per axis.
#
# "style_adherence" is deliberately NOT a fixed template here. A style question is inherently
# domain-specific (asking "does this look like Ukiyo-e" is meaningless for a Pattachitra image),
# and this harness previously hardcoded the Ukiyo-e question for every caller — a bug found only
# by direct code inspection, after it had already silently produced 360 wrong-question
# Pattachitra scores (docs/MODEL_VERDICT.md SS7's judge positive-control finding). The harness
# holds no domain knowledge: every caller of score_vlm_judge must pass its own `style_question`
# explicitly (no default), so a missing style fails loudly (TypeError) instead of silently
# reusing whatever domain happened to be hardcoded last.
SINGLE_AXIS_JUDGE_PROMPTS: dict[str, str] = {
    "figure_preservation": """You are judging a single image against one question only.
Question: are the subjects/figures in this image anatomically coherent and recognizable (not
melted, distorted, or missing)? The image was generated from this prompt: "{prompt}"
Respond with ONLY a JSON object: {{"figure_preservation": <float 0.0-1.0>}}""",
    "artifact_absence": """You are judging a single image against one question only.
Question: is this image FREE of embedded text, watermarks, signatures, cartouches, or seal/
script marks (1.0 = totally clean, 0.0 = heavily covered in text artifacts)?
Respond with ONLY a JSON object: {{"artifact_absence": <float 0.0-1.0>}}""",
}

VLM_JUDGE_AXES: tuple[str, ...] = ("style_adherence", "figure_preservation", "artifact_absence")


def load_prompts() -> list[dict]:
    """The canonical 30-prompt PartiPrompts benchmark (pp_001..pp_030) used throughout this
    project's other eval reports. scripts/eval_prompts.yaml also carries 4 extra lora_*
    entries added later for a different purpose — explicitly excluded here so every family
    is scored on the exact same 30-prompt set, per the task requirement."""
    with PROMPTS_YAML.open(encoding="utf-8") as f:
        all_prompts = yaml.safe_load(f)["prompts"]
    prompts = [p for p in all_prompts if p["id"].startswith("pp_")]
    assert len(prompts) == 30, f"expected 30 pp_* prompts, got {len(prompts)}"
    return prompts


def save_partial(path: Path, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def load_partial(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def done_keys(results: list[dict]) -> set[str]:
    return {f"{r['prompt_id']}_{r['seed']}" for r in results if not r.get("error")}


# ── Scorers ───────────────────────────────────────────────────────────────


def score_clip_one(img: Image.Image, prompt: str) -> float:
    from aetherart.clip_scorer import score

    return float(score(img, prompt))


def score_hps_one(img: Image.Image, prompt: str) -> float | None:
    try:
        from aetherart.eval_hps import score_hps

        return float(score_hps([img], [prompt])[0])
    except Exception as e:
        logger.warning("HPS scoring failed: %s", e)
        return None


_lpips_fn = None


def score_lpips_pair(img_a: Image.Image, img_b: Image.Image) -> float:
    global _lpips_fn
    import lpips as _lpips_lib
    import torchvision.transforms as T

    if _lpips_fn is None:
        _lpips_fn = _lpips_lib.LPIPS(net="alex", verbose=False)
        if torch.cuda.is_available():
            _lpips_fn = _lpips_fn.cuda()
    to_tensor = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
    ta = to_tensor(img_a.convert("RGB")).unsqueeze(0)
    tb = to_tensor(img_b.convert("RGB")).unsqueeze(0)
    if torch.cuda.is_available():
        ta, tb = ta.cuda(), tb.cuda()
    with torch.no_grad():
        return float(_lpips_fn(ta, tb).item())


def _ollama_generate_json(prompt_text: str, b64_image: str) -> dict:
    """One Ollama call, no axis-crossing: the request body carries exactly one prompt and one
    image. Split out so it's independently mockable/countable in tests — see
    tests/test_model_verdict_harness.py's assertion that scoring one image calls this once
    per axis (3 total), never once for all axes."""
    import requests

    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": VLM_MODEL,
            "prompt": prompt_text,
            "images": [b64_image],
            "stream": False,
            "format": "json",
            # Ollama's default served context window for this model can be as small as 4096
            # tokens - too small for a longer judge prompt plus a high-resolution image's token
            # count (root-caused via a real "exceeds the available context size" 400 error from
            # a Pattachitra-corpus curation run using a similarly-shaped prompt, not assumed).
            # qwen2.5vl:7b supports up to 128k context; 8192 is ample headroom here.
            "options": {"num_ctx": 8192},
        },
        timeout=180,
    )
    resp.raise_for_status()
    return json.loads(resp.json()["response"])


def score_vlm_judge(img: Image.Image, prompt: str, *, style_question: str) -> dict | None:
    """Independent single-axis scoring: one Ollama call per axis (3 total per image), each call
    seeing only that axis's question — no other axis is named or implied. Returns the same
    {"style_adherence": ..., "figure_preservation": ..., "artifact_absence": ...} shape the
    prior single-call design returned, so callers/downstream JSON consumers are unaffected.
    Returns None (whole record) if any one axis call fails, matching prior all-or-nothing
    failure semantics.

    style_question: the complete, self-contained style_adherence question for the caller's own
    domain (e.g. "does this image look like an authentic Ukiyo-e (Japanese woodblock print) -
    flat color planes, characteristic line work, traditional palette?"). Required, no default —
    this harness holds no domain knowledge of its own (see SINGLE_AXIS_JUDGE_PROMPTS's comment
    for why a hardcoded default previously produced 360 wrong-question Pattachitra scores).
    """
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    scores: dict[str, float] = {}
    for axis in VLM_JUDGE_AXES:
        if axis == "style_adherence":
            prompt_text = (
                "You are judging a single image against one question only.\n"
                f"Question: {style_question}\n"
                'Respond with ONLY a JSON object: {"style_adherence": <float 0.0-1.0>}'
            )
        else:
            prompt_text = SINGLE_AXIS_JUDGE_PROMPTS[axis].format(prompt=prompt)
        try:
            result = _ollama_generate_json(prompt_text, b64)
            scores[axis] = float(result[axis])
        except Exception as e:
            logger.warning("VLM judge call failed (axis=%s): %s", axis, e)
            return None
    return validate_judge_scores(scores)


# ── Harness integrity self-checks ────────────────────────────────────────
# Three silent measurement bugs have surfaced across separate verdict runs (num_ctx judge
# failures, a phantom VRAM counter, and a mid-run CUDA "illegal memory access" that poisoned
# 68/90 records in one checkpoint's first attempt). Each was caught by manual, one-off
# investigation after the fact. These checks make the same failure classes fail loudly, inline,
# instead of silently writing poisoned records that a later analysis has to reconstruct.


class DegenerateImageError(RuntimeError):
    """Raised when a generated image matches a known corruption failure signature. Callers must
    NOT catch this alongside ordinary generation errors — it aborts the run rather than being
    recorded as a per-item error, because a corrupted CUDA context can poison every subsequent
    generation in the same process, not just the one that happened to raise."""


def check_cuda_health() -> None:
    """Pre-flight probe: force a real CUDA kernel launch + readback and verify the result is
    numerically correct before spending minutes loading a multi-GB pipeline onto a context that
    may already be poisoned (e.g. left over from a crashed prior process sharing the same GPU).
    No-ops on CPU-only environments. Raises RuntimeError on any failure or wrong result."""
    if not torch.cuda.is_available():
        return
    try:
        probe = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        result = (probe * 2.0).sum().item()
        torch.cuda.synchronize()
    except RuntimeError as e:
        raise RuntimeError(
            f"CUDA context health probe failed - GPU/driver context is unusable: {e}"
        ) from e
    if math.isnan(result) or not math.isclose(result, 12.0):
        raise RuntimeError(
            f"CUDA context health probe returned a corrupted result ({result}, expected 12.0) - "
            "refusing to generate on a poisoned context"
        )


def detect_degenerate_image(
    img: Image.Image, *, black_white_threshold: float = 0.5, min_std: float = 2.0
) -> list[str]:
    """Return a list of degeneracy issue names; empty means the image shows no known corruption
    signature. This is NOT a style/quality judgment - it only catches the failure modes a
    poisoned CUDA context or a broken generation call actually produces (NaN/Inf pixels, solid-
    color collapse, near-uniform noise-free output), the same signature figure-dropout-from-
    corruption would leave behind."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float64)
    issues = []
    if not np.isfinite(arr).all():
        issues.append("non_finite_pixels")
    pct_black = float((arr.sum(axis=-1) == 0).mean())
    pct_white = float((arr.sum(axis=-1) == 765).mean())
    if pct_black > black_white_threshold:
        issues.append(f"mostly_black({pct_black:.0%})")
    if pct_white > black_white_threshold:
        issues.append(f"mostly_white({pct_white:.0%})")
    if arr.std() < min_std:
        issues.append(f"near_uniform(std={arr.std():.2f})")
    return issues


def assert_no_degenerate_image(img: Image.Image, context: str) -> None:
    """Raises DegenerateImageError (never a plain Exception a caller's try/except could
    silently absorb into an 'error' record) if img matches a known corruption signature."""
    issues = detect_degenerate_image(img)
    if issues:
        raise DegenerateImageError(
            f"Degenerate image detected for {context}: {issues} - aborting run rather than "
            "writing a poisoned record"
        )


def assert_unique_records(results: list[dict], key_fields: tuple[str, ...]) -> None:
    """Raise if any two non-errored records share the same key_fields composite key. Generalizes
    the retry-duplicate bug found in scripts/_pattachitra_ab_base_comparison.py (a retry
    appended a fresh record instead of replacing a stale errored one, producing 91 records
    instead of 90) into a check every harness run applies to itself."""
    keys = [tuple(r[f] for f in key_fields) for r in results if not r.get("error")]
    dupes = {k: v for k, v in Counter(keys).items() if v > 1}
    if dupes:
        raise AssertionError(f"Duplicate records for key fields {key_fields}: {dupes}")


def validate_judge_scores(
    scores: dict | None, axes: tuple[str, ...] = VLM_JUDGE_AXES
) -> dict | None:
    """Validate a VLM judge response before it's accepted into the dataset. Returns scores
    unchanged if every axis is present, numeric, and in [0.0, 1.0]; otherwise logs a warning and
    returns None - matching score_vlm_judge's existing all-or-nothing failure semantics, so a
    hallucinated out-of-range score (e.g. 1.5) can no longer silently pass through
    `float(result[axis])` and corrupt downstream paired-diff stats."""
    if scores is None:
        return None
    for axis in axes:
        value = scores.get(axis)
        if not isinstance(value, int | float) or isinstance(value, bool) or math.isnan(value):
            logger.warning("Judge response invalid for axis %s: %r", axis, value)
            return None
        if not 0.0 <= value <= 1.0:
            logger.warning("Judge response out of range for axis %s: %s", axis, value)
            return None
    return scores


# ── Family pipeline loaders ──────────────────────────────────────────────


def _img_stats(img: Image.Image) -> dict:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    return {"std": round(float(arr.std()), 3), "mean": round(float(arr.mean()), 3)}


def run_generation_family(
    family: str,
    build_pipe_fn,
    gen_fn,
    scorers: list[str],
    width: int,
    height: int,
    args: argparse.Namespace,
) -> None:
    """Generic runner: build_pipe_fn() -> pipe; gen_fn(pipe, prompt, seed) -> PIL.Image.

    Two-phase, matching scripts/eval.py's proven pattern: (1) generate + CLIP-score every
    image with the generation pipeline resident, (2) release the pipeline, then load HPS and
    score all saved images in a second pass. Scoring HPS immediately after each generation
    (interleaved) keeps the CPU-offloaded generation pipeline AND the HPS model resident in
    VRAM simultaneously — this was tried first and measured to cause severe VRAM-oversubscription
    slowdown (54s -> 335s -> 380s per image, escalating), the same pathology documented in
    docs/LATENCY_ROOT_CAUSE.md. Deferring HPS avoids it entirely.
    """
    check_cuda_health()
    out_json = REPORTS_DIR / f"verdict_{family}.json"
    img_dir = OUT_DIR / family
    img_dir.mkdir(parents=True, exist_ok=True)

    results = load_partial(out_json) if args.resume else []
    completed = done_keys(results)

    prompts = load_prompts()
    if args.limit:
        prompts = prompts[: args.limit]

    logger.info("[%s] Loading pipeline...", family)
    pipe = build_pipe_fn()
    logger.info("[%s] Pipeline ready. %d prompts x %d seeds = %d combos (%d already done)",
                family, len(prompts), len(SEEDS), len(prompts) * len(SEEDS), len(completed))

    # ── Phase 1: generate + CLIP (pipeline stays loaded throughout) ──
    t_start = time.time()
    n_done_this_run = 0
    for prompt_entry in prompts:
        for seed in SEEDS:
            key = f"{prompt_entry['id']}_{seed}"
            if key in completed:
                continue

            record: dict[str, Any] = {
                "family": family,
                "prompt_id": prompt_entry["id"],
                "prompt": prompt_entry["prompt"],
                "seed": seed,
                "hps_score": None,
                "error": None,
            }
            try:
                t0 = time.time()
                img = gen_fn(pipe, prompt_entry["prompt"], seed, width, height)
                latency = time.time() - t0
                assert_no_degenerate_image(img, context=f"{family}:{key}")

                img_path = img_dir / f"{prompt_entry['id']}_seed{seed}.png"
                img.save(img_path)
                record["image_path"] = str(img_path)
                record["latency_s"] = round(latency, 2)
                record.update(_img_stats(img))

                if "clip" in scorers:
                    record["clip_score"] = round(score_clip_one(img, prompt_entry["prompt"]), 4)

            except DegenerateImageError:
                raise
            except Exception as e:
                logger.exception("[%s] Error on %s", family, key)
                record["error"] = str(e)

            results.append(record)
            completed.add(key)
            save_partial(out_json, results)
            n_done_this_run += 1

            status = "ERROR" if record.get("error") else (
                f"clip={record.get('clip_score')} lat={record.get('latency_s')}s "
                f"std={record.get('std')}"
            )
            logger.info("[%s] [%d done this run] %s: %s", family, n_done_this_run, key, status)

    elapsed = time.time() - t_start
    logger.info("[%s] Generation phase done. %d generated this run in %.1f min. Total: %d",
                family, n_done_this_run, elapsed / 60, len(results))
    assert_unique_records(results, ("prompt_id", "seed"))

    # ── Phase 2: release pipeline, score HPS on every image missing it ──
    if "hps" in scorers:
        need_hps = [r for r in results if not r.get("error") and r.get("hps_score") is None]
        if need_hps:
            logger.info("[%s] Releasing generation pipeline before HPS scoring...", family)
            del pipe
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("[%s] Scoring HPS for %d images...", family, len(need_hps))
            for i, r in enumerate(need_hps):
                img = Image.open(r["image_path"])
                hps = score_hps_one(img, r["prompt"])
                r["hps_score"] = round(hps, 6) if hps is not None else None
                save_partial(out_json, results)
                if (i + 1) % 10 == 0 or i == len(need_hps) - 1:
                    logger.info("[%s] HPS [%d/%d] %s_%s: hps=%s",
                                family, i + 1, len(need_hps), r["prompt_id"], r["seed"], r["hps_score"])
            from aetherart.eval_hps import release_hps

            release_hps()

    logger.info("[%s] Done. Total records: %d", family, len(results))


def build_sd21_base():
    from aetherart.model import AetherModel

    m = AetherModel()
    m.init()
    if m.backend != "local":
        raise RuntimeError(f"SD 2.1 did not load locally (backend={m.backend})")
    return m.pipe


def gen_sd21_base(pipe, prompt, seed, width, height):
    gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
    out = pipe(prompt, num_inference_steps=30, guidance_scale=GUIDANCE_DEFAULT,
               width=width, height=height, generator=gen)
    return out.images[0]


def build_sdxl_base():
    from aetherart.sdxl_pipeline import load_sdxl_base

    return load_sdxl_base()


def gen_sdxl_base(pipe, prompt, seed, width, height):
    gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
    out = pipe(prompt, num_inference_steps=30, guidance_scale=GUIDANCE_DEFAULT,
               width=width, height=height, generator=gen)
    return out.images[0]


def build_hyper(variant: str):
    def _build():
        from aetherart.hyper import load_hyper_lora
        from aetherart.sdxl_pipeline import load_sdxl_base

        pipe = load_sdxl_base()
        load_hyper_lora(pipe, variant)
        return pipe

    return _build


def gen_hyper(variant: str):
    from aetherart.hyper import HYPER_DEFAULTS

    defaults = HYPER_DEFAULTS[variant]

    def _gen(pipe, prompt, seed, width, height):
        gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
        out = pipe(
            prompt,
            num_inference_steps=defaults["num_inference_steps"],
            guidance_scale=defaults["guidance_scale"],
            width=width, height=height, generator=gen,
        )
        return out.images[0]

    return _gen


def build_ukiyo_e_lora_sdxl():
    from aetherart.sdxl_pipeline import load_sdxl_base

    pipe = load_sdxl_base()
    lora_path = ROOT / "data" / "lora" / "ukiyo-e" / "ukiyo-e-sdxl-lora.safetensors"
    pipe.load_lora_weights(str(lora_path.parent), weight_name=lora_path.name, adapter_name="ukiyo_e")
    pipe.set_adapters(["ukiyo_e"], adapter_weights=[1.0])
    return pipe


UKIYO_E_NEGATIVE = "text, watermark, calligraphy, signature, words, letters"
UKIYO_E_TRIGGER = "ukyowood"
UKIYO_E_STYLE_QUESTION = (
    "does this image look like an authentic Ukiyo-e (Japanese woodblock print) - flat color "
    "planes, characteristic line work, traditional palette?"
)


def gen_ukiyo_e_lora(pipe, prompt, seed, width, height):
    gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
    full_prompt = f"{prompt}, {UKIYO_E_TRIGGER}, ukiyo-e woodblock print style"
    out = pipe(
        full_prompt, negative_prompt=UKIYO_E_NEGATIVE,
        num_inference_steps=30, guidance_scale=GUIDANCE_DEFAULT,
        width=width, height=height, generator=gen,
    )
    return out.images[0]


def run_ukiyo_e_lora_family(args: argparse.Namespace, adapter_path: Path | None, label: str) -> None:
    """LPIPS-vs-sdxl_base + VLM judge (+ CLIP recorded for context, NOT the verdict metric).

    Two-phase, same rationale as run_generation_family's HPS deferral: the VLM judge is a
    separate Ollama-served model. Calling it while the SDXL+LoRA generation pipeline is still
    GPU-resident reproduces the exact VRAM-oversubscription pathology documented in
    docs/LATENCY_ROOT_CAUSE.md (measured here: image 1 44s, image 2 escalating past 7min/30
    steps before this fix). Phase 1 generates + scores LPIPS/CLIP (both lightweight, already
    GPU-resident alongside the pipeline); phase 2 releases the pipeline, then scores VLM judge
    on every saved image.
    """
    check_cuda_health()
    out_json = REPORTS_DIR / f"verdict_{label}.json"
    img_dir = OUT_DIR / label
    img_dir.mkdir(parents=True, exist_ok=True)

    base_json = REPORTS_DIR / "verdict_sdxl_base.json"
    if not base_json.exists():
        raise RuntimeError(f"sdxl_base must run first (missing {base_json})")
    base_results = {f"{r['prompt_id']}_{r['seed']}": r for r in json.loads(base_json.read_text())}

    results = load_partial(out_json) if args.resume else []
    completed = done_keys(results)

    prompts = load_prompts()
    if args.limit:
        prompts = prompts[: args.limit]

    logger.info("[%s] Loading SDXL + Ukiyo-e LoRA...", label)
    from aetherart.sdxl_pipeline import load_sdxl_base

    pipe = load_sdxl_base()
    path = adapter_path or (ROOT / "data" / "lora" / "ukiyo-e" / "ukiyo-e-sdxl-lora.safetensors")
    pipe.load_lora_weights(str(path.parent), weight_name=path.name, adapter_name="ukiyo_e")
    pipe.set_adapters(["ukiyo_e"], adapter_weights=[1.0])
    logger.info("[%s] Pipeline ready (adapter=%s).", label, path)

    # ── Phase 1: generate + LPIPS + CLIP(context) — pipeline stays loaded throughout ──
    n_done = 0
    for prompt_entry in prompts:
        for seed in SEEDS:
            key = f"{prompt_entry['id']}_{seed}"
            if key in completed:
                continue
            base_key = key
            if base_key not in base_results or base_results[base_key].get("error"):
                logger.warning("[%s] no matching sdxl_base result for %s, skipping", label, key)
                continue

            record: dict[str, Any] = {
                "family": label, "prompt_id": prompt_entry["id"],
                "prompt": prompt_entry["prompt"], "seed": seed, "vlm_judge": None, "error": None,
            }
            try:
                t0 = time.time()
                img = gen_ukiyo_e_lora(pipe, prompt_entry["prompt"], seed, 1024, 1024)
                latency = time.time() - t0
                assert_no_degenerate_image(img, context=f"{label}:{key}")
                img_path = img_dir / f"{prompt_entry['id']}_seed{seed}.png"
                img.save(img_path)
                record["image_path"] = str(img_path)
                record["latency_s"] = round(latency, 2)
                record.update(_img_stats(img))

                record["clip_score_context_only"] = round(
                    score_clip_one(img, prompt_entry["prompt"]), 4
                )

                base_img_path = base_results[base_key]["image_path"]
                base_img = Image.open(base_img_path)
                record["lpips_vs_base"] = round(score_lpips_pair(img, base_img), 6)

            except DegenerateImageError:
                raise
            except Exception as e:
                logger.exception("[%s] Error on %s", label, key)
                record["error"] = str(e)

            results.append(record)
            completed.add(key)
            save_partial(out_json, results)
            n_done += 1
            logger.info(
                "[%s] [%d done] %s: lpips=%.4f clip(ctx)=%.4f",
                label, n_done, key, record.get("lpips_vs_base", -1),
                record.get("clip_score_context_only", -1),
            )

    logger.info("[%s] Generation phase done. %d records total.", label, len(results))
    assert_unique_records(results, ("prompt_id", "seed"))

    # ── Phase 2: release pipeline, then score VLM judge on every image missing it ──
    need_vlm = [r for r in results if not r.get("error") and r.get("vlm_judge") is None]
    if need_vlm:
        logger.info("[%s] Releasing generation pipeline before VLM judge scoring...", label)
        del pipe
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("[%s] Scoring VLM judge for %d images...", label, len(need_vlm))
        for i, r in enumerate(need_vlm):
            img = Image.open(r["image_path"])
            r["vlm_judge"] = score_vlm_judge(
                img, r["prompt"], style_question=UKIYO_E_STYLE_QUESTION
            )
            save_partial(out_json, results)
            if (i + 1) % 10 == 0 or i == len(need_vlm) - 1:
                logger.info("[%s] VLM [%d/%d] %s_%s: vlm=%s",
                            label, i + 1, len(need_vlm), r["prompt_id"], r["seed"], r["vlm_judge"])

    logger.info("[%s] Done. %d records total.", label, len(results))


def build_sdxl_controlnet_union():
    from aetherart.controlnet_sdxl import load_sdxl_controlnet_pipeline

    return load_sdxl_controlnet_pipeline()


def run_controlnet_family(args: argparse.Namespace) -> None:
    """Self-conditioned: canny-extract from sdxl_base's own saved output, same prompt/seed."""
    from aetherart.controlnet_sdxl import generate_sdxl_controlnet, preprocess_canny

    check_cuda_health()
    out_json = REPORTS_DIR / "verdict_sdxl_controlnet_union.json"
    img_dir = OUT_DIR / "sdxl_controlnet_union"
    img_dir.mkdir(parents=True, exist_ok=True)

    base_json = REPORTS_DIR / "verdict_sdxl_base.json"
    if not base_json.exists():
        raise RuntimeError(f"sdxl_base must run first (missing {base_json})")
    base_results = {f"{r['prompt_id']}_{r['seed']}": r for r in json.loads(base_json.read_text())}

    results = load_partial(out_json) if args.resume else []
    completed = done_keys(results)

    prompts = load_prompts()
    if args.limit:
        prompts = prompts[: args.limit]

    logger.info("[sdxl_controlnet_union] Loading pipeline...")
    pipe = build_sdxl_controlnet_union()
    logger.info("[sdxl_controlnet_union] Pipeline ready.")

    n_done = 0
    for prompt_entry in prompts:
        for seed in SEEDS:
            key = f"{prompt_entry['id']}_{seed}"
            if key in completed:
                continue
            if key not in base_results or base_results[key].get("error"):
                logger.warning("[sdxl_controlnet_union] no matching sdxl_base result for %s, skipping", key)
                continue

            record: dict[str, Any] = {
                "family": "sdxl_controlnet_union", "prompt_id": prompt_entry["id"],
                "prompt": prompt_entry["prompt"], "seed": seed, "hps_score": None, "error": None,
            }
            try:
                base_img = Image.open(base_results[key]["image_path"])
                canny = preprocess_canny(base_img)

                t0 = time.time()
                img = generate_sdxl_controlnet(
                    pipe, prompt_entry["prompt"], canny, ctype="canny",
                    guidance_scale=GUIDANCE_DEFAULT, num_inference_steps=30,
                    width=1024, height=1024, seed=seed,
                )
                latency = time.time() - t0
                assert_no_degenerate_image(img, context=f"sdxl_controlnet_union:{key}")
                img_path = img_dir / f"{prompt_entry['id']}_seed{seed}.png"
                img.save(img_path)
                record["image_path"] = str(img_path)
                record["latency_s"] = round(latency, 2)
                record.update(_img_stats(img))
                record["clip_score"] = round(score_clip_one(img, prompt_entry["prompt"]), 4)

            except DegenerateImageError:
                raise
            except Exception as e:
                logger.exception("[sdxl_controlnet_union] Error on %s", key)
                record["error"] = str(e)

            results.append(record)
            completed.add(key)
            save_partial(out_json, results)
            n_done += 1
            logger.info("[sdxl_controlnet_union] [%d done] %s: clip=%s lat=%ss",
                        n_done, key, record.get("clip_score"), record.get("latency_s"))

    logger.info("[sdxl_controlnet_union] Generation phase done. %d records total.", len(results))
    assert_unique_records(results, ("prompt_id", "seed"))

    # Defer HPS to a second pass, same rationale as run_generation_family: avoid the
    # generation-pipeline + HPS-model VRAM contention that caused 335-380s/image slowdowns.
    need_hps = [r for r in results if not r.get("error") and r.get("hps_score") is None]
    if need_hps:
        logger.info("[sdxl_controlnet_union] Releasing pipeline before HPS scoring...")
        del pipe
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("[sdxl_controlnet_union] Scoring HPS for %d images...", len(need_hps))
        for i, r in enumerate(need_hps):
            img = Image.open(r["image_path"])
            hps = score_hps_one(img, r["prompt"])
            r["hps_score"] = round(hps, 6) if hps is not None else None
            save_partial(out_json, results)
            if (i + 1) % 10 == 0 or i == len(need_hps) - 1:
                logger.info("[sdxl_controlnet_union] HPS [%d/%d] %s_%s: hps=%s",
                            i + 1, len(need_hps), r["prompt_id"], r["seed"], r["hps_score"])
        from aetherart.eval_hps import release_hps

        release_hps()

    logger.info("[sdxl_controlnet_union] Done. %d records total.", len(results))


FAMILIES = {
    "sd21_base": lambda args: run_generation_family(
        "sd21_base", build_sd21_base, gen_sd21_base, ["clip", "hps"], 512, 512, args
    ),
    "sdxl_base": lambda args: run_generation_family(
        "sdxl_base", build_sdxl_base, gen_sdxl_base, ["clip", "hps"], 1024, 1024, args
    ),
    "hyper_4step": lambda args: run_generation_family(
        "hyper_4step", build_hyper("4step"), gen_hyper("4step"), ["clip", "hps"], 1024, 1024, args
    ),
    "hyper_8step": lambda args: run_generation_family(
        "hyper_8step", build_hyper("8step"), gen_hyper("8step"), ["clip", "hps"], 1024, 1024, args
    ),
    "sdxl_controlnet_union": run_controlnet_family,
    "ukiyo_e_lora_sdxl": lambda args: run_ukiyo_e_lora_family(args, None, "ukiyo_e_lora_sdxl"),
    "ukiyo_e_lora_sdxl_curated": lambda args: run_ukiyo_e_lora_family(
        args,
        ROOT / "data" / "lora" / "ukiyo-e" / "training_output_sdxl_curated" / "checkpoint-1000"
        / "pytorch_lora_weights.safetensors",
        "ukiyo_e_lora_sdxl_curated",
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=list(FAMILIES))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="Limit to first N prompts (debug)")
    args = ap.parse_args()

    logger.info("=== Model verdict harness: family=%s seeds=%s ===", args.family, SEEDS)
    FAMILIES[args.family](args)
    print("VERDICT_FAMILY_DONE")


if __name__ == "__main__":
    main()
