"""Tests for scripts/model_verdict_harness.py's VLM judge — independent single-axis scoring.

A halo-effect check (docs/MODEL_VERDICT.md §4.5) found that scoring style_adherence,
figure_preservation, and artifact_absence in one Ollama call per image produces correlated
ratings that don't hold up under independent scoring. The harness now scores each axis in its
own call, with no other axis named in that call's prompt. This test asserts that invocation
pattern directly, so a future edit can't silently reintroduce the single-call design.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image


def _import_harness():
    spec = importlib.util.spec_from_file_location(
        "model_verdict_harness",
        Path(__file__).resolve().parent.parent / "scripts" / "model_verdict_harness.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["model_verdict_harness"] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _import_harness()


def _mock_ollama_response(axis: str) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"response": f'{{"{axis}": 0.9}}'}
    return resp


class TestVlmJudgeIsIndependentPerAxis:
    def test_scores_one_image_with_exactly_three_calls(self):
        """One Ollama call per axis (3 total) — never one call for all three axes."""
        img = Image.new("RGB", (8, 8))

        def fake_post(url, json, timeout):
            # The request body must carry exactly one axis's worth of prompt/response schema —
            # asserting this here, not just the call count, is what would catch a regression
            # back to "ask for all three scores in one call."
            axis = next(a for a in _mod.VLM_JUDGE_AXES if f'"{a}"' in json["prompt"])
            other_axes = [a for a in _mod.VLM_JUDGE_AXES if a != axis]
            assert not any(f'"{other}"' in json["prompt"] for other in other_axes), (
                f"axis {axis}'s prompt must not mention other axes {other_axes}"
            )
            return _mock_ollama_response(axis)

        with patch("requests.post", side_effect=fake_post) as mock_post:
            result = _mod.score_vlm_judge(img, "a test prompt")

        assert mock_post.call_count == 3
        assert result == {
            "style_adherence": 0.9,
            "figure_preservation": 0.9,
            "artifact_absence": 0.9,
        }

    def test_each_axis_prompt_names_only_itself(self):
        """Static check on the prompt templates themselves, independent of any mocking —
        catches a regression even if a future refactor changes how calls are dispatched."""
        for axis, template in _mod.SINGLE_AXIS_JUDGE_PROMPTS.items():
            rendered = template.format(prompt="irrelevant")
            other_axes = [a for a in _mod.VLM_JUDGE_AXES if a != axis]
            for other in other_axes:
                assert f'"{other}"' not in rendered, (
                    f"{axis}'s judge prompt must not request the {other} score"
                )
            assert f'"{axis}"' in rendered

    def test_one_axis_failure_fails_the_whole_record(self):
        """Preserves prior all-or-nothing failure semantics: if any one axis call fails, the
        whole image's vlm_judge record is None, not partially populated."""
        img = Image.new("RGB", (8, 8))
        call_count = {"n": 0}

        def fake_post(url, json, timeout):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise ConnectionError("simulated failure on the second axis call")
            axis = next(a for a in _mod.VLM_JUDGE_AXES if f'"{a}"' in json["prompt"])
            return _mock_ollama_response(axis)

        with patch("requests.post", side_effect=fake_post):
            result = _mod.score_vlm_judge(img, "a test prompt")

        assert result is None
