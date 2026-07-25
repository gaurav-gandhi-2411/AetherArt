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

import numpy as np
import pytest
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


TEST_STYLE_QUESTION = "does this image look like an authentic Test-Domain style?"


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
            result = _mod.score_vlm_judge(img, "a test prompt", style_question=TEST_STYLE_QUESTION)

        assert mock_post.call_count == 3
        assert result == {
            "style_adherence": 0.9,
            "figure_preservation": 0.9,
            "artifact_absence": 0.9,
        }

    def test_each_axis_prompt_names_only_itself(self):
        """Static check on the domain-neutral prompt templates (figure_preservation,
        artifact_absence — style_adherence is no longer a fixed template, see
        TestStyleQuestionIsNeverHardcoded), independent of any mocking — catches a regression
        even if a future refactor changes how calls are dispatched."""
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
            result = _mod.score_vlm_judge(img, "a test prompt", style_question=TEST_STYLE_QUESTION)

        assert result is None

    def test_out_of_range_axis_score_fails_the_whole_record(self):
        """A judge hallucinating a value outside [0, 1] (e.g. 1.5) must not silently pass
        through float(result[axis]) into the dataset - added after a CUDA-corruption audit
        found no range check existed on the judge's raw response."""
        img = Image.new("RGB", (8, 8))

        def fake_post(url, json, timeout):
            axis = next(a for a in _mod.VLM_JUDGE_AXES if f'"{a}"' in json["prompt"])
            if axis == "figure_preservation":
                resp = MagicMock()
                resp.raise_for_status.return_value = None
                resp.json.return_value = {"response": '{"figure_preservation": 1.5}'}
                return resp
            return _mock_ollama_response(axis)

        with patch("requests.post", side_effect=fake_post):
            result = _mod.score_vlm_judge(img, "a test prompt", style_question=TEST_STYLE_QUESTION)

        assert result is None


class TestStyleQuestionIsNeverHardcoded:
    """Regression suite for the bug found in docs/MODEL_VERDICT.md SS7's judge positive-control
    audit: style_adherence's question was hardcoded to ask about Ukiyo-e regardless of caller,
    silently producing 360 wrong-question Pattachitra scores. This is a NEW bug class for this
    project — semantically wrong but syntactically valid (a well-formed request, a well-formed
    in-range response, a real, working Ollama call) — invisible to every integrity self-check
    added after the CUDA-corruption audit (CUDA health probe, degenerate-image detection,
    uniqueness assertion, judge-score range validation), because none of those check whether the
    *question itself* is correct for the domain being scored. These tests close that specific
    gap: they assert the judge question actually asked matches the domain under test, not just
    that some well-formed question was asked."""

    def _capture_style_adherence_prompt(self, style_question: str) -> str:
        img = Image.new("RGB", (8, 8))
        captured = {}

        def fake_post(url, json, timeout):
            axis = next(a for a in _mod.VLM_JUDGE_AXES if f'"{a}"' in json["prompt"])
            if axis == "style_adherence":
                captured["prompt"] = json["prompt"]
            return _mock_ollama_response(axis)

        with patch("requests.post", side_effect=fake_post):
            _mod.score_vlm_judge(img, "a test prompt", style_question=style_question)
        return captured["prompt"]

    def test_style_question_is_required_with_no_default(self):
        """No default to silently fall back to - a caller that forgets to pass a style fails
        loudly (TypeError) rather than reusing whatever domain was hardcoded last."""
        img = Image.new("RGB", (8, 8))
        with pytest.raises(TypeError, match="style_question"):
            _mod.score_vlm_judge(img, "a test prompt")

    def test_judge_question_contains_the_domain_under_test(self):
        """The actual text sent to Ollama for the style_adherence axis must contain the
        caller-supplied style_question verbatim - proves the parameter is threaded through, not
        a decorative no-op."""
        prompt_sent = self._capture_style_adherence_prompt(
            "does this image look like an authentic Pattachitra painting?"
        )
        assert "Pattachitra" in prompt_sent

    def test_different_domains_produce_different_questions(self):
        """Two different style_question values must produce two different requests - if this
        failed, some hardcoded text would be overriding the parameter, exactly like the
        original bug."""
        ukiyo_e_prompt = self._capture_style_adherence_prompt("does this look like Ukiyo-e?")
        pattachitra_prompt = self._capture_style_adherence_prompt(
            "does this look like Pattachitra?"
        )
        assert ukiyo_e_prompt != pattachitra_prompt
        assert "Ukiyo-e" in ukiyo_e_prompt
        assert "Pattachitra" not in ukiyo_e_prompt
        assert "Pattachitra" in pattachitra_prompt
        assert "Ukiyo-e" not in pattachitra_prompt

    def test_no_leftover_style_name_for_an_unrelated_domain(self):
        """Behavioral check for the same defect class the original bug belonged to: if any style
        name were still hardcoded anywhere in score_vlm_judge's prompt construction (e.g.
        silently appended alongside the caller's text, not just replaced by it), it would leak
        into a request even when the caller names a completely unrelated domain. This targets the
        actual attack surface (the request Ollama receives) rather than source text, which would
        also flag legitimate documentation examples (e.g. this file's own docstrings mention
        Ukiyo-e/Pattachitra by name as illustrations)."""
        prompt_sent = self._capture_style_adherence_prompt(
            "does this image look like an authentic Bauhaus poster?"
        )
        lowered = prompt_sent.lower()
        for leftover in ("ukiyo", "woodblock", "pattachitra", "pattascroll"):
            assert leftover not in lowered, (
                f"found leftover hardcoded {leftover!r} in a request for an unrelated domain"
            )

    def test_figure_preservation_and_artifact_absence_are_domain_neutral(self):
        """Audited for the same defect class, per the task that found the style_adherence bug:
        neither of the other two axes' fixed templates may name a style either. Both currently
        pass (this is a regression guard, not a report of a second bug)."""
        forbidden_style_names = ["ukiyo", "woodblock", "pattachitra", "pattascroll"]
        for axis in ("figure_preservation", "artifact_absence"):
            template = _mod.SINGLE_AXIS_JUDGE_PROMPTS[axis].lower()
            for name in forbidden_style_names:
                assert name not in template, (
                    f"{axis}'s template must not hardcode a style name (found {name!r})"
                )


class TestCudaHealthProbe:
    """docs/MODEL_VERDICT.md's CUDA-corruption audit found no pre-flight check existed to catch
    a poisoned CUDA context before spending minutes loading a pipeline onto it."""

    def test_noop_when_cuda_unavailable(self):
        with patch.object(_mod.torch.cuda, "is_available", return_value=False):
            _mod.check_cuda_health()  # must not raise

    def test_raises_on_broken_context(self):
        cuda_error = RuntimeError("CUDA error: illegal memory access")
        with (
            patch.object(_mod.torch.cuda, "is_available", return_value=True),
            patch.object(_mod.torch, "tensor", side_effect=cuda_error),
            pytest.raises(RuntimeError, match="health probe failed"),
        ):
            _mod.check_cuda_health()

    def test_raises_on_wrong_result(self):
        mock_tensor = MagicMock()
        mock_tensor.__mul__ = MagicMock(return_value=mock_tensor)
        mock_tensor.sum.return_value = mock_tensor
        mock_tensor.item.return_value = 99.0  # correct probe result is 12.0

        with (
            patch.object(_mod.torch.cuda, "is_available", return_value=True),
            patch.object(_mod.torch, "tensor", return_value=mock_tensor),
            patch.object(_mod.torch.cuda, "synchronize"),
            pytest.raises(RuntimeError, match="corrupted result"),
        ):
            _mod.check_cuda_health()


class TestDetectDegenerateImage:
    """Pixel-level corruption signatures (NaN, solid-color collapse) are the failure mode a
    poisoned CUDA context actually produces - this is not a style/quality judgment."""

    def test_normal_varied_image_has_no_issues(self):
        arr = np.random.default_rng(42).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        assert _mod.detect_degenerate_image(img) == []

    def test_all_black_image_flagged(self):
        img = Image.new("RGB", (32, 32), (0, 0, 0))
        issues = _mod.detect_degenerate_image(img)
        assert any(i.startswith("mostly_black") for i in issues)

    def test_all_white_image_flagged(self):
        img = Image.new("RGB", (32, 32), (255, 255, 255))
        issues = _mod.detect_degenerate_image(img)
        assert any(i.startswith("mostly_white") for i in issues)

    def test_uniform_gray_image_flagged_near_uniform(self):
        img = Image.new("RGB", (32, 32), (128, 128, 128))
        issues = _mod.detect_degenerate_image(img)
        assert any(i.startswith("near_uniform") for i in issues)


class TestAssertNoDegenerateImage:
    def test_raises_degenerate_image_error_on_black_image(self):
        img = Image.new("RGB", (32, 32), (0, 0, 0))
        with pytest.raises(_mod.DegenerateImageError):
            _mod.assert_no_degenerate_image(img, context="test:pat_001_42")

    def test_no_raise_on_normal_image(self):
        arr = np.random.default_rng(7).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        _mod.assert_no_degenerate_image(img, context="test:pat_001_42")  # must not raise


class TestAssertUniqueRecords:
    """Generalizes the retry-duplicate bug (a retry appended a fresh record instead of
    replacing a stale errored one, producing 91 records instead of 90) into a check every
    harness run applies to its own output."""

    def test_no_dupes_does_not_raise(self):
        results = [
            {"prompt_id": "pat_001", "seed": 42, "error": None},
            {"prompt_id": "pat_001", "seed": 43, "error": None},
        ]
        _mod.assert_unique_records(results, ("prompt_id", "seed"))  # must not raise

    def test_duplicate_non_errored_record_raises(self):
        results = [
            {"prompt_id": "pat_001", "seed": 42, "error": None},
            {"prompt_id": "pat_001", "seed": 42, "error": None},
        ]
        with pytest.raises(AssertionError, match="Duplicate"):
            _mod.assert_unique_records(results, ("prompt_id", "seed"))

    def test_errored_record_sharing_key_with_success_is_not_a_duplicate(self):
        """A retry's successful record legitimately shares a key with its own stale errored
        predecessor - that predecessor must have already been stripped by the caller, but this
        check itself must not treat error+success as a collision."""
        results = [
            {"prompt_id": "pat_001", "seed": 42, "error": "RuntimeError: CUDA error"},
            {"prompt_id": "pat_001", "seed": 42, "error": None},
        ]
        _mod.assert_unique_records(results, ("prompt_id", "seed"))  # must not raise


class TestValidateJudgeScores:
    def test_valid_scores_returned_unchanged(self):
        scores = {"style_adherence": 0.9, "figure_preservation": 0.95, "artifact_absence": 1.0}
        assert _mod.validate_judge_scores(scores) == scores

    def test_none_input_returns_none(self):
        assert _mod.validate_judge_scores(None) is None

    def test_missing_axis_returns_none(self):
        scores = {"style_adherence": 0.9, "figure_preservation": 0.95}
        assert _mod.validate_judge_scores(scores) is None

    def test_out_of_range_value_returns_none(self):
        scores = {"style_adherence": 1.5, "figure_preservation": 0.95, "artifact_absence": 1.0}
        assert _mod.validate_judge_scores(scores) is None

    def test_non_numeric_value_returns_none(self):
        scores = {"style_adherence": "high", "figure_preservation": 0.95, "artifact_absence": 1.0}
        assert _mod.validate_judge_scores(scores) is None

    def test_nan_value_returns_none(self):
        scores = {"style_adherence": float("nan"), "figure_preservation": 0.95,
                   "artifact_absence": 1.0}
        assert _mod.validate_judge_scores(scores) is None
