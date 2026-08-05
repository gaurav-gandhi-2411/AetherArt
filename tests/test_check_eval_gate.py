"""Tests for scripts/check_eval_gate.py — the CI CLIP-regression gate.

Uses synthetic fixture JSON, not real generation — the gate's comparison logic is what's
under test here; the real-generation smoke test lives in test_generation_smoke.py.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def _import_gate():
    spec = importlib.util.spec_from_file_location(
        "check_eval_gate",
        Path(__file__).resolve().parent.parent / "scripts" / "check_eval_gate.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_eval_gate"] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _import_gate()


def _write_eval_json(
    path: Path, scores: list[float], scheduler: str = "DPM", steps: int = 30
) -> None:
    results = [
        {
            "prompt_id": f"pp_{i:03d}",
            "scheduler": scheduler,
            "steps": steps,
            "clip_score": s,
            "error": None,
        }
        for i, s in enumerate(scores)
    ]
    path.write_text(json.dumps({"results": results}), encoding="utf-8")


def _real_prompt_records() -> list[dict]:
    """The exact 30 (prompt_id, prompt, category, expected_difficulty) records the real,
    pinned BASELINE_PROMPT_SET_SHA256 was computed from -- fixtures that need to pass the
    gate's identity check must carry this content (clip_score is unconstrained)."""
    with _mod.DEFAULT_BASELINE.open(encoding="utf-8") as f:
        data = json.load(f)
    return _mod._cell_prompt_identity_records(data["results"], "DPM", 30)


def _write_eval_json_matching_baseline_prompts(
    path: Path, scores: list[float], scheduler: str = "DPM", steps: int = 30
) -> None:
    """Like _write_eval_json, but pairs each score with the real baseline's prompt-identity
    content in order, so the candidate's content hash matches BASELINE_PROMPT_SET_SHA256 --
    for tests that need to get past the identity gate to exercise the mean/SEM comparison."""
    records = _real_prompt_records()
    assert len(scores) <= len(records), "only 30 real prompt records available"
    results = [
        {**records[i], "scheduler": scheduler, "steps": steps, "clip_score": s, "error": None}
        for i, s in enumerate(scores)
    ]
    path.write_text(json.dumps({"results": results}), encoding="utf-8")


class TestComputeBaseline:
    def test_mean_and_sem_match_known_values(self, tmp_path):
        # Known small dataset: mean=3, sample stdev(n-1)=sqrt(2.5)=1.5811..., sem=stdev/sqrt(5)
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        p = tmp_path / "baseline.json"
        _write_eval_json(p, scores)

        mean, sem, n = _mod.compute_baseline(p, "DPM", 30)

        assert n == 5
        assert mean == 3.0
        assert abs(sem - (1.5811388300841898 / (5**0.5))) < 1e-9

    def test_ignores_errored_and_mismatched_cells(self, tmp_path):
        p = tmp_path / "baseline.json"
        results = [
            {"scheduler": "DPM", "steps": 30, "clip_score": 0.30, "error": None},
            {"scheduler": "DPM", "steps": 30, "clip_score": 0.32, "error": None},
            {"scheduler": "DPM", "steps": 30, "clip_score": 0.99, "error": "OOM"},  # excluded
            {"scheduler": "DDIM", "steps": 30, "clip_score": 0.10, "error": None},  # wrong sched
            {"scheduler": "DPM", "steps": 20, "clip_score": 0.10, "error": None},  # wrong steps
        ]
        p.write_text(json.dumps({"results": results}), encoding="utf-8")

        mean, _sem, n = _mod.compute_baseline(p, "DPM", 30)

        assert n == 2
        assert mean == 0.31

    def test_raises_on_insufficient_samples(self, tmp_path):
        p = tmp_path / "baseline.json"
        _write_eval_json(p, [0.30])

        import pytest

        with pytest.raises(ValueError, match="need >= 2"):
            _mod.compute_baseline(p, "DPM", 30)


class TestComputeCandidateMean:
    def test_mean_of_matching_cell(self, tmp_path):
        p = tmp_path / "candidate.json"
        _write_eval_json(p, [0.28, 0.30, 0.32])

        mean, n = _mod.compute_candidate_mean(p, "DPM", 30)

        assert n == 3
        assert abs(mean - 0.3) < 1e-9

    def test_raises_when_no_matching_scores(self, tmp_path):
        p = tmp_path / "candidate.json"
        _write_eval_json(p, [0.30], scheduler="EulerA", steps=20)

        import pytest

        with pytest.raises(ValueError, match="no usable"):
            _mod.compute_candidate_mean(p, "DPM", 30)


class TestGateCliExitCodes:
    """End-to-end: invoke the script as a subprocess and check the real exit code."""

    def _make_baseline(self, tmp_path) -> Path:
        # 30 identical-ish scores centered at 0.32 with small spread -> tight, real SEM.
        scores = [0.32 + (0.001 * (i % 5 - 2)) for i in range(30)]
        p = tmp_path / "baseline.json"
        _write_eval_json(p, scores)
        return p

    def test_exits_zero_when_candidate_matches_baseline(self, tmp_path):
        baseline = self._make_baseline(tmp_path)
        candidate = tmp_path / "candidate_ok.json"
        _write_eval_json_matching_baseline_prompts(candidate, [0.320] * 30)

        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().parent.parent / "scripts" / "check_eval_gate.py"),
                "--candidate",
                str(candidate),
                "--baseline",
                str(baseline),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stdout + result.stderr
        assert "PASS" in result.stdout

    def test_exits_nonzero_on_real_regression(self, tmp_path):
        baseline = self._make_baseline(tmp_path)
        candidate = tmp_path / "candidate_bad.json"
        # Far below any plausible SEM-derived threshold -> must fail, not be averaged away.
        _write_eval_json_matching_baseline_prompts(candidate, [0.05] * 30)

        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().parent.parent / "scripts" / "check_eval_gate.py"),
                "--candidate",
                str(candidate),
                "--baseline",
                str(baseline),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1, result.stdout + result.stderr
        assert "FAIL" in result.stdout

    def test_default_baseline_reproduces_documented_anchor(self):
        """The real baseline file, DPM/30 steps, must reproduce the documented CLIP=0.3199."""
        mean, sem, n = _mod.compute_baseline(_mod.DEFAULT_BASELINE, "DPM", 30)

        assert n == 30
        assert round(mean, 4) == 0.3199
        assert 0.004 <= sem <= 0.007  # matches reports/clip_blindness.md's ~0.004-0.007/cell note


class TestComputePromptSetHash:
    """Content-hash identity pin: --num-images 30 pins the COUNT of prompts a candidate run
    uses, not their IDENTITY. These verify the hash actually distinguishes 'same 30 prompts'
    from 'a reordered or edited version of the same 30 prompts'."""

    def test_pinned_constant_matches_the_real_baseline_file(self):
        """BASELINE_PROMPT_SET_SHA256 must actually be derived from the real baseline this
        repo's threshold traces to -- if this ever fails, the pinned hash was hand-edited or
        the baseline file changed without recomputing it (exactly the drift this gate exists
        to prevent one layer up, in check_eval_gate.main())."""
        real_hash = _mod.compute_candidate_prompt_set_hash(_mod.DEFAULT_BASELINE, "DPM", 30)

        assert real_hash == _mod.BASELINE_PROMPT_SET_SHA256

    def test_identical_content_hashes_identically(self):
        records = _real_prompt_records()
        results_a = [{**r, "scheduler": "DPM", "steps": 30, "error": None} for r in records]
        results_b = [{**r, "scheduler": "DPM", "steps": 30, "error": None} for r in records]

        assert _mod.compute_prompt_set_hash(results_a, "DPM", 30) == _mod.compute_prompt_set_hash(
            results_b, "DPM", 30
        )

    def test_reordering_the_same_30_prompts_changes_the_hash(self):
        records = _real_prompt_records()
        reordered = [records[1], records[0], *records[2:]]  # swap first two
        original_results = [{**r, "scheduler": "DPM", "steps": 30, "error": None} for r in records]
        reordered_results = [
            {**r, "scheduler": "DPM", "steps": 30, "error": None} for r in reordered
        ]

        original_hash = _mod.compute_prompt_set_hash(original_results, "DPM", 30)
        reordered_hash = _mod.compute_prompt_set_hash(reordered_results, "DPM", 30)

        assert original_hash != reordered_hash
        assert reordered_hash != _mod.BASELINE_PROMPT_SET_SHA256

    def test_editing_one_prompts_text_changes_the_hash(self):
        records = _real_prompt_records()
        edited = [dict(r) for r in records]
        edited[0]["prompt"] = edited[0]["prompt"] + " (edited)"
        edited_results = [{**r, "scheduler": "DPM", "steps": 30, "error": None} for r in edited]

        edited_hash = _mod.compute_prompt_set_hash(edited_results, "DPM", 30)

        assert edited_hash != _mod.BASELINE_PROMPT_SET_SHA256


class TestGatePromptIdentityCli:
    """End-to-end: the identity check runs as part of the real CLI, before the mean/SEM
    comparison -- a mismatch must exit non-zero with a distinct message, never a silent
    pass-through into the numeric check."""

    def _make_baseline(self, tmp_path) -> Path:
        scores = [0.32 + (0.001 * (i % 5 - 2)) for i in range(30)]
        p = tmp_path / "baseline.json"
        _write_eval_json(p, scores)
        return p

    def _run_gate(self, candidate: Path, baseline: Path) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().parent.parent / "scripts" / "check_eval_gate.py"),
                "--candidate",
                str(candidate),
                "--baseline",
                str(baseline),
            ],
            capture_output=True,
            text=True,
        )

    def test_unchanged_prompt_set_passes_the_identity_check(self, tmp_path):
        baseline = self._make_baseline(tmp_path)
        candidate = tmp_path / "candidate.json"
        _write_eval_json_matching_baseline_prompts(candidate, [0.320] * 30)

        result = self._run_gate(candidate, baseline)

        assert result.returncode == 0, result.stdout + result.stderr
        assert "identity mismatch" not in result.stdout
        assert "PASS" in result.stdout

    def test_identity_check_emits_its_own_success_line_on_match(self, tmp_path):
        """The identity check must log its own PASS line, distinct from the mean/SEM PASS
        line below it -- otherwise a silently-skipped check and one that actually ran and
        matched produce identical output, which is the observability gap this test guards."""
        baseline = self._make_baseline(tmp_path)
        candidate = tmp_path / "candidate.json"
        _write_eval_json_matching_baseline_prompts(candidate, [0.320] * 30)

        result = self._run_gate(candidate, baseline)

        assert result.returncode == 0, result.stdout + result.stderr
        expected_hash_prefix = _mod.BASELINE_PROMPT_SET_SHA256[:16]
        assert (
            f"PASS: baseline prompt-set identity check ({expected_hash_prefix}" in result.stdout
        ), result.stdout

    def test_reordered_first_30_fails_loudly_not_silently(self, tmp_path):
        baseline = self._make_baseline(tmp_path)
        candidate = tmp_path / "candidate.json"
        records = _real_prompt_records()
        reordered = [records[1], records[0], *records[2:]]  # same 30 IDs, different order
        results = [
            {**r, "scheduler": "DPM", "steps": 30, "clip_score": 0.320, "error": None}
            for r in reordered
        ]
        candidate.write_text(json.dumps({"results": results}), encoding="utf-8")

        result = self._run_gate(candidate, baseline)

        assert result.returncode == 1, result.stdout + result.stderr
        assert "identity mismatch" in result.stdout
        assert "PASS" not in result.stdout  # must not fall through to the numeric check
        assert "DO NOT" in result.stdout  # must not silently suggest auto-updating the hash

    def test_edited_first_30_fails_loudly_not_silently(self, tmp_path):
        baseline = self._make_baseline(tmp_path)
        candidate = tmp_path / "candidate.json"
        records = _real_prompt_records()
        edited = [dict(r) for r in records]
        edited[5]["prompt"] = "a completely different prompt"
        results = [
            {**r, "scheduler": "DPM", "steps": 30, "clip_score": 0.320, "error": None}
            for r in edited
        ]
        candidate.write_text(json.dumps({"results": results}), encoding="utf-8")

        result = self._run_gate(candidate, baseline)

        assert result.returncode == 1, result.stdout + result.stderr
        assert "identity mismatch" in result.stdout
