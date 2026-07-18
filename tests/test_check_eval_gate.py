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
        _write_eval_json(candidate, [0.320] * 30)

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
        _write_eval_json(candidate, [0.05] * 30)

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
