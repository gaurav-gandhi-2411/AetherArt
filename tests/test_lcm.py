"""Tests for LCM mode scheduler switching.
No GPU or model downloads required — only diffusers.
"""
from unittest.mock import MagicMock, patch

import pytest

# Entire module requires diffusers — skip cleanly when not installed.
diffusers = pytest.importorskip("diffusers", reason="diffusers required for scheduler tests")
LCMScheduler = diffusers.LCMScheduler
DPMSolverMultistepScheduler = diffusers.DPMSolverMultistepScheduler

from aetherart.lcm import (  # noqa: E402
    LCM_GUIDANCE,
    LCM_STEPS,
    apply_lcm_mode,
    is_lcm_scheduler,
    restore_standard_mode,
)


class TestLCMConstants:
    def test_lcm_steps_is_4(self):
        assert LCM_STEPS == 4

    def test_lcm_guidance_is_low(self):
        assert 1.0 <= LCM_GUIDANCE <= 2.0


class TestSchedulerSwitch:
    def _make_mock_pipe(self, scheduler_class_name: str):
        """Return a mock pipeline whose scheduler reports the given class name."""
        scheduler = MagicMock()
        scheduler.__class__.__name__ = scheduler_class_name
        scheduler.config = {}
        pipe = MagicMock()
        pipe.scheduler = scheduler
        return pipe

    def test_apply_lcm_mode_sets_lcm_scheduler(self):
        pipe = self._make_mock_pipe("DPMSolverMultistepScheduler")
        with patch.object(
            LCMScheduler,
            "from_config",
            return_value=MagicMock(__class__=LCMScheduler),
        ) as mock_fc:
            apply_lcm_mode(pipe)
            mock_fc.assert_called_once_with({})

    def test_restore_standard_mode_sets_dpm_scheduler(self):
        pipe = self._make_mock_pipe("LCMScheduler")
        with patch.object(
            DPMSolverMultistepScheduler,
            "from_config",
            return_value=MagicMock(__class__=DPMSolverMultistepScheduler),
        ) as mock_fc:
            restore_standard_mode(pipe)
            mock_fc.assert_called_once_with({})

    def test_is_lcm_scheduler_true_for_lcm(self):
        pipe = MagicMock()
        pipe.scheduler = MagicMock(spec=LCMScheduler)
        assert is_lcm_scheduler(pipe) is True

    def test_is_lcm_scheduler_false_for_dpm(self):
        pipe = MagicMock()
        pipe.scheduler = MagicMock(spec=DPMSolverMultistepScheduler)
        assert is_lcm_scheduler(pipe) is False
