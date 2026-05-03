"""Tests for SDXL Turbo module.
No actual model loading — verifies constants and API shape only.
"""

from aetherart.sdxl_turbo import TURBO_GUIDANCE, TURBO_MODEL_ID, TURBO_STEPS


class TestTurboConstants:
    def test_model_id(self):
        assert TURBO_MODEL_ID == "stabilityai/sdxl-turbo"

    def test_steps_is_one(self):
        assert TURBO_STEPS == 1

    def test_guidance_is_zero(self):
        assert TURBO_GUIDANCE == 0.0


class TestTurboAPI:
    def test_generate_turbo_signature(self):
        import inspect

        from aetherart.sdxl_turbo import generate_turbo

        sig = inspect.signature(generate_turbo)
        params = list(sig.parameters.keys())
        assert "pipe" in params
        assert "prompt" in params
        assert "seed" in params
        assert "width" in params
        assert "height" in params

    def test_free_pipeline_handles_none(self):
        """free_turbo_pipeline should not crash if called with a mock."""
        from unittest.mock import MagicMock

        from aetherart.sdxl_turbo import free_turbo_pipeline

        mock_pipe = MagicMock()
        # Should not raise
        free_turbo_pipeline(mock_pipe)
