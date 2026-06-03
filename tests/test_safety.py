from __future__ import annotations

from unittest.mock import MagicMock, patch

import aetherart.safety as safety_mod


def test_check_prompt_no_op_when_disabled() -> None:
    with patch.object(safety_mod, "_ENABLED", False):
        assert safety_mod.check_prompt("nude explicit content") is None


def test_check_prompt_blocks_when_enabled() -> None:
    with patch.object(safety_mod, "_ENABLED", True):
        result = safety_mod.check_prompt("a beautiful nude portrait")
        assert result is not None
        assert isinstance(result, str)


def test_check_prompt_blocks_uppercase_when_enabled() -> None:
    with patch.object(safety_mod, "_ENABLED", True):
        result = safety_mod.check_prompt("A NUDE PAINTING")
        assert result is not None


def test_check_prompt_allows_clean_prompt_when_enabled() -> None:
    with patch.object(safety_mod, "_ENABLED", True):
        assert safety_mod.check_prompt("ukiyo-e print of Mount Fuji at sunset") is None


def test_check_prompt_allows_clean_prompt_when_disabled() -> None:
    with patch.object(safety_mod, "_ENABLED", False):
        assert safety_mod.check_prompt("ukiyo-e print of Mount Fuji at sunset") is None


def test_is_enabled_true() -> None:
    with patch.object(safety_mod, "_ENABLED", True):
        assert safety_mod.is_enabled() is True


def test_is_enabled_false() -> None:
    with patch.object(safety_mod, "_ENABLED", False):
        assert safety_mod.is_enabled() is False


def test_apply_safety_checker_no_op_when_disabled() -> None:
    pipe = MagicMock()
    with patch.object(safety_mod, "_ENABLED", False):
        safety_mod.apply_safety_checker(pipe)  # must not raise


def test_apply_safety_checker_no_op_on_sdxl_pipe() -> None:
    # SDXL pipelines have no safety_checker attribute
    pipe = MagicMock(spec=[])
    with patch.object(safety_mod, "_ENABLED", True):
        safety_mod.apply_safety_checker(pipe)  # must not raise


def test_blocklist_covers_minimum_terms() -> None:
    assert len(safety_mod._BLOCKLIST) >= 10
