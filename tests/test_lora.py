"""Tests for aetherart/lora.py — registry lookups and load/unload helpers.

No real LoRA weights are loaded. Pipeline is mocked throughout.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from aetherart.lora import (
    LORA_REGISTRY,
    _unload_safe,
    get_default_negative,
    get_trigger_token,
    load_lora,
    unload_lora,
)


class TestLoraRegistry:
    def test_none_entry_is_null(self):
        assert LORA_REGISTRY["none"] is None

    def test_ukiyo_e_has_required_keys(self):
        cfg = LORA_REGISTRY["ukiyo-e"]
        assert cfg is not None
        assert "path" in cfg
        assert "trigger_token" in cfg
        assert "default_negative" in cfg

    def test_all_named_loras_have_path(self):
        for name, cfg in LORA_REGISTRY.items():
            if name == "none":
                assert cfg is None
            else:
                assert isinstance(cfg, dict) and "path" in cfg


class TestGetTriggerToken:
    def test_known_lora(self):
        assert get_trigger_token("ukiyo-e") == "ukyowood"

    def test_none_lora(self):
        assert get_trigger_token("none") == ""

    def test_unknown_lora(self):
        assert get_trigger_token("does-not-exist") == ""


class TestGetDefaultNegative:
    def test_known_lora_is_nonempty(self):
        neg = get_default_negative("ukiyo-e")
        assert isinstance(neg, str)
        assert len(neg) > 0

    def test_none_lora(self):
        assert get_default_negative("none") == ""

    def test_unknown_lora(self):
        assert get_default_negative("not-a-lora") == ""


class TestUnloadSafe:
    def test_calls_unload_lora_weights(self):
        pipe = MagicMock()
        _unload_safe(pipe)
        pipe.unload_lora_weights.assert_called_once()

    def test_swallows_exception(self):
        pipe = MagicMock()
        pipe.unload_lora_weights.side_effect = RuntimeError("no adapter loaded")
        _unload_safe(pipe)  # must not raise


class TestUnloadLora:
    def test_delegates_to_unload_safe(self):
        pipe = MagicMock()
        unload_lora(pipe)
        pipe.unload_lora_weights.assert_called_once()


class TestLoadLora:
    def test_none_name_skips_pipeline(self):
        pipe = MagicMock()
        load_lora(pipe, "none")
        pipe.load_lora_weights.assert_not_called()

    def test_unknown_name_skips_pipeline(self):
        pipe = MagicMock()
        load_lora(pipe, "not-a-real-lora")
        pipe.load_lora_weights.assert_not_called()

    def test_known_name_calls_load_and_set_adapters(self):
        pipe = MagicMock()
        load_lora(pipe, "ukiyo-e", alpha=0.8)
        pipe.load_lora_weights.assert_called_once()
        pipe.set_adapters.assert_called_once_with(["ukiyo_e"], adapter_weights=[0.8])

    def test_known_name_default_alpha_is_one(self):
        pipe = MagicMock()
        load_lora(pipe, "ukiyo-e")
        _, kwargs = pipe.set_adapters.call_args
        assert kwargs.get("adapter_weights") == [1.0] or pipe.set_adapters.call_args[0][1] == [1.0]
