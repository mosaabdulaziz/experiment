"""The app starts without a VLM key unless a VLM is actually switched on."""
import main


def test_no_key_and_no_vlm_starts_with_a_placeholder():
    cfg = {"monitoring": {"vlm_enabled": False}, "api": {"api_key": ""}}
    assert main.ensure_api_key(cfg) is True
    assert cfg["api"]["api_key"]      # the VLM client refuses to be built with None


def test_no_key_with_vlm_enabled_refuses_to_start():
    assert main.ensure_api_key({"monitoring": {"vlm_enabled": True}, "api": {"api_key": ""}}) is False


def test_a_real_key_is_left_alone():
    cfg = {"monitoring": {"vlm_enabled": True}, "api": {"api_key": "sk-real"}}
    assert main.ensure_api_key(cfg) is True and cfg["api"]["api_key"] == "sk-real"


def test_vlm_disabled_is_the_default_when_the_key_is_absent_from_config():
    assert main.ensure_api_key({}) is True
