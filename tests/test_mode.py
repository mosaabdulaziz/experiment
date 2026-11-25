from app.mode import VALID_MODES, analysis_enabled


def test_valid_modes():
    assert VALID_MODES == ("proactive", "chat")


def test_analysis_enabled_proactive_idle():
    assert analysis_enabled({"active_mode": "proactive", "chat_in_flight": 0}) is True


def test_analysis_disabled_in_chat_mode():
    assert analysis_enabled({"active_mode": "chat", "chat_in_flight": 0}) is False


def test_analysis_disabled_while_chat_in_flight():
    assert analysis_enabled({"active_mode": "proactive", "chat_in_flight": 1}) is False
    assert analysis_enabled({"active_mode": "proactive", "chat_in_flight": 2}) is False


def test_analysis_enabled_defaults_to_proactive():
    # Missing keys → behaves as proactive/idle.
    assert analysis_enabled({}) is True
