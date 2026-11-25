"""Unit tests for perception/modules/mtmc/vlm_arbiter.py's response parsing.

Covers the parse paths _parse_verdict must handle: full JSON, a bare verdict
word, JSON wrapped in markdown fences or stray prose, and -- the case added
after live deployment showed the model occasionally loop into a repetition
that burns the num_predict budget before closing the JSON -- a truncated
response recovered via prefix regex (decision is always the first field).
"""
from perception.modules.mtmc.vlm_arbiter import _parse_verdict


def test_parses_full_json():
    text = '{"decision": "same", "confidence": 0.9, "matched_features": ["color"], "conflicting_features": [], "explanation": "ok"}'
    result = _parse_verdict(text)
    assert result["decision"] == "same"
    assert result["confidence"] == 0.9


def test_parses_bare_verdict_word():
    result = _parse_verdict("different")
    assert result["decision"] == "different"
    assert result["confidence"] is None


def test_parses_json_in_markdown_fence():
    text = '```json\n{"decision": "uncertain", "confidence": 0.4}\n```'
    result = _parse_verdict(text)
    assert result["decision"] == "uncertain"
    assert result["confidence"] == 0.4


def test_parses_json_embedded_in_prose():
    text = 'Sure, here is my answer: {"decision": "same", "confidence": 0.7} Hope that helps!'
    result = _parse_verdict(text)
    assert result["decision"] == "same"
    assert result["confidence"] == 0.7


def test_recovers_truncated_output_via_prefix_regex():
    """Repetition-loop / num_predict cutoff: JSON never closes, but the
    leading 'decision'/'confidence' fields are intact and recoverable."""
    text = ('{"decision": "different", "confidence": 0.95, "matched_features": '
            '["The roof rack is different. The roof rack is different. '
            'The roof rack is different. ')
    result = _parse_verdict(text)
    assert result["decision"] == "different"
    assert result["confidence"] == 0.95


def test_returns_none_for_unparseable_garbage():
    assert _parse_verdict("???????") is None
    assert _parse_verdict("") is None


def test_parse_verdict_accepts_verdict_key_alias():
    v = _parse_verdict('{"verdict": "same", "confidence": 0.9}')
    assert v["decision"] == "same"
