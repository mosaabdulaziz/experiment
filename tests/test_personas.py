import re
import yaml

from app.personas import PERSONAS, DEFAULT_PERSONA
from app.server import create_app


def test_right_turn_on_red_exception_present():
    instructions = PERSONAS["traffic"]["instructions"]
    assert "right turn on red" in instructions.lower(), (
        "Traffic persona must explicitly exclude right-turn-on-red from red-light alerts"
    )


def test_vehicle_description_anchors_type_to_yolo():
    instructions = PERSONAS["traffic"]["instructions"]
    assert "yolo" in instructions.lower(), (
        "Vehicle type must be anchored to the YOLO class label"
    )


def test_vehicle_description_color_fallback_terms():
    instructions = PERSONAS["traffic"]["instructions"]
    assert "light-colored" in instructions.lower(), (
        "Vehicle description must specify 'light-colored' as a color fallback term"
    )


def test_vehicle_description_lane_position():
    instructions = PERSONAS["traffic"]["instructions"]
    assert "lane position" in instructions.lower() or "lane" in instructions.lower(), (
        "Vehicle description must include lane position attribute"
    )


def test_industrial_persona_exists_with_the_standard_shape():
    p = PERSONAS["industrial"]
    assert p["name"]
    assert p["role"].strip()
    assert p["instructions"].strip()


def test_industrial_catalogue_names_all_four_categories():
    """Each category string must appear so the model can fill the `category` field."""
    text = PERSONAS["industrial"]["instructions"].lower()
    for category in ("safety", "security", "storage", "operations"):
        assert category in text


def test_industrial_catalogue_frames_itself_as_kinds_not_a_checklist():
    """The spec calls for open-ended reasoning; a literal checklist misses everything unlisted."""
    text = PERSONAS["industrial"]["instructions"].lower()
    assert "not a checklist" in text or "kinds" in text
    assert "spirit" in text


def test_industrial_catalogue_covers_the_in_scope_situations():
    """Spot-check the situations the spec puts in scope for the demo."""
    text = PERSONAS["industrial"]["instructions"].lower()
    for topic in ("fallen", "collision", "fire", "smoke", "restricted",
                  "stack", "pallet", "blocked"):
        assert topic in text, f"catalogue missing: {topic}"


def test_industrial_persona_explains_the_track_id_convention():
    """Boxes are drawn as `class confidence #id`; grounded alerts depend on the model
    citing them."""
    text = PERSONAS["industrial"]["instructions"]
    # Check for the distinctive explanation sentence, not just isolated symbols
    assert "detection boxes" in text.lower()
    assert "class confidence #id" in text


def test_industrial_persona_does_not_restate_severity_rules():
    """The rubric lives in ANALYZE_SCHEMA. Two statements of one rule would drift."""
    text = PERSONAS["industrial"]["instructions"].lower()
    assert "critical" not in text
    assert '"info"' not in text


def test_existing_personas_are_untouched():
    for pid in ("traffic", "indoor_security"):
        assert pid in PERSONAS
        assert PERSONAS[pid]["instructions"].strip()


def test_industrial_catalogue_excludes_removed_and_deferred_situations():
    """Spec §3 removed item 7 (PPE) and deferred items 9, 10, 11, 19, 20. A catalogue
    bullet for any of them promises behaviour the system does not implement.

    Matching is word-boundary anchored: a bare substring test for "ppe" also matches
    the word "appears", which is why the plain-substring version could not include it.
    """
    text = PERSONAS["industrial"]["instructions"].lower()
    banned = ("ppe", "helmet", "hi-vis", "high-vis", "loiter", "dwell", "unattended",
              "abandoned", "tamper", "overcrowd", "congestion", "wrong.way")
    for term in banned:
        assert not re.search(rf"\b{term}", text), \
            f"catalogue reintroduces a removed/deferred situation: {term!r}"


def test_shipped_config_selects_a_persona_that_exists():
    """A typo here silently falls back to DEFAULT_PERSONA at startup — no error."""
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    selected = cfg["monitoring"]["persona"]
    assert selected in PERSONAS, f"config.yaml selects unknown persona {selected!r}"


def test_shipped_config_selects_the_traffic_persona_and_the_vlm_preset_the_industrial_one():
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    assert cfg["monitoring"]["persona"] == "traffic"
    vlm = yaml.safe_load(open("configs/presets/vlm/config.yaml", encoding="utf-8"))
    assert vlm["monitoring"]["persona"] == "industrial"


def test_default_persona_constant_is_unchanged():
    """DEFAULT_PERSONA is the fallback for an unknown id, not the shipped choice."""
    assert DEFAULT_PERSONA == "traffic"


def test_create_app_bootstraps_the_configured_persona(tmp_path):
    """config.yaml's persona id must actually reach monitoring.active_persona and seed
    its instructions — regression coverage for the bootstrap logic at app/server.py:433-440."""
    cfg = {
        "perception": {"backend": "local"},
        "api": {"backend": "cloud", "api_key": "x", "model": "m"},
        "monitoring": {"mode": "proactive", "persona": "industrial"},
        "sampling": {},
        "events": {"db_path": str(tmp_path / "e.db")},
        "chat": {"tools_enabled": False},
        "cv": {"enabled": False},
        "camera": {"source": 0},
        "segments": {"enabled": False},
        "alerts": {"clip": {"enabled": False}},
        "zones": {"file_path": str(tmp_path / "z.txt"),
                  "sidecar_config_path": str(tmp_path / "s.yaml")},
    }
    create_app(cfg)
    assert cfg["monitoring"]["active_persona"] == "industrial"
    assert cfg["monitoring"]["instructions"] == PERSONAS["industrial"]["instructions"]


def test_create_app_falls_back_unknown_persona_to_default(tmp_path):
    """An unknown persona id in config must fall back to DEFAULT_PERSONA, not raise.
    This is how a typo fails silently in production — the test pins the behavior."""
    cfg = {
        "perception": {"backend": "local"},
        "api": {"backend": "cloud", "api_key": "x", "model": "m"},
        "monitoring": {"mode": "proactive", "persona": "nonexistent"},
        "sampling": {},
        "events": {"db_path": str(tmp_path / "e.db")},
        "chat": {"tools_enabled": False},
        "cv": {"enabled": False},
        "camera": {"source": 0},
        "segments": {"enabled": False},
        "alerts": {"clip": {"enabled": False}},
        "zones": {"file_path": str(tmp_path / "z.txt"),
                  "sidecar_config_path": str(tmp_path / "s.yaml")},
    }
    create_app(cfg)
    assert cfg["monitoring"]["active_persona"] == DEFAULT_PERSONA
    assert cfg["monitoring"]["instructions"] == PERSONAS[DEFAULT_PERSONA]["instructions"]
