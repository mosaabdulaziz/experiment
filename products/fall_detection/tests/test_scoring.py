"""The scoring policy maths — no Redis, no torch, no GPU, no camera (AC-N7.1).

ACs: AC-F10.1, AC-F10.2, AC-F11.1, AC-F11.2, AC-F12.1, AC-F12.2, AC-F13.1, AC-F13.2,
AC-F14.1, AC-F15.1, AC-F30.3 (score half), AC-F35.1, AC-F35.2, AC-N7.1.
Formulas: FALL_DETECTION_DESIGN.md §4.2, §6.1-6.4, §13.3.
"""
from pathlib import Path

import pytest

import scoring
from fall_helpers import (AMBIGUOUS_FALL_LYING, DECISIVE_FALL, DECISIVE_LYING, POLICY, VENDOR_CFG,
                          make_rec, policy_with)

PRODUCT = Path(__file__).resolve().parents[1]


def _scoring_cfg():
    return POLICY["scoring"]


# ── the evaluation is detector-first (AC-F11.1, AC-F11.2) ──────────────────────────────────────────────────

def test_an_immobility_candidate_whose_top_class_is_a_decisive_fall_evaluates_as_falling():
    ev = _scoring_cfg()["evaluations"]
    assert scoring.evaluate("immobility", 6, ev) == "falling"
    assert scoring.evaluate("immobility", 3, ev) == "immobile"
    assert scoring.evaluate("immobility", 2, ev) == "immobile"


def test_a_fall_candidate_takes_the_group_its_top_class_falls_into():
    ev = _scoring_cfg()["evaluations"]
    assert scoring.evaluate("fall", 6, ev) == "falling"
    assert scoring.evaluate("fall", 3, ev) == "lying"
    assert scoring.evaluate("fall", 7, ev) == "unstable"
    assert scoring.evaluate("fall", 1, ev) == "upright"


def test_a_lying_candidate_evaluates_as_lying_whatever_the_reading_says():
    ev = _scoring_cfg()["evaluations"]
    assert scoring.evaluate("lying", 6, ev) == "lying"
    assert scoring.evaluate("lying", 0, ev) == "lying"


# ── the reading's margin (AC-F12.1, AC-F12.2) ──────────────────────────────────────────────────────────────

def test_a_classifier_that_cannot_tell_fall_from_lying_scores_near_zero():
    s = scoring.reading_score(AMBIGUOUS_FALL_LYING, "falling", _scoring_cfg())
    assert s <= 0.05


def test_a_decisive_reading_scores_near_one():
    s = scoring.reading_score(DECISIVE_FALL, "falling", _scoring_cfg())
    assert s >= 0.90


# ── the combination (AC-F13.1, AC-F13.2, AC-F14.1) ─────────────────────────────────────────────────────────

def test_a_cold_only_candidate_is_capped_below_the_threshold_and_delegated():
    cfg = _scoring_cfg()
    combined = scoring.combine({"action": None, "kinematic": None, "cold": 1.0}, cfg["trust"])
    assert combined == pytest.approx(0.40)
    severity, review = scoring.route("falling", combined, cfg["confirm_threshold"])
    assert severity == "warning" and review is True


def test_a_decisive_action_only_candidate_confirms_on_its_own():
    cfg = _scoring_cfg()
    combined = scoring.combine({"action": 0.95, "kinematic": None, "cold": None}, cfg["trust"])
    assert combined == pytest.approx(0.95)
    severity, review = scoring.route("falling", combined, cfg["confirm_threshold"])
    assert severity == "critical" and review is False


def test_a_path_that_did_not_speak_leaves_its_trust_unclaimed():
    cfg = _scoring_cfg()
    assert scoring.combine({"action": 0.5, "kinematic": None, "cold": None}, cfg["trust"]) == pytest.approx(0.5)
    both = scoring.combine({"action": 0.5, "kinematic": 1.0, "cold": None}, cfg["trust"])
    assert both == pytest.approx(1.0 - 0.5 * 0.5)          # 0.75, no renormalization


def test_the_trust_weights_come_from_the_config_not_the_code():
    cfg = policy_with(trust={"action": 0.60})["scoring"]
    combined = scoring.combine({"action": 1.0, "kinematic": None, "cold": None}, cfg["trust"])
    assert combined == pytest.approx(0.60)
    assert scoring.route("falling", combined, cfg["confirm_threshold"])[1] is True   # delegated


# ── the class sets live in the product's config only (AC-F15.1) ────────────────────────────────────────────

def test_the_evaluation_class_sets_exist_only_in_the_products_config():
    import re
    group = re.compile(r"\b(falling|lying|immobile|unstable|upright)\s*:\s*\[")
    policy_yaml = (PRODUCT / "fall_detection.yaml").read_text()
    for name in ("falling", "lying", "immobile", "unstable", "upright"):
        assert re.search(rf"\b{name}\s*:\s*\[", policy_yaml), f"the {name} group is not in the policy yaml"
    # the mapping — not the English word: the pristine vendor source legitimately says "falling"
    # in labels and comments, and AC-F7.1 requires most of it to stay byte-identical
    vendor_dir = PRODUCT / "fall_detection"
    offenders = [str(p.relative_to(PRODUCT)) for p in vendor_dir.rglob("*")
                 if p.suffix in (".py", ".yaml") and group.search(p.read_text(errors="ignore"))]
    assert offenders == []
    assert not group.search((PRODUCT / "scoring.py").read_text())   # nor a second copy in the code


# ── the physical scores read the gates' own thresholds (AC-F10.1, AC-F10.2) ────────────────────────────────

# the hip quantity runs the other way: `hip_rel` is the hip's height over the box's bottom
# (fall_monitor.py:226), the gate fires when it falls *below* `low_hip`, and design §4.2's
# s_hip = clip((hip_max - hip_rel) / hip_span) is inverted to match — so a fully collapsed
# person reads hip_rel = 0.0, not 2x the threshold.
KIN_COLLAPSE = {"drop": 0.28, "vdown": 0.32, "hip_rel": 0.0, "angle": 80.0, "aspect": 1.8}


def test_the_collapse_score_moves_with_the_gate_block_it_reads():
    gate = dict(VENDOR_CFG["path_b"])
    at_double = scoring.collapse_score(KIN_COLLAPSE, gate, scoring.default_spans(gate))
    assert at_double == pytest.approx(1.0)                   # every quantity at its span's end
    gate["drop"] = 0.28                                      # now drop sits exactly at the threshold
    assert scoring.collapse_score(KIN_COLLAPSE, gate, scoring.default_spans(gate)) == pytest.approx(0.0)


def test_one_collapsed_condition_cannot_be_averaged_away():
    gate = VENDOR_CFG["path_b"]
    spans = scoring.default_spans(gate)
    kin = dict(KIN_COLLAPSE, vdown=gate["vdown"])            # vdown exactly at the gate: s_vdown = 0
    assert scoring.collapse_score(kin, gate, spans) == pytest.approx(0.0)


def test_the_collapse_aggregation_is_the_geometric_mean_of_design_4_2():
    # s_collapse = (s_drop · s_vdown · s_hip · s_posture) ** 0.25 — pinned at an interior point,
    # where geometric mean (0.632), min (0.16) and arithmetic mean (0.79) all differ
    gate = VENDOR_CFG["path_b"]
    spans = scoring.default_spans(gate)
    kin = dict(KIN_COLLAPSE, vdown=gate["vdown"] + 0.16 * spans["vdown_span"])   # s_vdown = 0.16
    assert scoring.collapse_score(kin, gate, spans) == pytest.approx(0.16 ** 0.25, abs=1e-6)


def test_all_seven_spans_default_to_their_gates_own_threshold_and_are_overridable():
    gate = dict(VENDOR_CFG["path_b"])
    spans = scoring.default_spans(gate)
    for name, source in (("drop_span", "drop"), ("vdown_span", "vdown"), ("hip_span", "low_hip"),
                         ("angle_span", "angle"), ("aspect_span", "aspect")):
        assert spans[name] == pytest.approx(gate[source])
    lying_spans = scoring.default_spans(VENDOR_CFG["lying"])
    assert lying_spans["motion_span"] == pytest.approx(VENDOR_CFG["lying"]["max_motion"])
    imm_spans = scoring.default_spans(VENDOR_CFG["immobility"])
    assert imm_spans["disp_span"] == pytest.approx(VENDOR_CFG["immobility"]["disp_thresh"])
    overridden = scoring.collapse_score(KIN_COLLAPSE, gate, dict(spans, drop_span=10.0))
    assert overridden < 1.0                                  # the override changed the score


def test_all_seven_spans_sit_in_the_vendors_config_yaml_beside_their_gates():
    # ruling A6: the spans live in config.yaml next to the gates they measure — overridable
    # without a code change means the file, not the default_spans fallback, carries them
    import yaml
    cfg_path = PRODUCT / "config.yaml"
    assert cfg_path.is_file(), "config.yaml (the vendor's master config + the spans) does not exist yet"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    blocks = {name: cfg[name] for name in ("path_b", "path_c", "lying", "immobility")
              if isinstance(cfg.get(name), dict)}
    found = {key for block in blocks.values() for key in block if key.endswith("_span")}
    seven = {"drop_span", "vdown_span", "hip_span", "angle_span", "aspect_span",
             "motion_span", "disp_span"}
    assert seven <= found, f"missing spans: {sorted(seven - found)}"
    gate_keys = {"drop_span": ("drop",), "vdown_span": ("vdown",), "hip_span": ("low_hip",),
                 "angle_span": ("angle", "angle_in"), "aspect_span": ("aspect", "aspect_in"),
                 "motion_span": ("max_motion",), "disp_span": ("disp_thresh",)}
    for name, block in blocks.items():
        for span, value in block.items():
            if span not in gate_keys:
                continue
            gate_key = next(k for k in gate_keys[span] if k in block)
            assert value == pytest.approx(block[gate_key]), \
                f"{name}.{span} must default to the gate's own threshold ({name}.{gate_key})"


def test_the_immobility_physical_score_is_the_stillness_margin():
    cfg = VENDOR_CFG["immobility"]
    spans = scoring.default_spans(cfg)
    assert scoring.stillness_score(0.0, cfg, spans) == pytest.approx(1.0)
    assert scoring.stillness_score(cfg["disp_thresh"], cfg, spans) == pytest.approx(0.0)
    assert scoring.stillness_score(cfg["disp_thresh"] * 2, cfg, spans) == pytest.approx(0.0)


# ── the lying physical score (AC-F30.3 score half) ─────────────────────────────────────────────────────────

def test_the_lying_score_is_the_geometric_mean_of_posture_and_stillness():
    cfg = VENDOR_CFG["lying"]
    spans = scoring.default_spans(cfg)
    strong = {"angle": cfg["angle_in"] * 2, "aspect": cfg["aspect_in"] * 2, "motion": 0.0}
    assert scoring.lying_score(strong, cfg, spans) == pytest.approx(1.0)
    moving = dict(strong, motion=cfg["max_motion"])
    assert scoring.lying_score(moving, cfg, spans) == pytest.approx(0.0)
    half = {"angle": cfg["angle_in"] * 1.5, "aspect": cfg["aspect_in"] * 2,
            "motion": cfg["max_motion"] / 2}
    # s_posture = min((52.5-35)/35, (1.0-0.5)/0.5) = 0.5 ; s_still = (1.0-0.5)/1.0 = 0.5
    assert scoring.lying_score(half, cfg, spans) == pytest.approx(0.5)


# ── the action slot rule (AC-F35.1, AC-F35.2) ──────────────────────────────────────────────────────────────

def test_a_path_b_fall_with_a_window_and_a_decisive_reading_can_reach_the_threshold():
    rec = make_rec(probs=DECISIVE_FALL, kinematic=0.9)       # latched by Path B only: action None
    scored = scoring.score_candidate(rec, POLICY, VENDOR_CFG)
    assert scored["scores"]["action"] is not None and scored["scores"]["action"] >= 0.90
    assert scored["combined"] >= POLICY["scoring"]["confirm_threshold"]


def test_a_candidate_with_no_window_never_reaches_the_threshold():
    for rec in (make_rec(probs=None, cold=1.0), make_rec(probs=None, kinematic=1.0),
                make_rec(kind="lying", probs=None)):
        scored = scoring.score_candidate(rec, POLICY, VENDOR_CFG)
        assert scored["scores"]["action"] is None
        # no reading => no corroboration => the physical score falls to the cold ceiling (0.40),
        # never the kinematic one (0.50): which slot spoke is part of the contract
        assert scored["scores"]["kinematic"] is None
        assert scored["scores"]["cold"] is not None
        assert scored["combined"] < POLICY["scoring"]["confirm_threshold"]


def test_a_lying_candidate_is_scored_from_its_emission_moment_kinematics():
    kin = {"angle": 70.0, "aspect": 1.4, "motion": 0.1}
    rec = make_rec(kind="lying", probs=DECISIVE_LYING, kinematics=kin)
    scored = scoring.score_candidate(rec, POLICY, VENDOR_CFG)
    cfg = VENDOR_CFG["lying"]
    physical = scoring.lying_score(kin, cfg, scoring.default_spans(cfg))
    assert scored["evaluation"] == "lying"
    assert scored["scores"]["kinematic"] == pytest.approx(physical)   # window existed: kinematic trust
    assert scored["scores"]["cold"] is None


# ── N7.1: the policy module itself stays pure ───────────────────────────────────────────────────────────────

def test_the_scoring_module_imports_no_torch_no_redis_and_no_opencv():
    import re
    src = (PRODUCT / "scoring.py").read_text()
    for forbidden in ("torch", "redis", "cv2", "socket", "fakeredis"):
        assert not re.search(rf"^\s*(import|from)\s+{forbidden}\b", src, re.M), \
            f"scoring.py imports {forbidden}"


# ── the alert's words: what an operator reads on the incident and in Telegram ────────────────────────────────

def test_a_confirmed_alert_says_what_happened_in_plain_words_with_its_confidence():
    assert scoring.alert_statement("falling", 0.823, review=False) == "Person fell (confidence 82%)"
    assert scoring.alert_statement("immobile", 0.9, review=False) == "Person down and not moving (confidence 90%)"
    assert scoring.alert_statement("lying", 0.75, review=False) == "Person lying on the floor (confidence 75%)"
    assert scoring.alert_statement("unstable", 0.7, review=False) == \
        "Person unsteady, at risk of falling (confidence 70%)"


def test_an_unconfident_alert_says_the_fall_is_only_possible_and_needs_a_review():
    # what happened first: a narrow card (the console's banner) cuts the end off
    assert scoring.alert_statement("falling", 0.39, review=True) == \
        "A person may have fallen: needs review (confidence 39%)"
    assert scoring.alert_statement("lying", 0.0, review=True) == \
        "A person may be lying on the floor: needs review (confidence 0%)"
    assert scoring.alert_statement("upright", 0.314, review=True) == \
        "A person may have fallen and got up again: needs review (confidence 31%)"   # says a fall was suspected


def test_every_evaluation_the_policy_knows_has_its_own_words_not_the_bare_label():
    for evaluation in scoring.EVALUATIONS:
        for review in (False, True):
            assert f" {evaluation} (" not in scoring.alert_statement(evaluation, 0.5, review=review)
