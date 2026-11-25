"""The ppe-compliance rule tests, ported to the vendored copy in perception/modules/ppe/compliance/.

Copied from ppe-compliance (/home/iris2/IRIS/ppe-compliance), commit b07c335, tests/test_compliance.py: the 34 tests of
association, per-person decisions and temporal smoothing, with their assertions unchanged. That is the check that the
vendored copy behaves like the evaluated one. Only the imports and the `cfg` fixture differ (the configs are built
from the vendored dataclasses instead of the YAML loader, which is not vendored). Not ported, because the code they
test stays in ppe-compliance or is replaced in IRIS: tracker and stitching, events, visualisation, model profiles,
weights download, the person-level evaluation and the config loader (the module's Settings replaces it).
Rule changes must be made in BOTH places; see perception/modules/ppe/README.md.
"""
import types

import pytest

from perception.modules.ppe.compliance.compliance import evaluate_frame
from perception.modules.ppe.compliance.config import ComplianceConfig, TemporalConfig
from perception.modules.ppe.compliance.temporal import TemporalFilter
from perception.modules.ppe.compliance.types import ComplianceStatus, Detection, ItemStatus


@pytest.fixture
def cfg():
    # ppe-compliance's load_config() with no file: the code defaults of its compliance and temporal sections
    return types.SimpleNamespace(compliance=ComplianceConfig(), temporal=TemporalConfig())


IMG = (1000, 1000)
PERSON = (100, 100, 200, 400)  # 100 x 300 px person
HEAD = (125, 85, 175, 160)     # helmet/head area
TORSO = (110, 170, 190, 300)   # vest area


def det(cls, conf, box):
    return Detection(cls, conf, box)


def person_result(dets, cfg, image_size=IMG):
    persons, unassigned = evaluate_frame(dets, cfg.compliance, image_size)
    assert len(persons) == 1
    return persons[0], unassigned


def test_compliant_when_both_items_clearly_present(cfg):
    p, _ = person_result([det("Person", 0.9, PERSON), det("Hardhat", 0.8, HEAD),
                          det("Safety Vest", 0.7, TORSO)], cfg)
    assert p.items["helmet"].status is ItemStatus.PRESENT
    assert p.items["vest"].status is ItemStatus.PRESENT
    assert p.status is ComplianceStatus.COMPLIANT


def test_non_compliant_on_confident_negative(cfg):
    p, _ = person_result([det("Person", 0.9, PERSON), det("NO-Hardhat", 0.7, HEAD),
                          det("Safety Vest", 0.7, TORSO)], cfg)
    assert p.items["helmet"].status is ItemStatus.ABSENT
    assert p.status is ComplianceStatus.NON_COMPLIANT
    assert p.missing_items == ["helmet"]


def test_no_evidence_is_unknown_not_violation(cfg):
    p, _ = person_result([det("Person", 0.9, PERSON)], cfg)
    assert p.items["helmet"].status is ItemStatus.UNKNOWN
    assert p.items["helmet"].reason == "no_evidence"
    assert p.status is ComplianceStatus.UNKNOWN


def test_low_confidence_negative_is_unknown(cfg):
    p, _ = person_result([det("Person", 0.9, PERSON), det("NO-Hardhat", 0.2, HEAD)], cfg)
    assert p.items["helmet"].status is ItemStatus.UNKNOWN
    assert p.items["helmet"].reason == "low_confidence"


def test_conflicting_evidence_is_unknown(cfg):
    p, _ = person_result([det("Person", 0.9, PERSON), det("Hardhat", 0.55, HEAD),
                          det("NO-Hardhat", 0.50, HEAD)], cfg)
    assert p.items["helmet"].status is ItemStatus.UNKNOWN
    assert p.items["helmet"].reason == "conflicting_evidence"


def test_clear_winner_over_weak_conflict(cfg):
    p, _ = person_result([det("Person", 0.9, PERSON), det("Hardhat", 0.85, HEAD),
                          det("NO-Hardhat", 0.2, HEAD)], cfg)
    assert p.items["helmet"].status is ItemStatus.PRESENT


def test_small_person_is_unknown(cfg):
    p, _ = person_result([det("Person", 0.9, (10, 10, 20, 40)), det("NO-Hardhat", 0.9, (11, 9, 19, 18))], cfg)
    assert p.items["helmet"].reason == "person_too_small"
    assert p.status is ComplianceStatus.UNKNOWN


def test_head_cut_off_at_top_gives_helmet_unknown(cfg):
    # Even with absence inference switched on, a head cut by the top edge is unknown.
    cfg.compliance.item("helmet").infer_absence_from_missing_positive = True
    p, _ = person_result([det("Person", 0.9, (100, 0, 200, 300))], cfg)
    assert p.items["helmet"].status is ItemStatus.UNKNOWN
    assert p.items["helmet"].reason == "person_truncated"


def test_infer_absence_option(cfg):
    cfg.compliance.item("vest").infer_absence_from_missing_positive = True
    p, _ = person_result([det("Person", 0.9, PERSON), det("Hardhat", 0.8, HEAD)], cfg)
    assert p.items["vest"].status is ItemStatus.ABSENT
    assert p.items["vest"].reason == "no_positive_detection"


def test_required_items_configurable(cfg):
    cfg.compliance.required_items = ["helmet"]
    p, _ = person_result([det("Person", 0.9, PERSON), det("Hardhat", 0.8, HEAD)], cfg)
    assert list(p.items) == ["helmet"]
    assert p.status is ComplianceStatus.COMPLIANT


def test_low_confidence_person_ignored(cfg):
    persons, _ = evaluate_frame([det("Person", 0.2, PERSON)], cfg.compliance, IMG)
    assert persons == []


def test_ppe_assigned_to_correct_person_and_unassigned_reported(cfg):
    other = (220, 100, 320, 400)
    dets = [det("Person", 0.9, PERSON), det("Person", 0.9, other),
            det("NO-Hardhat", 0.8, (245, 85, 295, 160)),  # head of the second person
            det("Hardhat", 0.8, HEAD),
            det("Hardhat", 0.8, (700, 700, 740, 730))]    # nobody there
    persons, unassigned = evaluate_frame(dets, cfg.compliance, IMG)
    assert persons[0].items["helmet"].status is ItemStatus.PRESENT
    assert persons[1].items["helmet"].status is ItemStatus.ABSENT
    assert [d.bbox for d in unassigned] == [(700, 700, 740, 730)]


def test_helmet_on_torso_not_counted(cfg):
    p, unassigned = person_result([det("Person", 0.9, PERSON), det("Hardhat", 0.9, (120, 300, 180, 360))], cfg)
    assert p.items["helmet"].status is ItemStatus.UNKNOWN
    assert len(unassigned) == 1


def _frame(cfg, helmet_cls, conf=0.8):
    persons, _ = evaluate_frame([det("Person", 0.9, PERSON), det(helmet_cls, conf, HEAD),
                                 det("Safety Vest", 0.8, TORSO)], cfg.compliance, IMG)
    persons[0].track_id = 1
    return persons


def test_temporal_requires_persistent_violation_and_dedupes(cfg):
    cfg.temporal.window, cfg.temporal.min_absent_frames, cfg.temporal.min_present_frames = 5, 3, 2
    cfg.temporal.cooldown_s = 10
    tf = TemporalFilter(cfg.temporal, cfg.compliance.required_items)
    new = []
    seq = ["NO-Hardhat", "Hardhat", "NO-Hardhat", "NO-Hardhat", "NO-Hardhat", "NO-Hardhat"]
    for i, cls in enumerate(seq):
        sp = tf.update(_frame(cfg, cls), timestamp_s=i)[0]
        new.append(sp.new_violations)
    # first noisy frame alone doesn't fire; fires once at the 3rd absent, then no repeats
    assert new == [[], [], [], ["helmet"], [], []]
    assert sp.status is ComplianceStatus.NON_COMPLIANT


def test_temporal_new_episode_after_compliance_and_cooldown(cfg):
    cfg.temporal.window, cfg.temporal.min_absent_frames, cfg.temporal.min_present_frames = 3, 2, 2
    cfg.temporal.cooldown_s = 5
    tf = TemporalFilter(cfg.temporal, cfg.compliance.required_items)
    fired = []
    seq = ["NO-Hardhat"] * 2 + ["Hardhat"] * 3 + ["NO-Hardhat"] * 2 + ["Hardhat"] * 3 + ["NO-Hardhat"] * 2
    for i, cls in enumerate(seq):
        if tf.update(_frame(cfg, cls), timestamp_s=i)[0].new_violations:
            fired.append(i)
    # each violation episode fires once; episodes are closed by PRESENT and spaced >= cooldown
    assert fired == [1, 6, 11]


def test_contested_vest_goes_to_smaller_person_when_big_box_has_its_own(cfg):
    # Frame 291 of test/test.mp4 (1920x1080), real boxes: an oversized person
    # box (touching the top/left edge) overlaps a smaller person. The small
    # person's vest is 100% inside the big box's region but only 88% inside
    # its own torso band, so coverage alone gives it to the big box.
    big = (2, 0, 974, 1072)
    small = (578, 28, 963, 1051)
    small_vest = (644, 276, 922, 865)
    big_vest = (186, 67, 571, 950)
    dets = [det("Person", 0.54, big), det("Person", 0.47, small),
            det("Safety Vest", 0.80, small_vest), det("Safety Vest", 0.58, big_vest)]
    persons, unassigned = evaluate_frame(dets, cfg.compliance, (1920, 1080))
    by_box = {p.person.bbox: p for p in persons}
    assert [d.bbox for d in by_box[small].items["vest"].evidence] == [small_vest]
    assert [d.bbox for d in by_box[big].items["vest"].evidence] == [big_vest]
    assert unassigned == []


def test_side_by_side_people_keep_their_own_vest(cfg):
    # Overlapping neighbours: each vest also covers >= min_overlap of the other's torso band.
    a, b = (100, 100, 200, 400), (160, 100, 260, 400)
    vest_a, vest_b = (120, 170, 200, 300), (160, 170, 240, 300)
    dets = [det("Person", 0.9, a), det("Person", 0.9, b),
            det("Safety Vest", 0.6, vest_a), det("Safety Vest", 0.9, vest_b)]
    persons, unassigned = evaluate_frame(dets, cfg.compliance, IMG)
    by_box = {p.person.bbox: p for p in persons}
    assert [d.bbox for d in by_box[a].items["vest"].evidence] == [vest_a]
    assert [d.bbox for d in by_box[b].items["vest"].evidence] == [vest_b]
    assert unassigned == []


def test_one_box_per_class_per_person(cfg):
    # One Hardhat and one NO-Hardhat may both stay on a person (so conflicting
    # evidence can be weighed); a second Hardhat on the same head is left over.
    p, unassigned = person_result([det("Person", 0.9, PERSON), det("Hardhat", 0.85, HEAD),
                                   det("NO-Hardhat", 0.3, HEAD), det("Hardhat", 0.4, (130, 90, 170, 150))], cfg)
    assert sorted((d.class_name, d.confidence) for d in p.items["helmet"].evidence) ==         [("Hardhat", 0.85), ("NO-Hardhat", 0.3)]
    assert [(d.class_name, d.confidence) for d in unassigned] == [("Hardhat", 0.4)]


def test_duplicate_upper_body_box_dropped(cfg):
    # Frame 19 of test/test.mp4: a bent-over worker detected twice, an
    # upper-body box inside the full-body box.
    full, upper = (867, 238, 1137, 1031), (866, 238, 1124, 580)
    dets = [det("Person", 0.38, full), det("Person", 0.64, upper),
            det("Hardhat", 0.95, (899, 240, 1044, 362)), det("Safety Vest", 0.74, (979, 345, 1126, 588))]
    persons, unassigned = evaluate_frame(dets, cfg.compliance, (1920, 1080))
    assert [p.person.bbox for p in persons] == [full]
    assert persons[0].items["helmet"].status is ItemStatus.PRESENT
    assert unassigned == []


def test_partially_overlapping_people_both_kept(cfg):
    a, b = (100, 100, 200, 400), (150, 120, 250, 420)
    persons, _ = evaluate_frame([det("Person", 0.9, a), det("Person", 0.9, b)], cfg.compliance, IMG)
    assert sorted(p.person.bbox for p in persons) == [a, b]


def test_person_inside_box_around_two_people_kept(cfg):
    # Frame 85: one box covers two workers side by side; the box of one of them
    # lies fully inside it but is much narrower, so it is not a duplicate.
    group, one = (1269, 83, 1792, 950), (1269, 84, 1599, 903)
    persons, _ = evaluate_frame([det("Person", 0.5, group), det("Person", 0.47, one)],
                                cfg.compliance, (1920, 1080))
    assert sorted(p.person.bbox for p in persons) == sorted([group, one])


TOP_CUT = (100, 0, 200, 300)      # head cut off by the top edge
BOTTOM_CUT = (100, 700, 200, 1000)  # legs cut off by the bottom edge (IMG is 1000 high)
LEFT_CUT = (0, 100, 100, 400)


def test_truncated_person_with_confident_no_hardhat_is_unknown(cfg):
    # Known issue #1: truncation is checked before any negative detection.
    p, _ = person_result([det("Person", 0.9, TOP_CUT), det("NO-Hardhat", 0.9, (125, 0, 175, 60))], cfg)
    assert p.items["helmet"].status is ItemStatus.UNKNOWN
    assert p.items["helmet"].reason == "person_truncated"
    assert p.status is not ComplianceStatus.NON_COMPLIANT


def test_truncated_person_search_region_not_widened(cfg):
    # A NO-Hardhat box at torso height stays outside the head band of a truncated person.
    p, unassigned = person_result([det("Person", 0.9, TOP_CUT), det("NO-Hardhat", 0.9, (125, 150, 175, 210))], cfg)
    assert p.items["helmet"].evidence == []
    assert [d.class_name for d in unassigned] == ["NO-Hardhat"]


def test_confident_helmet_on_truncated_person_is_present(cfg):
    p, _ = person_result([det("Person", 0.9, TOP_CUT), det("Hardhat", 0.8, (125, 0, 175, 60))], cfg)
    assert p.items["helmet"].status is ItemStatus.PRESENT


def test_legs_cut_off_vest_still_decided(cfg):
    p, _ = person_result([det("Person", 0.9, BOTTOM_CUT), det("NO-Safety Vest", 0.8, (110, 760, 190, 900)),
                          det("Hardhat", 0.8, (125, 690, 175, 760))], cfg)
    assert p.items["vest"].status is ItemStatus.ABSENT
    assert p.items["vest"].reason == "negative_detection"
    assert p.items["helmet"].status is ItemStatus.PRESENT


def test_head_cut_off_vest_still_decided(cfg):
    p, _ = person_result([det("Person", 0.9, TOP_CUT), det("NO-Safety Vest", 0.8, (110, 50, 190, 200))], cfg)
    assert p.items["vest"].status is ItemStatus.ABSENT


def test_person_at_left_edge_never_absent(cfg):
    p, _ = person_result([det("Person", 0.9, LEFT_CUT), det("NO-Hardhat", 0.9, (25, 85, 75, 160)),
                          det("NO-Safety Vest", 0.9, (10, 170, 90, 300))], cfg)
    assert p.items["helmet"].reason == "person_truncated"
    assert p.items["vest"].reason == "person_truncated"


def test_sensitive_edges_configurable(cfg):
    cfg.compliance.item("vest").sensitive_edges = ["top", "bottom"]
    p, _ = person_result([det("Person", 0.9, LEFT_CUT), det("NO-Safety Vest", 0.9, (10, 170, 90, 300))], cfg)
    assert p.items["vest"].status is ItemStatus.ABSENT


def test_vest_absent_only_without_any_safety_vest_box(cfg):
    # Frame 103 (open vest over a dark shirt): NO-Safety Vest 0.81 next to Safety Vest 0.26.
    p, _ = person_result([det("Person", 0.9, PERSON), det("NO-Safety Vest", 0.81, TORSO),
                          det("Safety Vest", 0.26, TORSO)], cfg)
    assert p.items["vest"].status is ItemStatus.UNKNOWN
    assert p.items["vest"].reason == "conflicting_evidence"
    p, _ = person_result([det("Person", 0.9, PERSON), det("NO-Safety Vest", 0.81, TORSO)], cfg)
    assert p.items["vest"].status is ItemStatus.ABSENT


def test_absent_requires_no_positive_is_a_helmet_switch(cfg):
    dets = [det("Person", 0.9, PERSON), det("NO-Hardhat", 0.8, HEAD), det("Hardhat", 0.2, HEAD)]
    p, _ = person_result(dets, cfg)
    assert p.items["helmet"].status is ItemStatus.ABSENT          # off for the helmet by default
    cfg.compliance.item("helmet").absent_requires_no_positive = True
    p, _ = person_result(dets, cfg)
    assert p.items["helmet"].reason == "conflicting_evidence"


FRAME291_GROUP = (2, 0, 974, 1072)      # box around two workers
FRAME291_LEFT = (0, 3, 576, 1063)
FRAME291_CENTER = (578, 28, 963, 1051)


def test_box_around_two_workers_dropped(cfg):
    dets = [det("Person", 0.54, FRAME291_GROUP), det("Person", 0.82, FRAME291_LEFT),
            det("Person", 0.47, FRAME291_CENTER)]
    persons, _ = evaluate_frame(dets, cfg.compliance, (1920, 1080))
    assert sorted(p.person.bbox for p in persons) == sorted([FRAME291_LEFT, FRAME291_CENTER])
    cfg.compliance.group_min_members = 0
    persons, _ = evaluate_frame(dets, cfg.compliance, (1920, 1080))
    assert len(persons) == 3


def test_full_body_box_with_own_duplicate_not_a_group(cfg):
    # The upper-body duplicate is removed first, so the full-body box contains
    # only one other box and is kept.
    full, upper = (867, 238, 1137, 1031), (866, 238, 1124, 580)
    other = (900, 600, 1000, 1000)  # another (small) person inside the full box
    persons, _ = evaluate_frame([det("Person", 0.4, full), det("Person", 0.6, upper),
                                 det("Person", 0.5, other)], cfg.compliance, (1920, 1080))
    assert full in [p.person.bbox for p in persons]


def test_absent_needs_six_of_last_eight_frames(cfg):
    assert (cfg.temporal.window, cfg.temporal.min_absent_frames) == (8, 6)
    tf = TemporalFilter(cfg.temporal, cfg.compliance.required_items)
    fired = [i for i in range(8) if tf.update(_frame(cfg, "NO-Hardhat"), timestamp_s=i)[0].new_violations]
    assert fired == [5]  # the 6th absent frame


def test_short_absent_burst_gives_no_event(cfg):
    # Like the open-vest worker in frames 109-112: 4 absent frames between unknown ones.
    tf = TemporalFilter(cfg.temporal, cfg.compliance.required_items)
    unknown = lambda: _frame(cfg, "NO-Hardhat", conf=0.2)  # low confidence -> unknown
    seq = [unknown] * 2 + [lambda: _frame(cfg, "NO-Hardhat")] * 4 + [unknown] * 6
    assert not any(tf.update(f(), timestamp_s=i)[0].new_violations for i, f in enumerate(seq))
