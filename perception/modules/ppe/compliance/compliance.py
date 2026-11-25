# Vendored from ppe-compliance (/home/iris2/IRIS/ppe-compliance), commit b07c335, ppe_compliance/compliance.py.
# Rule changes must be made in BOTH places, here and in ppe-compliance (whose evaluation the numbers in
# docs/EVALUATION.md come from); see perception/modules/ppe/README.md.
# Unchanged from the source.
"""Per-frame, per-person PPE compliance decisions.

The decision for each required item is three-valued. ABSENT is only produced
from confident negative evidence on a person whose item region is fully in
the frame; anything uncertain (person too small, conflicting positive and
negative evidence, item region cut by the frame edge, low confidence, no
evidence at all) becomes UNKNOWN so it is never auto-escalated.

Order of the checks for one item (p / n = best positive / negative conf):
  1. person shorter than min_person_height_px      -> UNKNOWN person_too_small
  2. p and n both seen, one strong, within margin  -> UNKNOWN conflicting_evidence
  3. p >= present_conf and p > n                   -> PRESENT positive_detection
     (also on a truncated person: a seen helmet can't cause a false violation)
  4. item region reaches a sensitive frame edge    -> UNKNOWN person_truncated
     (before any negative evidence is used)
  5. n >= absent_conf, n > p, but some positive box and
     absent_requires_no_positive                   -> UNKNOWN conflicting_evidence
  6. n >= absent_conf and n > p                    -> ABSENT  negative_detection
  7. infer_absence_from_missing_positive (opt-in)  -> ABSENT  no_positive_detection
  8. some evidence below the thresholds            -> UNKNOWN low_confidence
  9. nothing                                       -> UNKNOWN no_evidence
"""
from __future__ import annotations

from .association import PersonContext, associate, item_region, select_persons, touched_edges
from .config import ComplianceConfig, PPEItemConfig
from .types import ComplianceStatus, Detection, ItemResult, ItemStatus, PersonResult


def evaluate_item(
    item: PPEItemConfig,
    ctx: PersonContext,
    evidence: list[Detection],
) -> ItemResult:
    pos = [d for d in evidence if d.class_name in item.positive_classes]
    neg = [d for d in evidence if d.class_name in item.negative_classes]
    p = max((d.confidence for d in pos), default=0.0)
    n = max((d.confidence for d in neg), default=0.0)

    def result(status: ItemStatus, reason: str) -> ItemResult:
        return ItemResult(item.name, status, reason, p, n, sorted(evidence, key=lambda d: -d.confidence))

    height = ctx.person.bbox[3] - ctx.person.bbox[1]
    if height < item.min_person_height_px:
        return result(ItemStatus.UNKNOWN, "person_too_small")

    strong_pos = p >= item.present_conf
    strong_neg = n >= item.absent_conf
    # Both kinds of evidence close together (e.g. helmet and bare head both
    # detected on the same person) -> don't guess.
    if (strong_pos or strong_neg) and p > 0 and n > 0 and abs(p - n) < item.conflict_margin:
        return result(ItemStatus.UNKNOWN, "conflicting_evidence")
    if strong_pos and p > n:
        return result(ItemStatus.PRESENT, "positive_detection")

    _, in_frame = item_region(ctx, item)
    if not in_frame:
        return result(ItemStatus.UNKNOWN, "person_truncated")
    if strong_neg and n > p:
        if item.absent_requires_no_positive and p > 0:
            return result(ItemStatus.UNKNOWN, "conflicting_evidence")
        return result(ItemStatus.ABSENT, "negative_detection")
    if item.infer_absence_from_missing_positive and p < item.missing_positive_max_conf:
        return result(ItemStatus.ABSENT, "no_positive_detection")
    if p > 0 or n > 0:
        return result(ItemStatus.UNKNOWN, "low_confidence")
    return result(ItemStatus.UNKNOWN, "no_evidence")


def overall_status(items: dict[str, ItemResult], required: list[str]) -> ComplianceStatus:
    statuses = [items[name].status for name in required]
    if any(s is ItemStatus.ABSENT for s in statuses):
        return ComplianceStatus.NON_COMPLIANT
    if statuses and all(s is ItemStatus.PRESENT for s in statuses):
        return ComplianceStatus.COMPLIANT
    return ComplianceStatus.UNKNOWN


def evaluate_frame(
    detections: list[Detection],
    cfg: ComplianceConfig,
    image_size: tuple[int, int] | None = None,
) -> tuple[list[PersonResult], list[Detection]]:
    """Evaluate every confidently detected person in one frame.

    Returns person results and PPE detections not attributed to any person.
    """
    persons, _ = select_persons(
        [d for d in detections if d.class_name in cfg.person_classes and d.confidence >= cfg.min_person_conf], cfg)
    contexts = [PersonContext(p, touched_edges(p.bbox, image_size, cfg.edge_margin_px)) for p in persons]
    assigned, unassigned = associate(contexts, detections, cfg)

    results = []
    for ctx, evidence in zip(contexts, assigned):
        items = {name: evaluate_item(cfg.item(name), ctx, evidence[name]) for name in cfg.required_items}
        results.append(PersonResult(ctx.person, items, overall_status(items, cfg.required_items)))
    return results, unassigned
