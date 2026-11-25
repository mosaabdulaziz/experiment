# Vendored from ppe-compliance (/home/iris2/IRIS/ppe-compliance), commit b07c335, ppe_compliance/association.py.
# Rule changes must be made in BOTH places, here and in ppe-compliance (whose evaluation the numbers in
# docs/EVALUATION.md come from); see perception/modules/ppe/README.md.
# Unchanged from the source.
"""Attach PPE detections to the person they belong to.

Duplicate person boxes (an upper-body box inside a full-body box of the same
worker) and boxes drawn around several workers are dropped first, see
`select_persons`.

Then one-to-one per item and polarity: each person gets at most one positive
and one negative box per item (one Hardhat + one NO-Hardhat, one Safety Vest +
one NO-Safety Vest), so positive and negative evidence on the same person can
still be weighed against each other (conflicting_evidence). Each box goes to
at most one person. Candidate pairs are those where the person's expected
region for the item (head band, torso band) covers at least `min_overlap` of
the box. Pairs are taken greedily per frame, best first, by:
  1. IoU of box and region (rounded to 0.01) - how well the box fits the
     region. Not the covered fraction alone: an oversized person box covers
     everything near it and would win every contested box,
  2. box confidence (a person keeps its most confident box),
  3. pixel distance between box centre and region centre (closer wins),
  4. region area (smaller wins).
The person and the box of a taken pair are removed; boxes left over are
reported as unassigned.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import ComplianceConfig, PPEItemConfig
from .geometry import area, ioa, iou, person_region
from .types import BBox, Detection


@dataclass
class PersonContext:
    person: Detection
    touched_edges: set[str]


def touched_edges(box: BBox, image_size: tuple[int, int] | None, margin: float) -> set[str]:
    if image_size is None:
        return set()
    w, h = image_size
    x1, y1, x2, y2 = box
    edges = set()
    if y1 <= margin:
        edges.add("top")
    if y2 >= h - margin:
        edges.add("bottom")
    if x1 <= margin:
        edges.add("left")
    if x2 >= w - margin:
        edges.add("right")
    return edges


def region_truncated_edges(ctx: PersonContext, item: PPEItemConfig) -> set[str]:
    """Sensitive frame edges reached by the item's region limited to the person
    box. The band spans the full box width, so it reaches left/right whenever
    the box does; it reaches top/bottom only if the band extends to that end of
    the box (head band: top; torso band: neither)."""
    reached = set(ctx.touched_edges) & {"left", "right"}
    if "top" in ctx.touched_edges and item.region_top <= 0.0:
        reached.add("top")
    if "bottom" in ctx.touched_edges and item.region_bottom >= 1.0:
        reached.add("bottom")
    return reached & set(item.sensitive_edges)


def item_region(ctx: PersonContext, item: PPEItemConfig) -> tuple[BBox, bool]:
    """Expected region of `item` on this person, and whether it is fully inside
    the frame (False = truncated). The region is the same either way: a
    truncated person is not searched more widely."""
    region = person_region(ctx.person.bbox, item.region_top, item.region_bottom, item.region_pad_x)
    return region, not region_truncated_edges(ctx, item)


def _center(b: BBox) -> tuple[float, float]:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def drop_duplicate_persons(persons: list[Detection], cfg: ComplianceConfig) -> list[Detection]:
    """Drop a person box that lies (almost) entirely inside a larger, horizontally
    aligned person box: the same worker detected twice. Keeps input order."""
    order = sorted(range(len(persons)), key=lambda i: -area(persons[i].bbox))
    kept: list[int] = []
    for i in order:
        b = persons[i].bbox
        duplicate = False
        for j in kept:
            big = persons[j].bbox
            if ioa(b, big) < cfg.duplicate_min_inside:
                continue
            x_inter = min(b[2], big[2]) - max(b[0], big[0])
            x_union = max(b[2], big[2]) - min(b[0], big[0])
            if x_union > 0 and x_inter / x_union >= cfg.duplicate_min_x_overlap:
                duplicate = True
                break
        if not duplicate:
            kept.append(i)
    return [persons[i] for i in sorted(kept)]


def drop_group_persons(persons: list[Detection], cfg: ComplianceConfig) -> list[Detection]:
    """Drop a person box drawn around several workers: one that contains at
    least `group_min_members` other person boxes, each with at least
    `duplicate_min_inside` of its area inside. Keeps input order."""
    if cfg.group_min_members <= 0:
        return list(persons)
    return [p for p in persons
            if sum(1 for q in persons if q is not p and area(q.bbox) < area(p.bbox)
                   and ioa(q.bbox, p.bbox) >= cfg.duplicate_min_inside) < cfg.group_min_members]


def select_persons(persons: list[Detection], cfg: ComplianceConfig) -> tuple[list[Detection], dict[int, str]]:
    """Person boxes to evaluate, plus {id(dropped box): reason} for the rest.
    Duplicates go first, so a full-body box with its own upper-body duplicate
    isn't mistaken for a box around two people."""
    unique = drop_duplicate_persons(persons, cfg)
    kept = drop_group_persons(unique, cfg)
    kept_ids = {id(p) for p in kept}
    unique_ids = {id(p) for p in unique}
    dropped = {id(p): ("duplicate" if id(p) not in unique_ids else "group")
               for p in persons if id(p) not in kept_ids}
    return kept, dropped


def associate(
    persons: list[PersonContext],
    detections: list[Detection],
    cfg: ComplianceConfig,
) -> tuple[list[dict[str, list[Detection]]], list[Detection]]:
    """Return, per person, {item_name: [ppe detections]} plus unassigned PPE."""
    assigned: list[dict[str, list[Detection]]] = [{it.name: [] for it in cfg.items} for _ in persons]
    unassigned: list[Detection] = []
    regions = [[item_region(p, it)[0] for it in cfg.items] for p in persons]

    for k, item in enumerate(cfg.items):
        for classes in (set(item.positive_classes), set(item.negative_classes)):
            boxes = [d for d in detections if d.class_name in classes]
            pairs = []
            for j, det in enumerate(boxes):
                cx, cy = _center(det.bbox)
                for i in range(len(persons)):
                    region = regions[i][k]
                    if ioa(det.bbox, region) < item.min_overlap:
                        continue
                    rx, ry = _center(region)
                    dist = ((cx - rx) ** 2 + (cy - ry) ** 2) ** 0.5
                    pairs.append(((round(iou(det.bbox, region), 2), det.confidence, -dist, -area(region)), i, j))
            pairs.sort(key=lambda p: p[0], reverse=True)
            used_persons: set[int] = set()
            used_boxes: set[int] = set()
            for _, i, j in pairs:
                if i in used_persons or j in used_boxes:
                    continue
                assigned[i][item.name].append(boxes[j])
                used_persons.add(i)
                used_boxes.add(j)
            unassigned.extend(d for j, d in enumerate(boxes) if j not in used_boxes)
    return assigned, unassigned
