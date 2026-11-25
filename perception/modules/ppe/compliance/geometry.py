# Vendored from ppe-compliance (/home/iris2/IRIS/ppe-compliance), commit b07c335, ppe_compliance/geometry.py.
# Rule changes must be made in BOTH places, here and in ppe-compliance (whose evaluation the numbers in
# docs/EVALUATION.md come from); see perception/modules/ppe/README.md.
# Unchanged from the source.
from __future__ import annotations

from .types import BBox


def area(b: BBox) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersection(a: BBox, b: BBox) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def iou(a: BBox, b: BBox) -> float:
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def ioa(inner: BBox, outer: BBox) -> float:
    """Fraction of `inner` covered by `outer`."""
    a = area(inner)
    return intersection(inner, outer) / a if a > 0 else 0.0


def person_region(person: BBox, top: float, bottom: float, pad_x: float) -> BBox:
    x1, y1, x2, y2 = person
    w, h = x2 - x1, y2 - y1
    return (x1 - pad_x * w, y1 + top * h, x2 + pad_x * w, y1 + bottom * h)


def truncation(region: BBox, image_size: tuple[int, int] | None) -> float:
    """Fraction of `region` lying outside the image (0 = fully inside)."""
    if image_size is None:
        return 0.0
    w, h = image_size
    a = area(region)
    if a <= 0:
        return 1.0
    return 1.0 - intersection(region, (0, 0, w, h)) / a
