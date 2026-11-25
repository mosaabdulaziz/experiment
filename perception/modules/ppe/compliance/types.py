# Vendored from ppe-compliance (/home/iris2/IRIS/ppe-compliance), commit b07c335, ppe_compliance/types.py.
# Rule changes must be made in BOTH places, here and in ppe-compliance (whose evaluation the numbers in
# docs/EVALUATION.md come from); see perception/modules/ppe/README.md.
# Unchanged from the source.
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ItemStatus(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"
    UNKNOWN = "unknown"


class ComplianceStatus(str, Enum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    UNKNOWN = "unknown"


BBox = tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: BBox

    def to_dict(self) -> dict:
        return {"class": self.class_name, "confidence": round(self.confidence, 4),
                "bbox": [round(v, 1) for v in self.bbox]}


@dataclass
class ItemResult:
    item: str
    status: ItemStatus
    reason: str
    positive_conf: float = 0.0
    negative_conf: float = 0.0
    evidence: list[Detection] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"item": self.item, "status": self.status.value, "reason": self.reason,
                "positive_conf": round(self.positive_conf, 4),
                "negative_conf": round(self.negative_conf, 4),
                "evidence": [d.to_dict() for d in self.evidence]}


@dataclass
class PersonResult:
    person: Detection
    items: dict[str, ItemResult]
    status: ComplianceStatus
    track_id: int | None = None

    @property
    def missing_items(self) -> list[str]:
        return [n for n, r in self.items.items() if r.status is ItemStatus.ABSENT]

    def to_dict(self) -> dict:
        return {"track_id": self.track_id, "status": self.status.value,
                "person": self.person.to_dict(),
                "items": {n: r.to_dict() for n, r in self.items.items()}}


@dataclass
class FrameResult:
    frame_index: int
    timestamp_s: float
    detections: list[Detection]
    persons: list[PersonResult]
    # PPE detections that could not be attributed to any person.
    unassigned: list[Detection] = field(default_factory=list)
