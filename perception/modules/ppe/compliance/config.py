# Vendored from ppe-compliance (/home/iris2/IRIS/ppe-compliance), commit b07c335, ppe_compliance/config.py.
# Rule changes must be made in BOTH places, here and in ppe-compliance (whose evaluation the numbers in
# docs/EVALUATION.md come from); see perception/modules/ppe/README.md.
# Trimmed: only PPEItemConfig, ComplianceConfig and TemporalConfig, unchanged. ModelConfig, TrackingConfig,
# EventConfig, PipelineConfig and the YAML loader are not vendored: the module's Settings builds these configs.
# IRIS change, not in ppe-compliance yet: TemporalConfig.close_after_present_s, default 0 = the source's behaviour.
"""Configuration of the PPE compliance rules (per-item thresholds and geometry, temporal smoothing)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PPEItemConfig:
    """How one PPE item maps onto model classes and person geometry."""
    name: str
    positive_classes: list[str]
    negative_classes: list[str] = field(default_factory=list)
    # Region of the person box where the item is expected, as fractions of
    # person height measured from the top of the box (may be negative to
    # extend above the box, e.g. helmet brims sticking out).
    region_top: float = 0.0
    region_bottom: float = 1.0
    # Horizontal padding of the region as a fraction of person width.
    region_pad_x: float = 0.1
    # Minimum fraction of the PPE box that must fall inside the region.
    min_overlap: float = 0.5
    # A positive detection with conf >= present_conf marks the item PRESENT.
    present_conf: float = 0.40
    # A negative detection with conf >= absent_conf marks the item ABSENT.
    absent_conf: float = 0.40
    # Positive and negative evidence must differ by at least this much,
    # otherwise the observation is treated as conflicting -> UNKNOWN.
    conflict_margin: float = 0.15
    # If True, a clearly visible person with no positive detection at all is
    # treated as ABSENT even without an explicit negative detection. Off by
    # default: absence of evidence is not evidence of absence.
    infer_absence_from_missing_positive: bool = False
    # Positive detections below this conf count as "no positive" for the
    # inference rule above.
    missing_positive_max_conf: float = 0.15
    # If True, the item is ABSENT only when no positive box at all (any conf
    # >= model.min_detection_conf) is on the person; a confident negative next
    # to a weak positive gives UNKNOWN (conflicting_evidence). On for the vest:
    # an open vest over a dark shirt is often detected as NO-Safety Vest.
    absent_requires_no_positive: bool = False
    # Person box must reach at least this many pixels in height for the item
    # to be judged at all (small people -> UNKNOWN).
    min_person_height_px: int = 48
    # Frame edges ("top", "bottom", "left", "right") at which the item may be
    # cut off. The item is truncated when its region - limited to the person
    # box - reaches one of these edges: the head band reaches the top edge
    # when the person box does, the torso band never reaches top or bottom
    # (it starts 12% below the box top and ends 25% above the box bottom), and
    # both bands span the full box width, so they reach left/right when the
    # box does. A truncated item is never ABSENT; positive evidence can still
    # make it PRESENT.
    sensitive_edges: list[str] = field(default_factory=lambda: ["top", "bottom", "left", "right"])


@dataclass
class ComplianceConfig:
    person_classes: list[str] = field(default_factory=lambda: ["Person"])
    min_person_conf: float = 0.35
    # A person box within this many pixels of the image border "touches" it.
    edge_margin_px: int = 3
    # Duplicate person boxes (e.g. an upper-body box inside a full-body box of
    # the same worker): the smaller box is dropped before association when at
    # least `duplicate_min_inside` of its area lies inside a larger person box
    # AND the two are horizontally aligned: the IoU of their x-intervals is at
    # least `duplicate_min_x_overlap`. A box around two people side by side
    # fails the second test, so the individual boxes inside it are kept.
    # Set duplicate_min_inside above 1 to disable.
    duplicate_min_inside: float = 0.85
    duplicate_min_x_overlap: float = 0.80
    # A person box that contains at least this many other person boxes (each
    # with >= duplicate_min_inside of its area inside) is a box drawn around
    # several workers, not a person: it is dropped before association. 0 = off.
    group_min_members: int = 2
    # Required PPE for this site/camera. Names refer to `items` below.
    required_items: list[str] = field(default_factory=lambda: ["helmet", "vest"])
    items: list[PPEItemConfig] = field(default_factory=lambda: [
        PPEItemConfig(
            name="helmet",
            positive_classes=["Hardhat"],
            negative_classes=["NO-Hardhat"],
            region_top=-0.15, region_bottom=0.30,
        ),
        PPEItemConfig(
            name="vest",
            positive_classes=["Safety Vest"],
            negative_classes=["NO-Safety Vest"],
            region_top=0.12, region_bottom=0.75,
            absent_requires_no_positive=True,
        ),
    ])

    def item(self, name: str) -> PPEItemConfig:
        for it in self.items:
            if it.name == name:
                return it
        raise KeyError(f"Required item {name!r} has no entry in compliance.items")


@dataclass
class TemporalConfig:
    # Sliding window of processed frames per track used for smoothing.
    window: int = 8
    # An item is ABSENT for the track only if at least this many frames in the
    # window were ABSENT and ABSENT outnumbers PRESENT.
    min_absent_frames: int = 6
    min_present_frames: int = 3
    # After an event for a track, don't emit another for the same items for
    # this many seconds of video time.
    cooldown_s: float = 30.0
    # IRIS change, not in ppe-compliance yet: an open episode closes
    # only once the item has been smoothed PRESENT for this many seconds without
    # a break. 0 = on the first PRESENT frame, as evaluated. The ppe module
    # sets 2 s: on test2.mp4 a bare head or cap read as a hardhat for 0.4-1.45 s
    # closed open episodes, and the same violation alerted again after the cooldown.
    close_after_present_s: float = 0.0
    # Smoothing state of a track not seen for this long is dropped.
    forget_after_s: float = 120.0
