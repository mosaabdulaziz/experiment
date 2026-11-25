"""One configured zone's live state on a stream."""
from dataclasses import dataclass, field

from perception.lib.overcrowding_dedup import OvercrowdingLatch

from .dedup import ZoneReentryGrace


@dataclass
class ZoneRuntime:
    """`label` is the nvdsanalytics ROI id (roi-<label>) the module reads back from each object's roiStatus; each
    zone owns its own dedup so alerts latch per-zone. Rebuilt from zones_config.txt on every version bump."""
    label: str
    name: str
    mode: str = "off"                 # restricted | valid | off
    intrusion_classes: set = field(default_factory=lambda: {"person"})
    overcrowding_enabled: bool = False
    overcrowding_threshold: int = 5
    overcrowding_classes: set = field(default_factory=lambda: {"person"})
    # Zone-alert dedup: once per person, tolerant of tracker re-IDs.
    _zone_grace: ZoneReentryGrace = field(default_factory=ZoneReentryGrace)
    # Overcrowding dedup: threshold-crossing with debounced release.
    _overcrowd_latch: OvercrowdingLatch = field(default_factory=OvercrowdingLatch)
    # End markers (Settings.end_markers): when the open intrusion / overcrowding began, and since when the zone
    # has had nobody violating it.
    _intrusion_open_at: "float | None" = None
    _intrusion_clear_since: "float | None" = None
    _overcrowd_open_at: "float | None" = None

    @property
    def intrusion_enabled(self) -> bool:
        return self.mode != "off"
