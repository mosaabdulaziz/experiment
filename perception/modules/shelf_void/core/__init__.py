"""The heat-map void detector and the RAISED/CLEARED alerter, copied from shelf-monitoring (commit 02538da), and the
accepted shelf-demo's per-region logic (stability: tracker), vendored unchanged from shelf-demo service/: tracker.py
(StableTracker), person_gate.py (PersonGate) and situations.py (Situation). Each vendored file names its source commit
in its header."""
from .alerts import AlertConfig, EmptyAlertEvent, EmptyShelfAlerter
from .detector import DetectorConfig, ShelfVoidDetector, VoidRegion
from .person_gate import PersonGate
from .situations import Intent, Situation, Spot
from .tracker import StableTracker

__all__ = ["AlertConfig", "EmptyAlertEvent", "EmptyShelfAlerter", "DetectorConfig", "ShelfVoidDetector", "VoidRegion",
           "PersonGate", "Intent", "Situation", "Spot", "StableTracker"]
