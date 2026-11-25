"""VEMA Tracker: Velocity-EMA Three-Regime Multi-Object Tracking research & benchmark suite."""

from .core.config import VEMAConfig
from .core.track import VEMATrack
from .core.tracker import VEMATracker

__all__ = ["VEMAConfig", "VEMATrack", "VEMATracker"]
