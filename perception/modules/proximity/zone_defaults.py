"""The per-zone settings' defaults and allowed values, from zone_defaults.json next to this file. The module reads them
here; the app reads the same file by path (app/zones.py), because importing this package would load the pipeline."""
from __future__ import annotations

import json
from pathlib import Path

PATH = Path(__file__).with_name("zone_defaults.json")
_RAW = json.loads(PATH.read_text())

DEFAULTS: dict = _RAW["defaults"]
VEHICLE_CLASSES: tuple = tuple(_RAW["vehicle_classes"])
HIDDEN_FEET: tuple = tuple(_RAW["hidden_feet"])
POSITIVE: tuple = tuple(_RAW["positive"])
NON_NEGATIVE: tuple = tuple(_RAW["non_negative"])
