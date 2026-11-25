"""Severity and category vocabulary for events.

Severity is the primary field; `event_type` is derived from it, so every existing
consumer (cooldowns, embedding dedup, the Incidents filter, AlertFlash) keeps
working untouched while the UI gains severity styling incrementally.
"""

SEVERITIES = ("critical", "warning", "info")
CATEGORIES = ("safety", "security", "storage", "operations")

# Deterministic, VLM-free alerts carry a fixed severity/category per behaviour.
# prompt_kind is already the behaviour name on the zone-alert path.
_ZONE_ALERT_DEFAULTS = {
    "intrusion": ("warning", "security"),
    "exit": ("warning", "security"),
    "overcrowding": ("warning", "safety"),
    "fire_smoke": ("critical", "safety"),
}
_ZONE_ALERT_FALLBACK = ("warning", "security")


def derive_event_type(severity: str) -> str:
    """`info` -> "comment", everything else -> "alert"."""
    return "comment" if severity == "info" else "alert"


def normalise_severity(raw, *, alert_hint: bool = False) -> str:
    """Coerce a model-supplied severity into the closed set, clamping DOWNWARD.

    An invalid or missing value never resolves to "critical": it becomes
    "warning" when something else about the entry reads as an alert, otherwise
    "info". A spurious full-screen flash costs more operator trust than a
    genuine alert sitting one notch too quiet.
    """
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in SEVERITIES:
            return s
    return "warning" if alert_hint else "info"


def normalise_category(raw) -> "str | None":
    """Coerce to the closed set; anything unrecognised becomes None (uncategorized)."""
    if isinstance(raw, str):
        c = raw.strip().lower()
        if c in CATEGORIES:
            return c
    return None


def zone_alert_defaults(kind: "str | None") -> "tuple[str, str]":
    """(severity, category) for a deterministic zone alert of this behaviour kind."""
    return _ZONE_ALERT_DEFAULTS.get(kind or "", _ZONE_ALERT_FALLBACK)
