"""Working-mode gate shared by the perception backends.

Two modes: `proactive` (run the VLM analysis cycle) and `chat` (pause it).
A `chat_in_flight` counter auto-pauses analysis for the duration of a chat turn
even while in proactive mode. State lives in the shared `monitoring` dict.
"""

VALID_MODES = ("proactive", "chat")


def analysis_enabled(monitoring: dict) -> bool:
    """True when proactive VLM analysis cycles should run."""
    return (
        monitoring.get("active_mode", "proactive") == "proactive"
        and monitoring.get("chat_in_flight", 0) == 0
    )
