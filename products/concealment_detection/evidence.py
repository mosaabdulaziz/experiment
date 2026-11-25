"""Evidence builder, severity mapper, and non-accusatory wording guard."""
from __future__ import annotations

from datetime import datetime, timezone

from concealment.types import TIERS, CandidateEvent
from footage_clock import FootageClock, display_timestamp

FORBIDDEN_PHRASES = (
    "confirmed theft",
    "theft detected",
    "theft confirmed",
    "stolen",
    "shoplifter",
    "guilty",
)


def tier_to_severity(tier: str) -> str:
    """Map candidate tier (`high` | `medium` | `low`) to platform severity (`critical` | `warning` | `info`)."""
    if tier not in TIERS:
        raise ValueError(f"Unsupported candidate tier '{tier}'")
    if tier == "high":
        return "critical"
    if tier == "medium":
        return "warning"
    return "info"


def validate_non_accusatory_text(text: str) -> None:
    """Raise `ValueError` if `text` contains forbidden accusatory phrases or omits suspected/verification framing."""
    low = text.lower()
    for phrase in FORBIDDEN_PHRASES:
        if phrase in low:
            raise ValueError(f"Forbidden accusatory phrase '{phrase}' in description")
    if "suspected" not in low or "human verification" not in low:
        raise ValueError(
            "Description must identify the alert as a suspected incident requiring human verification"
        )


def build_description(
    candidate: CandidateEvent,
    display_time: str | None = None,
    *,
    location: str | None = None,
) -> str:
    """Build a traceable, template-based behavioral description from `candidate` stage timelines."""
    stages = " -> ".join(f"{s.stage}@{s.entered_pts:.1f}s" for s in candidate.stage_timeline)
    if candidate.target_kind == "personal_bag":
        target_phrase = "a personal bag"
    else:
        target_phrase = f"clothing ({candidate.body_zone_name} area)"

    loc = location or candidate.camera
    time_part = f" at {display_time}" if display_time else f" at pts={candidate.pts:.2f}s"
    text = (
        f"Suspected suspicious activity requiring human verification on camera {loc} "
        f"(zone {candidate.zone_id}){time_part}: tracked person #{candidate.track_id} "
        f"dwelled {candidate.dwell_duration_s:.1f}s, reached toward shelf, and held hand near "
        f"{target_phrase} for {candidate.hold_duration_s:.1f}s ({candidate.terminal_stage}; "
        f"stages: {stages}; tier={candidate.tier}, score={candidate.score:.2f})."
    )
    validate_non_accusatory_text(text)
    return text


def build_candidate_record(
    candidate: CandidateEvent,
    clock: FootageClock | None = None,
    device_dt: datetime | None = None,
    *,
    location: str | None = None,
) -> dict:
    """Build a complete candidate dictionary for `storage.add_candidate()` and `acl.write_alert()`."""
    now_dt = device_dt or datetime.now(timezone.utc)
    disp_time = display_timestamp(now_dt, pts=candidate.pts, clock=clock)
    severity = tier_to_severity(candidate.tier)
    loc = location or candidate.camera
    description = build_description(candidate, display_time=disp_time, location=loc)

    return {
        "camera": candidate.camera,
        "location": loc,
        "track_id": candidate.track_id,
        "pts": candidate.pts,
        "display_time": disp_time,
        "tier": candidate.tier,
        "severity": severity,
        "category": "security",
        "kind": "suspected_concealment",
        "requires_human_verification": True,
        "score": candidate.score,
        "zone_id": candidate.zone_id,
        "target_kind": candidate.target_kind,
        "body_zone_name": candidate.body_zone_name,
        "terminal_stage": candidate.terminal_stage,
        "hold_duration_s": candidate.hold_duration_s,
        "dwell_duration_s": candidate.dwell_duration_s,
        "stage_timeline": [
            {"stage": s.stage, "entered_pts": s.entered_pts, "detail": s.detail}
            for s in candidate.stage_timeline
        ],
        "clip_window": [candidate.clip_window[0], candidate.clip_window[1]],
        "keyframes_pts": list(candidate.keyframes_pts),
        "description": description,
        "status": "candidate",
        "clip_path": None,
        "thumbnail_path": None,
    }
