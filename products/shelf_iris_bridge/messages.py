# Vendored from shelf-demo (~/sarah/shelf-demo), branch feat/iris-integration, commit 195390728433235b0daced2c0f4dc8614a2d212e,
# service/messages.py. Copied unchanged below this header: fix it there first, then copy it again.
"""Bodies of the messages we publish to IRIS (validated by contract.ts zod schemas)."""
import uuid
from datetime import datetime, timezone

from .catalog import EVENT_TYPE

MAX_SPOTS_IN_METADATA = 20

_CLEAR_TEXT = {
    "gone": "No empty spots for {s:g} s",
    "paused": "Pipeline paused",
    "camera_removed": "Camera removed from the pipeline",
    "removed": "Pipeline removed",
}


def utc_iso(dt):
    """Aware datetime -> '2026-11-12T10:00:47.080Z' (zod datetime({offset: true}) accepts it)."""
    if dt.tzinfo is None:
        raise ValueError("occurredAt needs a timezone-aware datetime")
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def utc_now():
    return datetime.now(timezone.utc)


def correlation_key(camera_id):
    return f"shelf-void:{camera_id}"


def build_event(pipeline_id, camera_id, intent, settings, model_version=None):
    """event.raised / event.cleared body for an Intent. Publish with routing key f"event.{state}".

    confidence = best spot stability (settings.confidence_mode == "stability") or best raw
    detector score ("detector"); both always go in metadata.

    detectorScore is the score of each spot's LATEST detection at the moment the event is built
    (the tracker overwrites it on every hit), not the max and not the score when first seen.
    metadata.detectorScoreAt says so explicitly. On test.mp4 the R1 spot's latest score is 0.15
    at confirmation (47.08 s) and 0.109 at the end of the video (what the baseline printed). Severity is omitted, so IRIS uses
    the catalog defaultSeverity (medium). No evidence until hosting is decided (GAPS §1).
    """
    body = {
        "eventId": str(uuid.uuid4()),
        "pipelineId": pipeline_id,
        "cameraId": camera_id,
        "type": EVENT_TYPE,
        "state": intent.state,
        "correlationKey": correlation_key(camera_id),
        "occurredAt": utc_iso(intent.occurred_at),
    }
    meta = {"reason": intent.reason}
    if model_version:
        meta["modelVersion"] = model_version
    if intent.state == "raised":
        spots = sorted(intent.spots, key=lambda s: (-s.stability, -s.detector_score))
        stability = max(s.stability for s in spots)
        score = max(s.detector_score for s in spots)
        body["confidence"] = round(stability if settings.confidence_mode == "stability" else score, 4)
        n = len(spots)
        body["description"] = f"{n} empty spot{'s' if n != 1 else ''} on the shelf"
        meta.update(confidenceMode=settings.confidence_mode, spotCount=n,
                    stability=round(stability, 4), detectorScore=round(score, 4),
                    detectorScoreAt="latest detection at event time",
                    spots=[{"bbox": s.bbox, "detectorScore": round(s.detector_score, 4),
                            "stability": round(s.stability, 4)} for s in spots[:MAX_SPOTS_IN_METADATA]])
    else:
        body["description"] = _CLEAR_TEXT[intent.reason].format(s=intent.clear_seconds)
    body["metadata"] = meta
    return body
