"""Shelf alerts (iris:alert:<camera>[@<role>], keys.py; written by perception/modules/shelf_void) -> iris-platform
events, and the iris-platform side of being an AI backend (docs/ai-integration.md in iris-platform):

  on (re)connect   catalog.snapshot (cloud, PUBLISH_CATALOG) then state.sync.request
  catalog.sync     catalog.snapshot
  pipeline.upsert  pipeline.status: synced when every camera of the pipeline is in the mapping file, else failed
                   with the unmapped cameras in the detail (IRIS retries every minute until the file is fixed)
  pipeline.remove  pipeline.status synced "removed"
  shelf_empty      event.raised  (correlationKey shelf-void:<cameraId>:<region>)
  shelf_restocked  event.cleared (same key)

Detection itself (regions, confirm, clear, confidence filter, person pause) is the DeepStream module's: the pipeline's
`config` and `zonePolygon` from IRIS are acknowledged but not applied (the status detail says so). Events of a pipeline
IRIS says is paused, or for a camera it no longer assigns, are not sent; when that happens to a region with an open
situation, a cleared (reason paused / camera_removed) is sent, as the shelf-demo service does.
"""
import json
import threading
import uuid
from datetime import datetime
from types import SimpleNamespace

from .catalog import PIPELINE_ID, catalog_snapshot
from .log import log
from .messages import build_event, correlation_key

# eventId = uuid5(EVENT_NS, "<camera>/<redis entry id>"): an alert read twice (a restart before its ack) is the same
# event, which IRIS de-duplicates.
EVENT_NS = uuid.UUID("5d0c6b8e-3f1a-4d7e-9a42-6c1f0b2e7a91")
KINDS = {"shelf_empty": "raised", "shelf_restocked": "cleared"}


def region_key(camera_id: str, region: str) -> str:
    return f"{correlation_key(camera_id)}:{region}"


class Bridge:
    def __init__(self, settings, mapping, publish, mapping_name: str = "the mapping file"):
        self.settings, self.mapping, self.publish = settings, mapping, publish
        self.mapping_name = mapping_name
        self.pipelines = {}          # pipelineId -> {"state": "active" | "paused", "cameras": set(cameraId)}
        self.open = {}               # (pipelineId, cameraId, region) -> alert_id of an open situation
        self._lock = threading.Lock()

    # ---------- startup (broker on_connected) ----------
    def startup_messages(self) -> list:
        msgs = []
        if self.settings.is_cloud and self.settings.publish_catalog:
            msgs.append(("catalog.snapshot", catalog_snapshot()))
        msgs.append(("state.sync.request", {"target": self.settings.target}))
        return msgs

    # ---------- iris-platform -> us (broker thread) ----------
    def handle(self, routing_key: str, body: dict) -> None:
        kind = _kind(routing_key)
        with self._lock:
            if kind == "catalog.sync":
                if self.settings.is_cloud and self.settings.publish_catalog:
                    self.publish("catalog.snapshot", catalog_snapshot())
            elif kind == "pipeline.upsert":
                self._upsert(body)
            elif kind == "pipeline.remove":
                self._remove(body)
            else:
                log("ignored", routing_key=routing_key)

    def _upsert(self, msg: dict) -> None:
        pid = msg.get("pipelineId")
        if not isinstance(pid, str) or not pid:
            log("upsert without pipelineId, ignored")
            return
        pid = pid.lower()
        if msg.get("catalogPipelineId") != PIPELINE_ID:
            self._forget(pid, "removed")
            return self._status(pid, "failed", f"unknown pipeline {msg.get('catalogPipelineId')!r}; "
                                               f"this backend runs {PIPELINE_ID!r}")
        state = msg.get("state")
        if state not in ("active", "paused"):
            return self._status(pid, "failed", f"unknown state {state!r}")
        cams = [c for c in msg.get("cameras") or [] if isinstance(c, dict) and isinstance(c.get("cameraId"), str)]
        unmapped = [c for c in cams if not self.mapping.regions_of(pid, c["cameraId"])]
        if unmapped:
            self._forget(pid, "camera_removed")
            names = ", ".join(f"{c.get('name') or c['cameraId']} ({c['cameraId']})" for c in unmapped)
            return self._status(pid, "failed", f"not in {self.mapping_name}: {names}; map each to a DeepStream "
                                               "camera and shelf region")
        ids = {c["cameraId"].lower() for c in cams}
        old = self.pipelines.get(pid)
        for key in [k for k in self.open if k[0] == pid and k[1] not in ids]:
            self._close(key, "camera_removed")
        if state == "paused":
            for key in [k for k in self.open if k[0] == pid]:
                self._close(key, "paused")
        self.pipelines[pid] = {"state": state, "cameras": ids}
        sources = sorted(f"{cam}/{region}" for c in ids for cam, region in self.mapping.regions_of(pid, c))
        detail = f"{state} on {len(ids)} camera{'s' if len(ids) != 1 else ''}"
        if sources:
            detail += f" (DeepStream {', '.join(sources)})"
        if any(c.get("zonePolygon") for c in cams) or msg.get("config"):
            detail += "; config and zones are set in the DeepStream shelf preset"
        log("pipeline applied", pipelineId=pid, state=state, cameras=len(ids), sources=sources,
            state_changed=(old or {}).get("state") != state)
        self._status(pid, "synced", detail)

    def _remove(self, msg: dict) -> None:
        pid = msg.get("pipelineId")
        if not isinstance(pid, str) or not pid:
            return
        pid = pid.lower()
        self._forget(pid, None)            # IRIS stops accepting events of a removed pipeline (guide §5.3)
        log("pipeline removed", pipelineId=pid)
        self._status(pid, "synced", "removed")

    def _forget(self, pid: str, reason) -> None:
        self.pipelines.pop(pid, None)
        for key in [k for k in self.open if k[0] == pid]:
            if reason:
                self._close(key, reason)
            else:
                self.open.pop(key)

    def _close(self, key: tuple, reason: str) -> None:
        pid, cid, region = key
        self.open.pop(key, None)
        intent = SimpleNamespace(state="cleared", reason=reason, occurred_at=_now(), spots=(), clear_seconds=0)
        body = build_event(pid, cid, intent, self.settings)
        body["correlationKey"] = region_key(cid, region)
        body["metadata"]["region"] = region
        self.publish("event.cleared", body)

    def _status(self, pid: str, status: str, detail: str) -> None:
        self.publish("pipeline.status", {"pipelineId": pid, "status": status, "detail": detail[:500]})

    # ---------- shelf alerts -> iris-platform (alert reader thread) ----------
    def on_alert(self, camera: str, entry_id: str, f: dict) -> bool:
        """One alert stream entry; `camera` is the mapping's (@<role> suffix stripped). True when published."""
        state = KINDS.get(f.get("kind"))
        if state is None:
            return False                   # another module's alert on the same camera
        region = f.get("region", "")
        target = self.mapping.target(camera, region)
        if target is None:
            log("shelf alert of an unmapped region, skipped", camera=camera, region=region, kind=f.get("kind"))
            return False
        pid, cid = target
        with self._lock:
            known = self.pipelines.get(pid)
            if known is not None and (known["state"] != "active" or cid not in known["cameras"]):
                log("shelf alert for a pipeline that is not active on this camera, skipped", camera=camera,
                    region=region, pipelineId=pid, cameraId=cid, state=known["state"])
                return False
            try:
                intent = _intent(state, f)
                body = build_event(pid, cid, intent, self.settings)
            except (KeyError, ValueError, TypeError) as e:
                log("malformed shelf alert, skipped", camera=camera, entry=entry_id, error=f"{type(e).__name__}: {e}")
                return False
            body["eventId"] = str(uuid.uuid5(EVENT_NS, f"{camera}/{entry_id}"))
            body["correlationKey"] = region_key(cid, region)
            body["metadata"].update(region=region, deepstreamCamera=camera, alertId=f.get("alert_id", ""))
            if state == "raised":
                self.open[(pid, cid, region)] = f.get("alert_id", "")
            else:
                self.open.pop((pid, cid, region), None)
            self.publish(f"event.{state}", body)
        log("shelf alert sent", camera=camera, region=region, state=state, reason=intent.reason,
            pipelineId=pid, cameraId=cid, eventId=body["eventId"])
        return True


def _intent(state: str, f: dict) -> SimpleNamespace:
    """The module's alert fields as the Intent messages.build_event takes (core/situations.py Intent)."""
    occurred = datetime.fromisoformat(f["occurred_at"].replace("Z", "+00:00"))
    if occurred.tzinfo is None:
        raise ValueError("occurred_at has no timezone")
    spots = ()
    if state == "raised":
        spots = tuple(SimpleNamespace(bbox=s["bbox"], detector_score=float(s["detectorScore"]),
                                      stability=float(s["stability"])) for s in json.loads(f["spots"]))
        if not spots:
            raise ValueError("a shelf_empty without spots")
    return SimpleNamespace(state=state, reason=f.get("reason") or ("new" if state == "raised" else "gone"),
                           occurred_at=occurred, spots=spots, clear_seconds=float(f.get("clear_s") or 0))


def _now():
    from .messages import utc_now
    return utc_now()


def _kind(rk: str) -> str:
    """'pipeline.upsert.cloud' / 'pipeline.upsert.edge.acme' -> 'pipeline.upsert'."""
    if rk.endswith(".cloud"):
        return rk[: -len(".cloud")]
    i = rk.find(".edge.")
    return rk[:i] if i >= 0 else rk
