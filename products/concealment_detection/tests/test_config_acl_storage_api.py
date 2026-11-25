"""Unit and integration tests for `config.py`, `acl.py`, `storage.py`, `consumer.py`, and `server.py`."""
from __future__ import annotations

import json
from pathlib import Path
from fastapi.testclient import TestClient
import pytest

import acl
from concealment.state_machine import ConcealmentMonitor
from concealment.types import DetectorSettings
from config import load_config
from consumer import CameraWorker, FrameQueue
from conftest import make_skeleton_obs
from evidence import build_candidate_record
from server import create_app
from storage import ReviewStore, connect
from test_evidence_and_clock import _sample_candidate


class FakeRedis:
    """In-memory stand-in for redis.Redis streams (`xrevrange`, `xread`, `xadd`)."""

    def __init__(self) -> None:
        self.streams: dict[str, list[tuple[str, dict]]] = {}
        self._seq = 0

    def xadd(self, key: str, fields: dict, maxlen: int | None = None) -> str:
        self._seq += 1
        entry_id = f"{self._seq}-0"
        bucket = self.streams.setdefault(key, [])
        bucket.append((entry_id, dict(fields)))
        if maxlen is not None and len(bucket) > maxlen:
            del bucket[:-maxlen]
        return entry_id

    def xrevrange(self, key: str, count: int = 1) -> list:
        bucket = self.streams.get(key, [])
        return list(reversed(bucket[-count:])) if bucket else []

    def xread(self, streams: dict[str, str], count: int = 64) -> list:
        out = []
        for key, last_id in streams.items():
            bucket = self.streams.get(key, [])
            last_seq = int(str(last_id).split("-")[0]) if last_id and last_id != "0-0" else 0
            recs = [(eid, f) for eid, f in bucket if int(eid.split("-")[0]) > last_seq][:count]
            if recs:
                out.append((key, recs))
        return out


def test_load_config_reads_shipped_yaml_and_zones_and_validates(tmp_path):
    cfg = load_config()
    assert cfg.port == 8005
    assert [c.id for c in cfg.cameras] == ["cam-1"]
    assert cfg.cameras[0].location == "cam-1"
    assert "cam-1" in cfg.camera_zones
    assert cfg.detector.dwell_min_s == 2.0

    # Env override for port and data_dir
    cfg_env = load_config(env={"CONCEALMENT_BACKEND_PORT": "9011", "CONCEALMENT_DATA_DIR": str(tmp_path)})
    assert cfg_env.port == 9011
    assert cfg_env.db_path == tmp_path / "concealment_detection.db"

    # Empty cameras raises ValueError
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("cameras: []\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path=bad_yaml, env={})


def test_acl_parse_pose_frame_and_alert_and_clip_streams():
    r = FakeRedis()
    reader = acl.MetaStreamReader(r)

    # First read primes cursor at stream tip
    assert reader.read("cam-1") == []

    obs = make_skeleton_obs(track_id=5, center_x=960.0)
    pose_payload = {
        "shape": [1920, 1080],
        "items": [
            {
                "id": 5,
                "box": list(obs.box),
                "kps": [list(kp) for kp in obs.kps],
            }
        ],
    }
    dets_payload = [
        ["handbag", 0.88, 940, 500, 70, 70],
    ]
    r.xadd(
        "iris:meta:cam-1",
        {
            "frame_num": "12",
            "pts": "1.500",
            "pose": json.dumps(pose_payload),
            "dets": json.dumps(dets_payload),
        },
    )
    entries = reader.read("cam-1")
    assert len(entries) == 1
    frame = acl.parse_pose_frame("cam-1", entries[0])
    assert frame.camera == "cam-1"
    assert frame.frame_num == 12
    assert frame.pts == 1.5
    assert frame.observed is True
    assert len(frame.tracks) == 1
    assert frame.tracks[0].track_id == 5
    assert frame.tracks[0].has_bag is True
    assert frame.tracks[0].bag_box is not None

    # Entry without `pose` field -> observed=False
    frame_unobs = acl.parse_pose_frame("cam-1", {"id": "2-0", "fields": {"frame_num": "13", "pts": "1.54"}})
    assert frame_unobs.observed is False
    assert frame_unobs.tracks == ()

    # write_alert validates non-accusatory wording and writes to iris:alert:cam-1
    msg = "Suspected concealment near shelf-a on cam-1. Suspected incident — requires human verification."
    assert acl.write_alert(r, "cam-1", severity="critical", message=msg, pts=1.5, zone="shelf-a") is True
    assert acl.write_alert(r, "cam-1", severity="warning", message=msg, pts=2.0, zone="shelf-a") is True
    assert len(r.streams["iris:alert:cam-1"]) == 2
    _, crit_fields = r.streams["iris:alert:cam-1"][0]
    _, written_fields = r.streams["iris:alert:cam-1"][1]
    assert crit_fields["severity"] == "critical"
    assert written_fields["kind"] == "suspected_concealment"
    assert written_fields["severity"] == "warning"
    assert written_fields["category"] == "security"

    # Accusatory alert text is rejected
    with pytest.raises(ValueError):
        acl.write_alert(r, "cam-1", severity="warning", message="Confirmed theft on cam-1", pts=1.5)

    assert acl.request_clip(r, "cam-1", 1.5) is True
    assert r.streams["iris:svr_request:cam-1"][0][1]["pts"] == "1.5"


def test_storage_review_store_and_fastapi_endpoints(tmp_path):
    conn = connect(tmp_path / "test.db")
    store = ReviewStore(conn)
    rec = build_candidate_record(_sample_candidate("high", "clothing"))
    cid = store.add_candidate(rec)
    assert cid == 1

    store.attach_clip(cid, "data/alert_svr/cam-1_12.mp4")
    fetched = store.get_candidate(cid)
    assert fetched is not None
    assert fetched["clip_path"] == "data/alert_svr/cam-1_12.mp4"
    assert fetched["status"] == "candidate"
    assert fetched["requires_human_verification"] is True

    # AC-5: "confirmed" is never an allowed status in storage or HTTP API
    with pytest.raises(ValueError):
        store.set_review_status(cid, "confirmed", note="bad")

    aid = store.add_assessment({"camera": "cam-1", "track_id": 9, "pts": 3.0, "state": "insufficient_evidence", "reason": "small_bbox"})
    assert aid == 1
    assert len(store.list_assessments("cam-1")) == 1

    app = create_app({"ready": True}, store, cameras=["cam-1"])
    client = TestClient(app)

    assert client.get("/status").json() == {"ready": True}
    assert client.get("/cameras").json() == ["cam-1"]
    cands = client.get("/candidates").json()
    assert len(cands) == 1
    assert cands[0]["id"] == cid

    detail = client.get(f"/candidates/{cid}").json()
    assert detail["location"] == "cam-1"
    assert detail["clip_path"] == "data/alert_svr/cam-1_12.mp4"
    assert client.get("/candidates/999").status_code == 404

    # Staff review: 400 on "confirmed", 200 on "concern_noted"
    res_bad = client.post(f"/review/{cid}", json={"status": "confirmed", "note": "nope"})
    assert res_bad.status_code == 400

    res_ok = client.post(f"/review/{cid}", json={"status": "concern_noted", "note": "needs floor check", "reviewer": "alice"})
    assert res_ok.status_code == 200
    assert store.get_candidate(cid)["status"] == "concern_noted"
    store.close()


def test_frame_queue_drops_oldest_and_worker_processes_entries(tmp_path):
    q = FrameQueue(maxsize=2)
    q.put({"seq": 1})
    q.put({"seq": 2})
    q.put({"seq": 3})
    assert q.qsize() == 2
    assert q.get_nowait() == {"seq": 2}
    assert q.get_nowait() == {"seq": 3}

    r = FakeRedis()
    store = ReviewStore(connect(tmp_path / "worker.db"))
    settings = DetectorSettings(min_track_s=0.1, dwell_min_s=0.3, min_hold_s=0.2)
    cfg = load_config()
    mon = ConcealmentMonitor("cam-1", settings, cfg.camera_zones["cam-1"])
    worker = CameraWorker("cam-1", mon, r, store)

    # Insufficient quality entry -> logged to assessments, zero Redis alerts
    tiny = make_skeleton_obs(track_id=2, box=(10.0, 10.0, 20.0, 30.0))
    bad_entry = {
        "id": "1-0",
        "fields": {
            "frame_num": "1",
            "pts": "0.1",
            "pose": json.dumps({"shape": [1920, 1080], "items": [{"id": 2, "box": list(tiny.box), "kps": [list(k) for k in tiny.kps]}]}),
        },
    }
    res = worker.process_entry(bad_entry)
    assert res["candidates"] == []
    assert len(res["assessments"]) == 1
    assert "iris:alert:cam-1" not in r.streams
    store.close()
