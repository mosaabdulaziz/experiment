"""products/safety/service.py: the safety product's settings, cameras, Apply and the incidents still waiting for an acknowledgement
(safety integration, ticket 2)."""
import asyncio
import json
import pathlib

import fakeredis
import pytest
import yaml

from app.events import EventStore
from products.safety.service import SafetyService

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _service(tmp_path, cameras=None, store=None):
    cams = tmp_path / "cameras.yaml"
    cams.write_text(yaml.safe_dump({"cameras": cameras or [
        {"id": "bay-a", "name": "Bay A", "clip": "a.mp4", "features": ["proximity", "zones", "ppe"]},
        {"id": "store", "name": "Store", "clip": "s.mp4", "features": ["fire"]}]}))
    config = {"safety": {"data_dir": str(tmp_path / "data"), "cameras": str(cams),
                         "product": str(ROOT / "products/safety/safety.yaml")}}
    return SafetyService(config, store=store, redis_client=fakeredis.FakeStrictRedis())


# ── settings ────────────────────────────────────────────────────────────────────────────────────────────────────

def test_every_feature_has_settings_by_default(tmp_path):
    s = _service(tmp_path).settings()
    assert set(s["features"]) == {"proximity", "zones", "ppe", "fall", "fire", "health"}
    assert s["features"]["fire"] == {"ack_required": True, "notify": True, "min_severity": "warning"}
    assert s["language"] == "en" and s["telegram"] == {"enabled": False, "chat_id": "", "token_set": False}


def test_settings_are_saved_and_merged_feature_by_feature(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    svc = _service(tmp_path)
    svc.update_settings({"features": {"ppe": {"ack_required": True, "min_severity": "critical"}},
                         "telegram": {"chat_id": "-100200"}, "language": "ar"})
    s = _service(tmp_path).settings()                       # read back from disk by a new service
    assert s["features"]["ppe"] == {"ack_required": True, "notify": False, "min_severity": "critical"}
    assert s["telegram"] == {"enabled": False, "chat_id": "-100200", "token_set": True}
    assert s["language"] == "ar"
    assert "123:abc" not in json.dumps(s)                   # the token never leaves .env


@pytest.mark.parametrize("bad", [
    {"features": {"teleport": {"notify": True}}},
    {"features": {"fire": {"min_severity": "apocalyptic"}}},
    {"features": {"fire": {"notify": "yes"}}},
    {"features": {"fire": {"colour": "red"}}},
    {"language": "fr"},
    {"telegram": {"enabled": "yes"}},
    {"telegram": {"token": "123:abc"}},
])
def test_bad_settings_are_refused(tmp_path, bad):
    with pytest.raises(ValueError):
        _service(tmp_path).update_settings(bad)


def test_telegram_is_off_until_turned_on_and_the_switch_and_the_chat_are_saved_apart(tmp_path):
    svc = _service(tmp_path)
    assert svc.update_settings({"telegram": {"enabled": True}})["telegram"]["enabled"] is True
    s = svc.update_settings({"telegram": {"chat_id": " -100 "}})            # the chat alone keeps the switch
    assert s["telegram"]["enabled"] is True and s["telegram"]["chat_id"] == "-100"
    s = svc.update_settings({"telegram": {"enabled": False}})               # and the switch alone keeps the chat
    assert s["telegram"]["enabled"] is False and s["telegram"]["chat_id"] == "-100"


def test_the_notify_rule_reads_feature_and_severity(tmp_path):
    svc = _service(tmp_path)
    svc.update_settings({"features": {"proximity": {"notify": True, "min_severity": "critical"}}})
    assert svc.should_notify("proximity", "critical") and not svc.should_notify("proximity", "warning")
    assert not svc.should_notify("zones", "critical")             # notify off by default
    assert not svc.should_notify(None, "critical")


# ── cameras and Apply ───────────────────────────────────────────────────────────────────────────────────────────

def test_cameras_list_their_features_and_the_streams_running_for_them(tmp_path):
    streams = [{"id": "bay-a@ppe", "camera": "bay-a", "feature": "ppe", "running": True},
               {"id": "bay-a@zones", "camera": "bay-a", "feature": "zones", "running": False}]
    (bay, store) = _service(tmp_path).cameras(streams)
    assert (bay["id"], bay["name"], bay["features"]) == ("bay-a", "Bay A", ["proximity", "zones", "ppe"])
    assert bay["streams"] == [{"id": "bay-a@ppe", "feature": "ppe", "running": True, "fps": None},
                              {"id": "bay-a@zones", "feature": "zones", "running": False, "fps": None}]
    assert store["streams"] == []


def test_a_camera_is_renamed_and_toggled_in_the_state_the_controller_reads(tmp_path):
    svc = _service(tmp_path)
    cam = svc.update_camera("bay-a", name="Loading bay", features=["ppe", "fire"])
    assert (cam["name"], cam["features"]) == ("Loading bay", ["ppe", "fire"])
    state = json.loads((tmp_path / "data" / "safety" / "cameras_state.json").read_text())
    assert state == {"bay-a": {"name": "Loading bay", "features": ["ppe", "fire"]}}


def test_an_unknown_camera_or_feature_is_refused(tmp_path):
    svc = _service(tmp_path)
    with pytest.raises(KeyError):
        svc.update_camera("nope", name="x")
    with pytest.raises(ValueError):
        svc.update_camera("bay-a", features=["teleport"])


def test_apply_bumps_the_version_and_reports_applying_until_the_controller_catches_up(tmp_path):
    svc = _service(tmp_path)
    assert svc.apply_status() == {"version": 0, "applied_version": 0, "applying": False}
    assert svc.request_apply() == 1
    assert svc.apply_status()["applying"] is True
    svc._redis.set("iris:analytics:safety_applied", 1)
    assert svc.apply_status() == {"version": 1, "applied_version": 1, "applying": False}


# ── waiting for an acknowledgement ──────────────────────────────────────────────────────────────────────────────

def test_unacknowledged_lists_only_features_that_require_it_including_ended_incidents(tmp_path):
    store = EventStore({"events": {"db_path": str(tmp_path / "events.db"), "retention_hours": 48}})
    svc = _service(tmp_path, store=store)

    async def run():
        await store.init()
        fire = await store.save("alert", "Fire", ["fire_smoke"], camera="store@fire",
                                incident_status="ongoing", incident_started_at="2026-11-08T10:00:00Z")
        await store.update_incident(fire["id"], "resolved", 12.0)          # ended, still not acked
        ppe = await store.save("alert", "No helmet", ["ppe_violation"], camera="bay-a@ppe")
        acked = await store.save("alert", "Fire again", ["fire_smoke"], camera="store@fire")
        await store.acknowledge(acked["id"])
        return fire["id"], ppe["id"], await svc.unacknowledged()

    fire_id, _ppe_id, waiting = asyncio.run(run())
    assert [e["id"] for e in waiting] == [fire_id]
    assert waiting[0]["incident_status"] == "resolved"


def test_an_unacknowledged_fall_is_not_pushed_out_by_frequent_alerts_that_need_no_acknowledgement(tmp_path):
    store = EventStore({"events": {"db_path": str(tmp_path / "events.db"), "retention_hours": 48}})
    svc = _service(tmp_path, store=store)

    async def run():
        await store.init()
        fall = await store.save("alert", "Fall", ["fall"], camera="rest-room@fall")
        for i in range(250):
            await store.save("alert", f"No helmet {i}", ["ppe_violation"], camera="bay-a@ppe")
        return fall["id"], await svc.unacknowledged()

    fall_id, waiting = asyncio.run(run())
    assert [e["id"] for e in waiting] == [fall_id]


def test_the_feature_filter_matches_role_streams_and_plain_tags():
    from app.features import features_where
    clause, params = features_where(["fall", "proximity"])
    assert "camera LIKE ?" in clause and "%@fall" in params and '%"fall"%' in params and '%"proximity%' in params
