import json
from datetime import datetime

from fastapi.testclient import TestClient

from config import CameraConfig, ServiceConfig
from server import create_app
from zones import FrameGeometry


def client(tmp_path):
    cfg = ServiceConfig(redis_url="redis://localhost:1/0", frame=FrameGeometry(1000, 1000),
                        cameras=(CameraConfig("barber-c"), CameraConfig("barber-l")),
                        data_dir=tmp_path, stream_starts_file=tmp_path / "starts.json",
                        crops_dir=tmp_path / "crops")
    (tmp_path / "crops").mkdir()
    return TestClient(create_app(cfg, start_consumer=False), client=("127.0.0.1", 50000))


def site_client(tmp_path):
    cfg = ServiceConfig(redis_url="redis://localhost:1/0", frame=FrameGeometry(1000, 1000),
                        cameras=(CameraConfig("ph-1", name="Wash area"), CameraConfig("ph-3")),
                        data_dir=tmp_path, stream_starts_file=tmp_path / "starts.json",
                        crops_dir=tmp_path / "crops", wait_camera="ph-3",
                        policy={"wt_end_mode": 1.0, "visits_source": 1.0})
    (tmp_path / "crops").mkdir()
    return TestClient(create_app(cfg, start_consumer=False), client=("127.0.0.1", 50000))


def test_a_policy_put_keeps_the_site_defaults(tmp_path):
    c = site_client(tmp_path)
    assert c.get("/api/barber/policy").json()["wt_end_mode"] == 1.0
    r = c.put("/api/barber/policy", json={"wt_out_s": 4})
    assert r.status_code == 200
    body = r.json()
    assert body["wt_out_s"] == 4.0 and body["wt_end_mode"] == 1.0 and body["visits_source"] == 1.0
    assert c.get("/api/barber/policy").json()["wt_end_mode"] == 1.0


def site_client_with(tmp_path, policy):
    cfg = ServiceConfig(redis_url="redis://localhost:1/0", frame=FrameGeometry(1000, 1000),
                        cameras=(CameraConfig("ph-3"),), data_dir=tmp_path, stream_starts_file=tmp_path / "starts.json",
                        crops_dir=tmp_path / "crops", wait_camera="ph-3", policy=policy)
    (tmp_path / "crops").mkdir(exist_ok=True)
    return TestClient(create_app(cfg, start_consumer=False), client=("127.0.0.1", 50000))


def test_the_site_policy_block_is_a_layer_under_the_keys_set_in_setup(tmp_path):
    # final review I3: no file is written on start, a PUT stores only its keys, a yaml change to any other key shows
    c = site_client_with(tmp_path, {"wt_end_mode": 1.0, "wt_serve_s": 150.0})
    assert c.get("/api/barber/policy").json()["wt_serve_s"] == 150.0
    assert not (tmp_path / "policy.json").exists()
    assert c.put("/api/barber/policy", json={"wt_out_s": 4}).json()["wt_serve_s"] == 150.0
    assert c.put("/api/barber/policy", json={"wt_return_sim": 0.85}).status_code == 200
    assert json.loads((tmp_path / "policy.json").read_text()) == {"wt_out_s": 4.0, "wt_return_sim": 0.85}
    c = site_client_with(tmp_path, {"wt_end_mode": 1.0, "wt_serve_s": 120.0, "wt_out_s": 9.0})   # the yaml edited
    got = c.get("/api/barber/policy").json()
    assert (got["wt_serve_s"], got["wt_out_s"], got["wt_return_sim"]) == (120.0, 4.0, 0.85)   # set in Setup: wins


def test_a_full_policy_file_written_before_still_wins_key_by_key(tmp_path):
    from policy import DEFAULT_POLICY
    (tmp_path / "policy.json").write_text(json.dumps({**DEFAULT_POLICY, "wt_serve_s": 90.0}))
    c = site_client_with(tmp_path, {"wt_end_mode": 1.0, "wt_serve_s": 150.0})
    got = c.get("/api/barber/policy").json()
    assert got["wt_serve_s"] == 90.0 and got["wt_end_mode"] == 0.0         # every key pinned, as before
    c.put("/api/barber/policy", json={"wt_out_s": 4})
    assert json.loads((tmp_path / "policy.json").read_text()) == {**DEFAULT_POLICY, "wt_serve_s": 90.0, "wt_out_s": 4.0}


def test_cameras_carry_their_display_name(tmp_path):
    cams = site_client(tmp_path).get("/api/barber/cameras").json()
    assert cams == [{"id": "ph-1", "stream": "ph-1", "name": "Wash area"}, {"id": "ph-3", "stream": "ph-3", "name": None}]


SQ = [[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]]


def test_zone_problems_report_a_half_drawn_pair(tmp_path):
    with client(tmp_path) as c:
        r = c.put("/api/monitoring/zones/barber-c", json=[{"id": "c-ch1", "type": "chair", "points": SQ, "pair": "ch1"}])
        assert r.status_code == 200
        assert {"code": "pair_single_camera", "pair": "ch1", "camera": "barber-c"} \
            in c.get("/api/monitoring/zones/problems").json()
        c.put("/api/monitoring/zones/barber-l", json=[
            {"id": "l-ch1", "type": "chair", "points": SQ, "pair": "ch1", "owner": True},
            {"id": "l-s1", "type": "seat", "points": SQ, "seat_of": "l-ch1"}])
        assert c.get("/api/monitoring/zones/problems").json() == []


def test_bindings_empty_on_a_fresh_db(tmp_path):
    with client(tmp_path) as c:
        assert c.get("/api/barber/bindings").json() == []


def test_roster_assign_and_read(tmp_path):
    with client(tmp_path) as c:
        lisa = c.post("/api/staff", json={"name": "lisa"}).json()["id"]
        assert c.put("/api/barber/roster/ch1", json={"staff_id": lisa}).json() == {"ch1": lisa}
        assert c.get("/api/barber/roster").json() == {"ch1": lisa}
        assert c.put("/api/barber/roster/ch1", json={"staff_id": "staff-99"}).status_code == 404
        assert c.put("/api/barber/roster/ch1", json={"staff_id": None}).json() == {}


def test_live_payload_shape(tmp_path):
    with client(tmp_path) as c:
        live = c.get("/api/barber/live").json()
        for key in ("customers", "chairs", "cameras", "scene", "zone_problems", "roster", "customers_waiting",
                    "today"):
            assert key in live


def test_date_only_ranges_start_at_the_shop_day_hour(tmp_path):
    from server import _parse_range
    assert _parse_range("2026-09-25", "2026-09-26", 5) == (datetime(2026, 9, 25, 5), datetime(2026, 9, 26, 5))
    assert _parse_range("2026-09-25T10:00:00", None, 5) == (datetime(2026, 9, 25, 10), datetime(2026, 9, 26, 10))


def test_roster_detail_has_since(tmp_path):
    with client(tmp_path) as c:
        lisa = c.post("/api/staff", json={"name": "lisa"}).json()["id"]
        c.put("/api/barber/roster/ch1", json={"staff_id": lisa})
        detail = c.get("/api/barber/roster", params={"detail": 1}).json()
        assert detail["ch1"]["staff_id"] == lisa and len(detail["ch1"]["since"]) == 19
        assert c.get("/api/barber/roster").json() == {"ch1": lisa}          # plain GET unchanged


def test_roster_confirmation_is_per_shop_day(tmp_path):
    with client(tmp_path) as c:
        first = c.get("/api/barber/roster/confirmation").json()
        assert first["confirmed"] is False and first["confirmed_at"] is None
        assert c.get("/api/barber/live").json()["roster_confirmed"] is False
        done = c.post("/api/barber/roster/confirmation").json()
        assert done["confirmed"] is True and done["day"] == first["day"]
        assert c.post("/api/barber/roster/confirmation").json()["confirmed_at"] == done["confirmed_at"]  # idempotent
        assert c.get("/api/barber/live").json()["roster_confirmed"] is True


def test_live_payload_names_the_shop(tmp_path):
    """2026-10-18 (user): the shop's name on every page (the rail's wordmark)."""
    from dataclasses import replace
    cfg = ServiceConfig(redis_url="redis://localhost:1/0", frame=FrameGeometry(1000, 1000),
                        cameras=(CameraConfig("barber-c"),), data_dir=tmp_path,
                        stream_starts_file=tmp_path / "starts.json", crops_dir=tmp_path / "crops")
    (tmp_path / "crops").mkdir()
    cfg = replace(cfg, place="BekkyBarber", branch="Palm Hills")
    with TestClient(create_app(cfg, start_consumer=False), client=("127.0.0.1", 50000)) as c:
        assert c.get("/api/barber/live").json()["site"] == {"place": "BekkyBarber", "branch": "Palm Hills", "logo": False}
        assert c.get("/api/barber/logo").status_code == 404
    logo = tmp_path / "logo.png"
    logo.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    with TestClient(create_app(replace(cfg, logo=logo), start_consumer=False), client=("127.0.0.1", 50000)) as c:
        assert c.get("/api/barber/live").json()["site"]["logo"] is True
        r = c.get("/api/barber/logo")
        assert r.status_code == 200 and r.content == logo.read_bytes() and r.headers["content-type"] == "image/png"


def test_live_payload_has_the_shop_day_and_lost_and_returning(tmp_path):
    with client(tmp_path) as c:
        live = c.get("/api/barber/live").json()
        assert len(live["shop_day"]) == 10 and live["day_start_hour"] == 5
        assert "lost" in live["today"] and "returning_visits" in live["today"]


def test_day_start_hour_is_bounded(tmp_path):
    with client(tmp_path) as c:
        assert c.put("/api/barber/policy", json={"day_start_hour": 24}).status_code == 400
        assert c.put("/api/barber/policy", json={"day_start_hour": 4}).json()["day_start_hour"] == 4


def seed_customer(tmp_path):
    import itertools
    from datetime import datetime
    import numpy as np
    from domain import EngineUpdate, FaceCandidate, Registration, Visit
    from policy import effective
    from register import apply
    from storage import apply_update, connect
    conn = connect(str(tmp_path / "barber-mc.db"))
    t = datetime(2026, 9, 25, 10)
    v = Visit("v1", "p1", t, True, state="finished", confirmed=True, waiting_started_at=t, haircuts=1,
              exited_at=t, outcome="completed", open=False)
    apply_update(conn, EngineUpdate(visits={"v1": v}), t)
    (tmp_path / "crops" / "faces").mkdir(parents=True, exist_ok=True)
    faces = []
    for i, e in enumerate([(1.0, 0.0), (0.99, 0.14)]):
        (tmp_path / "crops" / "faces" / f"{i}.jpg").write_bytes(b"\xff\xd8jpeg")
        a = np.asarray(e + (0.0,) * 510, np.float32)
        faces.append(FaceCandidate("barber-c", 1, 0.0, f"faces/{i}.jpg", tuple(a / np.linalg.norm(a)), 0.9 - i / 10))
    n = itertools.count()
    (cid,) = apply(conn, EngineUpdate(registrations=[Registration("v1", t, tuple(faces), ())]),
                   tmp_path / "crops", effective({}), lambda: f"cust{next(n)}")
    return cid


def test_customers_api_lists_details_and_deletes(tmp_path):
    with client(tmp_path) as c:
        cid = seed_customer(tmp_path)
        page = c.get("/api/barber/customers").json()
        assert page["total"] == 1 and page["rows"][0]["id"] == cid and page["rows"][0]["thumb"]
        detail = c.get(f"/api/barber/customers/{cid}").json()
        assert detail["visit_count"] == 1 and detail["visits"][0]["id"] == "v1" and len(detail["faces"]) == 2
        img = c.get(detail["faces"][0]["url"])
        assert img.status_code == 200 and img.content.startswith(b"\xff\xd8")
        assert c.delete(f"/api/barber/customers/{cid}").status_code == 200
        assert c.get(f"/api/barber/customers/{cid}").status_code == 404
        assert not (tmp_path / "crops" / "faces" / "0.jpg").exists()


def test_crop_api_serves_only_register_rows(tmp_path):
    with client(tmp_path) as c:
        (tmp_path / "secret.jpg").write_bytes(b"x")
        assert c.get("/api/barber/crops/faces/999").status_code == 404
        assert c.get("/api/barber/crops/../../secret.jpg").status_code == 404
        assert c.get("/api/barber/crops/other/1").status_code == 404


def test_crops_are_refused_to_a_public_client(tmp_path):
    seed_customer(tmp_path)
    cfg = ServiceConfig(redis_url="redis://localhost:1/0", frame=FrameGeometry(1000, 1000),
                        cameras=(CameraConfig("barber-c"),), data_dir=tmp_path,
                        stream_starts_file=tmp_path / "starts.json", crops_dir=tmp_path / "crops")
    with TestClient(create_app(cfg, start_consumer=False), client=("8.8.8.8", 1)) as c:
        assert c.get("/api/barber/crops/faces/1").status_code == 403


def test_alert_snapshot_is_refused_to_a_public_client_and_404_without_a_picture(tmp_path):
    with client(tmp_path) as c:
        assert c.get("/api/barber/alerts/nope/snapshot").status_code == 404
    cfg = ServiceConfig(redis_url="redis://localhost:1/0", frame=FrameGeometry(1000, 1000),
                        cameras=(CameraConfig("barber-c"),), data_dir=tmp_path,
                        stream_starts_file=tmp_path / "starts.json", crops_dir=tmp_path / "crops")
    with TestClient(create_app(cfg, start_consumer=False), client=("8.8.8.8", 1)) as c:
        assert c.get("/api/barber/alerts/nope/snapshot").status_code == 403


def test_crops_follow_the_forwarded_client_behind_a_local_proxy(tmp_path):
    """The Vite dev proxy connects from 127.0.0.1 and forwards the browser's
    address in X-Forwarded-For; the guard must judge the browser, not the proxy."""
    with client(tmp_path) as c:
        seed_customer(tmp_path)
        assert c.get("/api/barber/crops/faces/1", headers={"X-Forwarded-For": "8.8.8.8"}).status_code == 403
        assert c.get("/api/barber/crops/faces/1", headers={"X-Forwarded-For": "8.8.8.8, 192.168.1.20"}).status_code == 200
        assert c.get("/api/barber/crops/faces/1", headers={"X-Forwarded-For": "192.168.1.20"}).status_code == 200
        assert c.get("/api/barber/crops/faces/1").status_code == 200


def test_forwarded_for_is_ignored_from_an_untrusted_peer(tmp_path):
    seed_customer(tmp_path)
    cfg = ServiceConfig(redis_url="redis://localhost:1/0", frame=FrameGeometry(1000, 1000),
                        cameras=(CameraConfig("barber-c"),), data_dir=tmp_path,
                        stream_starts_file=tmp_path / "starts.json", crops_dir=tmp_path / "crops")
    with TestClient(create_app(cfg, start_consumer=False), client=("8.8.8.8", 1)) as c:
        assert c.get("/api/barber/crops/faces/1", headers={"X-Forwarded-For": "192.168.1.20"}).status_code == 403


def test_stats_day_buckets_start_at_the_policy_hour(tmp_path):
    with client(tmp_path) as c:
        c.put("/api/barber/policy", json={"day_start_hour": 5})
        st = c.get("/api/barber/stats", params={"from": "2026-09-24", "to": "2026-09-26", "bucket": "day"}).json()
        assert [b["bucket"] for b in st["series"]] == ["2026-09-24T05:00:00", "2026-09-25T05:00:00"]


def test_live_today_has_the_door_visitor_count(tmp_path):
    """Spec 4 / final review I-4: /live carries the door count (non-staff "in" crossings today)."""
    with client(tmp_path) as c:
        assert c.get("/api/barber/live").json()["today"]["visitors"] == 0


def test_policy_endpoint_takes_per_chair_overrides(tmp_path):
    """2026-10-05 (far-chair per-chair rules): "<key>@<chair>" keys go through the policy endpoint like any knob."""
    with client(tmp_path) as c:
        r = c.put("/api/barber/policy", json={"ce_seat_min@R9": 0.08, "ce_end_gap_s@R9": 400})
        assert r.status_code == 200 and r.json()["ce_seat_min@R9"] == 0.08
        got = c.get("/api/barber/policy").json()
        assert got["ce_end_gap_s@R9"] == 400 and got["ce_seat_min"] == 0.10
        assert c.put("/api/barber/policy", json={"bogus@R9": 1}).status_code == 400
        assert c.put("/api/barber/policy", json={"ce_seat_min@R9": 0}).status_code == 400
        assert c.put("/api/barber/policy", json={"ce_lower_frac@R9": 0.5}).status_code == 400


def test_visit_endpoints_are_gone_customer_endpoints_stay(tmp_path):
    with client(tmp_path) as c:
        assert c.get("/api/barber/visits").status_code in (404, 405)
        assert c.get("/api/barber/customers").status_code == 200


def test_visit_events_route_is_gone(tmp_path):
    with client(tmp_path) as c:
        assert c.get("/api/barber/visits/x/events").status_code in (404, 405)
        assert c.get("/api/barber/customers").status_code == 200


def test_live_payload_carries_the_zone_display_switches(tmp_path):
    with client(tmp_path) as c:
        c.put("/api/barber/policy", json={"zone_show_seat": 0, "zone_show_face_area": 0})
        d = c.get("/api/barber/live").json()["zone_display"]
        assert d["seat"] is False and d["chair"] is True and d["face_area"] is False
        assert set(d) == {"entrance_line", "chair", "seat", "waiting", "washing", "row", "door_area", "face_area"}


def test_a_pair_half_on_a_camera_the_site_leaves_out_is_parked(tmp_path):
    """The live Palm run leaves ph-4 out; its stored R3/R4 halves keep the ph-6 owners from showing as half-drawn."""
    (tmp_path / "zones.json").write_text(json.dumps({
        "ph-1": [{"id": "1-R3", "type": "chair", "points": SQ, "pair": "R3", "owner": True},
                 {"id": "1-R3-seat", "type": "seat", "points": SQ, "seat_of": "1-R3"}],
        "ph-4": [{"id": "4-R3", "type": "chair", "points": SQ, "pair": "R3"}]}))
    assert site_client(tmp_path).get("/api/monitoring/zones/problems").json() == []


def test_resume_check_scores_per_chair_over_shop_days(tmp_path):
    """2026-10-18 (user): every re-ID score is kept; per chair, same / new customer mean, min, max over shop days."""
    from domain import EngineUpdate, ResumeCheckRecord
    from storage import apply_update, connect
    c = client(tmp_path)
    conn = connect(str(tmp_path / "barber-mc.db"))
    t = datetime(2026, 10, 17, 12)
    recs = [ResumeCheckRecord("L6", "return", t, t, 90.0, sim, 0.62, same, "h", "h0", 5, 5)
            for sim, same in ((0.9, True), (0.7, True), (0.4, False))]
    late = datetime(2026, 10, 18, 3)                     # 03:00 on the 18th: still the 17th's shop day (05:00)
    recs.append(ResumeCheckRecord("R2", "reseat", late, late, 60.0, 0.95, 0.62, True, "h", "h", 5, 5))
    apply_update(conn, EngineUpdate(resume_checks=recs), t)
    r = c.get("/api/barber/resume-checks", params={"first": "2026-10-17", "last": "2026-10-17"})
    assert r.status_code == 200
    body = r.json()
    assert (body["first"], body["last"]) == ("2026-10-17", "2026-10-17")
    l6, r2 = body["chairs"]
    assert l6["chair_id"] == "L6" and l6["same"]["n"] == 2 and l6["same"]["mean"] == 0.8 and l6["new"]["max"] == 0.4
    assert r2["chair_id"] == "R2" and r2["same"]["n"] == 1
    assert c.get("/api/barber/resume-checks", params={"first": "2026-10-18", "last": "2026-10-18"}).json()["chairs"] == []
    assert c.get("/api/barber/resume-checks", params={"first": "x", "last": "2026-10-18"}).status_code == 400
