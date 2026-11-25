import pytest

from config import SERVICE_DIR, load_config
from saved_reports import SavedReports


def write(tmp_path, text):
    p = tmp_path / "barber.yaml"
    p.write_text(text)
    return p


def test_loads_cameras_frame_and_env_overrides(tmp_path):
    p = write(tmp_path, """
redis_url: redis://yaml:6379/0
frame: {width: 1920, height: 1080}
cameras:
  - {id: shop, clip_duration_s: 300.5, clear_on_loop: true}
  - {id: live}
""")
    cfg = load_config(p, env={"REDIS_URL": "redis://env:6379/1", "BARBER_DATA_DIR": str(tmp_path / "d"),
                              "COMPOSE_PROJECT_NAME": "km"})
    assert cfg.redis_url == "redis://env:6379/1"
    assert (cfg.frame.width, cfg.frame.height) == (1920.0, 1080.0)
    assert cfg.camera("shop").clip_duration_s == 300.5 and cfg.camera("shop").clear_on_loop
    assert cfg.camera("live").clip_duration_s is None
    assert cfg.db_path == tmp_path / "d" / "barber-mc.db"


def test_requires_a_camera(tmp_path):
    with pytest.raises(ValueError, match="no cameras"):
        load_config(write(tmp_path, "frame: {width: 10, height: 10}\n"), env={})


def test_crops_dir_defaults_next_to_the_repo_data(tmp_path):
    p = write(tmp_path, "cameras:\n  - {id: a}\n")
    cfg = load_config(p, env={"BARBER_DATA_DIR": str(tmp_path / "d")})
    assert cfg.crops_dir.name == "crops" and cfg.crops_dir.parent.name == "barber"
    assert load_config(p, env={"BARBER_CROPS_DIR": str(tmp_path / "c")}).crops_dir == tmp_path / "c"


def test_engine_defaults_to_shop_and_reads_chair(tmp_path):
    p = write(tmp_path, "cameras:\n  - {id: shop}\n")
    assert load_config(p, env={}).engine == "shop"
    p = write(tmp_path, "cameras:\n  - {id: shop}\nengine: chair\n")
    assert load_config(p, env={}).engine == "chair"
    p = write(tmp_path, "cameras:\n  - {id: shop}\nengine: bogus\n")
    with pytest.raises(ValueError, match="engine"):
        load_config(p, env={})


def test_camera_names_wait_camera_and_site_policy(tmp_path):
    p = write(tmp_path, """
cameras:
  - {id: ph-1, name: Wash area}
  - {id: ph-3}
wait_camera: ph-3
policy: {wt_end_mode: 1, wt_serve_s: 150, visits_source: 1}
""")
    cfg = load_config(p, env={})
    assert cfg.camera("ph-1").name == "Wash area" and cfg.camera("ph-3").name is None
    assert cfg.wait_camera == "ph-3"
    assert cfg.policy == {"wt_end_mode": 1.0, "wt_serve_s": 150.0, "visits_source": 1.0}


def test_no_wait_camera_and_no_policy_block_is_bekky(tmp_path):
    cfg = load_config(write(tmp_path, "cameras: [{id: barber-c}]\n"), env={})
    assert cfg.wait_camera is None and cfg.policy == {}


def test_wait_camera_must_be_a_configured_camera(tmp_path):
    with pytest.raises(ValueError, match="wait_camera ph-9 is not one of the cameras"):
        load_config(write(tmp_path, "cameras: [{id: ph-1}]\nwait_camera: ph-9\n"), env={})


def test_site_policy_block_is_validated(tmp_path):
    with pytest.raises(ValueError, match="policy"):
        load_config(write(tmp_path, "cameras: [{id: ph-1}]\npolicy: {no_such_knob: 1}\n"), env={})
    with pytest.raises(ValueError, match="policy"):
        load_config(write(tmp_path, "cameras: [{id: ph-1}]\npolicy: {wt_return_sim: 3}\n"), env={})


def test_the_palm_hills_site_file_loads():
    from config import SERVICE_DIR
    cfg = load_config(SERVICE_DIR / "barber.palm-hills.yaml", env={})
    assert [c.id for c in cfg.cameras] == [f"ph-{i}" for i in range(1, 8)]
    assert all(c.clip_duration_s == 7200 and c.name for c in cfg.cameras)
    assert cfg.wait_camera == "ph-3" and cfg.data_dir == SERVICE_DIR / "data" / "palm-hills"
    assert cfg.policy["wt_end_mode"] == 1.0 and cfg.policy["visits_source"] == 1.0 and cfg.policy["wt_serve_s"] == 150.0


def test_every_site_has_its_own_crops_root():
    # Each service's orphan sweep walks its whole crops root and deletes what its own DB doesn't reference: two
    # sites sharing a root (or one root inside another) delete each other's customer photos (final review C1).
    from config import SERVICE_DIR
    sites = sorted(SERVICE_DIR.glob("barber*.yaml"))
    assert len(sites) >= 2
    # sites on one data_dir share one DB (palm-hills and palm-hills-live): one DB, one sweep, one root is fine
    cfgs = {p.name: load_config(p, env={}) for p in sites}
    for a, ca in cfgs.items():
        for b, cb in cfgs.items():
            ra, rb = ca.crops_dir.resolve(), cb.crops_dir.resolve()
            if a != b and ca.data_dir != cb.data_dir:
                assert ra != rb and not ra.is_relative_to(rb), f"{a}'s crops root {ra} is {b}'s or inside it"


def test_each_site_names_its_place_and_branch():
    from config import SERVICE_DIR, load_config
    bekky = load_config(env={"BARBER_CONFIG": str(SERVICE_DIR / "barber.yaml")})
    palm = load_config(env={"BARBER_CONFIG": str(SERVICE_DIR / "barber.palm-hills.yaml")})
    assert (bekky.place, bekky.branch, bekky.branch_slug) == ("BekkyBarber", "Madinty", "madinty")
    assert (palm.place, palm.branch, palm.branch_slug) == ("BekkyBarber", "Palm Hills", "palmhills")


def test_a_branch_without_a_slug_gets_one(tmp_path):
    from config import SERVICE_DIR, load_config
    raw = (SERVICE_DIR / "barber.yaml").read_text().replace("branch_slug: madinty\n", "")
    p = tmp_path / "b.yaml"
    p.write_text(raw.replace("branch: Madinty", "branch: New Cairo 2"))
    assert load_config(env={"BARBER_CONFIG": str(p)}).branch_slug == "newcairo2"


def test_a_derived_slug_is_ascii(tmp_path):
    from config import SERVICE_DIR, load_config
    raw = (SERVICE_DIR / "barber.yaml").read_text().replace("branch_slug: madinty\n", "")
    p = tmp_path / "b.yaml"
    p.write_text(raw.replace("branch: Madinty", "branch: مدينتي 2"))
    assert load_config(env={"BARBER_CONFIG": str(p)}).branch_slug == "2"


def test_a_site_logo_is_read_when_the_file_exists(tmp_path):
    from config import SERVICE_DIR, load_config
    logo = tmp_path / "shop.png"
    logo.write_bytes(b"\x89PNG")
    raw = (SERVICE_DIR / "barber.yaml").read_text() + f"\nlogo: {logo}\n"
    p = tmp_path / "b.yaml"
    p.write_text(raw)
    assert load_config(env={"BARBER_CONFIG": str(p)}).logo == logo
    p.write_text(raw.replace(str(logo), str(tmp_path / "missing.png")))
    assert load_config(env={"BARBER_CONFIG": str(p)}).logo is None


def test_footage_start_is_read_quoted_or_not_and_a_site_can_clear_it(tmp_path):
    """2026-10-18 (user): a run on recordings reports the footage's time; footage_start is when its clips were filmed."""
    from datetime import datetime
    from config import SERVICE_DIR
    base = (SERVICE_DIR / "barber.yaml").read_text()
    p = tmp_path / "b.yaml"
    for line in ("footage_start: 2026-10-16T09:29:58", 'footage_start: "2026-10-16T09:29:58"'):
        p.write_text(base + f"\n{line}\n")
        assert load_config(env={"BARBER_CONFIG": str(p)}).footage_start == datetime(2026, 10, 16, 9, 29, 58)
    (tmp_path / "live.yaml").write_text("extends: b.yaml\nfootage_start: null\n")
    assert load_config(env={"BARBER_CONFIG": str(tmp_path / "live.yaml")}).footage_start is None
    p.write_text(base + "\nfootage_start: not a time\n")
    with pytest.raises(ValueError, match="footage_start"):
        load_config(env={"BARBER_CONFIG": str(p)})


def test_a_site_file_can_extend_another(tmp_path):
    """2026-10-15, the live Palm run: barber.palm-hills-live.yaml = the clip site with other cameras. Nested dicts
    merge key by key, lists (cameras) are replaced -- the sidecar overlay's rule."""
    (tmp_path / "base.yaml").write_text("place: P\nbranch: B\nwait_camera: a\npolicy: {wt_serve_s: 150, wt_out_s: 3}\n"
                                        "cameras:\n  - {id: a, clip_duration_s: 7200}\n  - {id: b}\n")
    live = write(tmp_path, "extends: base.yaml\npolicy: {wt_out_s: 5}\ncameras:\n  - {id: a}\n")
    cfg = load_config(live, env={})
    assert [(c.id, c.clip_duration_s) for c in cfg.cameras] == [("a", None)]
    assert cfg.policy == {"wt_serve_s": 150.0, "wt_out_s": 5.0} and (cfg.place, cfg.wait_camera) == ("P", "a")


def test_an_extends_cycle_is_an_error(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: barber.yaml\n")
    with pytest.raises(ValueError, match="extends cycle: "):
        load_config(write(tmp_path, "extends: a.yaml\n"), env={})


def test_the_palm_hills_live_site_drops_ph4_and_the_clip_settings():
    from config import SERVICE_DIR
    clips = load_config(SERVICE_DIR / "barber.palm-hills.yaml", env={})
    live = load_config(SERVICE_DIR / "barber.palm-hills-live.yaml", env={})
    assert [c.id for c in live.cameras] == ["ph-1", "ph-2", "ph-3", "ph-5", "ph-6", "ph-7"]
    assert all(c.clip_duration_s is None and not c.clear_on_loop and c.name for c in live.cameras)
    assert (live.data_dir, live.crops_dir, live.policy) == (clips.data_dir, clips.crops_dir, clips.policy)


def test_palm_hills_needs_five_minutes_of_sitting_for_a_haircut():
    """2026-10-18 (user): staff the classifier misses opened haircuts of a few minutes (23 of 110 on 16-17 Oct under
    5 min); a sitting becomes a haircut only after 5 minutes there. Bekky keeps the default."""
    from config import SERVICE_DIR
    from policy import DEFAULT_POLICY
    for site in ("barber.palm-hills.yaml", "barber.palm-hills-live.yaml"):
        assert load_config(SERVICE_DIR / site, env={}).policy["ce_min_haircut_s"] == 300.0
    assert "ce_min_haircut_s" not in load_config(SERVICE_DIR / "barber.yaml", env={}).policy
    assert DEFAULT_POLICY["ce_min_haircut_s"] == 60.0



def test_palm_hills_vip_haircuts_need_a_detection_the_detector_is_sure_of():
    """2026-10-22 (user): clothes on VIP2 were a haircut; the confidence check is on the two VIP chairs only."""
    from config import SERVICE_DIR
    from policy import DEFAULT_POLICY
    for site in ("barber.palm-hills.yaml", "barber.palm-hills-live.yaml", "barber.palm-hills-16oct.yaml"):
        pol = load_config(SERVICE_DIR / site, env={}).policy
        assert {k: v for k, v in pol.items() if k.startswith("ce_object_conf")} == {
            "ce_object_conf@ph7-VIP1": 0.72, "ce_object_conf@ph7-VIP2": 0.72}
    assert not any(k.startswith("ce_object_conf") for k in load_config(SERVICE_DIR / "barber.yaml", env={}).policy)
    assert DEFAULT_POLICY["ce_object_conf"] == 0.0

def test_palm_hills_barber_cover_lasts_four_minutes_on_l5_l6_r2_l7():
    """2026-10-18 (user): customers on these chairs went "away" under their barber; his cover holds them 4 min there,
    the default 2 min elsewhere."""
    from config import SERVICE_DIR
    from policy import DEFAULT_POLICY, effective, policy_for
    for site in ("barber.palm-hills.yaml", "barber.palm-hills-live.yaml"):
        p = effective(None, base={**DEFAULT_POLICY, **load_config(SERVICE_DIR / site, env={}).policy})
        assert {c: policy_for(p, c)["ce_carry_max_s"] for c in ("ph3-L5", "L6", "ph5-R1", "L7", "L8", "ph5-R2")} == \
            {"ph3-L5": 240.0, "L6": 240.0, "ph5-R1": 240.0, "L7": 240.0, "L8": 120.0, "ph5-R2": 120.0}


def test_palm_hills_takes_haircut_photos():
    """2026-10-22 (user): the haircut photo report is for Palm Hills; Bekky stays without."""
    from config import SERVICE_DIR
    for site in ("barber.palm-hills.yaml", "barber.palm-hills-live.yaml", "barber.palm-hills-16oct.yaml"):
        assert load_config(SERVICE_DIR / site, env={}).policy["hp_photos"] == 1.0
    assert "hp_photos" not in load_config(SERVICE_DIR / "barber.yaml", env={}).policy


def test_palm_hills_saves_its_reports_and_madinty_does_not():
    want = SavedReports(("en", "ar"), ("pdf", "html"), 5)
    for site in ("barber.palm-hills.yaml", "barber.palm-hills-live.yaml", "barber.palm-hills-16oct.yaml"):
        assert load_config(SERVICE_DIR / site, env={}).saved_reports == want, site
    assert load_config(SERVICE_DIR / "barber.yaml", env={}).saved_reports is None


def test_the_old_daily_report_key_is_refused_with_its_new_name(tmp_path):
    y = tmp_path / "barber.yaml"
    y.write_text('cameras: [{id: c1}]\ndaily_report: {at: "19:00"}\n')
    with pytest.raises(ValueError, match="daily_report was renamed saved_reports"):
        load_config(y, env={})
