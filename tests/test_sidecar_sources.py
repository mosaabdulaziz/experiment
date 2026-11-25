import pytest
from perception.engine.sources import (resolve_sources, make_stream_state,
                             select_sources, apply_role_overrides, owns_recording,
                             normalize_roles, load_sidecar_config, known_roles)


def test_resolve_sources_list():
    cfg = {"sources": [{"id": "cam0", "uri": "file://a"}, {"id": "cam1", "uri": "rtsp://b"}]}
    assert resolve_sources(cfg) == [
        {"id": "cam0", "uri": "file://a", "roles": []},
        {"id": "cam1", "uri": "rtsp://b", "roles": []},
    ]


def test_resolve_sources_defaults_id_by_index():
    cfg = {"sources": [{"uri": "file://a"}, {"uri": "file://b"}]}
    assert resolve_sources(cfg) == [
        {"id": "cam0", "uri": "file://a", "roles": []},
        {"id": "cam1", "uri": "file://b", "roles": []},
    ]


def test_resolve_sources_skips_entries_without_uri():
    cfg = {"sources": [{"id": "cam0", "uri": "file://a"}, {"id": "cam1"}]}
    assert resolve_sources(cfg) == [{"id": "cam0", "uri": "file://a", "roles": []}]


def test_resolve_sources_single_uri_shim():
    cfg = {"uri": "file://a", "source_id": "cam9"}
    assert resolve_sources(cfg) == [{"id": "cam9", "uri": "file://a", "roles": []}]


def test_resolve_sources_single_uri_default_id():
    cfg = {"uri": "file://a"}
    assert resolve_sources(cfg) == [{"id": "cam0", "uri": "file://a", "roles": []}]


def test_resolve_sources_preserves_roles_tag():
    cfg = {"sources": [{"id": "cam-7", "uri": "file://a", "roles": ["fire"]}]}
    assert resolve_sources(cfg) == [{"id": "cam-7", "uri": "file://a", "roles": ["fire"]}]


def test_resolve_sources_defaults_roles_to_empty_list():
    cfg = {"sources": [{"id": "cam-1", "uri": "file://a"}]}
    assert resolve_sources(cfg) == [{"id": "cam-1", "uri": "file://a", "roles": []}]


def test_resolve_sources_empty():
    assert resolve_sources({}) == []


def test_make_stream_state_keys():
    st = make_stream_state("cam2")
    assert st.meta_key == "iris:meta:cam2"
    assert st.jpeg_key == "iris:jpeg:cam2"
    assert st.status_key == "iris:status:cam2"
    assert st._last_pts is None
    assert st._last_stream == 0.0


def test_make_stream_state_starts_with_no_module_state():
    # Modules keep their per-stream scratch here, keyed by module name.
    st = make_stream_state("cam3")
    assert st.module_state == {}


# ── ViewerGate (viewer-gated preview publishing) ──────────────────────────────

def test_viewer_gate_misses_without_key():
    import fakeredis
    from perception.engine.sources import ViewerGate
    r = fakeredis.FakeStrictRedis()
    gate = ViewerGate(r)
    assert gate.is_watched("cam0") is False


def test_viewer_gate_hits_after_key_appears():
    import fakeredis
    from perception.engine.sources import ViewerGate
    r = fakeredis.FakeStrictRedis()
    gate = ViewerGate(r, check_interval=2.0)
    assert gate.is_watched("cam0") is False
    r.set("iris:watched:cam0", "1")
    # cached miss until the interval elapses
    assert gate.is_watched("cam0") is False
    gate._cache.clear()
    assert gate.is_watched("cam0") is True


def test_viewer_gate_refreshes_after_interval():
    import fakeredis
    from perception.engine.sources import ViewerGate
    r = fakeredis.FakeStrictRedis()
    gate = ViewerGate(r, check_interval=2.0)
    r.set("iris:watched:cam0", "1")
    now = 1000.0
    assert gate.is_watched("cam0", now=now) is True
    r.delete("iris:watched:cam0")
    # still cached
    assert gate.is_watched("cam0", now=now + 0.5) is True
    # past the interval it re-checks and sees the deletion
    assert gate.is_watched("cam0", now=now + 2.1) is False


def test_viewer_gate_fails_open_on_redis_error():
    from perception.engine.sources import ViewerGate
    class _BoomRedis:
        def exists(self, *a, **kw):
            raise RuntimeError("redis down")
    gate = ViewerGate(_BoomRedis())
    assert gate.is_watched("cam0") is True  # never starve previews


def test_viewer_gate_prefix():
    from perception.engine.sources import ViewerGate
    assert ViewerGate.KEY_PREFIX == "iris:watched:"


# ── preview_plan (pipeline_preview mode: annotated JPEGs vs inference-only) ───
# preview_plan(need_meta, need_stream, pipeline_preview, mtmc_active, has_dets) -> (do_preview, need_bgr).
# A module that crops the frame (faces, bodies) asks for pixels on its own, through its wants_pixels hook.

def test_preview_plan_pipeline_mode_publishes_on_stream_cadence():
    from perception.engine.sources import preview_plan
    # pipeline mode: every need_stream frame is drawn+encoded (and copied).
    assert preview_plan(False, True, True) == (True, True)


def test_preview_plan_meta_only_frame_needs_no_copy_by_itself():
    from perception.engine.sources import preview_plan
    assert preview_plan(True, False, True) == (False, False)


def test_preview_plan_raw_mode_never_publishes_jpeg():
    from perception.engine.sources import preview_plan
    # inference-only: do_preview is always False, even on the stream cadence.
    assert preview_plan(False, True, False) == (False, False)
    assert preview_plan(True, True, False) == (False, False)


def test_preview_plan_no_work_at_all():
    from perception.engine.sources import preview_plan
    assert preview_plan(False, False, True) == (False, False)
    assert preview_plan(False, False, False) == (False, False)


def test_stream_state_starts_with_no_module_state():
    assert make_stream_state("cam-1").module_state == {}


# ── select_sources (role + shard filtering) ──────────────────────────────────

def _srcs(*roles_per_source):
    return [{"id": f"cam{i}", "uri": "file://x", "roles": list(r)}
            for i, r in enumerate(roles_per_source)]


def test_select_sources_no_role_no_sharding_returns_everything():
    all_sources = _srcs(["fire"], [], ["attendance"])
    assert select_sources(all_sources, 0, 1, None) == all_sources


def test_select_sources_role_filter_keeps_matching_and_untagged():
    all_sources = _srcs(["fire"], [], ["attendance"])
    result = select_sources(all_sources, 0, 1, "fire")
    assert [s["id"] for s in result] == ["cam0", "cam1"]  # cam1 untagged -> matches everything


def test_select_sources_role_filter_excludes_other_role():
    all_sources = _srcs(["fire"], ["attendance"])
    result = select_sources(all_sources, 0, 1, "attendance")
    assert [s["id"] for s in result] == ["cam1"]


def test_select_sources_role_and_shard_compose():
    all_sources = _srcs([], [], [], [])  # 4 untagged sources
    result = select_sources(all_sources, 1, 2, "fire")
    # role filter keeps all 4 (untagged), then index-modulo keeps odd indices
    assert [s["id"] for s in result] == ["cam1", "cam3"]


def test_select_sources_dual_role_source_matches_both():
    all_sources = _srcs(["fire", "attendance"])
    assert [s["id"] for s in select_sources(all_sources, 0, 1, "fire")] == ["cam0"]
    assert [s["id"] for s in select_sources(all_sources, 0, 1, "attendance")] == ["cam0"]


# ── apply_role_overrides (the role_overrides table in the settings) ─────────

_FIRE = {"fire_alerts": {"interval": 0}}
_ZONES = {"zones": {}}
_TABLE = {
    "fire": {"pgie_enabled": False, "drop_modules": ["zones", "face_detect", "face_recognition", "customer_faces"]},
    "attendance": {"drop_modules": ["fire_alerts"]},
}


def test_apply_role_overrides_none_role_is_a_no_op():
    cfg = {"modules": {**_FIRE, **_ZONES}, "role_overrides": _TABLE}
    assert apply_role_overrides(cfg, None) == cfg


def test_a_role_with_no_entry_changes_nothing():
    cfg = {"modules": {**_FIRE, **_ZONES}, "role_overrides": _TABLE}
    assert apply_role_overrides(cfg, "barber") == cfg
    assert apply_role_overrides({"modules": dict(_FIRE)}, "fire") == {"modules": dict(_FIRE)}   # no table at all


def test_a_role_can_set_engine_keys_and_drop_modules():
    faces = {"face_detect": {}, "face_recognition": {}, "customer_faces": {}}
    cfg = {"modules": {**_FIRE, **_ZONES, **faces, "person_boxes": {}}, "role_overrides": _TABLE}
    result = apply_role_overrides(cfg, "fire")
    assert result["pgie_enabled"] is False
    assert result["modules"] == {**_FIRE, "person_boxes": {}}      # the fire module stays; zones and faces go
    assert "drop_modules" not in result and result["role_overrides"] == _TABLE


def test_a_role_only_takes_modules_away():
    cfg = {"modules": {**_FIRE, **_ZONES, "other": {}}, "role_overrides": _TABLE}
    result = apply_role_overrides(cfg, "attendance")
    assert result["modules"] == {**_ZONES, "other": {}}
    assert "pgie_enabled" not in result


def test_known_roles_are_the_table_plus_the_tags_in_use():
    cfg = {"role_overrides": _TABLE}
    assert known_roles(cfg, _srcs_for_roles(["barber"], [], ["fire"])) == {"fire", "attendance", "barber"}
    assert known_roles({}, []) == set()


def _srcs_for_roles(*roles_per_source):
    return [{"id": f"cam{i}", "uri": "file://x", "roles": list(r)} for i, r in enumerate(roles_per_source)]


def test_load_sidecar_config_overlay_deep_merges_and_replaces_lists(tmp_path):
    (tmp_path / "base.yaml").write_text("a: 1\nnested: {x: 1, y: 2}\nsources: [{id: cam-1}]\n")
    (tmp_path / "over.yaml").write_text("extends: base.yaml\nnested: {y: 3}\nsources: [{id: barber-1}]\n")
    assert load_sidecar_config(str(tmp_path / "over.yaml")) == {
        "a": 1, "nested": {"x": 1, "y": 3}, "sources": [{"id": "barber-1"}]}


def test_load_sidecar_config_plain_file_and_cycle(tmp_path):
    (tmp_path / "plain.yaml").write_text("k: v\n")
    assert load_sidecar_config(str(tmp_path / "plain.yaml")) == {"k": "v"}
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n")
    with pytest.raises(ValueError, match="cycle"):
        load_sidecar_config(str(tmp_path / "a.yaml"))


def test_apply_role_overrides_does_not_mutate_input():
    cfg = {"modules": dict(_FIRE)}
    apply_role_overrides(cfg, "attendance")
    assert cfg == {"modules": _FIRE}


# ── normalize_roles (scalar `roles:` yaml typo) ─────────────────────────────

def test_normalize_roles_wraps_a_bare_string():
    # `roles: fire` in yaml -- list("fire") would give ['f','i','r','e']
    assert normalize_roles("fire") == ["fire"]


def test_normalize_roles_passes_lists_through():
    assert normalize_roles(["fire", "attendance"]) == ["fire", "attendance"]


def test_normalize_roles_empty_and_missing():
    assert normalize_roles(None) == []
    assert normalize_roles([]) == []
    assert normalize_roles("") == []


def test_resolve_sources_wraps_scalar_roles_value():
    cfg = {"sources": [{"id": "cam-7", "uri": "file://a", "roles": "fire"}]}
    assert resolve_sources(cfg) == [{"id": "cam-7", "uri": "file://a", "roles": ["fire"]}]


# ── owns_recording (single-owner scoping for destructive boot flushes) ──────

def test_owns_recording_without_a_role_owns_everything():
    assert owns_recording({"id": "cam-1", "roles": []}, None) is True
    assert owns_recording({"id": "cam-7", "roles": ["fire"]}, None) is True


def test_owns_recording_tagged_source_matches_its_own_role():
    assert owns_recording({"id": "cam-7", "roles": ["fire"]}, "fire") is True


def test_owns_recording_tagged_source_excludes_other_roles():
    assert owns_recording({"id": "cam-7", "roles": ["fire"]}, "attendance") is False


def test_owns_recording_untagged_source_belongs_to_the_named_owner():
    # Deliberately NOT select_sources' "untagged matches every role" default:
    # the flushes are destructive, so exactly one process may own them.
    assert owns_recording({"id": "cam-1", "roles": []}, "attendance", "attendance") is True
    assert owns_recording({"id": "cam-1"}, "attendance", "attendance") is True


def test_owns_recording_untagged_source_is_not_owned_by_another_role_or_by_nobody_named():
    assert owns_recording({"id": "cam-1", "roles": []}, "fire", "attendance") is False
    assert owns_recording({"id": "cam-1", "roles": []}, "attendance") is False        # no owner named: no role owns it


def test_the_attendance_product_names_the_owner_of_untagged_recordings_and_the_shared_settings_name_none():
    base = load_sidecar_config("perception/pipeline/sidecar.yaml")
    assert "untagged_recording_owner" not in base and "role_overrides" not in base
    attendance = load_sidecar_config("products/attendance/sidecar.yaml")
    assert attendance["untagged_recording_owner"] == "attendance"
    assert set(attendance["role_overrides"]) == {"fire", "attendance"}


def test_owns_recording_dual_tagged_source_is_owned_by_both():
    src = {"id": "cam-7", "roles": ["fire", "attendance"]}
    assert owns_recording(src, "fire") is True
    assert owns_recording(src, "attendance") is True


def test_owns_recording_handles_scalar_roles_value():
    assert owns_recording({"id": "cam-7", "roles": "fire"}, "fire") is True
    assert owns_recording({"id": "cam-7", "roles": "fire"}, "attendance") is False
