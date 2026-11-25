"""tools/zones/validate_reload.py's measurement logic, off-GPU: when a probe stream shows a zone shape in effect."""
import importlib.util
import pathlib
import time

import fakeredis

SPEC = importlib.util.spec_from_file_location(
    "validate_reload", pathlib.Path(__file__).resolve().parents[1] / "tools" / "zones" / "validate_reload.py")
vr = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(vr)


def _probe(r, at, objects, labels=()):
    r.xadd(vr.PROBE + "cam-a", {"pts": "0", "at": f"{at:.3f}", "objects": objects, "labels": ",".join(labels)})


def _watch(r, shape, t0, version=3, timeout=0.3):
    vr.watch.cursor = {"cam-a": "0-0"}
    return vr.watch(r, "cam-a", shape, version, t0, timeout)


def test_b_applies_on_the_first_frame_carrying_the_full_frame_zone():
    r, t0 = fakeredis.FakeRedis(), time.time()
    _probe(r, t0 - 1, 1, ["z1"])                     # before the save: ignored
    _probe(r, t0 + 0.1, 2, ["z0"])
    _probe(r, t0 + 0.4, 1, ["z1"])
    r.set(f"{vr.APPLIED}:cam-a", 3)
    out = _watch(r, "B", t0)
    assert out["effect_frame_s"] == 0.4 and not out["timed_out"] and out["ack_s"] is not None


def test_a_needs_two_object_frames_in_a_row_without_z1():
    r, t0 = fakeredis.FakeRedis(), time.time()
    _probe(r, t0 + 0.1, 1, [])                       # one edge frame...
    _probe(r, t0 + 0.2, 2, ["z1"])                   # ...then B is still loaded
    _probe(r, t0 + 0.3, 0, [])                       # nobody in the frame: neither confirms nor breaks
    _probe(r, t0 + 0.5, 1, ["z0"])
    _probe(r, t0 + 0.6, 1, ["z0"])
    out = _watch(r, "A", t0)
    assert out["effect_frame_s"] == 0.5


def test_no_effect_times_out():
    r, t0 = fakeredis.FakeRedis(), time.time()
    _probe(r, t0 + 0.1, 1, ["z1"])
    out = _watch(r, "A", t0, timeout=0.2)
    assert out["timed_out"] and out["effect_s"] is None and out["ack_s"] is None


def test_a_is_not_applied_on_frames_without_labels():
    """After (c) removes cam-a's section, frames carry no labels: a re-saved A must not count until z0 is back."""
    r, t0 = fakeredis.FakeRedis(), time.time()
    for k in range(5):
        _probe(r, t0 + 0.1 * (k + 1), 2, [])
    assert _watch(r, "A", t0, timeout=0.2)["timed_out"]
    _probe(r, t0 + 0.7, 1, ["z0"])
    _probe(r, t0 + 0.8, 1, ["z0"])
    out = _watch(r, "A", t0)
    assert out["effect_frame_s"] == 0.7 and not out["timed_out"]


def test_wait_for_people_needs_a_run_of_object_frames():
    r, t0 = fakeredis.FakeRedis(), time.time()
    for objects in (1, 1, 0, *[1] * (vr.PEOPLE_RUN - 1)):   # a run broken by an empty frame does not count
        _probe(r, t0, objects)
    try:
        vr.wait_for_people(r, "cam-a", 0.3, start="0-0")
        raise AssertionError("an incomplete run of object frames was accepted")
    except TimeoutError:
        pass
    _probe(r, t0, 1)
    vr.wait_for_people(r, "cam-a", 0.3, start="0-0")
    assert vr.watch.cursor["cam-a"] == r.xrevrange(vr.PROBE + "cam-a", count=1)[0][0]   # watch starts after it


def _report(**c):
    run = {"ack_s": 0.5, "effect_s": 0.5, "frames": 10, "timed_out": False}
    return {"a_latency": {"effect_s": {"min": 0.4, "median": 0.5, "max": 0.5}, "ack_s": None, "timeouts": 0,
                          "meets_target": True},
            "b_only_owner_reloads": {"ok": True, "owner_reload_lines": 10, "edits": 10, "bystander_reload_lines": [],
                                     "bystander_file_unchanged": True},
            "c_removed_section": {"verdict": "removed section stops applying", "frames_still_labelled": 0,
                                  "frames_with_objects_after_ack": 50, "stale_labels": [],
                                  "reload_after_removal": run, "resaved_zones_come_back": True, **c}}


# ── safety guards (fix: never clobber a shared machine's running stack or its Redis) ─────────────────────────

def test_running_iris_services_flags_any_deepstream_sidecar(monkeypatch):
    monkeypatch.setattr(vr, "compose", lambda *a, **k: "redis\niris\ndeepstream-sidecar-2\n")
    assert vr.running_iris_services({}) == ["deepstream-sidecar-2", "iris"]
    monkeypatch.setattr(vr, "compose", lambda *a, **k: "redis\nminio\n")   # nothing of ours running
    assert vr.running_iris_services({}) == []


def test_redis_guard_refuses_a_redis_already_holding_iris_status(monkeypatch):
    import redis
    r = fakeredis.FakeRedis()
    r.set("iris:status:cam-a", b"up")
    monkeypatch.setattr(redis.Redis, "from_url", staticmethod(lambda *a, **k: r))
    assert vr.has_iris_status(r)
    assert vr.redis_iris_status_problem({}) is not None
    r.flushall()
    assert not vr.has_iris_status(r)
    assert vr.redis_iris_status_problem({}) is None


def test_scan_delete_removes_only_the_matching_pattern():
    r = fakeredis.FakeRedis()
    r.set("iris:zones:roi_probe:cam-a", 1)
    r.set("iris:zones:roi_probe:cam-b", 1)
    r.set("someone-elses:key", 1)                        # must survive: FLUSHALL would have taken it
    assert vr.scan_delete(r, "iris:zones:roi_probe:*") == 2
    assert r.exists("someone-elses:key") == 1
    assert r.exists("iris:zones:roi_probe:cam-a") == 0


def test_clear_run_keys_deletes_only_this_runs_keys_and_spares_other_iris_keys():
    """The cleanup must take only the keys this run creates. A blanket iris:* delete (or FLUSHALL) would wipe another
    IRIS product sharing the Redis; this must not."""
    r = fakeredis.FakeRedis()
    r.set("iris:ctrl:zones_version", 5)
    r.set(vr.APPLIED, 5)
    r.set(f"{vr.APPLIED}:cam-a", 5)
    r.xadd(f"{vr.PROBE}cam-a", {"x": 1})
    r.xadd(f"{vr.PROBE}cam-b", {"x": 1})
    r.set("iris:barber:status:lane-1", b"up")           # a different product's key on the same Redis
    r.set("iris:ctrl:some_other_version", 9)            # a sibling ctrl key the run does not own
    removed = vr.clear_run_keys(r)
    assert removed == 5                                 # exactly this run's five keys
    assert r.exists("iris:barber:status:lane-1") == 1  # survived
    assert r.exists("iris:ctrl:some_other_version") == 1
    assert r.exists("iris:ctrl:zones_version") == 0
    assert r.exists(f"{vr.APPLIED}:cam-a") == 0
    assert r.exists(f"{vr.PROBE}cam-b") == 0


# ── Redis ownership gate (first layer: only ever clean our own, empty redis container) ───────────────────────

def test_ownership_refuses_a_redis_from_another_project():
    problem = vr.redis_ownership_problem("iris-zonerun", "6399", owner="someone-elses-project", dbsize=0)
    assert problem is not None and "someone-elses-project" in problem
    assert vr.redis_ownership_problem("iris-zonerun", "6399", owner=None, dbsize=0) is not None   # external Redis


def test_ownership_refuses_a_non_empty_redis():
    problem = vr.redis_ownership_problem("iris-zonerun", "6399", owner="iris-zonerun", dbsize=12)
    assert problem is not None and "DBSIZE=12" in problem


def test_ownership_allows_our_own_empty_redis():
    assert vr.redis_ownership_problem("iris-zonerun", "6399", owner="iris-zonerun", dbsize=0) is None


# ── Tegra-aware memory preflight (fix: int("[N/A]") must not crash) ───────────────────────────────────────────

def test_memory_check_falls_back_to_meminfo_on_tegra_when_smi_reports_na():
    smi = "0, Orin (nvgpu), [N/A], [N/A]"                 # Jetson: nvidia-smi reports memory as [N/A]
    probs, source = vr.memory_problems(smi, 4096, tegra=True, meminfo_mib=8000)
    assert probs == [] and "meminfo" in source.lower()
    probs, source = vr.memory_problems(smi, 4096, tegra=True, meminfo_mib=1000)
    assert len(probs) == 1 and "1000" in probs[0]        # not enough shared memory


def test_memory_check_reads_nvidia_smi_when_the_figures_parse():
    probs, source = vr.memory_problems("0, RTX, 1000, 8000", 4096, tegra=False, meminfo_mib=None)
    assert probs == [] and source == "nvidia-smi"
    probs, _ = vr.memory_problems("0, RTX, 7000, 8000", 4096, tegra=False, meminfo_mib=None)
    assert len(probs) == 1 and "GPU 0" in probs[0]        # 1000 MiB free < 4096


def test_memory_check_never_crashes_on_an_unparsable_value_off_tegra():
    probs, source = vr.memory_problems("0, GPU, [N/A], [N/A]", 4096, tegra=False, meminfo_mib=None)
    assert len(probs) == 1 and "not a Tegra" in probs[0] and source == "nvidia-smi"


def test_summary_passes_only_when_every_step_completed():
    assert vr.summarize(_report(), 2.0) == 0
    assert vr.summarize(_report(resaved_zones_come_back=False,
                                reload_after_removal={"ack_s": 0.9, "effect_s": None, "frames": 256,
                                                      "timed_out": True}), 2.0) == 2
    assert vr.summarize(_report(verdict="no frames with objects observed: inconclusive"), 2.0) == 2
    timed_out = _report()
    timed_out["a_latency"]["timeouts"] = 1
    assert vr.summarize(timed_out, 2.0) == 2
    assert vr.summarize({"error": "TimeoutError: timed out waiting for people in cam-a's frames"}, 2.0) == 2


# ── (b) counted per edit version ─────────────────────────────────────────────────────────────────────────────

def _log(service, entries):
    """A `docker compose logs` excerpt: (version, outcome) -> the zones module's line, with telemetry noise."""
    rebuilt = {"reloaded": "['cam-a']", "unchanged": "no stream"}
    lines = []
    for i, (v, outcome) in enumerate(entries):
        lines.append(f"{service}  | 10:04:{15 + 2 * i:02d} [INFO] iris.sidecar: zones version {v}: analytics config "
                     f"{outcome}; zone state rebuilt for {rebuilt[outcome]}")
        lines.append(f"{service}  | 10:04:{16 + 2 * i:02d} [INFO] iris.sidecar: [telemetry] gauge bus_error_total_cam-a=0")
    return "\n".join(lines)


EDITS = list(range(3, 13))                          # v3..v12: the 10 edits; v2 is the setup save (first A)


def _check(owner, by, **kw):
    kw = {"bystander_file_unchanged": True, "bystander_acked_last_version": True, **kw}
    return vr.reload_check(_log("deepstream-sidecar-1", owner), _log("deepstream-sidecar-1-1", by), EDITS, **kw)


def test_reload_count_ignores_the_setup_save_caught_by_the_log_window():
    # The 2026-11-19 cmubuntu run: the window opened at 10:04:15 and caught the owner's v2 (setup) reload, and the
    # old time-window count scored it 11/10. Per version it is one reload per edit.
    b = _check([(2, "reloaded")] + [(v, "reloaded") for v in EDITS],
               [(2, "unchanged")] + [(v, "unchanged") for v in EDITS])
    assert b["owner_reload_lines"] == 10 and b["bystander_version_lines"] == 10
    assert set(b["owner_reloads_by_version"]) == set(EDITS)
    assert b["ok"]


def test_reload_count_ignores_the_bystanders_baseline_reload():
    b = _check([(v, "reloaded") for v in EDITS], [(1, "reloaded"), (2, "unchanged")] + [(v, "unchanged") for v in EDITS])
    assert b["bystander_reload_lines"] == [] and b["ok"]


def test_a_doubled_reload_does_not_cover_a_missing_one():
    # 10 reload lines in total, but v5 reloaded twice and v6 never: a total count would pass this
    owner = [(v, "reloaded") for v in EDITS if v != 6] + [(5, "reloaded")]
    b = _check(owner, [(v, "unchanged") for v in EDITS])
    assert b["owner_reload_lines"] == 10 and b["owner_reloads_by_version"][6] == 0
    assert not b["ok"]


def test_an_extra_owner_reload_for_one_edit_fails():
    # every edit reloaded, but v8 twice: an extra reload is a failure, not just a missing one
    b = _check([(v, "reloaded") for v in EDITS] + [(8, "reloaded")], [(v, "unchanged") for v in EDITS])
    assert b["owner_reloads_by_version"][8] == 2 and not b["ok"]


def test_owner_unchanged_for_an_edit_fails():
    owner = [(v, "unchanged" if v == 12 else "reloaded") for v in EDITS]
    assert not _check(owner, [(v, "unchanged") for v in EDITS])["ok"]


def test_bystander_reloading_an_edit_version_fails():
    b = _check([(v, "reloaded") for v in EDITS], [(v, "reloaded" if v == 7 else "unchanged") for v in EDITS])
    assert len(b["bystander_reload_lines"]) == 1 and "zones version 7" in b["bystander_reload_lines"][0]
    assert not b["ok"]


def test_bystander_missing_an_edit_version_fails_even_with_the_setup_line():
    by = [(2, "unchanged")] + [(v, "unchanged") for v in EDITS if v != 9]
    b = _check([(v, "reloaded") for v in EDITS], by)
    assert b["bystander_version_lines"] == 9 and not b["ok"]


def test_version_numbers_match_whole():
    # "zones version 1" must not be read as 12, nor 12 as 1
    out = vr.version_outcomes(_log("s", [(1, "reloaded"), (12, "unchanged")]), [1, 12])
    assert [o for o, _ in out[1]] == ["reloaded"] and [o for o, _ in out[12]] == ["unchanged"]


def test_file_or_ack_failures_still_fail():
    owner, by = [(v, "reloaded") for v in EDITS], [(v, "unchanged") for v in EDITS]
    assert not _check(owner, by, bystander_file_unchanged=False)["ok"]
    assert not _check(owner, by, bystander_acked_last_version=False)["ok"]
