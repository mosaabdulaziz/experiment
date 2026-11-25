"""The barber sidecar overlay must agree with products/barber's own config:
same camera ids, same face gallery, same frame size. A mismatch fails
silently at runtime (no tracks, no staff names, zones off by a scale)."""
from pathlib import Path

import yaml

from perception.engine.sources import apply_role_overrides, known_roles, load_sidecar_config, normalize_roles

REPO = Path(__file__).resolve().parents[1]


def _barber():
    return load_sidecar_config(str(REPO / "products" / "barber" / "sidecar.yaml"))


def test_overlay_inherits_base_and_overrides_barber_keys():
    base = load_sidecar_config(str(REPO / "perception" / "pipeline" / "sidecar.yaml"))
    cfg = _barber()
    assert cfg["sidecar_role"] == "barber" and "barber" in known_roles(cfg, cfg["sources"])
    assert cfg["pgie_config"] != base["pgie_config"]            # barber runs the seg model (project 3)
    # face-ID tuning is inherited: the barber's recogniser lists no thresholds of its own
    assert set(cfg["modules"]["face_recognition"]) <= {"gallery_key", "enhancement", "model"}
    assert cfg["modules"]["face_detect"]["config"].endswith("config_sgie_face_yolov8n.txt")
    assert cfg["modules"]["face_recognition"]["model"] == "buffalo_l"
    assert cfg["modules"]["customer_faces"]["model"] == "buffalo_l"
    assert cfg["relevant_classes"] == ["person"]
    assert (REPO / cfg["tracker_config"]).is_file()


def test_every_source_is_tagged_barber():
    cfg = apply_role_overrides(_barber(), "barber")
    assert cfg["sources"] and all(normalize_roles(s.get("roles")) == ["barber"] for s in cfg["sources"])
    assert not {"zones", "fire_alerts"} & set(cfg.get("modules") or {})


def test_tracker_reid_model_is_the_one_barber_up_fetches():
    text = (REPO / _barber()["tracker_config"]).read_text()
    tracker = yaml.safe_load(text.split("\n", 1)[1])   # past the %YAML:1.0 directive
    # re-association itself is off (see the tracker config header); the model
    # stays wired so switching it back on needs no download or path changes
    onnx, engine = tracker["ReID"]["onnxFile"], tracker["ReID"]["modelEngineFile"]
    # the repo, as the sidecar container mounts it; the engine where the
    # tracker caches it (next to the model), or every start rebuilds it
    assert onnx.startswith("/workspace/iris/") and engine.startswith(onnx + "_b")
    assert f"f={onnx.removeprefix('/workspace/iris/')}\n" in (REPO / "justfile").read_text()


def test_agrees_with_barber_service_config():
    cfg = _barber()
    service = yaml.safe_load((REPO / "products" / "barber" / "barber.yaml").read_text())
    # barber-service enrols no face gallery (d4cac48); the sidecar keeps its own key only so it
    # never matches against attendance's employee gallery
    attendance = load_sidecar_config(str(REPO / "products" / "attendance" / "sidecar.yaml"))
    assert (cfg["modules"]["face_recognition"]["gallery_key"]
            != attendance["modules"]["face_recognition"]["gallery_key"])
    assert {s["id"] for s in cfg["sources"]} == {c["id"] for c in service["cameras"]}
    assert (cfg["mux_width"], cfg["mux_height"]) == (service["frame"]["width"], service["frame"]["height"])


def test_the_face_sgie_runs_without_a_gallery():
    # customer faces need the face detector; attendance lets it idle while its gallery is empty, barber must not
    # (its own gallery is never filled)
    assert "idle_when_unused" not in _barber()["modules"]["face_detect"]
    assert "idle_when_unused" in load_sidecar_config(
        str(REPO / "products" / "attendance" / "sidecar.yaml"))["modules"]["face_detect"]


def test_barber_runs_without_the_annotated_preview():
    # The barber UI plays go2rtc's raw streams; the annotated preview only
    # cost streaming-thread frame copies and GPU JPEG encodes (2026-09-26).
    assert _barber()["pipeline_preview"] is False
    assert load_sidecar_config(str(REPO / "perception" / "pipeline" / "sidecar.yaml"))["pipeline_preview"] is True


def test_a_wedged_main_loop_is_ended_quickly():
    # A wedge never clears by itself; the default 60 s only lengthens the gap.
    base = load_sidecar_config(str(REPO / "perception" / "pipeline" / "sidecar.yaml"))
    assert 0 < _barber()["main_loop_stall_seconds"] < base["main_loop_stall_seconds"]


def test_the_barber_sidecar_writes_no_core_files():
    # nvstreammux segfaults wrote 12 GB cores into the mounted repo root.
    compose = yaml.safe_load((REPO / "docker-compose.yml").read_text())
    ulimits = compose["services"]["deepstream-sidecar-barber"]["ulimits"]
    assert ulimits["core"] == {"soft": 0, "hard": 0}
    assert ulimits["nofile"] == {"soft": 65536, "hard": 65536}   # kept from the shared anchor


def test_a_camera_that_stays_dead_ends_the_barber_sidecar_within_minutes():
    # 2026-09-26: after the test streams restarted, two cameras stayed dead
    # through an hour of stall-watch restarts. Barber escalates to a process
    # exit; other deployments keep the old behaviour (off).
    from perception.engine.source_recovery import SourceStallWatch
    base = load_sidecar_config(str(REPO / "perception" / "pipeline" / "sidecar.yaml"))
    assert base["source_stall_max_failed_restarts"] == 0
    cfg = _barber()
    assert cfg["source_stall_max_failed_restarts"] > 0
    watch = SourceStallWatch(
        cfg["source_stall_seconds"],
        max_backoff_seconds=cfg["source_stall_max_backoff_seconds"],
        max_failed_restarts=cfg["source_stall_max_failed_restarts"],
        failed_restart_window_seconds=cfg["source_stall_failed_restart_window_seconds"])
    watch.note_buffer(0, now=0.0)
    watch.note_buffer(1, now=0.0)
    gave_up = None
    for t in range(1, 600):
        watch.note_buffer(1, now=float(t))
        if watch.stalled(now=float(t)).hopeless:
            gave_up = t
            break
    assert gave_up is not None and gave_up <= 150


def test_face_sgie_runs_on_sampled_frames_only():
    # the gate is refused with the preview on or a process_fps throttle
    cfg = _barber()
    assert cfg["modules"]["face_detect"]["sample_only"] is True
    assert cfg["pipeline_preview"] is False and not cfg.get("process_fps")
    assert "face_detect" not in (load_sidecar_config(str(REPO / "perception" / "pipeline" / "sidecar.yaml")).get("modules") or {})


def test_barber_uses_the_seg_pgie_and_publishes_masks():
    cfg = _barber()
    assert cfg["pgie_config"] == "/workspace/iris/perception/models/DeepStream-Yolo-Seg/config_infer_primary_yolo26l_seg_r384.txt"
    assert {"person_masks", "person_boxes"} <= set(cfg["modules"])


def test_barber_mux_interpolation_method_is_2_and_base_has_none():
    assert _barber()["mux_interpolation_method"] == 2
    assert "mux_interpolation_method" not in load_sidecar_config(str(REPO / "perception" / "pipeline" / "sidecar.yaml"))


def test_barber_pgie_config_file_is_the_l_seg_384_fp16_mask_model():
    txt = (REPO / "perception/models/DeepStream-Yolo-Seg" / "config_infer_primary_yolo26l_seg_r384.txt").read_text()
    assert "onnx-file=yolo26l-seg-r384.onnx" in txt
    assert "network-mode=2" in txt
    assert "output-instance-mask=1" in txt


def test_barber_face_stage_runs_only_in_the_door_area():
    # face register at the door (2026-10-06): faces on, but only inside the regions barber publishes
    modules = _barber()["modules"]
    assert "customer_faces" in modules and "face_detect" in modules
    assert modules["face_detect"]["regions_key"] == "barber:face_regions"
    assert "capture" not in modules["face_recognition"]          # unknown door faces must not enter the CCTV gallery


def test_base_does_not_publish_masks():
    base = yaml.safe_load((REPO / "perception" / "pipeline" / "sidecar.yaml").read_text())
    assert "person_masks" not in (base.get("modules") or {})


def test_barber_container_preloads_libgcc_s():
    # nvinfer's TensorRT engine build for the seg model crashes when libunwind is loaded first
    compose = yaml.safe_load((REPO / "docker-compose.yml").read_text())
    env = compose["services"]["deepstream-sidecar-barber"]["environment"]
    assert "LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgcc_s.so.1" in env


def test_barber_enables_body_crops():
    # Chair crops stay available for the register's temporary chair-crop source
    # (register_source 0); body_crops_enabled overrides the default.
    cfg = _barber()
    assert cfg["modules"]["appearance"]["crops_dir"]


def test_barber_seg_parser_and_model_are_reproducible_from_the_repo():
    """Final review I-6: the seg parser .so is built into the image (the bind-mounted one is gitignored)
    and the ONNX comes from a just recipe that barber-up runs."""
    import re
    txt = (REPO / "perception/models/DeepStream-Yolo-Seg" / "config_infer_primary_yolo26l_seg_r384.txt").read_text()
    lib = re.search(r"^custom-lib-path=(.+)$", txt, re.M).group(1)
    assert lib == "/opt/nvidia/deepstream/deepstream/custom_parsers/libnvdsinfer_custom_impl_Yolo_seg.so"
    docker = (REPO / "perception" / "pipeline" / "Dockerfile").read_text()
    # the image builds every perception/models/*/nvdsinfer_custom_impl_*/ (perception/models/build_parsers.sh) and
    # installs the library under custom_parsers/; this model's directory is one of them
    assert "build_parsers.sh" in docker and "/opt/nvidia/deepstream/deepstream/custom_parsers" in (
        REPO / "perception/models/build_parsers.sh").read_text()
    makefile = (REPO / "perception/models/DeepStream-Yolo-Seg/nvdsinfer_custom_impl_Yolo_seg/Makefile").read_text()
    assert "TARGET_LIB:= libnvdsinfer_custom_impl_Yolo_seg.so" in makefile
    just = (REPO / "justfile").read_text()
    assert re.search(r"^barber-seg-model\b", just, re.M)
    up = just[just.index("\nbarber-up "):]
    up = up[:up.index("\n\n")]
    assert "just barber-seg-model" in up
    assert "yolo26l-seg-r384.onnx" in just


def test_barber_overlay_comment_names_the_model_it_runs():
    """Final review M-5: the overlay said YOLO26m-seg while running YOLO26l-seg."""
    assert "YOLO26m-seg is the only detector" not in (REPO / "products" / "barber" / "sidecar.yaml").read_text()



def _service(barber_yaml):
    """A site's barber yaml with its `extends:` chain resolved (products/barber/config.py; top-level keys suffice here)."""
    raw = yaml.safe_load(barber_yaml.read_text()) or {}
    base = raw.pop("extends", None)
    return {**_service(barber_yaml.parent / base), **raw} if base else raw


def _sites():
    """(site, its barber yaml, its sidecar config) for every products/barber/barber*.yaml; the pairs come from
    products/barber/site.sh."""
    import shlex
    import subprocess
    service_dir = REPO / "products" / "barber"
    for barber_yaml in sorted(service_dir.glob("barber*.yaml")):
        site = barber_yaml.name.removeprefix("barber.").removesuffix("yaml").rstrip(".") or "bekky"
        out = subprocess.run(["bash", str(service_dir / "site.sh"), site], check=True, capture_output=True,
                             text=True).stdout
        env = dict(kv.split("=", 1) for kv in shlex.split(out.removeprefix("export ")))
        assert Path(env["BARBER_CONFIG"]) == barber_yaml
        sidecar = load_sidecar_config(str(REPO / env["BARBER_SIDECAR_CONFIG"]))
        assert [s["id"] for s in sidecar["sources"]] == [c["id"] for c in _service(barber_yaml)["cameras"]], site
        yield site, barber_yaml, sidecar


def test_each_sites_sidecar_saves_crops_where_its_service_reads_them():
    """Final review C1: per-site crops roots. The sidecar sees the repo as /workspace/iris (docker-compose.yml)."""
    import os
    for site, barber_yaml, sidecar in _sites():
        crops = Path(_service(barber_yaml)["crops_dir"])
        service_root = Path(os.path.normpath(crops if crops.is_absolute() else barber_yaml.parent / crops))
        expected = "/workspace/iris/" + service_root.relative_to(REPO).as_posix()
        assert sidecar["modules"]["appearance"].get("crops_dir") == expected, site
        assert sidecar["modules"]["customer_faces"].get("crops_dir", "/workspace/iris/data/barber/crops") == expected, site


def test_each_sites_face_boxes_are_in_its_services_frame():
    """A customer face's face_box is in capture px; barber-service reads it in its frame (= the mux, like person
    boxes) -- final review I1."""
    for site, barber_yaml, sidecar in _sites():
        frame = _service(barber_yaml)["frame"]
        assert ((sidecar["capture_width"], sidecar["capture_height"]) == (sidecar["mux_width"], sidecar["mux_height"])
                == (frame["width"], frame["height"])), site
