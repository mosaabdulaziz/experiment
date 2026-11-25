"""engine_path: where nvinfer will actually serialize each GIE's TRT engine.

sidecar.sidecar needs gi/pyds (GPU container only), so this logic lives in its
own pure module -- same convention as box_color.py and zone_dedup.py."""

import hashlib
import os
from pathlib import Path

import pytest

from perception.engine.engine_path import (assert_distinct_engine_paths, claim_engine,
                                 gie_engine_path, invalidate_stale_engine)


def _write_config(tmp_path, name, *, custom_engine, mode=2, onnx="model.onnx"):
    lines = [f"onnx-file={onnx}", f"network-mode={mode}"]
    if custom_engine:
        lines.append("engine-create-func-name=NvDsInferYoloCudaEngineGet")
    path = tmp_path / name
    path.write_text("[property]\n" + "\n".join(lines) + "\n")
    return str(path)


def test_custom_engine_branch_uses_cwd_and_batch(tmp_path, monkeypatch):
    # Custom-engine paths are cwd-based by definition -- chdir into tmp_path so the
    # test (and invalidate_stale_engine's isfile/getsize probe of that path) never
    # touches the real repo cwd or a real cached engine there.
    monkeypatch.chdir(tmp_path)
    cfg = _write_config(tmp_path, "pgie.txt", custom_engine=True, mode=2)
    assert gie_engine_path(cfg, 8) == f"{os.getcwd()}/model_b8_gpu0_fp16.engine"


def test_custom_engine_branch_mode_suffix_follows_network_mode(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _write_config(tmp_path, "pgie.txt", custom_engine=True, mode=1)
    assert gie_engine_path(cfg, 8).endswith("_int8.engine")


def test_onnx_branch_derives_name_beside_its_own_onnx(tmp_path):
    cfg = _write_config(tmp_path, "fire.txt", custom_engine=False, onnx="fire_s.onnx")
    expected = str(tmp_path / "fire_s.onnx") + "_b8_gpu0_fp16.engine"
    assert gie_engine_path(cfg, 8) == expected


def test_onnx_branch_without_onnx_file_raises(tmp_path):
    path = tmp_path / "broken.txt"
    path.write_text("[property]\nnetwork-mode=2\n")
    with pytest.raises(ValueError, match="onnx-file"):
        gie_engine_path(str(path), 8)


def test_missing_config_raises_unreadable_message_not_missing_key(tmp_path):
    """A config file that cannot be opened at all must not be blamed for a missing
    onnx-file= line -- that sends an operator debugging a boot failure looking for
    the wrong thing (a bad line in a file that isn't even there)."""
    missing = str(tmp_path / "does_not_exist.txt")
    with pytest.raises(ValueError, match="missing or unreadable") as exc:
        gie_engine_path(missing, 8)
    assert "onnx-file" not in str(exc.value)


def test_empty_engine_create_func_name_value_is_not_custom_branch(tmp_path):
    """nvinfer only takes the custom-engine branch when engine-create-func-name
    carries an actual function name; an empty value must be treated the same as
    the key being absent."""
    path = tmp_path / "pgie.txt"
    path.write_text(
        "[property]\nonnx-file=model.onnx\nnetwork-mode=2\nengine-create-func-name=\n")
    expected = str(tmp_path / "model.onnx") + "_b8_gpu0_fp16.engine"
    assert gie_engine_path(str(path), 8) == expected


def test_two_onnx_branch_gies_in_separate_dirs_cannot_collide(tmp_path):
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    a_dir.mkdir()
    b_dir.mkdir()
    a = _write_config(a_dir, "gie.txt", custom_engine=False, onnx="m.onnx")
    b = _write_config(b_dir, "gie.txt", custom_engine=False, onnx="m.onnx")
    assert gie_engine_path(a, 8) != gie_engine_path(b, 8)


def test_assert_distinct_passes_on_unique_paths():
    assert_distinct_engine_paths([
        ("pgie", "/cfg/pgie.txt", "/x/model_b8_gpu0_fp16.engine"),
        ("pgie_fire", "/cfg/fire.txt", "/y/fire_s.onnx_b8_gpu0_fp16.engine"),
    ])


def test_assert_distinct_raises_on_collision_naming_both_configs():
    with pytest.raises(RuntimeError) as exc:
        assert_distinct_engine_paths([
            ("pgie", "/cfg/pgie.txt", "/x/model_b8_gpu0_fp16.engine"),
            ("pgie_fire", "/cfg/fire.txt", "/x/model_b8_gpu0_fp16.engine"),
        ])
    message = str(exc.value)
    assert "/cfg/pgie.txt" in message and "/cfg/fire.txt" in message


def test_assert_distinct_normalizes_equivalent_relative_paths():
    """Two paths that are textually different but resolve to the same real file
    (a "." / ".." traversal difference) must still be caught -- otherwise the
    guard can be defeated by nothing more than how a path happens to be written."""
    with pytest.raises(RuntimeError):
        assert_distinct_engine_paths([
            ("pgie", "/cfg/pgie.txt", "/x/sub/../model_b8_gpu0_fp16.engine"),
            ("pgie_fire", "/cfg/fire.txt", "/x/model_b8_gpu0_fp16.engine"),
        ])


def test_assert_distinct_catches_relative_vs_absolute_spelling(tmp_path, monkeypatch):
    """A relative and an absolute spelling of the SAME file must still collide.
    os.path.normpath is purely lexical -- it never resolves a relative path
    against cwd -- so it would let this slip past as two textually different
    strings. os.path.realpath does resolve against cwd, so this is the guard
    against exactly that gap."""
    monkeypatch.chdir(tmp_path)
    absolute_path = str(tmp_path / "model_b8_gpu0_fp16.engine")
    relative_path = "model_b8_gpu0_fp16.engine"
    with pytest.raises(RuntimeError):
        assert_distinct_engine_paths([
            ("pgie", "/cfg/pgie.txt", absolute_path),
            ("pgie_fire", "/cfg/fire.txt", relative_path),
        ])


def test_two_custom_engine_gies_at_same_batch_and_mode_collide(tmp_path):
    """The exact defect this guard exists for: a fire config copied from the pgie
    config keeps engine-create-func-name and lands on the pgie's engine."""
    a = _write_config(tmp_path, "pgie.txt", custom_engine=True)
    b_dir = tmp_path / "elsewhere"
    b_dir.mkdir()
    b = _write_config(b_dir, "fire.txt", custom_engine=True)
    entries = [("pgie", a, gie_engine_path(a, 8)),
               ("pgie_fire", b, gie_engine_path(b, 8))]
    with pytest.raises(RuntimeError):
        assert_distinct_engine_paths(entries)


def test_invalidate_removes_truncated_engine(tmp_path):
    engine = tmp_path / "model_b8_gpu0_fp16.engine"
    engine.write_bytes(b"x" * 1024)
    cfg = _write_config(tmp_path, "pgie.txt", custom_engine=True)
    invalidate_stale_engine(str(engine), cfg, None)
    assert not engine.exists()


def test_invalidate_keeps_plausible_engine(tmp_path):
    engine = tmp_path / "model_b8_gpu0_fp16.engine"
    engine.write_bytes(b"x" * 2_000_000)
    cfg = _write_config(tmp_path, "pgie.txt", custom_engine=True)
    invalidate_stale_engine(str(engine), cfg, None)
    assert engine.exists()


def test_invalidate_resolves_onnx_relative_to_config_directory_not_cwd(tmp_path, monkeypatch):
    """invalidate_stale_engine must resolve onnx-file the same abspath-based way
    gie_engine_path does, not relative to the process cwd -- so a relative
    gie_config still finds its own onnx and deletes a genuinely stale engine even
    when cwd is a different directory than the config's own (here: tmp_path, the
    config's parent, not the "cfgs" directory the config and its onnx live in)."""
    cfg_dir = tmp_path / "cfgs"
    cfg_dir.mkdir()
    onnx = cfg_dir / "model.onnx"
    onnx.write_bytes(b"onnx")
    cfg = _write_config(cfg_dir, "fire.txt", custom_engine=False, onnx="model.onnx")

    monkeypatch.chdir(tmp_path)
    relative_cfg = os.path.join("cfgs", "fire.txt")
    engine_path = gie_engine_path(relative_cfg, 8)

    engine = Path(engine_path)
    engine.write_bytes(b"x" * 2_000_000)
    old = onnx.stat().st_mtime - 100
    os.utime(engine, (old, old))  # engine now predates the onnx -> stale

    invalidate_stale_engine(engine_path, relative_cfg, "model.onnx")
    assert not engine.exists()


# ── two products, one checkout: the cwd-named engine must belong to one model ─────

def _two_products(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "attendance.onnx").write_bytes(b"A" * 100)
    (tmp_path / "vehicle.onnx").write_bytes(b"V" * 200)
    a = _write_config(tmp_path, "attendance.txt", custom_engine=True, onnx="attendance.onnx")
    v = _write_config(tmp_path, "vehicle.txt", custom_engine=True, onnx="vehicle.onnx")
    return a, v


def test_an_engine_built_for_another_model_is_deleted_not_reused(tmp_path, monkeypatch):
    """Attendance and multi-camera at the same batch size and precision name the SAME engine
    file. The second model must not deserialize the first one's network."""
    a, v = _two_products(tmp_path, monkeypatch)
    engine = Path(gie_engine_path(a, 3))
    engine.write_bytes(b"x" * 2_000_000)          # attendance's engine, built and cached
    claim_engine(str(engine), a, "attendance.onnx")
    assert gie_engine_path(a, 3) == str(engine) and engine.exists()   # same model: kept
    assert gie_engine_path(v, 3) == str(engine)                       # other model, same name...
    assert not engine.exists()                                        # ...so it is rebuilt


def test_the_same_model_keeps_its_engine_across_restarts(tmp_path, monkeypatch):
    a, _ = _two_products(tmp_path, monkeypatch)
    engine = Path(gie_engine_path(a, 3))              # first boot: marker written, then nvinfer builds
    engine.write_bytes(b"x" * 2_000_000)
    for _ in range(3):
        assert gie_engine_path(a, 3) == str(engine)
        assert engine.exists()


def test_an_engine_with_no_marker_is_rebuilt_once(tmp_path, monkeypatch):
    """An engine with no marker cannot be tied to the ONNX it was built from: it is rebuilt once (2026-11), then
    kept, since the new marker names the ONNX's sha256."""
    a, v = _two_products(tmp_path, monkeypatch)
    engine = Path(f"{os.getcwd()}/model_b3_gpu0_fp16.engine")
    engine.write_bytes(b"x" * 2_000_000)
    assert gie_engine_path(a, 3) == str(engine) and not engine.exists()
    marker = Path(str(engine) + ".src").read_text().splitlines()
    assert marker[1] == "100" and marker[2] == "sha256:" + hashlib.sha256(b"A" * 100).hexdigest()
    engine.write_bytes(b"x" * 2_000_000)                       # nvinfer rebuilt it
    assert gie_engine_path(a, 3) == str(engine) and engine.exists()


def test_an_old_marker_without_sha256_means_rebuild(tmp_path, monkeypatch):
    a, _ = _two_products(tmp_path, monkeypatch)
    engine = Path(f"{os.getcwd()}/model_b3_gpu0_fp16.engine")
    engine.write_bytes(b"x" * 2_000_000)
    onnx = tmp_path / "attendance.onnx"
    Path(str(engine) + ".src").write_text(f"{os.path.realpath(onnx)}\n100\n")   # the pre-2026-11 format
    gie_engine_path(a, 3)
    assert not engine.exists()


def test_a_same_size_onnx_swap_triggers_a_rebuild(tmp_path, monkeypatch):
    """A retrain or rollback of the same architecture: same path, same size, other weights."""
    monkeypatch.chdir(tmp_path)
    onnx = tmp_path / "det.onnx"
    onnx.write_bytes(b"A" * 100)
    cfg = _write_config(tmp_path, "det.txt", custom_engine=False, onnx="det.onnx")
    engine = Path(gie_engine_path(cfg, 3))
    engine.write_bytes(b"x" * 2_000_000)
    assert gie_engine_path(cfg, 3) == str(engine) and engine.exists()
    onnx.write_bytes(b"B" * 100)                                # same size
    os.utime(onnx, (engine.stat().st_mtime - 100, engine.stat().st_mtime - 100))   # and older: mtime check passes
    gie_engine_path(cfg, 3)
    assert not engine.exists()


def test_the_manifest_check_digest_is_reused(tmp_path, monkeypatch):
    """The startup check hashed the ONNX already: the engine marker must not read it again."""
    from perception.engine import model_manifest
    monkeypatch.chdir(tmp_path)
    onnx = tmp_path / "det.onnx"
    onnx.write_bytes(b"D" * 300)
    cfg = _write_config(tmp_path, "det.txt", custom_engine=False, onnx="det.onnx")
    model_manifest.file_sha256(onnx)                            # what verify() does for every checked file
    calls = []
    real = model_manifest.sha256_file
    monkeypatch.setattr(model_manifest, "sha256_file", lambda p: calls.append(p) or real(p))
    gie_engine_path(cfg, 3)
    assert calls == []
    onnx.write_bytes(b"E" * 301)                                # a changed file is hashed afresh
    gie_engine_path(cfg, 3)
    assert len(calls) == 1


def test_a_replaced_onnx_of_a_different_size_triggers_a_rebuild(tmp_path, monkeypatch):
    a, _ = _two_products(tmp_path, monkeypatch)
    engine = Path(gie_engine_path(a, 3))
    engine.write_bytes(b"x" * 2_000_000)
    claim_engine(str(engine), a, "attendance.onnx")
    (tmp_path / "attendance.onnx").write_bytes(b"A" * 150)      # re-exported, different size
    old = (tmp_path / "attendance.onnx").stat().st_mtime - 1000
    os.utime(engine, (old + 2000, old + 2000))                    # engine newer: mtime check passes
    gie_engine_path(a, 3)
    assert not engine.exists()


def test_onnx_branch_engines_get_a_marker_too(tmp_path, monkeypatch):
    """Their names carry the ONNX path, not its content: a same-size swap at that path needs the sha256 marker."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "fire.onnx").write_bytes(b"F" * 50)
    cfg = _write_config(tmp_path, "fire.txt", custom_engine=False, onnx="fire.onnx")
    engine = Path(gie_engine_path(cfg, 3))
    assert Path(str(engine) + ".src").read_text().endswith(
        "sha256:" + hashlib.sha256(b"F" * 50).hexdigest() + "\n")
