"""perception/engine/model_manifest.py: the startup verifier (stdlib only), against fixture files in a temp repo."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "perception" / "engine" / "model_manifest.py"

spec = importlib.util.spec_from_file_location("model_manifest_under_test", VERIFIER)   # as the services load it
mm = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mm                    # dataclasses resolve annotations through sys.modules
spec.loader.exec_module(mm)


def test_verifier_imports_only_the_standard_library():
    tree = ast.parse(VERIFIER.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative import: the services load this file by path, outside its package"
            imported.add(node.module.split(".")[0])
    non_stdlib = sorted(imported - set(sys.stdlib_module_names) - {"__future__"})
    assert non_stdlib == [], f"model_manifest.py must stay stdlib-only, imports {non_stdlib}"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / "models" / "det").mkdir(parents=True)
    (r / "models" / "det" / "det.onnx").write_bytes(b"det-v1")
    (r / "models" / "det" / "labels.txt").write_text("thing\n")
    (r / "models" / "det" / "det.onnx.dvc").write_text("outs:\n- md5: x\n  path: det.onnx\n")
    (r / "models" / "det" / "config.txt").write_text(
        "[property]\ngpu-id=0\nonnx-file=det.onnx   # beside the config\n"
        "model-engine-file=det.onnx_b1_gpu0_fp16.engine\nlabelfile-path=/workspace/iris/models/det/labels.txt\n")
    (r / "models" / "cls.onnx").write_bytes(b"cls-v1")
    (r / "models" / "fallback.onnx").write_bytes(b"fallback")
    (r / "models" / "reid.onnx").write_bytes(b"reid")
    (r / "tracker.yml").write_text('ReID:\n  reidType: 0   # off\n  onnxFile: "/workspace/iris/models/reid.onnx"\n')
    return r


def _entry(path, content: bytes, name="m", version="1", alias="production-x"):
    return {"path": path, "registered_model": name, "version": version, "alias": alias, "run_id": "r" * 32,
            "sha256": _sha(content), "size_bytes": len(content), "loaded_by": "test"}


def _manifest(repo, *entries, name="models.manifest.json"):
    p = repo / name
    p.write_text(json.dumps({"schema": 1, "product": "p", "site": None, "generated_at": "2026-11-04T00:00:00+00:00",
                             "generated_from_commit": "abc", "tracking_uri": "sqlite:///x", "models": list(entries)}))
    return name


@pytest.fixture
def full(repo):
    return _manifest(repo, _entry("models/det/det.onnx", b"det-v1", "detector"),
                     _entry("models/det/labels.txt", b"thing\n", "detector"),
                     _entry("models/cls.onnx", b"cls-v1", "classifier"),
                     _entry("models/fallback.onnx", b"fallback", "fallback"),
                     _entry("models/reid.onnx", b"reid", "reid"))


REFS = [mm.ModelRef("/workspace/iris/models/det/config.txt", "pgie_config", "nvinfer"),
        mm.ModelRef("models/cls.onnx", "modules.x.model"),
        mm.ModelRef("tracker.yml", "tracker_config", "tracker")]


def test_matching_files_start_and_hashes_are_returned(repo, full, caplog):
    v = mm.verify(full, REFS, repo, env={})
    assert sorted(v.checked) == ["models/cls.onnx", "models/det/det.onnx", "models/det/labels.txt"]
    assert v.sha256[str((repo / "models" / "det" / "det.onnx").resolve())] == _sha(b"det-v1")
    assert "models/fallback.onnx" not in v.checked              # listed, not loaded by this config: not checked
    assert "models/reid.onnx" not in v.checked                  # tracker reidType 0: no ReID net loaded


def test_hash_mismatch_refuses(repo, full):
    (repo / "models" / "det" / "det.onnx").write_bytes(b"det-v2")
    with pytest.raises(mm.ManifestError) as e:
        mm.verify(full, REFS, repo, env={})
    msg = str(e.value)
    assert msg.startswith("model manifest models.manifest.json:") and msg.endswith("Refusing to start.")
    assert f"models/det/det.onnx is not detector v1 (production-x): sha256 expected {_sha(b'det-v1')[:12]}" in msg
    assert "Run `dvc checkout models/det/det.onnx.dvc`" in msg and "same commit" in msg


def test_missing_file_refuses_with_the_dvc_command(repo, full):
    (repo / "models" / "det" / "det.onnx").unlink()
    (repo / "models" / "cls.onnx").unlink()
    with pytest.raises(mm.ManifestError) as e:
        mm.verify(full, REFS, repo, env={})
    msg = str(e.value)
    assert "lists models/det/det.onnx (detector v1 (production-x)), which does not exist. " \
           "Run `dvc pull models/det/det.onnx.dvc`" in msg
    assert "lists models/cls.onnx (classifier v1 (production-x)), which does not exist. Run (it has no DVC " \
           "pointer" in msg                                    # both problems in one refusal


def test_unlisted_model_refuses(repo):
    name = _manifest(repo, _entry("models/det/det.onnx", b"det-v1"), _entry("models/det/labels.txt", b"thing\n"))
    with pytest.raises(mm.ManifestError, match=r"modules\.x\.model loads models/cls\.onnx, which this manifest does "
                                               r"not list\. Every model a deployment loads must be a registered"):
        mm.verify(name, REFS, repo, env={})


def test_unlisted_nvinfer_companion_names_the_config_key(repo):
    name = _manifest(repo, _entry("models/det/det.onnx", b"det-v1"), _entry("models/cls.onnx", b"cls-v1"))
    with pytest.raises(mm.ManifestError, match=r"pgie_config -> labelfile-path loads models/det/labels\.txt"):
        mm.verify(name, REFS, repo, env={})


def test_tracker_reid_counts_only_when_enabled(repo, full):
    (repo / "tracker.yml").write_text('ReID:\n  reidType: 2\n  onnxFile: "/workspace/iris/models/reid.onnx"\n')
    assert "models/reid.onnx" in mm.verify(full, REFS, repo, env={}).checked


@pytest.mark.parametrize("env, refuses", [({}, False), ({"MODEL_MANIFEST_REQUIRED": "0"}, False),
                                          ({"MODEL_MANIFEST_REQUIRED": "1"}, True),
                                          ({"MODEL_MANIFEST_REQUIRED": "true"}, True)])
@pytest.mark.parametrize("setting", [None, "products/p/models.manifest.json"])
def test_missing_manifest_warns_or_refuses(repo, caplog, env, refuses, setting):
    if refuses:
        with pytest.raises(mm.ManifestError, match="no model manifest for .* MODEL_MANIFEST_REQUIRED is set"):
            mm.verify(setting, REFS, repo, env=env)
    else:
        v = mm.verify(setting, REFS, repo, env=env)
        assert v.manifest is None and "NOT verified" in v.warning and "NOT verified" in caplog.text


def test_an_existing_manifest_is_enforced_without_the_setting(repo, full):
    (repo / "models" / "cls.onnx").write_bytes(b"other")
    with pytest.raises(mm.ManifestError):
        mm.verify(full, REFS, repo, env={"MODEL_MANIFEST_REQUIRED": "0"})


def test_model_pack_loads_every_onnx(repo, tmp_path, monkeypatch):
    home = tmp_path / "home"
    pack = home / ".pack" / "models" / "p1"
    pack.mkdir(parents=True)
    for n in ("a.onnx", "b.onnx"):
        (pack / n).write_bytes(n.encode())
    monkeypatch.setenv("HOME", str(home))
    refs = [mm.ModelRef("~/.pack/models/p1", "modules.y.model=p1", "pack")]
    name = _manifest(repo, _entry("~/.pack/models/p1/a.onnx", b"a.onnx"))
    with pytest.raises(mm.ManifestError, match=r"modules\.y\.model=p1 loads ~/\.pack/models/p1/b\.onnx, which this"):
        mm.verify(name, refs, repo, env={})
    name = _manifest(repo, _entry("~/.pack/models/p1/a.onnx", b"a.onnx"), _entry("~/.pack/models/p1/b.onnx", b"b.onnx"))
    assert sorted(mm.verify(name, refs, repo, env={}).checked) == ["~/.pack/models/p1/a.onnx", "~/.pack/models/p1/b.onnx"]
    for f in pack.iterdir():
        f.unlink()
    pack.rmdir()
    with pytest.raises(mm.ManifestError) as e:                     # the whole pack is gone: its files are missing
        mm.verify(name, refs, repo, env={})
    assert "lists ~/.pack/models/p1/a.onnx" in str(e.value) and "has no DVC pointer" in str(e.value)


def test_unreadable_config_and_malformed_manifest_refuse(repo, full):
    with pytest.raises(mm.ManifestError, match=r"pgie_config names models/none\.txt, which cannot be read"):
        mm.verify(full, [mm.ModelRef("models/none.txt", "pgie_config", "nvinfer")], repo, env={})
    (repo / "bad.json").write_text(json.dumps({"schema": 1, "models": [{"path": "x"}]}))
    with pytest.raises(mm.ManifestError, match=r"models\[0\] lacks"):
        mm.verify("bad.json", REFS, repo, env={})
    (repo / "old.json").write_text(json.dumps({"schema": 2, "models": []}))
    with pytest.raises(mm.ManifestError, match="schema 2"):
        mm.verify("old.json", REFS, repo, env={})


@pytest.mark.parametrize("path, key", [
    ("/workspace/iris/models/a.onnx", "models/a.onnx"),
    ("models/./a.onnx", "models/a.onnx"),
    ("~/.insightface/models/buffalo_l/det_10g.onnx", "~/.insightface/models/buffalo_l/det_10g.onnx"),
    ("/opt/elsewhere/a.onnx", "/opt/elsewhere/a.onnx"),
])
def test_manifest_keys(repo, path, key):
    assert mm.manifest_key(path, repo) == key
    assert mm.manifest_key(repo / "models" / "a.onnx", repo) == "models/a.onnx"
