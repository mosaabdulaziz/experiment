"""The service's model check runs at startup (the lifespan), not at import: importing server.py has no side effect,
and the real service (`uvicorn server:app`, create_app() with no config) still refuses to start when its manifest lists
a model file that is missing or not the registered version.

Fixture manifests and a fixture model only (tmp_path), never the committed deployment manifests: these tests must
not depend on which model files a checkout happens to have pulled.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import server
from config import REPO_DIR, load_config
from model_files import load_verifier

SERVICE_DIR = Path(server.__file__).resolve().parent


def _site(tmp_path: Path, model: Path, listed_sha: str) -> Path:
    """A site yaml whose kids car model is `model` and whose manifest lists it with `listed_sha`."""
    mm = load_verifier(REPO_DIR)
    manifest = tmp_path / "models.manifest.json"
    manifest.write_text(json.dumps({"schema": 1, "product": "barber", "models": [{
        "path": mm.manifest_key(model, REPO_DIR), "registered_model": "kids-car-detector", "version": "1",
        "alias": "production-barber", "run_id": "r" * 32, "sha256": listed_sha, "size_bytes": 7}]}))
    site = tmp_path / "barber.yaml"
    site.write_text(f"model_manifest: {manifest}\nkids_car_model: {model}\ndata_dir: {tmp_path / 'data'}\n"
                    f"crops_dir: {tmp_path / 'crops'}\nredis_url: redis://localhost:1/0\n"
                    "cameras:\n  - {id: barber-c}\n")
    return site


@pytest.fixture
def missing_model_site(tmp_path, monkeypatch):
    site = _site(tmp_path, tmp_path / "kids_car.onnx", hashlib.sha256(b"kids-v1").hexdigest())   # never written
    monkeypatch.setenv("BARBER_CONFIG", str(site))
    monkeypatch.delenv("MODEL_MANIFEST_REQUIRED", raising=False)
    return site


def test_importing_the_server_checks_no_models(missing_model_site):
    """A fresh interpreter: the import is not hidden by an earlier one in this session."""
    env = {**os.environ, "BARBER_CONFIG": str(missing_model_site)}
    done = subprocess.run([sys.executable, "-c", "import server; print(type(server.app).__name__)"],
                          cwd=SERVICE_DIR, env=env, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "FastAPI"


def test_the_real_service_refuses_to_start_when_its_manifest_lists_a_missing_model(missing_model_site):
    app = server.create_app()                       # as `uvicorn server:app` builds it: no config handed in
    refusal = r"(?s)barber service: refusing to start -- .*kids_car\.onnx.*does not exist"
    with pytest.raises(RuntimeError, match=refusal):
        with TestClient(app):
            pass


def test_a_mismatched_model_refuses_and_the_registered_one_starts(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_MANIFEST_REQUIRED", raising=False)
    model = tmp_path / "kids_car.onnx"
    model.write_bytes(b"kids-v1")
    cfg = load_config(env={"BARBER_CONFIG": str(_site(tmp_path, model, hashlib.sha256(b"kids-v1").hexdigest()))})
    assert not cfg.data_dir.exists()
    with TestClient(server.create_app(cfg, start_consumer=False, check_models=True)):
        assert cfg.data_dir.is_dir()                # the lifespan went past the check (it makes data_dir right after)

    model.write_bytes(b"kids-v2")                   # a different file, same path: not the registered version
    with pytest.raises(RuntimeError, match=r"(?s)refusing to start -- .*kids_car\.onnx is not kids-car-detector v1"):
        with TestClient(server.create_app(cfg, start_consumer=False, check_models=True)):
            pass


def test_a_test_config_is_not_checked_unless_asked(tmp_path):
    """The default: a caller handing in its own config (the tests) is not checked; check_models=True opts in."""
    cfg = load_config(env={"BARBER_CONFIG": str(_site(tmp_path, tmp_path / "absent.onnx", "0" * 64))})
    with TestClient(server.create_app(cfg, start_consumer=False)):
        pass
