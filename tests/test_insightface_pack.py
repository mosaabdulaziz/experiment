"""The InsightFace model packs are loaded from the repo's DVC copy everywhere, and never downloaded.

No insightface, no GPU: a fake `insightface.app.FaceAnalysis` records how it is called. A pack folder is faked in a
temp dir (perception.insightface_pack.ROOT patched), so the tests do not need the ONNX weights pulled.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import sys
import types
from pathlib import Path

import pytest

from perception import insightface_pack as ip

REPO = Path(__file__).resolve().parents[1]
PACK = "perception/models/insightface/models/buffalo_l"
FILES = ("1k3d68.onnx", "2d106det.onnx", "det_10g.onnx", "genderage.onnx", "w600k_r50.onnx")
DEFAULT_PACK = "perception/models/insightface/models/auraface"
DEFAULT_FILES = ("glintr100.onnx",)


@pytest.fixture
def pack(tmp_path, monkeypatch):
    """A fake insightface root with initially complete auraface and buffalo_l packs."""
    root = tmp_path / "insightface"
    d = root / "models" / "buffalo_l"
    d.mkdir(parents=True)
    for f in FILES:
        (d / f).write_bytes(b"onnx")
    ad = root / "models" / "auraface"
    ad.mkdir(parents=True)
    for f in DEFAULT_FILES:
        (ad / f).write_bytes(b"onnx")
    monkeypatch.setattr(ip, "ROOT", root)
    return types.SimpleNamespace(root=root, dir=d, aura_dir=ad)


@pytest.fixture
def fake_insightface(monkeypatch):
    """insightface.app.FaceAnalysis that records its kwargs and never downloads."""
    calls: list[dict] = []

    class FaceAnalysis:
        def __init__(self, name="auraface", root="~/.insightface", allowed_modules=None, **kw):
            calls.append({"name": name, "root": root, "allowed_modules": allowed_modules})
            self.models = {"recognition": types.SimpleNamespace(get_feat=lambda x: [[0.0]])}

        def prepare(self, **kw):
            pass

    pkg = types.ModuleType("insightface")
    app = types.ModuleType("insightface.app")
    utils = types.ModuleType("insightface.utils")
    app.FaceAnalysis = FaceAnalysis
    utils.face_align = types.SimpleNamespace(norm_crop=lambda *a, **k: None)
    pkg.app, pkg.utils = app, utils
    for name, mod in (("insightface", pkg), ("insightface.app", app), ("insightface.utils", utils)):
        monkeypatch.setitem(sys.modules, name, mod)
    return calls


@pytest.fixture
def stub_libs(monkeypatch):
    for name in ("cv2", "redis"):
        if importlib.util.find_spec(name) is None:
            mod = types.ModuleType(name)
            if name == "redis":                          # used in annotations and as a client class
                mod.Redis = type("Redis", (), {"from_url": staticmethod(lambda *a, **k: None)})
            monkeypatch.setitem(sys.modules, name, mod)


# -- the helper ---------------------------------------------------------------------------------------------------
def test_root_is_the_repo_folder_insightface_expects():
    assert ip.ROOT == REPO / "perception" / "models" / "insightface"
    assert ip.DEFAULT_PACK == "auraface"
    assert ip.pack_dir() == REPO / DEFAULT_PACK and ip.pack_rel() == DEFAULT_PACK
    assert ip.PACKS["auraface"] == DEFAULT_FILES
    assert ip.pack_dir("buffalo_l") == REPO / PACK and ip.pack_rel("buffalo_l") == PACK
    assert ip.PACKS["buffalo_l"] == FILES


def test_complete_pack_gives_the_root(pack):
    assert ip.root() == str(pack.root) and ip.missing() == []
    assert ip.root("buffalo_l") == str(pack.root) and ip.missing("buffalo_l") == []


def test_missing_files_refuse_with_the_dvc_pull(pack):
    (pack.dir / "w600k_r50.onnx").unlink()
    (pack.dir / "det_10g.onnx").unlink()
    with pytest.raises(ip.InsightFacePackMissing) as exc:
        ip.root("buffalo_l")
    msg = str(exc.value)
    assert f"dvc pull {PACK}/det_10g.onnx.dvc {PACK}/w600k_r50.onnx.dvc" in msg
    assert "1k3d68" not in msg.split("dvc pull")[1]                      # only what is missing
    assert "never downloaded" in msg
    with pytest.raises(SystemExit, match=r"^error: .*dvc pull"):
        ip.cli_root("buffalo_l")


def test_unknown_pack_is_refused(pack):
    with pytest.raises(ip.InsightFacePackMissing, match="not versioned in this repo"):
        ip.root("antelopev2")


# -- the sidecar modules ------------------------------------------------------------------------------------------
def _module(name, stub_libs):
    return importlib.import_module(f"perception.modules.{name}")


@pytest.mark.parametrize("name", ["face_recognition", "customer_faces"])
def test_modules_declare_the_repo_pack(name, stub_libs):
    mod = _module(name, stub_libs)
    refs = mod.MODULE.model_files(mod.MODULE.Settings())
    assert [(r.path, r.kind) for r in refs] == [(DEFAULT_PACK, "pack")]


@pytest.mark.parametrize("name", ["face_recognition", "customer_faces"])
def test_modules_refuse_startup_without_the_pack(name, pack, stub_libs, fake_insightface, tmp_path):
    from perception.engine.modules import ModuleStartupError
    (pack.aura_dir / "glintr100.onnx").unlink()
    mod = _module(name, stub_libs)
    settings = mod.MODULE.Settings(**({"crops_dir": str(tmp_path / "crops")} if name == "customer_faces" else {}))
    m = mod.MODULE(settings, types.SimpleNamespace(redis=None))
    with pytest.raises(ModuleStartupError, match=f"modules.{name} could not start: .*dvc pull {DEFAULT_PACK}/glintr100.onnx.dvc"):
        m.on_built(None)
    assert fake_insightface == []                                         # refused before insightface was touched


def test_customer_faces_loads_from_the_repo_root(pack, stub_libs, fake_insightface, tmp_path, monkeypatch):
    mod = _module("customer_faces", stub_libs)
    monkeypatch.setattr(mod, "CustomerFaceCollector", lambda *a, **k: object())
    m = mod.MODULE(mod.MODULE.Settings(crops_dir=str(tmp_path / "crops")), types.SimpleNamespace(redis=None))
    m.on_built(None)
    assert fake_insightface == [{"name": "auraface", "root": str(pack.root),
                                 "allowed_modules": ["detection", "recognition"]}]


def test_recognizer_loads_from_the_repo_root(pack, stub_libs, fake_insightface, monkeypatch):
    monkeypatch.delitem(sys.modules, "perception.modules.face_recognition.recognizer", raising=False)
    rec = importlib.import_module("perception.modules.face_recognition.recognizer")
    src = Path(rec.__file__).read_text()
    assert "root=insightface_pack.root(model_name)" in src
    rec.FaceAnalysis(name="auraface", root=rec.insightface_pack.root("auraface"))     # the call as written
    assert fake_insightface[-1]["root"] == str(pack.root)


# -- the app ------------------------------------------------------------------------------------------------------
def test_app_answers_503_without_the_pack_and_uses_the_repo_root(pack, stub_libs, fake_insightface, monkeypatch):
    fastapi = pytest.importorskip("fastapi")
    np = pytest.importorskip("numpy")
    monkeypatch.delitem(sys.modules, "app.face_recognition", raising=False)
    fr = importlib.import_module("app.face_recognition")
    (pack.aura_dir / "glintr100.onnx").unlink()
    with pytest.raises(fastapi.HTTPException) as exc:
        fr._embed_face(np.zeros((8, 8, 3), np.uint8))
    assert exc.value.status_code == 503 and f"dvc pull {DEFAULT_PACK}/glintr100.onnx.dvc" in exc.value.detail
    assert fake_insightface == []
    (pack.aura_dir / "glintr100.onnx").write_bytes(b"onnx")
    with pytest.raises(Exception):                       # the fake has no .get(): only the constructor call matters
        fr._embed_face(np.zeros((8, 8, 3), np.uint8))
    assert fake_insightface[-1]["root"] == str(pack.root)
    assert fake_insightface[-1]["name"] == "auraface"


# -- _FaceDetLiteDetectionModel and _install_auraface_shim --------------------------------------------------------
def test_face_det_lite_detection_model_lazy_session_and_detect(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")

    det = ip._FaceDetLiteDetectionModel(onnx_path=tmp_path / "missing_face_det_lite.onnx")
    assert det.taskname == "detection"
    assert det.session is None
    det.prepare(ctx_id=-1, det_thresh=0.5, input_size=(640, 640))
    assert det.session is None
    assert det.ctx_id == -1
    assert det.det_thresh == pytest.approx(0.35)
    det.prepare(ctx_id=0, det_thresh=0.20, input_size=(640, 640))
    assert det.det_thresh == pytest.approx(0.20)
    det.prepare(ctx_id=0, det_thresh=0.65, input_size=(640, 640))
    assert det.det_thresh == pytest.approx(0.65)

    # Empty image exits before touching session.
    b_empty, k_empty = det.detect(np.zeros((0, 0, 3), dtype=np.uint8))
    assert b_empty.shape == (0, 5) and k_empty.shape == (0, 5, 2)
    assert det.session is None

    # Missing ONNX raises InsightFacePackMissing with dvc pull command on first real detect().
    with pytest.raises(ip.InsightFacePackMissing, match="dvc pull perception/models/DeepStream-Yolo-Face/face_det_lite.onnx.dvc"):
        det.detect(np.zeros((120, 160, 3), dtype=np.uint8))

    # Synthetic session verifies lazy creation, session reuse, unmapping, NMS, and max_num filtering.
    fake_onnx = tmp_path / "face_det_lite.onnx"
    fake_onnx.write_bytes(b"onnx")
    det = ip._FaceDetLiteDetectionModel(onnx_path=fake_onnx)
    det.prepare(ctx_id=-1, det_thresh=0.5)

    created = []

    class FakeSession:
        def __init__(self):
            self.runs = 0
            created.append(self)

        def get_inputs(self):
            return [types.SimpleNamespace(name="images")]

        def run(self, _out, feed):
            self.runs += 1
            blob = feed["images"]
            assert blob.shape == (1, 3, 480, 640) and blob.dtype == np.float32
            out = np.zeros((1, 4800, 15), dtype=np.float32)
            # Face 0: centered, larger box
            out[0, 0] = [240, 160, 400, 340, 0.92, 280, 210, 360, 210, 320, 255, 285, 300, 355, 300]
            # Face 1: near duplicate of Face 0 (suppressed by NMS)
            out[0, 1] = [242, 162, 398, 338, 0.70, 280, 210, 360, 210, 320, 255, 285, 300, 355, 300]
            # Face 2: smaller box near corner
            out[0, 2] = [20, 20, 90, 100, 0.85, 35, 45, 75, 45, 55, 65, 40, 85, 70, 85]
            return [out]

    det._ensure_session = lambda: det.session if det.session is not None else setattr(det, "session", FakeSession()) or det.session

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    bboxes, kpss = det.detect(img, max_num=0)
    assert len(created) == 1 and det.session is created[0]
    assert bboxes.shape == (2, 5) and bboxes.dtype == np.float32
    assert kpss.shape == (2, 5, 2) and kpss.dtype == np.float32
    assert bboxes[0, 4] == pytest.approx(0.92)

    # Second detect() reuses the existing session without creating a new one, and respects max_num=1.
    bboxes1, kpss1 = det.detect(img, max_num=1)
    assert len(created) == 1 and created[0].runs == 2
    assert bboxes1.shape == (1, 5) and kpss1.shape == (1, 5, 2)
    assert np.allclose(bboxes1[0, :4], [240, 160, 400, 340], atol=1e-4)

    # Grayscale (2D and 1-ch) and 4-channel BGRA inputs are normalized cleanly.
    for alt in (np.zeros((480, 640), dtype=np.uint8),
                np.zeros((480, 640, 1), dtype=np.uint8),
                np.zeros((480, 640, 4), dtype=np.uint8)):
        b_alt, k_alt = det.detect(alt, max_num=1)
        assert b_alt.shape == (1, 5) and k_alt.shape == (1, 5, 2)


def test_install_auraface_shim_idempotent_and_populates_models(pack, monkeypatch):
    rec_model = types.SimpleNamespace(taskname="recognition", prepare=lambda **kw: None)

    class RealFaceAnalysis:
        __module__ = "insightface.app.face_analysis"

        def __init__(self, name="buffalo_l", root="~/.insightface", allowed_modules=None, **kw):
            raise AssertionError("original __init__ would fail on assert 'detection' in self.models for auraface")

    pkg = types.ModuleType("insightface")
    app_mod = types.ModuleType("insightface.app")
    zoo_mod = types.ModuleType("insightface.model_zoo")
    app_mod.FaceAnalysis = RealFaceAnalysis
    zoo_mod.model_zoo = types.SimpleNamespace(get_model=lambda p, **kw: rec_model if p.endswith("glintr100.onnx") else None)
    pkg.app, pkg.model_zoo = app_mod, zoo_mod
    for name, mod in (("insightface", pkg), ("insightface.app", app_mod), ("insightface.model_zoo", zoo_mod)):
        monkeypatch.setitem(sys.modules, name, mod)

    # Calling root("auraface") 100 times wraps RealFaceAnalysis.__init__ exactly once.
    first_root = ip.root("auraface")
    first_init = RealFaceAnalysis.__init__
    for _ in range(99):
        assert ip.root("auraface") == first_root
    assert RealFaceAnalysis.__init__ is first_init
    assert getattr(RealFaceAnalysis, "_auraface_shim_installed", False) is True

    fa = RealFaceAnalysis(name="auraface", root=first_root, allowed_modules=["detection", "recognition"])
    assert fa.models["recognition"] is rec_model
    assert isinstance(fa.models["detection"], ip._FaceDetLiteDetectionModel)
    assert fa.det_model is fa.models["detection"]
    assert fa.det_model.session is None
    assert fa.addons == {} and fa.liveness_mode == "normal" and fa.model_package is None


# -- every call site, the images, the catalog ---------------------------------------------------------------------
def _face_analysis_calls() -> list[tuple[str, int, ast.Call]]:
    out = []
    files = [p for p in REPO.rglob("*.py")
             if not any(part in {"helper-repos", "tests", ".venv", "node_modules", ".data-remote", ".git"}
                        for part in p.relative_to(REPO).parts)]
    for p in files:
        try:
            src = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if "FaceAnalysis(" not in src:
            continue
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", None)) == "FaceAnalysis":
                out.append((str(p.relative_to(REPO)), node.lineno, node))
    return out


def test_every_face_analysis_call_passes_the_repo_root():
    calls = _face_analysis_calls()
    assert len(calls) >= 9, calls                        # modules, recognizer, app, attendance CLIs, register scripts
    bad = []
    for path, line, call in calls:
        root = next((k.value for k in call.keywords if k.arg == "root"), None)
        text = ast.unparse(root) if root is not None else ""
        if not (text.startswith("insightface_pack.") or text == "pack_root"):
            bad.append(f"{path}:{line} root={text or '(default ~/.insightface: downloads)'}")
    assert bad == []


@pytest.mark.parametrize("dockerfile", ["Dockerfile", "perception/pipeline/Dockerfile"])
def test_images_no_longer_download_the_pack(dockerfile):
    lines = [l for l in (REPO / dockerfile).read_text().splitlines() if not l.lstrip().startswith("#")]
    body = "\n".join(lines)
    assert "FaceAnalysis" not in body and ".insightface" not in body
    if dockerfile == "Dockerfile":
        assert "insightface" in body                     # the package install stays


def test_catalog_entry_points_at_the_repo_pack_and_pins_all_five():
    yaml = pytest.importorskip("yaml")
    cat = yaml.safe_load((REPO / "mtmc/tracking/imported_models.yaml").read_text())
    (e,) = [x for x in cat["entries"] if x["id"] == "face-recognizer/buffalo_l"]
    assert e["files"][0] == f"{PACK}/w600k_r50.onnx" and sorted(Path(f).name for f in e["files"]) == sorted(FILES)
    assert set(e["expected_sha256"]) == set(e["files"])
    assert e["provenance"]["origin"].endswith("/releases/download/v0.7/buffalo_l.zip")
    for f in e["files"]:
        assert (REPO / (f + ".dvc")).is_file(), f"no DVC pointer for {f}"
        local = REPO / f
        if local.is_file():                              # pulled: the bytes are the pinned ones
            assert hashlib.sha256(local.read_bytes()).hexdigest() == e["expected_sha256"][f]
