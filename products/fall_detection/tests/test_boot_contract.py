"""The boot contract: the classifier is mandatory, the GPU check is a real op, no CPU fallback.

ACs: AC-F23.1-3, AC-F24.1, AC-N9.1, AC-N11.1. Design: FALL_DETECTION_DESIGN.md §8; plan risk 4.
The refusal paths run anywhere by faking torch; the positive path is gated on the real weights
(and skips cleanly without them, like the vendor's own gated tests — ruling A10).
"""
import hashlib
import sys
import types
from importlib import metadata
from pathlib import Path

import pytest

import boot

VENDOR_WEIGHTS = Path(__file__).resolve().parents[1] / "weights"
REGISTRY = {
    "mmaction2_joint.pth": "484a394a50246032fef9694396096902909784e8618ccc845d2859dc2b952fea",
    "mmaction2_jm.pth": "f18eb408a42eebced02fc0e0dfeed2c6edb320cc1752abc6df855dff0845fc6d",
}


def _cfg(tmp_path, device="cuda:0"):
    return {"device": device, "weights_dir": str(tmp_path / "weights"), "streams": ["j", "jm"]}


def _fake_torch(monkeypatch, available=True, op_raises=None):
    def _op(*a, **k):
        if op_raises:
            raise op_raises
        return types.SimpleNamespace(sum=lambda: types.SimpleNamespace(item=lambda: 1.0))

    fake = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: available),
        ones=_op, matmul=_op, __version__="2.9.0", version=types.SimpleNamespace(cuda="12.8"),
    )
    monkeypatch.setattr(boot, "torch", fake)
    return fake


def _good_weights(tmp_path, monkeypatch):
    """Weights whose bytes this helper owns, checked against the sha it pins for them.

    The vendor's checkpoints are not in the repo (they are placed by hand), so a
    test that needs a *verifiable* pair of checkpoints writes its own and pins their digests in
    the product's registry. The refusal paths are then reached for the reason they name.
    """
    d = tmp_path / "weights"
    d.mkdir()
    shas = {}
    for name in REGISTRY:
        src = VENDOR_WEIGHTS / name
        blob = src.read_bytes() if src.is_file() else b"x" * 64
        (d / name).write_bytes(blob)
        shas[name] = hashlib.sha256(blob).hexdigest()
    monkeypatch.setattr(boot, "WEIGHT_SHA256_REGISTRY", shas)
    return d


# ── refusals (AC-F23.1-3, AC-F24.1, AC-N9.1) ───────────────────────────────────────────────────────────────

def test_a_missing_checkpoint_refuses_startup_naming_the_file(tmp_path, monkeypatch, capsys):
    _fake_torch(monkeypatch)
    (tmp_path / "weights").mkdir()
    with pytest.raises(SystemExit) as e:
        boot.boot(_cfg(tmp_path))
    assert e.value.code != 0
    assert "mmaction2_joint.pth" in capsys.readouterr().err


def test_a_corrupt_checkpoint_refuses_startup_naming_the_checksum(tmp_path, monkeypatch, capsys):
    _fake_torch(monkeypatch)
    d = _good_weights(tmp_path, monkeypatch)
    for name in d.iterdir():                                   # whatever the copy produced, break it
        name.write_bytes(b"corrupted")
    with pytest.raises(SystemExit) as e:
        boot.boot(_cfg(tmp_path))
    assert e.value.code != 0
    out = capsys.readouterr().err
    assert "sha" in out.lower() or "checksum" in out.lower()


def test_cuda_configured_with_no_gpu_refuses_startup_and_does_not_fall_back(tmp_path, monkeypatch, capsys):
    _fake_torch(monkeypatch, available=False)
    _good_weights(tmp_path, monkeypatch)
    with pytest.raises(SystemExit) as e:
        boot.boot(_cfg(tmp_path))
    assert e.value.code != 0
    assert "cuda:0" in capsys.readouterr().err                 # the message names the device; no CPU retry


def test_a_cuda_wheel_without_this_cards_kernels_refuses_startup(tmp_path, monkeypatch, capsys):
    # the RTX-5090 failure mode: is_available() lies, the first real op raises
    _fake_torch(monkeypatch, available=True, op_raises=RuntimeError("no kernel image for sm_120"))
    _good_weights(tmp_path, monkeypatch)
    with pytest.raises(SystemExit) as e:
        boot.boot(_cfg(tmp_path))
    assert e.value.code != 0
    assert "cuda" in capsys.readouterr().err.lower()


# ── the positive path ───────────────────────────────────────────────────────────────────────────────────────

def test_a_classifier_that_cannot_be_built_refuses_startup_naming_the_cause(tmp_path, monkeypatch, capsys):
    # the copy's own import fallback swallows the cause, so the gate has to name it
    _fake_torch(monkeypatch)
    _good_weights(tmp_path, monkeypatch)

    def _boom(**kwargs):
        raise RuntimeError("No module named 'mmaction.models.localizers.drn'")

    monkeypatch.setattr(boot, "TSSTGClassifier", _boom)
    with pytest.raises(SystemExit) as e:
        boot.boot(_cfg(tmp_path))
    assert e.value.code != 0
    assert "drn" in capsys.readouterr().err


def test_with_verified_weights_and_a_working_device_the_classifier_is_built(tmp_path, monkeypatch):
    _fake_torch(monkeypatch)
    _good_weights(tmp_path, monkeypatch)
    built = []
    monkeypatch.setattr(boot, "TSSTGClassifier", lambda **kw: built.append(kw) or "CLASSIFIER")
    classifier, state = boot.boot(_cfg(tmp_path))
    assert classifier == "CLASSIFIER"
    assert built[0]["device"] == "cuda:0"
    # the state is the record of the steps that succeeded, and `/status` reports exactly it
    assert state == {"device": "cuda:0", "weights_verified": True, "classifier_ready": True}
    assert boot.status(state) == {"device": "cuda:0", "weights_verified": True,
                                  "classifier_ready": True,
                                  "paths_live": ["action", "cold", "kinematic"]}


# ── the single-torch environment (AC-N11.1) ────────────────────────────────────────────────────────────────

def test_the_resolved_environment_carries_exactly_one_torch():
    torches = [d for d in metadata.distributions() if (d.metadata["Name"] or "").lower() == "torch"]
    if not torches:
        pytest.skip("this interpreter is not the product's synced environment (no torch at all)")
    assert len(torches) == 1, f"expected exactly one torch distribution, found {len(torches)}"


def test_the_installed_build_carries_the_kernels_of_the_card_it_runs_on():
    """The invariant behind the cu130 pin, checked against the card actually in the box.

    A wheel compiled for other architectures loads and reports a healthy CUDA, then fails at
    the first op -- exactly the boot gate's refusal (AC-F24.1). So the check is arch membership,
    not a version floor: cu130 is what covers the deployment's sm_120 card, and this box's own
    card must be covered by whatever build is installed here.
    """
    torch = pytest.importorskip("torch", reason="runs inside the product venv; torch absent here")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device on this machine")
    major, minor = torch.cuda.get_device_capability()
    built = torch.cuda.get_arch_list()
    assert f"sm_{major}{minor}" in built, (
        f"the installed torch ({torch.__version__}, cuda {torch.version.cuda}) carries {built}, "
        f"not sm_{major}{minor} -- the card in this box"
    )


def test_the_product_pins_the_cuda_build_that_covers_the_deployment_card():
    """The deployment card is an RTX 5090 (sm_120); the cu130 index is what covers it.

    cu130 is also the build the sidecar's own image installs (perception/pipeline/Dockerfile),
    so the product and the engine share one CUDA generation -- and one driver -- on the server.
    """
    tomllib = pytest.importorskip("tomllib", reason="tomllib is stdlib from 3.11")
    pyproject = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    indexes = {index["name"]: index["url"] for index in pyproject["tool"]["uv"]["index"]}
    assert indexes.get("pytorch-cu130") == "https://download.pytorch.org/whl/cu130"
    assert pyproject["tool"]["uv"]["sources"]["torch"] == [{"index": "pytorch-cu130"}]
    assert pyproject["tool"]["uv"]["sources"]["torchvision"] == [{"index": "pytorch-cu130"}]


def test_the_resolved_build_is_the_pinned_one_in_the_products_own_environment():
    """In the product's uv-synced environment, the resolved torch is the pinned cu130 build.

    The pin is what the deployment runs; this is the check that the environment really resolved
    it, so a machine that synced from another index is caught here rather than at the first op.
    """
    torch = pytest.importorskip("torch", reason="runs inside the product venv; torch absent here")
    product_env = Path(__file__).resolve().parents[1] / ".venv"
    if Path(sys.prefix).resolve() != product_env.resolve():
        pytest.skip("not the product's own synced environment")
    assert torch.version.cuda, "the resolved torch carries no CUDA build"
    major, minor = (int(part) for part in torch.version.cuda.split(".")[:2])
    assert (major, minor) == (13, 0), (
        f"the pinned index is cu130, the environment resolved {torch.version.cuda}"
    )


@pytest.mark.skipif(not (VENDOR_WEIGHTS / "mmaction2_joint.pth").is_file(),
                    reason="the pinned checkpoints are not on this machine")
def test_the_real_classifier_constructs_against_the_resolved_environment():
    # AC-N11.1's positive half: no fake, no monkeypatch — the copied model class really builds
    # against the one installed torch (+mmaction2) in the product's own environment.
    # CPU on purpose: construction is what this pins, and the card is the server's (the boot
    # gate's cuda path is covered above by the faked torch, and really exercised on the 5090).
    pytest.importorskip("torch", reason="runs inside the product venv (S2); torch absent here")
    pytest.importorskip("mmaction", reason="mmaction2 is a runtime dep of the product venv")
    from fall_detection.tsstg_model import TSSTGClassifier
    clf = TSSTGClassifier(weight_dir=VENDOR_WEIGHTS, variant="mmaction2",
                          streams=("j", "jm"), device="cpu")
    assert clf.device == "cpu"
