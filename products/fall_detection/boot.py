"""The boot contract: the classifier is mandatory, the GPU check is a real op, no CPU fallback.

The gate (design §8):

  1. read the product's yaml -> the device and the weights dir
  2. verify both checkpoints against the product's model manifest (the registered versions; no manifest only
     warns unless MODEL_MANIFEST_REQUIRED), then against the vendor's own SHA-256 registry -> mismatch: refuse
  3. `torch.cuda.is_available()` AND a real CUDA op on the device -> no GPU: refuse
  4. build the classifier and hand it to the monitors
  5. only then: the Redis consumer, the per-camera workers, the HTTP API

The vendor's own code swallows all three failures (`except Exception` -> `classifier = None`).
Here each one exits non-zero with a message naming the cause.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch

import mmaction_compat  # noqa: F401  (installs the import shim before the copy is loaded)
import model_files
from fall_detection.fall_monitor import CFG
from fall_detection.tsstg_model import STREAM_FILE_MAP, WEIGHT_SHA256_REGISTRY, TSSTGClassifier

VARIANT = "mmaction2"
# the classifier's clips are the vendor's own (config.yaml's detector.streams): one source per quantity
DEFAULT_STREAMS = tuple(CFG["detector"]["streams"])


def _refuse(message: str) -> None:
    """Exit non-zero, naming the cause."""
    print(f"fall detection: refusing to start -- {message}", file=sys.stderr)
    raise SystemExit(1)


def _checkpoint_paths(classifier: dict) -> list[Path]:
    """The checkpoints the configured streams need, by name."""
    weights_dir = Path(classifier["weights_dir"])
    streams = tuple(classifier.get("streams") or DEFAULT_STREAMS)
    file_map = STREAM_FILE_MAP[VARIANT]
    paths = []
    for stream in streams:
        name = file_map.get(stream)
        if name is None:
            _refuse(f"the classifier stream {stream!r} has no {VARIANT} checkpoint")
        paths.append(weights_dir / name)
    return paths


def _verify_weights(classifier: dict) -> None:
    """Both checkpoints, against the vendor's own registry."""
    for path in _checkpoint_paths(classifier):
        if not path.is_file():
            _refuse(f"the checkpoint {path.name} is missing ({path}); the checkpoints are versioned with DVC: "
                    "run `just data-pull models-fall`")
        expected = WEIGHT_SHA256_REGISTRY.get(path.name)
        if expected is None:
            _refuse(f"the checkpoint {path.name} has no pinned sha256 in the vendor's registry")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            _refuse(f"the checkpoint {path.name} does not match its pinned sha256 "
                    f"(expected {expected}, got {digest})")


def _check_device(device: str) -> None:
    """A real CUDA op, not `cuda.is_available()`.

    A CUDA wheel without this card's kernels loads, reports a healthy CUDA and fails at the
    first op, so the check has to be an op (design §8). There is no CPU fallback.
    """
    if not torch.cuda.is_available():
        _refuse(f"the configured device {device} is not available -- there is no CPU fallback")
    try:
        probe = torch.ones((4, 4), device=device)
        torch.matmul(probe, probe).sum().item()
    except Exception as exc:
        _refuse(f"a real CUDA op on {device} failed: {exc}")


def _verify_manifest(classifier: dict, model_manifest: str | None) -> None:
    """Both checkpoints, against the registered versions the model manifest lists (perception/engine/model_manifest.py)."""
    verifier = model_files.load_verifier()
    try:
        model_files.verify_checkpoints(_checkpoint_paths(classifier), model_manifest)
    except verifier.ManifestError as exc:
        _refuse(str(exc))


def boot(classifier: dict, model_manifest: str | None = None):
    """The gate. Returns the built classifier and the state it proved -- or exits non-zero.

    The returned state is not a declaration: it is the record of the steps that just succeeded,
    which is what the service's `/status` reports.
    """
    device = str(classifier["device"])
    _verify_manifest(classifier, model_manifest)
    _verify_weights(classifier)
    _check_device(device)
    streams = tuple(classifier.get("streams") or DEFAULT_STREAMS)
    try:
        built = TSSTGClassifier(weight_dir=classifier["weights_dir"], variant=VARIANT,
                                streams=streams, device=device)
    except Exception as exc:
        # a broken classifier is a refusal, not a traceback: the copy's own import fallback
        # swallows the cause, so the gate names it
        _refuse(f"the classifier could not be built: {exc}")
    return built, {"device": device, "weights_verified": True, "classifier_ready": True}


def paths_live(state: dict) -> set[str]:
    """The scoring paths this state runs with.

    The action path is the classifier's reading, so it lives only while the classifier does.
    The two physical paths are pure kinematics, so they live either way -- the product's own
    lying and immobility candidates score against them without a reading.
    """
    if state.get("classifier_ready"):
        return {"action", "kinematic", "cold"}
    return {"kinematic", "cold"}


def status(state: dict) -> dict:
    """What the service reports about itself: the device, the weights, the classifier, the paths."""
    return {"device": state.get("device"),
            "weights_verified": bool(state.get("weights_verified")),
            "classifier_ready": bool(state.get("classifier_ready")),
            "paths_live": sorted(paths_live(state))}
