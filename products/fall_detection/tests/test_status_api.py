"""The HTTP surface: /status tells the operator the truth about the classifier and the live paths.

ACs: AC-F25.1, AC-N10.1. Design: FALL_DETECTION_DESIGN.md §8; plan §The boot contract.
The invariant "no code path runs with fewer paths than reported" is pinned at its source of truth:
`boot` derives both the worker's live paths and the reported status from the same state, so the
report cannot be hand-fed a rosier picture than the service runs with.
"""
from fastapi.testclient import TestClient
import pytest

import boot
import storage
from server import create_app
from storage import ReviewQueue


@pytest.fixture
def queue(tmp_path):
    q = ReviewQueue(storage.connect(tmp_path / "reviews.db"))
    yield q
    q.close()


def _app(boot_state, queue):
    return create_app(status=boot.status(boot_state), storage=queue)


READY = {"device": "cuda:0", "weights_verified": True, "classifier_ready": True}


def test_status_reports_the_device_the_weights_and_the_classifier(queue):
    body = TestClient(_app(READY, queue)).get("/status").json()
    assert body["device"] == "cuda:0"
    assert body["weights_verified"] is True
    assert body["classifier_ready"] is True
    assert set(body["paths_live"]) == {"action", "kinematic", "cold"}


def test_the_api_names_the_cameras_the_service_watches(queue):
    """The tab's camera list comes from the product's own config, not from a second source."""
    app = create_app(status=boot.status(READY), storage=queue, cameras=["cam-1", "cam-2"])
    assert TestClient(app).get("/cameras").json() == ["cam-1", "cam-2"]


def test_the_live_paths_are_derived_from_the_boot_state_not_declared_beside_it():
    assert set(boot.paths_live(READY)) == {"action", "kinematic", "cold"}
    degraded = dict(READY, classifier_ready=False)
    assert "action" not in boot.paths_live(degraded)          # no reading => no action path
    assert set(boot.paths_live(degraded)) == {"kinematic", "cold"}


def test_status_never_reports_more_paths_than_are_live(queue):
    degraded = dict(READY, classifier_ready=False)
    body = TestClient(_app(degraded, queue)).get("/status").json()
    assert body["classifier_ready"] is False
    assert "action" not in body["paths_live"]
    assert set(body["paths_live"]) == set(boot.paths_live(degraded))   # reports exactly what runs
