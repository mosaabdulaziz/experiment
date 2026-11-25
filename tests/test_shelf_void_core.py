"""modules.shelf_void core: the heat map and the alerter, with a fake model.

Ported unchanged from the standalone shelf-monitoring project (tests/test_shelf_void.py, commit 02538da);
only the import changed."""
import numpy as np

from perception.modules.shelf_void.core import AlertConfig, DetectorConfig, EmptyShelfAlerter, ShelfVoidDetector

H, W, FPS = 240, 320, 10
FRAME = np.zeros((H, W, 3), np.uint8)
VOID = (100, 100, 180, 140)


def make_det(script):
    """script(i) -> list of (x1,y1,x2,y2,conf) for frame i."""
    state = {"i": 0}

    def predict(_frame):
        out = script(state["i"]); state["i"] += 1
        return out
    return ShelfVoidDetector(DetectorConfig(min_area=100), predict_fn=predict)


def run(script, n, alert_cfg=None):
    det = make_det(script)
    al = EmptyShelfAlerter(alert_cfg or AlertConfig(confirm_s=1.0, clear_s=1.0))
    events = []
    for i in range(n):
        events += [(i, e) for e in al.update(det.update(FRAME), i / FPS, FRAME)]
    return events


def test_flicker_noise_never_alerts():
    # box appears every 4th frame only -> heat never reaches threshold
    evs = run(lambda i: [(*VOID, 0.03)] if i % 4 == 0 else [], 100)
    assert evs == []


def test_persistent_void_raises_with_snapshot():
    evs = run(lambda i: [(*VOID, 0.05)], 60)
    raised = [e for _, e in evs if e.event == "RAISED"]
    assert len(raised) == 1
    x1, y1, x2, y2 = raised[0].bbox
    assert abs(x1 - 100) <= 2 and abs(y2 - 140) <= 2
    assert raised[0].snapshot is not None and raised[0].snapshot.size > 0


def test_restock_clears_and_reempty_raises_new_alert():
    # empty 0-49, restocked 50-109, empty again 110-179
    script = lambda i: [(*VOID, 0.05)] if i < 50 or i >= 110 else []
    evs = [e for _, e in run(script, 180)]
    kinds = [e.event for e in evs]
    assert kinds == ["RAISED", "CLEARED", "RAISED"], kinds
    assert evs[0].alert_id == evs[1].alert_id != evs[2].alert_id


def test_fragments_joined_into_one_region():
    frags = [(100, 100, 130, 140, 0.05), (140, 100, 180, 140, 0.05)]  # 10px gap
    evs = run(lambda i: frags, 60)
    assert len([e for _, e in evs if e.event == "RAISED"]) == 1


def test_roi_ignores_voids_outside():
    det = ShelfVoidDetector(
        DetectorConfig(min_area=100, roi=[(0, 0), (90, 0), (90, H), (0, H)]),
        predict_fn=lambda f: [(*VOID, 0.05)])
    for _ in range(60):
        regions = det.update(FRAME)
    assert regions == []
