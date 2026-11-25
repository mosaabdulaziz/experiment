import numpy as np

from perception.modules.person_masks.outline import outline


def test_full_mask_gives_the_box_corners():
    scores = np.ones((20, 10), np.float32)
    pts = outline(scores, 0.5, (100.0, 50.0, 100.0, 200.0), (1000.0, 500.0))
    xy = np.array(pts).reshape(-1, 2)
    assert 3 <= len(xy) <= 40
    assert abs(xy[:, 0].min() - 0.1) < 0.01 and abs(xy[:, 0].max() - 0.2) < 0.01
    assert abs(xy[:, 1].min() - 0.1) < 0.02 and abs(xy[:, 1].max() - 0.5) < 0.02


def test_empty_mask_is_none():
    assert outline(np.zeros((20, 10), np.float32), 0.5, (0, 0, 10, 10), (100, 100)) is None


def test_threshold_is_applied_and_largest_blob_kept():
    s = np.zeros((40, 40), np.float32)
    s[2:6, 2:6] = 0.9            # small blob
    s[10:38, 10:38] = 0.9        # big blob
    s[0:40, 39] = 0.4            # below threshold
    xy = np.array(outline(s, 0.5, (0, 0, 40, 40), (40, 40))).reshape(-1, 2)
    assert xy[:, 0].min() > 0.2 and xy[:, 1].min() > 0.2


def test_point_budget_is_respected():
    yy, xx = np.mgrid[0:200, 0:200]
    s = (((xx - 100) ** 2 + (yy - 100) ** 2) < 90 ** 2).astype(np.float32)
    pts = outline(s, 0.5, (0, 0, 200, 200), (200, 200), max_points=12)
    assert 3 <= len(pts) // 2 <= 12 and len(pts) % 2 == 0


def test_values_are_clamped_and_rounded():
    pts = outline(np.ones((4, 4), np.float32), 0.5, (-10.0, -10.0, 30.0, 30.0), (20.0, 20.0))
    assert all(0.0 <= v <= 1.0 for v in pts)
    assert all(round(v, 4) == v for v in pts)
