"""Unit tests for the MTMC pipeline (tracker + association), GPU-free."""
import numpy as np
import pytest

from mtmc.associate import TrackletWithEmbedding, associate
from mtmc.cameramodel import CameraLinkModel, TrackletObs, learn_pairwise_homography
from mtmc.config import PipelineConfig
from mtmc.tracker import Tracklet, iou_matrix, track_detections


def _cfg(**kw) -> PipelineConfig:
    base = dict(scene="S01", cameras=["c001", "c002"])
    base.update(kw)
    return PipelineConfig(**base)


def test_iou_matrix_basic():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[5, 0, 10, 10], [100, 100, 5, 5]])
    iou = iou_matrix(a, b)
    assert iou.shape == (1, 2)
    assert abs(iou[0, 0] - 0.5) < 1e-9  # 50 overlap / 100 union
    assert iou[0, 1] == 0.0


def test_tracker_follows_moving_box():
    dets = {}
    for f in range(1, 31):
        x = (f - 1) * 2
        dets[f] = np.array([[x, 0, 10, 10]])
    tls = track_detections("c001", dets, _cfg(), max_frame=30)
    assert len(tls) == 1
    assert tls[0].start_frame == 1 and tls[0].end_frame == 30


def test_tracker_separates_distant_boxes():
    dets = {}
    for f in range(1, 21):
        dets[f] = np.array([[0, 0, 10, 10], [200, 200, 10, 10]])
    tls = track_detections("c001", dets, _cfg(), max_frame=20)
    assert len(tls) == 2


def test_tracker_bridges_small_gaps():
    dets = {f: np.array([[0, 0, 10, 10]]) for f in range(1, 21) if f != 12}
    tls = track_detections("c001", dets, _cfg(), max_frame=20)
    assert len(tls) == 1  # gap of 1 frame bridged by max_unmatched_frames=10


def test_tracker_drops_short_tracklets():
    dets = {1: np.array([[0, 0, 10, 10]]), 2: np.array([[1, 0, 10, 10]])}
    tls = track_detections("c001", dets, _cfg(), max_frame=2)
    assert len(tls) == 0  # below min_tracklet_frames=8


def test_associate_merges_similar_tracklets_across_cams():
    cfg = _cfg()
    emb = np.array([1.0, 0.0])
    tl = [
        TrackletWithEmbedding("c001", 0, 0.0, 3.0, emb),
        TrackletWithEmbedding("c002", 0, 0.1, 3.2, emb),
    ]
    gid = associate(tl, cfg)
    assert len(gid) == 1 and len(gid[0]) == 2


def test_associate_rejects_time_incompatible():
    cfg = _cfg(time_overlap_tolerance_s=0.1)
    emb = np.array([1.0, 0.0])
    tl = [
        TrackletWithEmbedding("c001", 0, 0.0, 1.0, emb),
        TrackletWithEmbedding("c002", 0, 100.0, 101.0, emb),
    ]
    gid = associate(tl, cfg)
    assert len(gid) == 2


def test_associate_rejects_low_similarity():
    cfg = _cfg(reid_sim_threshold=0.9)
    tl = [
        TrackletWithEmbedding("c001", 0, 0.0, 1.0, np.array([1.0, 0.0])),
        TrackletWithEmbedding("c002", 0, 0.0, 1.0, np.array([0.0, 1.0])),
    ]
    gid = associate(tl, cfg)
    assert len(gid) == 2


def test_associate_never_merges_same_camera():
    cfg = _cfg()
    emb = np.array([1.0, 0.0])
    tl = [
        TrackletWithEmbedding("c001", 0, 0.0, 1.0, emb),
        TrackletWithEmbedding("c001", 1, 0.1, 1.1, emb),
    ]
    gid = associate(tl, cfg)
    assert len(gid) == 2


def test_tracklet_to_xyxy():
    t = Tracklet("c001", 0)
    t.add(1, np.array([5.0, 6.0, 7.0, 8.0]))
    assert np.allclose(t.to_xyxy(), [[5, 6, 12, 14]])


# ---- camera-link model gates -------------------------------------------

def _obs(cam, tid, frames, boxes, start_s, end_s, offset_s=0.0):
    times = offset_s + (np.asarray(frames, dtype=float) - 1) / 10.0
    return TrackletObs(cam, tid, frames, np.asarray(boxes, dtype=float), start_s, end_s, times_s=times)


def test_spatial_gate_passes_for_consistent_position():
    # identity-ish homography: cam_a point projects to same place in cam_b
    H = np.eye(3)
    model = CameraLinkModel(homographies={("c001", "c002"): H})
    a = _obs("c001", 0, [1, 2, 3], [[100, 200, 40, 40]] * 3, 0.0, 2.0)
    b = _obs("c002", 0, [1, 2, 3], [[100, 200, 40, 40]] * 3, 0.0, 2.0)
    assert model.spatial_gate(a, b, max_dist_px=50.0)


def test_spatial_gate_uses_global_times_not_frames():
    # c001 has no offset; c002 starts 20s later. Frame 1 of c002 is 20s in
    # wall time, so the tracklets do NOT overlap and the gate has no points.
    H = np.eye(3)
    model = CameraLinkModel(homographies={("c001", "c002"): H})
    a = _obs("c001", 0, [1, 2, 3], [[100, 200, 40, 40]] * 3, 0.0, 2.0, offset_s=0.0)
    b = _obs("c002", 0, [201, 202, 203], [[100, 200, 40, 40]] * 3, 20.0, 22.0, offset_s=20.0)
    # intervals don't overlap -> spatial gate has nothing to compare -> passes
    assert model.spatial_gate(a, b, max_dist_px=50.0)


def test_spatial_gate_rejects_far_apart():
    H = np.eye(3)
    model = CameraLinkModel(homographies={("c001", "c002"): H})
    a = _obs("c001", 0, [1, 2, 3], [[100, 200, 40, 40]] * 3, 0.0, 2.0)
    b = _obs("c002", 0, [1, 2, 3], [[500, 500, 40, 40]] * 3, 0.0, 2.0)
    assert not model.spatial_gate(a, b, max_dist_px=50.0)


def test_spatial_gate_passes_without_homography():
    model = CameraLinkModel()
    a = _obs("c001", 0, [1], [[100, 200, 40, 40]], 0.0, 1.0)
    b = _obs("c002", 0, [1], [[500, 500, 40, 40]], 0.0, 1.0)
    assert model.spatial_gate(a, b, max_dist_px=10.0)  # no info -> don't reject


def test_temporal_gate_overlap_and_gap():
    model = CameraLinkModel()
    a = _obs("c001", 0, [1], [[0, 0, 10, 10]], 0.0, 1.0)
    b = _obs("c002", 0, [1], [[0, 0, 10, 10]], 0.5, 1.5)
    assert model.temporal_gate(a, b, tol_s=1.5)  # overlapping intervals
    c = _obs("c003", 0, [1], [[0, 0, 10, 10]], 100.0, 101.0)
    assert not model.temporal_gate(a, c, tol_s=1.5)  # disjoint, no window


def test_temporal_gate_window_allows_sequential_transit():
    # disjoint intervals but inside the learned window -> allowed
    model = CameraLinkModel(windows={("c001", "c002"): (10.0, 20.0)})
    a = _obs("c001", 0, [1], [[0, 0, 10, 10]], 0.0, 1.0)      # mid 0.5
    b = _obs("c002", 0, [1], [[0, 0, 10, 10]], 12.0, 13.0)    # mid 12.5, delta 12
    assert model.temporal_gate(a, b, tol_s=1.5)
    c = _obs("c002", 1, [1], [[0, 0, 10, 10]], 100.0, 101.0)  # mid 100.5, delta 100
    assert not model.temporal_gate(a, c, tol_s=1.5)


def test_learn_pairwise_homography_identity():
    # two GT dicts describing the same objects at the same positions,
    # spread across the frame (non-collinear so the fit is well-posed)
    n = 12
    rng = np.random.default_rng(0)
    pts = [(150, 150), (400, 300), (700, 200), (300, 500), (600, 600), (800, 700)] * 2
    gt_a, gt_b = {}, {}
    for f in range(1, n + 1):
        x, y = pts[f - 1]
        gt_a[f] = np.array([[1, x, y, 40, 40, 1, -1, -1, -1]])
        gt_b[f] = np.array([[1, x, y, 40, 40, 1, -1, -1, -1]])
    h = learn_pairwise_homography(gt_a, gt_b, 0.0, 0.0, 10.0)
    assert h is not None
    # projecting a point should reproduce it (identity up to scale)
    p = np.array([150.0, 250.0, 1.0]) @ h.T
    p = p[:2] / p[2]
    assert np.allclose(p, [150.0, 250.0], atol=1e-6)


def test_associate_respects_spatial_gate():
    cfg = _cfg(spatial_gate_px=50.0)
    H = np.eye(3)
    model = CameraLinkModel(homographies={("c001", "c002"): H})
    emb = np.array([1.0, 0.0])
    near = TrackletWithEmbedding(
        "c001", 0, 0.0, 2.0, emb, [1, 2], np.array([[100, 200, 40, 40]] * 2)
    )
    same_vehicle = TrackletWithEmbedding(
        "c002", 0, 0.0, 2.0, emb, [1, 2], np.array([[100, 200, 40, 40]] * 2)
    )
    far_vehicle = TrackletWithEmbedding(
        "c002", 1, 0.0, 2.0, emb, [1, 2], np.array([[900, 900, 40, 40]] * 2)
    )
    gid = associate([near, same_vehicle, far_vehicle], cfg, camera_model=model)
    # near and same_vehicle merge; far stays separate even though embedding matches
    assert len(gid) == 2
    merged = [m for m in gid.values() if len(m) == 2]
    assert len(merged) == 1
    members = {c for c, _ in merged[0]}
    assert members == {"c001", "c002"} and ("c002", 1) not in merged[0]


def test_associate_spatial_proof_merges_without_appearance():
    # time-overlapping tracklets projecting to the same ground point merge
    # even when appearance says nothing (orthogonal embeddings, sim ~ 0)
    cfg = _cfg(spatial_merge_px=50.0, reid_sim_threshold=0.95)
    H = np.eye(3)
    model = CameraLinkModel(homographies={("c001", "c002"): H})
    a = TrackletWithEmbedding(
        "c001", 0, 0.0, 2.0, np.array([1.0, 0.0]), [1, 2],
        np.array([0.0, 0.0]), np.array([[100, 200, 40, 40]] * 2),
    )
    b = TrackletWithEmbedding(
        "c002", 0, 0.0, 2.0, np.array([0.0, 1.0]), [1, 2],
        np.array([0.0, 0.0]), np.array([[100, 200, 40, 40]] * 2),
    )
    gid = associate([a, b], cfg, camera_model=model)
    assert len(gid) == 1  # spatial proof, appearance ignored


def test_associate_appearance_requires_sim_threshold():
    # time-overlapping pairs without a homography (no spatial proof) merge
    # only when the appearance similarity clears the threshold
    cfg = _cfg(spatial_merge_px=50.0, reid_sim_threshold=0.9)
    model = CameraLinkModel()  # no homography -> no spatial proof
    a = TrackletWithEmbedding(
        "c001", 0, 0.0, 1.0, np.array([1.0, 0.0]), [1],
        np.array([0.0]), np.array([[100, 200, 40, 40]]),
    )
    b = TrackletWithEmbedding(
        "c002", 0, 0.0, 1.0, np.array([1.0, 0.0]), [1],
        np.array([0.0]), np.array([[100, 200, 40, 40]]),
    )
    # time-overlapping, same appearance, but sim 1.0 >= 0.9 -> merges
    assert len(associate([a, b], cfg, camera_model=model)) == 1
    # orthogonal embeddings (sim 0.0) fail the high threshold -> stay separate
    a2 = TrackletWithEmbedding(
        "c001", 1, 0.0, 1.0, np.array([1.0, 0.0]), [1],
        np.array([0.0]), np.array([[100, 200, 40, 40]]),
    )
    b2 = TrackletWithEmbedding(
        "c002", 1, 0.0, 1.0, np.array([0.0, 1.0]), [1],
        np.array([0.0]), np.array([[100, 200, 40, 40]]),
    )
    assert len(associate([a2, b2], cfg, camera_model=model)) == 2


# ── CamGraph tests ────────────────────────────────────────────────────────

from mtmc.camgraph import CamGraph, EdgeConfig, load_cam_graph


def test_camgraph_uniform_defaults():
    """Without a YAML file, every pair gets the same EdgeConfig."""
    g = load_cam_graph(None, ["c001", "c002", "c003"])
    e = g.edge("c001", "c002")
    assert e.reid_sim == 0.55
    assert e.time_tolerance_s == 1.5
    assert e.use_camera_model is False
    # reverse order is identical
    assert g.edge("c002", "c001").reid_sim == e.reid_sim


def test_camgraph_per_edge_overrides(tmp_path):
    """Edges listed in the YAML override the defaults."""
    yaml = tmp_path / "graph.yaml"
    yaml.write_text("""
defaults:
  reid_sim: 0.55
  time_tolerance_s: 1.5
edges:
  - pair: [c001, c002]
    reid_sim: 0.80
    use_camera_model: true
    spatial_merge_px: 50.0
""")
    g = load_cam_graph(str(yaml), ["c001", "c002", "c003"])
    # listed edge gets overrides
    e12 = g.edge("c001", "c002")
    assert e12.reid_sim == 0.80
    assert e12.use_camera_model is True
    assert e12.spatial_merge_px == 50.0
    # unlisted edge falls back to defaults
    e13 = g.edge("c001", "c003")
    assert e13.reid_sim == 0.55
    assert e13.use_camera_model is False


def test_camgraph_cli_overrides_yaml(tmp_path):
    """CLI overrides beat the YAML defaults."""
    yaml = tmp_path / "graph.yaml"
    yaml.write_text("""
defaults:
  reid_sim: 0.55
  time_tolerance_s: 1.5
""")
    g = load_cam_graph(str(yaml), ["c001", "c002"], cli_defaults={"reid_sim": 0.90})
    e = g.edge("c001", "c002")
    assert e.reid_sim == 0.90  # CLI override wins


def test_camgraph_with_associate():
    """CamGraph is threaded through associate() correctly."""
    cfg = _cfg()
    # Build a graph where c001↔c002 has a VERY high sim threshold (0.99)
    defaults = EdgeConfig(reid_sim=0.99)
    graph = CamGraph(cameras=["c001", "c002"], defaults=defaults)
    a = TrackletWithEmbedding(
        "c001", 0, 0.0, 1.0, np.array([1.0, 0.0]), [1],
        np.array([0.0]), np.array([[100, 200, 40, 40]]),
    )
    b = TrackletWithEmbedding(
        "c002", 0, 0.0, 1.0, np.array([1.0, 0.0]), [1],
        np.array([0.0]), np.array([[100, 200, 40, 40]]),
    )
    # Same appearance (sim=1.0) but graph says 0.99 -> merges
    assert len(associate([a, b], cfg, cam_graph=graph)) == 1
    # Now set threshold above 1.0 -> cannot merge
    graph2 = CamGraph(
        cameras=["c001", "c002"],
        defaults=EdgeConfig(reid_sim=1.5),
    )
    assert len(associate([a, b], cfg, cam_graph=graph2)) == 2


def test_camgraph_summary():
    g = load_cam_graph(None, ["c001", "c002"])
    s = g.summary()
    assert "2 cameras" in s
