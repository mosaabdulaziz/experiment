"""Unit tests for MTMC Multi-Camera Vehicle Tracking pipeline.

Tests vehicle-only tracking, persistent global IDs across cameras, 20-second disappearance rules,
appearance similarity over historical crops, camera topology gating, and separation of local/global IDs.
"""
import time

import numpy as np
import pytest

from perception.modules.mtmc.core import MTMCConfig, MTMCManager
from perception.modules.mtmc.core._associate import GlobalVehicleAssociation, GlobalVehicleTrack


def _make_config(**kwargs) -> MTMCConfig:
    defaults = {
        "enabled": True,
        "reid_enabled": False,  # disable actual model loading in unit test; we mock embeddings
        "min_disappearance_s": 20.0,
        "max_inactive_ttl_s": 300.0,
        "reid_sim_threshold": 0.55,
        "topology": ["cam-1", "cam-2", "cam-3"],
        "tracked_classes": ("car", "truck", "bus", "motorcycle", "bicycle"),
    }
    defaults.update(kwargs)
    return MTMCConfig(**defaults)


def test_vehicle_only_filtering():
    """Ensure person detections are ignored and only vehicles are tracked."""
    cfg = _make_config()
    mtmc = MTMCManager(cfg)

    # Frame with 1 person and 1 car
    boxes = [
        (100.0, 100.0, 50.0, 50.0, "person", 0.9, 1),
        (200.0, 200.0, 60.0, 60.0, "car", 0.85, 2),
    ]
    dets = [["person", 0.9, 1], ["car", 0.85, 2]]

    gids = mtmc.process_frame("cam-1", 1, 10.0, dets, boxes=boxes)
    # Only car (ds_oid = 2) should get a global ID, person (ds_oid = 1) should be ignored
    assert 1 not in gids
    assert 2 in gids


def test_global_id_persists_cam1_cam2_cam3():
    """Verify Global ID persists across Cam 1 -> (disappears > 20s) -> Cam 2 -> (disappears > 20s) -> Cam 3."""
    cfg = _make_config()
    mtmc = MTMCManager(cfg)

    # Vehicle embedding vector (normalized)
    emb_veh_a = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    # --- Step 1: Vehicle A in Camera 1 at t = 10.0s ---
    boxes_c1 = [(100.0, 100.0, 50.0, 50.0, "car", 0.9, 101)]
    dets_c1 = [["car", 0.9, 101]]

    gids_c1 = mtmc.process_frame("cam-1", 1, 10.0, dets_c1, boxes=boxes_c1)
    gid_a = gids_c1[101]

    # Inject embedding directly into active tracklet
    cam1_tr = mtmc._cameras["cam-1"].all_tracklets()[0]
    cam1_tr.embedding = emb_veh_a

    # Process frame 2 to solidify association
    gids_c1 = mtmc.process_frame("cam-1", 2, 10.5, dets_c1, boxes=boxes_c1)
    assert gids_c1[101] == gid_a

    # --- Step 2: Vehicle A disappears from Camera 1 for > 20 seconds ---
    # Advance empty frames on cam-1 until tracklet ages out
    for f in range(3, 25):
        mtmc.process_frame("cam-1", f, 10.5 + (f - 2) * 0.1, [], boxes=[])

    # Check tracklet was popped and global track transitioned to INACTIVE
    assert len(mtmc._cameras["cam-1"].all_tracklets()) == 0
    gtr_a = mtmc.association.global_tracks[gid_a]
    assert gtr_a.state == "INACTIVE"
    assert gtr_a.last_camera_id == "cam-1"

    # --- Step 3: Vehicle A appears in Camera 2 at t = 35.0s (> 20s gap) ---
    t_cam2 = 35.0
    boxes_c2 = [(150.0, 150.0, 50.0, 50.0, "car", 0.92, 201)]
    dets_c2 = [["car", 0.92, 201]]

    mtmc.process_frame("cam-2", 1, t_cam2, dets_c2, boxes=boxes_c2)
    cam2_tr = mtmc._cameras["cam-2"].all_tracklets()[0]
    cam2_tr.embedding = emb_veh_a  # Same appearance embedding

    gids_c2 = mtmc.process_frame("cam-2", 2, t_cam2 + 0.1, dets_c2, boxes=boxes_c2)

    # Vehicle in Cam 2 MUST reuse the persistent Global ID from Cam 1!
    assert gids_c2[201] == gid_a

    # --- Step 4: Vehicle A disappears from Camera 2 for > 20 seconds ---
    for f in range(3, 25):
        mtmc.process_frame("cam-2", f, t_cam2 + 0.1 + (f - 2) * 0.1, [], boxes=[])

    assert len(mtmc._cameras["cam-2"].all_tracklets()) == 0
    assert gtr_a.state == "INACTIVE"
    assert gtr_a.last_camera_id == "cam-2"
    # Should have stored embeddings from both cam-1 and cam-2
    assert "cam-1" in gtr_a.mean_embeddings
    assert "cam-2" in gtr_a.mean_embeddings

    # --- Step 5: Vehicle A appears in Camera 3 at t = 60.0s (> 20s gap) ---
    t_cam3 = 60.0
    boxes_c3 = [(200.0, 200.0, 50.0, 50.0, "car", 0.88, 301)]
    dets_c3 = [["car", 0.88, 301]]

    mtmc.process_frame("cam-3", 1, t_cam3, dets_c3, boxes=boxes_c3)
    cam3_tr = mtmc._cameras["cam-3"].all_tracklets()[0]
    cam3_tr.embedding = emb_veh_a

    gids_c3 = mtmc.process_frame("cam-3", 2, t_cam3 + 0.1, dets_c3, boxes=boxes_c3)

    # Vehicle in Cam 3 MUST reuse the SAME persistent Global ID!
    assert gids_c3[301] == gid_a


def test_disappearance_timing_gate():
    """Verify vehicle disappearing < 20s is gated out, while >= 20s is candidate."""
    assoc = GlobalVehicleAssociation(
        topology=["cam-1", "cam-2"],
        min_disappearance_s=20.0,
        reid_sim_threshold=0.55,
    )
    emb = np.array([1.0, 0.0])

    # Vehicle 0 in cam-1 at t=10.0s
    gid = assoc.get_or_create_global_id("cam-1", 1, 10.0, embedding=emb)
    assoc.mark_tracklet_inactive("cam-1", 1, 12.0)  # disappeared at t=12.0s

    # Attempt match at t=25.0s (elapsed = 13.0s < 20.0s) -> timing gate rejects match
    gid_too_soon = assoc.get_or_create_global_id("cam-2", 1, 25.0, embedding=emb)
    assert gid_too_soon != gid

    # Attempt match at t=35.0s (elapsed = 23.0s >= 20.0s) -> timing gate passes & matches!
    gid_matched = assoc.get_or_create_global_id("cam-2", 2, 35.0, embedding=emb)
    assert gid_matched == gid


def test_low_similarity_creates_new_global_id():
    """Verify candidate below similarity threshold receives a new Global ID."""
    assoc = GlobalVehicleAssociation(
        topology=["cam-1", "cam-2"],
        min_disappearance_s=20.0,
        reid_sim_threshold=0.70,
    )
    emb_veh1 = np.array([1.0, 0.0])
    emb_veh2 = np.array([0.0, 1.0])  # Orthogonal -> sim = 0.0

    gid1 = assoc.get_or_create_global_id("cam-1", 1, 10.0, embedding=emb_veh1)
    assoc.mark_tracklet_inactive("cam-1", 1, 12.0)

    # Vehicle 2 appears in cam-2 after > 20s with orthogonal embedding
    gid2 = assoc.get_or_create_global_id("cam-2", 1, 35.0, embedding=emb_veh2)
    assert gid2 != gid1


def test_topology_gating():
    """Verify camera topology rejects invalid transitions."""
    assoc = GlobalVehicleAssociation(
        topology=[("cam-1", "cam-2"), ("cam-2", "cam-3")],  # cam-1 cannot jump directly to cam-3
        min_disappearance_s=20.0,
        reid_sim_threshold=0.50,
    )
    emb = np.array([1.0, 0.0])

    gid1 = assoc.get_or_create_global_id("cam-1", 1, 10.0, embedding=emb)
    assoc.mark_tracklet_inactive("cam-1", 1, 12.0)

    # Attempt invalid jump: cam-1 -> cam-3 directly
    gid3 = assoc.get_or_create_global_id("cam-3", 1, 35.0, embedding=emb)
    assert gid3 != gid1


def test_historical_crop_and_multi_camera_embedding_retention():
    """Verify GlobalVehicleTrack retains historical crops/embeddings from all visited cameras."""
    assoc = GlobalVehicleAssociation(topology=["cam-1", "cam-2", "cam-3"])
    crop1 = np.zeros((50, 50, 3), dtype=np.uint8)
    crop2 = np.ones((50, 50, 3), dtype=np.uint8) * 255
    emb1 = np.array([1.0, 0.0])
    emb2 = np.array([0.8, 0.6])

    gid = assoc.get_or_create_global_id("cam-1", 1, 10.0, embedding=emb1, crops=[crop1])
    assoc.mark_tracklet_inactive("cam-1", 1, 12.0)

    assoc.get_or_create_global_id("cam-2", 1, 35.0, embedding=emb2, crops=[crop2])

    gtr = assoc.global_tracks[gid]
    assert "cam-1" in gtr.crops and "cam-2" in gtr.crops
    assert "cam-1" in gtr.mean_embeddings and "cam-2" in gtr.mean_embeddings
    assert gtr.combined_embedding() is not None


def test_license_plate_tracking():
    """Verify license_plate class detections are tracked by MTMC."""
    cfg = _make_config(tracked_classes=("license_plate", "plate", "car", "truck"))
    mtmc = MTMCManager(cfg)

    boxes = [(100.0, 100.0, 40.0, 20.0, "license_plate", 0.95, 10)]
    dets = [["license_plate", 0.95, 10]]

    gids = mtmc.process_frame("cam-1", 1, 10.0, dets, boxes=boxes)
    assert 10 in gids
    assert gids[10] >= 0


def test_topology_only_association_without_embeddings():
    """Verify cross-camera vehicle track association works using map topology & timing when Re-ID embeddings are None."""
    assoc = GlobalVehicleAssociation(
        topology=["cam-1", "cam-2"],
        min_disappearance_s=20.0,
        reid_sim_threshold=0.55,
    )

    # Vehicle track in cam-1 at t=10.0s without embedding
    gid1 = assoc.get_or_create_global_id("cam-1", 1, 10.0, embedding=None)
    assoc.mark_tracklet_inactive("cam-1", 1, 12.0)

    # Appears in cam-2 after > 20s (t=35.0s) without embedding
    gid2 = assoc.get_or_create_global_id("cam-2", 1, 35.0, embedding=None)

    # Topology-only mode MUST reuse global identity across cameras!
    assert gid2 == gid1


def _topology_assoc():
    return GlobalVehicleAssociation(
        topology=["cam-1", "cam-2", "cam-3"],
        min_disappearance_s=20.0,
        reid_sim_threshold=0.55,
    )


def test_without_embeddings_the_vehicle_closest_in_time_wins_and_stale_tracks_go_inactive_on_their_own():
    """Topology and timing only (no Re-ID embedding): of the vehicles that could have come from the previous camera,
    the one that disappeared most recently wins (the rule since 2026-09-03; it used to be first-in-first-out). A track
    nobody marked inactive goes stale by itself once it has not been seen for min_disappearance_s."""
    assoc = _topology_assoc()
    gid0 = assoc.get_or_create_global_id("cam-1", local_id=0, pts=26.0)   # last seen 26 s (NOT marked inactive)
    gid1 = assoc.get_or_create_global_id("cam-1", local_id=1, pts=44.0)   # last seen 44 s (NOT marked inactive)
    assert gid0 != gid1
    assert assoc.global_tracks[gid0].state == "ACTIVE" and assoc.global_tracks[gid1].state == "ACTIVE"

    # A vehicle appears on cam-2 at 76 s: 50 s after gid0 was last seen, 32 s after gid1 -- gid1 is the closest.
    assert assoc.get_or_create_global_id("cam-2", local_id=4, pts=76.0) == gid1

    # Another appears on cam-2 at 80 s. gid1 is now on cam-2 itself, so the one still waiting on cam-1 is gid0.
    assert assoc.get_or_create_global_id("cam-2", local_id=5, pts=80.0) == gid0


def test_without_embeddings_a_tie_in_time_goes_to_the_lowest_global_id():
    assoc = _topology_assoc()
    first = assoc.get_or_create_global_id("cam-1", local_id=0, pts=30.0)
    second = assoc.get_or_create_global_id("cam-1", local_id=1, pts=30.0)    # same last-seen time
    assert assoc.get_or_create_global_id("cam-2", local_id=9, pts=70.0) == min(first, second)


def test_a_vehicle_that_shows_up_before_min_disappearance_is_a_new_one():
    assoc = _topology_assoc()
    gid0 = assoc.get_or_create_global_id("cam-1", local_id=0, pts=50.0)
    # only 10 s after cam-1 last saw it (< 20 s): it cannot have driven there
    assert assoc.get_or_create_global_id("cam-2", local_id=3, pts=60.0) != gid0


def test_a_camera_the_topology_does_not_connect_is_never_matched():
    assoc = _topology_assoc()
    gid0 = assoc.get_or_create_global_id("cam-1", local_id=0, pts=10.0)
    # cam-1 -> cam-3 is not an edge of the chain cam-1 -> cam-2 -> cam-3, however long ago it was seen
    assert assoc.get_or_create_global_id("cam-3", local_id=7, pts=200.0) != gid0


def test_per_camera_pair_timing_gaps():
    """Verify per-camera-pair timing gaps override global min_disappearance_s."""
    assoc = GlobalVehicleAssociation(
        topology=["cam-1", "cam-2", "cam-3"],
        min_disappearance_s=50.0,  # global default 50s
        camera_gaps={
            "cam-1->cam-2": 10.0,  # 10s gap between cam-1 and cam-2
            "cam-2->cam-3": 30.0,  # 30s gap between cam-2 and cam-3
        },
    )

    # Vehicle 0 in cam-1 at t=10.0s
    gid0 = assoc.get_or_create_global_id("cam-1", local_id=0, pts=10.0)
    assoc.mark_tracklet_inactive("cam-1", local_id=0, pts=12.0)

    # Attempt match at cam-2 at t=25.0s (elapsed = 13.0s).
    # Global default is 50.0s, but pair gap is 10.0s -> 13.0s >= 10.0s PASSES!
    gid_cam2 = assoc.get_or_create_global_id("cam-2", local_id=1, pts=25.0)
    assert gid_cam2 == gid0
    assoc.mark_tracklet_inactive("cam-2", local_id=1, pts=30.0)

    # Attempt match at cam-3 at t=45.0s (elapsed = 15.0s).
    # Pair gap for cam-2->cam-3 is 30.0s -> 15.0s < 30.0s FAILS (new global ID)!
    gid_cam3_too_soon = assoc.get_or_create_global_id("cam-3", local_id=2, pts=45.0)
    assert gid_cam3_too_soon != gid0

    # Attempt match at cam-3 at t=65.0s (elapsed = 35.0s >= 30.0s) -> PASSES!
    gid_cam3_matched = assoc.get_or_create_global_id("cam-3", local_id=3, pts=65.0)
    assert gid_cam3_matched == gid0


# ---------------------------------------------------------------------------
# VLM arbiter gating: every proposed association -- topology-only,
# high-confidence re-id, or ambiguous re-id -- must be confirmed by the VLM
# before it is allowed to merge. See perception/modules/mtmc/vlm_arbiter.py.
# ---------------------------------------------------------------------------

class _FakeVlmArbiter:
    """Deterministic stand-in for sidecar.vlm_arbiter.VLMArbiter.

    Resolves synchronously: submit() immediately queues a canned verdict that
    take_result() returns on the very next call, so tests don't need to poll
    across frames the way a real (async, network-backed) arbiter requires.
    """

    def __init__(self, decision="same", confidence=0.9, enabled=True):
        self.enabled = enabled
        self.min_confidence = 0.5
        self.max_wait_s = 12.0
        self._decision = decision
        self._confidence = confidence
        self._next_job_id = 0
        self._results: dict[int, dict] = {}
        self.submit_calls = 0

    def submit(self, crop_a, crop_b, context=None) -> int:
        self.submit_calls += 1
        job_id = self._next_job_id
        self._next_job_id += 1
        self._results[job_id] = {"decision": self._decision, "confidence": self._confidence}
        return job_id

    def take_result(self, job_id: int):
        return self._results.pop(job_id, None)


def _crop() -> np.ndarray:
    return np.zeros((50, 50, 3), dtype=np.uint8)


def test_vlm_gates_topology_only_association():
    """A topology/timing-only candidate (no embedding) must wait for VLM 'same'."""
    vlm = _FakeVlmArbiter(decision="same")
    assoc = GlobalVehicleAssociation(
        topology=["cam-1", "cam-2"], min_disappearance_s=20.0, vlm_arbiter=vlm,
    )
    gid0 = assoc.get_or_create_global_id("cam-1", 0, 10.0, crops=[_crop()])
    assoc.mark_tracklet_inactive("cam-1", 0, 12.0)

    # First call: candidate found, VLM job submitted, no gid yet (waiting).
    pending = assoc.get_or_create_global_id("cam-2", 1, 35.0, crops=[_crop()])
    assert pending == -1
    assert vlm.submit_calls == 1

    # Second call (next frame): fake arbiter's canned "same" verdict resolves it.
    gid1 = assoc.get_or_create_global_id("cam-2", 1, 36.0, crops=[_crop()])
    assert gid1 == gid0


def test_vlm_rejects_topology_only_association():
    """A topology-only candidate is NOT merged if the VLM says 'different' --
    even though the pre-VLM code would have merged it outright."""
    vlm = _FakeVlmArbiter(decision="different", confidence=0.95)
    assoc = GlobalVehicleAssociation(
        topology=["cam-1", "cam-2"], min_disappearance_s=20.0, vlm_arbiter=vlm,
    )
    gid0 = assoc.get_or_create_global_id("cam-1", 0, 10.0, crops=[_crop()])
    assoc.mark_tracklet_inactive("cam-1", 0, 12.0)

    assert assoc.get_or_create_global_id("cam-2", 1, 35.0, crops=[_crop()]) == -1
    gid1 = assoc.get_or_create_global_id("cam-2", 1, 36.0, crops=[_crop()])
    assert gid1 != gid0 and gid1 != -1


def test_vlm_rejects_high_similarity_reid_association():
    """Even a similarity above reid_sim_threshold must still clear the VLM --
    the old immediate-merge-on-high-sim behavior no longer applies."""
    vlm = _FakeVlmArbiter(decision="different", confidence=0.9)
    assoc = GlobalVehicleAssociation(
        topology=["cam-1", "cam-2"], min_disappearance_s=20.0,
        reid_sim_threshold=0.55, vlm_arbiter=vlm,
    )
    emb = np.array([1.0, 0.0])
    gid0 = assoc.get_or_create_global_id("cam-1", 0, 10.0, embedding=emb, crops=[_crop()])
    assoc.mark_tracklet_inactive("cam-1", 0, 12.0)

    # Identical embedding -> sim=1.0, well above threshold -- would have
    # auto-merged before this change.
    assert assoc.get_or_create_global_id("cam-2", 1, 35.0, embedding=emb, crops=[_crop()]) == -1
    gid1 = assoc.get_or_create_global_id("cam-2", 1, 36.0, embedding=emb, crops=[_crop()])
    assert gid1 != gid0 and gid1 != -1


def test_vlm_confirms_ambiguous_association():
    """Ambiguous-zone candidate (gate <= sim < match) merges once VLM says 'same'."""
    vlm = _FakeVlmArbiter(decision="same", confidence=0.8)
    assoc = GlobalVehicleAssociation(
        topology=["cam-1", "cam-2"], min_disappearance_s=20.0,
        reid_sim_threshold=0.80, reid_gate_threshold=0.15, vlm_arbiter=vlm,
    )
    emb1 = np.array([1.0, 0.0])
    emb2 = np.array([0.7071, 0.7071])  # sim ~= 0.707: gate <= sim < match
    gid0 = assoc.get_or_create_global_id("cam-1", 0, 10.0, embedding=emb1, crops=[_crop()])
    assoc.mark_tracklet_inactive("cam-1", 0, 12.0)

    assert assoc.get_or_create_global_id("cam-2", 1, 35.0, embedding=emb2, crops=[_crop()]) == -1
    gid1 = assoc.get_or_create_global_id("cam-2", 1, 36.0, embedding=emb2, crops=[_crop()])
    assert gid1 == gid0


def test_vlm_adjudicates_gate_rejected_candidate():
    """A candidate below reid_gate_threshold (would be a hard reject with no
    VLM) still reaches the VLM for adjudication when one is configured, and
    merges on a 'same' verdict -- full-vehicle appearance similarity is too
    weak a signal to hard-veto on alone."""
    vlm = _FakeVlmArbiter(decision="same", confidence=0.85)
    assoc = GlobalVehicleAssociation(
        topology=["cam-1", "cam-2"], min_disappearance_s=20.0,
        reid_sim_threshold=0.80, reid_gate_threshold=0.15, vlm_arbiter=vlm,
    )
    emb1 = np.array([1.0, 0.0])
    emb2 = np.array([0.0, 1.0])  # sim = 0.0: below reid_gate_threshold
    gid0 = assoc.get_or_create_global_id("cam-1", 0, 10.0, embedding=emb1, crops=[_crop()])
    assoc.mark_tracklet_inactive("cam-1", 0, 12.0)

    assert assoc.get_or_create_global_id("cam-2", 1, 35.0, embedding=emb2, crops=[_crop()]) == -1
    assert vlm.submit_calls == 1
    gid1 = assoc.get_or_create_global_id("cam-2", 1, 36.0, embedding=emb2, crops=[_crop()])
    assert gid1 == gid0


def test_vlm_no_crop_blocks_merge_without_waiting():
    """No crop on the new tracklet -> VLM can't be asked -> no merge this frame
    (falls straight through to creating a new track, no async wait)."""
    vlm = _FakeVlmArbiter(decision="same")
    assoc = GlobalVehicleAssociation(
        topology=["cam-1", "cam-2"], min_disappearance_s=20.0, vlm_arbiter=vlm,
    )
    gid0 = assoc.get_or_create_global_id("cam-1", 0, 10.0, crops=[_crop()])
    assoc.mark_tracklet_inactive("cam-1", 0, 12.0)

    gid1 = assoc.get_or_create_global_id("cam-2", 1, 35.0, crops=None)
    assert gid1 != gid0 and gid1 != -1
    assert vlm.submit_calls == 0


def test_vlm_disabled_preserves_pre_vlm_behavior():
    """With the arbiter disabled, topology/high-sim matches auto-merge exactly
    as they did before the VLM gate existed (no regression for that config)."""
    vlm = _FakeVlmArbiter(decision="same", enabled=False)
    assoc = GlobalVehicleAssociation(
        topology=["cam-1", "cam-2"], min_disappearance_s=20.0, vlm_arbiter=vlm,
    )
    gid0 = assoc.get_or_create_global_id("cam-1", 0, 10.0, crops=[_crop()])
    assoc.mark_tracklet_inactive("cam-1", 0, 12.0)

    gid1 = assoc.get_or_create_global_id("cam-2", 1, 35.0, crops=[_crop()])
    assert gid1 == gid0
    assert vlm.submit_calls == 0






class _ScriptedVlm(_FakeVlmArbiter):
    """Answers 'same' only for jobs whose candidate global id is in ``same_gids``."""

    def __init__(self, same_gids, confidence=0.9):
        super().__init__(decision="different", confidence=confidence)
        self._same_gids = set(same_gids)

    def submit(self, crop_a, crop_b, context=None) -> int:
        job_id = super().submit(crop_a, crop_b, context)
        gid = (context or {})["candidate_global_id"]
        self._results[job_id]["decision"] = "same" if gid in self._same_gids else "different"
        return job_id


def test_vlm_primary_judges_every_eligible_candidate_not_just_closest():
    """Two eligible tracks; the closest-in-time one is a different vehicle. The
    VLM picks the farther one -- no FIFO / closest-wins rule decides."""
    far = _ScriptedVlm(same_gids=set())
    assoc = GlobalVehicleAssociation(
        topology=["cam-1", "cam-2"], min_disappearance_s=2.0, vlm_arbiter=far,
    )
    g_old = assoc.get_or_create_global_id("cam-1", 0, 10.0, crops=[_crop()])
    assoc.mark_tracklet_inactive("cam-1", 0, 12.0)
    g_new = assoc.get_or_create_global_id("cam-1", 1, 30.0, crops=[_crop()])
    assoc.mark_tracklet_inactive("cam-1", 1, 32.0)
    assert g_old != g_new

    vlm = _ScriptedVlm(same_gids={g_old})
    assoc._vlm = vlm
    assert assoc.get_or_create_global_id("cam-2", 7, 40.0, crops=[_crop()]) == -1
    assert vlm.submit_calls == 2  # both candidates were asked
    assert assoc.get_or_create_global_id("cam-2", 7, 41.0, crops=[_crop()]) == g_old


def test_vlm_primary_all_different_creates_new_track():
    vlm = _ScriptedVlm(same_gids=set())
    assoc = GlobalVehicleAssociation(
        topology=["cam-1", "cam-2"], min_disappearance_s=2.0, vlm_arbiter=vlm,
    )
    g0 = assoc.get_or_create_global_id("cam-1", 0, 10.0, crops=[_crop()])
    assoc.mark_tracklet_inactive("cam-1", 0, 12.0)
    assert assoc.get_or_create_global_id("cam-2", 1, 30.0, crops=[_crop()]) == -1
    g1 = assoc.get_or_create_global_id("cam-2", 1, 31.0, crops=[_crop()])
    assert g1 not in (-1, g0)


def test_vlm_primary_topology_and_timing_still_filter_candidates():
    """Wrong camera order or a too-early appearance never reaches the VLM."""
    vlm = _ScriptedVlm(same_gids={0})
    assoc = GlobalVehicleAssociation(
        topology=["cam-1", "cam-2", "cam-3"], min_disappearance_s=20.0, vlm_arbiter=vlm,
    )
    assoc.get_or_create_global_id("cam-1", 0, 10.0, crops=[_crop()])
    assoc.mark_tracklet_inactive("cam-1", 0, 12.0)
    # cam-1 -> cam-3 is not a topology edge
    assoc.get_or_create_global_id("cam-3", 1, 60.0, crops=[_crop()])
    # cam-1 -> cam-2 but only 5s after disappearing (< 20s gap)
    assoc.get_or_create_global_id("cam-2", 2, 17.0, crops=[_crop()])
    assert vlm.submit_calls == 0


def test_vlm_primary_reid_similarity_is_not_a_veto():
    """Opposite embeddings (sim 0) with a 'same' verdict still merge."""
    vlm = _ScriptedVlm(same_gids={0})
    assoc = GlobalVehicleAssociation(
        topology=["cam-1", "cam-2"], min_disappearance_s=2.0,
        reid_sim_threshold=0.9, reid_gate_threshold=0.5, vlm_arbiter=vlm,
    )
    g0 = assoc.get_or_create_global_id("cam-1", 0, 10.0, embedding=np.array([1.0, 0.0]), crops=[_crop()])
    assoc.mark_tracklet_inactive("cam-1", 0, 12.0)
    assert assoc.get_or_create_global_id("cam-2", 1, 30.0, embedding=np.array([0.0, 1.0]), crops=[_crop()]) == -1
    assert assoc.get_or_create_global_id("cam-2", 1, 31.0, embedding=np.array([0.0, 1.0]), crops=[_crop()]) == g0


def test_tracklet_views_span_time_and_keep_largest_per_bucket():
    from perception.modules.mtmc.core._tracklets import TrackletRecord
    tr = TrackletRecord(cam_id="cam-1", object_id=1)
    small, big = np.zeros((10, 10, 3), np.uint8), np.zeros((40, 40, 3), np.uint8)
    tr.add_observation(1, 0.1, (0, 0, 10, 10), small)
    tr.add_observation(2, 0.5, (0, 0, 40, 40), big)      # same 1s bucket: larger wins
    tr.add_observation(3, 5.2, (0, 0, 10, 10), small)    # much later bucket
    views = tr.get_view_crops()
    assert len(views) == 2 and views[0] is big and views[1] is small


def test_pick_views_spreads_over_time_and_prefers_larger():
    crops = [np.zeros((s, s, 3), np.uint8) for s in (10, 30, 20, 20, 50, 10, 15, 40, 12)]
    picked = GlobalVehicleAssociation._pick_views(crops, 3)
    assert [c.shape[0] for c in picked] == [30, 50, 40]      # largest of each third, in time order
    assert len(GlobalVehicleAssociation._pick_views(crops[:2], 3)) == 2


def test_vlm_receives_multiple_views_per_vehicle():
    seen = {}

    class Spy(_FakeVlmArbiter):
        def submit(self, a, b, context=None):
            seen["a"], seen["b"] = a, b
            return super().submit(a, b, context)

    assoc = GlobalVehicleAssociation(
        topology=["cam-1", "cam-2"], min_disappearance_s=2.0, vlm_arbiter=Spy(), vlm_views=3,
    )
    views = [_crop() for _ in range(5)]
    assoc.get_or_create_global_id("cam-1", 0, 10.0, crops=[_crop()], views=views)
    assoc.mark_tracklet_inactive("cam-1", 0, 12.0)
    assoc.get_or_create_global_id("cam-2", 1, 30.0, crops=[_crop()], views=views)
    assert len(seen["a"]) == 3 and len(seen["b"]) == 3



def test_gallery_gets_the_highest_confidence_shot_whatever_its_size():
    from perception.modules.mtmc.core._tracklets import TrackletRecord
    calls = []

    class Gallery:
        def maybe_submit(self, *a):
            calls.append(a)

    mgr = MTMCManager.__new__(MTMCManager)
    mgr.vehicle_gallery = Gallery()
    small, big, mid = (np.full((20, 20, 3), 1, np.uint8), np.full((60, 90, 3), 2, np.uint8),
                       np.full((30, 40, 3), 3, np.uint8))
    tr = TrackletRecord(cam_id="cam-03", object_id=1)
    tr.add_observation(1, 1.0, (0, 0, 20, 20), small, 0.91)   # small but most confident
    tr.add_observation(2, 2.0, (0, 0, 90, 60), big, 0.80)     # big, less confident: not taken
    tr.add_observation(3, 3.0, (0, 0, 40, 30), mid, 0.85)
    mgr._submit_gallery("cam-03", tr, 188, 3, 3.0, np.zeros((720, 1280, 3), np.uint8))
    assert len(calls) == 1 and calls[0][2].tolist() == small.tolist() and calls[0][3] == 0.91


def test_gallery_replaces_only_on_a_strictly_better_score(tmp_path, monkeypatch):
    from perception.modules.mtmc.vehicle_gallery import VehicleGallery
    g = VehicleGallery(enabled=True, storage_backend="local", local_dir=str(tmp_path))
    saved = []
    monkeypatch.setattr(g, "_persist", lambda cam, gid, crop, score, *a: saved.append(score))
    img = np.zeros((10, 10, 3), np.uint8)
    for score in (0.7, 0.7, 0.6, 0.9, 0.85):
        g.maybe_submit("cam-01", 5, img, score, score, (0, 0, 10, 10), 0.0)
    g.close()
    time.sleep(0.2)
    assert saved == [0.7, 0.9]


def test_origin_camera_only_tracks_vehicles_that_start_there():
    assoc = GlobalVehicleAssociation(
        topology=["cam-1", "cam-2"], min_disappearance_s=2.0, max_inactive_ttl_s=20.0,
        origin_cameras=["cam-1"])
    g0 = assoc.get_or_create_global_id("cam-1", 0, 10.0)       # origin: starts a track
    assert g0 >= 0
    # non-origin camera, nothing to match (no eligible candidate): never gets an id
    assert assoc.get_or_create_global_id("cam-2", 5, 100.0) == -1
    assert assoc.get_or_create_global_id("cam-2", 5, 101.0) == -1   # decision is final
    assert len(assoc.global_tracks) == 1


def test_origin_camera_still_lets_other_cameras_join_an_origin_track():
    vlm = _FakeVlmArbiter(decision="same")
    assoc = GlobalVehicleAssociation(
        topology=["cam-1", "cam-2"], min_disappearance_s=2.0, vlm_arbiter=vlm,
        origin_cameras=["cam-1"])
    g0 = assoc.get_or_create_global_id("cam-1", 0, 10.0, crops=[_crop()])
    assoc.mark_tracklet_inactive("cam-1", 0, 12.0)
    assert assoc.get_or_create_global_id("cam-2", 1, 30.0, crops=[_crop()]) == -1  # waiting
    assert assoc.get_or_create_global_id("cam-2", 1, 31.0, crops=[_crop()]) == g0


def test_origin_camera_vlm_rejection_leaves_vehicle_untracked():
    vlm = _FakeVlmArbiter(decision="different", confidence=0.95)
    assoc = GlobalVehicleAssociation(
        topology=["cam-1", "cam-2"], min_disappearance_s=2.0, vlm_arbiter=vlm,
        origin_cameras=["cam-1"])
    assoc.get_or_create_global_id("cam-1", 0, 10.0, crops=[_crop()])
    assoc.mark_tracklet_inactive("cam-1", 0, 12.0)
    assoc.get_or_create_global_id("cam-2", 1, 30.0, crops=[_crop()])       # waiting
    assert assoc.get_or_create_global_id("cam-2", 1, 31.0, crops=[_crop()]) == -1
    assert vlm.submit_calls == 1                                            # not re-asked
    assert assoc.get_or_create_global_id("cam-2", 1, 32.0, crops=[_crop()]) == -1
    assert vlm.submit_calls == 1
    assert len(assoc.global_tracks) == 1


def _sized(n):
    return np.zeros((n, n, 3), dtype=np.uint8)


def test_vlm_gets_history_from_every_camera_the_candidate_passed_through():
    """New vehicle on cam-3 is compared against the candidate's crops from cam-1 AND
    cam-2, each captioned; with fewer cameras than vlm_views the spare images go to the latest."""
    seen = {}

    class Spy(_FakeVlmArbiter):
        def submit(self, a, b, context=None):
            seen.update(a=a, b=b, ctx=context)
            return super().submit(a, b, context)

    assoc = GlobalVehicleAssociation(
        topology=["cam-1", "cam-2", "cam-3"], min_disappearance_s=2.0,
        vlm_arbiter=Spy(), vlm_views=3, vlm_history_views=1)
    v1 = [_sized(s) for s in (20, 30, 40, 50, 60)]
    v2 = [_sized(s) for s in (25, 35, 45, 55, 65)]
    g = assoc.get_or_create_global_id("cam-1", 0, 10.0, crops=[_crop()], views=v1)
    assoc.mark_tracklet_inactive("cam-1", 0, 12.0)
    assoc.get_or_create_global_id("cam-2", 1, 30.0, crops=[_crop()], views=v2)       # submit
    assert assoc.get_or_create_global_id("cam-2", 1, 31.0, crops=[_crop()], views=v2) == g
    assoc.mark_tracklet_inactive("cam-2", 1, 33.0)

    seen.clear()
    assoc.get_or_create_global_id("cam-3", 2, 60.0, crops=[_crop()], views=[_crop()] * 3)
    assert len(seen["a"]) == 3
    assert len(seen["b"]) == 3                                   # 3 in all: 1 from cam-1, 2 from cam-2
    caps = seen["ctx"]["captions_b"]
    assert "cam-1" in caps[0] and "earlier" in caps[0]
    assert all("cam-2" in c and "most recent" in c for c in caps[1:])
    assert len(seen["ctx"]["captions_a"]) == 3


def test_a_candidate_seen_on_three_cameras_sends_one_image_per_camera():
    seen = {}

    class Spy(_FakeVlmArbiter):
        def submit(self, a, b, context=None):
            seen.update(a=a, b=b, ctx=context)
            return super().submit(a, b, context)

    assoc = GlobalVehicleAssociation(topology="any", min_disappearance_s=2.0, vlm_arbiter=Spy(), vlm_views=3)
    views = [_sized(s) for s in (20, 30, 40, 50, 60)]
    g = assoc.get_or_create_global_id("cam-1", 0, 10.0, crops=[_crop()], views=views)
    for k, cam in ((1, "cam-2"), (2, "cam-3")):
        assoc.mark_tracklet_inactive("cam-%d" % k, k - 1, 12.0 + 20 * k)
        assoc.get_or_create_global_id(cam, k, 30.0 * k + 10, crops=[_crop()], views=views)
        assert assoc.get_or_create_global_id(cam, k, 30.0 * k + 11, crops=[_crop()], views=views) == g
    assoc.mark_tracklet_inactive("cam-3", 2, 100.0)
    seen.clear()
    assoc.get_or_create_global_id("cam-4", 3, 140.0, crops=[_crop()], views=[_crop()] * 3)
    caps = seen["ctx"]["captions_b"]
    assert len(seen["b"]) == 3 and [c.split(" on ")[1].split(" ")[0] for c in caps] == ["cam-1", "cam-2", "cam-3"]
    assert all("view" not in c for c in caps)


def test_vlm_waits_for_views_before_asking_then_asks_once():
    vlm = _FakeVlmArbiter(decision="same")
    assoc = GlobalVehicleAssociation(
        topology=["cam-1", "cam-2"], min_disappearance_s=2.0, vlm_arbiter=vlm)
    g = assoc.get_or_create_global_id("cam-1", 0, 10.0, crops=[_crop()])
    assoc.mark_tracklet_inactive("cam-1", 0, 12.0)
    for t in (30.0, 30.5):                                       # still collecting views
        assert assoc.get_or_create_global_id("cam-2", 1, t, crops=[_crop()], views_ready=False) == -1
    assert vlm.submit_calls == 0
    assert assoc.get_or_create_global_id("cam-2", 1, 31.0, crops=[_crop()], views_ready=True) == -1
    assert vlm.submit_calls == 1                                 # submitted once ready
    assert assoc.get_or_create_global_id("cam-2", 1, 31.5, crops=[_crop()], views_ready=False) == g


# ---------------------------------------------------------------------------
# A full cycle through the gate: cameras connect both ways, a vehicle can leave through
# the camera it came in by, and leaving closes its track.
# ---------------------------------------------------------------------------

def test_topology_any_lets_a_vehicle_go_back_and_skip_cameras():
    vlm = _ScriptedVlm(same_gids={0})
    assoc = GlobalVehicleAssociation(
        topology="any", min_disappearance_s=2.0, vlm_arbiter=vlm)
    g0 = assoc.get_or_create_global_id("cam-3", 0, 10.0, crops=[_crop()])
    assoc.mark_tracklet_inactive("cam-3", 0, 12.0)
    # cam-3 -> cam-1 (backwards, skipping cam-2) is allowed now
    assert assoc.get_or_create_global_id("cam-1", 1, 30.0, crops=[_crop()]) == -1     # asking the VLM
    assert vlm.submit_calls == 1
    assert assoc.get_or_create_global_id("cam-1", 1, 31.0, crops=[_crop()]) == g0


def test_pair_gaps_apply_in_both_directions():
    assoc = GlobalVehicleAssociation(
        topology="any", min_disappearance_s=2.0, camera_gaps={"cam-1->cam-2": 9.0})
    assert assoc.get_min_disappearance_s("cam-1", "cam-2") == 9.0
    assert assoc.get_min_disappearance_s("cam-2", "cam-1") == 9.0      # same pair, other way
    assert assoc.get_min_disappearance_s("cam-2", "cam-3") == 2.0      # unlisted: the default


def test_same_camera_reappearance_only_for_a_vehicle_leaving_through_the_gate():
    vlm = _ScriptedVlm(same_gids={0})
    assoc = GlobalVehicleAssociation(topology="any", min_disappearance_s=2.0, vlm_arbiter=vlm)
    g0 = assoc.get_or_create_global_id("cam-1", 0, 10.0, crops=[_crop()])
    # no gate information: a second tracklet on the same camera is just another vehicle
    assoc.get_or_create_global_id("cam-1", 1, 11.0, crops=[_crop()])
    assert vlm.submit_calls == 0
    # leaving through the gate while g0 is still in view: still not a candidate (it is another car)
    assoc.note_gate("cam-1", 2, "exit")
    assoc.get_or_create_global_id("cam-1", 2, 12.0, crops=[_crop()])
    assert vlm.submit_calls == 0
    # once g0 has left the view, a vehicle leaving through the same gate may be g0
    assoc.mark_tracklet_inactive("cam-1", 0, 13.0)
    assoc.note_gate("cam-1", 3, "exit")
    assert assoc.get_or_create_global_id("cam-1", 3, 40.0, crops=[_crop()]) == -1
    assert vlm.submit_calls >= 1
    assert assoc.get_or_create_global_id("cam-1", 3, 41.0, crops=[_crop()]) == g0


def test_tracklet_aging_out_with_a_pending_verdict_does_not_crash():
    vlm = _ScriptedVlm(same_gids=set())
    assoc = GlobalVehicleAssociation(topology="any", min_disappearance_s=2.0, vlm_arbiter=vlm)
    assoc.get_or_create_global_id("cam-1", 0, 10.0, crops=[_crop()])
    assoc.mark_tracklet_inactive("cam-1", 0, 12.0)
    assoc.get_or_create_global_id("cam-2", 1, 30.0, crops=[_crop()])      # verdict pending
    assoc.mark_tracklet_inactive("cam-2", 1, 31.0)                        # ages out first: logs, no KeyError


def test_entering_vehicle_is_new_without_asking_the_vlm():
    vlm = _ScriptedVlm(same_gids={0})
    assoc = GlobalVehicleAssociation(
        topology="any", min_disappearance_s=2.0, vlm_arbiter=vlm, origin_cameras=["cam-1"])
    assoc.get_or_create_global_id("cam-1", 0, 10.0, crops=[_crop()])
    assoc.mark_tracklet_inactive("cam-1", 0, 12.0)
    assoc.note_gate("cam-1", 7, "enter")
    g = assoc.get_or_create_global_id("cam-1", 7, 40.0, crops=[_crop()])
    assert g not in (-1, 0) and vlm.submit_calls == 0


def test_leaving_vehicle_is_matched_to_one_inside_and_its_track_closed():
    vlm = _ScriptedVlm(same_gids={0})
    assoc = GlobalVehicleAssociation(
        topology="any", min_disappearance_s=2.0, vlm_arbiter=vlm, origin_cameras=["cam-1"])
    g0 = assoc.get_or_create_global_id("cam-1", 0, 10.0, crops=[_crop()])      # came in
    g1 = assoc.get_or_create_global_id("cam-1", 1, 11.0, crops=[_crop()])      # another came in
    assoc.mark_tracklet_inactive("cam-1", 0, 12.0)
    assoc.mark_tracklet_inactive("cam-1", 1, 13.0)
    # later a vehicle crosses the line going out: the VLM says it is g0
    assoc.note_gate("cam-1", 9, "exit")
    assert assoc.get_or_create_global_id("cam-1", 9, 60.0, crops=[_crop()]) == -1
    assert assoc.get_or_create_global_id("cam-1", 9, 61.0, crops=[_crop()]) == g0
    assert assoc.close_track(g0) is True and assoc.close_track(g0) is False    # closed once
    asked = vlm.submit_calls
    # g0 has left: a new tracklet on another camera is never compared with it again (only g1)
    assoc.get_or_create_global_id("cam-2", 10, 90.0, crops=[_crop()])
    assert vlm.submit_calls == asked + 1
    assert assoc.global_tracks[g1].exited is False


def test_leaving_vehicle_that_matches_nothing_inside_starts_no_track():
    vlm = _ScriptedVlm(same_gids=set())
    assoc = GlobalVehicleAssociation(
        topology="any", min_disappearance_s=2.0, vlm_arbiter=vlm, origin_cameras=["cam-1"])
    assoc.get_or_create_global_id("cam-1", 0, 10.0, crops=[_crop()])
    assoc.mark_tracklet_inactive("cam-1", 0, 12.0)
    assoc.note_gate("cam-1", 9, "exit")
    assoc.get_or_create_global_id("cam-1", 9, 60.0, crops=[_crop()])           # waiting
    assert assoc.get_or_create_global_id("cam-1", 9, 61.0, crops=[_crop()]) == -1
    assert len(assoc.global_tracks) == 1                                       # no phantom track


def test_exit_of_a_tracklet_that_already_has_a_track_closes_it():
    assoc = GlobalVehicleAssociation(topology="any", min_disappearance_s=2.0, origin_cameras=["cam-1"])
    g = assoc.get_or_create_global_id("cam-1", 0, 10.0)
    assert assoc.note_gate("cam-1", 0, "exit") == g
    assert assoc.global_tracks[g].exited is True
    assert assoc.note_gate("cam-1", 0, "exit") is None           # already closed


def test_topology_mesh_of_listed_cameras_excludes_other_cameras():
    assoc = GlobalVehicleAssociation(topology={"any": ["cam-1", "cam-2", "cam-3"]}, min_disappearance_s=2.0)
    assert assoc.is_valid_transition("cam-3", "cam-1") and assoc.is_valid_transition("cam-1", "cam-3")
    assert assoc.is_valid_transition("cam-2", "cam-1")
    # a camera that isn't listed is not part of it, in either role
    assert not assoc.is_valid_transition("cam-1", "cam-9")
    assert not assoc.is_valid_transition("cam-9", "cam-1")


def test_new_vehicle_on_a_gate_camera_waits_for_the_line_then_falls_back():
    assoc = GlobalVehicleAssociation(topology="any", min_disappearance_s=2.0, origin_cameras=["cam-1"],
                                     gate_wait_s=10.0)
    assoc.set_gate_cameras({"cam-1"})
    # first seen at pts 100: no direction yet -> no id, no track
    assert assoc.get_or_create_global_id("cam-1", 0, 100.0, first_pts=100.0) == -1
    assert assoc.get_or_create_global_id("cam-1", 0, 105.0, first_pts=100.0) == -1
    assert len(assoc.global_tracks) == 0
    # the line says it is entering: a new vehicle right away
    assoc.note_gate("cam-1", 0, "enter")
    assert assoc.get_or_create_global_id("cam-1", 0, 106.0, first_pts=100.0) == 0
    # a vehicle the line never ruled on is judged without it once the wait is over
    assert assoc.get_or_create_global_id("cam-1", 1, 200.0, first_pts=200.0) == -1
    assert assoc.get_or_create_global_id("cam-1", 1, 211.0, first_pts=200.0) >= 0
    # a camera without a gate line never waits
    assoc.set_gate_cameras(set())
    assert assoc.get_or_create_global_id("cam-1", 2, 400.0, first_pts=400.0) >= 0


def test_leaving_with_nothing_inside_starts_no_track_even_with_the_wait():
    assoc = GlobalVehicleAssociation(topology="any", min_disappearance_s=2.0, origin_cameras=["cam-1"])
    assoc.set_gate_cameras({"cam-1"})
    assoc.note_gate("cam-1", 4, "exit")
    assert assoc.get_or_create_global_id("cam-1", 4, 50.0, first_pts=49.0) == -1
    assert len(assoc.global_tracks) == 0


def _manager():
    from perception.modules.mtmc.core import MTMCManager
    return MTMCManager(_make_config(reid_enabled=False, origin_cameras=["cam-1"], topology="any"))


def test_manager_publishes_an_exit_when_the_gate_reports_it_for_a_tracked_vehicle():
    m = _manager()
    a = m.association
    g = a.get_or_create_global_id("cam-1", 7, 10.0)          # tracked vehicle on the gate camera
    m.apply_gate_event("cam-1", 7, "exit", at=123.0)
    ev = m.pop_exit_events()
    assert len(ev) == 1 and ev[0]["global_id"] == g and ev[0]["at"] == 123.0
    assert a.global_tracks[g].exited is True
    m.apply_gate_event("cam-1", 7, "exit", at=124.0)           # already closed: no second event
    assert m.pop_exit_events() == []


def test_manager_publishes_an_exit_when_the_vehicle_is_identified_after_the_gate_ruled():
    vlm = _ScriptedVlm(same_gids={0})
    m = _manager()
    m.association = GlobalVehicleAssociation(topology="any", min_disappearance_s=2.0, vlm_arbiter=vlm,
                                             origin_cameras=["cam-1"])
    a = m.association
    g0 = a.get_or_create_global_id("cam-1", 0, 10.0, crops=[_crop()])
    a.mark_tracklet_inactive("cam-1", 0, 12.0)
    m.apply_gate_event("cam-1", 9, "exit", at=50.0)            # ruled before anyone knows who it is
    assert m.pop_exit_events() == []
    assert a.get_or_create_global_id("cam-1", 9, 60.0, crops=[_crop()]) == -1
    gid = a.get_or_create_global_id("cam-1", 9, 61.0, crops=[_crop()])
    assert gid == g0
    m._close_exited(gid, "cam-1", 9)                           # what process_frame does on assignment
    ev = m.pop_exit_events()
    assert len(ev) == 1 and ev[0]["global_id"] == g0 and ev[0]["at"] == 50.0


def test_overlap_tolerance_zero_never_merges_simultaneous_tracks_on_different_cameras():
    def run(overlap):
        vlm = _ScriptedVlm(same_gids={0})
        assoc = GlobalVehicleAssociation(topology="any", min_disappearance_s=2.0, vlm_arbiter=vlm,
                                         origin_cameras=["cam-1"], overlap_tolerance_s=overlap)
        assoc.get_or_create_global_id("cam-1", 0, 10.0, crops=[_crop()])     # still in view on cam-1
        assoc.get_or_create_global_id("cam-1", 0, 12.0, crops=[_crop()])
        r = assoc.get_or_create_global_id("cam-2", 7, 13.0, crops=[_crop()])  # cam-2 sees one at once
        return vlm.submit_calls, r
    calls_default, _ = run(5.0)
    calls_off, r_off = run(0.0)
    assert calls_default >= 1            # with the overlap window the VLM is asked
    assert calls_off == 0 and r_off == -1  # without it a simultaneous sighting is never a candidate


# ---------------------------------------------------------------------------
# Camera groups: cameras that overlap, linked to their neighbours with a minimum travel time.
# The real layout: {D41, D16} -5s- {D52} -3s- {CH x6} -2s- {D15, D18} -2s- {D28, D31}.
# ---------------------------------------------------------------------------

GROUPS = {
    "gate": ["cam-01", "cam-02"],
    "weigh": ["cam-03"],
    "yard": ["cam-04", "cam-05", "cam-06", "cam-07", "cam-08", "cam-09"],
    "store": ["cam-10", "cam-11"],
    "back": ["cam-12", "cam-13"],
}
LINKS = [["gate", "weigh", 5.0], ["weigh", "yard", 3.0], ["yard", "store", 2.0], ["store", "back", 2.0]]


def _layout(**kw):
    return GlobalVehicleAssociation(camera_groups=GROUPS, group_links=LINKS, min_disappearance_s=30.0,
                                    reid_sim_threshold=0.5, **kw)


def test_groups_expand_to_the_allowed_moves_both_ways_and_nothing_else():
    a = _layout()
    ok = a.is_valid_transition
    assert ok("cam-01", "cam-03") and ok("cam-02", "cam-03")                  # gate -> weigh
    assert ok("cam-03", "cam-01")                                             # ...and back out
    assert ok("cam-03", "cam-08") and ok("cam-09", "cam-03")                  # weigh <-> yard
    assert ok("cam-06", "cam-10") and ok("cam-11", "cam-04")                  # yard <-> store
    assert ok("cam-10", "cam-13") and ok("cam-12", "cam-11")                  # store <-> back
    assert ok("cam-04", "cam-09") and ok("cam-01", "cam-02")                  # inside a group
    assert not ok("cam-01", "cam-06")                                         # gate -> yard skips weigh
    assert not ok("cam-03", "cam-10")                                         # weigh -> store skips yard
    assert not ok("cam-01", "cam-13") and not ok("cam-13", "cam-02")          # entrance <-> far end


def test_minimum_travel_times_follow_the_links_and_are_zero_inside_a_group():
    a = _layout()
    g = a.get_min_disappearance_s
    assert g("cam-01", "cam-03") == 5.0 and g("cam-03", "cam-02") == 5.0
    assert g("cam-03", "cam-05") == 3.0 and g("cam-07", "cam-03") == 3.0
    assert g("cam-06", "cam-10") == 2.0 and g("cam-11", "cam-09") == 2.0
    assert g("cam-10", "cam-12") == 2.0 and g("cam-13", "cam-11") == 2.0
    assert g("cam-04", "cam-05") == 0.0 and g("cam-12", "cam-13") == 0.0


def test_only_cameras_of_the_same_group_may_overlap():
    a = _layout(overlap_tolerance_s=5.0)
    assert a.overlap_tolerance_for("cam-04", "cam-08") == 5.0      # both in the yard
    assert a.overlap_tolerance_for("cam-01", "cam-02") == 5.0
    assert a.overlap_tolerance_for("cam-03", "cam-04") == 0.0      # different groups
    assert a.overlap_tolerance_for("cam-06", "cam-10") == 0.0


def test_a_vehicle_still_in_view_matches_an_overlapping_neighbour_but_not_a_far_camera():
    vlm = _ScriptedVlm(same_gids={0})
    a = _layout(overlap_tolerance_s=5.0, vlm_arbiter=vlm, origin_cameras=["cam-01"])
    a.get_or_create_global_id("cam-01", 0, 10.0, crops=[_crop()])            # in view at the gate
    a.get_or_create_global_id("cam-01", 0, 12.0, crops=[_crop()])
    # the other gate camera sees a vehicle 1 s later: overlapping views, so the VLM is asked
    assert a.get_or_create_global_id("cam-02", 7, 13.0, crops=[_crop()]) == -1
    assert vlm.submit_calls == 1
    # a yard camera is not adjacent to the gate at all
    a.get_or_create_global_id("cam-06", 8, 13.0, crops=[_crop()])
    assert vlm.submit_calls == 1


def test_a_vehicle_cannot_hop_to_the_next_group_faster_than_the_link_time():
    vlm = _ScriptedVlm(same_gids={0})
    a = _layout(overlap_tolerance_s=5.0, vlm_arbiter=vlm, origin_cameras=["cam-01"])
    a.get_or_create_global_id("cam-01", 0, 10.0, crops=[_crop()])
    a.mark_tracklet_inactive("cam-01", 0, 12.0)
    a.get_or_create_global_id("cam-03", 5, 15.0, crops=[_crop()])            # 3 s after: < 5 s link time
    assert vlm.submit_calls == 0
    assert a.get_or_create_global_id("cam-03", 6, 20.0, crops=[_crop()]) == -1   # 8 s after: allowed, asks the VLM
    assert vlm.submit_calls == 1


def test_groups_reject_a_camera_in_two_groups_and_links_to_unknown_groups():
    import pytest
    with pytest.raises(ValueError):
        GlobalVehicleAssociation(camera_groups={"a": ["cam-1"], "b": ["cam-1"]}, group_links=[])
    with pytest.raises(ValueError):
        GlobalVehicleAssociation(camera_groups={"a": ["cam-1"]}, group_links=[["a", "nope", 1.0]])


def test_global_ids_continue_after_a_restart_instead_of_reusing_old_ones():
    store = {}
    a = GlobalVehicleAssociation(topology=["cam-1"], origin_cameras=["cam-1"])
    a.on_new_gid = lambda n: store.__setitem__("next", n)
    assert [a.get_or_create_global_id("cam-1", i, 10.0 + i) for i in range(3)] == [0, 1, 2]
    assert store["next"] == 3
    # the sidecar restarts: a fresh matcher is told where numbering stopped
    b = GlobalVehicleAssociation(topology=["cam-1"], origin_cameras=["cam-1"])
    b.on_new_gid = lambda n: store.__setitem__("next", n)
    b.start_gid_at(store["next"])
    assert b.get_or_create_global_id("cam-1", 0, 10.0) == 3 and store["next"] == 4
    b.start_gid_at(1)                       # a stale lower value never moves numbering backwards
    assert b.get_or_create_global_id("cam-1", 1, 11.0) == 4


def test_a_vehicle_its_own_camera_lost_can_be_refound_under_a_new_tracker_id():
    def run(regain):
        vlm = _FakeVlmArbiter(decision="same", confidence=0.95)
        assoc = GlobalVehicleAssociation(
            topology=["cam-1"], min_disappearance_s=2.0, vlm_arbiter=vlm, same_camera_regain_s=regain,
        )
        gid0 = assoc.get_or_create_global_id("cam-1", 0, 10.0, crops=[_crop()])
        assoc.get_or_create_global_id("cam-1", 0, 12.0, crops=[_crop()])   # last seen at 12; tracker then loses it
        gids = [assoc.get_or_create_global_id("cam-1", 1, 30.0 + i, crops=[_crop()]) for i in range(3)]
        return gid0, [g for g in gids if g != -1][0] if any(g != -1 for g in gids) else -1
    old, new = run(0.0)
    assert new != old                      # off: the new tracker id is a different vehicle
    old, new = run(45.0)
    assert new == old                      # on: the VLM re-joins it to the vehicle the camera lost


def test_a_new_tracker_id_on_top_of_a_live_track_is_judged_against_it_not_skipped():
    def run(overlapping):
        vlm = _FakeVlmArbiter(decision="same", confidence=0.95)
        assoc = GlobalVehicleAssociation(
            topology=["cam-1"], min_disappearance_s=2.0, vlm_arbiter=vlm, same_camera_regain_s=45.0,
        )
        gid0 = assoc.get_or_create_global_id("cam-1", 0, 10.0, crops=[_crop()])
        assoc.get_or_create_global_id("cam-1", 0, 12.0, crops=[_crop()])   # still in view
        gids = [assoc.get_or_create_global_id("cam-1", 1, 12.0 + 0.1 * i, crops=[_crop()],
                                              overlapping_gids=frozenset({gid0}) if overlapping else frozenset())
                for i in range(3)]
        return gid0, [g for g in gids if g != -1][0] if any(g != -1 for g in gids) else -1
    old, new = run(False)
    assert new != old                      # another vehicle in view on the same camera
    old, new = run(True)
    assert new == old                      # the tracker's second box on the same vehicle: same vehicle


def test_box_overlap_is_measured_against_the_smaller_box():
    from perception.modules.mtmc.core import _boxes_overlap
    assert _boxes_overlap((0, 0, 100, 100), (10, 10, 40, 40))        # one inside the other
    assert not _boxes_overlap((0, 0, 100, 100), (200, 0, 50, 50))    # apart
    assert not _boxes_overlap((0, 0, 100, 100), (90, 90, 100, 100))  # a corner touching


def _two_cams(**kw):
    vlm = _FakeVlmArbiter(decision="same", confidence=0.95)
    return GlobalVehicleAssociation(camera_groups={"entrance": ["cam-1", "cam-2"]}, group_links=[],
                                    min_disappearance_s=2.0, vlm_arbiter=vlm, origin_cameras=["cam-1", "cam-2"],
                                    same_camera_regain_s=45.0, **kw)


def test_two_tracker_ids_of_one_vehicle_on_a_camera_both_keep_its_global_id():
    a = _two_cams()
    gid = a.get_or_create_global_id("cam-1", 30, 10.0, crops=[_crop()])
    for i in range(3):   # 28 sits on top of 30: the tracker split one vehicle in two
        a.get_or_create_global_id("cam-1", 30, 11.0 + i, crops=[_crop()])
        a.get_or_create_global_id("cam-1", 28, 11.0 + i, crops=[_crop()], overlapping_gids=frozenset({gid}))
    assert a.get_or_create_global_id("cam-1", 28, 15.0, crops=[_crop()]) == gid
    assert a.get_or_create_global_id("cam-1", 30, 15.0, crops=[_crop()]) == gid   # not a new id


def test_a_vehicle_still_in_view_on_the_other_camera_is_not_new_when_it_shows_up_again():
    a = _two_cams()
    gid = a.get_or_create_global_id("cam-1", 1, 10.0, crops=[_crop()])
    for i in range(3):
        a.get_or_create_global_id("cam-2", 5, 10.0 + i, crops=[_crop()])   # overlap: cam-2 joins it
    got = [a.get_or_create_global_id("cam-2", 5, 14.0, crops=[_crop()])]
    a.get_or_create_global_id("cam-1", 1, 14.0, crops=[_crop()])
    a.mark_tracklet_inactive("cam-1", 1, 14.0)      # it left cam-1's view but stays on cam-2
    assert a.global_tracks[gid].state == "ACTIVE"
    a.get_or_create_global_id("cam-2", 5, 20.0, crops=[_crop()])            # last seen on cam-2 ...
    a.global_tracks[gid].last_camera_id = "cam-1"                           # ... whichever camera updated last
    new = [a.get_or_create_global_id("cam-1", 9, 21.0 + 0.1 * i, crops=[_crop()]) for i in range(3)]
    assert [g for g in new if g != -1][0] == gid


# ---------------------------------------------------------------------------
# Licence plates: found by the plate detector on a vehicle's crop, attached to that vehicle
# ---------------------------------------------------------------------------

def _plate_frame(mtmc, frame, pts, plates, cars=((2, 200.0), (3, 600.0))):
    boxes = [(x, 200.0, 300.0, 200.0, "car", 0.9, oid) for oid, x in cars]
    dets = [["car", 0.9, oid] for oid, _ in cars]
    frame_bgr = np.zeros((720, 1280, 3), np.uint8)
    return mtmc.process_frame("cam-1", frame, pts, dets, frame_bgr=frame_bgr, boxes=boxes, plates=plates)


def test_plate_is_attached_to_the_vehicle_it_was_found_on():
    mtmc = MTMCManager(_make_config())
    gids = _plate_frame(mtmc, 1, 10.0, [(260.0, 330.0, 60.0, 20.0, 0.8, 2)])
    ev = mtmc.pop_plate_events()
    assert [(e["global_id"], e["cam_id"]) for e in ev] == [(gids[2], "cam-1")]
    assert ev[0]["crop"].shape[:2] == (20, 60) and ev[0]["conf"] == 0.8
    assert mtmc.pop_plate_events() == []        # drained


def test_plate_without_a_live_parent_goes_to_the_smallest_box_holding_it():
    mtmc = MTMCManager(_make_config())
    # parent id 99 is not a live tracklet; the plate sits inside the second car's box
    gids = _plate_frame(mtmc, 1, 10.0, [(650.0, 330.0, 60.0, 20.0, 0.7, 99)])
    assert [e["global_id"] for e in mtmc.pop_plate_events()] == [gids[3]]


def test_plate_outside_every_vehicle_is_dropped():
    mtmc = MTMCManager(_make_config())
    _plate_frame(mtmc, 1, 10.0, [(1100.0, 600.0, 60.0, 20.0, 0.9, -1)])
    assert mtmc.pop_plate_events() == []


def test_only_a_better_plate_is_announced_again():
    mtmc = MTMCManager(_make_config())
    _plate_frame(mtmc, 1, 10.0, [(260.0, 330.0, 40.0, 14.0, 0.6, 2)])
    assert len(mtmc.pop_plate_events()) == 1
    _plate_frame(mtmc, 2, 10.1, [(260.0, 330.0, 30.0, 10.0, 0.9, 2)])   # smaller: worse
    assert mtmc.pop_plate_events() == []
    _plate_frame(mtmc, 3, 10.2, [(260.0, 330.0, 70.0, 24.0, 0.9, 2)])   # bigger and surer: better
    assert len(mtmc.pop_plate_events()) == 1


def test_every_camera_announces_its_own_best_plate_for_the_same_vehicle():
    from types import SimpleNamespace
    mtmc = MTMCManager(_make_config())
    tr = SimpleNamespace(global_id=7, object_id=1,
                         plate={"score": 100.0, "conf": 0.9, "box": (0, 0, 10, 10), "crop": None})
    mtmc._publish_plate("cam-1", tr)
    weaker = SimpleNamespace(global_id=7, object_id=4,
                             plate={"score": 20.0, "conf": 0.5, "box": (0, 0, 5, 4), "crop": None})
    mtmc._publish_plate("cam-2", weaker)         # another camera: announced although weaker
    mtmc._publish_plate("cam-2", weaker)         # the same camera, not better: silent
    assert [(e["global_id"], e["cam_id"]) for e in mtmc.pop_plate_events()] == [(7, "cam-1"), (7, "cam-2")]


def test_plate_seen_before_the_vehicle_has_an_id_is_announced_once_it_gets_one():
    mtmc = MTMCManager(_make_config())
    gids = _plate_frame(mtmc, 1, 10.0, [(260.0, 330.0, 60.0, 20.0, 0.8, 2)])
    assert gids[2] >= 0
    assert len(mtmc.pop_plate_events()) == 1


# ---------------------------------------------------------------------------
# The same car seen by both gate cameras at once: whichever is judged second must be compared
# with the id the first one got in the meantime.
# ---------------------------------------------------------------------------

def _two_older_vehicles(vlm):
    a = _layout(overlap_tolerance_s=5.0, vlm_arbiter=vlm, origin_cameras=["cam-01", "cam-02"])
    assert a.get_or_create_global_id("cam-02", 0, 10.0, crops=[_crop()]) == 0       # gid 0, on cam-02
    a.mark_tracklet_inactive("cam-02", 0, 12.0)
    assert a.get_or_create_global_id("cam-01", 1, 60.0, crops=[_crop()]) == -1      # asked about gid 0
    assert a.get_or_create_global_id("cam-01", 1, 61.0, crops=[_crop()]) == 1       # 'different': gid 1
    a.mark_tracklet_inactive("cam-01", 1, 63.0)
    return a


def test_new_sighting_is_compared_with_a_vehicle_that_appeared_while_it_waited():
    vlm = _ScriptedVlm(same_gids={2})                 # the car the other camera is about to create
    a = _two_older_vehicles(vlm)
    # both gate cameras see the same new car; each question goes out before the other has an id
    assert a.get_or_create_global_id("cam-02", 10, 120.0, crops=[_crop()]) == -1     # vs gid 1
    assert a.get_or_create_global_id("cam-01", 92, 120.1, crops=[_crop()]) == -1     # vs gid 0
    assert a.get_or_create_global_id("cam-02", 10, 121.0, crops=[_crop()]) == 2      # 'different': new
    # cam-01's answer is 'different from gid 0' too, but gid 2 now exists and was never compared
    assert a.get_or_create_global_id("cam-01", 92, 121.1, crops=[_crop()]) == -1
    assert a.get_or_create_global_id("cam-01", 92, 122.0, crops=[_crop()]) == 2
    assert a._next_gid == 3                                                         # no second id


def test_new_sighting_with_nothing_new_to_compare_still_becomes_a_new_vehicle():
    vlm = _ScriptedVlm(same_gids=set())
    a = _two_older_vehicles(vlm)
    asked = vlm.submit_calls
    assert a.get_or_create_global_id("cam-02", 10, 120.0, crops=[_crop()]) == -1
    assert a.get_or_create_global_id("cam-02", 10, 121.0, crops=[_crop()]) == 2
    assert vlm.submit_calls == asked + 1                  # nothing new appeared: no extra question


def test_a_tracklet_is_held_by_exactly_one_global_id():
    """Every real assignment path gives one tracker id to one global id; the invariant check stays at 0, and
    if two ids ever did hold it the conflict is counted and only the first keeps it."""
    assoc = GlobalVehicleAssociation(topology="any", min_disappearance_s=2.0, vlm_arbiter=_ScriptedVlm(same_gids={0, 1}),
                                     origin_cameras=["cam-1"])
    g0 = assoc.get_or_create_global_id("cam-1", 0, 10.0, crops=[_crop()])
    assoc.mark_tracklet_inactive("cam-1", 0, 12.0)
    g1 = assoc.get_or_create_global_id("cam-1", 5, 40.0, crops=[_crop()])      # a different vehicle
    assoc.mark_tracklet_inactive("cam-1", 5, 42.0)
    # two candidates both say "same" for one new tracklet: it still joins exactly one of them
    for pts in (80.0, 81.0, 82.0):
        assoc.get_or_create_global_id("cam-2", 7, pts, crops=[_crop()])
    holders = [g.global_id for g in assoc.global_tracks.values() if 7 in g.local_track_aliases.get("cam-2", ())]
    assert len(holders) <= 1 and assoc.alias_conflicts == 0
    # force the impossible and see it counted
    a, b = assoc.global_tracks[g0], assoc.global_tracks[g1]
    a.local_track_aliases.setdefault("cam-9", set()).add(99)
    b.local_track_aliases.setdefault("cam-9", set()).add(99)
    assoc.get_or_create_global_id("cam-9", 99, 100.0)
    assert assoc.alias_conflicts == 1
    assert [g.global_id for g in assoc.global_tracks.values() if 99 in g.local_track_aliases.get("cam-9", ())] == [g0]


def _one_vehicle_exits_on_cam1():
    a = _two_cams(overlap_tolerance_s=5.0)
    gid = a.get_or_create_global_id("cam-1", 1, 10.0, crops=[_crop()])
    a.get_or_create_global_id("cam-1", 1, 12.0, crops=[_crop()])
    return a, gid


def test_a_question_already_out_is_still_answered_when_the_vehicle_leaves_through_the_overlapping_camera():
    a, gid = _one_vehicle_exits_on_cam1()
    assert a.get_or_create_global_id("cam-2", 5, 12.0, crops=[_crop()]) == -1     # asked about gid
    a.close_track(gid)                                                              # cam-1 saw it leave
    a.get_or_create_global_id("cam-1", 1, 12.5, crops=[_crop()])
    assert a.get_or_create_global_id("cam-2", 5, 13.0, crops=[_crop()]) == gid     # not a new id
    assert a._next_gid == 1


def test_only_the_overlapping_camera_may_still_join_a_vehicle_that_has_left():
    a, gid = _one_vehicle_exits_on_cam1()
    a.close_track(gid)
    # the overlapping neighbour, within the overlap window of its last sighting: still that vehicle
    assert a.get_or_create_global_id("cam-2", 5, 13.0, crops=[_crop()]) == -1
    assert a.get_or_create_global_id("cam-2", 5, 14.0, crops=[_crop()]) == gid
    # ... but not once the window has passed: a vehicle seen after it left is a new visit
    b, gid2 = _one_vehicle_exits_on_cam1()
    b.close_track(gid2)
    assert b.get_or_create_global_id("cam-2", 6, 40.0, crops=[_crop()]) not in (gid2, -1)


def test_a_vehicle_live_on_the_neighbouring_camera_is_always_asked_about_even_past_the_cap():
    import numpy as np
    vlm = _ScriptedVlm(same_gids={4})                 # gid 4 is the one live on cam-01
    a = _layout(overlap_tolerance_s=5.0, vlm_arbiter=vlm, origin_cameras=["cam-01", "cam-02"], vlm_max_candidates=2)
    look_alike, other = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    for i in range(4):            # older vehicles that look like the newcomer on paper ...
        a.get_or_create_global_id("cam-01", i, 430.0 + 15 * i, embedding=look_alike, crops=[_crop()])
        a.mark_tracklet_inactive("cam-01", i, 432.0 + 15 * i)
    for k in range(4):            # ... and one in view on cam-01 right now that does not
        live = a.get_or_create_global_id("cam-01", 40, 500.0 + 0.1 * k, embedding=other, crops=[_crop()])
    assert live == 4
    r = [a.get_or_create_global_id("cam-02", 50, 502.0 + 0.1 * i, embedding=look_alike, crops=[_crop()])
         for i in range(4)]
    assert [g for g in r if g != -1][0] == 4


def test_gallery_is_offered_the_shot_with_the_plate_showing_over_a_more_confident_one():
    from types import SimpleNamespace
    from perception.modules.mtmc.core._tracklets import TrackletRecord
    tr = TrackletRecord(cam_id="cam-1", object_id=1)
    plain, with_plate = np.zeros((40, 80, 3), np.uint8), np.full((40, 80, 3), 9, np.uint8)
    tr.add_observation(1, 0.0, (0, 0, 80, 40), plain, 0.95)           # the most confident shot has no plate
    offers = []
    gallery = SimpleNamespace(maybe_submit=lambda *a: offers.append(a))
    mgr = SimpleNamespace(vehicle_gallery=gallery)
    MTMCManager._submit_gallery(mgr, "cam-1", tr, 5, 1, 1.0, None)
    assert offers[-1][2] is not tr.plate and offers[-1][3] == 0.95     # no plate yet: the best shot, by confidence
    tr.note_plate((10, 25, 20, 8), 0.6, 1.0, plain[:8, :20], with_plate)
    MTMCManager._submit_gallery(mgr, "cam-1", tr, 5, 1, 1.0, None)
    assert offers[-1][2] is with_plate and offers[-1][3] == pytest.approx(1.6)


def test_a_vehicle_seen_only_too_small_has_no_representative_crop_and_no_global_id():
    from perception.modules.mtmc.core import MTMCConfig, MTMCManager
    from perception.modules.mtmc.core._tracklets import CameraTracklets
    cfg = MTMCConfig(tracked_classes=("vehicle",))
    cams = CameraTracklets("cam-07", cfg)
    frame = np.full((720, 1280, 3), 80, np.uint8)
    # vehicle 1: 109 x 77 (the size of truck 43's crops on cam-07); vehicle 2: 300 x 200
    boxes = [(100.0, 100.0, 109.0, 77.0, "vehicle", 0.9, 1), (500.0, 300.0, 300.0, 200.0, "vehicle", 0.9, 2)]
    for i in range(3):
        cams.update(i, i * 0.1, [], frame, boxes)
    small, big = cams._tracklets[1], cams._tracklets[2]
    assert small.best_shot is None and small.crops == [] and small.small_seen == 3
    assert big.best_shot is not None and len(big.crops) == 3 and big.small_seen == 0
