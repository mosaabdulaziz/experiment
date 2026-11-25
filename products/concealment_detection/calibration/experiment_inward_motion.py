"""Read-only experiment: require inward hand-to-body motion (`r_max_prior - r_now >= inward_min_torsos`) on the same hand.

Evaluates:
1. Idle/dwell `in_at_body_pose` false-trigger rate on `gettyimages1` and `gettyimages2`.
2. Default-config and tuned-config alert behavior on `gettyimages1` and `gettyimages2` for both `yolo11m-pose` and `yolo26x-pose`.
3. Staged **development** recall + false-alarm rate for each inward threshold. The staged test
   split, `PoseLift` real-world pairs and the negative holdout are never loaded here: an
   operating point must be frozen by the calibration runner before any of them is reported.

Does NOT modify any production file.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

CAL_DIR = Path(__file__).resolve().parent
PRODUCT_DIR = CAL_DIR.parent
for p in (str(PRODUCT_DIR), str(CAL_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from calibration.calibrate import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    _preload_negative_split,
    _preload_positive_split,
    _stratified_sample,
    evaluate_settings_on_splits,
)
from calibration.dataset_utils import discover_retails_splits, partition_staged_pairs  # noqa: E402
from calibration.render_rgb_demo import (  # noqa: E402
    OUT_DIR,
    VIDEOS,
    _relabel_handbag_to_basket,
)
import concealment.features as feat_mod  # noqa: E402
import concealment.state_machine as sm_mod  # noqa: E402
from concealment.features import (  # noqa: E402
    _hand_kinematics,
    find_containing_zone,
    ground_anchor,
    reach_alignment,
    torso_geometry,
)
from concealment.state_machine import ConcealmentMonitor  # noqa: E402
from concealment.types import (  # noqa: E402
    CameraZones,
    DetectorSettings,
    L_SHOULDER,
    L_WRIST,
    R_SHOULDER,
    R_WRIST,
    TrackFeatures,
    TrackObservation,
)
from config import load_config  # noqa: E402


def make_inward_feature_extractor(inward_min_torsos: float):
    """Build a drop-in replacement for `extract_track_features` requiring same-hand inward motion to body."""

    def _extract(
        obs: TrackObservation,
        history: list[tuple[float, TrackObservation, float]],
        camera_zones: CameraZones,
        shape: tuple[int, int],
        settings: DetectorSettings,
        active_hand: str | None = None,
    ) -> TrackFeatures:
        pts = history[-1][0] if history else 0.0
        torso_center, torso_length, _ = torso_geometry(obs)
        gp = ground_anchor(obs, shape, conf_threshold=settings.keypoint_conf_threshold)
        zone = find_containing_zone(gp, camera_zones)
        in_shelf_zone = zone is not None
        zone_id = zone.id if zone is not None else None

        l_info = _hand_kinematics(obs, "left", torso_center, torso_length, settings)
        r_info = _hand_kinematics(obs, "right", torso_center, torso_length, settings)

        left_elbow_deg = l_info[0]
        right_elbow_deg = r_info[0]
        min_elbow_deg = min(left_elbow_deg, right_elbow_deg)

        def _side_motion(side: str, cur_wr: tuple[float, float], base_in_at_body: bool) -> tuple[float, bool, bool]:
            wr_idx = L_WRIST if side == "left" else R_WRIST
            sh_idx = L_SHOULDER if side == "left" else R_SHOULDER
            cur_rad = math.hypot(cur_wr[0] - torso_center[0], cur_wr[1] - torso_center[1]) / max(torso_length, 1.0)
            if not history:
                reach_ok = cur_rad >= (0.65 + settings.reach_excursion_torsos)
                return (cur_rad, reach_ok, False)

            min_radial = float("inf")
            max_radial = 0.0
            max_prior_radial = 0.0
            best_delta = (0.0, 0.0)

            all_candidates = list(history) + [(pts, obs, torso_length)]
            lookback_s = max(settings.reach_window_s, 4.0)
            for idx_h, (h_pts, h_obs, h_tlen) in enumerate(all_candidates):
                if pts > 0.0 and (pts - h_pts) > lookback_s + 1e-6:
                    continue
                if len(h_obs.kps) <= wr_idx:
                    continue
                wx, wy, wc = h_obs.kps[wr_idx]
                h_tc, h_tl, _ = torso_geometry(h_obs)
                tl = max(h_tlen if h_tlen > 0 else h_tl, 1.0)
                rad = math.hypot(wx - h_tc[0], wy - h_tc[1]) / tl
                if (pts - h_pts) <= settings.reach_window_s + 1e-6:
                    if rad < min_radial:
                        min_radial = rad
                    if rad > max_radial:
                        max_radial = rad
                        sx, sy, _ = h_obs.kps[sh_idx]
                        best_delta = (wx - sx, wy - sy)
                # Prior confident frames strictly before current frame for inward motion check
                if idx_h < len(all_candidates) - 1 and (pts - h_pts) >= 0.06 and wc >= settings.keypoint_conf_threshold:
                    if rad > max_prior_radial:
                        max_prior_radial = rad

            if min_radial == float("inf"):
                return (0.0, False, False)

            excursion = max(0.0, max_radial - min_radial)
            align = reach_alignment(best_delta, zone)
            reach_ok = (excursion >= settings.reach_excursion_torsos) and (align >= -0.25)

            inward_drop = max(0.0, max_prior_radial - cur_rad)
            inward_ok = base_in_at_body and (inward_drop >= inward_min_torsos)
            return (excursion, reach_ok, inward_ok)

        l_exc, l_reach, l_at_body = _side_motion("left", l_info[1], l_info[7])
        r_exc, r_reach, r_at_body = _side_motion("right", r_info[1], r_info[7])

        # Choose active hand: honor locked active_hand, else prefer hand that reached + moved inward, then reaching hand
        if active_hand in ("left", "right"):
            pass
        elif r_at_body and not l_at_body:
            active_hand = "right"
        elif l_at_body and not r_at_body:
            active_hand = "left"
        elif r_reach and not l_reach:
            active_hand = "right"
        elif l_reach and not r_reach:
            active_hand = "left"
        elif (r_exc - r_info[5]) >= (l_exc - l_info[5]):
            active_hand = "right"
        else:
            active_hand = "left"

        if active_hand == "right":
            act = r_info
            act_exc = r_exc
            act_reach = r_reach
            act_in_at_body = r_at_body
        else:
            act = l_info
            act_exc = l_exc
            act_reach = l_reach
            act_in_at_body = l_at_body

        wr_idx = R_WRIST if active_hand == "right" else L_WRIST

        confs: list[float] = []
        for h_pts, h_obs, _ in history:
            if pts > 0.0 and (pts - h_pts) > settings.baseline_window_s + 1e-6:
                continue
            if len(h_obs.kps) > wr_idx:
                c = float(h_obs.kps[wr_idx][2])
                if c >= settings.keypoint_conf_threshold:
                    confs.append(c)
        if confs:
            s_confs = sorted(confs)
            mid = len(s_confs) // 2
            wrist_conf_baseline = (
                s_confs[mid]
                if len(s_confs) % 2 == 1
                else 0.5 * (s_confs[mid - 1] + s_confs[mid])
            )
        else:
            wrist_conf_baseline = max(act[2], settings.keypoint_conf_threshold)

        active_wrist_conf = act[2]
        wrist_hidden_now = (
            active_wrist_conf < settings.keypoint_conf_threshold
            or (
                wrist_conf_baseline >= settings.keypoint_conf_threshold
                and active_wrist_conf <= settings.hidden_conf_drop_fraction * wrist_conf_baseline
            )
        )

        return TrackFeatures(
            track_id=obs.track_id,
            pts=pts,
            torso_center=torso_center,
            torso_length=torso_length,
            ground_point_norm=gp,
            zone_id=zone_id,
            in_shelf_zone=in_shelf_zone,
            left_elbow_deg=left_elbow_deg,
            right_elbow_deg=right_elbow_deg,
            min_elbow_deg=min_elbow_deg,
            active_hand=active_hand,
            wrist_to_body_torsos=act[5],
            body_zone_name=act[3],
            target_kind=act[4],
            elbow_flexed_or_medial=act[6],
            in_at_body_pose=act_in_at_body,
            wrist_excursion_torsos=act_exc,
            reach_detected=act_reach,
            active_wrist_conf=active_wrist_conf,
            wrist_conf_baseline=wrist_conf_baseline,
            wrist_hidden_now=wrist_hidden_now,
        )

    return _extract


def eval_rgb_videos(cfg) -> list[dict]:
    import acl

    results = []
    for model_tag in ("yolo11m", "yolo26x"):
        for video in VIDEOS:
            cache = OUT_DIR / (
                f"cache_{video['key']}_meta.json"
                if model_tag == "yolo11m"
                else f"cache_{model_tag}_{video['key']}_meta.json"
            )
            data = json.loads(cache.read_text(encoding="utf-8"))
            entries = data["entries"]
            # Check both raw entries and basket-relabelled entries for video 1
            for variant_name, use_entries in [
                ("raw_default", entries),
                (
                    "basket_relabel_default",
                    _relabel_handbag_to_basket(entries) if video["tuned_relabel_handbag_to_basket"] else entries,
                ),
            ]:
                if variant_name == "basket_relabel_default" and not video["tuned_relabel_handbag_to_basket"]:
                    continue
                zones = CameraZones(camera=video["camera"], zones=(video["zone"],))
                mon = ConcealmentMonitor(camera=video["camera"], settings=cfg.detector, camera_zones=zones)
                events = []
                idle_frames = 0
                idle_at_body = 0
                for e in use_entries:
                    pf = acl.parse_pose_frame(camera=video["camera"], entry=e)
                    res = mon.update(pf)
                    events.extend(res.candidates)
                    for obs in pf.tracks:
                        if obs.track_id == 1:
                            sm = mon._machines.get(1)
                            hist = list(mon._histories.get(1, ()))
                            feat = sm_mod.extract_track_features(obs, hist, zones, pf.shape, cfg.detector)
                            if sm and sm.stage in ("IDLE", "DWELL"):
                                idle_frames += 1
                                if feat.in_at_body_pose:
                                    idle_at_body += 1
                results.append(
                    {
                        "video": video["key"],
                        "model": model_tag,
                        "variant": variant_name,
                        "idle_at_body_frac": round(idle_at_body / max(idle_frames, 1), 3),
                        "alerts": [
                            (ev.track_id, round(ev.pts, 2), ev.target_kind, ev.body_zone_name, ev.tier)
                            for ev in events
                        ],
                    }
                )
    return results


def main() -> None:
    cfg = load_config()
    # Each module keeps its own original: restoring one from the other would silently un-patch
    # a caller that swapped only one of the two.
    orig_feat = feat_mod.extract_track_features
    orig_sm = sm_mod.extract_track_features

    try:
        print("Preloading the frozen staged development split ...")
        splits = discover_retails_splits(DEFAULT_DATA_ROOT)
        staged_dev, _staged_test = partition_staged_pairs(splits.staged_pairs)
        staged_pos = _preload_positive_split(staged_dev)
        staged_neg = _preload_negative_split(_stratified_sample(splits.train_tune_files, 36))

        all_rows = []
        for label, fn in [
            ("baseline_current", orig_feat),
            ("inward_0.08T", make_inward_feature_extractor(0.08)),
            ("inward_0.12T", make_inward_feature_extractor(0.12)),
            ("inward_0.15T", make_inward_feature_extractor(0.15)),
            ("inward_0.20T", make_inward_feature_extractor(0.20)),
        ]:
            feat_mod.extract_track_features = fn
            sm_mod.extract_track_features = fn

            # Selection happens on development data only. Staged test, PoseLift and the negative
            # holdout stay untouched here; the calibration runner reports them after freezing a point.
            st_metrics = evaluate_settings_on_splits(cfg.detector, cfg.camera_zones, staged_pos, staged_neg)
            rgb_res = eval_rgb_videos(cfg)
            all_rows.append({"variant": label, "staged_dev": st_metrics, "rgb": rgb_res})

            print(f"\n=== {label} ===")
            print(
                f"  Staged dev Recall: {st_metrics['recall_all']*100:.1f}% "
                f"(single: {st_metrics['recall_single_person']*100:.1f}%) "
                f"| dev FA/h: {st_metrics['fa_per_span_hour']:.2f} "
                f"(warn: {st_metrics['warning_fa_per_span_hour']:.2f})"
            )
            for r in rgb_res:
                print(
                    f"  [{r['video']} | {r['model']} | {r['variant']}] "
                    f"idle_at_body={r['idle_at_body_frac']:.3f} alerts={r['alerts']}"
                )

        (OUT_DIR / "inward_motion_experiment.json").write_text(
            json.dumps(all_rows, indent=2), encoding="utf-8"
        )
    finally:
        feat_mod.extract_track_features = orig_feat
        sm_mod.extract_track_features = orig_sm


if __name__ == "__main__":
    main()
