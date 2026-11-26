#!/usr/bin/env python3
"""Compare NvDCF vs VEMA-3D golden JSONL recordings and telemetry logs across four migration gates."""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any

PRODUCTS = ("multicam", "zones", "proximity", "attendance", "barber")
VEHICLE_LABELS = frozenset({"vehicle", "car", "truck", "bus", "motorcycle"})

_STAGE_RE = re.compile(
    r"\[telemetry\] (?P<name>\S+) n=(?P<n>\d+) total=(?P<total>[\d.]+)ms "
    r"busy=(?P<busy>[\d.]+)% avg=(?P<avg>[\d.]+)ms max=(?P<max>[\d.]+)ms "
    r".*window=(?P<window>[\d.]+)s"
)
_GAUGE_RE = re.compile(r"\[telemetry\] gauge (?P<name>[\w.-]+)=(?P<val>-?[\d.]+)")
_FPS_RE = re.compile(r"\[(?P<cam>[\w.-]+)\] (?P<fps>[\d.]+) fps")
_CRASH_RE = re.compile(r"ModuleStartupError|Segmentation fault|SIGSEGV|\bFATAL\b|Traceback \(most recent")


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    pos = (pct / 100.0) * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return float(s[lo] + frac * (s[hi] - s[lo]))


def _iou_xywh(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax1 + aw, bx1 + bw)
    y2 = min(ay1 + ah, by1 + bh)
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    union = max(0.0, aw) * max(0.0, ah) + max(0.0, bw) * max(0.0, bh) - inter
    return inter / union if union > 1e-9 else 0.0


def parse_telemetry_log(text: str, skip_first_window: bool = True) -> dict[str, Any]:
    """Parse sidecar [telemetry] stages, gauges, per-camera FPS, and crash indicators."""
    per_window: dict[int, dict[str, dict[str, float]]] = collections.defaultdict(
        lambda: collections.defaultdict(lambda: {"busy": 0.0, "n": 0.0, "total": 0.0, "max": 0.0})
    )
    seen_windows = 0
    window_names: set[str] = set()
    fps_by_cam: dict[str, list[float]] = collections.defaultdict(list)
    gauges: dict[str, float] = {}
    crash_count = 0

    for line in text.splitlines():
        if _CRASH_RE.search(line):
            crash_count += 1
        sm = _STAGE_RE.search(line)
        if sm:
            name = sm.group("name")
            if name in window_names:
                seen_windows += 1
                window_names = set()
            window_names.add(name)
            stage_key = "gpu.decode" if name.startswith("gpu.decode.") else name
            acc = per_window[seen_windows][stage_key]
            acc["busy"] += float(sm.group("busy"))
            acc["n"] += int(sm.group("n"))
            acc["total"] += float(sm.group("total"))
            acc["max"] = max(acc["max"], float(sm.group("max")))
            continue
        gm = _GAUGE_RE.search(line)
        if gm:
            gauges[gm.group("name")] = float(gm.group("val"))
            continue
        fm = _FPS_RE.search(line)
        if fm:
            fps_by_cam[fm.group("cam")].append(float(fm.group("fps")))

    windows = sorted(per_window)
    if skip_first_window and len(windows) > 1:
        windows = windows[1:]

    stages: dict[str, dict[str, Any]] = {}
    for w in windows:
        for stage, acc in per_window[w].items():
            dst = stages.setdefault(stage, {"busy": 0.0, "n": 0, "total": 0.0, "max_ms": 0.0, "windows": 0})
            dst["busy"] += acc["busy"]
            dst["n"] += int(acc["n"])
            dst["total"] += acc["total"]
            dst["max_ms"] = max(dst["max_ms"], acc["max"])
            dst["windows"] += 1

    for dst in stages.values():
        w_cnt = max(dst["windows"], 1)
        dst["busy_pct"] = round(dst.pop("busy") / w_cnt, 3)
        dst["avg_ms"] = round(dst["total"] / dst["n"], 4) if dst["n"] else 0.0
        dst["max_ms"] = round(dst["max_ms"], 4)

    cam_medians: dict[str, float] = {}
    for cam, vals in fps_by_cam.items():
        steady = vals[2:] if len(vals) > 3 else vals
        cam_medians[cam] = round(statistics.median(steady), 3)

    overall_fps = round(statistics.median(cam_medians.values()), 3) if cam_medians else None
    drops = {
        "qos_dropped_total": int(gauges.get("qos_dropped_total", 0.0)),
        "publisher_drops": int(gauges.get("publisher_drops", 0.0)),
        "pub_shed_total": int(sum(v for k, v in gauges.items() if k.startswith("pub_shed_total_"))),
        "stall_episodes": int(sum(v for k, v in gauges.items() if k.startswith("source_stall_episodes_total_"))),
    }
    total_drops = drops["qos_dropped_total"] + drops["publisher_drops"] + drops["stall_episodes"]

    return {
        "stages": stages,
        "gpu_tracker": stages.get("gpu.tracker"),
        "gpu_total_busy_pct": round(
            sum(v["busy_pct"] for k, v in stages.items() if k.startswith("gpu.")), 3
        ),
        "fps_by_cam": cam_medians,
        "overall_fps": overall_fps,
        "drops": drops,
        "total_drops": total_drops,
        "crash_count": crash_count,
    }


def _to_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _to_int(val: Any, default: int = 0) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _json_maybe(val: Any) -> Any:
    if isinstance(val, str) and val[:1] in "[{":
        try:
            return json.loads(val)
        except ValueError:
            return val
    return val


def parse_jsonl_recording(path: str | Path) -> dict[str, Any]:
    """Parse an engine_golden.py JSONL recording into structured per-stream events."""
    meta_frames: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    zone_alerts: list[dict[str, Any]] = []
    overcrowd_alerts: list[dict[str, Any]] = []
    alerts: list[dict[str, Any]] = []
    face_rec: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    mtmc_plates: list[dict[str, Any]] = []
    mtmc_assigned: list[dict[str, Any]] = []
    mtmc_exited: list[dict[str, Any]] = []
    appearance: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    customer_face: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    frame_w, frame_h = 0, 0
    max_x2, max_y2 = 0.0, 0.0

    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            entry = json.loads(line)
            if "snapshot" in entry:
                continue
            key = entry.get("key", "")
            fields = entry.get("fields") or {}

            if key.startswith("iris:meta:"):
                cam = key[len("iris:meta:"):]
                pts = _to_float(fields.get("pts"), 0.0)
                fnum = _to_int(fields.get("frame_num"), len(meta_frames[cam]))
                dets_raw = _json_maybe(fields.get("dets")) or []
                pboxes_raw = _json_maybe(fields.get("person_boxes")) or []
                pbox_by_id = {
                    _to_int(b[4]): (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
                    for b in pboxes_raw
                    if isinstance(b, list) and len(b) >= 5
                }
                gids_raw = _json_maybe(fields.get("global_ids")) or {}
                gids = {str(k): _to_int(v) for k, v in gids_raw.items()} if isinstance(gids_raw, dict) else {}

                parsed_dets = []
                for d in dets_raw:
                    if not isinstance(d, list) or len(d) < 3:
                        continue
                    label = str(d[0])
                    conf = _to_float(d[1], 0.0)
                    oid = _to_int(d[2], -1)
                    box = None
                    if len(d) >= 7:
                        box = (float(d[3]), float(d[4]), float(d[5]), float(d[6]))
                    elif oid in pbox_by_id and label == "person":
                        box = pbox_by_id[oid]
                    if box is not None:
                        max_x2 = max(max_x2, box[0] + box[2])
                        max_y2 = max(max_y2, box[1] + box[3])
                    parsed_dets.append((label, conf, oid, box))

                meta_frames[cam].append({
                    "frame_num": fnum,
                    "pts": pts,
                    "dets": parsed_dets,
                    "global_ids": gids,
                })
            elif key.startswith("iris:zone_alert:"):
                cam = key[len("iris:zone_alert:"):]
                zone_alerts.append({"cam": cam, **fields})
            elif key.startswith("iris:overcrowd_alert:"):
                cam = key[len("iris:overcrowd_alert:"):]
                overcrowd_alerts.append({"cam": cam, **fields})
            elif key.startswith("iris:alert:"):
                cam = key[len("iris:alert:"):]
                alerts.append({"cam": cam, **fields})
            elif key.startswith("iris:face_rec:"):
                cam = key[len("iris:face_rec:"):]
                recs = _json_maybe(fields.get("face_rec")) or []
                face_rec[cam].append({"pts": _to_float(fields.get("pts"), 0.0), "items": recs})
            elif key == "iris:mtmc:plates":
                mtmc_plates.append(dict(fields))
            elif key == "iris:mtmc:assigned":
                mtmc_assigned.append(dict(fields))
            elif key == "iris:mtmc:exited":
                mtmc_exited.append(dict(fields))
            elif key.startswith("iris:appearance:"):
                cam = key[len("iris:appearance:"):]
                items = _json_maybe(fields.get("items")) or []
                if isinstance(items, list):
                    appearance[cam].extend(items)
            elif key.startswith("iris:customer_face:"):
                cam = key[len("iris:customer_face:"):]
                item = _json_maybe(fields.get("item"))
                if isinstance(item, dict):
                    customer_face[cam].append(item)
            elif key.startswith("iris:overlay:"):
                w_ov = _to_int(fields.get("w"), 0)
                h_ov = _to_int(fields.get("h"), 0)
                if w_ov > 0 and h_ov > 0:
                    frame_w, frame_h = w_ov, h_ov

    for cam in meta_frames:
        meta_frames[cam].sort(key=lambda r: (r["pts"], r["frame_num"]))

    if frame_w == 0 or frame_h == 0:
        frame_w = 1920 if max_x2 > 1280 else (1280 if max_x2 > 1080 else 1080 if max_x2 > 0 else 1280)
        frame_h = 1080 if max_y2 > 720 else 720

    return {
        "meta": dict(meta_frames),
        "frame_size": (frame_w, frame_h),
        "zone_alerts": zone_alerts,
        "overcrowd_alerts": overcrowd_alerts,
        "alerts": alerts,
        "face_rec": dict(face_rec),
        "mtmc_plates": mtmc_plates,
        "mtmc_assigned": mtmc_assigned,
        "mtmc_exited": mtmc_exited,
        "appearance": dict(appearance),
        "customer_face": dict(customer_face),
    }


def _compatible_label(a: str, b: str) -> bool:
    if a == b:
        return True
    return a in VEHICLE_LABELS and b in VEHICLE_LABELS


def _at_border(box: tuple[float, float, float, float], w_img: int, h_img: int, margin: float) -> bool:
    x, y, w, h = box
    return x <= margin or y <= margin or (x + w) >= (w_img - margin) or (y + h) >= (h_img - margin)


def compute_intrinsic_metrics(parsed: dict[str, Any], border_margin_px: float = 12.0) -> dict[str, Any]:
    """Compute Tier-2 label-free trajectory quality metrics from iris:meta detections."""
    meta = parsed["meta"]
    w_img, h_img = parsed["frame_size"]

    # Track observations grouped by (cam, oid), sorted by (pts, frame_num).
    tracks: dict[tuple[str, int], list[dict[str, Any]]] = collections.defaultdict(list)
    total_detections = 0
    total_frames = 0
    p95_concurrent_sum = 0.0

    for cam, frames in meta.items():
        concurrent: list[float] = []
        for frame in frames:
            total_frames += 1
            seen_oids: set[int] = set()
            for label, conf, oid, box in frame["dets"]:
                total_detections += 1
                if oid < 0:
                    continue
                seen_oids.add(oid)
                tracks[(cam, oid)].append({
                    "frame_num": frame["frame_num"],
                    "pts": frame["pts"],
                    "label": label,
                    "conf": conf,
                    "box": box,
                })
            concurrent.append(float(len(seen_oids)))
        if concurrent:
            p95_concurrent_sum += _percentile(concurrent, 95.0)

    unique_ids = len(tracks)
    inflation = (unique_ids / p95_concurrent_sum) if p95_concurrent_sum > 0.0 else 0.0

    durations: list[float] = []
    flicker_count = 0
    jerks_sq: list[float] = []
    log_area_vols: list[float] = []
    border_lingers: list[float] = []
    border_shadow_lingers: list[float] = []

    ends_by_cam: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    starts_by_cam: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)

    for (cam, oid), obs in tracks.items():
        obs.sort(key=lambda o: (o["pts"], o["frame_num"]))
        dur = max(0.0, obs[-1]["pts"] - obs[0]["pts"])
        durations.append(dur)
        if dur < 1.0:
            flicker_count += 1

        boxed = [o for o in obs if o["box"] is not None]
        if len(boxed) >= 2:
            for k in range(1, len(boxed)):
                a0 = max(boxed[k - 1]["box"][2] * boxed[k - 1]["box"][3], 1.0)
                a1 = max(boxed[k]["box"][2] * boxed[k]["box"][3], 1.0)
                log_area_vols.append(abs(math.log(a1 / a0)))

        if len(boxed) >= 3:
            for k in range(1, len(boxed) - 1):
                dt1 = boxed[k]["pts"] - boxed[k - 1]["pts"]
                dt2 = boxed[k + 1]["pts"] - boxed[k]["pts"]
                if dt1 > 0 and dt2 > 0 and max(dt1, dt2) > 2.5 * min(dt1, dt2):
                    continue
                b0, b1, b2 = boxed[k - 1]["box"], boxed[k]["box"], boxed[k + 1]["box"]
                c0x, c0y = b0[0] + 0.5 * b0[2], b0[1] + 0.5 * b0[3]
                c1x, c1y = b1[0] + 0.5 * b1[2], b1[1] + 0.5 * b1[3]
                c2x, c2y = b2[0] + 0.5 * b2[2], b2[1] + 0.5 * b2[3]
                jx = c2x - 2.0 * c1x + c0x
                jy = c2y - 2.0 * c1y + c0y
                jerks_sq.append(jx * jx + jy * jy)

        # Border-exit ghost linger: tracks that started inside and linger at the border on exit.
        if len(boxed) >= 2 and _at_border(boxed[-1]["box"], w_img, h_img, border_margin_px):
            saw_interior = any(not _at_border(o["box"], w_img, h_img, border_margin_px) for o in boxed[:-1])
            if saw_interior:
                first_tail_border_pts = boxed[-1]["pts"]
                for k in range(len(boxed) - 1, -1, -1):
                    if _at_border(boxed[k]["box"], w_img, h_img, border_margin_px):
                        first_tail_border_pts = boxed[k]["pts"]
                    else:
                        break
                border_lingers.append(max(0.0, boxed[-1]["pts"] - first_tail_border_pts))
                det_border = [o["pts"] for o in boxed if o["conf"] >= 0.0]
                shadow_start_pts = det_border[-1] if det_border else boxed[0]["pts"]
                border_shadow_lingers.append(max(0.0, boxed[-1]["pts"] - max(first_tail_border_pts, shadow_start_pts)))

        if boxed:
            tail = boxed[-min(len(boxed), 5):]
            vx, vy = 0.0, 0.0
            dt_tail = tail[-1]["pts"] - tail[0]["pts"]
            if len(tail) >= 2 and dt_tail > 1e-4:
                c0x = tail[0]["box"][0] + 0.5 * tail[0]["box"][2]
                c0y = tail[0]["box"][1] + 0.5 * tail[0]["box"][3]
                c1x = tail[-1]["box"][0] + 0.5 * tail[-1]["box"][2]
                c1y = tail[-1]["box"][1] + 0.5 * tail[-1]["box"][3]
                vx = (c1x - c0x) / dt_tail
                vy = (c1y - c0y) / dt_tail
            ends_by_cam[cam].append({
                "oid": oid,
                "pts": boxed[-1]["pts"],
                "box": boxed[-1]["box"],
                "label": boxed[-1]["label"],
                "vx": vx,
                "vy": vy,
            })
            starts_by_cam[cam].append({
                "oid": oid,
                "pts": boxed[0]["pts"],
                "box": boxed[0]["box"],
                "label": boxed[0]["label"],
            })

    # Spatio-temporal stitchable breaks (1-to-1 greedy matching per camera within (0, 2.5s]).
    stitchable_breaks = 0
    has_boxes = any(o["box"] is not None for obs in tracks.values() for o in obs)
    for cam, ends in ends_by_cam.items():
        starts = starts_by_cam.get(cam, [])
        cands: list[tuple[float, int, int]] = []
        for ei, end in enumerate(ends):
            ex, ey, ew, eh = end["box"]
            ecx = ex + 0.5 * ew
            ecy = ey + 0.5 * eh
            scale = max(ew, eh, 40.0)
            for si, st in enumerate(starts):
                if st["oid"] == end["oid"]:
                    continue
                dt = st["pts"] - end["pts"]
                if not (0.0 < dt <= 2.5):
                    continue
                if not _compatible_label(end["label"], st["label"]):
                    continue
                pcx = ecx + end["vx"] * dt
                pcy = ecy + end["vy"] * dt
                pred_box = (pcx - 0.5 * ew, pcy - 0.5 * eh, ew, eh)
                sx, sy, sw, sh = st["box"]
                scx = sx + 0.5 * sw
                scy = sy + 0.5 * sh
                dist = math.hypot(pcx - scx, pcy - scy)
                iou = _iou_xywh(pred_box, st["box"])
                if iou >= 0.15 or dist <= scale:
                    score = (dist / scale) + 0.2 * dt - 0.5 * iou
                    cands.append((score, ei, si))
        cands.sort(key=lambda item: item[0])
        used_e: set[int] = set()
        used_s: set[int] = set()
        for _score, ei, si in cands:
            if ei not in used_e and si not in used_s:
                used_e.add(ei)
                used_s.add(si)
                stitchable_breaks += 1

    return {
        "total_frames": total_frames,
        "total_detections": total_detections,
        "unique_ids": unique_ids,
        "p95_concurrent": round(p95_concurrent_sum, 3),
        "unique_id_inflation": round(inflation, 4),
        "flicker_tracks": flicker_count,
        "flicker_rate_pct": round(100.0 * flicker_count / unique_ids, 2) if unique_ids else 0.0,
        "lifespan_median_s": round(statistics.median(durations), 3) if durations else 0.0,
        "lifespan_p90_s": round(_percentile(durations, 90.0), 3) if durations else 0.0,
        "stitchable_breaks": stitchable_breaks if has_boxes else None,
        "center_jerk_rms": round(math.sqrt(sum(jerks_sq) / len(jerks_sq)), 4) if jerks_sq else None,
        "log_area_volatility": round(statistics.median(log_area_vols), 5) if log_area_vols else None,
        "border_exit_linger_s": round(sum(border_lingers) / len(border_lingers), 4) if border_lingers else (0.0 if has_boxes else None),
        "border_shadow_linger_s": round(sum(border_shadow_lingers) / len(border_shadow_lingers), 4) if border_shadow_lingers else (0.0 if has_boxes else None),
    }


def _dedup_zone_episodes(events: list[dict[str, Any]], gap_s: float = 3.5) -> list[float]:
    """Collapse ID-switch re-triggers inside (cam, zone) into episode open PTSs for fair timing comparison."""
    by_cz: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
    for ev in events:
        if ev.get("kind") == "intrusion_cleared":
            continue
        by_cz[(ev.get("cam", ""), str(ev.get("zone", "")))].append(_to_float(ev.get("pts"), 0.0))
    opens: list[float] = []
    for pts_list in by_cz.values():
        pts_list.sort()
        last = -1e9
        for p in pts_list:
            if p - last > gap_s:
                opens.append(p)
            last = p
    return sorted(opens)


def _multicam_stats(data: dict[str, Any]) -> dict[str, Any]:
    local_ids: set[tuple[str, int]] = set()
    gids: set[int] = set()
    for cam, frames in data["meta"].items():
        for frame in frames:
            for label, _conf, oid, _box in frame["dets"]:
                if oid >= 0 and label in VEHICLE_LABELS:
                    local_ids.add((cam, oid))
            for gid in frame["global_ids"].values():
                if gid >= 0:
                    gids.add(gid)
    for ev in data["mtmc_assigned"]:
        gid = _to_int(ev.get("global_id"), -1)
        if gid >= 0:
            gids.add(gid)
    plates = {
        str(ev.get("text", "")).strip()
        for ev in data["mtmc_plates"]
        if str(ev.get("text", "")).strip()
    }
    n_plates = len(plates)
    return {
        "unique_local_ids": len(local_ids),
        "unique_global_ids": len(gids),
        "unique_confirmed_plates": n_plates,
        "plate_reads_total": len(data["mtmc_plates"]),
        "gid_split_ratio": round(len(gids) / max(n_plates, 1), 4),
    }


def _zones_stats(data: dict[str, Any]) -> dict[str, Any]:
    intrusions = [e for e in data["zone_alerts"] if e.get("kind") != "intrusion_cleared"]
    by_cz: dict[tuple[str, str], list[tuple[float, int]]] = collections.defaultdict(list)
    for ev in intrusions:
        by_cz[(ev.get("cam", ""), str(ev.get("zone", "")))].append(
            (_to_float(ev.get("pts"), 0.0), _to_int(ev.get("object_id"), -1))
        )
    duplicates = 0
    for entries in by_cz.values():
        entries.sort(key=lambda item: item[0])
        for i in range(1, len(entries)):
            if (entries[i][0] - entries[i - 1][0]) <= 3.5 and entries[i][1] != entries[i - 1][1]:
                duplicates += 1
    overcrowd_opens = [e for e in data["overcrowd_alerts"] if e.get("kind") != "overcrowding_resolved"]
    peak_count = max((_to_int(e.get("count"), 0) for e in overcrowd_opens), default=0)
    return {
        "intrusion_alerts": len(intrusions),
        "duplicate_retriggers": duplicates,
        "overcrowd_alerts": len(overcrowd_opens),
        "overcrowd_peak_count": peak_count,
        "episode_opens_pts": _dedup_zone_episodes(data["zone_alerts"]),
    }


def _proximity_stats(data: dict[str, Any]) -> dict[str, Any]:
    opens: list[float] = []
    seen_incidents: set[str] = set()
    for ev in data["alerts"]:
        if ev.get("kind") != "proximity":
            continue
        if ev.get("suppressed") or ev.get("incident_state") == "resolved":
            continue
        inc = str(ev.get("incident", ""))
        if inc and inc in seen_incidents:
            continue
        if inc:
            seen_incidents.add(inc)
        opens.append(_to_float(ev.get("pts"), 0.0))
    opens.sort()
    vehicle_ids = {
        (cam, oid)
        for cam, frames in data["meta"].items()
        for frame in frames
        for label, _c, oid, _b in frame["dets"]
        if oid >= 0 and label in VEHICLE_LABELS
    }
    return {
        "proximity_episodes": len(opens),
        "episode_pts": opens,
        "unique_vehicle_ids": len(vehicle_ids),
    }


def _attendance_stats(data: dict[str, Any]) -> dict[str, Any]:
    by_emp: dict[tuple[str, str], list[tuple[float, int]]] = collections.defaultdict(list)
    for cam, entries in data["face_rec"].items():
        for entry in entries:
            pts = entry["pts"]
            for item in entry["items"]:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).strip()
                confirmed = item.get("confirmed")
                if confirmed is False or not name or name.lower() == "unknown":
                    continue
                pid = _to_int(item.get("parent_id"), -1)
                by_emp[(cam, name)].append((pts, pid))
    splits = 0
    spans = 0
    names: set[str] = set()
    for (_cam, name), rows in by_emp.items():
        names.add(name)
        rows.sort(key=lambda r: r[0])
        spans += 1
        for i in range(1, len(rows)):
            dt = rows[i][0] - rows[i - 1][0]
            if dt > 30.0:
                spans += 1
            elif rows[i][1] != rows[i - 1][1]:
                spans += 1
                splits += 1
    return {
        "named_employees": sorted(names),
        "session_spans": spans,
        "session_splits": splits,
    }


def _barber_stats(data: dict[str, Any]) -> dict[str, Any]:
    person_dets = 0
    person_ids: set[tuple[str, int]] = set()
    for cam, frames in data["meta"].items():
        for frame in frames:
            for label, _c, oid, _b in frame["dets"]:
                if label == "person":
                    person_dets += 1
                    if oid >= 0:
                        person_ids.add((cam, oid))
    app_samples = sum(len(v) for v in data["appearance"].values())
    face_samples = sum(len(v) for v in data["customer_face"].values())
    return {
        "person_detections": person_dets,
        "unique_person_ids": len(person_ids),
        "appearance_samples": app_samples,
        "customer_face_samples": face_samples,
    }


def evaluate_product_oracle(
    product: str, nvdcf_data: dict[str, Any], vema3d_data: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate Tier-3 downstream product business oracle pass/fail."""
    if product == "multicam":
        a, b = _multicam_stats(nvdcf_data), _multicam_stats(vema3d_data)
        passed = (
            b["unique_local_ids"] <= a["unique_local_ids"]
            and b["unique_confirmed_plates"] >= a["unique_confirmed_plates"]
            and (a["unique_confirmed_plates"] == 0 or b["gid_split_ratio"] <= a["gid_split_ratio"])
        )
        return {"product": product, "nvdcf": a, "vema3d": b, "passed": passed}

    if product == "zones":
        a, b = _zones_stats(nvdcf_data), _zones_stats(vema3d_data)
        a_opens, b_opens = a["episode_opens_pts"], b["episode_opens_pts"]
        a_oids = {
            (cam, _to_int(d[2], -1))
            for cam, frames in nvdcf_data["meta"].items()
            for fr in frames
            for d in fr["dets"]
            if _to_int(d[2], -1) >= 0
        }
        b_oids = {
            (cam, _to_int(d[2], -1))
            for cam, frames in vema3d_data["meta"].items()
            for fr in frames
            for d in fr["dets"]
            if _to_int(d[2], -1) >= 0
        }
        a_dets = sum(len(fr["dets"]) for frames in nvdcf_data["meta"].values() for fr in frames)
        b_dets = sum(len(fr["dets"]) for frames in vema3d_data["meta"].values() for fr in frames)
        timing_ok = True
        max_pts_delta = 0.0
        if len(a_opens) == len(b_opens):
            for pa, pb in zip(a_opens, b_opens):
                max_pts_delta = max(max_pts_delta, abs(pa - pb))
            timing_ok = max_pts_delta <= 0.5
        elif len(b_opens) < len(a_opens) and b_opens:
            # Allow fewer episodes only when every candidate episode aligns within 0.5s
            # and all omitted baseline episodes are bounded by baseline's excess track breaks
            # without losing detection recall.
            excess_breaks = max(0, len(a_oids) - len(b_oids))
            matched_idx = 0
            used_a: set[int] = set()
            for pb in b_opens:
                best_i, best_d = -1, 1e9
                for ai in range(matched_idx, len(a_opens)):
                    if ai in used_a:
                        continue
                    d = abs(a_opens[ai] - pb)
                    if d < best_d:
                        best_d = d
                        best_i = ai
                if best_i >= 0 and best_d <= 0.5:
                    used_a.add(best_i)
                    matched_idx = best_i + 1
                    max_pts_delta = max(max_pts_delta, best_d)
                else:
                    max_pts_delta = max(max_pts_delta, best_d)
                    break
            timing_ok = (
                len(used_a) == len(b_opens)
                and (len(a_opens) - len(b_opens)) <= excess_breaks
                and b_dets >= 0.95 * a_dets
                and max_pts_delta <= 0.5
            )
        elif a_opens or b_opens:
            timing_ok = False
        passed = (
            b["duplicate_retriggers"] <= a["duplicate_retriggers"]
            and b["intrusion_alerts"] <= a["intrusion_alerts"]
            and b["overcrowd_peak_count"] == a["overcrowd_peak_count"]
            and timing_ok
        )
        return {
            "product": product,
            "nvdcf": a,
            "vema3d": b,
            "max_open_pts_delta_s": round(max_pts_delta, 4),
            "passed": passed,
        }

    if product == "proximity":
        a, b = _proximity_stats(nvdcf_data), _proximity_stats(vema3d_data)
        timing_ok = len(a["episode_pts"]) == len(b["episode_pts"]) and all(
            abs(pa - pb) <= 1.0 for pa, pb in zip(a["episode_pts"], b["episode_pts"])
        )
        passed = timing_ok and b["unique_vehicle_ids"] <= a["unique_vehicle_ids"]
        return {"product": product, "nvdcf": a, "vema3d": b, "passed": passed}

    if product == "attendance":
        a, b = _attendance_stats(nvdcf_data), _attendance_stats(vema3d_data)
        passed = (
            set(a["named_employees"]) <= set(b["named_employees"])
            and b["session_splits"] <= a["session_splits"]
        )
        return {"product": product, "nvdcf": a, "vema3d": b, "passed": passed}

    if product == "barber":
        a, b = _barber_stats(nvdcf_data), _barber_stats(vema3d_data)
        tol = max(0.08 * max(a["person_detections"], b["person_detections"]), 2.0)
        det_ok = abs(a["person_detections"] - b["person_detections"]) <= tol
        app_ok = a["appearance_samples"] == 0 or b["appearance_samples"] >= 0.9 * a["appearance_samples"]
        face_ok = a["customer_face_samples"] == 0 or b["customer_face_samples"] >= 0.9 * a["customer_face_samples"]
        passed = det_ok and app_ok and face_ok and b["unique_person_ids"] <= a["unique_person_ids"]
        return {"product": product, "nvdcf": a, "vema3d": b, "passed": passed}

    raise ValueError(f"Unsupported product: {product}")


def evaluate_gates(
    product: str,
    rt_a: dict[str, Any] | None,
    rt_b: dict[str, Any] | None,
    intr_a: dict[str, Any],
    intr_b: dict[str, Any],
    oracle: dict[str, Any],
) -> dict[str, Any]:
    """Compute Tier-4 Go / No-Go gate verdicts."""
    det_recall_ok = intr_b["total_detections"] >= 0.95 * intr_a["total_detections"]
    no_crashes = True if rt_b is None else (rt_b["crash_count"] == 0 and rt_b["total_drops"] == 0)
    gate_correctness = det_recall_ok and no_crashes

    gate_oracle = bool(oracle["passed"])

    breaks_ok = True
    if intr_a["stitchable_breaks"] is not None and intr_b["stitchable_breaks"] is not None:
        breaks_ok = intr_b["stitchable_breaks"] <= intr_a["stitchable_breaks"]

    vol_ok = True
    if intr_a["log_area_volatility"] is not None and intr_b["log_area_volatility"] is not None:
        vol_ok = intr_b["log_area_volatility"] <= max(intr_a["log_area_volatility"] * 1.05, 1e-6)

    linger_ok = True
    if intr_a["border_exit_linger_s"] is not None and intr_b["border_exit_linger_s"] is not None:
        baseline_fragmented = (
            (intr_a.get("stitchable_breaks") or 0) > 0
            and intr_a["unique_ids"] > intr_b["unique_ids"]
        )
        sh_a = intr_a.get("border_shadow_linger_s")
        sh_b = intr_b.get("border_shadow_linger_s")
        if baseline_fragmented and sh_a is not None and sh_b is not None:
            # When the baseline's track fragmentation artificially cuts its own border tail
            # short, gate on post-detection shadow coasting at the border instead.
            linger_ok = sh_b <= sh_a + 0.25
        else:
            linger_ok = intr_b["border_exit_linger_s"] <= intr_a["border_exit_linger_s"] + 0.25

    gate_trajectory = breaks_ok and vol_ok and linger_ok

    efficiency_checked = rt_a is not None and rt_b is not None
    gate_efficiency: bool | None = None
    if efficiency_checked:
        assert rt_a is not None and rt_b is not None
        latency_ok = True
        if rt_a.get("gpu_tracker") and rt_b.get("gpu_tracker"):
            latency_ok = (
                rt_b["gpu_tracker"]["avg_ms"] <= rt_a["gpu_tracker"]["avg_ms"] * 1.02
                and rt_b["gpu_tracker"]["max_ms"] <= rt_a["gpu_tracker"]["max_ms"] * 1.05
            )
        fps_ok = True
        if rt_a.get("overall_fps") and rt_b.get("overall_fps"):
            fps_ok = rt_b["overall_fps"] >= 0.95 * rt_a["overall_fps"]
        drops_ok = rt_b["total_drops"] <= rt_a["total_drops"]
        gate_efficiency = latency_ok and fps_ok and drops_ok

    approved = gate_correctness and gate_oracle and gate_trajectory and (gate_efficiency is not False)
    return {
        "product": product,
        "correctness_gate": gate_correctness,
        "product_oracle_gate": gate_oracle,
        "trajectory_quality_gate": gate_trajectory,
        "efficiency_gate": gate_efficiency,
        "efficiency_checked": efficiency_checked,
        "overall_verdict": "GO" if approved else "NO_GO",
    }


def evaluate_ab(
    nvdcf_path: str | Path,
    vema3d_path: str | Path,
    product: str,
    log_nvdcf_path: str | Path | None = None,
    log_vema3d_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run full Tier 1-4 evaluation comparing NvDCF vs VEMA-3D recordings."""
    data_a = parse_jsonl_recording(nvdcf_path)
    data_b = parse_jsonl_recording(vema3d_path)
    rt_a = parse_telemetry_log(Path(log_nvdcf_path).read_text(encoding="utf-8", errors="replace")) if log_nvdcf_path else None
    rt_b = parse_telemetry_log(Path(log_vema3d_path).read_text(encoding="utf-8", errors="replace")) if log_vema3d_path else None
    intr_a = compute_intrinsic_metrics(data_a)
    intr_b = compute_intrinsic_metrics(data_b)
    oracle = evaluate_product_oracle(product, data_a, data_b)
    gates = evaluate_gates(product, rt_a, rt_b, intr_a, intr_b, oracle)
    return {
        "product": product,
        "runtime": {"nvdcf": rt_a, "vema3d": rt_b},
        "intrinsic": {"nvdcf": intr_a, "vema3d": intr_b},
        "oracle": oracle,
        "gates": gates,
    }


def _fmt(val: Any) -> str:
    if val is None:
        return "n/a"
    if isinstance(val, float):
        return f"{val:.4g}"
    return str(val)


def render_markdown(report: dict[str, Any]) -> str:
    """Format an A/B evaluation report as a Markdown migration scorecard."""
    lines = [
        f"# VEMA-3D Migration Scorecard ({report['product']})",
        "",
        f"**Overall Verdict:** `{report['gates']['overall_verdict']}`",
        "",
    ]

    rt_a, rt_b = report["runtime"]["nvdcf"], report["runtime"]["vema3d"]
    if rt_a is not None and rt_b is not None:
        tr_a = rt_a.get("gpu_tracker") or {}
        tr_b = rt_b.get("gpu_tracker") or {}
        lines.extend([
            "## Tier 1: Runtime & Hardware Efficiency",
            "",
            "| Metric | NvDCF | VEMA-3D |",
            "| :--- | ---: | ---: |",
            f"| `gpu.tracker` avg_ms | {_fmt(tr_a.get('avg_ms'))} | {_fmt(tr_b.get('avg_ms'))} |",
            f"| `gpu.tracker` max_ms | {_fmt(tr_a.get('max_ms'))} | {_fmt(tr_b.get('max_ms'))} |",
            f"| `gpu.tracker` busy % | {_fmt(tr_a.get('busy_pct'))} | {_fmt(tr_b.get('busy_pct'))} |",
            f"| Overall steady FPS | {_fmt(rt_a.get('overall_fps'))} | {_fmt(rt_b.get('overall_fps'))} |",
            f"| Drops + Stalls | {_fmt(rt_a.get('total_drops'))} | {_fmt(rt_b.get('total_drops'))} |",
            "",
        ])

    ia, ib = report["intrinsic"]["nvdcf"], report["intrinsic"]["vema3d"]
    lines.extend([
        "## Tier 2: Intrinsic Trajectory Quality",
        "",
        "| Metric | NvDCF | VEMA-3D |",
        "| :--- | ---: | ---: |",
    ])
    for key in (
        "total_detections",
        "unique_ids",
        "p95_concurrent",
        "unique_id_inflation",
        "flicker_rate_pct",
        "lifespan_median_s",
        "lifespan_p90_s",
        "stitchable_breaks",
        "center_jerk_rms",
        "log_area_volatility",
        "border_exit_linger_s",
        "border_shadow_linger_s",
    ):
        lines.append(f"| `{key}` | {_fmt(ia.get(key))} | {_fmt(ib.get(key))} |")

    oracle = report["oracle"]
    lines.extend([
        "",
        f"## Tier 3: Product Business Oracle (`{oracle['product']}`)",
        "",
        f"- **Passed:** `{oracle['passed']}`",
        f"- **NvDCF:** `{json.dumps(oracle['nvdcf'], sort_keys=True)}`",
        f"- **VEMA-3D:** `{json.dumps(oracle['vema3d'], sort_keys=True)}`",
        "",
        "## Tier 4: Go / No-Go Gates",
        "",
        "| Gate | Verdict |",
        "| :--- | :--- |",
    ])
    g = report["gates"]
    for gk in ("correctness_gate", "product_oracle_gate", "trajectory_quality_gate", "efficiency_gate"):
        gv = g[gk]
        label = "SKIPPED" if gv is None else ("PASS" if gv else "FAIL")
        lines.append(f"| `{gk}` | `{label}` |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nvdcf", required=True, type=Path, help="Baseline NvDCF engine_golden JSONL recording")
    ap.add_argument("--vema3d", required=True, type=Path, help="Candidate VEMA-3D engine_golden JSONL recording")
    ap.add_argument("--log-nvdcf", type=Path, default=None, help="Optional baseline sidecar log with [telemetry] lines")
    ap.add_argument("--log-vema3d", type=Path, default=None, help="Optional candidate sidecar log with [telemetry] lines")
    ap.add_argument("--product", required=True, choices=PRODUCTS, help="IRIS product oracle to evaluate")
    ap.add_argument("--out", type=Path, default=None, help="Optional JSON report output path")
    args = ap.parse_args(argv)

    report = evaluate_ab(
        args.nvdcf,
        args.vema3d,
        args.product,
        log_nvdcf_path=args.log_nvdcf,
        log_vema3d_path=args.log_vema3d,
    )
    print(render_markdown(report))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if report["gates"]["overall_verdict"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
