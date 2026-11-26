#!/usr/bin/env python3
"""Render a synchronized Before/After side-by-side tracking video on IRIS footage.

Two workflows:
  1. `render`: Render immediately from an input video + two golden JSONL recordings
     (e.g. `/tmp/barber_nvdcf.jsonl` and `/tmp/barber_vema2.jsonl`).
  2. `run`: End-to-end server runner — stages a golden clip or custom `.mp4` slice,
     runs both DeepStream tracker arms (`golden-<product>.yaml` vs `-vema2.yaml`) via
     `tools/engine_golden_run.sh`, prints the Tier 1-4 scorecard, and renders the
     side-by-side MP4 for visual sign-off.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import colorsys
import json
import math
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class BoxObs:
    """Single tracked object detection or shadow coasting prediction on one frame."""

    oid: int
    label: str
    conf: float
    left: float
    top: float
    width: float
    height: float
    global_id: int | None = None
    vehicle_state: str | None = None

    @property
    def is_shadow(self) -> bool:
        return self.conf < 0.0

    @property
    def center(self) -> tuple[float, float]:
        return (self.left + 0.5 * self.width, self.top + 0.5 * self.height)


@dataclass
class OverlayLine:
    """Proximity distance line recorded in iris:overlay."""

    p1: tuple[float, float]
    p2: tuple[float, float]
    label: str
    style: str = "close"


@dataclass
class FrameTracks:
    """All tracked objects and optional overlay lines at one recorded frame timestamp."""

    pts: float
    frame_num: int
    boxes: list[BoxObs] = field(default_factory=list)
    lines: list[OverlayLine] = field(default_factory=list)


@dataclass
class TrackTimeline:
    """Sorted timeline for a single camera extracted from an engine_golden JSONL file."""

    cam: str
    frames: list[FrameTracks]
    mux_size: tuple[int, int]

    @property
    def pts_list(self) -> list[float]:
        return [f.pts for f in self.frames]


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


def _load_yaml_chain(path: Path) -> dict[str, Any]:
    """Load a sidecar YAML file resolving relative `extends:` without importing pyds."""
    import yaml

    visited: set[Path] = set()

    def _merge(base: dict[str, Any], ov: dict[str, Any]) -> dict[str, Any]:
        out = dict(base)
        for k, v in ov.items():
            if k == "extends":
                continue
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = _merge(out[k], v)
            else:
                out[k] = v
        return out

    def _load(p: Path) -> dict[str, Any]:
        rp = p.resolve()
        if rp in visited:
            return {}
        visited.add(rp)
        data = yaml.safe_load(rp.read_text(encoding="utf-8")) or {}
        ext = data.get("extends")
        if ext:
            parent = _load((rp.parent / str(ext)).resolve())
            return _merge(parent, data)
        return {k: v for k, v in data.items() if k != "extends"}

    return _load(path)


def parse_mux_size_arg(raw: str | None) -> tuple[int, int] | None:
    if not raw:
        return None
    parts = raw.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Expected WxH for --mux-size, got {raw!r}")
    w, h = int(parts[0]), int(parts[1])
    if w <= 0 or h <= 0:
        raise ValueError(f"Expected positive dimensions for --mux-size, got {raw!r}")
    return (w, h)


def load_track_timeline(
    jsonl_path: str | Path,
    cam: str | None = None,
    mux_size_override: tuple[int, int] | None = None,
) -> TrackTimeline:
    """Parse iris:meta and iris:overlay records from a golden JSONL file for one camera."""
    meta_by_cam: dict[str, list[FrameTracks]] = collections.defaultdict(list)
    overlay_by_cam: dict[str, dict[float, tuple[list[BoxObs], list[OverlayLine]]]] = collections.defaultdict(dict)
    overlay_w, overlay_h = 0, 0
    max_x2, max_y2 = 0.0, 0.0

    with open(jsonl_path, encoding="utf-8") as fh:
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
                c = key[len("iris:meta:") :]
                pts = _to_float(fields.get("pts"), 0.0)
                fnum = _to_int(fields.get("frame_num"), len(meta_by_cam[c]))
                dets_raw = _json_maybe(fields.get("dets")) or []
                pboxes_raw = _json_maybe(fields.get("person_boxes")) or []
                pbox_by_id: dict[int, tuple[float, float, float, float]] = {}
                for b in pboxes_raw:
                    if isinstance(b, list) and len(b) >= 5:
                        oid_b = _to_int(b[4], -1)
                        if oid_b >= 0:
                            pbox_by_id[oid_b] = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))

                gids_raw = _json_maybe(fields.get("global_ids")) or {}
                gids = {str(k): _to_int(v, -1) for k, v in gids_raw.items()} if isinstance(gids_raw, dict) else {}

                boxes: list[BoxObs] = []
                seen_oids: set[int] = set()
                for d in dets_raw:
                    if not isinstance(d, list) or len(d) < 3:
                        continue
                    label = str(d[0])
                    conf = _to_float(d[1], 0.0)
                    oid = _to_int(d[2], -1)
                    if oid < 0:
                        continue
                    rect: tuple[float, float, float, float] | None = None
                    if len(d) >= 7:
                        rect = (float(d[3]), float(d[4]), float(d[5]), float(d[6]))
                    elif label == "person" and oid in pbox_by_id:
                        rect = pbox_by_id[oid]
                    if rect is None:
                        continue
                    max_x2 = max(max_x2, rect[0] + rect[2])
                    max_y2 = max(max_y2, rect[1] + rect[3])
                    gid = gids.get(str(oid))
                    if gid is not None and gid < 0:
                        gid = None
                    boxes.append(
                        BoxObs(
                            oid=oid,
                            label=label,
                            conf=conf,
                            left=rect[0],
                            top=rect[1],
                            width=rect[2],
                            height=rect[3],
                            global_id=gid,
                        )
                    )
                    seen_oids.add(oid)

                # Include any person_boxes not matched above (e.g. if dets omitted a person box)
                for oid_b, rect in pbox_by_id.items():
                    if oid_b not in seen_oids:
                        max_x2 = max(max_x2, rect[0] + rect[2])
                        max_y2 = max(max_y2, rect[1] + rect[3])
                        boxes.append(
                            BoxObs(
                                oid=oid_b,
                                label="person",
                                conf=1.0,
                                left=rect[0],
                                top=rect[1],
                                width=rect[2],
                                height=rect[3],
                                global_id=gids.get(str(oid_b)),
                            )
                        )

                meta_by_cam[c].append(FrameTracks(pts=pts, frame_num=fnum, boxes=boxes))

            elif key.startswith("iris:overlay:"):
                c = key[len("iris:overlay:") :]
                w_ov = _to_int(fields.get("w"), 0)
                h_ov = _to_int(fields.get("h"), 0)
                if w_ov > 0 and h_ov > 0:
                    overlay_w, overlay_h = w_ov, h_ov
                pts = round(_to_float(fields.get("pts"), 0.0), 3)
                shapes = _json_maybe(fields.get("shapes")) or []
                ov_boxes: list[BoxObs] = []
                ov_lines: list[OverlayLine] = []
                for s in shapes:
                    if not isinstance(s, dict):
                        continue
                    stype = s.get("type")
                    if stype == "box":
                        rect = s.get("rect") or []
                        oid_s = _to_int(s.get("id"), -1)
                        if isinstance(rect, list) and len(rect) >= 4 and oid_s >= 0:
                            l_x, t_y, b_w, b_h = [float(v) for v in rect[:4]]
                            max_x2 = max(max_x2, l_x + b_w)
                            max_y2 = max(max_y2, t_y + b_h)
                            ov_boxes.append(
                                BoxObs(
                                    oid=oid_s,
                                    label=str(s.get("style") or "object"),
                                    conf=1.0,
                                    left=l_x,
                                    top=t_y,
                                    width=b_w,
                                    height=b_h,
                                    vehicle_state=s.get("state"),
                                )
                            )
                    elif stype == "line":
                        p1, p2 = s.get("p1"), s.get("p2")
                        if isinstance(p1, list) and len(p1) >= 2 and isinstance(p2, list) and len(p2) >= 2:
                            ov_lines.append(
                                OverlayLine(
                                    p1=(float(p1[0]), float(p1[1])),
                                    p2=(float(p2[0]), float(p2[1])),
                                    label=str(s.get("label") or ""),
                                    style=str(s.get("style") or "close"),
                                )
                            )
                if ov_boxes or ov_lines:
                    overlay_by_cam[c][pts] = (ov_boxes, ov_lines)

    all_cams = set(meta_by_cam.keys()) | set(overlay_by_cam.keys())
    if not all_cams:
        raise ValueError(f"No iris:meta:* or iris:overlay:* frames found in {jsonl_path}")

    chosen_cam = cam
    if chosen_cam is None:
        chosen_cam = max(all_cams, key=lambda c: (len(meta_by_cam.get(c, [])), len(overlay_by_cam.get(c, {}))))
    elif chosen_cam not in all_cams:
        raise ValueError(f"Camera {chosen_cam!r} not in {jsonl_path}; available: {sorted(all_cams)}")

    frames = list(meta_by_cam.get(chosen_cam, []))
    ov_map = overlay_by_cam.get(chosen_cam, {})

    if not frames and ov_map:
        for idx, (pts, (ov_b, ov_l)) in enumerate(sorted(ov_map.items()), start=1):
            frames.append(FrameTracks(pts=pts, frame_num=idx, boxes=ov_b, lines=ov_l))
    elif frames and ov_map:
        for fr in frames:
            pair = ov_map.get(round(fr.pts, 3))
            if pair is not None:
                ov_b, ov_l = pair
                if not fr.boxes and ov_b:
                    fr.boxes = ov_b
                if ov_l:
                    fr.lines = ov_l

    frames.sort(key=lambda f: (f.pts, f.frame_num))

    if mux_size_override is not None:
        mux_size = mux_size_override
    elif overlay_w > 0 and overlay_h > 0:
        mux_size = (overlay_w, overlay_h)
    else:
        mw = 1920 if max_x2 > 1280 else (1280 if max_x2 > 1080 else 1080 if max_x2 > 0 else 1280)
        mh = 1080 if max_y2 > 720 else 720
        mux_size = (mw, mh)

    return TrackTimeline(cam=chosen_cam, frames=frames, mux_size=mux_size)


def id_color_bgr(oid: int) -> tuple[int, int, int]:
    """Return deterministic, high-contrast BGR color via golden-angle hue spacing."""
    hue = ((abs(int(oid)) * 137.508) % 360.0) / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.88, 0.96)
    return (int(round(b * 255.0)), int(round(g * 255.0)), int(round(r * 255.0)))


def scale_box(
    box: BoxObs,
    mux_size: tuple[int, int],
    panel_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Scale (left, top, width, height) from mux pixels to panel pixels as (x1, y1, x2, y2)."""
    mw, mh = max(1, mux_size[0]), max(1, mux_size[1])
    pw, ph = panel_size
    sx = pw / float(mw)
    sy = ph / float(mh)
    x1 = int(round(box.left * sx))
    y1 = int(round(box.top * sy))
    x2 = int(round((box.left + box.width) * sx))
    y2 = int(round((box.top + box.height) * sy))
    return (
        max(0, min(pw - 1, x1)),
        max(0, min(ph - 1, y1)),
        max(0, min(pw - 1, x2)),
        max(0, min(ph - 1, y2)),
    )


class TimelineCursor:
    """Synchronizes video PTS to recorded TrackTimeline frames and maintains ID trails/stats."""

    def __init__(self, timeline: TrackTimeline, hold_seconds: float, trail_seconds: float) -> None:
        self.timeline = timeline
        self.pts_list = timeline.pts_list
        self.hold_seconds = hold_seconds
        self.trail_seconds = trail_seconds
        self.trails: dict[int, collections.deque[tuple[float, float, float]]] = collections.defaultdict(
            collections.deque
        )
        self.first_seen_pts: dict[int, float] = {}
        self.seen_ids: set[int] = set()
        self.spawns_after_warmup: set[int] = set()
        self._last_consumed_idx = -1

    def at(self, video_pts: float) -> FrameTracks | None:
        if not self.pts_list:
            return None
        idx = bisect.bisect_right(self.pts_list, video_pts + 0.04) - 1
        if idx < 0:
            return None
        fr = self.timeline.frames[idx]
        if video_pts - fr.pts > self.hold_seconds:
            return None

        # Advance internal stats/trails up through `idx` so trails stay continuous even on skipped steps
        if idx > self._last_consumed_idx:
            for k in range(self._last_consumed_idx + 1, idx + 1):
                step_fr = self.timeline.frames[k]
                t_fr = step_fr.pts
                for b in step_fr.boxes:
                    self.seen_ids.add(b.oid)
                    if b.oid not in self.first_seen_pts:
                        self.first_seen_pts[b.oid] = t_fr
                        if t_fr > 1.0:
                            self.spawns_after_warmup.add(b.oid)
                    cx, cy = b.center
                    q = self.trails[b.oid]
                    q.append((t_fr, cx, cy))
                    while q and (t_fr - q[0][0]) > self.trail_seconds:
                        q.popleft()
            self._last_consumed_idx = idx
        return fr


def _draw_dashed_rect(
    img: Any,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
    dash: int = 8,
    gap: int = 5,
) -> None:
    import cv2

    x1, y1 = pt1
    x2, y2 = pt2
    edges = [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ]
    for (ax, ay), (bx, by) in edges:
        length = math.hypot(bx - ax, by - ay)
        if length < 1.0:
            continue
        ux, uy = (bx - ax) / length, (by - ay) / length
        s = 0.0
        while s < length:
            e = min(s + dash, length)
            cv2.line(
                img,
                (int(round(ax + ux * s)), int(round(ay + uy * s))),
                (int(round(ax + ux * e)), int(round(ay + uy * e))),
                color,
                thickness,
                cv2.LINE_AA,
            )
            s += dash + gap


def _annotate_panel(
    panel_bgr: Any,
    cursor: TimelineCursor,
    fr: FrameTracks | None,
    panel_size: tuple[int, int],
) -> tuple[int, int]:
    """Draw trails, bounding boxes, ID pills, shadow markers, and overlay lines onto `panel_bgr`."""
    import cv2

    pw, ph = panel_size
    mux_size = cursor.timeline.mux_size
    sx = pw / float(max(1, mux_size[0]))
    sy = ph / float(max(1, mux_size[1]))

    active_cnt = 0
    shadow_cnt = 0
    if fr is None:
        return (0, 0)

    # 1. Proximity overlay lines (under boxes)
    line_colors = {"latched": (0, 0, 255), "close": (0, 165, 255), "far": (170, 255, 200)}
    for ln in fr.lines:
        p1 = (int(round(ln.p1[0] * sx)), int(round(ln.p1[1] * sy)))
        p2 = (int(round(ln.p2[0] * sx)), int(round(ln.p2[1] * sy)))
        c_ln = line_colors.get(ln.style, (0, 165, 255))
        cv2.line(panel_bgr, p1, p2, (20, 20, 20), 4, cv2.LINE_AA)
        cv2.line(panel_bgr, p1, p2, c_ln, 2, cv2.LINE_AA)
        if ln.label:
            mid = ((p1[0] + p2[0]) // 2 + 4, (p1[1] + p2[1]) // 2)
            cv2.putText(panel_bgr, ln.label, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 3, cv2.LINE_AA)
            cv2.putText(panel_bgr, ln.label, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.48, c_ln, 1, cv2.LINE_AA)

    # 2. Motion trails
    for b in fr.boxes:
        q = cursor.trails.get(b.oid)
        if not q or len(q) < 2:
            continue
        color = id_color_bgr(b.oid)
        pts = [(int(round(cx * sx)), int(round(cy * sy))) for _, cx, cy in q]
        n_seg = len(pts) - 1
        for i in range(n_seg):
            thick = 1 if i < n_seg // 2 else 2
            cv2.line(panel_bgr, pts[i], pts[i + 1], color, thick, cv2.LINE_AA)

    # 3. Bounding boxes + labels
    for b in fr.boxes:
        x1, y1, x2, y2 = scale_box(b, mux_size, panel_size)
        color = id_color_bgr(b.oid)
        if b.is_shadow:
            shadow_cnt += 1
            _draw_dashed_rect(panel_bgr, (x1, y1), (x2, y2), color, thickness=2)
        else:
            active_cnt += 1
            cv2.rectangle(panel_bgr, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        first_t = cursor.first_seen_pts.get(b.oid, fr.pts)
        is_new_spawn = (first_t > 1.0) and (0.0 <= fr.pts - first_t <= 0.6)
        if is_new_spawn:
            cv2.rectangle(
                panel_bgr,
                (max(0, x1 - 3), max(0, y1 - 3)),
                (min(pw - 1, x2 + 3), min(ph - 1, y2 + 3)),
                (0, 240, 255),
                2,
                cv2.LINE_AA,
            )

        tag_parts = [f"#{b.oid}"]
        if b.global_id is not None:
            tag_parts.append(f"G{b.global_id}")
        if b.is_shadow:
            tag_parts.append("coast")
        else:
            tag_parts.append(f"{b.conf:.2f}")
        if is_new_spawn:
            tag_parts.append("NEW")
        label_txt = " ".join(tag_parts)

        (tw, th), baseline = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        pill_y1 = max(0, y1 - th - baseline - 6)
        pill_y2 = pill_y1 + th + baseline + 6
        pill_x2 = min(pw - 1, x1 + tw + 8)
        bg_col = (0, 215, 255) if is_new_spawn else color
        fg_col = (15, 15, 15)
        cv2.rectangle(panel_bgr, (x1, pill_y1), (pill_x2, pill_y2), bg_col, -1)
        cv2.putText(
            panel_bgr,
            label_txt,
            (x1 + 4, pill_y2 - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            fg_col,
            1,
            cv2.LINE_AA,
        )

    return (active_cnt, shadow_cnt)


class _VideoSink:
    """Writes BGR frames to MP4 using H.264 ffmpeg pipe (preferred) or OpenCV fallback."""

    def __init__(self, out_path: Path, width: int, height: int, fps: float) -> None:
        import cv2

        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.proc: subprocess.Popen[bytes] | None = None
        self.writer: Any = None
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            cmd = [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{width}x{height}",
                "-r",
                f"{fps:.4f}",
                "-i",
                "-",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(out_path),
            ]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
            if not self.writer.isOpened():
                raise RuntimeError(f"Failed to open cv2.VideoWriter for {out_path}")

    def write(self, frame_bgr: Any) -> None:
        if self.proc is not None and self.proc.stdin is not None:
            self.proc.stdin.write(frame_bgr.tobytes())
        elif self.writer is not None:
            self.writer.write(frame_bgr)

    def close(self) -> None:
        if self.proc is not None:
            if self.proc.stdin is not None:
                self.proc.stdin.close()
            rc = self.proc.wait()
            if rc != 0:
                raise RuntimeError(f"ffmpeg encoding failed with exit code {rc}")
        elif self.writer is not None:
            self.writer.release()


def render_side_by_side(
    video_path: str | Path,
    before_jsonl: str | Path,
    after_jsonl: str | Path,
    out_path: str | Path,
    *,
    config_path: str | Path | None = None,
    cam: str | None = None,
    mux_size_override: tuple[int, int] | None = None,
    start_s: float = 0.0,
    duration_s: float = 0.0,
    panel_width: int = 960,
    trail_seconds: float = 1.2,
    hold_seconds: float = 0.35,
    before_label: str = "BEFORE: NVIDIA NvDCF",
    after_label: str = "AFTER: VEMA v2 (Polar-EMA)",
) -> dict[str, Any]:
    """Render side-by-side Before vs After video overlay and return summary counts."""
    import cv2
    import numpy as np

    cfg_mux: tuple[int, int] | None = mux_size_override
    if cfg_mux is None and config_path is not None:
        cfg_data = _load_yaml_chain(Path(config_path))
        mw = _to_int(cfg_data.get("mux_width"), 0)
        mh = _to_int(cfg_data.get("mux_height"), 0)
        if mw > 0 and mh > 0:
            cfg_mux = (mw, mh)

    tl_before = load_track_timeline(before_jsonl, cam=cam, mux_size_override=cfg_mux)
    tl_after = load_track_timeline(after_jsonl, cam=tl_before.cam, mux_size_override=tl_before.mux_size)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        if fps <= 0.1 or fps > 240.0:
            fps = 25.0
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        pw = max(320, int(panel_width) - (int(panel_width) % 2))
        ph = int(round(pw * (vid_h / float(max(1, vid_w)))))
        ph -= ph % 2
        gap = 4
        header_h = 58
        footer_h = 26
        canvas_w = 2 * pw + gap
        canvas_h = header_h + ph + footer_h
        canvas_w += canvas_w % 2
        canvas_h += canvas_h % 2

        if start_s > 0.0:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)

        max_frames = int(round(duration_s * fps)) if duration_s > 0.0 else 0
        est_total_dur = duration_s if duration_s > 0.0 else (total_src_frames / fps if total_src_frames > 0 else 0.0)

        # Normalize timeline PTS base if DeepStream started near t0 != start_s
        min_meta_pts = min(
            tl_before.frames[0].pts if tl_before.frames else 0.0,
            tl_after.frames[0].pts if tl_after.frames else 0.0,
        )
        # When replaying a clipped slice directly, DeepStream PTS starts at ~0.0 even if --start>0 on original video
        timeline_starts_at_zero = min_meta_pts < 2.0 and start_s > 5.0

        cur_before = TimelineCursor(tl_before, hold_seconds=hold_seconds, trail_seconds=trail_seconds)
        cur_after = TimelineCursor(tl_after, hold_seconds=hold_seconds, trail_seconds=trail_seconds)

        out_file = Path(out_path)
        sink = _VideoSink(out_file, canvas_w, canvas_h, fps)
        frame_idx = 0

        try:
            while True:
                if max_frames > 0 and frame_idx >= max_frames:
                    break
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                t_rel = frame_idx / fps
                t_lookup = t_rel if timeline_starts_at_zero else (start_s + t_rel)
                frame_idx += 1

                base_panel = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)
                left_panel = base_panel.copy()
                right_panel = base_panel.copy()

                fr_b = cur_before.at(t_lookup)
                fr_a = cur_after.at(t_lookup)
                act_b, sh_b = _annotate_panel(left_panel, cur_before, fr_b, (pw, ph))
                act_a, sh_a = _annotate_panel(right_panel, cur_after, fr_a, (pw, ph))

                canvas = np.full((canvas_h, canvas_w, 3), 24, dtype=np.uint8)
                canvas[header_h : header_h + ph, 0:pw] = left_panel
                canvas[header_h : header_h + ph, pw + gap : pw + gap + pw] = right_panel
                canvas[header_h : header_h + ph, pw : pw + gap] = (70, 70, 70)

                # Header HUD
                n_uid_b = len(cur_before.seen_ids)
                n_uid_a = len(cur_after.seen_ids)
                sp_b = len(cur_before.spawns_after_warmup)
                sp_a = len(cur_after.spawns_after_warmup)

                cv2.putText(
                    canvas, before_label, (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (235, 235, 235), 2, cv2.LINE_AA
                )
                cv2.putText(
                    canvas,
                    f"Active:{act_b}  Coast:{sh_b}  |  Unique IDs:{n_uid_b}  |  New Spawns(>1s):{sp_b}",
                    (14, 47),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (180, 210, 255),
                    1,
                    cv2.LINE_AA,
                )

                rx = pw + gap + 14
                after_title_col = (120, 245, 160) if n_uid_a <= n_uid_b else (120, 200, 255)
                cv2.putText(
                    canvas, after_label, (rx, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.64, after_title_col, 2, cv2.LINE_AA
                )
                uid_stat_col = (120, 245, 160) if (n_uid_a <= n_uid_b and sp_a <= sp_b) else (90, 200, 255)
                cv2.putText(
                    canvas,
                    f"Active:{act_a}  Coast:{sh_a}  |  Unique IDs:{n_uid_a}  |  New Spawns(>1s):{sp_a}",
                    (rx, 47),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    uid_stat_col,
                    1,
                    cv2.LINE_AA,
                )

                # Footer timeline & progress bar
                fy = header_h + ph + 18
                mm, ss = divmod(t_lookup, 60.0)
                dur_str = f" / {int(est_total_dur // 60):02d}:{est_total_dur % 60:04.1f}" if est_total_dur > 0 else ""
                cv2.putText(
                    canvas,
                    f"cam={tl_before.cam}  t={int(mm):02d}:{ss:04.1f}{dur_str}  frame={frame_idx}",
                    (14, fy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )
                if est_total_dur > 0.1:
                    frac = max(0.0, min(1.0, t_rel / est_total_dur))
                    bx1 = canvas_w // 2
                    bx2 = canvas_w - 20
                    cv2.rectangle(canvas, (bx1, header_h + ph + 9), (bx2, header_h + ph + 17), (55, 55, 55), -1)
                    fill_x = bx1 + int(round((bx2 - bx1) * frac))
                    cv2.rectangle(canvas, (bx1, header_h + ph + 9), (fill_x, header_h + ph + 17), (100, 220, 150), -1)

                sink.write(canvas)
        finally:
            sink.close()
    finally:
        cap.release()

    return {
        "out_path": str(out_path),
        "cam": tl_before.cam,
        "frames_rendered": frame_idx,
        "mux_size": list(tl_before.mux_size),
        "before_unique_ids": len(cur_before.seen_ids),
        "after_unique_ids": len(cur_after.seen_ids),
        "before_spawns_after_1s": len(cur_before.spawns_after_warmup),
        "after_spawns_after_1s": len(cur_after.spawns_after_warmup),
    }


def _run_checked(cmd: list[str], env: dict[str, str] | None = None, capture: bool = False) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        check=True,
    )
    return proc.stdout if capture and proc.stdout is not None else ""


def _stage_clip(src_video: Path, dst_clip: Path, start_s: float, duration_s: float) -> None:
    if start_s <= 0.0 and duration_s <= 0.0:
        try:
            os.link(src_video, dst_clip)
            return
        except OSError:
            shutil.copy2(src_video, dst_clip)
            return

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required on PATH to slice a clip with --start / --duration")
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
    if start_s > 0.0:
        cmd += ["-ss", f"{start_s:.3f}"]
    cmd += ["-i", str(src_video)]
    if duration_s > 0.0:
        cmd += ["-t", f"{duration_s:.3f}"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-an", str(dst_clip)]
    subprocess.run(cmd, check=True)


def run_and_render(args: argparse.Namespace) -> int:
    """Execute both DeepStream arms on the server and render the side-by-side MP4."""
    import yaml
    from tools.tracker_ab_eval import evaluate_ab, render_markdown

    product = args.product
    before_golden = REPO_ROOT / "tests" / "configs" / f"golden-{product}.yaml"
    after_golden = REPO_ROOT / "tests" / "configs" / f"golden-{product}-vema2.yaml"
    if not before_golden.is_file() or not after_golden.is_file():
        raise FileNotFoundError(
            f"Missing golden config pair for product {product!r}: {before_golden} / {after_golden}"
        )

    base_cfg = _load_yaml_chain(before_golden)
    base_sources = base_cfg.get("sources") or []
    if not base_sources:
        raise ValueError(f"No sources declared in {before_golden}")
    cam_id = args.cam or str(base_sources[0].get("id") or "cam-1")
    cam_roles = base_sources[0].get("roles")

    if args.video:
        src_video = Path(args.video).resolve()
    else:
        raw_uri = str(base_sources[0].get("uri") or "")
        if not raw_uri.startswith("file:///workspace/iris/"):
            raise ValueError(f"Golden URI is not a repo file path ({raw_uri}); pass --video explicitly.")
        src_video = (REPO_ROOT / raw_uri[len("file:///workspace/iris/") :]).resolve()

    if not src_video.is_file():
        raise FileNotFoundError(f"Source video not found: {src_video}")

    run_id = uuid.uuid4().hex[:8]
    stage_dir = REPO_ROOT / "data" / "uploads" / "_ab_tmp" / run_id
    stage_dir.mkdir(parents=True, exist_ok=True)
    clip_path = stage_dir / "clip.mp4"

    try:
        print(f"[ab-render] Staging clip from {src_video} (start={args.start}s, duration={args.duration}s)...")
        _stage_clip(src_video, clip_path, start_s=args.start, duration_s=args.duration)

        src_entry: dict[str, Any] = {
            "id": cam_id,
            "uri": f"file:///workspace/iris/data/uploads/_ab_tmp/{run_id}/clip.mp4",
        }
        if cam_roles:
            src_entry["roles"] = cam_roles

        before_cfg_path = stage_dir / "before.yaml"
        after_cfg_path = stage_dir / "after.yaml"
        before_rel = Path("..") / ".." / ".." / ".." / "tests" / "configs" / before_golden.name
        after_rel = Path("..") / ".." / ".." / ".." / "tests" / "configs" / after_golden.name

        for dst_cfg, rel_parent in ((before_cfg_path, before_rel), (after_cfg_path, after_rel)):
            payload: dict[str, Any] = {
                "extends": rel_parent.as_posix(),
                "file_loop": False,
                "sources": [src_entry],
            }
            if args.sample_fps > 0.0:
                payload["sample_fps"] = float(args.sample_fps)
            dst_cfg.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

        redis_seed = REPO_ROOT / "tests" / "configs" / f"golden-{product}.redis"
        if redis_seed.is_file():
            shutil.copy2(redis_seed, stage_dir / "before.redis")
            shutil.copy2(redis_seed, stage_dir / "after.redis")

        before_jsonl = stage_dir / "before.jsonl"
        after_jsonl = stage_dir / "after.jsonl"
        before_log = stage_dir / "before.log"
        after_log = stage_dir / "after.log"

        run_env = dict(os.environ)
        run_env["GOLDEN_IDLE"] = str(args.idle)
        run_env["GOLDEN_MAX"] = str(args.max_seconds)

        for arm_name, cfg_file, jsonl_file, log_file in (
            ("BEFORE (NvDCF)", before_cfg_path, before_jsonl, before_log),
            ("AFTER (VEMA v2)", after_cfg_path, after_jsonl, after_log),
        ):
            print(f"[ab-render] Running DeepStream arm: {arm_name} ...")
            subprocess.run(
                ["docker", "compose", "rm", "-sf", "deepstream-sidecar"],
                cwd=str(REPO_ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            rel_cfg = cfg_file.relative_to(REPO_ROOT).as_posix()
            _run_checked(["bash", "tools/engine_golden_run.sh", rel_cfg, str(jsonl_file)], env=run_env)
            logs_txt = _run_checked(
                ["docker", "compose", "logs", "--no-log-prefix", "deepstream-sidecar"],
                capture=True,
            )
            log_file.write_text(logs_txt, encoding="utf-8")

        report = evaluate_ab(
            before_jsonl,
            after_jsonl,
            product=product,
            log_nvdcf_path=before_log,
            log_vema3d_path=after_log,
        )
        md = render_markdown(report)
        print("\n" + md)

        out_mp4 = Path(args.out).resolve()
        out_mp4.with_suffix(".scorecard.md").write_text(md + "\n", encoding="utf-8")
        out_mp4.with_suffix(".scorecard.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

        print(f"[ab-render] Rendering side-by-side MP4 to {out_mp4} ...")
        summary = render_side_by_side(
            clip_path,
            before_jsonl,
            after_jsonl,
            out_mp4,
            config_path=before_cfg_path,
            cam=cam_id,
            mux_size_override=parse_mux_size_arg(args.mux_size),
            start_s=0.0,
            duration_s=0.0,
            panel_width=args.panel_width,
            trail_seconds=args.trail_seconds,
            hold_seconds=args.hold_seconds,
            before_label=args.before_label,
            after_label=args.after_label,
        )
        print(json.dumps(summary, indent=2))
        return 0
    finally:
        if not args.keep_temp:
            shutil.rmtree(stage_dir, ignore_errors=True)
            parent_tmp = REPO_ROOT / "data" / "uploads" / "_ab_tmp"
            if parent_tmp.is_dir() and not any(parent_tmp.iterdir()):
                parent_tmp.rmdir()


def _add_render_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--mux-size", default=None, help="Override mux resolution WxH (e.g. 1280x720)")
    p.add_argument("--cam", default=None, help="Camera stream id (auto-detected if omitted)")
    p.add_argument("--start", type=float, default=0.0, help="Start offset seconds inside source video")
    p.add_argument("--duration", type=float, default=0.0, help="Seconds to render/replay (0 = full clip)")
    p.add_argument("--panel-width", type=int, default=960, help="Width in px per side-by-side panel")
    p.add_argument("--trail-seconds", type=float, default=1.2, help="Motion trail duration in seconds")
    p.add_argument("--hold-seconds", type=float, default=0.35, help="Hold last meta frame between sample_fps ticks")
    p.add_argument("--before-label", default="BEFORE: NVIDIA NvDCF")
    p.add_argument("--after-label", default="AFTER: VEMA v2 (Polar-EMA)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Before/After Side-by-Side DeepStream Tracker Visualization")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_render = sub.add_parser("render", help="Render MP4 directly from video + before/after JSONLs")
    p_render.add_argument("--video", required=True, help="Input MP4 path")
    p_render.add_argument("--before", required=True, help="Baseline tracker JSONL (engine_golden.py)")
    p_render.add_argument("--after", required=True, help="Candidate tracker JSONL (engine_golden.py)")
    p_render.add_argument("--out", required=True, help="Output side-by-side MP4 path")
    p_render.add_argument("--config", default=None, help="Optional sidecar YAML to infer mux_width x mux_height")
    _add_render_opts(p_render)

    p_run = sub.add_parser("run", help="Run DeepStream NvDCF & VEMA v2 arms on server, then render MP4")
    p_run.add_argument("--product", required=True, choices=["barber", "attendance", "zones", "proximity", "multicam"])
    p_run.add_argument("--video", default=None, help="Custom video file path (defaults to golden clip)")
    p_run.add_argument("--out", required=True, help="Output side-by-side MP4 path")
    p_run.add_argument("--sample-fps", type=float, default=30.0, help="Metadata publish rate for crisp overlays")
    p_run.add_argument("--idle", type=int, default=90, help="GOLDEN_IDLE seconds")
    p_run.add_argument("--max-seconds", type=int, default=1800, help="GOLDEN_MAX seconds")
    p_run.add_argument("--keep-temp", action="store_true", help="Keep staged clip, JSONLs, and logs under data/uploads/_ab_tmp")
    _add_render_opts(p_run)

    args = parser.parse_args()
    if args.cmd == "render":
        res = render_side_by_side(
            args.video,
            args.before,
            args.after,
            args.out,
            config_path=args.config,
            cam=args.cam,
            mux_size_override=parse_mux_size_arg(args.mux_size),
            start_s=args.start,
            duration_s=args.duration,
            panel_width=args.panel_width,
            trail_seconds=args.trail_seconds,
            hold_seconds=args.hold_seconds,
            before_label=args.before_label,
            after_label=args.after_label,
        )
        print(json.dumps(res, indent=2))
        return 0
    return run_and_render(args)


if __name__ == "__main__":
    raise SystemExit(main())
