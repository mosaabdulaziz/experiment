"""Box stabilization: the algorithm behind the box_stabilizer module.

Mirrors mtmc/annotate_videos.py's per-track canonical box so the live preview
(and the iris:meta feed) shows stable whole-vehicle boxes instead of flicker:

  * containment suppression drops a nested cab-only box in the same frame when
    the whole-vehicle box is also present (plain NMS won't remove it; see
    mtmc/detection_postprocess.py);
  * a per-(stream, object_id) canonical box keeps the vehicle's full size
    while the detector alternates between whole-vehicle and cab-only boxes
    (hysteresis: a much-smaller update translates the held box instead of
    shrinking it);
  * held "virtual" boxes stay draw-able for ``hold_seconds`` even when the
    track emits no box, and they still suppress a cab-only box that surfaces
    under a NEW tracker id.

Boxes are ``(left, top, width, height, label, conf, object_id)`` in mux
coordinates — the same shape the sidecar probe builds and draws.
"""
from __future__ import annotations

import numpy as np

from .postprocess import containment_suppression


def _to_tuple(box, label, conf, oid):
    """canon box (x1, y1, x2, y2) -> probe box tuple in mux coords."""
    return (box[0], box[1], box[2] - box[0], box[3] - box[1],
            label, conf, oid)


class BoxStabilizer:
    def __init__(self, labels=("vehicle",), ratio=0.85, hold_seconds=0.75,
                 ema=0.55, ratio_hysteresis=0.85):
        self._labels = frozenset(labels)
        self._ratio = float(ratio)
        self._hold_s = float(hold_seconds)
        self._ema = float(ema)
        self._hyst = float(ratio_hysteresis)
        # (stream_id, object_id) -> {"box": [x1,y1,x2,y2], "lab": str,
        #                            "conf": float, "cx": float, "cy": float,
        #                            "pts": float}
        self._canon: dict = {}

    def _virtuals(self, stream_id, pts, exclude=frozenset()):
        """Canon entries of this stream still within the hold window."""
        out = []
        for key, rec in list(self._canon.items()):
            if key[0] != stream_id:
                continue
            if rec["pts"] - pts > 0:  # PTS wrapped/rewound: pretend stale
                continue
            if key[1] in exclude:
                continue
            if pts - rec["pts"] <= self._hold_s:
                out.append((rec["box"], rec["lab"], rec["conf"], key[1]))
        return out

    def _prune(self, stream_id, pts):
        for key in [k for k, r in self._canon.items()
                    if k[0] == stream_id and pts - r["pts"] > self._hold_s * 6]:
            del self._canon[key]

    def stabilize(self, stream_id, boxes, pts):
        """Stabilize one frame's boxes for one stream; returns the box list.

        Labels in ``self._labels`` are stabilized; everything else passes
        through untouched. Face boxes should not be passed in here.
        """
        stab = [b for b in boxes if b[4] in self._labels]
        out = [b for b in boxes if b[4] not in self._labels]

        if not stab:
            out.extend(_to_tuple(b, l, c, o)
                       for b, l, c, o in self._virtuals(stream_id, pts))
            self._prune(stream_id, pts)
            return out

        oids = [int(b[6]) for b in stab]
        xyxy = np.asarray([[b[0], b[1], b[0] + b[2], b[1] + b[3]]
                           for b in stab], dtype=np.float32)
        confs = np.asarray([float(b[5]) for b in stab], dtype=np.float32)
        labs = [b[4] for b in stab]

        virt = self._virtuals(stream_id, pts, exclude=set(oids))
        n_cur = len(stab)
        if virt:
            vxy = np.asarray([tuple(v[0]) for v in virt], dtype=np.float32)
            all_xyxy = np.concatenate([xyxy, vxy], axis=0)
            all_conf = np.concatenate([confs, [float(v[2]) for v in virt]])
            all_lab = labs + [v[1] for v in virt]
        else:
            all_xyxy, all_conf, all_lab = xyxy, confs, labs

        # Containment suppression per label (so a person box inside a vehicle
        # can never be dropped — only same-label nesting is).
        keep = np.ones(len(all_xyxy), dtype=bool)
        for lab in set(all_lab):
            m = np.array([l == lab for l in all_lab])
            if m.sum() >= 2:
                keep[m] = containment_suppression(
                    all_xyxy[m], self._ratio, all_conf[m])
        cur_keep = keep[:n_cur]
        virt_keep = keep[n_cur:] if virt else []

        # Emit surviving virtual (held) boxes as-is.
        for (box, lab, conf, oid), kk in zip(virt, virt_keep):
            if kk:
                out.append(_to_tuple(box, lab, conf, oid))

        for i in range(n_cur):
            if not cur_keep[i]:
                continue
            oid = oids[i]
            bx = (float(xyxy[i, 0]), float(xyxy[i, 1]),
                  float(xyxy[i, 2]), float(xyxy[i, 3]))
            conf = float(confs[i])
            lab = labs[i]
            cx, cy = (bx[0] + bx[2]) / 2.0, (bx[1] + bx[3]) / 2.0
            area = (bx[2] - bx[0]) * (bx[3] - bx[1])

            key = (stream_id, oid)
            rec = self._canon.get(key)
            # pts < rec["pts"]: file-loop sources rewind PTS at the loop
            # boundary — treat the box as fresh rather than blending across
            # the video restart.
            if (rec is None or pts < rec["pts"]
                    or pts - rec["pts"] > self._hold_s):
                rec = {"box": list(bx), "lab": lab, "conf": conf,
                       "cx": cx, "cy": cy, "pts": pts}
                self._canon[key] = rec
                out.append(_to_tuple(rec["box"], lab, conf, oid))
                continue

            obx = rec["box"]
            oarea = (obx[2] - obx[0]) * (obx[3] - obx[1])
            if area >= self._hyst * max(oarea, 1e-9):
                # consistent size -> blend both position and size
                nb = [self._ema * new + (1 - self._ema) * old
                      for new, old in zip(bx, obx)]
                rec.update({"box": nb, "lab": lab, "conf": conf,
                            "cx": cx, "cy": cy, "pts": pts})
            else:
                # cab-only (much smaller) box: move the held whole-vehicle box
                # by this detection's motion, keep the held SIZE.
                dx, dy = cx - rec["cx"], cy - rec["cy"]
                rec["box"] = [obx[0] + dx, obx[1] + dy,
                              obx[2] + dx, obx[3] + dy]
                rec.update({"cx": cx, "cy": cy, "conf": conf, "pts": pts})
            out.append(_to_tuple(rec["box"], rec["lab"], rec["conf"], oid))

        self._prune(stream_id, pts)
        return out