"""Keeps a vehicle's plate box steady between the frames the plate detector actually ran on.

The plate detector is a secondary GIE that looks at each vehicle every few frames, and its boxes
exist only on those frames, so drawn straight they blink. Here each plate is remembered as a
position RELATIVE to its vehicle's box (smoothed over detections), so on every frame it is redrawn
on the vehicle's current box and moves with it, and it is dropped only when no detection has
confirmed it for `hold_s` seconds or its vehicle is gone.
"""


def plate_size_ok(w: float, h: float, min_size=(0.0, 0.0), aspect=(0.0, float("inf"))) -> str:
    """"" when a plate box of w x h pixels is big and shaped enough to be read, else why not. A plate a few pixels
    across (a far vehicle, a box cut by the frame edge) cannot be read: the reader blows it up to 48 px high and
    returns characters it made up, which then count as a reading."""
    if w < min_size[0] or h < min_size[1]:
        return "too small to read"
    if h <= 0 or not (aspect[0] <= w / h <= aspect[1]):
        return "odd shape for a plate"
    return ""


def one_plate_per_vehicle(plates, vehicle_boxes, max_area_frac: float = 0.25,
                          remembered: dict | None = None, stick: float = 0.12,
                          dropped: list | None = None, min_size=(0.0, 0.0),
                          aspect=(0.0, float("inf")), frame: tuple | None = None,
                          edge_margin: float = 2.0) -> list[tuple]:
    """At most one plate per vehicle: the one nearest the bottom-centre of the vehicle's box.

    plates: (l, t, w, h, conf, parent_id); vehicle_boxes: (l, t, w, h, label, conf, object_id).
    A real plate sits low and central on the vehicle, so a second detection on the same vehicle
    (a sticker, a headlight, a plate on the vehicle behind) loses to the one closest to that
    spot. A plate whose centre is outside its vehicle's box, or that is big next to it, is not on
    the vehicle at all (it floats) and is dropped.

    `remembered` maps a vehicle id to the plate centre shown for it so far, relative to its box
    ((x, y) in 0..1). A detection within `stick` of it (in vehicle sizes) is kept over the
    bottom-centre rule, so the shown plate does not jump between two candidates frame to frame.

    `min_size` (width, height in pixels) and `aspect` (min, max width/height) drop plates too small or too oddly
    shaped to be read, BEFORE the nearest-to-bottom-centre choice: a 6 x 6 box must not win over a real plate
    just because it sits lower on the vehicle, nor be read at all.

    `frame` (width, height of the frame the boxes are in): a plate touching its edge is cut off by it (a vehicle
    driving out of view, its plate half in), and half a plate reads as made-up digits, so it is dropped. A
    later frame with the whole plate wins instead.
    """
    veh = {b[6]: b for b in vehicle_boxes}
    best: dict[int, tuple[float, tuple]] = {}
    for pl in plates:
        l, t, w, h, _conf, pid = pl
        v = veh.get(pid)
        if v is None or v[2] <= 0 or v[3] <= 0:
            if dropped is not None:
                dropped.append((pl, v, "no vehicle"))
            continue
        why = plate_size_ok(w, h, min_size, aspect)
        if not why and frame is not None and (l <= edge_margin or t <= edge_margin or
                                              l + w >= frame[0] - edge_margin or t + h >= frame[1] - edge_margin):
            why = "cut by the frame edge"
        if why:
            if dropped is not None:
                dropped.append((pl, v, why))
            continue
        cx, cy = l + w / 2.0, t + h / 2.0
        if not (v[0] <= cx <= v[0] + v[2] and v[1] <= cy <= v[1] + v[3]):
            if dropped is not None:
                dropped.append((pl, v, "centre outside its vehicle's box"))
            continue
        if w * h > max_area_frac * v[2] * v[3]:
            if dropped is not None:
                dropped.append((pl, v, "too big for its vehicle"))
            continue
        # distance to the bottom-centre, in units of the vehicle's own size
        d = (((cx - (v[0] + v[2] / 2.0)) / v[2]) ** 2 + ((cy - (v[1] + v[3])) / v[3]) ** 2) ** 0.5
        mem = (remembered or {}).get(pid)
        if mem is not None:
            dm = (((cx - v[0]) / v[2] - mem[0]) ** 2 + ((cy - v[1]) / v[3] - mem[1]) ** 2) ** 0.5
            if dm <= stick:
                d = -1.0 + dm          # the plate already shown for this vehicle: keep it
        if pid not in best or d < best[pid][0]:
            best[pid] = (d, pl)
    return [pl for _, pl in best.values()]


class PlateHold:
    def __init__(self, hold_s: float = 1.5, smooth: float = 0.5):
        self.hold_s = hold_s
        self.smooth = smooth          # weight of a new detection against the remembered position
        self._plates: dict[int, dict] = {}   # vehicle object id -> {"rel", "conf", "seen"}

    def centres(self) -> dict[int, tuple[float, float]]:
        """Plate centre shown per vehicle id, relative to its box (what one_plate_per_vehicle
        needs to keep the same plate)."""
        return {pid: (p["rel"][0] + p["rel"][2] / 2.0, p["rel"][1] + p["rel"][3] / 2.0)
                for pid, p in self._plates.items()}

    def update(self, plates, vehicle_boxes, pts: float) -> list[tuple]:
        """plates: (l, t, w, h, conf, parent_id) detected this frame; vehicle_boxes: (l, t, w, h,
        label, conf, object_id) of the vehicles this frame. Returns (l, t, w, h, conf, parent_id)
        for every plate to show now: detected ones (smoothed) plus held ones."""
        veh = {b[6]: b for b in vehicle_boxes}
        for l, t, w, h, conf, pid in plates:
            v = veh.get(pid)
            if v is None or v[2] <= 0 or v[3] <= 0:
                continue
            rel = ((l - v[0]) / v[2], (t - v[1]) / v[3], w / v[2], h / v[3])
            old = self._plates.get(pid)
            if old is not None:
                a = self.smooth
                rel = tuple(a * n + (1 - a) * o for n, o in zip(rel, old["rel"]))
            self._plates[pid] = {"rel": rel, "conf": conf, "seen": pts}
        out = []
        for pid in list(self._plates):
            p = self._plates[pid]
            v = veh.get(pid)
            if v is None or pts - p["seen"] > self.hold_s:
                del self._plates[pid]
                continue
            rx, ry, rw, rh = p["rel"]
            out.append((v[0] + rx * v[2], v[1] + ry * v[3], rw * v[2], rh * v[3], p["conf"], pid))
        return out


def resolve_parent(pid: int, box: tuple, vehicle_boxes) -> int:
    """The vehicle a plate detection belongs to: its parent id when that is a vehicle in this frame, else the
    smallest vehicle box holding the plate's centre (the sgie often reports no parent), else -1."""
    veh = list(vehicle_boxes)
    if any(b[6] == pid for b in veh):
        return pid
    cx, cy = box[0] + box[2] / 2.0, box[1] + box[3] / 2.0
    inside = [b for b in veh if b[0] <= cx <= b[0] + b[2] and b[1] <= cy <= b[1] + b[3]]
    return min(inside, key=lambda b: b[2] * b[3])[6] if inside else -1


class PlateDiag:
    """Why a vehicle did or did not get a plate: one summary line per vehicle once it has been out of sight
    for `quiet_s` seconds of stream time.

    For each (camera, vehicle id) it counts the analysed frames the vehicle was in, the frames the plate detector
    found anything on it (at any confidence the detector reports), the frames a plate survived the filters, and the
    reasons plates were dropped. A truck with a plate you can see in its picture and "plate detected in 0 of 9
    frames" was not looked at (or not found); "detected in 5, kept in 0, dropped: too small" was found and filtered.
    """

    def __init__(self, quiet_s: float = 3.0, emit=None):
        self._quiet = float(quiet_s)
        self._emit = emit or (lambda line: None)
        self._rec: dict[tuple, dict] = {}

    def frame(self, cam: str, pts: float, vehicle_ids, raw, kept, lost) -> None:
        """raw: [(parent vehicle id, conf)] of every plate the detector reported this frame; kept: plates that
        survived the filters, (l, t, w, h, conf, vehicle id); lost: [(plate, vehicle box or None, reason)]."""
        def rec(vid):
            r = self._rec.get((cam, vid))
            if r is None:
                r = self._rec[(cam, vid)] = {"first": pts, "last": pts, "frames": 0, "det": 0, "best": 0.0,
                                             "kept": 0, "drops": {}, "last_det": None}
            return r
        for vid in set(vehicle_ids):
            r = rec(vid)
            r["frames"] += 1
            r["last"] = pts
        for vid in {v for v, _c in raw}:
            r = rec(vid)
            r["det"] += 1
            r["last_det"] = pts
        for v, c in raw:
            rec(v)["best"] = max(rec(v)["best"], float(c))
        for vid in {p[5] for p in kept}:
            rec(vid)["kept"] += 1
        for pl, _v, why in lost:
            d = rec(pl[5])["drops"]
            d[why] = d.get(why, 0) + 1
        for key in [k for k, r in self._rec.items() if k[0] == cam and (pts - r["last"] > self._quiet or pts < r["first"])]:
            self._finish(key)

    def _finish(self, key) -> None:
        r = self._rec.pop(key)
        if r["frames"] < 2 and not r["det"]:
            return                       # a flicker
        drops = ", ".join(f"{why} x{n}" for why, n in sorted(r["drops"].items())) or "none"
        self._emit(f"[plate-diag] {key[0]} vehicle #{key[1]}: in {r['frames']} analysed frames "
                   f"({r['last'] - r['first']:.1f} s of video), a plate was detected in {r['det']}"
                   f"{(' (best conf %.2f)' % r['best']) if r['det'] else ''}, kept in {r['kept']}, "
                   f"dropped: {drops}")
