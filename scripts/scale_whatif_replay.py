#!/usr/bin/env python3
"""What-if replay: drive the REAL tracker (perception/modules/mtmc/core MTMCManager) with the tracklets of the day run, under old or new rules.

The day run kept no embeddings, boxes or per-tracklet crops, but its sidecar log says, per tracklet (camera + local id),
when it was seen, which old track it joined or started, and what the VLM answered; the run also saved one best crop per
truck / camera / visit. This rebuilds the tracklets from the log (13 sidecar segments: a heal restart empties the
in-memory tracks, as it did in the real run), shows each one to the manager as 1 fps frames whose picture is its saved crop,
and answers the manager's VLM questions from (1) identity (the same old track), (2) the old run's recorded verdicts, (3) fresh
hosted-VLM calls on the crops for pairs the old run never asked. Direction answers come from the offline pass
(scale_offline_direction.py, ~/URCA/IRIS/review/offline-direction/results.jsonl) or fresh calls. Unknown answers are
collected and asked in bulk, then the replay is repeated until nothing is missing (answers cached in cache.jsonl).

  scale_whatif_replay.py run --rules old|new [--from HH:MM --to HH:MM] [--continuous] [--dry-run] [--max-calls N]
  scale_whatif_replay.py compare            old vs new summary (and old vs the real run)

Limits: no Re-ID embeddings (candidate ranking falls back to time), no per-frame boxes, one crop per truck/camera/visit,
the tracklets that never joined a track in the real run (80%) have no picture and are left out, VLM latency is not modelled.
It estimates the EFFECT of the rules; it is not the authoritative day. Writes only under ~/URCA/IRIS/review/whatif/.
"""
import argparse, gc, collections, concurrent.futures as cf, datetime as dt, hashlib, json, logging, os, pickle, re, sys, time, types
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")                # Re-ID on the CPU: the GPU belongs to the running stack

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
import cv2, numpy as np, yaml
import perception.modules.mtmc.core as _mtmc
from perception.modules.mtmc.core import MTMCConfig, MTMCManager
_mtmc._boxes_overlap = lambda a, b, min_frac=0.5: round((a[2] % 1) * 1e6) == round((b[2] % 1) * 1e6)

ARCHIVE = os.path.expanduser("~/URCA/IRIS/archive/day-export-2026-09-13-ORIGINAL")
LOG = os.path.join(ARCHIVE, "logs/scale-deepstream-sidecar-1.log")
CROPS = os.path.join(ARCHIVE, "data/vehicle_crops")
OFFLINE = os.path.expanduser("~/URCA/IRIS/review/offline-direction/results.jsonl")
OUT = os.path.expanduser("~/URCA/IRIS/review/whatif")
T0 = dt.datetime(2026, 10, 6, 9, 46, 0).timestamp()
FPS = 30                                                       # frame numbers: max_unmatched_frames (500) is counted in frames
LEAD_S = 4.0                                                   # a tracklet is first mentioned about this long after it appeared
DAY_S = 86400
KEEP = ("NEW GLOBAL", "EXITED", "NOT TRACKED", "DIRECTION", "MATCH:", "NOT asked", "ARBITER verdict", "timeout")
GAP_S = 150                                                    # mentions further apart than this are different tracklets


def footage_s(wall_hms):
    h, m, s = map(int, wall_hms.split(":"))
    x = (h * 3600 + m * 60 + s - (6 * 3600 + 46 * 60)) % DAY_S
    return x - DAY_S if x > DAY_S - 1200 else x


# ---------------------------------------------------------------- the day run's tracklets, from its log
RX = {"new": re.compile(r"NEW GLOBAL TRACK: (cam-\d+) local_id=(\d+) -> Global ID (\d+)"),
      "join": re.compile(r"VLM ARBITER MATCH: (cam-\d+) local_id=(\d+) -> Global ID (\d+)"),
      "verdict": re.compile(r"VLM ARBITER verdict=(\w+) conf=([\d.]+) (cam-\d+) local_id=(\d+) vs Global ID (\d+)"),
      "left": re.compile(r"(cam-\d+) local_id=(\d+) \(Global ID (\d+)\) left view"),
      "gone": re.compile(r"INACTIVE TRACK: (cam-\d+) local_id=(\d+) \(Global ID (\d+)\) disappeared"),
      "any": re.compile(r"(cam-\d+) local_id=(\d+)"),
      "exit": re.compile(r"EXITED: Global ID (\d+)")}


def parse_log():
    seg, offset, last = 0, 0, -10 ** 9
    tl, exits, cur = [], [], {}
    for line in open(LOG, errors="replace"):
        if "[vlm_arbiter] started" in line:
            seg += 1
            continue
        if "iris.mtmc" not in line:
            continue
        t = footage_s(line[:8]) + offset
        if t < last - 12 * 3600:                                # past midnight in the log's clock
            offset += DAY_S
            t += DAY_S
        last = t
        m = RX["exit"].search(line)
        if m:
            exits.append((seg, t, int(m[1])))
            continue
        m = RX["any"].search(line)
        if not m:
            continue
        k = (seg, m[1], int(m[2]))
        r = cur.get(k)
        if r is None or t - r["t1"] > GAP_S:                     # local ids are reused after a source reconnect: a new tracklet
            r = {"rid": len(tl), "seg": seg, "cam": m[1], "local": int(m[2]), "t0": t, "t1": t, "gid": None,
                 "end": None, "verdicts": {}, "started": False}
            tl.append(r)
            cur[k] = r
        r["t1"] = t
        for k in ("new", "join", "left", "gone"):
            g = RX[k].search(line)
            if g:
                if k in ("new", "join") and r["gid"] is None:
                    r["gid"] = int(g[3])
                r["started"] |= k == "new"
                if k in ("left", "gone"):
                    r["end"] = t
                    r["gid"] = r["gid"] if r["gid"] is not None else int(g[3])
        v = RX["verdict"].search(line)
        if v:
            r["verdicts"][int(v[5])] = (v[1], float(v[2]))
    return tl, exits


# ---------------------------------------------------------------- crops, offline direction answers
def crop_index():
    """(cam, gid) -> [(footage_s the crop was saved, stay, path)]"""
    idx = collections.defaultdict(list)
    for cam in sorted(os.listdir(CROPS)):
        for fn in os.listdir(os.path.join(CROPS, cam)):
            m = re.fullmatch(r"(\d+)(?:-(\d+))?\.jpg", fn)
            if m:
                p = os.path.join(CROPS, cam, fn)
                idx[(cam, int(m[1]))].append((int(os.path.getmtime(p) - T0), int(m[2] or 0), p))
    return idx


def pick_crop(idx, cam, gid, t0, t1):
    c = idx.get((cam, gid))
    if not c:
        return None
    return min(c, key=lambda x: 0 if t0 <= x[0] <= t1 else min(abs(x[0] - t0), abs(x[0] - t1)))[2]


def load_cache(path):
    out = {}
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l)
            out[r["key"]] = r["v"]
    return out


# ---------------------------------------------------------------- the stand-in for the VLM
class ReplayVlm:
    enabled = True
    min_confidence = 0.5
    max_wait_s = 60.0

    def __init__(self, world):
        self.w = world
        self.results, self.n = {}, 0
        self.stat = collections.Counter()

    def close(self):
        pass

    def take_result(self, job):
        return self.results.pop(job, None)

    def _tracklet(self, cam, oid):
        return self.w.by_oid.get((self.w.seg, cam, oid))

    def submit_direction(self, views, context=None):
        self.n += 1
        t = self._tracklet(context["cam_id"], context["local_id"])
        path = t["crop"] if t else None
        res = {"decision": "unclear", "confidence": None}
        if path:
            key = f"dir|{context['cam_id']}|{path}"
            if key in self.w.dir_answers:
                a = self.w.dir_answers[key]
                res = {"decision": a["orientation"] or "unclear", "confidence": a["confidence"]}
                self.stat["direction_known"] += 1
            else:
                self.w.need_dir[key] = {"gid": t["gid"], "cam": context["cam_id"], "file": os.path.basename(path), "stay": 0, "t": 0, "path": path}
                self.stat["direction_missing"] += 1
        self.results[self.n] = res
        return self.n

    def submit(self, views_a, views_b, context=None):
        self.n += 1
        ctx = context or {}
        t = self._tracklet(ctx.get("cam_id"), ctx.get("local_id"))
        cand = self.w.mgr.association.global_tracks.get(ctx.get("candidate_global_id"))
        lineage = set()
        if cand is not None:
            for cam, oids in cand.local_track_aliases.items():
                for oid in oids:
                    o = self.w.by_oid.get((self.w.seg, cam, oid))
                    if o and o["gid"] is not None:
                        lineage.add(o["gid"])
        res = None
        if t and t["gid"] is not None:
            if t["gid"] in lineage:
                res, why = {"decision": "same", "confidence": 0.95}, "identity"
            else:
                rec = [t["verdicts"][g] for g in lineage if g in t["verdicts"]]
                if rec:
                    yes = [r for r in rec if r[0] == "same"]
                    pick = max(yes or rec, key=lambda r: r[1])
                    res, why = {"decision": pick[0], "confidence": pick[1]}, "recorded"
        if res is None:
            ha, hb = [self.w.orig(v) for v in views_a], [self.w.orig(v) for v in views_b]
            key = self.w.pair_key(ha, hb)
            if key in self.w.cmp_answers:
                res, why = self.w.cmp_answers[key], "cached"
            else:
                self.w.need_cmp[key] = (ha, hb, {k: v for k, v in ctx.items() if k in ("captions_a", "captions_b", "cam_id", "local_id", "candidate_global_id", "candidate_last_camera", "mode", "n_candidates")})
                res, why = {"decision": "uncertain", "confidence": 0.0}, "missing"
        self.stat[why] += 1
        self.results[self.n] = dict(res)
        return self.n


class World:
    def __init__(self):
        self.seg = 0
        self.by_oid = {}
        self.mgr = None
        self.dir_answers, self.cmp_answers = {}, {}
        self.need_dir, self.need_cmp = {}, {}
        self.t2f = {}                                            # digest of the small picture -> digest of the original
        self.reg = {}                                            # md5 of a crop's pixels -> its file (arrays are not kept)

    @staticmethod
    def digest(im):
        return hashlib.md5(np.ascontiguousarray(im).tobytes()).hexdigest()

    def orig(self, im):
        d = self.digest(im)
        return self.t2f.get(d, d)

    def pair_key(self, ha, hb):
        return "cmp|" + hashlib.md5(("|".join(ha) + "#" + "|".join(hb)).encode()).hexdigest()


# ---------------------------------------------------------------- the replay
def make_config(rules, cont, rank="reid"):
    raw = yaml.safe_load(open(os.path.join(REPO, "products/multicam/sidecar.yaml")))["modules"]["mtmc"]["config"]
    raw = {k: v for k, v in raw.items() if k in MTMCConfig.__dataclass_fields__}
    raw.update(enabled=True, reid_enabled=(rank == "reid"), gold_dump_dir=None, reid_max_crops=2)     # all crops of a tracklet are the same picture here
    if rules == "old":                                           # what the day run had: D16 only, nothing required, one ask
        raw.update(direction_cameras={"cam-02": "front_is_exit"}, direction_wait_s=25.0, direction_retry_s=0.0, direction_require=False)
    return MTMCConfig(**raw)


def build_events(tl, idx, lo, hi, cont):
    tr = []
    for r in sorted(tl, key=lambda r: r["t0"]):
        if r["gid"] is None or r["t1"] > DAY_S or r["t0"] < -1200:
            continue
        end = max(r["t1"], r["end"] or 0)
        start = max(r["t0"] - LEAD_S, -1200)
        if start < lo or start > hi:
            continue
        path = pick_crop(idx, r["cam"], r["gid"], start, end)
        if not path:
            continue
        r["crop"], r["start"], r["end_t"] = path, start, max(end, start + 6)
        tr.append(r)
    return tr


def fake_embedding(gid):
    """Stand-in for Re-ID, used only to RANK candidates (the VLM decides): the same old track gets the same vector."""
    v = np.random.default_rng(10_000 + gid).normal(size=64).astype(np.float32)
    return v / np.linalg.norm(v)


def _tag(box_w, gid):
    return box_w + (gid + 1) * 1e-6                               # the old track hides in the box width's fraction


def _same_tag(a, b, min_frac=0.5):
    """Stand-in for the box-overlap test: in the real run two boxes overlapped when the camera's tracker had split one vehicle
    in two. Here every box is the whole crop, so overlap means 'same old track'."""
    return round((a[2] % 1) * 1e6) == round((b[2] % 1) * 1e6)


THUMB = 256                                                      # longest side of the pictures shown to the tracker
EMB = {}                                                          # crop file -> Re-ID embedding (computed once with the real ResNet50)
EMB_PATH = os.path.join(OUT, "emb.pkl")


def make_sync_embedder(w):
    def submit(self):
        for cam in self._cameras.values():
            for tr in cam.all_tracklets():
                if tr.embedding is not None or not tr.embeddable or len(tr.get_crops()) < self.cfg.reid_min_crops:
                    continue
                path = w.reg.get(w.orig(tr.get_crops()[0]))
                if path is None:
                    continue
                if path not in EMB:
                    f = self._reid.embed_crops([cv2.imread(path)])   # the original crop, not the thumbnail
                    EMB[path] = None if f is None else f[0]
                if EMB[path] is not None:
                    tr.embedding = EMB[path]
                    tr.embedding_count += 1
    return submit


def run_once(rules, tr, cont, dir_answers, cmp_answers, log_lines, rank="reid", step=2.0):
    w = World()
    w.dir_answers, w.cmp_answers = dir_answers, cmp_answers
    vlm = ReplayVlm(w)
    cfg = make_config(rules, cont, rank)
    imgs = {}

    def img(p):
        if p not in imgs:
            im = cv2.imread(p)
            fd = World.digest(im)                               # answers are cached under the ORIGINAL picture's digest
            f = THUMB / max(im.shape[:2])
            if f < 1:                                            # the tracker only needs a picture of the right shape: keep it small
                im = cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            imgs[p] = im
            w.reg[fd] = p                                       # fresh VLM calls and Re-ID load the original file
            w.t2f[World.digest(im)] = fd
        return imgs[p]

    class H(logging.Handler):
        def emit(self, rec):
            m = rec.getMessage()
            if "[mtmc]" in m and any(k in m for k in KEEP) and len(log_lines) < 300000:
                log_lines.append(m)
    h = H(level=logging.INFO)
    lg = logging.getLogger()
    lg.addHandler(h)
    old_level = lg.level
    lg.setLevel(logging.INFO)
    next_gid, assign, exits, events = 0, {}, [], []
    stats = collections.Counter()
    # events: (t, order, kind, tracklet)
    ev = []
    for r in tr:
        n = max(int((r["end_t"] - r["start"]) / step), 3)
        for i in range(n + 1):
            ev.append((r["start"] + i * step, 1, "frame", r))
        ev.append((r["end_t"] + 500 / FPS + 2, 2, "beat", r))
    ev.sort(key=lambda e: (e[0], e[1]))
    seg_of = lambda r: 0 if cont else r["seg"]
    cur = None
    oid_counter = collections.Counter()
    oid_of = {}
    for t, _, kind, r in ev:
        s = seg_of(r)
        if s != cur:                                             # a sidecar restart: the in-memory tracks are gone
            if w.mgr is not None:
                next_gid = max(next_gid, w.mgr.association._next_gid)
                stats["restart"] += 1
                w.mgr.close()                                   # stops its embed thread, so its crops can be freed
                w.mgr = None
                gc.collect()
            w.mgr = MTMCManager(cfg, vlm_arbiter=vlm)
            w.mgr.association.start_gid_at(next_gid)
            if rank == "reid":                                    # the real Re-ID model, run synchronously (no thread timing)
                w.mgr._submit_embeddings = types.MethodType(make_sync_embedder(w), w.mgr)
            if rank == "oracle":                                  # rank candidates as Re-ID would: the true track first
                orig = w.mgr.association.get_or_create_global_id

                done_emb = set()                                  # an embedding arrives once per tracklet, not at every frame

                def wrapped(*args, _orig=orig, _w=w, _done=done_emb, **kw):
                    k_ = (_w.seg, kw.get("cam_id"), kw.get("local_id"))
                    t_ = _w.by_oid.get(k_)
                    if t_ is not None and k_ not in _done:
                        kw["embedding"] = fake_embedding(t_["gid"])
                    r_ = _orig(*args, **kw)
                    if isinstance(r_, int) and r_ >= 0:
                        _done.add(k_)
                    return r_
                w.mgr.association.get_or_create_global_id = wrapped
            w.seg, cur = s, s
        key = r["rid"]
        if key not in oid_of:
            oid_counter[0] += 1
            oid_of[key] = oid_counter[0]
            w.by_oid[(s, r["cam"], oid_of[key])] = r
        oid = oid_of[key]
        fno = int(t * FPS)
        if kind == "frame":
            im = img(r["crop"])
            hh, ww = im.shape[:2]
            out = w.mgr.process_frame(r["cam"], fno, float(t), [["vehicle", 0.9, oid]],
                                      boxes=[(0.0, 0.0, _tag(float(ww), r["gid"]), float(hh), "vehicle", 0.9, oid)], frame_bgr=im)
            g = out.get(oid, -1)
            if g is not None and g >= 0:
                assign[key] = g
        else:
            w.mgr.process_frame(r["cam"], fno, float(t), [], boxes=[], frame_bgr=np.zeros((8, 8, 3), np.uint8))
        for e in w.mgr.pop_exit_events():
            exits.append((t, e.get("global_id")))
        for e in w.mgr.pop_direction_events():
            events.append((t, e))
    next_gid = max(next_gid, w.mgr.association._next_gid) if w.mgr else next_gid
    total = dict(w.mgr._direction_total) if w.mgr else {}
    if w.mgr is not None:
        w.mgr.close()
    lg.removeHandler(h)
    lg.setLevel(old_level)
    return {"assign": assign, "exits": exits, "dir_events": events, "need_dir": w.need_dir, "need_cmp": w.need_cmp, "reg": w.reg, "vlm": dict(vlm.stat),
            "stats": dict(stats), "total": total}


def resolve(res, arb, cache_path, workers, budget):
    """Ask the hosted VLM what the replay needed and could not find; append to the cache."""
    from scale_offline_direction import ask
    done = 0
    lock = __import__("threading").Lock()

    def one_dir(item):
        key, c = item
        r = ask(arb, c)
        return key, {"orientation": r["orientation"], "confidence": r["confidence"]}

    def one_cmp(item):
        key, (ha, hb, ctx) = item
        reg = res["reg"]
        v = arb._compare([cv2.imread(reg[h]) for h in ha], [cv2.imread(reg[h]) for h in hb], ctx)
        return key, {"decision": v.get("decision", "uncertain"), "confidence": v.get("confidence")}
    jobs = ([(one_dir, i) for i in res["need_dir"].items()] + [(one_cmp, i) for i in res["need_cmp"].items()])[:budget]
    out = {}
    with cf.ThreadPoolExecutor(workers) as ex, open(cache_path, "a") as f:
        futs = [ex.submit(fn, it) for fn, it in jobs]
        for fu in cf.as_completed(futs):
            try:
                key, v = fu.result()
            except Exception as exc:
                print("  call failed:", str(exc)[:100]); continue
            out[key] = v
            f.write(json.dumps({"key": key, "v": v}, ensure_ascii=False) + "\n")
            f.flush()
            done += 1
            if done % 50 == 0:
                print(f"    {done}/{len(jobs)} answers", flush=True)
    return out


def hms(t):
    return f"{int(t) // 3600:02d}:{int(t) % 3600 // 60:02d}:{int(t) % 60:02d}"


def cmd_run(a):
    tl, exits_old = parse_log()
    idx = crop_index()
    lo = _tsec(a.start) if a.start else -1200
    hi = _tsec(a.end) if a.end else DAY_S
    tr = build_events(tl, idx, lo, hi, a.continuous)
    seen = sum(1 for r in tl.values() if r["start"] if False) if False else 0
    print(f"{len(tl)} tracklets in the log, {len(tr)} replayable (joined a track, have a crop) in {hms(max(lo, 0))}-{hms(min(hi, DAY_S))}; "
          f"{len(set(r['gid'] for r in tr))} old tracks")
    os.makedirs(OUT, exist_ok=True)
    cache_path = os.path.join(OUT, "cache.jsonl")
    cache = load_cache(cache_path)
    dir_answers = {}
    if os.path.exists(OFFLINE):
        for l in open(OFFLINE):
            r = json.loads(l)
            dir_answers[f"dir|{r['cam']}|" + os.path.join(CROPS, r["cam"], r["file"])] = {"orientation": r["orientation"], "confidence": r["confidence"]}
    if os.path.exists(EMB_PATH):
        EMB.update(pickle.load(open(EMB_PATH, "rb")))
    cmp_answers = {k: v for k, v in cache.items() if k.startswith("cmp|")}
    dir_answers.update({k: v for k, v in cache.items() if k.startswith("dir|")})
    arb, budget, spent = None, a.max_calls, 0
    for rnd in range(1, a.rounds + 1):
        log_lines = []
        t0 = time.time()
        res = run_once(a.rules, tr, a.continuous, dir_answers, cmp_answers, log_lines, a.rank, a.step)
        gc.collect()
        pickle.dump(EMB, open(EMB_PATH, "wb"))
        miss = len(res["need_dir"]) + len(res["need_cmp"])
        save_summary(a, tr, res, tl, exits_old, log_lines, lo, hi, rnd, miss)        # usable at any point, after every round
        print(f"round {rnd}: {len(set(res['assign'].values()))} tracks, {len(res['exits'])} exits, VLM {res['vlm']}, missing answers: "
              f"{len(res['need_dir'])} direction + {len(res['need_cmp'])} compare ({time.time() - t0:.0f}s)", flush=True)
        if miss == 0 or a.dry_run or spent >= budget:
            break
        if arb is None:
            from scale_offline_direction import load_env
            load_env()
            from perception.modules.mtmc.vlm_arbiter import VLMArbiter
            arb = VLMArbiter(enabled=True, provider="openai", base_url=os.environ["QWEN_BASE_URL"], model=os.environ["QWEN_MODEL"],
                             api_key_env="QWEN_API_KEY", worker_threads=1, num_predict=400, request_timeout_s=40.0)
        got = resolve(res, arb, cache_path, a.workers, budget - spent)
        spent += len(got)
        for k, v in got.items():
            (cmp_answers if k.startswith("cmp|") else dir_answers)[k] = v
    pickle.dump(EMB, open(EMB_PATH, "wb"))
    print(json.dumps(json.load(open(os.path.join(run_dir(a), "summary.json")))["headline"], indent=1, ensure_ascii=False))


def run_dir(a):
    return os.path.join(OUT, f"{a.rules}-{a.rank}{'-cont' if a.continuous else ''}")


def save_summary(a, tr, res, tl, exits_old, log_lines, lo, hi, rnd, miss):
    summ = summarize(a.rules, a.rank, tr, res, tl, exits_old, log_lines, lo, hi)
    summ["headline"]["rounds_done"], summ["headline"]["answers_still_missing"] = rnd, miss
    d = run_dir(a)
    os.makedirs(d, exist_ok=True)
    json.dump(summ, open(os.path.join(d, "summary.json"), "w"), indent=1, ensure_ascii=False, default=str)
    with open(os.path.join(d, "mtmc.log"), "w") as f:
        f.write("\n".join(log_lines))


def _tsec(s):
    p = [int(x) for x in s.split(":")]
    return p[0] * 3600 + p[1] * 60 + (p[2] if len(p) > 2 else 0)


def summarize(rules, rank, tr, res, tl, exits_old, log_lines, lo, hi):
    assign = res["assign"]
    by_key = {r["rid"]: r for r in tr}
    sim_groups = collections.defaultdict(set)                     # sim gid -> old gids of its tracklets
    old_groups = collections.defaultdict(set)                     # old gid -> sim gids
    n_assigned = 0
    for k, g in assign.items():
        r = by_key.get(k)
        if r is None:
            continue
        n_assigned += 1
        sim_groups[g].add(r["gid"])
        old_groups[r["gid"]].add(g)
    not_tracked = [r for r in tr if r["rid"] not in assign]
    old_tracks = {r["gid"] for r in tr}
    old_started = {r["gid"] for r in tr if r["started"]}
    # pairwise agreement of tracklet grouping (same old gid <=> same sim gid), over assigned tracklets
    pair = collections.Counter()
    items = [(by_key[k]["gid"], g) for k, g in assign.items() if k in by_key]
    byo, bys = collections.Counter(o for o, _ in items), collections.Counter(s for _, s in items)
    both = collections.Counter(items)
    tp = sum(c * (c - 1) // 2 for c in both.values())
    po = sum(c * (c - 1) // 2 for c in byo.values())
    ps = sum(c * (c - 1) // 2 for c in bys.values())
    cap = sum(1 for l in log_lines if "NOT asked (cap" in l)
    exits_old_n = sum(1 for s, t, g in exits_old if lo <= t < hi)
    outcome = {}
    for g in sorted(old_tracks):
        sims = old_groups.get(g, set())
        if not sims:
            outcome[g] = "not tracked"
        elif len(sims) == 1 and len(sim_groups[next(iter(sims))]) == 1:
            outcome[g] = "own track"
        elif len(sims) == 1:
            outcome[g] = "merged with " + ",".join(str(x) for x in sorted(sim_groups[next(iter(sims))] - {g}))
        else:
            outcome[g] = "split into %d tracks" % len(sims)
    head = {"rules": rules, "rank": rank, "window": f"{hms(max(lo, 0))}-{hms(min(hi, DAY_S))}", "tracklets_replayed": len(tr), "old_tracks": len(old_tracks),
            "old_tracks_started_in_window": len(old_started), "sim_tracks": len(sim_groups), "tracklets_without_track": len(not_tracked),
            "exits_old_run": exits_old_n, "exits_sim": len(res["exits"]), "cap_hits_sim": cap,
            "same_track_pairs_agreement": {"precision": round(tp / ps, 3) if ps else None, "recall": round(tp / po, 3) if po else None},
            "direction": res["total"], "vlm": res["vlm"], "restarts": res["stats"].get("restart", 0),
            "outcomes": dict(collections.Counter(v.split(" ")[0] + (" " + v.split(" ")[1] if v.startswith(("own", "not")) else "") for v in outcome.values()))}
    return {"headline": head, "outcome_by_old_gid": outcome,
            "sim_tracks": {int(g): sorted(o) for g, o in sim_groups.items()},
            "not_tracked": [{"gid": r["gid"], "cam": r["cam"], "local": r["local"], "t": hms(r["t0"])} for r in not_tracked]}


def cmd_compare(a):
    for d in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, d, "summary.json")
        if os.path.exists(p):
            h = json.load(open(p))["headline"]
            print(d, json.dumps(h, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--rules", choices=["old", "new"], required=True)
    r.add_argument("--from", dest="start")
    r.add_argument("--to", dest="end")
    r.add_argument("--rank", choices=["reid", "oracle", "time"], default="reid",
                   help="candidate ranking: reid = the real Re-ID model on the saved crops, oracle = true track first (optimistic), time = elapsed time only")
    r.add_argument("--step", type=float, default=2.0, help="seconds between the frames shown for a tracklet")
    r.add_argument("--continuous", action="store_true", help="no sidecar restarts: one manager for the whole window")
    r.add_argument("--dry-run", action="store_true", help="one round, no VLM calls: report how many answers are missing")
    r.add_argument("--max-calls", type=int, default=3000)
    r.add_argument("--rounds", type=int, default=8)
    r.add_argument("--workers", type=int, default=6)
    r.set_defaults(fn=cmd_run)
    c = sub.add_parser("compare")
    c.set_defaults(fn=cmd_compare)
    a = ap.parse_args()
    a.fn(a)
