#!/usr/bin/env python3
"""Offline pass: judge the saved entrance crops of the day's trucks with the VLM and apply the new start rule.

The new rule (perception/modules/mtmc/core, direction_require): a track starts only for a vehicle decided as ENTERING. D41 (cam-01) is
a side view (the VLM says which way the front points: left/toward = entering, right/away = exiting); D16 (cam-02) is
front/rear (front = leaving). This asks that question again, on the single best crop the run saved, for every D41/D16
crop of the trucks in the platform database, and sorts each truck by what its start crop says:

  keep               the crop a track started from is decided ENTERING
  would_not_start    that crop is decided EXITING and nothing says it ever entered: a leaving vehicle that started a track
  exit_first         that crop is EXITING but a later stay (it came back) is ENTERING: a real truck, its first visit was a leave
  disputed           that crop is EXITING but the other entrance camera's crop of the same stay says ENTERING
  undecided          still unclear after the retries: under the new rule no track
  not_vehicle        the picture holds no vehicle at all ("none" answer): never a track
  no_crop            no entrance crop was saved

  scale_offline_direction.py run [--gids 49,13] [--workers 4]    ask the VLM (resumable; results.jsonl)
  scale_offline_direction.py report                              report.csv + index.html from the results

Nothing in the database or the archive is changed. Results and the report go to ~/URCA/IRIS/review/offline-direction/.
Single crops carry no movement, so D41 is judged by where the front points only (the live sidecar also checks movement).
"""
import argparse, collections, concurrent.futures as cf, csv, datetime as dt, html, json, os, re, shutil, subprocess, sys, threading

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
ARCHIVE = os.path.expanduser("~/URCA/IRIS/archive/day-export-2026-09-13-ORIGINAL")
OUT = os.path.expanduser("~/URCA/IRIS/review/offline-direction")
ENV = os.path.expanduser("~/URCA/IRIS/scale-local/.env")
PG = os.environ.get("PG_CONTAINER", "scale-ui-postgres-1")
T0 = dt.datetime(2026, 10, 6, 9, 46, 0).timestamp()            # the wall time footage 00:00 played (manifest footage_began_wall)
CAMS = {"cam-01": ("D41", "side_right_to_left_enters"), "cam-02": ("D16", "front_is_exit")}
SIZES = (1024, 640, 448)                                       # a different size is a real retry (same size repeats the answer)
MIN_CONF = 0.7
LIVE_LOG = os.path.join(ARCHIVE, "logs/scale-deepstream-sidecar-1.log")
PREFIX = "ااا-"


def load_env():
    for line in open(ENV):
        k, _, v = line.strip().partition("=")
        if k.startswith("QWEN_"):
            os.environ.setdefault(k, v.strip().strip('"').split(" #")[0].strip())


def psql(sql):
    r = subprocess.run(["docker", "exec", "-i", PG, "psql", "-U", "iris", "-d", "iris", "-At", "-F", ","], input=sql,
                       capture_output=True, text=True, check=True)
    return [l.split(",") for l in r.stdout.splitlines()]


def decide(cam, orientation, conf):
    """front/rear (D16) or left/right/toward/away (D41) -> entering | exiting | not_vehicle (no vehicle in the picture) | None (not decided)."""
    if orientation is None or (conf is not None and conf < MIN_CONF):
        return None
    if orientation == "none":
        return "not_vehicle"
    if cam == "cam-01":
        return {"left": "entering", "toward": "entering", "right": "exiting", "away": "exiting"}.get(orientation)
    return {"rear": "entering", "front": "exiting"}.get(orientation)


def crops_of(gid):
    out = []
    for cam in CAMS:
        d = os.path.join(ARCHIVE, "data/vehicle_crops", cam)
        for f in sorted(os.listdir(d)):
            m = re.fullmatch(rf"{gid}(?:-(\d+))?\.jpg", f)
            if m:
                out.append({"gid": gid, "cam": cam, "file": f, "stay": int(m[1] or 0), "path": os.path.join(d, f),
                            "t": int(os.path.getmtime(os.path.join(d, f)) - T0)})
    return out


def ask(arb, c):
    import cv2
    from perception.modules.mtmc.vlm_arbiter import DIRECTION_PROMPT, DIRECTION_PROMPT_SIDE, _parse_direction, _to_b64
    side = CAMS[c["cam"]][1] == "side_right_to_left_enters"
    img = cv2.imread(c["path"])
    last = None
    for size in SIZES:
        try:
            v = _parse_direction(arb._call(DIRECTION_PROMPT_SIDE if side else DIRECTION_PROMPT, [_to_b64(img, size)]), side=side)
        except Exception as exc:                                  # a failed call counts as an unclear try
            last = {"orientation": None, "confidence": None, "evidence": [f"error: {str(exc)[:80]}"], "size": size}
            continue
        if v is None:
            last = {"orientation": None, "confidence": None, "evidence": ["unparseable"], "size": size}
            continue
        last = {"orientation": v["decision"], "confidence": v["confidence"], "evidence": v["evidence"], "size": size}
        if decide(c["cam"], v["decision"], v["confidence"]):
            break
    return {k: c[k] for k in ("gid", "cam", "file", "stay", "t")} | last | {
        "direction": decide(c["cam"], last["orientation"], last["confidence"])}


def run(gids, workers):
    load_env()
    from perception.modules.mtmc.vlm_arbiter import VLMArbiter
    arb = VLMArbiter(enabled=True, provider="openai", base_url=os.environ["QWEN_BASE_URL"], model=os.environ["QWEN_MODEL"],
                     api_key_env="QWEN_API_KEY", worker_threads=1, num_predict=400, request_timeout_s=40.0)
    if not arb.enabled:
        sys.exit("the VLM client did not start (QWEN_API_KEY / QWEN_BASE_URL / QWEN_MODEL in scale-local/.env?)")
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "results.jsonl")
    done = {(r["cam"], r["file"]) for r in map(json.loads, open(path))} if os.path.exists(path) else set()
    todo = [c for g in gids for c in crops_of(g) if (c["cam"], c["file"]) not in done]
    print(f"{len(gids)} trucks, {len(todo)} crops to ask ({len(done)} already done)")
    lock, n = threading.Lock(), [0]

    def work(c):
        r = ask(arb, c)
        with lock:
            open(path, "a").write(json.dumps(r, ensure_ascii=False) + "\n")
            n[0] += 1
            if n[0] % 25 == 0 or n[0] == len(todo):
                print(f"  {n[0]}/{len(todo)}", flush=True)
    with cf.ThreadPoolExecutor(workers) as ex:
        list(ex.map(work, todo))


def start_cameras():
    """gid -> the entrance camera its track started on (the sidecar log, 'NEW GLOBAL TRACK')."""
    out = {}
    for l in open(LIVE_LOG, errors="replace"):
        m = re.search(r"NEW GLOBAL TRACK: (cam-\d+) local_id=\d+ -> Global ID (\d+)", l)
        if m:
            out[int(m[2])] = m[1]
    return out


def live_verdicts():
    """gid -> set of directions the live run decided (all on D16)."""
    out = collections.defaultdict(set)
    for l in open(LIVE_LOG, errors="replace"):
        m = re.search(r"direction: gid (\d+) on (cam-\d+) is (entering|exiting)", l)
        if m:
            out[int(m[1])].add(m[3])
    return out


def report():
    rows = psql("select external_vehicle_id, id from trucks where created_at < '2026-10-07 13:00' order by 1")
    uuid = {int(e.replace(PREFIX, "")): u for e, u in rows}
    res = collections.defaultdict(list)
    for r in map(json.loads, open(os.path.join(OUT, "results.jsonl"))):
        res[r["gid"]].append(r)
    start, live = start_cameras(), live_verdicts()
    os.makedirs(os.path.join(OUT, "images"), exist_ok=True)
    cards, table, counts, agree = [], [], collections.Counter(), collections.Counter()
    for gid in sorted(uuid):
        rs = sorted(res.get(gid, []), key=lambda r: (r["t"], r["cam"]))
        sc = start.get(gid)
        key = next((r for r in rs if r["cam"] == sc and r["stay"] == 0), None) or next((r for r in rs if r["stay"] == 0), None)
        cls = "no_crop" if key is None else {"entering": "keep", "exiting": "would_not_start", "not_vehicle": "not_vehicle", None: "undecided"}[key["direction"]]
        if cls == "would_not_start":
            if any(r["stay"] > key["stay"] and r["direction"] == "entering" for r in rs):
                cls = "exit_first"
            elif any(r["stay"] == key["stay"] and r["cam"] != key["cam"] and r["direction"] == "entering" for r in rs):
                cls = "disputed"
        counts[cls] += 1
        for r in rs:                                            # offline vs live, on D16 crops with a live verdict
            if r["cam"] == "cam-02" and live.get(gid) and r["direction"]:
                agree["same" if r["direction"] in live[gid] else "different"] += 1
        lv = "/".join(sorted(live[gid])) if live.get(gid) else "none"
        imgs = ""
        for r in rs:
            n = f"{gid:04d}_{r['cam']}_{r['file']}"
            shutil.copyfile(os.path.join(ARCHIVE, "data/vehicle_crops", r["cam"], r["file"]), os.path.join(OUT, "images", n))
            tag = (r["direction"] or "undecided") + f" ({r['orientation']}, {r['confidence']})"
            mark = " ★ start" if r is key else ""
            imgs += (f'<figure><img src="images/{n}" loading=lazy><figcaption>{CAMS[r["cam"]][0]} stay {r["stay"]} · '
                     f'{r["t"] // 3600:02d}:{r["t"] % 3600 // 60:02d}:{r["t"] % 60:02d} · <b>{html.escape(tag)}</b>{mark}</figcaption></figure>')
            table.append([gid, f"{PREFIX}{gid:04d}", cls, CAMS[r["cam"]][0], r["stay"], r["t"], r["orientation"], r["confidence"],
                          r["direction"], lv, "; ".join(r["evidence"])])
        t0 = key["t"] if key else (rs[0]["t"] if rs else 0)
        cards.append((t0, f'<div class="c {cls}"><div class=h><label style="float:right"><input type=checkbox class=sel data-gid="{gid}" '
                      f'onchange="u()"> false positive</label><a style="color:#8ab4ff" target=_blank href="https://localhost:35444/trucks/{uuid[gid]}">'
                      f'{PREFIX}{gid:04d}</a> · start on {CAMS.get(start.get(gid), ("?",))[0]}</div>{imgs}'
                      f'<div class=m><b>{cls.replace("_", " ")}</b> · live verdict: {lv}</div></div>'))
    with open(os.path.join(OUT, "report.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gid", "truck", "class", "camera", "stay", "footage_s", "vlm_orientation", "confidence", "direction", "live_verdict", "evidence"])
        w.writerows(table)
    btn = "".join(f"<button onclick=\"f('{k}')\">{k.replace('_', ' ')} ({v})</button> " for k, v in counts.items())
    page = ('<meta charset=utf-8><title>Offline VLM direction pass</title><style>body{font:14px sans-serif;background:#111;color:#eee;margin:16px}'
            '.g{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}.c{background:#1c1c1c;padding:8px;border-radius:6px}'
            '.h{font-weight:bold;margin-bottom:4px}figure{margin:0 0 6px}img{width:100%;display:block}figcaption,.m{font-size:12px;color:#aaa}'
            'button{margin:2px 4px 8px 0;padding:6px 10px}.would_not_start{outline:2px solid #ff9a6b}.exit_first,.disputed{outline:2px solid #6bb5ff}.undecided{outline:2px solid #e6c34d}.not_vehicle{outline:2px solid #c77dff}</style>'
            f'<h2>{len(uuid)} trucks: the new start rule applied to the saved entrance crops (footage clock)</h2>'
            '<div id=bar style="position:sticky;top:0;background:#222;padding:8px;margin-bottom:10px;z-index:5"><b id=n>0 selected</b> '
            '<button onclick="dl()">Download selection (selection.json)</button> <button onclick="clr()">Clear</button></div>'
            f'<button onclick="f(\'\')">all</button> {btn}<div class=g>' + "".join(c for _, c in sorted(cards, key=lambda x: x[0])) +
            '</div><script>var K="offl_sel";function sel(){return [...document.querySelectorAll(".sel:checked")].map(x=>+x.dataset.gid)}'
            'function u(){var s=sel();try{localStorage.setItem(K,JSON.stringify(s))}catch(e){}document.getElementById("n").textContent=s.length+" selected"}'
            'function dl(){var b=new Blob([JSON.stringify({gids:sel()})],{type:"application/json"}),a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="selection.json";a.click()}'
            'function clr(){document.querySelectorAll(".sel").forEach(x=>x.checked=false);u()}'
            'function f(k){document.querySelectorAll(".c").forEach(c=>c.style.display=!k||c.classList.contains(k)?"":"none")}'
            'try{JSON.parse(localStorage.getItem(K)||"[]").forEach(g=>{var x=document.querySelector(\'.sel[data-gid="\'+g+\'"]\');if(x)x.checked=true})}catch(e){}u()</script>')
    open(os.path.join(OUT, "index.html"), "w").write(page)
    print(dict(counts), "total", sum(counts.values()), "| offline vs live on D16 crops:", dict(agree))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--gids", help="comma list (default: every truck in the database)")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "run":
        gids = ([int(g) for g in a.gids.split(",")] if a.gids else
                sorted(int(e.replace(PREFIX, "")) for (e, _) in psql("select external_vehicle_id, id from trucks where created_at < '2026-10-07 13:00'")))
        run(gids, a.workers)
    else:
        report()
