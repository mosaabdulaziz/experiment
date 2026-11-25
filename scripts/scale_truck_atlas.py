#!/usr/bin/env python3
"""One page with every day truck and everything known about it across all 15 cameras, to review the associations at once.

  scale_truck_atlas.py [--out DIR]        default ~/URCA/IRIS/review/atlas-day-v1/  (index.html, atlas.csv, data.json, images/)

Per truck: a timeline row per camera (first to last sighting, footage clock), the saved crops per camera and visit, the
chain of association events from the run's sidecar log (new track, VLM "same" joins with confidence, the numbers of
"different"/"uncertain" answers, direction verdicts, exits), the plate, the visits at stops, the offline direction class
(scripts/scale_offline_direction.py), flags for suspicious cases, a link to the live profile and a pointer into the
recordings (file + offset) for every camera visit. Filters, search and a "false positive" selection (selection.json, for
scripts/scale_remove_trucks.py) work in the page. Read-only: the database, the archive and the recordings are not touched.
Footage clock: 00:00 = the start of the recorded day. Sightings are read from the platform database (imported day, shifted
to 2026-10-06 00:00 Cairo); the sidecar log's wall times are UTC and run 06:46 ahead of the footage clock.
"""
import argparse, collections, csv, datetime as dt, html, json, os, re, shutil, subprocess, sys

ARCHIVE = os.path.expanduser("~/URCA/IRIS/archive/day-export-2026-09-13-ORIGINAL")
OFFLINE = os.path.expanduser("~/URCA/IRIS/review/offline-direction")
PG = os.environ.get("PG_CONTAINER", "scale-ui-postgres-1")
LOG = os.path.join(ARCHIVE, "logs/scale-deepstream-sidecar-1.log")
T0 = dt.datetime(2026, 10, 6, 9, 46, 0).timestamp()            # wall time of footage 00:00 (crop file times)
CUTOFF = "2026-10-07 13:00"                                    # trucks created before this are the imported day
DAY0 = "2026-10-06 00:00:00+03"                                # shifted day start (Cairo)
PREFIX = "ااا-"
VISIT_GAP_S = 90                                               # sightings further apart than this are another visit
CAMS = {"cam-01": "D41", "cam-02": "D16", "cam-03": "D52", "cam-04": "CH13", "cam-05": "CH22", "cam-06": "CH23",
        "cam-07": "CH06", "cam-08": "CH30", "cam-09": "CH24", "cam-10": "D15", "cam-11": "D18", "cam-12": "D31",
        "cam-13": "D28", "cam-14": "D05", "cam-15": "D20"}
BYNAME = {v: k for k, v in CAMS.items()}
ENTRANCE = {"cam-01", "cam-02"}


def psql(sql):
    r = subprocess.run(["docker", "exec", "-i", PG, "psql", "-U", "iris", "-d", "iris", "-qAt", "-F", "\t"], input=sql,
                       capture_output=True, text=True, check=True)
    return [l.split("\t") for l in r.stdout.splitlines()]


def hms(s):
    s = int(s)
    sign = "-" if s < 0 else ""
    s = abs(s)
    return f"{sign}{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def footage_s(wall_hms):
    """The sidecar log's UTC wall time -> footage seconds (06:46 ahead; before the start it is negative)."""
    h, m, s = map(int, wall_hms.split(":"))
    x = (h * 3600 + m * 60 + s - (6 * 3600 + 46 * 60)) % 86400
    return x - 86400 if x > 86400 - 1200 else x


def load_db():
    trucks = {}
    for e, u, pl, pn, pc in psql(f"select external_vehicle_id, id, coalesce(plate_letters,''), coalesce(plate_numbers,''), "
                                 f"coalesce(plate_confidence::text,'') from trucks where created_at < '{CUTOFF}'"):
        trucks[int(e.replace(PREFIX, ""))] = {"gid": int(e.replace(PREFIX, "")), "uuid": u, "plate": "" if re.fullmatch(r"ا+", pl) else f"{pl} {pn}".strip(),     # ااا-NNNN is the placeholder of an unread plate
                                              "plate_conf": pc, "sightings": collections.defaultdict(list), "visits": []}
    by_uuid = {t["uuid"]: t for t in trucks.values()}
    for u, cam, sec in psql(f"select s.truck_id, c.name, extract(epoch from (s.seen_at - timestamptz '{DAY0}'))::int from sightings s "
                            f"join cameras c on c.id = s.camera_id join trucks t on t.id = s.truck_id where t.created_at < '{CUTOFF}'"):
        by_uuid[u]["sightings"][BYNAME.get(cam.replace("CAM-", ""), cam)].append(int(sec))
    for u, stop, a, b, why in psql(f"select v.truck_id, s.name->>'en', extract(epoch from (v.started_at - timestamptz '{DAY0}'))::int, "
                                   f"extract(epoch from (coalesce(v.ended_at, v.last_seen_at) - timestamptz '{DAY0}'))::int, coalesce(v.end_reason::text,'') "
                                   f"from visits v join stops s on s.id = v.stop_id join trucks t on t.id = v.truck_id where t.created_at < '{CUTOFF}' order by 3"):
        by_uuid[u]["visits"].append({"stop": stop, "a": int(a), "b": int(b), "why": why})
    return trucks


def parse_log(gids):
    ev = collections.defaultdict(list)       # gid -> [(t, kind, cam, text)]
    rx = [(re.compile(r"NEW GLOBAL TRACK: (cam-\d+) local_id=(\d+) -> Global ID (\d+)"), "new", (1, 2, 3)),
          (re.compile(r"VLM ARBITER MATCH: (cam-\d+) local_id=(\d+) -> Global ID (\d+) \(conf=([\d.]+)\)"), "join", (1, 2, 3, 4)),
          (re.compile(r"VLM ARBITER verdict=(different|uncertain) conf=([\d.]+) (cam-\d+) local_id=(\d+) vs Global ID (\d+)"), "no", None),
          (re.compile(r"direction: gid (\d+) on (cam-\d+) is (entering|exiting) \(.*conf ([\d.]+)\)"), "dir", None),
          (re.compile(r"EXITED: Global ID (\d+) left the premises \(last camera (cam-\d+)\)"), "exit", None),
          (re.compile(r"(cam-\d+) local_id=(\d+) \(Global ID (\d+)\) left view"), "left", (1, 2, 3)),
          (re.compile(r"INACTIVE TRACK: (cam-\d+) local_id=(\d+) \(Global ID (\d+)\) disappeared"), "gone", (1, 2, 3))]
    with open(LOG, errors="replace") as f:
        for line in f:
            if "Global ID" not in line and "direction: gid" not in line:
                continue
            t = footage_s(line[:8])
            for r, kind, idx in rx:
                m = r.search(line)
                if not m:
                    continue
                if kind == "new":
                    cam, loc, g = m[1], m[2], int(m[3]); ev[g].append((t, kind, cam, f"new track (local {loc})"))
                elif kind == "join":
                    cam, loc, g = m[1], m[2], int(m[3]); ev[g].append((t, kind, cam, f"VLM same {m[4]} -> joined (local {loc})"))
                elif kind == "no":
                    ev[int(m[5])].append((t, "no", m[3], f"VLM {m[1]} {m[2]} (local {m[4]})"))
                elif kind == "dir":
                    ev[int(m[1])].append((t, "dir", m[2], f"direction {m[3]} ({m[4]})"))
                elif kind == "exit":
                    ev[int(m[1])].append((t, "exit", m[2], "EXITED the premises"))
                elif kind == "left":
                    ev[int(m[3])].append((t, "left", m[1], f"left view (local {m[2]})"))
                else:
                    ev[int(m[3])].append((t, "gone", m[1], f"track inactive (local {m[2]})"))
                break
    return {g: sorted(v) for g, v in ev.items() if g in gids}


def load_files():
    """cam -> sorted [(start second, file name)] from the player's file lists."""
    out = {}
    for cam, name in CAMS.items():
        p = os.path.join(ARCHIVE, "data/day-player", f"{cam}.txt")
        lst = []
        if os.path.exists(p):
            for l in open(p):
                m = re.search(r"/([A-Za-z0-9]+_20260913(\d{2})(\d{2})(\d{2})\.mp4)", l)
                if m:
                    lst.append((int(m[2]) * 3600 + int(m[3]) * 60 + int(m[4]), m[1]))
        out[cam] = sorted(lst)
    return out


def pointer(files, cam, t):
    c = [x for x in files.get(cam, []) if x[0] <= t]
    return f"{c[-1][1]} @ {hms(t - c[-1][0])}" if c else ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.expanduser("~/URCA/IRIS/review/atlas-day-v1"))
    out = ap.parse_args().out
    trucks = load_db()
    gids = set(trucks)
    print(f"{len(trucks)} day trucks; reading the sidecar log ...", flush=True)
    ev = parse_log(gids)
    files = load_files()
    offline = {}
    if os.path.exists(os.path.join(OFFLINE, "report.csv")):
        for r in csv.DictReader(open(os.path.join(OFFLINE, "report.csv"), encoding="utf-8-sig")):
            offline[int(r["gid"])] = r["class"]
    ores = collections.defaultdict(dict)
    if os.path.exists(os.path.join(OFFLINE, "results.jsonl")):
        for r in map(json.loads, open(os.path.join(OFFLINE, "results.jsonl"))):
            ores[r["gid"]][(r["cam"], r["file"])] = r
    stop_of = {}                                                 # camera -> stop name (cameras of one stop see the same area)
    for cam, stop in psql("select c.name, s.name->>'en' from cameras c join stops s on s.id = c.stop_id"):
        stop_of[BYNAME.get(cam.replace("CAM-", ""), cam)] = stop
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    os.system(f"chmod -R u+w '{out}/images' 2>/dev/null")

    cards, rows, data = [], [], []
    for gid in sorted(trucks, key=lambda g: min((min(v) for v in trucks[g]["sightings"].values()), default=0)):
        tr = trucks[gid]
        visits = []                                              # (cam, start, end, count)
        for cam, secs in tr["sightings"].items():
            secs.sort()
            a = prev = secs[0]
            n = 1
            for s in secs[1:]:
                if s - prev > VISIT_GAP_S:
                    visits.append((cam, a, prev, n)); a, n = s, 0
                prev = s; n += 1
            visits.append((cam, a, prev, n))
        visits.sort(key=lambda v: v[1])
        t_min = min((v[1] for v in visits), default=0)
        t_max = max((v[2] for v in visits), default=t_min + 1)
        span = max(t_max - t_min, 1)
        events = ev.get(gid, [])
        joins = [float(e[3].split()[2]) for e in events if e[1] == "join"]
        no = collections.Counter(e[3].split()[1] for e in events if e[1] == "no")
        live_dirs = sorted({e[3].split()[1] for e in events if e[1] == "dir"})
        cams = sorted({v[0] for v in visits})
        cls = offline.get(gid, "")
        flags = []
        if joins and min(joins) < 0.9: flags.append("lowconf")
        if no["different"] + no["uncertain"] >= 10: flags.append("manyno")
        if len(live_dirs) > 1 or (cls == "disputed"): flags.append("dirconflict")
        if any(e[1] == "exit" for e in events): flags.append("exited")
        ent = sorted(v[1] for v in visits if v[0] in ENTRANCE)            # D41 and D16 see one pass at once: count passes
        if sum(1 for i, x in enumerate(ent) if i == 0 or x - ent[i - 1] > 300) >= 2: flags.append("entrances2")
        if tr["plate"] and tr["plate"].replace(" ", ""): flags.append("plate")
        # simultaneous on two cameras of different areas = a link that cannot be one vehicle (3 s tolerance)
        pts = sorted((s, c) for c, ss in tr["sightings"].items() for s in ss)
        for i, (s, c) in enumerate(pts):
            if any(s2 - s <= 3 and c2 != c and stop_of.get(c2) != stop_of.get(c) and not ({c, c2} <= ENTRANCE) for s2, c2 in pts[i + 1:i + 12]):
                flags.append("overlap"); break
        if any(b - a_ > 90 * 60 for (_, _, a_), (_, b, _) in zip([(0, 0, v[2]) for v in visits[:-1]], [(0, v[1], 0) for v in visits[1:]])):
            flags.append("longgap")
        if len(cams) >= 6: flags.append("manycams")
        # timeline rows
        rows_html = ""
        for cam in cams:
            segs = "".join(f'<i title="{CAMS[cam]} {hms(a)}-{hms(b)} ({n} sightings)" style="left:{(a - t_min) / span * 100:.2f}%;width:{max((b - a) / span * 100, 0.6):.2f}%"></i>'
                           for c, a, b, n in visits if c == cam)
            rows_html += f'<div class=tl><span>{CAMS[cam]}</span><b>{segs}</b></div>'
        # crops
        crops, cimgs = [], ""
        for cam in CAMS:
            d = os.path.join(ARCHIVE, "data/vehicle_crops", cam)
            for fn in sorted(os.listdir(d)):
                m = re.fullmatch(rf"{gid}(?:-(\d+))?\.jpg", fn)
                if m:
                    p = os.path.join(d, fn)
                    crops.append((int(os.path.getmtime(p) - T0), cam, fn, int(m[1] or 0), p))
        for t, cam, fn, stay, p in sorted(crops):
            name = f"{gid:04d}_{cam}_{fn}"
            shutil.copyfile(p, os.path.join(out, "images", name))
            r = ores.get(gid, {}).get((cam, fn))
            off = f' · offline <b>{html.escape(str(r["direction"] or "undecided"))}</b> ({html.escape(str(r["orientation"]))})' if r else ""
            ptr = pointer(files, cam, t)
            cimgs += (f'<figure><img src="images/{name}" loading=lazy><figcaption>{CAMS[cam]} stay {stay} · {hms(t)}{off}'
                      f'<br><small>{html.escape(ptr)}</small></figcaption></figure>')
        # association chain
        chain = "".join(f'<li class="{k}"><time>{hms(t)}</time> {CAMS.get(c, c)} · {html.escape(txt)}</li>'
                        for t, k, c, txt in events if k in ("new", "join", "dir", "exit") or (k == "no" and False))
        chain += f'<li class=sum>VLM answers against other trucks: different {no["different"]}, uncertain {no["uncertain"]}</li>'
        stops = " → ".join(f'{html.escape(v["stop"])} {hms(v["a"])}' for v in tr["visits"][:14]) or "no visits"
        plate = html.escape(tr["plate"]) if tr["plate"] else "no plate read"
        ftxt = " ".join(flags)
        cards.append((t_min, f'<div class="c {cls}" data-gid="{gid}" data-cls="{cls}" data-flags="{ftxt}" data-ncams="{len(cams)}" data-plate="{plate}">'
                      f'<div class=h><label style="float:right"><input type=checkbox class=sel data-gid="{gid}" onchange="u()"> false positive</label>'
                      f'<a style="color:#8ab4ff" target=_blank href="https://localhost:35444/trucks/{tr["uuid"]}">{PREFIX}{gid:04d}</a> · {hms(t_min)}–{hms(t_max)} · {plate}</div>'
                      f'<div class=chips>{"".join(f"<span class={x}>{x}</span>" for x in ([cls] if cls else []) + flags)}</div>'
                      f'{rows_html}<div class=crops>{cimgs}</div><ul>{chain}</ul><div class=m>stops: {stops}</div></div>'))
        for c, a, b, n in visits:
            rows.append([gid, f"{PREFIX}{gid:04d}", CAMS[c], hms(a), hms(b), n, cls, " ".join(flags), pointer(files, c, a),
                         ";".join(x[2] for x in crops if x[1] == c)])
        data.append({"gid": gid, "uuid": tr["uuid"], "plate": tr["plate"], "class": cls, "flags": flags, "cameras": [CAMS[c] for c in cams],
                     "visits": [{"cam": CAMS[c], "from": hms(a), "to": hms(b), "n": n, "recording": pointer(files, c, a)} for c, a, b, n in visits],
                     "events": [{"t": hms(t), "kind": k, "cam": CAMS.get(c, c), "text": x} for t, k, c, x in events]})
    json.dump(data, open(os.path.join(out, "data.json"), "w"), ensure_ascii=False)
    with open(os.path.join(out, "atlas.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gid", "truck", "camera", "visit_from", "visit_to", "sightings", "offline_class", "flags", "recording_at_start", "crop_files"])
        w.writerows(rows)
    per_hour = collections.Counter(t // 3600 for t, _ in cards)
    classes = collections.Counter(d["class"] or "-" for d in data)
    flagc = collections.Counter(f for d in data for f in d["flags"])
    chips = "".join(f'<button onclick="f(\'cls\',\'{k}\')">{k} ({v})</button> ' for k, v in classes.items() if k != "-")
    fchips = "".join(f'<button onclick="f(\'flag\',\'{k}\')">{k} ({v})</button> ' for k, v in sorted(flagc.items()))
    hours = "".join(f'<span title="{h:02d}:00 {per_hour.get(h, 0)} trucks"><i style="height:{per_hour.get(h, 0) * 4}px"></i>{h}</span>' for h in range(24))
    page = f"""<meta charset=utf-8><title>Truck atlas: all cameras</title><style>
body{{font:14px sans-serif;background:#111;color:#eee;margin:16px}}.g{{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:12px}}
.c{{background:#1c1c1c;padding:8px;border-radius:6px}}.h{{font-weight:bold;margin-bottom:4px}}.chips span{{display:inline-block;background:#333;border-radius:9px;padding:1px 8px;margin:0 4px 4px 0;font-size:11px}}
.tl{{display:flex;align-items:center;font-size:11px;color:#aaa;margin:1px 0}}.tl span{{width:44px}}.tl b{{position:relative;flex:1;height:10px;background:#2a2a2a;border-radius:3px}}
.tl i{{position:absolute;top:0;height:10px;background:#4a9eff;border-radius:3px}}.crops{{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0}}figure{{margin:0;width:128px}}
figure img{{width:100%;display:block}}figcaption{{font-size:10px;color:#aaa}}ul{{margin:4px 0;padding-left:16px;font-size:12px;color:#ccc}}li.join{{color:#8fd18f}}li.exit{{color:#ff9a6b}}li.sum{{color:#999}}
time{{color:#888}}.m{{font-size:11px;color:#888}}button{{margin:2px 4px 6px 0;padding:5px 9px}}.would_not_start{{outline:2px solid #ff9a6b}}.exit_first,.disputed{{outline:2px solid #6bb5ff}}.undecided{{outline:2px solid #e6c34d}}
.hrs{{display:flex;gap:3px;align-items:flex-end;height:70px;margin:6px 0}}.hrs span{{font-size:10px;color:#888;text-align:center;width:22px}}.hrs i{{display:block;background:#4a9eff;margin-bottom:2px}}
</style><h2>{len(data)} day trucks across all cameras (footage clock, 00:00 = start of the recorded day)</h2>
<div class=hrs>{hours}</div>
<div id=bar style="position:sticky;top:0;background:#222;padding:8px;margin-bottom:10px;z-index:5"><b id=n>0 selected</b>
<button onclick="dl()">Download selection (selection.json)</button> <button onclick="clr()">Clear</button>
<input id=q placeholder="gid or plate" oninput="s()" style="padding:5px"> shown: <b id=shown></b></div>
<div>offline class: <button onclick="f('cls','')">all</button> {chips}</div><div>flags (all selected must apply): {fchips} <button onclick="fl=[];ap()">clear flags</button></div>
<div style="font-size:12px;color:#999;margin:6px 0">lowconf = a VLM join below 0.9 · manyno = 10+ "different/uncertain" answers · dirconflict = direction verdicts disagree · overlap = seen on two areas at once · longgap = 90+ min between visits · entrances2 = two separate entrance passes (5+ min apart)</div>
<div class=g>{''.join(c for _, c in sorted(cards, key=lambda x: x[0]))}</div><script>
var K="atlas_sel",cl="",fl=[];function sel(){{return [...document.querySelectorAll(".sel:checked")].map(x=>+x.dataset.gid)}}
function u(){{var s=sel();try{{localStorage.setItem(K,JSON.stringify(s))}}catch(e){{}}document.getElementById("n").textContent=s.length+" selected"}}
function dl(){{var b=new Blob([JSON.stringify({{gids:sel()}})],{{type:"application/json"}}),a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="selection.json";a.click()}}
function clr(){{document.querySelectorAll(".sel").forEach(x=>x.checked=false);u()}}
function f(kind,v){{if(kind=="cls")cl=v;else if(!fl.includes(v))fl.push(v);ap()}}
function s(){{ap()}}
function ap(){{var q=document.getElementById("q").value.trim().toLowerCase(),n=0;document.querySelectorAll(".c").forEach(c=>{{var ok=(!cl||c.dataset.cls==cl)&&fl.every(x=>(" "+c.dataset.flags+" ").includes(" "+x+" "))&&(!q||c.dataset.gid==q.replace(/^0+/,"")||(c.dataset.plate||"").toLowerCase().includes(q));c.style.display=ok?"":"none";if(ok)n++}});document.getElementById("shown").textContent=n}}
try{{JSON.parse(localStorage.getItem(K)||"[]").forEach(g=>{{var x=document.querySelector('.sel[data-gid="'+g+'"]');if(x)x.checked=true}})}}catch(e){{}}u();ap()</script>"""
    open(os.path.join(out, "index.html"), "w").write(page)
    print(f"wrote {out}/index.html: {len(data)} trucks, {len(rows)} camera visits, flags {dict(flagc)}")


if __name__ == "__main__":
    main()
