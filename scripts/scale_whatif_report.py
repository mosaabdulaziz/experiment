#!/usr/bin/env python3
"""A page that puts the what-if replay next to the offline direction pass, one card per old track (truck).

  scale_whatif_report.py           ~/URCA/IRIS/review/whatif/index.html (+ images/, report.csv)

Reads the summaries written by scripts/scale_whatif_replay.py (old-reid/ and new-reid/), the offline direction classes
(review/offline-direction/report.csv), the list of trucks you removed by hand and the platform database (profile links).
Each card shows the first entrance crop, the offline class, what the old and the new rules do with that truck in the replay
(own track / merged with other trucks / split / not tracked) and flags the disagreements to look at first. A "false positive"
tick + Download selection gives a selection.json for scripts/scale_remove_trucks.py. Read-only.
"""
import collections, csv, html, json, os, re, shutil, subprocess

REVIEW = os.path.expanduser("~/URCA/IRIS/review")
OUT = os.path.join(REVIEW, "whatif")
ARCHIVE = os.path.expanduser("~/URCA/IRIS/archive/day-export-2026-09-13-ORIGINAL")
PG = os.environ.get("PG_CONTAINER", "scale-ui-postgres-1")
PREFIX = "ااا-"
KIND = lambda v: ("not tracked" if v.startswith("not") else v.split(" ")[0])


def main():
    new = json.load(open(os.path.join(OUT, "new-reid/summary.json")))
    old = json.load(open(os.path.join(OUT, "old-reid/summary.json")))
    cls = {int(r["gid"]): r["class"] for r in csv.DictReader(open(os.path.join(REVIEW, "offline-direction/report.csv"), encoding="utf-8-sig"))}
    removed = set(json.load(open(os.path.join(REVIEW, "removed_trucks.json")))["gids"])
    r = subprocess.run(["docker", "exec", "-i", PG, "psql", "-U", "iris", "-d", "iris", "-qAt", "-F", "\t", "-c",
                        "select external_vehicle_id, id from trucks where created_at < '2026-10-07 13:00'"], capture_output=True, text=True, check=True)
    uuid = {int(a.replace(PREFIX, "")): b for a, b in (l.split("\t") for l in r.stdout.splitlines())}
    os.makedirs(os.path.join(OUT, "images"), exist_ok=True)
    os.system(f"chmod -R u+w '{OUT}/images' 2>/dev/null")
    cards, rows, counts = [], [], collections.Counter()
    for g in sorted(int(x) for x in new["outcome_by_old_gid"]):
        o_new, o_old = new["outcome_by_old_gid"][str(g)], old["outcome_by_old_gid"].get(str(g), "not in replay")
        c = "removed by you" if g in removed else cls.get(g, "no class")
        k_new, k_old = KIND(o_new), KIND(o_old)
        should_drop = c in ("would_not_start", "removed by you", "not_vehicle")
        flag = ""
        if should_drop and k_new != "not tracked":
            flag = "leaving but still tracked"
        elif c == "keep" and k_new == "not tracked":
            flag = "entering but dropped"
        counts[(c, k_new)] += 1
        img = ""
        for cam in ("cam-01", "cam-02"):
            p = os.path.join(ARCHIVE, "data/vehicle_crops", cam, f"{g}.jpg")
            if os.path.exists(p):
                shutil.copyfile(p, os.path.join(OUT, "images", f"{g:04d}_{cam}.jpg"))
                img += f'<img src="images/{g:04d}_{cam}.jpg" loading=lazy>'
        link = f'<a style="color:#8ab4ff" target=_blank href="https://localhost:35444/trucks/{uuid[g]}">{PREFIX}{g:04d}</a>' if g in uuid else f"{PREFIX}{g:04d} (removed)"
        cards.append(f'<div class="c" data-cls="{html.escape(c)}" data-new="{k_new}" data-flag="{html.escape(flag)}">'
                     f'<div class=h><label style="float:right"><input type=checkbox class=sel data-gid="{g}" onchange="u()"> false positive</label>{link}</div>'
                     f'<div class=im>{img}</div><div class=m>offline class: <b>{html.escape(c)}</b><br>old rules (replay): {html.escape(o_old)}<br>'
                     f'new rules (replay): <b>{html.escape(o_new)}</b>{("<br><span class=flag>" + flag + "</span>") if flag else ""}</div></div>')
        rows.append([g, f"{PREFIX}{g:04d}", c, o_old, o_new, flag])
    with open(os.path.join(OUT, "report.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gid", "truck", "offline_class", "old_rules_replay", "new_rules_replay", "flag"])
        w.writerows(rows)
    h = new["headline"]
    ho = old["headline"]
    table = "".join(f"<tr><td>{c}</td>" + "".join(f"<td>{counts[(c, k)]}</td>" for k in ("own", "merged", "split", "not tracked")) + "</tr>"
                    for c in sorted({c for c, _ in counts}))
    flags = collections.Counter(r[5] for r in rows if r[5])
    fb = "".join(f'<button onclick="f(\'flag\',\'{k}\')">{k} ({v})</button> ' for k, v in flags.items())
    page = f"""<meta charset=utf-8><title>What-if replay</title><style>
body{{font:14px sans-serif;background:#111;color:#eee;margin:16px}}table{{border-collapse:collapse;margin:8px 0}}td,th{{border:1px solid #444;padding:3px 10px;text-align:right}}
th:first-child,td:first-child{{text-align:left}}.g{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:10px}}.c{{background:#1c1c1c;padding:8px;border-radius:6px}}
.im img{{width:48%;margin-right:2%}}.m{{font-size:12px;color:#bbb;margin-top:4px}}.flag{{color:#ff9a6b}}button{{margin:2px 4px 6px 0;padding:5px 9px}}
</style><h2>What-if replay of the day: old rules vs new rules, per truck</h2>
<p>Old rules: {ho['sim_tracks']} tracks, {ho['exits_sim']} exits (the real run: 365 tracks, {ho['exits_old_run']} exits). New rules: {h['sim_tracks']} tracks, {h['exits_sim']} exits.
A rough model (see the replay script): pairs that stayed unanswered count as "no match", so splits are overstated and exits understated.</p>
<table><tr><th>offline class</th><th>own track</th><th>merged</th><th>split</th><th>not tracked</th></tr>{table}</table>
<div id=bar style="position:sticky;top:0;background:#222;padding:8px;margin:8px 0;z-index:5"><b id=n>0 selected</b> <button onclick="dl()">Download selection (selection.json)</button>
<button onclick="clr()">Clear</button> shown: <b id=shown></b></div>
<div>new rules: <button onclick="f('new','')">all</button> <button onclick="f('new','not tracked')">not tracked</button> <button onclick="f('new','own')">own track</button>
<button onclick="f('new','merged')">merged</button> <button onclick="f('new','split')">split</button> {fb}</div>
<div class=g>{''.join(cards)}</div><script>
var K="whatif_sel",F={{new:"",flag:""}};function sel(){{return [...document.querySelectorAll(".sel:checked")].map(x=>+x.dataset.gid)}}
function u(){{var s=sel();try{{localStorage.setItem(K,JSON.stringify(s))}}catch(e){{}}document.getElementById("n").textContent=s.length+" selected"}}
function dl(){{var b=new Blob([JSON.stringify({{gids:sel()}})],{{type:"application/json"}}),a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="selection.json";a.click()}}
function clr(){{document.querySelectorAll(".sel").forEach(x=>x.checked=false);u()}}
function f(k,v){{if(k=="new"){{F.new=v;F.flag=""}}else{{F.flag=v;F.new=""}};ap()}}
function ap(){{var n=0;document.querySelectorAll(".c").forEach(c=>{{var ok=(!F.new||c.dataset.new==F.new)&&(!F.flag||c.dataset.flag==F.flag);c.style.display=ok?"":"none";if(ok)n++}});document.getElementById("shown").textContent=n}}
try{{JSON.parse(localStorage.getItem(K)||"[]").forEach(g=>{{var x=document.querySelector('.sel[data-gid="'+g+'"]');if(x)x.checked=true}})}}catch(e){{}}u();ap()</script>"""
    open(os.path.join(OUT, "index.html"), "w").write(page)
    print(f"wrote {OUT}/index.html: {len(cards)} trucks; flags {dict(flags)}")


if __name__ == "__main__":
    main()
