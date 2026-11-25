"""Live view of the VLM association arbiter: /vlm.

Reads what perception/modules/mtmc/vlm_arbiter.py writes to data/vlm_arbitration/ (run.json, one
arbitration_log.jsonl row per comparison, and the exact images sent) and serves a
self-refreshing page: the model + prompts at the top, then every comparison with the
images sent, the exact prompt, and the model's reply. The sidecar clears the folder
on every start, so the page always shows the current run only.
"""
import json
import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

router = APIRouter(tags=["vlm"])

_DIR = os.path.join("data", "vlm_arbitration")


@router.get("/api/vlm/log")
async def vlm_log():
    run = {}
    try:
        with open(os.path.join(_DIR, "run.json")) as f:
            run = json.load(f)
    except (OSError, ValueError):
        pass
    rows = []
    try:
        with open(os.path.join(_DIR, "arbitration_log.jsonl")) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue  # a line still being written
    except OSError:
        pass
    return JSONResponse({"run": run, "rows": rows})


@router.get("/api/vlm/image/{name}")
async def vlm_image(name: str):
    if "/" in name or ".." in name or not name.endswith(".jpg"):
        raise HTTPException(status_code=400, detail="bad name")
    path = os.path.join(_DIR, name)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(path, media_type="image/jpeg")


@router.get("/vlm", response_class=HTMLResponse)
async def vlm_page():
    return HTMLResponse(_PAGE)


_PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>VLM live</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#111;--card:#1a1a1a;--line:#333;--fg:#eee;--dim:#999}
@media (prefers-color-scheme: light){:root{--bg:#f6f6f6;--card:#fff;--line:#ddd;--fg:#111;--dim:#666}}
body{font-family:system-ui,sans-serif;background:var(--bg);color:var(--fg);margin:0;padding:16px 20px}
h1{font-size:19px;margin:0 0 4px}.dim{color:var(--dim);font-size:13px}
pre{white-space:pre-wrap;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:10px;font-size:12px;margin:6px 0}
details{margin:8px 0}summary{cursor:pointer;font-weight:600;font-size:13px}
.stats span{display:inline-block;margin-right:14px;font-size:13px}
.card{border:1px solid var(--line);border-radius:8px;background:var(--card);padding:10px;margin:10px 0}
.side{margin:4px 0}.side .t{font-size:12px;color:var(--dim);margin-bottom:3px}
.imgs{display:flex;gap:8px;flex-wrap:wrap}
.imgs img{height:170px;max-width:300px;object-fit:contain;background:#000;border-radius:4px;border:1px solid var(--line)}
.b{display:inline-block;padding:2px 10px;border-radius:4px;font-weight:600;color:#fff}
.same{background:#1e6b2a}.different{background:#8a2b2b}.uncertain{background:#8a6d1e}.error{background:#555}
.v{font-size:13px;margin-top:6px}
</style></head><body>
<h1>VLM association arbiter <span class="dim" id="live">connecting…</span></h1>
<div class="dim" id="runinfo"></div>
<details open><summary>Prompt sent (1 image per vehicle)</summary><pre id="p1"></pre></details>
<details><summary>Prompt sent (3 views per vehicle)</summary><pre id="p3"></pre></details>
<div class="stats" id="stats"></div>
<div id="list"></div>
<script>
const CAM={"cam-01":"Cam-41","cam-02":"Cam-16","cam-03":"Cam-52","cam-04":"Cam-05","cam-05":"Cam-23"};
const cn=c=>CAM[c]?CAM[c]+" ("+c+")":c;
const esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
let runKey=null, shown=new Set();
function imgs(list,caps){return list.map((n,i)=>`<figure style="margin:0"><a href="api/vlm/image/${n}" target="_blank"><img src="api/vlm/image/${n}" loading="lazy"></a><figcaption class="dim" style="max-width:300px;font-size:11px">${esc((caps||[])[i]||"")}</figcaption></figure>`).join("")}
function card(r){
  const c=r.context||{}, v=r.verdict||{}, d=v.decision||"error";
  const list=x=>(x||[]).map(esc).join(", ")||"-";
  if(c.task==="direction"){
    const meaning=(c.mode==="front_is_enter"?{front:"entering",rear:"exiting"}:{front:"exiting",rear:"entering"})[d]||"not decided";
    return `<div class="card" id="j${r.job_id}">
  <div class="side"><div class="t">DIRECTION - tracked vehicle: ${esc(cn(c.cam_id))} local ${esc(c.local_id)} @ ${Math.round(c.pts||0)}s (${r.images_a.length} image${r.images_a.length>1?"s":""})</div><div class="imgs">${imgs(r.images_a,c.captions_a)}</div></div>
  <div class="v"><span class="b ${d==="front"||d==="rear"?"same":"uncertain"}">${esc(d)} ${esc(v.confidence??"")}</span> &nbsp;${esc(r.compare_elapsed_s)}s &nbsp;job ${r.job_id} &nbsp;<b>${esc(meaning)}</b>
  <div>${esc(v.explanation||v.error||"")}</div>
  <div class="dim">rule on this camera: ${esc(c.mode)} (the end facing the camera tells entering from leaving)</div>
  <details><summary>Exact prompt + raw verdict</summary><pre>${esc(r.prompt)}</pre><pre>${esc(JSON.stringify(v,null,1))}</pre></details></div></div>`;
  }
  return `<div class="card" id="j${r.job_id}">
  <div class="side"><div class="t">Vehicle A - NEW tracklet: ${esc(cn(c.cam_id))} local ${esc(c.local_id)} @ ${Math.round(c.pts||0)}s (${r.images_a.length} image${r.images_a.length>1?"s":""})</div><div class="imgs">${imgs(r.images_a,c.captions_a)}</div></div>
  <div class="side"><div class="t">Vehicle B - CANDIDATE global ID ${esc(c.candidate_global_id)} from ${esc(cn(c.candidate_last_camera))}, gap ${esc(c.elapsed_s)}s${c.sim!=null?", reid sim "+esc(c.sim):""} (${r.images_b.length} image${r.images_b.length>1?"s":""})</div><div class="imgs">${imgs(r.images_b,c.captions_b)}</div></div>
  <div class="v"><span class="b ${d}">${esc(d)} ${esc(v.confidence??"")}</span> &nbsp;${esc(r.compare_elapsed_s)}s &nbsp;job ${r.job_id}
  <div>${esc(v.explanation||v.error||"")}</div>
  <div class="dim">matched: ${list(v.matched_features)} | conflicting: ${list(v.conflicting_features)}</div>
  <details><summary>Exact prompt + raw verdict</summary><pre>${esc(r.prompt)}</pre><pre>${esc(JSON.stringify(v,null,1))}</pre></details></div></div>`;
}
async function tick(){
  try{
    const j=await (await fetch("api/vlm/log",{cache:"no-store"})).json();
    const key=String((j.run||{}).started_at||"");
    if(key!==runKey){runKey=key;shown=new Set();document.getElementById("list").innerHTML="";}
    const run=j.run||{};
    document.getElementById("runinfo").textContent=run.model?`${run.provider} - ${run.model} - ${run.endpoint} - temp ${run.temperature} - started ${new Date(run.started_at*1000).toLocaleTimeString()}`:"waiting for the sidecar to start the arbiter…";
    document.getElementById("p1").textContent=run.prompt_single_image_each||"";
    document.getElementById("p3").textContent=run.prompt_three_views_each||"";
    const rows=j.rows, cnt={};let lat=0;
    for(const r of rows){const d=(r.verdict||{}).decision||"error";cnt[d]=(cnt[d]||0)+1;lat+=r.compare_elapsed_s||0}
    document.getElementById("stats").innerHTML=`<span><b>${rows.length}</b> comparisons</span>`+Object.entries(cnt).map(([k,v])=>`<span>${k}: <b>${v}</b></span>`).join("")+(rows.length?`<span>avg ${(lat/rows.length).toFixed(1)}s</span>`:"");
    const list=document.getElementById("list");
    const fresh=rows.filter(r=>!shown.has(r.job_id)).sort((a,b)=>a.submitted_at-b.submitted_at);
    for(const r of fresh){shown.add(r.job_id);list.insertAdjacentHTML("afterbegin",card(r))}
    document.getElementById("live").textContent="live - updated "+new Date().toLocaleTimeString();
  }catch(e){document.getElementById("live").textContent="disconnected"}
}
tick();setInterval(tick,2500);
</script></body></html>
"""
