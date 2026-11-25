#!/usr/bin/env python3
"""Run N local videos (plus the two LAN RTSP cams by default) through the IRIS
DeepStream pipeline and watch them live on the dashboard (http://localhost:8000).

Pipeline sources live in perception/pipeline/sidecar.yaml and the sidecar builds ONE batched
pipeline for all of them at startup, so "running N streams" = write the source
list + restart the sidecar container. The web UI discovers every stream
automatically (Redis iris:status:*) and shows it as a live tile.

Commands:
  python3 scripts/live_streams.py list                   what can run right now
  python3 scripts/live_streams.py run --n 6             6 videos + cam0/cam1 RTSP
  python3 scripts/live_streams.py run --n 6 --no-rtsp   videos only
  python3 scripts/live_streams.py run --ids cam-2,cam-4 exact selection (sorted list)
  python3 scripts/live_streams.py run --all             every usable upload
  python3 scripts/live_streams.py status                streams live right now
  python3 scripts/live_streams.py restore               back to the saved config
  python3 scripts/live_streams.py copy '<glob>' [--max 5] [--dry-run]
                                                        pull dataset clips into data/uploads

Notes:
- --n picks the first N usable uploads (sorted by name); --ids/--all for explicit
  selection. RTSP cams are added first as cam0 (exit) / cam1 (enter) unless --no-rtsp.
- The pre-tool sidecar.yaml is backed up once at data/live_streams.backup.yaml so
  `restore` always gets you back to the original config.
- Only NVDEC-decodable files are picked (h264/hevc/mpeg4/...). rawvideo, vp8/vp9
  and AV1 sources are skipped - NVDEC cannot decode them.
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIDECAR_YAML = os.path.join(REPO, "perception", "pipeline", "sidecar.yaml")
UPLOADS_DIR = os.path.join(REPO, "data", "uploads")
BACKUP = os.path.join(REPO, "data", "live_streams.backup.yaml")

VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v")
OK_CODECS = {"h264", "hevc", "h265", "mpeg4", "mpeg2video", "h263", "mjpeg"}
MIN_SIZE = 200_000   # bytes
MIN_DURATION = 2.0   # seconds

# The two LAN RTSP cams (cam0 = exit, cam1 = enter). Credentials are NEVER
# written to disk: the URIs carry ${RTSP_USER}/${RTSP_PASS} placeholders that
# perception/engine/sources.py expands from the container environment (host .env ->
# docker-compose -> container). Unset vars keep the placeholder and the source
# fails to authenticate at runtime — visible in the sidecar log.
RTSP_SOURCES = [
    ("cam0", "rtsp://${RTSP_USER}:${RTSP_PASS}@10.2.0.50:8554/east-entrance-exit"),
    ("cam1", "rtsp://${RTSP_USER}:${RTSP_PASS}@10.2.0.50:8554/east-entrance-enter"),
]

DASHBOARD_URL = "http://localhost:8000"


def _probe(path: str):
    """ffprobe one file -> {codec, w, h, fps, duration} or None."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json",
             "-show_streams", "-show_format", path],
            capture_output=True, text=True, timeout=30)
    except Exception:
        return None
    if r.returncode != 0:
        return None
    try:
        data = json.loads(r.stdout)
    except Exception:
        return None
    v = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if not v:
        return None
    num, _, den = (v.get("avg_frame_rate") or "0/0").partition("/")
    try:
        fps = float(num) / float(den) if den and float(den) else 0.0
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    try:
        duration = float(v.get("duration") or data.get("format", {}).get("duration") or 0)
    except (ValueError, TypeError):
        duration = 0.0
    return {"codec": v.get("codec_name", ""), "w": v.get("width"),
            "h": v.get("height"), "fps": fps, "duration": duration}


def _h264_twin(path: str):
    """If `path` is a non-h264 upload that already has a transcoded h264 twin
    (<stem>_h264.mp4), return the twin path; else None. The sidecar only loops
    sources NVDEC can decode (h264/hevc) — mpeg4 AVIs fall back to software
    decode where nvurisrcbin's file-loop never fires, so they play once and
    freeze. Transcoding to h264 makes every source NVDEC-decodable and looping."""
    stem, _ = os.path.splitext(path)
    twin = stem + "_h264.mp4"
    return twin if os.path.isfile(twin) else None


# Subfolders of data/uploads that hold something other than replay clips (generated, or other tools' sources).
_NOT_REPLAY_DIRS = {"golden", "barber", "fire", "safety"}


def _video_files():
    """Replay clips: those in data/uploads and in its use-case folders (attendance/, multicam/, misc/), as paths
    relative to it. Sorted by file name, as before the folders existed, so cam-N numbering is unchanged."""
    out = []
    for dirpath, dirnames, filenames in os.walk(UPLOADS_DIR):
        dirnames[:] = [d for d in dirnames if d not in _NOT_REPLAY_DIRS]
        for name in filenames:
            path = os.path.join(dirpath, name)
            if not name.lower().endswith(VIDEO_EXTS):
                continue
            # An original with a transcoded h264 twin is superseded by it.
            if _h264_twin(path):
                continue
            out.append(os.path.relpath(path, UPLOADS_DIR))
    return sorted(out, key=lambda rel: (os.path.basename(rel), rel))


def list_videos():
    """[(id, name, probe)] for usable uploads, sorted by name, id = cam-N."""
    usable, skipped = [], []
    for name in _video_files():
        path = os.path.join(UPLOADS_DIR, name)
        if os.path.getsize(path) < MIN_SIZE:
            skipped.append(name + " (tiny)")
            continue
        p = _probe(path)
        if not p:
            skipped.append(name + " (unprobeable)")
            continue
        if p["codec"] not in OK_CODECS:
            skipped.append(name + f" ({p['codec']})")
            continue
        if p["duration"] < MIN_DURATION:
            skipped.append(name + " (too short)")
            continue
        usable.append((name, p))
    return usable, skipped


def container_uri(rel_path: str) -> str:
    """file:// URI as seen INSIDE the sidecar container (/workspace/iris/<rel>)."""
    cpath = f"/workspace/iris/{rel_path}"
    return "file://" + urllib.parse.quote(cpath, safe="/")


def _replace_sources_block(text: str, entries: list, marker: str) -> str:
    """Surgically replace the `sources:` list in sidecar.yaml, keeping every other
    key and its comments intact. entries = [{id, uri}]."""
    lines = text.split("\n")
    start = next((i for i, ln in enumerate(lines) if ln.strip() == "sources:"), None)
    if start is None:
        raise SystemExit("sidecar.yaml has no `sources:` key - refusing to edit")
    end = None
    for j in range(start + 1, len(lines)):
        if lines[j] and not lines[j][0].isspace() and not lines[j].lstrip().startswith("#"):
            end = j
            break
    block = [marker]
    for e in entries:
        block.append(f"    - {{id: {e['id']}, uri: \"{e['uri']}\"}}")
    head = lines[:start + 1]
    tail = lines[end:] if end is not None else []
    return "\n".join(head + block + tail)


def _validate_yaml(path: str, label: str = ""):
    import yaml
    try:
        with open(path) as f:
            yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise SystemExit(f"{label}invalid YAML in {path}: {exc}")


def _find_container(substr: str):
    try:
        out = subprocess.run(["docker", "ps", "--format", "{{.Names}}"],
                             capture_output=True, text=True, timeout=15).stdout.split()
    except Exception:
        return None
    return next((n for n in out if substr in n), None)


def _restart_sidecar():
    ctr = _find_container("deepstream-sidecar")
    if not ctr:
        raise SystemExit("deepstream-sidecar container not found - is the stack up? (docker compose up -d)")
    print(f"[live] restarting {ctr} ...")
    subprocess.run(["docker", "restart", ctr], check=True, timeout=120)


def _api_cameras():
    try:
        with urllib.request.urlopen(f"{DASHBOARD_URL}/api/cameras", timeout=5) as resp:
            return json.loads(resp.read().decode()).get("cameras", [])
    except Exception:
        return None


def _redis_status():
    ctr = _find_container("redis")
    if not ctr:
        return None
    keys = subprocess.run(["docker", "exec", ctr, "redis-cli", "keys", "iris:status:*"],
                          capture_output=True, text=True, timeout=15).stdout.split()
    cams = []
    for k in keys:
        sid = k.split("iris:status:", 1)[-1]
        raw = subprocess.run(["docker", "exec", ctr, "redis-cli", "get", k],
                             capture_output=True, text=True, timeout=15).stdout
        try:
            st = json.loads(raw)
        except Exception:
            st = {}
        cams.append({"id": sid, "running": st.get("running"),
                     "is_file": st.get("is_file"), "source": st.get("source")})
    return cams


def _current_streams():
    return _api_cameras() or _redis_status() or []


def _redis_keys():
    """Set of stream ids currently reported by the sidecar (authoritative)."""
    ctr = _find_container("redis")
    if not ctr:
        return set()
    out = subprocess.run(["docker", "exec", ctr, "redis-cli", "keys", "iris:status:*"],
                         capture_output=True, text=True, timeout=15).stdout.split()
    return {k.split("iris:status:", 1)[-1] for k in out}


def wait_for_streams(expected_ids, timeout=180):
    """Wait until Redis reports EXACTLY the expected stream ids.

    The sidecar purges stale iris:status:* keys at boot and writes one per
    configured source as frames flow, so key-set equality (held for two
    consecutive polls) means the NEW pipeline is actually serving these
    sources -- not leftovers from a previous run. Returns (streams, ok).
    """
    expected = set(expected_ids)
    t0 = time.time()
    stable = 0
    while time.time() - t0 < timeout:
        keys = _redis_keys()
        if expected and keys == expected:
            stable += 1
            if stable >= 2:
                break
        else:
            stable = 0
        time.sleep(5)
    if not expected:  # fallback: wait for any status key to appear
        while time.time() - t0 < timeout and not _redis_keys():
            time.sleep(5)
    streams = _current_streams() or []
    seen = {c["id"]: c for c in streams}
    missing = sorted(expected - set(seen))
    return seen, (stable >= 2 and not missing)


def _print_streams(streams: dict):
    for sid in sorted(streams):
        c = streams[sid]
        flag = "OK " if c.get("running") else "off"
        print(f"    {sid:>6}  {flag}  {c.get('source', '')}")


def cmd_list(_args):
    usable, skipped = list_videos()
    if not os.path.isdir(UPLOADS_DIR):
        raise SystemExit(f"{UPLOADS_DIR} does not exist")
    print(f"[live] usable videos in data/uploads ({len(usable)}):")
    for i, (name, p) in enumerate(usable, start=1):
        print(f"    cam-{i:>2}  {name:<60} {p['codec']:<8} {p['w']}x{p['h']} "
              f"{p['fps']:.1f}fps {p['duration']:.0f}s")
    if skipped:
        print(f"[live] skipped ({len(skipped)}): {', '.join(skipped)}")
    print("[live] default RTSP (always added unless --no-rtsp):")
    for cid, uri in RTSP_SOURCES:
        print(f"    {cid:>4}  {uri}")


def _select_videos(args):
    usable, _ = list_videos()
    if not usable:
        raise SystemExit("no usable videos in data/uploads - run `copy` or drop files in first")
    if args.ids:
        wanted = [i.strip() for i in args.ids.split(",") if i.strip()]
        chosen, chosen_ids = [], []
        for w in wanted:
            if not w.startswith("cam-"):
                raise SystemExit(f"--ids entries must look like cam-N, got {w!r}")
            try:
                idx = int(w.split("-", 1)[1])
            except ValueError:
                raise SystemExit(f"bad id {w!r}")
            if not (1 <= idx <= len(usable)):
                print(f"  ! {w} out of range (1..{len(usable)}), skipping")
                continue
            chosen.append(usable[idx - 1][0])
            chosen_ids.append(w)
        if not chosen:
            raise SystemExit("no valid ids given")
        return chosen, chosen_ids
    if args.all:
        return [n for n, _ in usable], None
    n = args.n
    if n is None or n < 1:
        raise SystemExit("run requires --n N (or --all / --ids)")
    if n > len(usable):
        print(f"  ! only {len(usable)} usable videos; using all")
        n = len(usable)
    return [name for name, _ in usable[:n]], None


def _set_yaml_value(key: str, value: str):
    """Set or insert a top-level `key: value` line in sidecar.yaml (inserted right
    after the previous key if absent)."""
    text = open(SIDECAR_YAML).read()
    if re.search(rf"(?m)^{re.escape(key)}:", text):
        text = re.sub(rf"(?m)^{re.escape(key)}:.*$", f"{key}: {value}", text)
    else:
        lines = text.splitlines()
        # find the last top-level `name:` line to insert after
        idx = max((i for i, ln in enumerate(lines)
                   if ln and not ln[0].isspace() and not ln.lstrip().startswith("#")),
                  default=0)
        lines.insert(idx + 1, f"{key}: {value}")
        text = chr(10).join(lines)
    open(SIDECAR_YAML, "w").write(text)


def _clear_yaml_key(key: str):
    """Remove a top-level `key: ...` line (and its trailing comment lines)."""
    text = open(SIDECAR_YAML).read()
    new = re.sub(rf"(?m)^{re.escape(key)}:.*\n", "", text)
    if new != text:
        open(SIDECAR_YAML, "w").write(new)
        _validate_yaml(SIDECAR_YAML, "sidecar config: ")


def _set_process_fps(n: float):
    """Write process_fps: N into sidecar.yaml so the sidecar throttles pgie
    inference + probe analysis to ~N fps per stream while the sources still
    decode and play realtime (see sidecar.py infer_interval)."""
    text = open(SIDECAR_YAML).read()
    if re.search(r"(?m)^process_fps:", text):
        text = re.sub(r"(?m)^process_fps:.*$", f"process_fps: {n:g}", text)
    else:
        text = re.sub(r"(?m)^(stream_fps:.*)$",
                      lambda m: m.group(1) + "\n" + f"process_fps: {n:g}", text, count=1)
    open(SIDECAR_YAML, "w").write(text)
    _validate_yaml(SIDECAR_YAML, "sidecar config: ")
    m = re.search(r"(?m)^stream_fps:\s*([\d.]+)", text)
    sfps = float(m.group(1)) if m else 15.0
    print(f"[live] process_fps={n:g} (stream_fps={sfps:g} -> every "
          f"{max(1, int(round(sfps / n)))}th frame fully analyzed; "
          f"streams still play realtime)")


def _clear_process_fps():
    """Remove process_fps so every frame is analyzed again."""
    text = open(SIDECAR_YAML).read()
    new = re.sub(r"(?m)^process_fps:.*\n", "", text)
    if new != text:
        open(SIDECAR_YAML, "w").write(new)
        _validate_yaml(SIDECAR_YAML, "sidecar config: ")
        print("[live] process_fps cleared (every frame analyzed)")


def cmd_run(args):
    chosen, chosen_ids = _select_videos(args)
    entries = []
    if not args.no_rtsp:
        for cid, uri in RTSP_SOURCES:
            entries.append({"id": cid, "uri": uri})
    if chosen_ids:
        for cid, name in zip(chosen_ids, chosen):
            entries.append({"id": cid, "uri": container_uri(os.path.join("data/uploads", name))})
    else:
        for i, name in enumerate(chosen, start=1):
            entries.append({"id": f"cam-{i}", "uri": container_uri(os.path.join("data/uploads", name))})

    if not os.path.exists(BACKUP):
        shutil.copy2(SIDECAR_YAML, BACKUP)
        print(f"[live] saved pre-tool config -> {BACKUP}")

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    rtsp_note = "+ 2 rtsp" if not args.no_rtsp else ""
    marker = (f"    # generated by scripts/live_streams.py run "
              f"({len(chosen)} videos {rtsp_note}) {stamp}")
    text = open(SIDECAR_YAML).read()
    open(SIDECAR_YAML, "w").write(_replace_sources_block(text, entries, marker))
    if args.process_fps > 0:
        _set_process_fps(args.process_fps)
    else:
        _clear_process_fps()
    if args.no_record:
        _set_yaml_value("record_enabled", "false")
        _validate_yaml(SIDECAR_YAML, "sidecar config: ")
        print("[live] record_enabled=false (no NVENC recording branches)")
    else:
        # Preserve the existing record_enabled value instead of deleting the key:
        # the sidecar defaults record_enabled to True, so clearing the key silently
        # re-enables the splitmuxsink record branches that can back up the tee and
        # freeze the whole pipeline (observed in e2e verification). Explicit false
        # stays false across re-runs unless --no-record/--record flips it.
        if not re.search(r"(?m)^record_enabled:", open(SIDECAR_YAML).read()):
            _set_yaml_value("record_enabled", "false")
            print("[live] record_enabled unset -> pinned to false (run --record to enable recording)")
        _validate_yaml(SIDECAR_YAML, "sidecar config: ")

    print(f"[live] sidecar.yaml now defines {len(entries)} source(s):")
    for e in entries:
        print(f"    {e['id']:>5}  {e['uri']}")
    total = len(entries)
    if total >= 8:
        print(f"  ! note: {total} streams will likely saturate this GPU "
              f"(capacity sweep: ~6 comfortable @25fps)")
    if not args.no_rtsp:
        print("  ! note: RTSP sources (cam0/cam1) are registered but may not "
              "deliver frames through the batched nvstreammux pipeline "
              "(pre-existing issue with mixed file+RTSP sources)")

    _restart_sidecar()
    print(f"[live] waiting for {len(entries)} stream(s) to come up (engine load may take ~1-2 min)...")
    seen, ok = wait_for_streams([e["id"] for e in entries])
    missing = [e["id"] for e in entries if e["id"] not in seen]
    if missing:
        print(f"[live] ! not seen yet: {', '.join(missing)} (still booting? check docker logs)")
    print(f"[live] streams live now:")
    _print_streams(seen)
    print(f"[live] dashboard: {DASHBOARD_URL}")
    return 0 if not missing else 1


def cmd_status(_args):
    streams = _current_streams()
    if not streams:
        print("[live] no streams reported yet (sidecar starting? run `status` again)")
        return 1
    print("[live] streams now:")
    _print_streams({c["id"]: c for c in streams})
    print(f"[live] dashboard: {DASHBOARD_URL}")
    return 0


def _yaml_source_ids(path: str):
    """ids of the configured sources in a sidecar.yaml, or [] if unparseable."""
    try:
        import yaml
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
        return [s.get("id") for s in cfg.get("sources", []) if s.get("uri")]
    except Exception:
        return []


def cmd_restore(_args):
    if not os.path.exists(BACKUP):
        print("[live] no backup found (data/live_streams.backup.yaml) - nothing to restore")
        return 1
    shutil.copy2(BACKUP, SIDECAR_YAML)
    _validate_yaml(SIDECAR_YAML, "sidecar config: ")
    expected = _yaml_source_ids(SIDECAR_YAML)
    print("[live] restored sidecar.yaml from backup ("
          f"{len(expected)} source(s))")
    _restart_sidecar()
    print("[live] waiting for the pipeline to come back up...")
    wait_for_streams(expected, timeout=180)
    streams = _current_streams()
    if streams:
        print("[live] streams now:")
        _print_streams({c["id"]: c for c in streams})
    else:
        print("[live] no streams yet - check `docker logs <sidecar>` if it stays empty")
    return 0


def cmd_transcode(args):
    """Transcode non-h264 uploads (mpeg4 AVIs) to h264 MP4 twins.

    The sidecar loops file sources via nvurisrcbin's file-loop, which only fires
    on the NVDEC decode path. mpeg4 AVIs are software-decoded, never loop, and
    freeze at EOS — transcode them to h264 so NVDEC handles them and they loop
    like real CCTV streams. Each transcode writes <stem>_h264.mp4 next to the
    original; list/run prefer the twin automatically."""
    todo = []
    for name in _video_files():
        path = os.path.join(UPLOADS_DIR, name)
        p = _probe(path)
        if not p:
            continue
        if p["codec"] in ("h264", "hevc", "h265"):
            continue
        todo.append((name, p))
    if not todo:
        print("[live] no non-h264 uploads left to transcode")
        return 0
    print(f"[live] {len(todo)} upload(s) need h264 transcoding:")
    for name, p in todo:
        print(f"    {name}  ({p['codec']} {p['w']}x{p['h']} @{p['fps']:.0f}fps)")
    if args.dry_run:
        print("[live] dry-run - nothing transcoded")
        return 0
    failed = 0
    for name, p in todo:
        src = os.path.join(UPLOADS_DIR, name)
        stem, _ = os.path.splitext(src)
        dst = stem + "_h264.mp4"
        fps = p["fps"] or 25
        # crf 23 keeps quality ~source; veryfast keeps it quick for these clips.
        # Write to a temp path and atomically rename: a crash mid-transcode must
        # not leave a partial twin that silently supersedes the good original.
        tmp = dst + ".tmp"
        # -f mp4 is required because the temp name (<dst>.tmp) has no extension
        # ffmpeg can infer a muxer from; without it every transcode fails with
        # "Unable to choose an output format".
        cmd = ["ffmpeg", "-y", "-i", src, "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "23", "-pix_fmt", "yuv420p", "-r", str(fps), "-an", "-f", "mp4", tmp]
        print(f"    transcoding {name} -> {os.path.basename(dst)} ...", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode == 0 and os.path.isfile(tmp) and os.path.getsize(tmp) >= MIN_SIZE:
            os.replace(tmp, dst)
            print(f"      ok ({os.path.getsize(dst) / 1e6:.1f} MB)")
        else:
            failed += 1
            try:
                os.remove(tmp)
            except OSError:
                pass
            tail = (r.stderr or "").strip().splitlines()
            print(f"      FAILED: {tail[-1] if tail else 'unknown error'}")
    if failed:
        print(f"[live] {failed} transcode(s) failed - check the files and retry")
        return 1
    print("[live] done - run `list` to confirm, then `run --all --no-record`")
    return 0


def cmd_copy(args):
    pat = args.pattern
    if os.path.isdir(pat):
        pat = os.path.join(pat, "**", "*")
    files = [f for f in glob.glob(pat, recursive=True)
             if f.lower().endswith(VIDEO_EXTS) and os.path.isfile(f)]
    picked = []
    for f in sorted(files):
        p = _probe(f)
        if p and p["codec"] in OK_CODECS and p["duration"] >= MIN_DURATION:
            picked.append(f)
        if len(picked) >= args.max:
            break
    if not picked:
        print("[live] no decodable videos matched the pattern")
        return 1
    print(f"[live] {len(picked)} decodable video(s):")
    for f in picked:
        print(f"    {os.path.basename(f)}")
    if args.dry_run:
        print("[live] dry-run - nothing copied")
        return 0
    os.makedirs(args.to, exist_ok=True)
    for f in picked:
        dest = os.path.join(args.to, os.path.basename(f))
        n = 1
        stem, ext = os.path.splitext(os.path.basename(f))
        while os.path.exists(dest):
            dest = os.path.join(args.to, f"{stem}-{n}{ext}")
            n += 1
        shutil.copy2(f, dest)
        print(f"    copied -> {dest}")
    print("[live] run `python3 scripts/live_streams.py list` to see them")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="list usable videos + default RTSP sources")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("run", help="write N sources to sidecar.yaml and restart the sidecar")
    p.add_argument("--n", type=int, default=None, help="number of videos (1..available)")
    p.add_argument("--all", action="store_true", help="use every usable upload")
    p.add_argument("--ids", default=None, help="comma list, e.g. cam-2,cam-4 (positions in sorted list)")
    p.add_argument("--no-rtsp", action="store_true", help="do not add the cam0/cam1 RTSP streams")
    p.add_argument("--process-fps", type=float, default=0.0,
                   help="analyze only ~N fps per stream (streams still play realtime); "
                        "e.g. 5 = every 6th frame of a 30fps source")
    p.add_argument("--no-record", action="store_true",
                   help="set record_enabled: false (no NVENC recording branches; "
                        "consumer GPUs cap concurrent NVENC sessions, which kills the "
                        "pipeline beyond ~12-15 streams)")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("status", help="show streams currently reported live")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("restore", help="restore the pre-tool sidecar.yaml and restart")
    p.set_defaults(func=cmd_restore)

    p = sub.add_parser("transcode",
                       help="convert non-h264 uploads to h264 MP4 (NVDEC-decodable + looping)")
    p.add_argument("--dry-run", action="store_true", help="list what would be transcoded")
    p.set_defaults(func=cmd_transcode)

    p = sub.add_parser("copy", help="copy decodable videos from a folder/glob into data/uploads")
    p.add_argument("pattern", help="glob or directory, e.g. '/path/datasets/Oxford*/**/*.mp4'")
    p.add_argument("--max", type=int, default=5, help="max videos to copy (default 5)")
    p.add_argument("--to", default=UPLOADS_DIR, help="destination dir (default data/uploads)")
    p.add_argument("--dry-run", action="store_true", help="list matches without copying")
    p.set_defaults(func=cmd_copy)

    args = ap.parse_args()
    sys.exit(args.func(args) or 0)


if __name__ == "__main__":
    main()
