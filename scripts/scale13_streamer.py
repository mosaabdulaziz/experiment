#!/usr/bin/env python3
"""Play each camera's recordings back-to-back as a real-time RTSP stream, for as long as it runs.

The recordings are segments of a continuous day (each file starts where the last one ended),
so concatenating them in time order reproduces the camera's real timeline. Every camera starts
at the same footage clock time (--start), so the 13 streams are on one shared timeline, like 13
live cameras. ffmpeg -re paces the copy at the source frame rate; nothing is re-encoded. At the
end of the day a camera starts over from --start. A stream that stops is restarted.

  scale13_streamer.py run    --videos DIR --port 38554 [--start 2026-09-13T00:00:00]
  scale13_streamer.py stop | status

Timeline mode (--timeline, for a full-day run that must line up with the real clock):
every file starts at the footage time written in its name, a file is cut where the next one
begins (overlaps), a hole between files is filled with black frames in the camera's own codec
and size so the stream never goes silent, and a camera that is late or early re-aligns at the
next piece. --begin-at "2026-10-06T00:00:00" waits for that local wall time and starts the
footage clock then; --stop-at-end stops at the end of the recorded day instead of wrapping.
"""
import argparse, datetime as dt, glob, json, os, re, signal, subprocess, sys, time

CAMS = [("cam-01", "D41"), ("cam-02", "D16"), ("cam-03", "D52"), ("cam-04", "ch13"),
        ("cam-05", "ch22"), ("cam-06", "ch23"), ("cam-07", "ch06"), ("cam-08", "ch30"),
        ("cam-09", "ch24"), ("cam-10", "D15"), ("cam-11", "D18"), ("cam-12", "D31"),
        ("cam-13", "D28"),
        ("cam-14", "D05"), ("cam-15", "D20")]
# Cameras re-encoded to H.264 (capped bitrate) instead of stream-copied (none needed so far).
TRANSCODE = set()
# Cameras played one file at a time, each with its own ffmpeg, instead of through the concat
# demuxer. D28's concat is dropped by go2rtc ~100 s after it starts ("Broken pipe" /
# "streams: unknown error"), whatever the codec or timestamps; the same files pushed one at a
# time run fine. The price is a reconnect at each file boundary (every 27-60 min) for that camera.
# cam-04 (CH13, H.264) too: the concat demuxer cannot seek into its first file ("could not seek to
# position"), so it started from the top of the file, 22 min behind the others; a single-file ffmpeg
# seeks fine.
PER_FILE = {"cam-13", "cam-04"}
DAY = 86400.0
HOLE_MIN_S = 3.0   # a gap shorter than this is just a file boundary, not a hole to fill
STATE = os.environ.get("SCALE13_STATE", "/tmp/scale13-streamer")
TS = re.compile(r"_(\d{14})\.mp4$")


def recordings(videos, prefix):
    out = []
    for p in glob.glob(os.path.join(videos, f"{prefix}_*.mp4")):
        m = TS.search(p)
        if m:
            out.append((dt.datetime.strptime(m.group(1), "%Y%m%d%H%M%S"), p))
    return sorted(out)


def playlist(files, start):
    """concat-demuxer list that begins at `start` on this camera's timeline."""
    i = max((k for k, (t, _) in enumerate(files) if t <= start), default=0)
    lines = [f"file '{p}'" for _, p in files[i:]]
    # The offset into the first file is passed to ffmpeg as `-ss` on the concat input (see cmd): the concat
    # demuxer's own `inpoint` is ignored for stream copy, which started every camera at the top of its
    # file (cameras up to an hour apart on screen).
    return "\n".join(lines) + "\n", files[i][0], max(0.0, (start - files[i][0]).total_seconds())


def cmd(listfile, host, port, n, transcode=False, seek=0.0):
    codec = (["-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency", "-pix_fmt", "yuv420p",
              "-b:v", "3M", "-maxrate", "4M", "-bufsize", "6M", "-g", "40"]
             if transcode else ["-c", "copy"])
    return ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-re", *(["-ss", f"{seek:.3f}"] if seek else []), "-f", "concat", "-safe", "0",
            "-i", listfile, "-map", "0:v:0", *codec, "-an",
            "-max_muxing_queue_size", "1024", "-flush_packets", "1",
            "-rtsp_transport", "tcp", "-f", "rtsp", f"rtsp://{host}:{port}/raw-cam-{n}"]


def probe(path):
    """(duration s, codec, width, height, fps) of a recording's video stream, via ffprobe."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,width,height,r_frame_rate:format=duration", "-of", "json", path],
        capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    st = j["streams"][0]
    num, _, den = st.get("r_frame_rate", "20/1").partition("/")
    fps = float(num) / float(den or 1)
    return float(j["format"]["duration"]), st["codec_name"], int(st["width"]), int(st["height"]), fps


def probe_cached(path, cache):
    if path not in cache:
        cache[path] = list(probe(path))
    return cache[path]


def plan_timeline(entries, now):
    """Pieces that play one camera's recordings on the footage clock, from `now` on.

    `entries` is [(start datetime, path, (duration s, codec, w, h, fps))] sorted by start. Returns
    [{kind: 'file'|'fill', start, end, path, seek, codec, w, h, fps}]: a file piece plays from `seek`
    seconds into the file between `start` and `end` (the file's own end, or where the next file begins
    if they overlap); a fill piece is black video covering a hole, in the codec of the file before it
    (of the one after it when there is none). Pieces that end before `now` are left out and the one
    spanning `now` begins late (its seek and start advance by the lateness).
    """
    pieces = []
    for i, (t0, path, (dur, codec, w, h, fps)) in enumerate(entries):
        end = t0 + dt.timedelta(seconds=dur)
        if i + 1 < len(entries):
            end = min(end, entries[i + 1][0])
        if pieces and t0 > pieces[-1]["end"] + dt.timedelta(seconds=HOLE_MIN_S):
            pieces.append({"kind": "fill", "start": pieces[-1]["end"], "end": t0, "path": None, "seek": 0.0,
                           "codec": pieces[-1]["codec"], "w": pieces[-1]["w"], "h": pieces[-1]["h"], "fps": pieces[-1]["fps"]})
        pieces.append({"kind": "file", "start": t0, "end": end, "path": path, "seek": 0.0,
                       "codec": codec, "w": w, "h": h, "fps": fps})
    out = []
    for pc in pieces:
        if pc["end"] <= now:
            continue
        if pc["start"] < now:
            late = (now - pc["start"]).total_seconds()
            pc = {**pc, "start": now, "seek": pc["seek"] + late if pc["kind"] == "file" else 0.0}
        out.append(pc)
    return out


def piece_cmd(pc, host, port, n, now):
    """ffmpeg command line that pushes one piece to the camera's RTSP path."""
    secs = max(0.1, (pc["end"] - max(now, pc["start"])).total_seconds())
    url = f"rtsp://{host}:{port}/raw-cam-{n}"
    tail = ["-an", "-flush_packets", "1", "-rtsp_transport", "tcp", "-f", "rtsp", url]
    if pc["kind"] == "file":
        c = ["ffmpeg", "-hide_banner", "-loglevel", "warning"]
        if pc["seek"]:
            c += ["-ss", f"{pc['seek']:.3f}"]
        return c + ["-re", "-i", pc["path"], "-t", f"{secs:.3f}", "-map", "0:v:0", "-c", "copy",
                    "-max_muxing_queue_size", "1024"] + tail
    enc = (["-c:v", "libx265", "-preset", "ultrafast", "-x265-params", "log-level=error:keyint=40:bframes=0"]
           if pc["codec"] == "hevc" else
           ["-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency", "-g", "40", "-bf", "0"])
    return (["ffmpeg", "-hide_banner", "-loglevel", "warning", "-re", "-f", "lavfi", "-i",
             f"color=c=black:s={pc['w']}x{pc['h']}:r={pc['fps']:g}", "-t", f"{secs:.3f}"]
            + enc + ["-pix_fmt", "yuv420p"] + tail)


def playday(a):
    """One camera on the footage clock (see the module docstring, timeline mode). Exits 0 at the end of the
    recorded day, non-zero if a push fails (the supervisor then restarts it, rejoining at the clock)."""
    start = dt.datetime.fromisoformat(a.start)
    day0 = start.replace(hour=0, minute=0, second=0, microsecond=0)
    n = int(a.cam.split("-")[1])

    def footage_now():
        return day0 + dt.timedelta(seconds=(start - day0).total_seconds() + time.time() - a.t0)

    cache_path = os.path.join(STATE, "probe-cache.json")
    try:
        cache = json.load(open(cache_path))
    except (OSError, ValueError):
        cache = {}
    entries = [(t, p, tuple(probe_cached(p, cache))) for t, p in recordings(a.videos, a.prefix)]
    try:
        json.dump(cache, open(cache_path, "w"))
    except OSError:
        pass
    end_of_day = day0 + dt.timedelta(days=1)
    pieces = plan_timeline(entries, footage_now())
    if entries and pieces and footage_now() < pieces[0]["start"] - dt.timedelta(seconds=HOLE_MIN_S):
        # before the footage begins the camera shows black (pre-roll), so the sidecar can connect to a live stream
        first = pieces[0]
        pieces.insert(0, {"kind": "fill", "start": footage_now(), "end": first["start"], "path": None, "seek": 0.0,
                          "codec": first["codec"], "w": first["w"], "h": first["h"], "fps": first["fps"]})
    for pc in pieces:
        while True:                       # a piece never starts early: wait for the footage clock
            ahead = (pc["start"] - footage_now()).total_seconds()
            if ahead <= 0.05:
                break
            time.sleep(min(ahead, 1.0))
        now = footage_now()
        if a.stop_at_end and now >= end_of_day:
            return
        if pc["end"] <= now:
            continue                      # behind by a whole piece: skip to the next one
        pc = {**pc, "seek": pc["seek"] + (max(0.0, (now - pc["start"]).total_seconds()) if pc["kind"] == "file" else 0.0)}
        if pc["kind"] == "fill":
            print(f"{a.cam}: hole {pc['start']:%H:%M:%S}-{pc['end']:%H:%M:%S}, black {pc['w']}x{pc['h']} ({pc['codec']})", flush=True)
        r = subprocess.run(piece_cmd(pc, a.host, a.port, n, now), stdin=subprocess.DEVNULL)
        if r.returncode != 0:
            sys.exit(r.returncode)


def run(a):
    os.makedirs(STATE, exist_ok=True)
    open(os.path.join(STATE, "supervisor.pid"), "w").write(str(os.getpid()))
    start = dt.datetime.fromisoformat(a.start)
    t0 = time.time()
    if a.begin_at:                        # footage time `start` is reached at this local wall time
        t0 = dt.datetime.fromisoformat(a.begin_at).timestamp()
        print(f"waiting for {a.begin_at} to start the footage clock at {a.start}", flush=True)

    day0 = start.replace(hour=0, minute=0, second=0, microsecond=0)
    json.dump({"t0": t0, "start": a.start, "timeline": bool(a.timeline), "stop_at_end": bool(a.stop_at_end)},
              open(os.path.join(STATE, "clock.json"), "w"))   # for scale_day_status.py

    def footage_now():
        if a.timeline:
            return day0 + dt.timedelta(seconds=(start - day0).total_seconds() + time.time() - t0)
        # Every camera is on the same footage clock: `start` plus the time this has been running,
        # wrapping at the END of the recorded day (midnight), back to its beginning. Wrapping a
        # full day after `start` instead ran past the footage and every camera looped on nothing
        # (a run started at 11:49 only had 12 h of footage). A camera that restarts rejoins it.
        offset = (start - day0).total_seconds() + (time.time() - t0)
        return day0 + dt.timedelta(seconds=offset % DAY)

    def plan(cam, prefix):
        files = recordings(a.videos, prefix)
        if not files:
            return None
        text, _first, seek = playlist(files, footage_now())
        lf = os.path.join(STATE, f"{cam}.txt")
        open(lf, "w").write(text)
        return lf, text.splitlines()[0][6:-1], seek

    skip = {c.strip() for c in a.skip.split(",") if c.strip()}
    procs, stop = {}, []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    done = set()
    while not stop:
        if a.timeline and a.stop_at_end and footage_now() >= day0 + dt.timedelta(days=1):
            print("end of the recorded day: stopping", flush=True)
            break
        for cam, prefix in CAMS:
            if cam in skip or cam in done:
                continue
            p = procs.get(cam)
            if p is not None and p.poll() is None:
                continue
            if p is not None and p.returncode == 0 and a.timeline and a.stop_at_end:
                done.add(cam)
                continue
            if p is not None:
                print(f"{cam}: ffmpeg exited {p.returncode}; rejoining at footage time {footage_now():%H:%M:%S}", flush=True)
                time.sleep(2)
            made = plan(cam, prefix)
            if made is None:
                print(f"{cam}: no recordings for {prefix} in {a.videos}", flush=True)
                continue
            lf, first, seek = made
            n = int(cam.split("-")[1])
            log = open(os.path.join(STATE, f"{cam}.log"), "ab")
            if a.timeline:
                argv = [sys.executable, os.path.abspath(__file__), "playday", "--cam", cam, "--prefix", prefix,
                        "--videos", a.videos, "--host", a.host, "--port", str(a.port), "--start", a.start,
                        "--t0", f"{t0:.3f}"] + (["--stop-at-end"] if a.stop_at_end else [])
                procs[cam] = subprocess.Popen(argv, stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True)
                print(f"{cam} {prefix}: timeline player (footage {footage_now():%H:%M:%S})", flush=True)
                time.sleep(0.5)
                continue
            argv = ([sys.executable, os.path.abspath(__file__), "playfiles", "--cam", cam, "--list", lf,
                     "--host", a.host, "--port", str(a.port), "--seek", f"{seek:.3f}"] if (cam in PER_FILE or a.per_file == "all")
                    else cmd(lf, a.host, a.port, n, cam in TRANSCODE, seek))
            procs[cam] = subprocess.Popen(argv,
                                          stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                                          start_new_session=True)
            print(f"{cam} {prefix}: playing from {os.path.basename(first)} (footage {footage_now():%H:%M:%S})"
                  + (" [re-encoded]" if cam in TRANSCODE else " [file by file]" if (cam in PER_FILE or a.per_file == "all") else ""), flush=True)
            time.sleep(0.5)
        time.sleep(3)
    for p in procs.values():
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def playfiles(a):
    """Play the entries of a concat list one after another, one ffmpeg per file (see PER_FILE)."""
    entries = [[line[6:-1], 0.0] for line in open(a.list).read().splitlines() if line.startswith("file ")]
    if entries:
        entries[0][1] = a.seek
    n = int(a.cam.split("-")[1])
    for path, inpoint in entries:
        c = ["ffmpeg", "-hide_banner", "-loglevel", "warning"]
        if inpoint:
            c += ["-ss", f"{inpoint:.3f}"]
        c += ["-re", "-i", path, "-map", "0:v:0", "-c", "copy", "-an", "-max_muxing_queue_size", "1024",
              "-flush_packets", "1", "-rtsp_transport", "tcp", "-f", "rtsp", f"rtsp://{a.host}:{a.port}/raw-cam-{n}"]
        r = subprocess.run(c, stdin=subprocess.DEVNULL)
        if r.returncode != 0:
            sys.exit(r.returncode)   # the supervisor re-plans from the current footage time


def pid():
    try:
        return int(open(os.path.join(STATE, "supervisor.pid")).read())
    except (OSError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["run", "stop", "status", "playfiles", "playday"])
    ap.add_argument("--timeline", action="store_true", help="play on the footage clock: files at their recorded times, holes filled with black")
    ap.add_argument("--begin-at", default="", help='local wall time at which footage time --start begins, e.g. 2026-10-06T00:00:00')
    ap.add_argument("--stop-at-end", action="store_true", help="stop at the end of the recorded day instead of wrapping to its start")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--cam", default="")
    ap.add_argument("--list", default="")
    ap.add_argument("--seek", type=float, default=0.0)
    ap.add_argument("--videos", default="/data/IRIS/data/el-mohandes")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=38554)
    ap.add_argument("--start", default="2026-09-13T00:00:00")
    ap.add_argument("--per-file", default="", help='"all": play every camera file by file (for a machine where the concat '
                    'demuxer cannot seek into the first file, e.g. "could not seek to position" in the cam logs)')
    ap.add_argument("--skip", default="", help="comma-separated camera ids to leave out, e.g. cam-13")
    a = ap.parse_args()
    if a.action == "run":
        run(a)
    elif a.action == "playfiles":
        playfiles(a)
    elif a.action == "playday":
        playday(a)
    elif a.action == "stop":
        p = pid()
        if p:
            try:
                os.kill(p, signal.SIGTERM)
                print("stopped", p)
            except ProcessLookupError:
                print("not running")
        else:
            print("not running")
    else:
        p = pid()
        alive = bool(p) and os.path.exists(f"/proc/{p}")
        print("running" if alive else "not running", p or "")
        if alive:
            for cam, _ in CAMS:
                f = os.path.join(STATE, f"{cam}.txt")
                print(cam, "playlist" if os.path.exists(f) else "-")


if __name__ == "__main__":
    main()
