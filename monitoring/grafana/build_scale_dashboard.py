"""Generates dashboards/iris-scale.json ("IRIS - system health"). Run after editing:

    python3 monitoring/grafana/build_scale_dashboard.py

Every stat panel is coloured green / yellow / red by the thresholds below, so a quick look at the
Overview row says whether something is about to break; the rows underneath say where.
"""
import json
import pathlib

DS = {"type": "prometheus", "uid": "prometheus"}
G, Y, R = "green", "#EAB839", "red"

panels: list[dict] = []
_id = 0
_y = 0


def _next() -> int:
    global _id
    _id += 1
    return _id


def _steps(warn, crit, invert=False):
    """Thresholds: values >= warn yellow, >= crit red; invert=True for 'low is bad'."""
    if warn is None:
        return {"mode": "absolute", "steps": [{"color": G, "value": None}]}
    if invert:   # bad when the value falls BELOW crit / warn
        return {"mode": "absolute", "steps": [{"color": R, "value": None}, {"color": Y, "value": crit},
                                              {"color": G, "value": warn}]}
    return {"mode": "absolute", "steps": [{"color": G, "value": None}, {"color": Y, "value": warn},
                                          {"color": R, "value": crit}]}


def row(title: str) -> None:
    global _y
    panels.append({"type": "row", "title": title, "id": _next(), "collapsed": False,
                   "gridPos": {"h": 1, "w": 24, "x": 0, "y": _y}, "panels": []})
    _y += 1


def stat(title, expr, x, w=4, unit="none", warn=None, crit=None, invert=False, decimals=None,
         desc="", mappings=None, h=4):
    p = {"type": "stat", "title": title, "id": _next(), "datasource": DS, "description": desc,
         "gridPos": {"h": h, "w": w, "x": x, "y": _y},
         "targets": [{"expr": expr, "refId": "A", "datasource": DS}],
         "fieldConfig": {"defaults": {"unit": unit, "thresholds": _steps(warn, crit, invert),
                                      "color": {"mode": "thresholds"}, "mappings": mappings or []},
                         "overrides": []},
         "options": {"colorMode": "background", "graphMode": "area", "textMode": "value",
                     "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False}}}
    if decimals is not None:
        p["fieldConfig"]["defaults"]["decimals"] = decimals
    panels.append(p)


def ts(title, exprs, x, w=8, unit="none", warn=None, crit=None, invert=False, h=8, desc="",
       minv=None, maxv=None, stack=False):
    """exprs: list of (expr, legend)."""
    d = {"unit": unit, "thresholds": _steps(warn, crit, invert),
         "custom": {"drawStyle": "line", "lineWidth": 1, "fillOpacity": 8, "showPoints": "never",
                    "spanNulls": False, "stacking": {"mode": "normal" if stack else "none", "group": "A"},
                    # draw the thresholds as lines so the margin to trouble is visible
                    "thresholdsStyle": {"mode": "line" if warn is not None else "off"}}}
    if minv is not None:
        d["min"] = minv
    if maxv is not None:
        d["max"] = maxv
    panels.append({"type": "timeseries", "title": title, "id": _next(), "datasource": DS, "description": desc,
                   "gridPos": {"h": h, "w": w, "x": x, "y": _y},
                   "targets": [{"expr": e, "legendFormat": lg, "refId": chr(65 + i), "datasource": DS}
                               for i, (e, lg) in enumerate(exprs)],
                   "fieldConfig": {"defaults": d, "overrides": []},
                   "options": {"legend": {"showLegend": True, "displayMode": "list", "placement": "bottom",
                                          "calcs": ["lastNotNull", "max"]},
                               "tooltip": {"mode": "multi", "sort": "desc"}}})


def table(title, exprs, x, w=24, h=10, desc="", overrides=None, key="cam"):
    """exprs: list of (expr, column name); joined on `cam`."""
    panels.append({"type": "table", "title": title, "id": _next(), "datasource": DS, "description": desc,
                   "gridPos": {"h": h, "w": w, "x": x, "y": _y},
                   "targets": [{"expr": e, "legendFormat": "{{%s}}" % key, "refId": chr(65 + i), "format": "table",
                                "instant": True, "datasource": DS} for i, (e, _n) in enumerate(exprs)],
                   "transformations": [
                       {"id": "joinByField", "options": {"byField": key, "mode": "outer"}},
                       {"id": "organize", "options": {
                           "excludeByName": {**{f"Time {chr(65 + i)}": True for i in range(len(exprs))},
                                             "Time": True},
                           "renameByName": {f"Value #{chr(65 + i)}": n for i, (_e, n) in enumerate(exprs)},
                           "indexByName": {key: 0}}},
                       {"id": "sortBy", "options": {"fields": {}, "sort": [{"field": key}]}}],
                   "fieldConfig": {"defaults": {"custom": {"cellOptions": {"type": "color-background"}}},
                                   "overrides": overrides or []},
                   "options": {"showHeader": True, "cellHeight": "sm"}})


LOKI = {"type": "loki", "uid": "loki"}


def logs(title, expr, x, w=24, h=10, desc=""):
    panels.append({"type": "logs", "title": title, "id": _next(), "datasource": LOKI, "description": desc,
                   "gridPos": {"h": h, "w": w, "x": x, "y": _y},
                   "targets": [{"expr": expr, "refId": "A", "datasource": LOKI, "queryType": "range"}],
                   "options": {"showTime": True, "wrapLogMessage": True, "sortOrder": "Descending",
                               "enableLogDetails": True, "dedupStrategy": "none"}})


def problems_table(title, expr, x, w=24, h=8, desc=""):
    """Firing alerts as rows: severity, component, what, which camera/container."""
    panels.append({"type": "table", "title": title, "id": _next(), "datasource": DS, "description": desc,
                   "gridPos": {"h": h, "w": w, "x": x, "y": _y},
                   "targets": [{"expr": expr, "refId": "A", "format": "table", "instant": True, "datasource": DS}],
                   "transformations": [{"id": "organize", "options": {
                       "excludeByName": {"Time": True, "__name__": True, "alertstate": True, "job": True,
                                         "instance": True, "Value": True, "name": True},
                       "indexByName": {"severity": 0, "component": 1, "alertname": 2, "summary": 3},
                       "renameByName": {"severity": "level", "alertname": "problem", "summary": "what it means"}}}],
                   "fieldConfig": {"defaults": {"custom": {"cellOptions": {"type": "auto"}}},
                                   "overrides": [{"matcher": {"id": "byName", "options": "level"},
                                                  "properties": [{"id": "mappings", "value": [{"type": "value", "options": {
                                                      "critical": {"text": "CRITICAL", "color": "red"},
                                                      "warning": {"text": "warning", "color": "#EAB839"}}}]},
                                                                 {"id": "custom.cellOptions", "value": {"type": "color-background"}}]}]},
                   "options": {"showHeader": True, "cellHeight": "sm"}})


def adv(h):
    global _y
    _y += h


def col_override(name, unit, warn, crit, invert=False, mappings=None):
    return {"matcher": {"id": "byName", "options": name},
            "properties": [{"id": "unit", "value": unit},
                           {"id": "thresholds", "value": _steps(warn, crit, invert)},
                           {"id": "custom.cellOptions", "value": {"type": "color-background"}},
                           *([{"id": "mappings", "value": mappings}] if mappings else [])]}


YESNO = [{"type": "value", "options": {"1": {"text": "yes", "color": G}, "0": {"text": "NO", "color": R}}}]
G_ = 'iris_gauge{name="%s"}'

# =============================================================================================
row("Active problems - what cannot keep up right now (rules: monitoring/alerts.scale.yml)")
stat("Critical", 'count(ALERTS{alertstate="firing",severity="critical"}) or vector(0)', 0, w=3, warn=1, crit=1,
     desc="Rules in the critical state: something is lost or about to be.")
stat("Warnings", 'count(ALERTS{alertstate="firing",severity="warning"}) or vector(0)', 3, w=3, warn=1, crit=4,
     desc="A component is falling behind another (drops, full queues, no headroom).")
stat("Pending", 'count(ALERTS{alertstate="pending"}) or vector(0)', 6, w=3, warn=1, crit=5,
     desc="Conditions that are true but have not lasted long enough to fire yet.")
stat("Detections lost/min", 'sum(rate(iris_gauge{name=~"pub_drops_total_cam-.*"}[2m])) * 60',
     9, w=3, warn=1, crit=1, desc="Detection metadata really lost (the meta-only safety queue overflowed). Should always be 0.")
stat("Previews shed/min", 'sum(rate(iris_gauge{name=~"pub_shed_total_cam-.*"}[2m])) * 60',
     12, w=3, warn=30, crit=300, desc="Annotated preview frames skipped because the publisher is behind; their detections are still published.")
stat("GPU throttles (30 min)", 'sum(increase(iris_gpu_throttle_events_total[30m])) or vector(0)',
     15, w=3, warn=1, crit=3, desc="Times a throttle reason started; each one is in events.log with power limit, AC and temperatures.")
stat("Late frames/min (QoS)", 'sum(rate(iris_gauge{name="qos_dropped_total"}[2m])) * 60 or vector(0)',
     18, w=3, warn=12, crit=120, desc="GStreamer elements dropping frames as too late.")
adv(4)
problems_table("Firing now", 'ALERTS{alertstate="firing"}', 0, w=24, h=8,
               desc="One row per firing rule (per camera / container where it applies).")
adv(8)

row("Overview - is anything about to break?")
stat("Cameras alive", 'count(iris_camera_meta_age_seconds < 30) or vector(0)', 0, unit="none", warn=15, crit=13,
     invert=True, desc="Cameras the sidecar published a frame for in the last 30 s (of 15).")
stat("Worst camera silence", "max(iris_camera_meta_age_seconds)", 4, unit="s", warn=5, crit=30,
     desc="Seconds since the quietest camera was last seen by the pipeline.")
stat("GPU core", "max(iris_gpu_util_percent)", 8, unit="percent", warn=85, crit=95)
stat("NVDEC (decoders)", "max(iris_gpu_decoder_util_percent)", 12, unit="percent", warn=80, crit=95,
     desc="Hardware video decoders. At 100 % frames start to be dropped.")
stat("NVENC (encoders)", "max(iris_gpu_encoder_util_percent)", 16, unit="percent", warn=80, crit=95)
stat("GPU throttled", 'max(iris_gpu_throttle{reason=~"sw_power_cap|hw_slowdown|sw_thermal|hw_thermal|hw_power_brake"})',
     20, warn=1, crit=1, mappings=[{"type": "value", "options": {"0": {"text": "no"}, "1": {"text": "YES"}}}],
     desc="The GPU is slowing itself down (power cap, heat, hardware slowdown).")
adv(4)
stat("VLM backlog", G_ % "vlm_queue_depth", 0, warn=5, crit=15, desc="Association questions waiting for the VLM.")
stat("VLM slowest answer", G_ % "vlm_compare_max_s", 4, unit="s", warn=10, crit=30,
     desc="Slowest of the last 20 VLM calls.")
stat("VLM errors (10 min)", 'increase(iris_gauge{name="vlm_errors_total"}[10m])', 8, warn=1, crit=5)
stat("Platform feed backlog", "iris_truck_feed_backlog", 12, warn=50, crit=500,
     desc="Batches waiting for the truck-tracking platform to read them.")
stat("Platform ready", "iris_platform_ready", 16, mappings=YESNO, warn=1, crit=1, invert=True)
stat("Containers down", 'count(iris_container_up == 0) or vector(0)', 20, warn=1, crit=1)
adv(4)

# =============================================================================================
row("GPU - engines, memory, thermals")
ts("Core / decoder / encoder %", [("iris_gpu_util_percent", "core gpu{{gpu}}"),
                                  ("iris_gpu_decoder_util_percent", "NVDEC gpu{{gpu}}"),
                                  ("iris_gpu_encoder_util_percent", "NVENC gpu{{gpu}}")],
   0, unit="percent", warn=85, crit=95, minv=0, maxv=100,
   desc="NVDEC is the binding resource for many 1080p/4K streams: it saturates before the core does.")
ts("Memory used", [("iris_gpu_memory_used_bytes / iris_gpu_memory_total_bytes * 100", "gpu{{gpu}} %")],
   8, unit="percent", warn=85, crit=95, minv=0, maxv=100)
ts("Temperature / power", [("iris_gpu_temperature_celsius", "temp gpu{{gpu}} C"),
                           ("iris_gpu_power_watts", "power gpu{{gpu}} W")], 16, warn=80, crit=87)
adv(8)
ts("Power draw vs power limit (W)", [("iris_gpu_power_watts", "draw gpu{{gpu}}"),
                                     ("iris_gpu_power_limit_watts", "limit gpu{{gpu}}")], 0, unit="watt",
   desc="A laptop GPU's limit moves (power is shared with the CPU); a throttle starts when draw meets the limit.")
ts("Held back by policy (s per min)", [("rate(iris_gpu_violation_seconds_total{policy=~\"power|thermal|board_limit|reliability\"}[2m]) * 60", "{{policy}}"),
                                           ("sum(rate(iris_gpu_throttle_blips_total[5m])) * 60", "flickers/min (<3 s, not logged)")], 8,
   unit="s", warn=1, crit=10, desc="NVML violation status: seconds of every minute the GPU ran below its best clock because of a policy.")
ts("Host: CPU temperature, AC adapter", [("iris_host_cpu_temperature_celsius", "CPU C"), ("iris_host_ac_online * 100", "AC online (100 = yes)")], 16)
adv(8)
ts("Clocks (SM vs max) - falls when throttled",
   [("iris_gpu_sm_clock_mhz", "SM gpu{{gpu}}"), ("iris_gpu_sm_clock_max_mhz", "max gpu{{gpu}}")], 0, unit="MHz")
ts("Throttle reasons (1 = active)",
   [('iris_gpu_throttle{reason!~"gpu_idle|app_clocks|sync_boost"}', "{{reason}} gpu{{gpu}}")], 8,
   warn=1, crit=1, minv=0, maxv=1)
ts("NVENC sessions / PCIe", [("iris_gpu_encoder_sessions", "enc sessions gpu{{gpu}}"),
                             ("iris_gpu_pcie_rx_kbps / 1024", "PCIe rx MB/s"),
                             ("iris_gpu_pcie_tx_kbps / 1024", "PCIe tx MB/s")], 16)
adv(8)

# =============================================================================================
row("Cameras - input, rate, drops")
cam_over = [
    col_override("input alive", "none", 1, 1, invert=True, mappings=YESNO),
    col_override("player alive", "none", 1, 1, invert=True, mappings=YESNO),
    col_override("silent for", "s", 5, 30),
    col_override("fps", "none", 12, 6, invert=True),
    col_override("restarts (1 h)", "none", 1, 3),
    col_override("input Mbit/s", "none", None, None),
]
table("Per camera", [
    ("iris_go2rtc_input_alive", "input alive"),
    ("iris_player_alive", "player alive"),
    ("iris_camera_meta_age_seconds", "silent for"),
    ('label_replace(iris_gauge{name=~"fps_cam-.*"}, "cam", "$1", "name", "fps_(.*)")', "fps"),
    ('label_replace(increase(iris_gauge{name=~"stall_restarts_total_cam-.*"}[1h]), "cam", "$1", "name", "stall_restarts_total_(.*)")',
     "restarts (1 h)"),
    ("iris_go2rtc_bitrate_bps / 1e6", "input Mbit/s")], 0, overrides=cam_over,
    desc="input alive: the footage/camera feed reaches go2rtc.  silent for: time since the pipeline last "
         "published this camera.  restarts: source restarts by the stall watchdog.")
adv(10)
ts("Analysis rate per camera (fps)",
   [('iris_gauge{name=~"fps_cam-.*"}', "{{name}}")], 0, w=12, unit="none", warn=12, crit=6, invert=True,
   desc="Frames per second that got through inference. A dead source reads 0.")
ts("Camera silence (s)", [("iris_camera_meta_age_seconds", "{{cam}}")], 12, w=12, unit="s", warn=5, crit=30)
adv(8)

# =============================================================================================
row("Can't keep up - drops, waits, backlogs per component")
ts("Previews shed/s per camera", [('rate(iris_gauge{name=~"pub_shed_total_cam-.*"}[1m])', "{{name}}"),
                                 ('sum(rate(iris_gauge{name=~"pub_drops_total_cam-.*"}[1m]))', "DETECTIONS LOST/s")], 0, warn=1, crit=5,
   desc="A shed preview skips drawing/encoding only; its detections are still published. 'DETECTIONS LOST' must stay 0.")
ts("Publisher queue wait per camera (ms)", [('iris_gauge{name=~"pub_wait_ms_cam-.*"}', "{{name}}")], 8, unit="ms", warn=500, crit=1000,
   desc="How long the oldest recent job waited for its drawing worker; rising = the worker cannot keep up.")
ts("Publisher queue depth per camera", [('iris_gauge{name=~"pub_depth_cam-.*"}', "{{name}}"),
                                        (G_ % "pub_meta_q_depth", "meta-only queue"),
                                        (G_ % "pub_encoders", "GPU JPEG encoders")], 16, warn=6, crit=8)
adv(8)
ts("go2rtc -> sidecar bitrate per camera (Mbit/s)", [('iris_go2rtc_sent_bps / 1e6', "{{cam}}")], 0, w=8,
   desc="During a source stall this keeps flowing while the camera's frames stop: the sidecar, not the stream, is stuck.")
ts("Source stalls", [(G_ % "source_stalls_open", "cameras silent now"),
                     ('sum(iris_gauge{name=~"source_stall_episodes_total_cam-.*"})', "episodes since start"),
                     (G_ % "decoder_threads_sleeping", "decoder threads in a sleep loop")], 8, w=8, warn=1, crit=1,
   desc="[source-stall] START/END lines in events.log carry the evidence (queue levels, decoder-thread census).")
adv(8)
ts("Analysis queues vs their cap", [('iris_gauge{name=~"mux_queue_[0-9]+"}', "{{name}}"),
                                    ('iris_gauge{name="mux_queue_cap"}', "cap")], 0,
   desc="All lines on the cap = decoders are blocked waiting for the GPU/probe stage (no headroom).")
ts("Worker backlogs", [(G_ % "q_embed_depth", "Re-ID embed queue"), (G_ % "q_gallery_depth", "crop gallery queue"),
                       (G_ % "vlm_queue_depth", "VLM queue"), (G_ % "publisher_pending", "publisher (all cameras)")], 8)
ts("GStreamer late-frame drops/min", [('rate(iris_gauge{name="qos_dropped_total"}[2m]) * 60', "QoS drops/min"),
                                      ('rate(iris_gauge{name="flight_pressure_events_total"}[10m]) * 600', "[backpressure] log events / 10 min")],
   16, desc="QoS = an element dropped a frame as too late.")
adv(8)

# =============================================================================================
row("Pipeline - queues, stage time, drops")
ts("GPU stage busy %", [('iris_stage_busy_percent{stage=~"gpu.*"}', "{{stage}}")], 0, unit="percent",
   warn=80, crit=95, desc="Share of wall time each GPU element is busy (probe-measured).")
ts("Stage time per call (ms)", [("iris_stage_avg_ms", "{{stage}}")], 8, unit="ms")
ts("Analysis queue fill (buffers)", [('iris_gauge{name=~"mux_queue_.*"}', "{{name}}")], 16, warn=8, crit=14,
   desc="A queue pinned at its maximum is where back-pressure starts: the mux/decoder is falling behind.")
adv(8)
ts("Publisher pending / drops", [(G_ % "publisher_pending", "pending"),
                                 ('rate(iris_gauge{name="publisher_drops"}[1m])', "drops/s")], 0, warn=50, crit=200)
ts("Redis memory / clients", [("iris_redis_memory_bytes / 1048576", "memory MiB"),
                              ("iris_redis_clients", "clients")], 8)
ts("Redis ops/s and stream lengths", [("iris_redis_ops_per_second", "ops/s"),
                                      ("iris_redis_stream_length", "{{stream}}")], 16)
adv(8)

# =============================================================================================
row("VLM and association")
ts("VLM backlog / in flight", [(G_ % "vlm_queue_depth", "queued"), (G_ % "vlm_in_flight", "in flight")], 0,
   warn=5, crit=15)
ts("VLM answer time (s)", [(G_ % "vlm_compare_avg_s", "avg of last 20"), (G_ % "vlm_compare_max_s", "max of last 20"),
                           (G_ % "vlm_wait_max_s", "max queue wait")], 8, unit="s", warn=10, crit=30)
ts("VLM verdicts per minute", [('rate(iris_gauge{name="vlm_same_total"}[2m]) * 60', "same"),
                               ('rate(iris_gauge{name="vlm_different_total"}[2m]) * 60', "different"),
                               ('rate(iris_gauge{name="vlm_errors_total"}[2m]) * 60', "errors")], 16)
adv(8)
stat("Seconds since last VLM answer", G_ % "vlm_last_answer_age_s", 0, unit="s", warn=600, crit=1800,
     desc="-1 = none yet. Large while association work exists means the VLM is stuck.")
stat("Open vehicle tracks", G_ % "mtmc_tracks_open", 4)
stat("Pending verdicts", G_ % "mtmc_pending_verdicts", 8, warn=5, crit=15)
stat("Exits closed", G_ % "mtmc_exits_total", 12)
stat("Not tracked", G_ % "mtmc_untracked_total", 16)
stat("Next global id", G_ % "mtmc_next_gid", 20)
adv(4)

# =============================================================================================
row("Platform (truck-tracking) and the camera system app")
ts("Feed backlog and age of last push", [("iris_truck_feed_backlog", "backlog (batches)"),
                                         ("iris_truck_feed_last_push_age_seconds", "last push age s")], 0,
   warn=50, crit=500)
ts("Feed events, 5 min (by status)", [("iris_platform_feed_events_5m", "{{status}}")], 8,
   desc="rejected / failed should stay 0.")
ts("Newest stored sighting (s ago)", [("iris_platform_newest_sighting_age_seconds", "age")], 16, unit="s",
   warn=600, crit=1800)
adv(8)
ts("Sightings sent / dropped as stale", [("rate(iris_truck_feed_sightings_sent_total[5m]) * 60", "sent/min"),
                                         ("rate(iris_truck_feed_stale_dropped_total[5m]) * 60", "stale-dropped/min"),
                                         ("rate(iris_truck_feed_push_failures_total[5m]) * 60", "push failures/min")], 0)
ts("Tracking loop lag / poll errors", [("iris_tracking_loop_age_seconds", "loop age s"),
                                       ("increase(iris_tracking_poll_errors_total[5m])", "errors (5 m)")], 8,
   warn=5, crit=30)
ts("Trucks / sightings stored", [("iris_platform_trucks_total", "trucks"), ("iris_platform_sightings_total", "sightings")], 16)
adv(8)

# =============================================================================================
row("Stall forensics - what the sidecar was doing when cameras went quiet")
stat("Main loop lag", G_ % "flight_main_lag_s", 0, unit="s", warn=3, crit=10,
     desc="How long the sidecar's GLib main loop has not turned (the watchdog, bus handling and source "
          "restarts all run on it). Red = the loop is blocked; the stack of every thread is then written "
          "to data/flight/stall-*.txt.")
stat("Scheduling delay", G_ % "flight_sched_delay_ms", 4, unit="ms", warn=200, crit=1000,
     desc="How late a plain watcher thread wakes from a 2 s sleep: CPU or GIL starvation shows here first.")
stat("Stalls recorded", G_ % "flight_stalls_total", 8, warn=1, crit=1)
stat("Sidecar CPU", G_ % "flight_cpu_pct", 12, unit="percent", warn=1000, crit=1400,
     desc="100 = one core; this machine has 16 threads.")
stat("Sidecar threads", G_ % "flight_threads", 16)
stat("Publisher drops/min", 'rate(iris_gauge{name="publisher_drops"}[2m]) * 60', 20, warn=60, crit=600,
     desc="Annotated frames dropped because the publisher threads cannot keep up (CPU drawing).")
adv(4)
ts("Main-loop lag and scheduling delay", [(G_ % "flight_main_lag_s", "main loop lag s"),
                                         (G_ % "flight_sched_delay_ms" + " / 1000", "sched delay s")], 0,
   unit="s", warn=3, crit=10)
ts("Bus events per minute (all cameras)",
   [('sum by (kind) (label_replace(rate(iris_gauge{name=~"bus_.*_total_cam-.*"}[2m]) * 60, "kind", "$1", "name", "bus_(.*)_total_cam-.*"))', "{{kind}}")], 8,
   desc="nodata = 'No data from source since last 10 s'; writefail = 'Could not write to resource' (the "
        "reconnect that never recovers); error/warn = everything else on the bus.")
ts("CPU and publisher drops", [(G_ % "flight_cpu_pct", "process CPU %"),
                               ('rate(iris_gauge{name="publisher_drops"}[1m])', "dropped frames/s")], 16)
adv(8)
table("Per-camera bus events (total since start)", [
    ('label_replace(iris_gauge{name=~"bus_nodata_total_cam-.*"}, "cam", "$1", "name", "bus_nodata_total_(.*)")', "no data"),
    ('label_replace(iris_gauge{name=~"bus_writefail_total_cam-.*"}, "cam", "$1", "name", "bus_writefail_total_(.*)")', "write failed"),
    ('label_replace(iris_gauge{name=~"bus_error_total_cam-.*"}, "cam", "$1", "name", "bus_error_total_(.*)")', "errors"),
    ('label_replace(iris_gauge{name=~"bus_warn_total_cam-.*"}, "cam", "$1", "name", "bus_warn_total_(.*)")', "other warnings"),
    ('label_replace(iris_gauge{name=~"stall_restarts_total_cam-.*"}, "cam", "$1", "name", "stall_restarts_total_(.*)")', "watchdog restarts"),
    ("iris_camera_meta_age_seconds", "silent for (s)")], 0, w=12, h=9,
    overrides=[col_override("no data", "none", 1, 5), col_override("write failed", "none", 1, 3),
               col_override("errors", "none", 1, 3), col_override("watchdog restarts", "none", 1, 3),
               col_override("silent for (s)", "s", 5, 30)])
logs("Sidecar warnings and errors (Loki)",
     '{container="scale-deepstream-sidecar-1"} |~ `\\[(WARNING|ERROR|CRITICAL)\\]|\\[watchdog\\]|\\[flight\\]|\\[backpressure\\]|Resetting source`',
     12, w=12, h=9, desc="The same time range as the graphs. Stall dumps are announced here with the blocked call.")
adv(9)
logs("Other containers: warnings and errors (Loki)",
     '{container=~"scale-.*", container!="scale-deepstream-sidecar-1"} |~ `\\[(WARNING|ERROR|CRITICAL)\\]|ERROR|Error|error:`',
     0, h=8)
adv(8)

# =============================================================================================
row("Containers and host")
table("Containers", [
    ("iris_container_up", "up"),
    ("iris_container_cpu_percent", "CPU % (100 = 1 core)"),
    ("iris_container_memory_bytes / 1048576", "memory MiB"),
    ("iris_container_memory_bytes / (iris_container_memory_limit_bytes > 0) * 100",
     "memory % of limit"),
    ("iris_container_restarts", "restarts"),
    ("iris_container_oom_killed", "OOM killed")],
    0, h=9, key="name", overrides=[col_override("up", "none", 1, 1, invert=True, mappings=YESNO),
                       col_override("memory % of limit", "percent", 80, 95),
                       col_override("restarts", "none", 1, 3),
                       col_override("OOM killed", "none", 1, 1)])
adv(9)
ts("Host CPU %", [('100 - avg(rate(node_cpu_seconds_total{mode="idle"}[1m])) * 100', "cpu")], 0, unit="percent",
   warn=85, crit=95, minv=0, maxv=100)
ts("Host memory used %", [("(1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100", "memory")],
   8, unit="percent", warn=85, crit=95, minv=0, maxv=100)
ts("Disk free %", [("iris_disk_free_ratio * 100", "{{label}}")], 16, unit="percent", warn=15, crit=5,
   invert=True, desc="data = the camera system's data dir; footage = the recordings drive.")
adv(8)
ts("Exporter polls (1 = ok)", [("iris_probe_up", "{{source}}")], 0, w=12, minv=0, maxv=1, warn=1, crit=1,
   invert=True, desc="A source at 0 means that part of this dashboard is blind.")
ts("Scrape targets up", [("up", "{{job}}")], 12, w=12, minv=0, maxv=1, warn=1, crit=1, invert=True)

dash = {"title": "IRIS - system health", "uid": "iris-scale-health", "schemaVersion": 39, "version": 1,
        "editable": True, "refresh": "10s", "time": {"from": "now-30m", "to": "now"},
        "tags": ["iris", "health"], "templating": {"list": []},
        "annotations": {"list": [
            {"name": "Sidecar errors", "datasource": LOKI, "enable": True, "iconColor": "red",
             "expr": '{container="scale-deepstream-sidecar-1"} |~ `\\[ERROR\\]|\\[flight\\]|\\[backpressure\\]`', "titleFormat": "sidecar", "textFormat": "{{level}}"},
            {"name": "Source resets", "datasource": LOKI, "enable": True, "iconColor": "orange",
             "expr": '{container="scale-deepstream-sidecar-1"} |~ "Resetting source|No data from source"', "titleFormat": "source reset"}]},
        "panels": panels}

out = pathlib.Path(__file__).parent / "dashboards" / "iris-scale.json"
out.write_text(json.dumps(dash, indent=1, ensure_ascii=False) + "\n")
print(f"{out}: {len([p for p in panels if p['type'] != 'row'])} panels")
