"""Pure helpers for the exporter's event log: GPU throttle start/end tracking, alert firing/resolved
detection, laptop context (AC adapter, CPU temperature) and the one-line event format.

Kept free of NVML / network so it can be unit tested."""
from __future__ import annotations

import glob
import json
import os
import time

# not trouble: the GPU idling, clocks pinned by the application, boost sync
BENIGN = {"gpu_idle", "app_clocks", "sync_boost"}


def event_line(component: str, text: str, now: float | None = None) -> str:
    """'2026-10-04T07:41:03Z [event] gpu: ...' (UTC, one line per event, greppable)."""
    t = time.gmtime(time.time() if now is None else now)
    return f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', t)} [event] {component}: {text}"


def append_event(path: str, line: str) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(line.rstrip("\n") + "\n")
    except OSError:
        pass   # the log must never break the exporter


class ThrottleTracker:
    """Turns the GPU's throttle-reason flags into start / end events with durations.

    A throttle that clears within ``min_s`` seconds is only a blip (the driver's power cap flickers for
    a second when the load jumps): it is reported as a "blip" event (counted, not logged) so the log
    holds the throttles that matter. A longer one is announced when it passes ``min_s`` (with its real
    start time) and again when it ends."""

    def __init__(self, min_s: float = 3.0):
        self.min_s = min_s
        self._since: dict[str, float] = {}
        self._worst: dict[str, dict] = {}
        self._announced: set[str] = set()

    def update(self, now: float, reasons: dict[str, int], ctx: dict) -> list[dict]:
        active = {r for r, v in reasons.items() if v and r not in BENIGN}
        events = []
        for r in sorted(active - set(self._since)):
            self._since[r] = now
            self._worst[r] = dict(ctx)
        for r in sorted(active & set(self._since)):
            # keep the lowest power limit and highest temperature seen while it lasted
            w = self._worst[r]
            for k, better in (("power_limit_w", min), ("temp_c", max), ("power_w", max)):
                if ctx.get(k) is not None:
                    w[k] = better(w.get(k, ctx[k]), ctx[k])
            if r not in self._announced and now - self._since[r] >= self.min_s:
                self._announced.add(r)
                events.append({"kind": "start", "reason": r, "ctx": dict(self._worst[r]),
                               "since": self._since[r]})
        for r in sorted(set(self._since) - active):
            since = self._since.pop(r)
            worst = self._worst.pop(r, {})
            if r in self._announced:
                self._announced.discard(r)
                events.append({"kind": "end", "reason": r, "duration_s": now - since, "ctx": worst})
            else:
                events.append({"kind": "blip", "reason": r, "duration_s": now - since, "ctx": worst})
        return events


def format_throttle(ev: dict) -> str:
    c = ev["ctx"]
    bits = []
    if c.get("power_w") is not None and c.get("power_limit_w") is not None:
        bits.append(f"power {c['power_w']:.0f} W of limit {c['power_limit_w']:.0f} W")
    if c.get("sm_mhz") is not None and c.get("sm_max_mhz"):
        bits.append(f"SM clock {c['sm_mhz']:.0f} of max {c['sm_max_mhz']:.0f} MHz")
    if c.get("temp_c") is not None:
        bits.append(f"GPU {c['temp_c']:.0f} C")
    if c.get("cpu_temp_c") is not None:
        bits.append(f"CPU {c['cpu_temp_c']:.0f} C")
    if c.get("ac_online") is not None:
        bits.append("AC adapter " + ("online" if c["ac_online"] else "NOT connected (battery)"))
    if ev["kind"] == "start":
        return f"THROTTLE START {ev['reason']}: " + ", ".join(bits)
    return f"THROTTLE END {ev['reason']} after {ev['duration_s']:.0f} s (worst: " + ", ".join(bits) + ")"


def diff_alerts(prev: dict[str, dict], alerts: list[dict], now: float) -> tuple[dict[str, dict], list[dict]]:
    """prev: {key: {since, labels}} of alerts that were firing; alerts: Prometheus /api/v1/alerts data.
    Returns the new state and firing / resolved events."""
    state: dict[str, dict] = {}
    events = []
    for a in alerts:
        if a.get("state") != "firing":
            continue
        labels = a.get("labels", {})
        key = json.dumps(labels, sort_keys=True)
        if key in prev:
            state[key] = prev[key]
        else:
            state[key] = {"since": now, "labels": labels}
            events.append({"kind": "firing", "labels": labels})
    for key, v in prev.items():
        if key not in state:
            events.append({"kind": "resolved", "labels": v["labels"], "duration_s": now - v["since"]})
    return state, events


def format_alert(ev: dict) -> str:
    l = ev["labels"]
    which = " ".join(f"{k}={v}" for k, v in l.items()
                     if k not in ("alertname", "severity", "component", "summary", "alertstate"))
    head = f"{ev['kind'].upper()} {l.get('alertname', '?')} [{l.get('severity', '?')}] {l.get('component', '')}"
    if which:
        head += f" ({which})"
    if ev["kind"] == "resolved":
        return f"{head} after {ev['duration_s']:.0f} s"
    return f"{head}: {l.get('summary', '')}"


def read_ac_online(root: str = "/host/sys/class/power_supply") -> bool | None:
    """True/False when a mains adapter exists, None when unknown."""
    seen = None
    for d in glob.glob(os.path.join(root, "*")):
        try:
            if open(os.path.join(d, "type")).read().strip() == "Mains":
                seen = (seen or False) or open(os.path.join(d, "online")).read().strip() == "1"
        except OSError:
            continue
    return seen


def read_cpu_temp(root: str = "/host/sys/class/hwmon") -> float | None:
    """CPU package temperature in C (k10temp on AMD, coretemp on Intel), None if not readable."""
    for d in glob.glob(os.path.join(root, "hwmon*")):
        try:
            name = open(os.path.join(d, "name")).read().strip()
            if name in ("k10temp", "coretemp"):
                return int(open(os.path.join(d, "temp1_input")).read().strip()) / 1000.0
        except (OSError, ValueError):
            continue
    return None
