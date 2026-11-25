"""The safety stack's controller: cameras + per-camera feature toggles -> each role sidecar's config, and only the
role sidecars whose cameras changed restarted (products/safety/README.md).

Inputs
  products/safety/safety.yaml         the features, the role each extends, the settings every role shares
  products/safety/cameras.yaml        the cameras (SAFETY_CAMERAS), each with its default features
  <data>/safety/cameras_state.json    what the UI saved: {camera: {"name", "features"}} (overrides the defaults)
Outputs, in <data>/safety/generated/
  <role>.yaml                         a role sidecar's config: extends the role, the common settings, its cameras
  fall_detection.yaml                 the fall service's config: the fall role's cameras, the stack's zones file
  go2rtc.yaml                         the cameras' raw streams (demo clips are published into them)

Commands
  render          write the generated files; print each role's cameras and what is not applied yet
  apply           render, add new cameras to go2rtc (its API), then start / restart / stop the role sidecars whose
                  config differs from what was last applied (<data>/safety/applied.json; each step recorded once it
                  worked, so a failed apply is retried by the next)
  clips           list the demo cameras' clips (camera id, path), for `just safety-demo`
  watch           apply whenever the app asks (the UI's Apply button bumps iris:ctrl:safety_apply; the version applied
                  is written back to iris:analytics:safety_applied, so the UI can show "applying" vs "applied")

The app never gets the Docker socket: this runs on the host, beside the stack (`just safety-controller`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shlex
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))     # run as a script from tools/: products.safety.config is the repo's

from products.safety.config import (APPLY_APPLIED_KEY, APPLY_HOLD_S, APPLY_VERSION_KEY,  # noqa: E402
                                    APPLYING_PREFIX, ROLE_KEY_SEPARATOR, effective_cameras, load_cameras,
                                    load_product, load_state)

logger = logging.getLogger("safety_controller")

CONTAINER_REPO = "/workspace/iris"
SERVICE_PREFIX = "safety-"        # the compose service of role <r> is safety-<r> (docker-compose.safety.yml)
# <root>/data/safety/generated -> <root>: the generated configs extend the role files by this relative path, which
# resolves in every container (each mounts the stack's data dir at <root>/data).
GENERATED_TO_ROOT = "../../.."
APPLIED_FILE = "applied.json"     # <data>/safety/: a digest of each generated file as it was last applied


# ── rendering ─────────────────────────────────────────────────────────────────────────────────────────────────

def role_sources(cameras: list, product: dict) -> dict:
    """{role: [source, ...]}: every role, with the cameras that have its feature on (an empty list = not needed)."""
    return {role: [{"id": c["id"], "uri": f"rtsp://go2rtc:8554/raw-{c['id']}", "roles": [role]}
                   for c in cameras if role in c["features"]]
            for role in product["features"]}


def render_role(role: str, sources: list, product: dict) -> dict:
    """A role sidecar's config: the role's file, the shared settings, its own clip folder, its cameras."""
    return {"extends": f"{GENERATED_TO_ROOT}/{product['roles'][role]}", **(product.get("common") or {}),
            "sidecar_role": role,
            "svr_dir": f"{CONTAINER_REPO}/data/alert_svr/{role}", "svr_dir_app": f"data/alert_svr/{role}",
            "sources": sources}


def render_fall_service(sources: list, product: dict, repo: Path, data_dir: Path) -> dict:
    """The fall service's config (it runs on the host): the fall role's cameras and the stack's zones file, whose
    points are in the common mux size."""
    common = product.get("common") or {}
    return {"extends": str(Path(repo) / product["fall_service"]["extends"]),
            "cameras": [{"id": s["id"]} for s in sources],
            "zones": {"path": str(Path(data_dir) / "zones_config.txt"),
                      "mux": [common.get("mux_width", 1280), common.get("mux_height", 720)]}}


def render_go2rtc(cameras: list) -> dict:
    """go2rtc: raw-<id> is what the role sidecars pull (a real camera's RTSP, or empty for a demo clip the streamer
    publishes into); <id> is the browsers' stream, transcoded to H.264 on demand."""
    streams = {}
    for c in cameras:
        streams[f"raw-{c['id']}"] = [c["rtsp"]] if c.get("rtsp") else None
        streams[c["id"]] = [f"rtsp://127.0.0.1:8554/raw-{c['id']}",
                            f"ffmpeg:rtsp://127.0.0.1:8554/raw-{c['id']}#video=h264"]
    return {"log": {"level": "info"}, "api": {"origin": "*"}, "rtsp": {"pkt_size": 1472},
            "ffmpeg": {"bin": "ffmpeg", "h264": "-codec:v libx264 -g:v 30 -preset:v superfast -tune:v zerolatency "
                                               "-profile:v main -level:v 4.1"},
            "streams": streams}


def demo_clips(cameras: list, clips_dir) -> list:
    """[(camera id, clip path)] for the demo cameras (`clip:` instead of `rtsp:`), which `just safety-demo` loops
    into go2rtc's raw-<id> as if they were cameras."""
    return [(c["id"], str(Path(clips_dir) / c["clip"])) for c in cameras if c.get("clip")]


def _replace(path: Path, text: str) -> None:
    """Write whole or not at all: the app reads these files at any moment."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_generated(out_dir: Path, files: dict) -> set:
    """Write {name: content} as <out_dir>/<name>.yaml, each replaced whole; return the names whose content changed."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    changed = set()
    for name, content in files.items():
        path = out_dir / f"{name}.yaml"
        text = "# generated by tools/safety_controller.py -- edit products/safety/, not this file\n" + yaml.safe_dump(
            content, sort_keys=False)
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            _replace(path, text)
            changed.add(name)
    return changed


def render(product: dict, cameras: list, state: dict, data_dir, repo: Path = REPO, write: bool = True) -> tuple:
    """(role sources, {file name: content}) after writing every generated file (unless `write` is False: a report).
    Writing applies nothing: what is still to apply is what differs from the last applied state (pending). The app
    reads the written role files as its running config, so only what is about to be applied should write them."""
    eff = effective_cameras(cameras, state, product)
    sources = role_sources(eff, product)
    files = {role: render_role(role, srcs, product) for role, srcs in sources.items()}
    files["fall_detection"] = render_fall_service(sources.get("fall", []), product, repo, Path(data_dir))
    files["go2rtc"] = render_go2rtc(eff)
    if write:
        write_generated(Path(data_dir) / "safety" / "generated", files)
    return sources, files


def digest(content) -> str:
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def load_applied(data_dir) -> dict:
    """{file name: digest} of what the last successful steps applied; {} before the first apply."""
    return load_state(Path(data_dir) / "safety" / APPLIED_FILE)


def save_applied(data_dir, applied: dict) -> None:
    path = Path(data_dir) / "safety" / APPLIED_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    _replace(path, json.dumps(applied, indent=2, sort_keys=True))


def pending(files: dict, applied: dict) -> set:
    """The generated files whose content was not applied yet."""
    return {name for name, content in files.items() if applied.get(name) != digest(content)}


# ── applying ──────────────────────────────────────────────────────────────────────────────────────────────────

def plan(sources: dict, changed: set, running: set) -> dict:
    """What to do with each role sidecar: stop the running ones with no camera left, start the missing ones that
    have cameras, restart the running ones whose config changed. Everything else is left alone."""
    out = {"restart": [], "start": [], "stop": []}
    for role, srcs in sources.items():
        if not srcs:
            if role in running:
                out["stop"].append(role)
        elif role not in running:
            out["start"].append(role)
        elif role in changed:
            out["restart"].append(role)
    return out


def _run(args) -> str:
    return subprocess.run(list(args), capture_output=True, text=True, check=True).stdout


def running_roles(compose: list, runner=_run) -> set:
    out = runner([*compose, "ps", "--services", "--status", "running"])
    return {name[len(SERVICE_PREFIX):] for name in out.split() if name.startswith(SERVICE_PREFIX)}


def apply_plan(the_plan: dict, compose: list, runner=_run, on_done=lambda role: None) -> None:
    """Stops first (frees the GPU), then starts, then restarts; on_done(role) after each one that worked."""
    for action, verb, why in (("stop", ["stop"], "no camera has it on"), ("start", ["up", "-d", "--no-build"], ""),
                              ("restart", ["restart"], "its cameras changed")):
        for role in the_plan[action]:
            logger.info("%s %s%s%s", action, SERVICE_PREFIX, role, f" ({why})" if why else "")
            runner([*compose, *verb, f"{SERVICE_PREFIX}{role}"])
            on_done(role)


def go2rtc_missing(generated: dict, live: set) -> dict:
    """{stream name: its sources (None = publish-only)} for the generated streams go2rtc does not have yet."""
    return {name: srcs for name, srcs in (generated.get("streams") or {}).items() if name not in live}


def _http(method: str, url: str) -> str:
    with urllib.request.urlopen(urllib.request.Request(url, method=method), timeout=5) as resp:
        return resp.read().decode("utf-8", "replace")


def register_streams(missing: dict, api: str, http=_http) -> tuple:
    """Add streams to the running go2rtc through its API (PUT /api/streams?name=..&src=..), so a new camera needs no
    go2rtc restart (which would cut every camera's pull). A source whose ${VAR} credentials are not in this process's
    environment is not sent (go2rtc's own environment has them): that stream is returned as skipped, and needs a
    go2rtc restart to be picked up. Returns (sent, skipped) stream names."""
    sent, skipped = [], []
    for name, srcs in missing.items():
        expanded = [os.path.expandvars(src) for src in srcs or []]
        if any("${" in src for src in expanded):        # cameras.yaml names credentials as ${VAR}
            logger.warning("go2rtc: %s names credentials this process cannot read; restart go2rtc to add it", name)
            skipped.append(name)
            continue
        query = urllib.parse.urlencode([("name", name), *(("src", src) for src in expanded)])
        http("PUT", f"{api}/api/streams?{query}")
        logger.info("go2rtc: added stream %s", name)
        sent.append(name)
    return sent, skipped


def sync_go2rtc(data_dir: Path, api: str, http=_http) -> tuple:
    """Register the generated streams the running go2rtc lacks (idempotent: a stream it has is left alone)."""
    generated = yaml.safe_load((Path(data_dir) / "safety" / "generated" / "go2rtc.yaml").read_text(encoding="utf-8"))
    live = set(json.loads(http("GET", f"{api}/api/streams") or "{}"))
    return register_streams(go2rtc_missing(generated, live), api, http)


class IncompleteApply(RuntimeError):
    """The apply could not finish (its version stays unapplied); the message says what the operator must do."""


def _key(camera: str, role: str) -> str:
    return f"{camera}{ROLE_KEY_SEPARATOR}{role}"


def _boot_ids(redis_client, sources: dict, roles: list) -> dict:
    """{stream key: boot_id} of these roles' streams as their status says now (to tell a restart has happened)."""
    out = {}
    for role in roles:
        for s in sources.get(role, []):
            raw = redis_client.get(f"iris:status:{_key(s['id'], role)}")
            try:
                out[_key(s["id"], role)] = json.loads(raw).get("boot_id") if raw else None
            except (TypeError, ValueError):
                out[_key(s["id"], role)] = None
    return out


def mark_applying(roles: list, redis_client) -> None:
    """Tell camera health these roles are being (re)started, so it holds their incidents (best effort)."""
    for role in roles:
        try:
            redis_client.set(f"{APPLYING_PREFIX}{role}", "1", ex=APPLY_HOLD_S)
        except Exception as exc:  # noqa: BLE001 -- health may page; the apply goes on
            logger.warning("could not mark %s as applying: %s", role, exc)


def forget_status(role: str, redis_client) -> None:
    """A stopped role's streams are gone: drop their status keys (the sidecar deletes them itself on a clean stop;
    this covers one that died), so the app stops listing them as running."""
    keys = list(redis_client.scan_iter(match=f"iris:status:*{ROLE_KEY_SEPARATOR}{role}"))
    if keys:
        redis_client.delete(*keys)


def wait_until_live(redis_client, sources: dict, roles: list, before: dict, timeout: float = APPLY_HOLD_S,
                    interval: float = 2.0, now=time.monotonic, sleep=time.sleep) -> set:
    """Wait until every stream of these (re)started roles reports running with a boot other than `before`'s, then
    release each role's camera-health hold as it comes up. Returns the roles still not live at the timeout (their
    hold then simply expires)."""
    waiting = {role: {_key(s["id"], role) for s in sources.get(role, [])} for role in roles}
    deadline = now() + timeout
    while True:
        for role, keys in list(waiting.items()):
            for key in list(keys):
                raw = redis_client.get(f"iris:status:{key}")
                try:
                    status = json.loads(raw) if raw else {}
                except ValueError:
                    status = {}
                if status.get("running") and status.get("boot_id") != before.get(key):
                    keys.discard(key)
            if not keys:
                redis_client.delete(f"{APPLYING_PREFIX}{role}")
                logger.info("%s%s is live", SERVICE_PREFIX, role)
                waiting.pop(role)
        if not waiting or now() >= deadline:
            return set(waiting)
        sleep(interval)


def apply(args, runner=_run, http=_http, redis_client=None, wait: bool = False) -> dict:
    """Render, then bring the stack to it. What is done is what differs from the last APPLIED state (pending), and
    each step is recorded only once it worked: a failed or partial apply is simply retried by the next one, and
    rendering alone (`just safety-status`) applies nothing. With Redis: camera health is held for the roles it
    (re)starts, a stopped role's status keys are dropped, and with `wait` it returns once those roles are live."""
    product = load_product(args.product)
    sources, files = render(product, load_cameras(args.cameras), load_state(args.state), args.data)
    applied = load_applied(args.data)
    changed = pending(files, applied)

    def done(name: str) -> None:
        applied[name] = digest(files[name])
        save_applied(args.data, applied)

    if args.go2rtc_api:
        _sent, skipped = sync_go2rtc(Path(args.data), args.go2rtc_api, http)   # before any sidecar pulls a new camera
        if skipped:
            # A role sidecar pulling a stream go2rtc lacks stalls, and ends the role on its other cameras too: nothing
            # is started until go2rtc has it (it reads the generated file, with the credentials, when it restarts).
            raise IncompleteApply(f"go2rtc does not have {', '.join(sorted(skipped))}: its credentials are not in the "
                                  "controller's environment -- restart go2rtc (source scripts/safety_env.sh && "
                                  "safety_compose restart go2rtc), then press Apply again")
    if "go2rtc" in changed:
        done("go2rtc")          # registered above, or read by go2rtc at its next start
    running = running_roles(args.compose, runner)
    the_plan = plan(sources, changed, running)
    if redis_client is None and args.redis:
        import redis
        redis_client = redis.Redis.from_url(args.redis, decode_responses=True)
    up = the_plan["start"] + the_plan["restart"]
    before = _boot_ids(redis_client, sources, up) if redis_client is not None else {}
    if redis_client is not None:
        mark_applying(up, redis_client)
    apply_plan(the_plan, args.compose, runner, on_done=done)
    if redis_client is not None:
        for role in the_plan["stop"]:
            forget_status(role, redis_client)
    for role in changed & set(sources):
        if not sources[role] and role not in running:
            done(role)          # no camera and not running: nothing to do
    if "fall_detection" in changed:
        if args.fall_restart:
            logger.info("the fall service's cameras changed: %s", " ".join(args.fall_restart))
            runner(args.fall_restart)
        done("fall_detection")
    if wait and redis_client is not None and up:
        late = wait_until_live(redis_client, sources, up, before)
        if late:
            logger.warning("not live after %ds: %s", APPLY_HOLD_S, ", ".join(sorted(late)))
    return the_plan


RETRY_FIRST_S = 10.0    # a failed apply is tried again after 10 s, then 20, 40, 80 ...
RETRY_TRIES = 5         # ... and given up after 5 tries, until Apply is pressed again


class ApplyWatch:
    """One poll of `watch`: a newly requested version is applied once and marked applied. One that fails is tried
    again after a growing wait and given up after RETRY_TRIES, so a step that keeps failing (a fall service that will
    not start, a camera go2rtc lacks) is not re-run -- and the service restarted -- on every poll. Pressing Apply
    again (a new version) starts afresh."""

    def __init__(self, redis_client, run, now=time.monotonic):
        self.redis, self.run, self.now = redis_client, run, now
        self.applied = None          # the last version applied
        self.version = None          # the version being tried
        self.failures = 0
        self.retry_at = 0.0

    def poll(self) -> None:
        version = int(self.redis.get(APPLY_VERSION_KEY) or 0)
        if version == self.applied:
            return
        if version != self.version:
            self.version, self.failures, self.retry_at = version, 0, 0.0
        if self.failures >= RETRY_TRIES or self.now() < self.retry_at:
            return
        try:
            the_plan = self.run()
        except Exception as exc:  # noqa: BLE001 -- keep watching; the UI sees applied < version
            self.failures += 1
            if self.failures >= RETRY_TRIES:
                logger.error("apply of version %d failed %d times (%s); given up until Apply is pressed again",
                             version, self.failures, exc, exc_info=True)
            else:
                wait = RETRY_FIRST_S * 2 ** (self.failures - 1)
                self.retry_at = self.now() + wait
                logger.error("apply of version %d failed (%s); trying again in %.0f s", version, exc, wait,
                             exc_info=True)
            return
        logger.info("apply requested (version %d): %s", version, the_plan)
        self.redis.set(APPLY_APPLIED_KEY, version)
        self.applied = version


def watch(args) -> None:
    """Apply whenever the app asks; the version is marked applied once every (re)started role is live (or its
    wait timed out), so the UI's "applying" lasts until the change is really running."""
    if not args.redis:
        raise SystemExit("watch needs --redis (the stack's Redis, where the app asks for an apply)")
    import redis
    r = redis.Redis.from_url(args.redis, decode_responses=True)
    poll = ApplyWatch(r, lambda: apply(args, redis_client=r, wait=True)).poll
    while True:
        try:
            poll()
        except Exception:  # noqa: BLE001 -- Redis unreachable for a moment: keep watching
            logger.exception("cannot read the apply request; trying again on the next poll")
        time.sleep(args.interval)


def model_pointers(product: dict) -> list:
    """The DVC pointers (.dvc files, relative to the repo root) of the models the stack loads: safety.yaml's
    `models`, a folder standing for each pointer in it."""
    out = []
    for entry in product.get("models") or []:
        path = REPO / entry
        pointers = sorted(path.glob("*.dvc")) if path.is_dir() else [path.with_name(path.name + ".dvc")]
        out += [str(p.relative_to(REPO)) for p in pointers]
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("command", choices=("render", "apply", "clips", "watch", "models"))
    p.add_argument("--product", default=str(REPO / "products/safety/safety.yaml"))
    p.add_argument("--cameras", default=str(REPO / "products/safety/cameras.yaml"))
    p.add_argument("--data", required=True, help="the stack's data dir (mounted at data/ in its containers)")
    p.add_argument("--state", help="the UI's saved toggles (default <data>/safety/cameras_state.json)")
    p.add_argument("--compose", default="docker compose", help="the stack's compose command (project, files)")
    p.add_argument("--fall-restart", default="", help="command that restarts the fall service, when its cameras change")
    p.add_argument("--redis", default="", help="the stack's Redis (watch: required; apply: holds camera health "
                                                "for the roles it restarts, forgets a stopped role's status)")
    p.add_argument("--interval", type=float, default=2.0, help="watch: seconds between polls")
    p.add_argument("--clips", default=".", help="clips: the directory the demo cameras' clips are in")
    p.add_argument("--dry-run", action="store_true", help="render: report what each role runs and what is still to "
                                                          "apply, writing nothing (the app reads the written files)")
    p.add_argument("--go2rtc-api", default="", help="apply: the stack's go2rtc API (http://localhost:<port>), to add "
                                                    "new cameras' streams without restarting it")
    args = p.parse_args(argv)
    args.state = args.state or str(Path(args.data) / "safety" / "cameras_state.json")
    args.compose = shlex.split(args.compose)
    args.fall_restart = shlex.split(args.fall_restart)
    return args


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = parse_args(argv)
    if args.command == "render":
        sources, files = render(load_product(args.product), load_cameras(args.cameras), load_state(args.state),
                                args.data, write=not args.dry_run)
        waiting = pending(files, load_applied(args.data))
        for role, srcs in sources.items():
            print(f"{role:10s} {', '.join(s['id'] for s in srcs) or '-'}{'  (not applied yet)' if role in waiting else ''}")
        return 0
    if args.command == "apply":
        print(apply(args, wait=True))          # releases camera health's hold for each role once it is live
        return 0
    if args.command == "models":
        for pointer in model_pointers(load_product(args.product)):
            print(pointer)
        return 0
    if args.command == "clips":
        product = load_product(args.product)
        for cid, path in demo_clips(effective_cameras(load_cameras(args.cameras), load_state(args.state), product),
                                    args.clips):
            print(f"{cid}\t{path}")
        return 0
    watch(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
