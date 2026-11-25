"""Barber service API + live websocket. Run: uvicorn server:app --port 8003

The websocket is forked from products/attendance/server.py @ 1f5b27b --
changes: staff instead of employees, an app factory (create_app) so tests
can run the API without the Redis consumer, and no photo enrollment (staff
are recognized by uniform, not face ID).
"""
from __future__ import annotations

import asyncio
import functools
import json
import logging
import re
import shutil
import time
from contextlib import asynccontextmanager, contextmanager
from datetime import date, datetime, timedelta
from typing import Any, Callable
from urllib.parse import quote

from fastapi import Body, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

import crops
from acl import camera_clock
from config import REPO_DIR, ServiceConfig, load_config, load_stream_starts
from model_files import load_verifier, verify_service_models
from consumer import ConsumerThread
from netguard import is_local_client
from footage_clock import footage_clock
from report_data import KINDS, build_report, file_name, period_bounds
from report_docx import render_docx
from report_html import render_html
from photo_report import build_photo_report, photo_file_name, render_photo_html
from report_marks import mark
from report_pdf import RendererUnavailable, html_to_pdf, svg_to_png
from report_text import LANGS
from policy import effective, validate_policy, validate_policy_update
from saved_reports import Hooks, Pages, Period, ReportSaver, list_saved, saved_file
from shopday import day_start, shop_day, shop_day_range
from stats import compute_stats
from storage import (alert_snapshot, any_visit, alert_subject, assign_chair, confirm_roster, connect, count_customers, create_staff, crop_row,
                     current_roster, delete_customer, delete_staff, get_customer, get_staff, list_alerts,
                     list_bindings, list_customers, list_handoffs, list_staff, resume_check_stats,
                     roster_confirmation, roster_detail, update_staff)
from zones import ZoneValidationError, area_of, validate_zones, zone_set_problems

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("barber.server")


class ConnectionManager:
    """Broadcast to every connected websocket client; sends are handed to
    the event loop from the consumer thread. Each client has its own queue and
    sender task (2026-10-15): one awaited send per client in turn let a client
    that stopped reading hold up every broadcast, and the dashboards behind it
    froze on their first snapshot. A client QUEUE messages behind is dropped
    (closed: the dashboard reconnects and starts from a fresh snapshot)."""

    QUEUE = 16
    SEND_TIMEOUT_S = 10.0

    def __init__(self):
        self._clients: dict[WebSocket, asyncio.Queue] = {}
        self._senders: dict[WebSocket, asyncio.Task] = {}
        self._ending: set[asyncio.Task] = set()      # cancelled senders, held until done (tasks are weakly held)
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def connect(self, ws: WebSocket, first: dict | None = None) -> None:
        await ws.accept()
        q: asyncio.Queue = asyncio.Queue(self.QUEUE)
        if first is not None:
            q.put_nowait(first)
        self._clients[ws] = q
        self._senders[ws] = asyncio.create_task(self._send_loop(ws, q))

    async def _send_loop(self, ws: WebSocket, q: asyncio.Queue) -> None:
        try:
            while True:
                await asyncio.wait_for(ws.send_json(await q.get()), self.SEND_TIMEOUT_S)
        except asyncio.CancelledError:
            raise
        except Exception:
            self._drop(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._clients.pop(ws, None)
        task = self._senders.pop(ws, None)
        if task is not None and task is not asyncio.current_task():
            self._ending.add(task)
            task.add_done_callback(self._ending.discard)
            task.cancel()

    def _drop(self, ws: WebSocket) -> None:
        self.disconnect(ws)

        async def close() -> None:
            try:
                await asyncio.wait_for(ws.close(), 2.0)
            except Exception:
                pass
        task = asyncio.get_running_loop().create_task(close())
        self._ending.add(task)
        task.add_done_callback(self._ending.discard)

    def broadcast_threadsafe(self, message: dict) -> None:
        if self._loop is None or not self._clients:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast(message), self._loop)

    async def _broadcast(self, message: dict) -> None:
        for ws, q in list(self._clients.items()):
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                self._drop(ws)


def _parse_range(from_: str | None, to: str | None, hour: int) -> tuple[datetime, datetime]:
    """[from, to) from ISO dates or datetimes; a date-only value starts at the
    shop day's start hour; defaults to the current shop day."""
    def parse(value: str) -> datetime:
        return day_start(date.fromisoformat(value), hour) if len(value) == 10 else datetime.fromisoformat(value)
    try:
        start = parse(from_) if from_ else shop_day_range(datetime.now(), hour)[0]
        end = parse(to) if to else start + timedelta(days=1)
    except ValueError:
        raise HTTPException(status_code=400, detail="from/to must be ISO dates or datetimes") from None
    if end <= start:
        raise HTTPException(status_code=400, detail="to must be after from")
    return start, end


def verify_models(cfg: ServiceConfig, env: dict | None = None) -> None:
    """The startup model check: the kids car model must be the registered version the site's manifest lists
    (perception/engine/model_manifest.py). No manifest only warns, unless MODEL_MANIFEST_REQUIRED is set."""
    verifier = load_verifier(REPO_DIR)
    try:
        verify_service_models({"model_manifest": cfg.model_manifest, "kids_car_model": cfg.kids_car_path},
                              REPO_DIR, env)
    except verifier.ManifestError as exc:
        raise RuntimeError(f"barber service: refusing to start -- {exc}") from None


def create_app(cfg: ServiceConfig | None = None, start_consumer: bool = True,
               check_models: bool | None = None) -> FastAPI:
    """The app. `check_models`: run verify_models at startup (the lifespan, before the consumer starts or uvicorn
    binds the port: a refusal is "Application startup failed", exit 3); None = check when `cfg` is None, i.e. the real
    service (`uvicorn server:app`), which is the only caller without a config. Importing this module checks nothing."""
    if check_models is None:
        check_models = cfg is None
    if cfg is None:
        cfg = load_config()
    manager = ConnectionManager()

    @contextmanager
    def db():
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        conn = connect(str(cfg.db_path))
        try:
            yield conn
        finally:
            conn.close()

    def live_payload() -> dict:
        """Everything the Live floor shows, in one message."""
        engine_live = consumer.live()
        clip_starts = load_stream_starts(cfg.stream_starts_file)
        policy = consumer.effective_policy()
        zones = consumer.shown_zones()                 # the Kids chair where the car is (sub-project 4)
        stored = consumer.zones.load()
        now = datetime.now()
        today_start, today_end = shop_day_range(now, int(policy["day_start_hour"]))
        with db() as conn:
            open_alerts = list_alerts(conn, open_only=True)
            recent = [a for a in list_alerts(conn, start=today_start, limit=50) if not a["open"]][:5]
            today = compute_stats(conn, today_start, today_end, live=engine_live["customers"], now=now,
                                  min_service_s=policy["min_service_s"], min_wait_s=policy["avg_wait_min_s"],
                                  visits_source=int(policy["visits_source"]), serve_s=policy["wt_serve_s"])
            roster = current_roster(conn)
            roster_confirmed = roster_confirmation(conn, today_start.date())["confirmed"]
        foot = footage_clock(cfg.footage_start, consumer.clip_run)
        played = min(foot.footage(now), foot.end) if foot else None
        return {"type": "live_snapshot", "server_time": now.isoformat(timespec="seconds"),
                # a run on recordings (2026-10-18): the footage's day and how far it has played -- its reports' clock
                "footage": None if foot is None else {
                    "day": shop_day(played, int(policy["day_start_hour"])).isoformat(),
                    "start": foot.start.isoformat(timespec="seconds"), "end": foot.end.isoformat(timespec="seconds"),
                    "now": played.isoformat(timespec="seconds")},
                # the shop's name and logo on every page (2026-10-18): logo = GET /api/barber/logo has one
                "site": {"place": cfg.place, "branch": cfg.branch, "logo": cfg.logo is not None},
                "shop_day": today_start.date().isoformat(),
                "day_start_hour": int(policy["day_start_hour"]),
                "customers_waiting": engine_live["customers_waiting"],
                "customers": engine_live["customers"], "chairs": engine_live["chairs"],
                "cameras": engine_live["cameras"],     # {cam: {last_frame_age_s}}, engine clock
                "scene": consumer.scene,
                "clips": {c.id: {"started_ms": round(clip_starts[c.id].timestamp() * 1000),
                                 "duration_s": c.clip_duration_s}
                          for c in cfg.cameras if c.clip_duration_s and c.id in clip_starts},
                "zones": zones,
                "zone_display": {t: bool(policy[f"zone_show_{t}"])
                                 for t in ("entrance_line", "chair", "seat", "waiting", "washing", "row", "door_area", "face_area")},
                "zone_problems": _zone_problems(stored),
                "roster": roster,
                "roster_confirmed": roster_confirmed,
                "open_alerts": open_alerts, "recent_alerts": recent,
                "targets": {"wait_target_s": policy["wait_target_s"], "service_target_s": policy["service_target_s"]},
                "today": {k: today[k] for k in ("customer_visits", "completed_haircuts", "average_wait_s",
                                                "average_haircut_s", "walk_outs", "haircuts_per_barber",
                                                "lost", "returning_visits", "visitors", "served_customers")}}

    consumer = ConsumerThread(
        cfg,
        on_change=lambda: manager.broadcast_threadsafe(live_payload()),
        # boxes + chair states, at policy scene_fps -- so they follow the video between snapshots
        on_scene=lambda payload: manager.broadcast_threadsafe({"type": "scene", **payload}),
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if check_models:                                # first: nothing starts unless the models are the registered ones
            verify_models(cfg)
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        for left in pdf_tmp.glob("barber-report-*"):    # 2026-10-25: a stop mid-print leaves Chrome's folder behind
            shutil.rmtree(left, ignore_errors=True)
        manager.bind_loop(asyncio.get_running_loop())
        if start_consumer:
            consumer.start()
            if cfg.saved_reports is not None:           # live: the period checks; a recording run: its laps' PDFs
                _app.state.saver.start()
        yield
        _app.state.saver.stop()

    app = FastAPI(title="Barber service", lifespan=lifespan)
    # The Vite dev proxy connects from this box and forwards the browser's
    # address; trust that header from loopback only, whatever runs the app,
    # so the LAN-only photo guard judges the browser, not the proxy.
    app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="127.0.0.1,::1")
    app.state.cfg = cfg
    app.state.consumer = consumer

    def hour() -> int:
        return int(consumer.effective_policy()["day_start_hour"])

    # ── clock ───────────────────────────────────────────────────────────────

    @app.get("/api/time")
    def server_time():
        """This server's clock plus each camera's offset, so a client can
        show times on the same clock the rows were written with."""
        return {"epoch_ms": datetime.now().timestamp() * 1000,
                "camera_offsets_ms": {c.id: (None if (off := camera_clock.offset(c.id)) is None
                                             else round(off * 1000)) for c in cfg.cameras}}

    # ── staff ───────────────────────────────────────────────────────────────

    @app.get("/api/barber/cameras")
    def cameras():
        """Configured cameras. `stream` is the camera's go2rtc stream name
        (the browser-safe variant go2rtc.yaml derives from raw-<id>)."""
        return [{"id": c.id, "stream": c.id, "name": c.name} for c in cfg.cameras]

    @app.get("/api/barber/logo")
    def shop_logo():
        """The shop's logo (site yaml `logo`, as on the reports) for the dashboard's rail (2026-10-18)."""
        if cfg.logo is None:
            raise HTTPException(status_code=404, detail="no logo set for this site")
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".svg": "image/svg+xml"}
        return FileResponse(cfg.logo, media_type=mime.get(cfg.logo.suffix.lower(), "application/octet-stream"),
                            headers={"Cache-Control": "max-age=3600"})

    @app.get("/api/staff")
    def staff_list():
        with db() as conn:
            return list_staff(conn)

    @app.post("/api/staff", status_code=201)
    def staff_create(payload: dict):
        name = str(payload.get("name") or "").strip()
        if not name:
            raise HTTPException(status_code=400, detail="name is required")
        with db() as conn:
            return create_staff(conn, name)

    @app.patch("/api/staff/{staff_id}")
    def staff_update(staff_id: str, payload: dict):
        with db() as conn:
            row = update_staff(conn, staff_id, **payload)
        if row is None:
            raise HTTPException(status_code=404, detail="unknown staff member")
        return row

    @app.delete("/api/staff/{staff_id}")
    def staff_delete(staff_id: str):
        with db() as conn:
            if not delete_staff(conn, staff_id):
                raise HTTPException(status_code=404, detail="unknown staff member")
        return {"id": staff_id, "deleted": True}

    # ── roster: who works which chair ───────────────────────────────────────

    @app.get("/api/barber/roster")
    def roster_get(detail: bool = False):
        with db() as conn:
            return roster_detail(conn) if detail else current_roster(conn)

    @app.get("/api/barber/roster/confirmation")
    def roster_confirmation_get():
        with db() as conn:
            return roster_confirmation(conn, shop_day(datetime.now(), hour()))

    @app.post("/api/barber/roster/confirmation")
    def roster_confirmation_post():
        with db() as conn:
            day = shop_day(datetime.now(), hour())
            confirm_roster(conn, day, datetime.now())
            return roster_confirmation(conn, day)

    @app.put("/api/barber/roster/{chair_id}")
    def roster_put(chair_id: str, payload: dict):
        staff_id = payload.get("staff_id")
        with db() as conn:
            if staff_id is not None and get_staff(conn, staff_id) is None:
                raise HTTPException(status_code=404, detail="unknown staff member")
            assign_chair(conn, chair_id, staff_id, datetime.now())
            return current_roster(conn)

    # ── zones (same contract as attendance's editor) ────────────────────────

    def _zone_problems(stored: dict) -> list[dict]:
        ids = {c.id for c in cfg.cameras}
        return zone_set_problems({c.id: stored.get(c.id, []) for c in cfg.cameras},
                                 {cam: zs for cam, zs in stored.items() if cam not in ids})

    @app.get("/api/monitoring/zones")
    def zones_get():
        return consumer.zones.load()

    @app.put("/api/monitoring/zones/{cam_id}")
    def zones_put(cam_id: str, payload: Any = Body(...)):
        if cfg.camera(cam_id) is None:
            raise HTTPException(status_code=404, detail="unknown camera")
        try:
            cleaned = validate_zones(payload)
        except ZoneValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        zones = dict(consumer.zones.load())
        zones[cam_id] = cleaned
        consumer.zones.save(zones)
        return cleaned

    @app.get("/api/monitoring/zones/problems")
    def zones_problems():
        return _zone_problems(consumer.zones.load())

    # ── policy ──────────────────────────────────────────────────────────────

    @app.get("/api/barber/policy")
    def policy_get():
        return consumer.effective_policy()

    @app.put("/api/barber/policy")
    def policy_put(payload: dict):
        """Partial update: only the keys sent change, and only they are stored (with those set before): a key
        never set keeps following the site's policy block and the defaults (final review I3)."""
        try:
            changes = validate_policy_update(payload)
            stored = {**(consumer.policy.load() or {}), **changes}
            merged = effective(stored, base=consumer.base_policy)
            validate_policy(merged)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        consumer.policy.save(stored)
        return merged

    # ── live state, history, stats ──────────────────────────────────────────

    @app.get("/api/barber/live")
    def live():
        return live_payload()

    @app.get("/api/barber/stats")
    def stats(from_: str | None = Query(None, alias="from"), to: str | None = None,
              bucket: str | None = None, barber: str | None = None):
        h = hour()
        start, end = _parse_range(from_, to, h)
        engine_live = consumer.live()
        pol = consumer.effective_policy()
        with db() as conn:
            try:
                result = compute_stats(conn, start, end, bucket, barber_id=barber, day_start_hour=h,
                                       live=engine_live["customers"], now=datetime.now(),
                                       min_service_s=pol["min_service_s"], min_wait_s=pol["avg_wait_min_s"],
                                       visits_source=int(pol["visits_source"]), serve_s=pol["wt_serve_s"])
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from None
        result["customers_waiting"] = engine_live["customers_waiting"]
        return result

    @app.get("/api/barber/handoffs")
    def handoffs(from_: str | None = Query(None, alias="from"), to: str | None = None):
        start, end = _parse_range(from_, to, hour())
        with db() as conn:
            return list_handoffs(conn, start, end)

    @app.get("/api/barber/bindings")
    def bindings(from_: str | None = Query(None, alias="from"), to: str | None = None):
        start, end = _parse_range(from_, to, hour())
        with db() as conn:
            return list_bindings(conn, start, end)

    @app.get("/api/barber/alerts")
    def alerts(from_: str | None = Query(None, alias="from"), to: str | None = None, open: bool = False,
               visit: str | None = None, limit: int = 100):
        start, end = _parse_range(from_, to, hour()) if (from_ or to) else (None, None)
        with db() as conn:
            return list_alerts(conn, start, end, open_only=open, limit=min(limit, 1000), visit_id=visit)

    # ── customer register (project C) ───────────────────────────────────────

    @app.get("/api/barber/customers")
    def customers_list(limit: int = 25, offset: int = 0, sort: str = "last_seen"):
        with db() as conn:
            return {"rows": list_customers(conn, min(max(limit, 1), 500), max(offset, 0), sort),
                    "total": count_customers(conn)}

    @app.get("/api/barber/customers/{customer_id}")
    def customer_detail(customer_id: str):
        with db() as conn:
            c = get_customer(conn, customer_id)
        if c is None:
            raise HTTPException(status_code=404, detail="unknown customer")
        return c

    @app.delete("/api/barber/customers/{customer_id}")
    def customer_delete(customer_id: str):
        with db() as conn:
            paths = delete_customer(conn, customer_id)
        if paths is None:
            raise HTTPException(status_code=404, detail="unknown customer")
        for rel in paths:
            crops.delete(cfg.crops_dir, rel)
        return {"id": customer_id, "deleted": True}

    @app.get("/api/barber/crops/{kind}/{row_id}")
    def crop_image(kind: str, row_id: int, request: Request):
        """Serves only files a register row references, under the crops root,
        and only to clients on the shop's own network."""
        if not is_local_client(request.client.host if request.client else None):
            raise HTTPException(status_code=403, detail="photos are only served on the shop network")
        with db() as conn:
            row = crop_row(conn, kind, row_id)
        path = crops.inside(cfg.crops_dir, row["crop_path"]) if row else None
        if path is None or not path.is_file():
            raise HTTPException(status_code=404, detail="no such crop")
        return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/barber/alerts/{alert_id}/snapshot")
    def alert_snapshot_image(alert_id: str, request: Request):
        """The frame taken when the alert opened; shop network only (as the photos)."""
        if not is_local_client(request.client.host if request.client else None):
            raise HTTPException(status_code=403, detail="photos are only served on the shop network")
        with db() as conn:
            rel = alert_snapshot(conn, alert_id)
        path = crops.inside(cfg.crops_dir, rel) if rel else None
        if path is None or not path.is_file():
            raise HTTPException(status_code=404, detail="no snapshot")
        return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    # ── the kids car (sub-project 4; spec 2026-10-15-barber-kids-car-design.md) ─────────
    sizes_file = (cfg.kids_car_model.parent if cfg.kids_car_model else REPO_DIR / "perception/models/kids_car") / "sizes.json"
    try:
        car_sizes = {k: tuple(v) for k, v in json.loads(sizes_file.read_text()).items()}
    except (OSError, ValueError):
        car_sizes = {}

    def car_payload() -> dict:
        p = consumer.kids.place
        return {"place": None if p is None else {"camera": p.camera, "box": list(p.box), "source": p.source,
                                                 "conf": p.conf, "since": p.since},
                "enabled": bool(int(consumer.effective_policy()["kc_enabled"])), "locator": consumer.locator is not None,
                "sizes": {c.id: list(car_sizes.get(c.id, (0.12, 0.18))) for c in cfg.cameras}}

    @app.get("/api/barber/kids-car")
    def kids_car_get():
        return car_payload()

    @app.put("/api/barber/kids-car")
    def kids_car_pin(payload: Any = Body(...)):
        """The dashboard's "the kids car is here": a camera and a point on its picture (0..1)."""
        if not isinstance(payload, dict) or cfg.camera(str(payload.get("camera"))) is None:
            raise HTTPException(status_code=404 if isinstance(payload, dict) and "camera" in payload else 400,
                                detail="unknown camera")
        try:
            x, y = float(payload["x"]), float(payload["y"])
        except (KeyError, TypeError, ValueError):
            raise HTTPException(status_code=400, detail="x and y (0..1) are required") from None
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise HTTPException(status_code=400, detail="x and y must be within 0..1")
        cam = str(payload["camera"])
        with consumer.lock:
            consumer.kids.pin(cam, x, y, car_sizes.get(cam, (0.12, 0.18)), time.time())
        return car_payload()

    @app.delete("/api/barber/kids-car")
    def kids_car_clear():
        with consumer.lock:
            consumer.kids.clear()
        return car_payload()

    # ── reports (2026-10-15; spec 2026-10-15-barber-reports-design.md) ─────────
    REPORT_TYPES = {"html": "text/html; charset=utf-8", "pdf": "application/pdf",
                    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
    pdf_tmp = cfg.data_dir / ".tmp"                     # 2026-10-25: Chrome's temp folders on /data, not /tmp
    pdf_svg_to_png = functools.partial(svg_to_png, tmp_dir=pdf_tmp)

    def chair_names() -> dict[str, str]:
        chairs = [z for zs in consumer.zones.load().values() for z in zs if z.get("type") == "chair"]
        # every copy's own id too: rows recorded before a chair moved camera carry its old zone id (sub-project 5)
        return {**{z["id"]: z.get("name") or z["id"] for z in chairs}, **{area_of(z): z.get("name") or z["id"] for z in chairs}}

    def report_logo() -> tuple[bytes, str] | None:
        """The shop's logo on the reports (site yaml `logo`); not the /api/barber/logo route's shop_logo."""
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".svg": "image/svg+xml"}
        if cfg.logo is None or cfg.logo.suffix.lower() not in mime:
            return None
        return cfg.logo.read_bytes(), mime[cfg.logo.suffix.lower()]

    def file_response(body: bytes, name: str, format: str, download: bool) -> Response:
        how = "inline" if format == "html" and not download else "attachment"
        plain = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or f"report.{format}"   # latin-1 header, no quotes
        return Response(body, media_type=REPORT_TYPES[format],
                        headers={"Content-Disposition": f"{how}; filename=\"{plain}\"; filename*=UTF-8''{quote(name)}",
                                 "Cache-Control": "no-store"})

    def report_name(kind: str, first: date, format: str) -> str:
        s, e = period_bounds(kind, first, int(consumer.effective_policy()["day_start_hour"]))
        return file_name(kind, s, e, cfg.place, cfg.branch_slug or "branch", format)

    def analysis_parts(kind: str, first: date, lan: bool) -> tuple[dict, Callable[[str], bytes | None]]:
        """A report's model and its snapshot images, built once for all of its formats (ValueError: not a period
        with data yet). lan: alert snapshots in it (customer photos stay on the LAN)."""
        pol = consumer.effective_policy()
        names = chair_names()
        site = {"place": cfg.place, "branch": cfg.branch, "branch_slug": cfg.branch_slug}
        with db() as conn:
            # a run on recordings (2026-10-18): `first` is a footage day and the report gives the footage's times
            model = build_report(conn, kind, first, datetime.now(), pol, site, names, live=consumer.live()["customers"],
                                 clock=footage_clock(cfg.footage_start, consumer.clip_run))
            snaps = {aid: alert_snapshot(conn, aid) for aid in model["alerts"]["shown"]} if lan else {}
            subjects = {aid: alert_subject(conn, aid) for aid in snaps}
        zones = consumer.zones.load()

        def marked(aid: str, jpg: bytes) -> bytes:
            """2026-10-18 (user): the Preview's mark -- the waiting customer's box, or the haircut's chair zone."""
            s = subjects.get(aid) or {}
            outline = None
            if s.get("kind") == "excessive_service" and s.get("chair_id"):
                outline = next((z["points"] for z in zones.get(s.get("subject_cam"), [])
                                if z.get("type") == "chair" and s["chair_id"] in (z["id"], area_of(z))), None)
            tag = (names.get(s["chair_id"], s["chair_id"]) if s.get("kind") == "excessive_service" and s.get("chair_id")
                   else f"#{s['ticket']}" if s.get("ticket") is not None else None)
            return mark(jpg, s.get("subject_box"), outline, tag)

        def image(aid: str) -> bytes | None:
            path = crops.inside(cfg.crops_dir, snaps[aid]) if snaps.get(aid) else None
            return marked(aid, path.read_bytes()) if path is not None and path.is_file() else None

        return model, image

    def render_report(kind: str, first: date, lang: str, format: str, lan: bool) -> bytes:
        """A report's bytes for the download endpoint (RendererUnavailable: no Chrome for a PDF)."""
        model, image = analysis_parts(kind, first, lan)
        logo = report_logo()
        if format == "docx":
            return render_docx(model, lang, image, lan, pdf_svg_to_png, shop_logo=logo)
        if format == "html":
            return render_html(model, lang, image, lan, shop_logo=logo).encode()
        return html_to_pdf(render_html(model, lang, image, lan, printing=True, shop_logo=logo), tmp_dir=pdf_tmp)

    @app.get("/api/barber/resume-checks")
    def resume_checks(first: str, last: str):
        """Per chair, the re-ID resume checks' scores over shop days first..last (2026-10-18, user: to tune
        ce_resume_sim_low / _high per chair): the same customer's and a new one's count, mean, min and max."""
        try:
            a, b = date.fromisoformat(first), date.fromisoformat(last)
        except ValueError:
            raise HTTPException(status_code=400, detail="first and last must be YYYY-MM-DD") from None
        hour = int(consumer.effective_policy()["day_start_hour"])
        chairs = [z for zs in consumer.zones.load().values() for z in zs if z.get("type") == "chair"]
        names = {**{z["id"]: z.get("name") or z["id"] for z in chairs}, **{area_of(z): z.get("name") or z["id"] for z in chairs}}
        with db() as conn:
            stats = resume_check_stats(conn, day_start(a, hour), day_start(b, hour) + timedelta(days=1))
        return {"first": a.isoformat(), "last": b.isoformat(),
                "chairs": [{**c, "name": names.get(c["chair_id"], c["chair_id"])} for c in stats]}

    @app.get("/api/barber/report")
    def report(request: Request, kind: str, start: str, lang: str = "en", format: str = "html", download: bool = False):
        """A day / week / month report of this branch: html (inline; download=1 as a file), pdf or docx (attachment)."""
        if kind not in KINDS or lang not in LANGS or format not in REPORT_TYPES:
            raise HTTPException(status_code=400, detail="kind, lang or format not recognised")
        try:
            first = date.fromisoformat(start)
        except ValueError:
            raise HTTPException(status_code=400, detail="start must be YYYY-MM-DD") from None
        lan = is_local_client(request.client.host if request.client else None)
        try:
            body = render_report(kind, first, lang, format, lan)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        except RendererUnavailable as exc:
            logger.warning("report renderer unavailable: %s", exc)
            raise HTTPException(status_code=503, detail="report renderer unavailable") from None
        return file_response(body, report_name(kind, first, format), format, download)

    # ── haircut photo report (2026-10-22; spec 2026-10-22-barber-haircut-photos-design.md) ─────────
    def photo_parts(first: date) -> tuple[dict, Callable[[str], bytes | None]]:
        """ValueError: a day not started yet."""
        site = {"place": cfg.place, "branch": cfg.branch, "branch_slug": cfg.branch_slug}
        with db() as conn:
            model = build_photo_report(conn, first, datetime.now(), consumer.effective_policy(), site, chair_names(),
                                       live=consumer.live()["customers"],
                                       clock=footage_clock(cfg.footage_start, consumer.clip_run))

        def image(rel: str) -> bytes | None:
            p = crops.inside(cfg.crops_dir, rel)
            return p.read_bytes() if p is not None and p.is_file() else None

        return model, image

    def render_photo_report(first: date, lang: str, format: str) -> bytes:
        """RendererUnavailable: no Chrome for the PDF."""
        model, image = photo_parts(first)
        if format == "html":
            return render_photo_html(model, lang, image, shop_logo=report_logo()).encode()
        return html_to_pdf(render_photo_html(model, lang, image, printing=True, shop_logo=report_logo()), timeout=180,
                           tmp_dir=pdf_tmp)

    @app.get("/api/barber/report/photos/allowed")
    def photo_report_allowed(request: Request):
        """Whether this client may have the haircut photo report (the shop network only): the Report panel's links."""
        return {"allowed": is_local_client(request.client.host if request.client else None)}

    @app.get("/api/barber/report/photos")
    def photo_report(request: Request, start: str, lang: str = "en", format: str = "html", download: bool = False):
        """One shop day's haircuts with their start and end photos: html (inline; download=1 as a file) or pdf.
        Photos of customers: the shop network only."""
        if not is_local_client(request.client.host if request.client else None):
            raise HTTPException(status_code=403, detail="haircut photos are shown on the shop network only")
        if lang not in LANGS or format not in ("html", "pdf"):
            raise HTTPException(status_code=400, detail="lang or format not recognised")
        try:
            first = date.fromisoformat(start)
        except ValueError:
            raise HTTPException(status_code=400, detail="start must be YYYY-MM-DD") from None
        try:
            body = render_photo_report(first, lang, format)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        except RendererUnavailable as exc:
            logger.warning("report renderer unavailable: %s", exc)
            raise HTTPException(status_code=503, detail="report renderer unavailable") from None
        s, e = period_bounds("day", first, int(consumer.effective_policy()["day_start_hour"]))
        return file_response(body, photo_file_name(s, e, cfg.place, cfg.branch_slug or "branch", format), format, download)

    @app.get("/api/barber/reports/saved")
    def saved_reports_list(request: Request, kind: str = "day"):
        """The Reports page's list: the complete saved folders of a kind, newest first. Metadata only, so it answers
        off the shop network too; `allowed` says whether this client may open the files."""
        if kind not in KINDS:
            raise HTTPException(status_code=400, detail="kind must be day, week or month")
        return {"enabled": app.state.saver.cfg is not None,
                "allowed": is_local_client(request.client.host if request.client else None),
                "items": list_saved(app.state.saver.root, kind)}

    @app.get("/api/barber/reports/saved/file")
    def saved_report_file(request: Request, id: str, name: str, download: bool = False):
        """One saved file, only one its folder's manifest lists. The files hold customer pictures: the shop network
        only."""
        if not is_local_client(request.client.host if request.client else None):
            raise HTTPException(status_code=403, detail="saved reports open on the shop network only")
        path = saved_file(app.state.saver.root, id, name)
        fmt = path.suffix.lstrip(".") if path is not None else ""
        if path is None or fmt not in REPORT_TYPES:
            raise HTTPException(status_code=404, detail="no such saved report")
        return file_response(path.read_bytes(), name, fmt, download)

    # saved reports (2026-10-25, user; saved_reports.py): each finished period's reports on disk, on the download code
    def saved_pages(period: Period, report: str, lang: str, docx: bool) -> Pages:
        logo = report_logo()
        if report == "photos":
            model, image = photo_parts(period.first)
            return Pages(render_photo_html(model, lang, image, shop_logo=logo),
                         render_photo_html(model, lang, image, printing=True, shop_logo=logo))
        model, image = analysis_parts(period.kind, period.first, lan=True)
        return Pages(render_html(model, lang, image, True, shop_logo=logo),
                     render_html(model, lang, image, True, printing=True, shop_logo=logo),
                     render_docx(model, lang, image, True, pdf_svg_to_png, shop_logo=logo) if docx else None)

    def saved_name(period: Period, report: str, fmt: str) -> str:
        if report == "photos":
            s, e = period_bounds("day", period.first, hour())
            return photo_file_name(s, e, cfg.place, cfg.branch_slug or "branch", fmt)
        return report_name(period.kind, period.first, fmt)

    def has_visits(start: datetime, end: datetime) -> bool:
        with db() as conn:
            return any_visit(conn, start, end)

    # a run on recordings (2026-10-18, user: its report at the end of the lap, in the footage's time): saved when the
    # clips play to their end -- the whole lap, just before its rows are cleared -- never by the clock; its day is the
    # footage's
    # 2026-10-25 final review: a clip run is never "live" -- without a footage_start it saves nothing (test data
    # must not land in a live site's reports folder)
    clips = any(c.clip_duration_s for c in cfg.cameras)
    recording = cfg.footage_start is not None and clips
    app.state.saver = ReportSaver(
        cfg.saved_reports, cfg.data_dir / "reports",
        Hooks(build=saved_pages, name=saved_name, has_visits=has_visits,
              photos=lambda: bool(consumer.effective_policy()["hp_photos"]),
              print_pdf=lambda html: html_to_pdf(html, timeout=180, tmp_dir=pdf_tmp), day_start_hour=hour),
        live=not clips)
    if cfg.saved_reports is not None and recording:
        def lap_end() -> None:
            foot = footage_clock(cfg.footage_start, consumer.clip_run)
            if foot is not None:
                day = shop_day(foot.end - timedelta(seconds=1), hour())
                logger.info("recording's lap ended: saving its %s reports", day)
                app.state.saver.save_lap(day, datetime.now())
        consumer.on_lap_end = lap_end

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await manager.connect(ws, first=live_payload())
        try:
            while True:
                await ws.receive_text()  # keepalive pings; ignored
        except WebSocketDisconnect:
            manager.disconnect(ws)

    return app


app = create_app()
