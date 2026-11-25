"""/api/safety/*: the safety product's settings, cameras and Apply, and the incidents waiting for an acknowledgement.
Mounted when IRIS_PRODUCT names `safety`; the logic is products/safety/service.py (SafetyService, app.state.safety)."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

router = APIRouter(prefix="/api/safety", tags=["safety"])


def _error(message: str, status: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


def _streams(request: Request) -> list:
    sm = getattr(request.app.state, "stream_manager", None)
    return sm.list_streams() if sm is not None else []


def _fps(request: Request) -> dict:
    """Each stream's frames a second, as camera health last measured them (none without it)."""
    health = getattr(request.app.state, "safety_health", None)
    return dict(health.fps) if health is not None else {}


@router.get("/settings")
def get_settings(request: Request):
    return request.app.state.safety.settings()


@router.put("/settings")
async def put_settings(request: Request):
    try:
        return request.app.state.safety.update_settings(await request.json())
    except ValueError as exc:
        return _error(str(exc), 400)


@router.get("/cameras")
def get_cameras(request: Request):
    """Each camera with its name, features (toggles) and the role streams running for it."""
    return {"cameras": request.app.state.safety.cameras(_streams(request), _fps(request))}


@router.put("/cameras/{camera_id}")
async def put_camera(camera_id: str, request: Request):
    """Rename a camera and/or set its features. Saved, not yet live: POST /api/safety/apply applies them."""
    body = await request.json()
    try:
        request.app.state.safety.update_camera(camera_id, name=body.get("name"), features=body.get("features"))
    except KeyError:
        return _error(f"no camera {camera_id!r}", 404)
    except ValueError as exc:
        return _error(str(exc), 400)
    return next(c for c in request.app.state.safety.cameras(_streams(request), _fps(request)) if c["id"] == camera_id)


@router.get("/cameras/{camera_id}/snapshot")
def get_snapshot(camera_id: str, request: Request):
    """A still of the camera straight from go2rtc, no sidecar needed: the wall's tile and the zone editor's picture
    for a camera no feature runs on."""
    try:
        data = request.app.state.safety.snapshot(camera_id)
    except KeyError:
        return _error(f"no camera {camera_id!r}", 404)
    except ConnectionError as exc:
        return _error(f"no picture from the camera: {exc}", 502)
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.post("/apply")
def post_apply(request: Request):
    """Ask the host controller to apply the saved toggles; `applying` stays true until it has."""
    request.app.state.safety.request_apply()
    return request.app.state.safety.apply_status()


@router.get("/apply")
def get_apply(request: Request):
    return request.app.state.safety.apply_status()


@router.get("/unacknowledged")
async def get_unacknowledged(request: Request):
    """Incidents of features that require an acknowledgement, not yet acknowledged (ended ones included)."""
    return {"events": await request.app.state.safety.unacknowledged()}


async def _fall(request: Request, method: str, path: str, body: "dict | None" = None) -> JSONResponse:
    try:
        status, payload = await asyncio.to_thread(request.app.state.safety.fall_http, method, path, body)
    except LookupError as exc:
        return _error(str(exc), 503)
    except ConnectionError as exc:
        return _error(f"the fall service is unreachable: {exc}", 502)
    return JSONResponse(payload, status_code=status)


@router.get("/fall/candidates")
async def fall_candidates(request: Request):
    """The fall service's review queue (borderline candidates), passed through with its query (camera, status)."""
    query = request.url.query
    return await _fall(request, "GET", "/candidates" + (f"?{query}" if query else ""))


@router.post("/fall/review/{row_id}")
async def fall_review(row_id: int, request: Request):
    """The operator's verdict on a candidate ({status: acknowledged|confirmed|dismissed, note}), to the fall service."""
    return await _fall(request, "POST", f"/review/{row_id}", await request.json())


def _notifier(request: Request):
    return getattr(request.app.state, "safety_notifier", None)


async def _telegram(call) -> JSONResponse:
    try:
        return JSONResponse(await call())
    except LookupError as exc:
        return _error(str(exc), 400)
    except OSError as exc:
        return _error(str(exc), 502)


@router.post("/telegram/test")
async def telegram_test(request: Request):
    """Send a test message to the saved chat (setting Telegram up). 400: no token or no chat; 502: Telegram's reason."""
    async def send():
        await _notifier(request).send_test()
        return {"ok": True}
    return await _telegram(send)


@router.get("/telegram/chats")
async def telegram_chats(request: Request):
    """The chats the bot was recently added to or written in, to choose the one incidents go to."""
    async def chats():
        return {"chats": await _notifier(request).recent_chats()}
    return await _telegram(chats)
