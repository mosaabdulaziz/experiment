"""FastAPI HTTP endpoints for the concealment detection review queue and status."""
from __future__ import annotations

from collections.abc import Callable

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from storage import ReviewStore


class ReviewRequest(BaseModel):
    status: str
    note: str = ""
    reviewer: str = "staff"


def create_app(
    status: dict | Callable[[], dict], store: ReviewStore, cameras=(), *, lifespan=None,
) -> FastAPI:
    """Create the review API with optional runtime status and managed lifespan."""
    app = FastAPI(title="IRIS Suspected Concealment Review Service", lifespan=lifespan)
    camera_list = list(cameras)

    @app.get("/status")
    def get_status():
        return status() if callable(status) else status

    @app.get("/healthz")
    def get_health():
        snapshot = get_status()
        return JSONResponse(snapshot, status_code=200 if snapshot.get("ready", True) else 503)

    @app.get("/cameras")
    def get_cameras():
        return camera_list

    @app.get("/candidates")
    def get_candidates(
        camera: str | None = None,
        review_status: str | None = None,
        since: int = 0,
        limit: int = 50,
        before: int | None = None,
    ):
        return store.list_candidates(
            camera=camera,
            status=review_status,
            since=since,
            limit=limit,
            before=before,
        )

    @app.get("/candidates/{candidate_id}")
    def get_candidate_detail(candidate_id: int):
        row = store.get_candidate(candidate_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"Candidate {candidate_id} not found")
        return row

    @app.post("/review/{candidate_id}")
    def post_review(candidate_id: int, body: ReviewRequest):
        try:
            store.set_review_status(
                candidate_id,
                status=body.status,
                note=body.note,
                reviewer=body.reviewer,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return store.get_candidate(candidate_id)

    @app.get("/assessments")
    def get_assessments(camera: str | None = None, limit: int = 50):
        return store.list_assessments(camera=camera, limit=limit)

    return app
