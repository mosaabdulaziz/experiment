"""The HTTP API: what the product is doing, and the review queue an operator works.

Five endpoints (design §5.3, §5.4):

  GET  /status            the device, the weights, the classifier and the live paths
  GET  /cameras           the cameras this service watches
  GET  /candidates        the review queue, newest first
  GET  /events            the product's own history, newest first (since/camera/limit)
  POST /review/<id>       the operator's verdict: acknowledged | confirmed | dismissed

The app is a factory, so the API can be run without the Redis consumer (the tests do exactly
that) and the service's start-up order stays the boot gate's (design §8).
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from storage import STATUSES, ReviewQueue


class ReviewRequest(BaseModel):
    status: str
    note: str = ""


def create_app(status: dict, storage: ReviewQueue, cameras=()) -> FastAPI:
    """The API. `status` is the boot state; `storage` is the live review queue; `cameras` are
    the ids the service watches (the product's own config is the one source for them)."""
    app = FastAPI(title="Fall detection service")

    @app.get("/status")
    def get_status():
        """What the service runs with -- the same state the paths are derived from."""
        return status

    @app.get("/cameras")
    def get_cameras():
        """The ids the service watches, for the tab's camera list."""
        return list(cameras)

    @app.get("/candidates")
    def candidates(camera: str | None = None, status: str | None = None):
        """The review queue: the candidates that were too weak to confirm, with their evidence."""
        return storage.list_reviews(camera, status)

    @app.get("/events")
    def events(camera: str | None = None, since: int = 0, limit: int = 50):
        """The product's own history: every scored candidate, newest first.

        The tab polls this with its cursor; `since` is the last id it has seen, so the
        response is only what arrived since. A confirmed fall is never queued for review,
        so this is the only place it can live.
        """
        return storage.list_events(camera, since, limit)

    @app.post("/review/{row_id}")
    def review(row_id: int, body: ReviewRequest):
        """The operator's verdict on a queued candidate."""
        if body.status not in STATUSES:
            raise HTTPException(status_code=400,
                                detail=f"the status must be one of {', '.join(STATUSES)}")
        try:
            storage.set_review_status(row_id, body.status, body.note)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"no review row {row_id}") from None
        return {"id": row_id, "status": body.status, "note": body.note}

    return app
