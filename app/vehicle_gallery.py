"""Vehicle crop gallery API endpoints for IRIS FastAPI app.

Read-only from this side: the DeepStream sidecar (perception/modules/mtmc/vehicle_gallery.py)
picks the best-scoring crop per (camera, global vehicle id), uploads it to
MinIO (or local disk), and indexes it in the Redis hash ``iris:vehicle:crops``.
This module just serves that index and the image bytes back to the frontend.
"""
import json
import logging
import os
from typing import Optional, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, FileResponse
from pydantic import BaseModel
import redis

logger = logging.getLogger("iris.vehicle_gallery")

router = APIRouter(prefix="/api/vehicles", tags=["vehicles"])

_REDIS_KEY = "iris:vehicle:crops"

_redis_client: Optional[redis.Redis] = None
_minio_client = None
_bucket = "iris-vehicle-crops"
_storage_backend = "minio"
_local_dir = "data/vehicle_crops"
_plates_dir = "data/plates"


def init_redis(redis_url: str):
    global _redis_client
    _redis_client = redis.Redis.from_url(redis_url)


def init_storage(cfg: dict):
    """Configure how crop images are fetched, from config.yaml's vehicle_crops: section."""
    global _minio_client, _bucket, _storage_backend, _local_dir
    _storage_backend = cfg.get("storage_backend", "minio")
    _bucket = cfg.get("minio_bucket", "iris-vehicle-crops")
    _local_dir = cfg.get("local_dir", "data/vehicle_crops")
    if _storage_backend == "minio":
        try:
            from minio import Minio
            _minio_client = Minio(
                cfg.get("minio_endpoint", "minio:9000"),
                access_key=os.environ.get("MINIO_ACCESS_KEY", cfg.get("minio_access_key", "minioadmin")),
                secret_key=os.environ.get("MINIO_SECRET_KEY", cfg.get("minio_secret_key", "minioadmin")),
                secure=bool(cfg.get("minio_secure", False)),
            )
        except Exception as exc:
            logger.warning("Vehicle crop MinIO client init failed (image serving disabled): %s", exc)
            _minio_client = None


def _get_redis() -> redis.Redis:
    if _redis_client is None:
        raise HTTPException(status_code=503, detail="Redis not initialized")
    return _redis_client


def crop_field(cam_id: str, global_id: int, stay: int = 0) -> str:
    """Index field / file suffix of one stay's crop (stay 0 keeps the original names)."""
    return f"{cam_id}:{global_id}" + (f":{stay}" if stay else "")


class VehicleCropEntry(BaseModel):
    cam_id: str
    global_id: int
    stay: int = 0
    score: float = 0.0
    conf: float = 0.0
    updated_at: float = 0.0


@router.get("/crops", response_model=List[VehicleCropEntry])
async def list_crops(global_id: Optional[int] = None):
    """List best-shot vehicle crops, optionally filtered to one global_id."""
    r = _get_redis()
    try:
        data = r.hgetall(_REDIS_KEY)
        out = []
        for k, v in data.items():
            k = k.decode() if isinstance(k, bytes) else k
            v = v.decode() if isinstance(v, bytes) else v
            entry = json.loads(v)
            if global_id is not None and int(entry.get("global_id", -1)) != global_id:
                continue
            out.append(VehicleCropEntry(
                cam_id=entry.get("cam_id", ""),
                global_id=int(entry.get("global_id", -1)),
                stay=int(entry.get("stay", 0)),
                score=entry.get("score", 0.0),
                conf=entry.get("conf", 0.0),
                updated_at=entry.get("updated_at", 0.0),
            ))
        return out
    except Exception as e:
        logger.error("Failed to list vehicle crops: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/plates/{global_id}/readings")
async def get_plate_readings(global_id: int):
    """What the plate reader got from each camera's best plate of this vehicle (best-read first)."""
    path = os.path.join(_plates_dir, f"{global_id}.json")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="No plate read for this vehicle yet")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@router.get("/plates/{global_id}/image")
async def get_plate_image(global_id: int):
    """The picture of this vehicle's plates: each camera's crop with its reading beside it."""
    path = os.path.join(_plates_dir, f"{global_id}.png")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="No plate read for this vehicle yet")
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-cache"})


@router.get("/plates/{global_id}/{cam_id}/image")
async def get_plate_crop(global_id: int, cam_id: str):
    """The plate crop one camera got of this vehicle (what its reading was made from)."""
    if "/" in cam_id or ".." in cam_id:
        raise HTTPException(status_code=400, detail="bad camera id")
    path = os.path.join(_plates_dir, f"{global_id}_{cam_id}.jpg")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="No plate crop for this vehicle on this camera")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-cache"})


@router.get("/crops/{cam_id}/{global_id}/image")
async def get_crop_image(cam_id: str, global_id: int, stay: int = 0):
    """Serve the best-shot JPEG crop for one (camera, global vehicle id) pair; ``stay`` picks which of
    its stays on that camera (0 = the first, the default)."""
    r = _get_redis()
    raw = r.hget(_REDIS_KEY, crop_field(cam_id, global_id, stay))
    if raw is None:
        raise HTTPException(status_code=404, detail="No crop recorded for this vehicle on this camera")
    entry = json.loads(raw.decode() if isinstance(raw, bytes) else raw)

    backend = entry.get("backend", _storage_backend)
    if backend == "minio":
        if _minio_client is None:
            raise HTTPException(status_code=503, detail="MinIO not configured on this server")
        object_key = entry.get("object_key")
        bucket = entry.get("bucket", _bucket)
        if not object_key:
            raise HTTPException(status_code=404, detail="Crop index entry missing object_key")
        try:
            resp = _minio_client.get_object(bucket, object_key)
            try:
                return Response(content=resp.read(), media_type="image/jpeg")
            finally:
                resp.close()
                resp.release_conn()
        except Exception as exc:
            logger.warning("Failed to fetch vehicle crop %s/%s from MinIO: %s", bucket, object_key, exc)
            raise HTTPException(status_code=404, detail="Crop image not found in storage")
    else:
        # The sidecar records an absolute path from ITS container (/workspace/iris/...);
        # the app sees the same host dir at a different mount, so rebuild the path.
        path = os.path.join(_local_dir, cam_id, f"{global_id}{f'-{stay}' if stay else ''}.jpg")
        if os.path.exists(path):
            return FileResponse(path, media_type="image/jpeg")
        raise HTTPException(status_code=404, detail="Crop image not found on disk")
