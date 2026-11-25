"""Face recognition API endpoints for IRIS FastAPI app.

Provides REST endpoints for managing the face gallery and querying recognition results.
"""
import os
import json
import time
import logging
import threading
from typing import Optional, List
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import redis
import numpy as np
import cv2

logger = logging.getLogger("iris.face_recognition")

router = APIRouter(prefix="/api/faces", tags=["faces"])

# Redis connection (will be initialized from config)
_redis_client: Optional[redis.Redis] = None
_face_gallery_key = "iris:face:gallery"
_face_rec_key_prefix = "iris:face_rec:"
_face_captures_key = "iris:face:captures"
_face_capture_img_prefix = "iris:face:capture_img:"
_face_captures_dir = "/app/data/face_captures"  # shared volume, also written by sidecar
_face_app_lock = threading.Lock()
_face_app_cache: tuple = ()


def init_redis(redis_url: str):
    """Initialize Redis client."""
    global _redis_client
    _redis_client = redis.Redis.from_url(redis_url)


def get_redis() -> redis.Redis:
    if _redis_client is None:
        raise HTTPException(status_code=503, detail="Redis not initialized")
    return _redis_client


def _embed_face(img: np.ndarray) -> np.ndarray:
    """Compute the 512-d ArcFace embedding for the face in *img*.

    Runs InsightFace on CPU with the same ``auraface`` model the sidecar uses,
    so gallery embeddings are directly comparable to live capture embeddings.

    Raises:
        HTTPException(503) if InsightFace is not installed in this container.
        HTTPException(400) if no face is detected in the image.
    """
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        raise HTTPException(status_code=503, detail="InsightFace not available in API container")

    try:      # the repo's DVC copy (./perception is mounted into this container), never downloaded at run time
        from perception import insightface_pack
        pack_root = insightface_pack.root("auraface")
    except ImportError:
        raise HTTPException(status_code=503, detail="InsightFace model pack unavailable: perception/ (with "
                                                    "perception/insightface_pack.py) is not mounted in this container")
    except insightface_pack.InsightFacePackMissing as exc:
        raise HTTPException(status_code=503, detail=f"InsightFace model pack unavailable: {exc}")

    global _face_app_cache
    with _face_app_lock:
        if len(_face_app_cache) == 3 and _face_app_cache[0] is FaceAnalysis and _face_app_cache[1] == pack_root:
            app = _face_app_cache[2]
        else:
            app = FaceAnalysis(name="auraface", root=pack_root, allowed_modules=['detection', 'recognition'])
            app.prepare(ctx_id=-1, det_thresh=0.5, det_size=(640, 640))  # ctx_id=-1 = CPU
            _face_app_cache = (FaceAnalysis, pack_root, app)

    faces = app.get(img, max_num=1)
    if not faces:
        raise HTTPException(status_code=400, detail="No face detected in the uploaded image")
    return faces[0].embedding  # (512,)


# Pydantic models
class FaceRegisterRequest(BaseModel):
    name: str
    face_id: Optional[str] = None  # Auto-generated if not provided


class FaceResponse(BaseModel):
    id: str
    name: str
    photo_path: str = ""
    created_at: float = 0.0


class FaceRecognitionResult(BaseModel):
    parent_id: Optional[int] = None
    name: str
    confidence: float
    rec_id: Optional[str] = None


class FaceRecognitionResponse(BaseModel):
    source_id: str
    pts: float
    results: List[FaceRecognitionResult]


class FaceCaptureResponse(BaseModel):
    id: str
    source_id: str
    track_id: int
    pts: float
    captured_at: float
    assigned_to: Optional[str] = None


class AssignCaptureRequest(BaseModel):
    name: str  # Person name to assign this capture to


@router.get("", response_model=List[FaceResponse])
async def list_faces():
    """List all registered faces in the gallery."""
    r = get_redis()
    try:
        data = r.hgetall(_face_gallery_key)
        faces = []
        for k, v in data.items():
            k = k.decode() if isinstance(k, bytes) else k
            v = v.decode() if isinstance(v, bytes) else v
            entry = json.loads(v)
            faces.append(FaceResponse(
                id=k,
                name=entry.get("name", ""),
                photo_path=entry.get("photo_path", ""),
                created_at=entry.get("created_at", 0.0)
            ))
        return faces
    except Exception as e:
        logger.error("Failed to list faces: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/register", response_model=FaceResponse)
async def register_face(
    name: str = Form(...),
    face_id: Optional[str] = Form(None),
    file: UploadFile = File(...)
):
    """Register a new face from an uploaded photo.
    
    The photo should contain a clear face. The embedding will be computed
    and stored in the gallery for future recognition.
    """
    r = get_redis()
    
    # Read and decode image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    # Compute the face embedding (CPU, auraface, same model as the sidecar so
    # the stored embedding is directly comparable to live captures).
    embedding = _embed_face(img)
    
    # Generate face_id if not provided
    if face_id is None:
        import uuid
        face_id = str(uuid.uuid4())[:8]
    
    # Save photo
    photo_dir = "/app/data/face_photos"
    os.makedirs(photo_dir, exist_ok=True)
    photo_path = os.path.join(photo_dir, f"{face_id}.jpg")
    cv2.imwrite(photo_path, img)
    
    # Store in Redis (use ``embeddings`` list — supports multiple per person)
    entry = {
        "name": name,
        "embeddings": [embedding.tolist()],
        "photo_path": photo_path,
        "created_at": time.time()
    }
    r.hset(_face_gallery_key, face_id, json.dumps(entry))
    
    logger.info("Registered face: %s (%s)", face_id, name)
    
    return FaceResponse(
        id=face_id,
        name=name,
        photo_path=photo_path,
        created_at=entry["created_at"]
    )


@router.get("/{face_id}/image")
async def get_face_image(face_id: str):
    """Serve the registered face photo for a known person.
    
    Resolution order:
    1. Standard path: /app/data/face_photos/{face_id}.jpg
    2. Stored photo_path from the Redis gallery entry (handles CLI-registered
       faces whose photo was not copied into the standard directory)
    """
    # 1 ── Try standard path (API-registered faces) ────────────────────────
    photo_dir = "/app/data/face_photos"
    img_path = os.path.join(photo_dir, f"{face_id}.jpg")
    if os.path.exists(img_path):
        from fastapi.responses import FileResponse
        return FileResponse(img_path, media_type="image/jpeg")

    # 2 ── Fallback: read photo_path from Redis gallery entry ──────────────
    try:
        r = get_redis()
        raw = r.hget(_face_gallery_key, face_id)
        if raw is not None:
            entry = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
            stored_path = entry.get("photo_path", "")
            if stored_path and os.path.exists(stored_path):
                from fastapi.responses import FileResponse
                return FileResponse(stored_path, media_type="image/jpeg")
    except Exception as e:
        logger.warning("Failed to look up face %s in Redis fallback: %s", face_id, e)

    raise HTTPException(status_code=404, detail="Face image not found")


@router.delete("/{face_id}")
async def delete_face(face_id: str):
    """Delete a face from the gallery."""
    r = get_redis()
    deleted = r.hdel(_face_gallery_key, face_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Face not found")
    
    # Also delete photo file
    photo_dir = "/app/data/face_photos"
    photo_path = os.path.join(photo_dir, f"{face_id}.jpg")
    if os.path.exists(photo_path):
        os.remove(photo_path)
    
    logger.info("Deleted face: %s", face_id)
    return {"message": "Face deleted", "face_id": face_id}


@router.get("/recognition/{source_id}", response_model=FaceRecognitionResponse)
async def get_latest_recognition(source_id: str):
    """Get the latest face recognition results for a source."""
    r = get_redis()
    try:
        key = f"{_face_rec_key_prefix}{source_id}"
        entries = r.xrevrange(key, count=1)
        if not entries:
            raise HTTPException(status_code=404, detail="No recognition data available")
        
        _id, fields = entries[0]
        pts = float(fields.get(b"pts", b"0").decode())
        face_rec = json.loads(fields.get(b"face_rec", b"[]").decode())
        
        results = [
            FaceRecognitionResult(
                parent_id=r["parent_id"],
                name=r["name"],
                confidence=r["confidence"],
                rec_id=r.get("rec_id")
            )
            for r in face_rec
        ]
        
        return FaceRecognitionResponse(
            source_id=source_id,
            pts=pts,
            results=results
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get recognition results: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recognition/{source_id}/history")
async def get_recognition_history(
    source_id: str,
    count: int = Query(50, ge=1, le=200)
):
    """Get recent face recognition history for a source."""
    r = get_redis()
    try:
        key = f"{_face_rec_key_prefix}{source_id}"
        entries = r.xrevrange(key, count=count)
        
        history = []
        for _id, fields in entries:
            pts = float(fields.get(b"pts", b"0").decode())
            face_rec = json.loads(fields.get(b"face_rec", b"[]").decode())
            history.append({
                "pts": pts,
                "results": face_rec
            })
        
        return {"source_id": source_id, "history": history}
    except Exception as e:
        logger.error("Failed to get recognition history: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ── CCTV face capture endpoints ────────────────────────────────────────────

@router.get("/captures", response_model=List[FaceCaptureResponse])
async def list_captures():
    """List all face captures from CCTV footage (unknown faces)."""
    r = get_redis()
    try:
        data = r.hgetall(_face_captures_key)
        out = []
        for k, v in data.items():
            k = k.decode() if isinstance(k, bytes) else k
            v = v.decode() if isinstance(v, bytes) else v
            entry = json.loads(v)
            out.append(FaceCaptureResponse(
                id=k,
                source_id=entry.get("source_id", ""),
                track_id=entry.get("track_id", 0),
                pts=entry.get("pts", 0.0),
                captured_at=entry.get("captured_at", 0.0),
                assigned_to=entry.get("assigned_to"),
            ))
        return sorted(out, key=lambda x: x.captured_at, reverse=True)
    except Exception as e:
        logger.error("Failed to list face captures: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/captures/{capture_id}/image")
async def get_capture_image(capture_id: str):
    """Serve the thumbnail JPEG of a face capture."""
    r = get_redis()
    # Try Redis cache first
    try:
        cached = r.get(f"{_face_capture_img_prefix}{capture_id}")
        if cached is not None:
            img_bytes = cached if isinstance(cached, bytes) else cached.encode()
            from fastapi.responses import Response
            return Response(content=img_bytes, media_type="image/jpeg")
    except Exception:
        pass
    # Fallback: read from shared disk volume
    img_path = os.path.join(_face_captures_dir, f"{capture_id}.jpg")
    if os.path.exists(img_path):
        from fastapi.responses import FileResponse
        return FileResponse(img_path, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="Capture image not found")


@router.post("/captures/{capture_id}/assign")
async def assign_capture(capture_id: str, body: AssignCaptureRequest):
    """Assign a capture to a known person.
    
    1. Look up the existing person in the gallery by name.
    2. Append the capture's embedding to that person's ``embeddings``
       list so the sidecar can recognise this face in future frames.
    3. Mark the capture as assigned (kept in the captures list so it
       renders under that person in the frontend).
    """
    r = get_redis()
    
    # Read capture entry
    raw = r.hget(_face_captures_key, capture_id)
    if raw is None:
        raise HTTPException(status_code=404, detail="Capture not found")
    cap_entry = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
    
    if cap_entry.get("assigned_to"):
        raise HTTPException(status_code=400, detail=f"Capture already assigned to '{cap_entry['assigned_to']}'")
    
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name must not be empty")
    
    # Find the person in the gallery by name.
    # NOTE: hgetall returns bytes keys; decode once into a fresh dict so we can
    # safely index by the decoded person_id below (avoids KeyError with bytes keys).
    gallery_raw = r.hgetall(_face_gallery_key)
    gallery = {}
    for k, v in gallery_raw.items():
        k = k.decode() if isinstance(k, bytes) else k
        v = v.decode() if isinstance(v, bytes) else v
        gallery[k] = json.loads(v)
    person_id = None
    for k, entry in gallery.items():
        if entry.get("name", "") == name:
            person_id = k
            break
    if person_id is None:
        raise HTTPException(status_code=400, detail=f"No known person named '{name}' in the gallery")
    
    # Append the capture's embedding to the person's gallery entry
    capture_emb = cap_entry.get("embedding")
    if capture_emb:
        gallery_entry = gallery[person_id]
        # Support both old (embedding) and new (embeddings) format
        if "embeddings" not in gallery_entry:
            old_emb = gallery_entry.get("embedding")
            gallery_entry["embeddings"] = [old_emb] if old_emb is not None else []
        gallery_entry["embeddings"].append(capture_emb)
        r.hset(_face_gallery_key, person_id, json.dumps(gallery_entry))
        logger.info("Appended capture embedding to '%s' (now %d embeddings)",
                    name, len(gallery_entry["embeddings"]))
    
    # Mark the capture as assigned — keep it in the captures list
    cap_entry["assigned_to"] = name
    r.hset(_face_captures_key, capture_id, json.dumps(cap_entry))
    
    logger.info("Capture %s assigned to '%s'", capture_id, name)
    return {"message": f"Capture assigned to '{name}'", "face_id": person_id, "name": name}


@router.delete("/captures/{capture_id}")
async def delete_capture(capture_id: str):
    """Delete/reject a face capture."""
    r = get_redis()
    deleted = r.hdel(_face_captures_key, capture_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Capture not found")
    
    r.delete(f"{_face_capture_img_prefix}{capture_id}")
    img_path = os.path.join(_face_captures_dir, f"{capture_id}.jpg")
    if os.path.exists(img_path):
        os.remove(img_path)
    
    logger.info("Deleted face capture: %s", capture_id)
    return {"message": "Capture deleted", "capture_id": capture_id}


# ── Add images to existing persons ──────────────────────────────────────────

@router.post("/{face_id}/images", response_model=FaceCaptureResponse)
async def add_person_image(face_id: str, file: UploadFile = File(...)):
    """Upload an additional image for an existing person.

    The image is embedded (InsightFace CPU) and the embedding is appended to the
    person's gallery entry so the sidecar's recognizer can match against it (it
    reloads the gallery every ~30s).  The image is also stored as a capture-like
    entry assigned to that person, so it appears alongside their CCTV captures
    in the gallery.
    """
    r = get_redis()

    # 1. Verify the person exists and get their name
    raw = r.hget(_face_gallery_key, face_id)
    if raw is None:
        raise HTTPException(status_code=404, detail="Person not found")
    entry = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
    person_name = entry.get("name", "")
    if not person_name:
        raise HTTPException(status_code=400, detail="Person has no name")

    # 2. Read and save the image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # 3. Compute the face embedding. The whole point of adding an image is that
    #    recognition can now match this person, so the photo MUST be embedded
    #    (CPU, same auraface model as the sidecar). Non-face images get a 400
    #    that the frontend surfaces to the user.
    embedding = _embed_face(img)

    import uuid
    img_id = str(uuid.uuid4())[:8]
    os.makedirs(_face_captures_dir, exist_ok=True)
    dst_path = os.path.join(_face_captures_dir, f"{img_id}.jpg")
    cv2.imwrite(dst_path, img)

    # 4. Append the embedding to the person's gallery entry so the sidecar's
    #    recognizer can match against this photo (it reloads the gallery every
    #    ~30s).  Handles both the legacy ``embedding`` and the current
    #    ``embeddings`` list formats.
    if "embeddings" not in entry:
        old_emb = entry.get("embedding")
        entry["embeddings"] = [old_emb] if old_emb is not None else []
    entry["embeddings"].append(embedding.tolist())
    r.hset(_face_gallery_key, face_id, json.dumps(entry))
    logger.info("Appended image embedding to '%s' (now %d embeddings)",
                person_name, len(entry["embeddings"]))

    # 5. Store as a capture assigned to that person (with its real embedding)
    capture_entry = {
        "embedding": embedding.tolist(),
        "source_id": "manual_upload",
        "track_id": 0,
        "pts": 0.0,
        "captured_at": time.time(),
        "assigned_to": person_name,
    }
    r.hset(_face_captures_key, img_id, json.dumps(capture_entry))
    # Cache image bytes in Redis for fast serving
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if ok:
        r.setex(f"{_face_capture_img_prefix}{img_id}", 86400, buf.tobytes())

    logger.info("Uploaded additional image for '%s': %s", person_name, img_id)

    return FaceCaptureResponse(
        id=img_id,
        source_id="manual_upload",
        track_id=0,
        pts=0.0,
        captured_at=capture_entry["captured_at"],
        assigned_to=person_name,
    )