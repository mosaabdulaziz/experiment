"""iris:status:<id> writer."""

import json
import urllib.parse
import redis



def _write_status(r, source, running, error=None, pipeline_preview=True, boot_id=None, key_id=None):
    """`key_id` is the stream's key id (sources.stream_key_id); None = the camera id."""
    is_file = source["uri"].startswith("file://")
    status = {"running": running, "is_file": is_file, "source": urllib.parse.unquote(source["uri"].rsplit("/", 1)[-1])}
    if boot_id is not None:
        # One value per process, generated once in main() and unchanged across
        # this process's own play/pause transitions -- the app's StreamManager
        # watches for it changing to detect that this source's owning sidecar
        # process (re)started (e.g. after a crash), which is what tells it to
        # force-resolve any incident this source had left "ongoing": the
        # crashed process's in-memory FireAlertState is gone, so no
        # "fire_smoke_resolved" message for that incident will ever arrive.
        status["boot_id"] = boot_id
    # The raw endpoint the frontend can play directly (RTSP for live sources;
    # none exists for file:// uploads). In inference-only mode
    # (pipeline_preview=false) this is THE preview: the frontend plays
    # preview_uri and overlays boxes from iris:meta:<id>.
    if not is_file:
        status["preview_uri"] = source["uri"]
    status["preview_mode"] = "pipeline" if pipeline_preview else "raw"
    if error:
        status["error"] = error
    try:
        r.set(f"iris:status:{key_id or source['id']}", json.dumps(status))
    except redis.RedisError:
        pass
