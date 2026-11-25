import asyncio
import logging
import time

import numpy as np
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class CommentFilter:
    """Suppresses repetitive events using cosine similarity on embeddings.

    Comments are compared against every recent comment AND every recent alert
    still inside the window — if any is too similar the comment is suppressed.
    Alerts are compared against every recent alert in the window. Comparing
    against the whole recent list (not just the single most recent entry) matters
    for a standing condition: a stopped truck reported once should still be caught
    if it is reported again a few cycles later, even though a handful of OTHER,
    unrelated alerts (a forklift, a different vehicle) fired in between and would
    otherwise have overwritten a single "last embedding" reference.
    """

    _MAX_RECENT = 20  # hard cap per type, independent of window_seconds, so a very
                      # long window on a busy stream can't grow this without bound

    def __init__(self, config: dict):
        dedup = config.get("dedup", {})
        self._enabled = bool(dedup.get("enabled", True))
        self._threshold = float(dedup.get("threshold", 0.92))
        self._backend = dedup.get("backend", "api")

        api_cfg = config.get("api", {})
        default_model = "text-embedding-v3" if self._backend == "api" else "BAAI/bge-small-en-v1.5"
        self._model = dedup.get("model", default_model)

        self._alert_threshold = float(dedup.get("alert_threshold", self._threshold))
        self._window_seconds  = float(dedup.get("window_seconds", 30))

        # event_type -> list of (embedding, stored_at), oldest first, pruned to
        # window_seconds/_MAX_RECENT on every read+write.
        self._recent: dict[str, list[tuple[list[float], float]]] = {"comment": [], "alert": []}
        self._pending: tuple[str, list[float], float] | None = None  # (event_type, embedding, ts)
        self._lock = asyncio.Lock()
        self._local_model = None  # lazy-loaded for the "local" backend

        if self._backend == "api":
            self._client = AsyncOpenAI(
                api_key=api_cfg.get("api_key"),
                base_url=api_cfg.get("base_url", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
            )

    def _prune(self, event_type: str) -> None:
        now = time.monotonic()
        kept = [e for e in self._recent[event_type] if now - e[1] <= self._window_seconds]
        self._recent[event_type] = kept[-self._MAX_RECENT:]

    async def is_duplicate(self, text: str, event_type: str = "comment") -> bool:
        """Return True if text is too similar to a recent event and should be suppressed.

        Comments are checked against recent comments and recent alerts.
        Alerts are checked against recent alerts only.
        """
        if not self._enabled:
            return False

        async with self._lock:
            try:
                embedding = await self._embed(text)
            except Exception:
                # Fail open: a missing/broken embedding backend (e.g. fastembed not
                # installed, or a DashScope hiccup) must never crash the analysis
                # worker or block a real event from being saved -- treat as "not a
                # duplicate" and leave _pending untouched, same as if dedup were off
                # for this one call.
                logger.warning("embed failed for %s dedup check — treating as not "
                                "duplicate", event_type, exc_info=True)
                return False
            self._prune("comment")
            self._prune("alert")
            refs = {"comment": list(self._recent["comment"]),
                    "alert":   list(self._recent["alert"])}

        # Hold the embedding; only committed to the reference store if record_emitted() is called.
        # Suppressed events must not shift the anchor — otherwise a later distinct event can bypass
        # dedup because it only compares against the suppressed event's embedding, not the original.
        self._pending = (event_type, embedding, time.monotonic())

        threshold = self._alert_threshold if event_type == "alert" else self._threshold

        compare = [(event_type, refs[event_type])]
        if event_type == "comment" and refs["alert"]:
            compare.append(("alert", refs["alert"]))

        for label, entries in compare:
            best_similarity, best_age = None, None
            for prev_embedding, stored_at in entries:
                similarity = _cosine(prev_embedding, embedding)
                if best_similarity is None or similarity > best_similarity:
                    best_similarity = similarity
                    best_age = time.monotonic() - stored_at
            if best_similarity is None:
                continue
            suppress = best_similarity >= threshold
            logger.info("%s vs %d recent %s(s): best similarity %.3f (age %.0fs, threshold %.3f) → %s",
                        event_type, len(entries), label, best_similarity, best_age, threshold,
                        "suppress" if suppress else "pass")
            if suppress:
                return True

        return False

    def record_emitted(self, event_type: str) -> None:
        """Commit the pending embedding to the reference store.

        Must be called only when the event is actually saved/broadcast.
        Suppressed or cooldown-dropped events must never call this.
        """
        if self._pending is None or self._pending[0] != event_type:
            return
        _, embedding, ts = self._pending
        self._recent[event_type].append((embedding, ts))
        self._recent[event_type] = self._recent[event_type][-self._MAX_RECENT:]
        self._pending = None

    async def _embed(self, text: str) -> list[float]:
        if self._backend == "api":
            t0 = time.monotonic()
            response = await self._client.embeddings.create(
                model=self._model,
                input=text,
                encoding_format="float",
            )
            logger.info("embed (%s): %.2fs", self._model, time.monotonic() - t0)
            return response.data[0].embedding

        # Local backend (fastembed) — runs in a thread executor (sync library)
        t0 = time.monotonic()
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._embed_local, text)
        logger.info("embed (local/%s): %.2fs", self._model, time.monotonic() - t0)
        return result

    def _embed_local(self, text: str) -> list[float]:
        if self._local_model is None:
            from fastembed import TextEmbedding  # optional dependency
            self._local_model = TextEmbedding(self._model)
        return list(self._local_model.embed([text]))[0].tolist()


def _cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom else 0.0
