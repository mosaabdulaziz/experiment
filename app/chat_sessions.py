import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ChatSession:
    session_id: str
    turns: list[dict] = field(default_factory=list)  # [{"role": "user"|"assistant", "content": str}, ...]
    last_active: float = field(default_factory=time.monotonic)
    sweep: dict | None = None  # {"history_index": int, "since": float} for resume, or None


class ChatSessionStore:
    """In-memory chat session registry. Sessions die with the process by design —
    matches the same in-memory pattern used for monitoring instructions."""

    def __init__(self, config: dict):
        chat_cfg = config.get("chat", {})
        self._max_turns = max(int(chat_cfg.get("max_turns", 8)), 1)
        self._ttl_seconds = max(int(chat_cfg.get("session_ttl_seconds", 1800)), 30)
        self._sessions: dict[str, ChatSession] = {}
        self._lock = asyncio.Lock()

    @property
    def ttl_seconds(self) -> int:
        return self._ttl_seconds

    async def get_or_create(self, session_id: str | None) -> ChatSession:
        async with self._lock:
            if session_id and session_id in self._sessions:
                session = self._sessions[session_id]
                session.last_active = time.monotonic()
                return session
            new_id = session_id or uuid.uuid4().hex
            session = ChatSession(session_id=new_id)
            self._sessions[new_id] = session
            return session

    async def append_turn(self, session_id: str, role: str, content: str) -> None:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            session.turns.append({"role": role, "content": content})
            # Keep last `max_turns` user/assistant pairs (max_turns * 2 messages)
            limit = self._max_turns * 2
            if len(session.turns) > limit:
                session.turns = session.turns[-limit:]
            session.last_active = time.monotonic()

    async def delete(self, session_id: str) -> bool:
        async with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
        return False

    async def set_sweep(self, session_id: str, history_index: int, since: float) -> None:
        async with self._lock:
            s = self._sessions.get(session_id)
            if s is not None:
                s.sweep = {"history_index": int(history_index), "since": float(since)}

    async def get_sweep(self, session_id: str) -> dict | None:
        async with self._lock:
            s = self._sessions.get(session_id)
            return dict(s.sweep) if s is not None and s.sweep is not None else None

    async def clear_sweep(self, session_id: str) -> None:
        async with self._lock:
            s = self._sessions.get(session_id)
            if s is not None:
                s.sweep = None

    async def prune_expired(self) -> int:
        now = time.monotonic()
        async with self._lock:
            expired = [sid for sid, s in self._sessions.items() if now - s.last_active > self._ttl_seconds]
            for sid in expired:
                del self._sessions[sid]
        if expired:
            logger.info("Pruned %d idle chat session(s)", len(expired))
        return len(expired)

    async def run_prune_loop(self, check_interval: float = 60.0) -> None:
        """Background task: periodically drop sessions idle past their TTL."""
        while True:
            try:
                await asyncio.sleep(check_interval)
                await self.prune_expired()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # never let the loop die silently
                logger.error("Chat session prune failed: %s", exc, exc_info=True)
