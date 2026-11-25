"""Telegram messages for the safety product's incidents (products/safety; decisions D24/D27).

AlertEvents calls notify() when an incident opens and when it escalates. Whether it is sent is Settings' Telegram
switch (off by default), the feature's settings (products/safety/service.py: notify on, severity at or above its
minimum) and whether a bot token (TELEGRAM_BOT_TOKEN, .env) and a chat (Settings) exist. notify() only queues: the sending -- with the camera's current frame when there is one, and
retries -- runs on its own task, so a slow or unreachable Telegram never delays saving an alert. The queue is bounded;
when it is full the oldest message is dropped (and logged): a backlog of stale alerts helps nobody.
"""
from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
import uuid

from app.features import feature_of

logger = logging.getLogger("iris.notify")

API = "https://api.telegram.org"


def _post(url: str, body: bytes, content_type: str):
    """The call's `result`. Raises OSError with Telegram's own reason ("Unauthorized", "Bad Request: chat not
    found"): urllib turns its 4xx answers into HTTPError, whose text would only say "HTTP Error 400". Never the URL,
    which holds the token."""
    req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": content_type})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read() or b"{}")
        except ValueError:
            payload = {}
        raise OSError(f"telegram: {payload.get('description') or exc.reason}") from None
    except urllib.error.URLError as exc:
        raise OSError(f"telegram is unreachable: {exc.reason}") from None
    except Exception as exc:  # noqa: BLE001 -- e.g. http.client.InvalidURL, whose message quotes the URL (the token)
        raise OSError(f"telegram call failed ({type(exc).__name__}): is the bot token well formed?") from None
    if not payload.get("ok", False):
        raise OSError(f"telegram: {payload.get('description', 'refused the call')}")
    return payload.get("result")


def _env_token() -> str:
    """The bot token from the stack's .env, without the space or newline a pasted or quoted value can carry."""
    return os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()


def telegram_send(method: str, token: str, fields: dict, photo: "bytes | None" = None):
    """One Bot API call (sendMessage, getUpdates, or sendPhoto with the JPEG as multipart); its result. Raises on any
    failure."""
    url = f"{API}/bot{token}/{method}"
    if photo is None:
        return _post(url, urllib.parse.urlencode(fields).encode(), "application/x-www-form-urlencoded")
    boundary = uuid.uuid4().hex
    parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
             for k, v in fields.items()]
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="photo"; filename="frame.jpg"\r\n'
                 f'Content-Type: image/jpeg\r\n\r\n'.encode() + photo + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return _post(url, b"".join(parts), f"multipart/form-data; boundary={boundary}")


class TelegramNotifier:
    def __init__(self, safety, send=telegram_send, snapshot=None, maxsize: int = 100, attempts: int = 3,
                 retry_s: float = 2.0):
        self._safety = safety
        self._send = send
        self._snapshot = snapshot or (lambda sid: None)      # sid -> the camera's current JPEG, or None
        self._queue: collections.deque = collections.deque(maxlen=maxsize)
        self._attempts, self._retry_s = attempts, retry_s
        self._wake = asyncio.Event()
        self._busy = False
        self._task = None

    # ── producer side: never blocks, never raises ──────────────────────────────────────────────────────────────
    def notify(self, record: dict, how: str) -> None:
        """Queue an incident that `how` = "opened" or "escalated"; the worker decides whether it is sent."""
        try:
            if len(self._queue) == self._queue.maxlen:
                dropped, _ = self._queue[0]
                logger.warning("telegram queue full: dropping the message for event %s", dropped.get("id"))
            self._queue.append((record, how))
            self._wake.set()
        except Exception:  # noqa: BLE001 -- a notification problem must never fail the alert
            logger.exception("could not queue a telegram message")

    def pending(self) -> list:
        return list(self._queue)

    # ── worker ────────────────────────────────────────────────────────────────────────────────────────────────
    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def drain(self) -> None:
        """Wait until every queued message is handled (tests, shutdown)."""
        while self._queue or self._busy:
            await asyncio.sleep(0.01)

    async def _run(self) -> None:
        while True:
            if not self._queue:
                self._wake.clear()
                await self._wake.wait()
                continue
            record, how = self._queue.popleft()
            self._busy = True
            try:
                await self._handle(record, how)
            except Exception:  # noqa: BLE001
                logger.exception("telegram message for event %s failed", record.get("id"))
            finally:
                self._busy = False

    async def _handle(self, record: dict, how: str) -> None:
        feature = record.get("feature") or feature_of(record.get("camera"), record.get("tags"))
        telegram = self._safety.settings()["telegram"]
        if not telegram["enabled"] or not self._safety.should_notify(feature, record.get("severity")):
            return
        token = _env_token()
        chat = telegram["chat_id"]
        if not token or not chat:
            logger.info("event %s would be sent to Telegram, but the bot token or the chat is not set",
                        record.get("id"))
            return
        text = self._text(record, how, feature)
        photo = await asyncio.to_thread(self._snapshot, record.get("camera"))
        method, fields = (("sendPhoto", {"chat_id": chat, "caption": text}) if photo
                          else ("sendMessage", {"chat_id": chat, "text": text}))
        for attempt in range(1, self._attempts + 1):
            try:
                await asyncio.to_thread(self._send, method, token, fields, photo)
                logger.info("telegram: event %s (%s %s) sent%s", record.get("id"), feature, how,
                            "" if attempt == 1 else f" on attempt {attempt}")
                return
            except Exception as exc:  # noqa: BLE001
                if attempt == self._attempts:
                    logger.error("telegram: event %s not sent after %d attempts: %s", record.get("id"), attempt, exc)
                    return
                await asyncio.sleep(self._retry_s * attempt)

    # ── setting it up (Settings: Find chats, Send a test message) ────────────────────────────────────────────────
    def _token(self) -> str:
        token = _env_token()
        if not token:
            raise LookupError("no bot token: set TELEGRAM_BOT_TOKEN in the stack's .env and start the stack again")
        return token

    async def send_test(self) -> None:
        """One message to the saved chat, now, without retries, with Telegram on or off (to set it up before turning
        it on). LookupError: no token or no chat; OSError: Telegram refused it (its reason) or is unreachable."""
        token = self._token()
        chat = self._safety.settings()["telegram"]["chat_id"]
        if not chat:
            raise LookupError("no chat: choose the chat in Settings and save it")
        text = "Site safety: test message. Incidents of the features set to notify will arrive in this chat."
        await asyncio.to_thread(self._send, "sendMessage", token, {"chat_id": chat, "text": text})

    async def recent_chats(self) -> list:
        """The chats the bot was recently added to or written in (Telegram keeps a day of updates), each once:
        {id, title, type}. Read without an offset, so nothing is consumed."""
        updates = await asyncio.to_thread(self._send, "getUpdates", self._token(), {"timeout": 0}) or []
        chats: dict = {}
        for update in updates:
            for kind in ("my_chat_member", "message", "edited_message", "channel_post", "edited_channel_post"):
                chat = (update.get(kind) or {}).get("chat")
                if chat and str(chat["id"]) not in chats:
                    title = chat.get("title") or chat.get("username") or chat.get("first_name") or str(chat["id"])
                    chats[str(chat["id"])] = {"id": str(chat["id"]), "title": title, "type": chat.get("type", "")}
        return list(chats.values())

    def _text(self, record: dict, how: str, feature: "str | None") -> str:
        names = {c["id"]: c["name"] for c in self._safety.cameras()}
        camera = record.get("camera_id") or record.get("camera")
        name = names.get(camera, camera)
        message = str(record.get("message") or "")
        stream = record.get("camera") or ""
        if "@" in stream and name:
            message = message.replace(stream, name)      # the app's words name a role's stream key: "... on store@fire"
        head = "Escalated" if how == "escalated" else "New"
        return (f"{head} {str(record.get('severity') or 'warning').upper()} -- {feature or 'alert'} on "
                f"{name}\n{message}")
