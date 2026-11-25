"""products/safety/notifier.py: Telegram messages for the incidents that matter (safety integration, ticket 2, D24/D27).
Sent when an incident opens and when it escalates, only for features with notify on and at or above their minimum
severity; asynchronous and retried, never in the way of saving the alert."""
import asyncio
import pathlib

import fakeredis
import yaml

from products.safety.service import SafetyService
from products.safety.notifier import TelegramNotifier

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _service(tmp_path, monkeypatch, token="123:abc", chat="-100", enabled=True):
    if token:
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", token)
    else:
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    tmp_path.mkdir(parents=True, exist_ok=True)
    cams = tmp_path / "cameras.yaml"
    cams.write_text(yaml.safe_dump({"cameras": [{"id": "store", "name": "Store room", "clip": "s.mp4"}]}))
    svc = SafetyService({"safety": {"data_dir": str(tmp_path / "data"), "cameras": str(cams),
                                    "product": str(ROOT / "products/safety/safety.yaml")}},
                        redis_client=fakeredis.FakeStrictRedis())
    svc.update_settings({"telegram": {"chat_id": chat or "", "enabled": enabled}})   # Telegram is off by default
    return svc


def _event(feature_tag="fire_smoke", camera="store@fire", severity="warning", message="Fire detected"):
    return {"id": 7, "message": message, "tags": [feature_tag], "camera": camera, "severity": severity,
            "camera_id": camera.split("@")[0], "feature": None}


class _Telegram:
    def __init__(self, fail=0):
        self.sent, self.fail = [], fail

    def __call__(self, method, token, fields, photo=None):
        if self.fail:
            self.fail -= 1
            raise OSError("telegram unreachable")
        self.sent.append((method, token, fields, photo))


def _run(notifier, *notes):
    async def go():
        notifier.start()
        for record, how in notes:
            notifier.notify(record, how)
        await notifier.drain()
        notifier.stop()
    asyncio.run(go())


def test_an_opened_fire_incident_is_sent_with_its_snapshot(tmp_path, monkeypatch):
    tg = _Telegram()
    n = TelegramNotifier(_service(tmp_path, monkeypatch), send=tg, snapshot=lambda sid: b"jpeg", retry_s=0)
    _run(n, (_event(), "opened"))
    ((method, token, fields, photo),) = tg.sent
    assert (method, token, fields["chat_id"], photo) == ("sendPhoto", "123:abc", "-100", b"jpeg")
    assert "Fire detected" in fields["caption"] and "Store room" in fields["caption"]


def test_without_a_snapshot_a_text_message_is_sent(tmp_path, monkeypatch):
    tg = _Telegram()
    _run(TelegramNotifier(_service(tmp_path, monkeypatch), send=tg, snapshot=lambda sid: None, retry_s=0),
         (_event(), "escalated"))
    ((method, _token, fields, _photo),) = tg.sent
    assert method == "sendMessage" and fields["text"].startswith("Escalated")


def test_a_feature_with_notify_off_or_below_its_minimum_is_not_sent(tmp_path, monkeypatch):
    tg = _Telegram()
    n = TelegramNotifier(_service(tmp_path, monkeypatch), send=tg, snapshot=lambda sid: None, retry_s=0)
    _run(n, (_event("zone_intrusion", "office@zones"), "opened"),                  # zones: notify off by default
         (_event("proximity_warning", "bay-a@proximity", severity="warning"), "opened"))   # proximity: critical only
    assert tg.sent == []


def test_with_telegram_off_nothing_is_sent_whatever_the_features_say(tmp_path, monkeypatch):
    from products.safety.service import DEFAULT_FEATURES
    svc = _service(tmp_path, monkeypatch, enabled=False)
    svc.update_settings({"features": {name: {"notify": True, "min_severity": "info"} for name in DEFAULT_FEATURES}})
    tg = _Telegram()
    _run(TelegramNotifier(svc, send=tg, snapshot=lambda sid: None, retry_s=0),
         *(({**_event(severity="critical"), "id": i, "feature": name}, "opened") for i, name in enumerate(DEFAULT_FEATURES)))
    assert tg.sent == []


def test_nothing_is_sent_without_a_token_or_a_chat(tmp_path, monkeypatch):
    for token, chat in (("", "-100"), ("123:abc", "")):
        tg = _Telegram()
        _run(TelegramNotifier(_service(tmp_path / (token or "x"), monkeypatch, token=token, chat=chat), send=tg,
                              snapshot=lambda sid: None, retry_s=0), (_event(), "opened"))
        assert tg.sent == []


def test_a_failed_send_is_retried(tmp_path, monkeypatch):
    tg = _Telegram(fail=2)
    _run(TelegramNotifier(_service(tmp_path, monkeypatch), send=tg, snapshot=lambda sid: None, retry_s=0),
         (_event(), "opened"))
    assert len(tg.sent) == 1


def test_notify_never_blocks_a_full_queue_drops_the_oldest(tmp_path, monkeypatch):
    n = TelegramNotifier(_service(tmp_path, monkeypatch), send=_Telegram(), snapshot=lambda sid: None, maxsize=2)
    for i in range(5):
        n.notify({**_event(), "id": i}, "opened")           # no worker running: nothing drains
    assert [record["id"] for record, _how in n.pending()] == [3, 4]


# ── setting Telegram up: a test message, the chats the bot is in, Telegram's own reason when it refuses ──────────

def _notifier(svc, tg):
    return TelegramNotifier(svc, send=tg, snapshot=lambda sid: None, retry_s=0)


def test_a_test_message_goes_to_the_saved_chat(tmp_path, monkeypatch):
    tg = _Telegram()
    asyncio.run(_notifier(_service(tmp_path, monkeypatch), tg).send_test())
    ((method, token, fields, photo),) = tg.sent
    assert (method, token, fields["chat_id"], photo) == ("sendMessage", "123:abc", "-100", None)
    assert "test" in fields["text"].lower()


def test_a_test_message_works_before_telegram_is_turned_on(tmp_path, monkeypatch):
    tg = _Telegram()
    asyncio.run(_notifier(_service(tmp_path, monkeypatch, enabled=False), tg).send_test())
    assert len(tg.sent) == 1


def test_a_test_message_names_what_is_missing(tmp_path, monkeypatch):
    for token, chat, missing in (("", "-100", "TELEGRAM_BOT_TOKEN"), ("123:abc", "", "chat")):
        n = _notifier(_service(tmp_path / missing, monkeypatch, token=token, chat=chat), _Telegram())
        try:
            asyncio.run(n.send_test())
        except LookupError as exc:
            assert missing in str(exc)
        else:
            raise AssertionError("a test message with nothing to send it with must fail")


def test_a_refused_test_message_is_not_retried_and_says_why(tmp_path, monkeypatch):
    def refuse(method, token, fields, photo=None):
        refuse.calls += 1
        raise OSError("telegram: Bad Request: chat not found")
    refuse.calls = 0
    try:
        asyncio.run(_notifier(_service(tmp_path, monkeypatch), refuse).send_test())
    except OSError as exc:
        assert "chat not found" in str(exc) and refuse.calls == 1
    else:
        raise AssertionError("a refused test message must fail")


def test_the_chats_the_bot_was_added_to_or_written_in(tmp_path, monkeypatch):
    def updates(method, token, fields, photo=None):
        assert method == "getUpdates" and "offset" not in fields          # reading never consumes the updates
        return [
            {"my_chat_member": {"chat": {"id": -1001, "title": "Site safety", "type": "supergroup"}}},
            {"message": {"chat": {"id": 42, "first_name": "Kareem", "type": "private"}, "text": "hi"}},
            {"message": {"chat": {"id": -1001, "title": "Site safety", "type": "supergroup"}, "text": "x"}},
            {"channel_post": {"chat": {"id": -1002, "title": "Alerts", "type": "channel"}}},
        ]
    chats = asyncio.run(_notifier(_service(tmp_path, monkeypatch), updates).recent_chats())
    assert chats == [{"id": "-1001", "title": "Site safety", "type": "supergroup"},
                     {"id": "42", "title": "Kareem", "type": "private"},
                     {"id": "-1002", "title": "Alerts", "type": "channel"}]


def test_telegram_s_reason_is_kept_when_it_refuses(monkeypatch):
    import io
    import urllib.error
    import urllib.request

    from products.safety import notifier as mod

    def urlopen(req, timeout):
        body = io.BytesIO(b'{"ok": false, "error_code": 400, "description": "Bad Request: chat not found"}')
        raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", {}, body)
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    try:
        mod.telegram_send("sendMessage", "123:abc", {"chat_id": "-1", "text": "t"})
    except OSError as exc:
        assert "chat not found" in str(exc) and "123:abc" not in str(exc)
    else:
        raise AssertionError("a refused call must raise")


def test_a_malformed_bot_token_never_reaches_an_error_message():
    """A token pasted with a space or a newline makes http.client refuse the URL with the URL in its message (not an
    OSError); the error raised says what failed, never the token."""
    import pytest
    from products.safety import notifier
    for token in ("123:SECRET-tok en", "123:SECRET-token\n"):
        with pytest.raises(OSError) as err:
            notifier.telegram_send("sendMessage", token, {"chat_id": "1", "text": "x"})
        assert "SECRET" not in str(err.value) and err.value.__cause__ is None and err.value.__suppress_context__


def test_the_bot_token_is_read_without_surrounding_whitespace(monkeypatch):
    from products.safety import notifier
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", " 123:abc\n")
    assert notifier._env_token() == "123:abc"


def test_the_message_names_the_camera_where_the_app_wrote_its_stream_key(tmp_path, monkeypatch):
    tg = _Telegram()
    _run(TelegramNotifier(_service(tmp_path, monkeypatch), send=tg, snapshot=lambda sid: None, retry_s=0),
         (_event(message="Fire detected on store@fire"), "opened"))
    ((_method, _token, fields, _photo),) = tg.sent
    assert fields["text"].endswith("\nFire detected on Store room") and "store@fire" not in fields["text"]


def test_a_sent_message_is_logged_so_an_operator_can_see_it_went(tmp_path, monkeypatch, caplog):
    caplog.set_level("INFO", logger="iris.notify")
    _run(TelegramNotifier(_service(tmp_path, monkeypatch), send=_Telegram(), snapshot=lambda sid: None, retry_s=0),
         (_event(), "opened"))
    assert any("event 7" in r.getMessage() and "sent" in r.getMessage() for r in caplog.records)
