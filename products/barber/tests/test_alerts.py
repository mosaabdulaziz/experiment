import itertools
from datetime import datetime, timedelta

from alerts import EXCESSIVE_WAIT, AlertBook

T0 = datetime(2026, 9, 13, 10, 0, 0)


def t(s: float) -> datetime:
    return T0 + timedelta(seconds=s)


def book():
    ids = itertools.count(1)
    return AlertBook(lambda: f"a{next(ids)}")


def check(b, now, target=100.0):
    return b.check(EXCESSIVE_WAIT, "v1", camera="cam-1", visit_id="v1", session_id=None,
                   since=t(0), target_s=target, now=now)


def test_opens_once_when_target_crossed_dated_at_the_crossing():
    b = book()
    assert check(b, t(100)) is None
    alert = check(b, t(101))
    assert alert.started_at == t(100) and alert.open and alert.kind == EXCESSIVE_WAIT
    assert check(b, t(150)) is None  # already open
    assert b.open_alerts() == [alert]


def test_close_clamps_to_alert_start_and_allows_reopening():
    b = book()
    alert = check(b, t(101))
    closed = b.close(EXCESSIVE_WAIT, "v1", t(90))  # backdated end before the crossing
    assert closed is alert and closed.ended_at == t(100) and not closed.open
    assert b.close(EXCESSIVE_WAIT, "v1", t(200)) is None
    assert check(b, t(300)).id == "a2"


def test_no_target_never_alerts():
    assert check(book(), t(10_000), target=None) is None
