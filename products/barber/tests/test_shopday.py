from datetime import date, datetime

from shopday import TicketCounter, day_start, shop_day, shop_day_range


def test_before_the_start_hour_belongs_to_the_previous_day():
    assert shop_day(datetime(2026, 9, 26, 1, 30), 5) == date(2026, 9, 25)
    assert shop_day(datetime(2026, 9, 26, 5, 0), 5) == date(2026, 9, 26)


def test_hour_zero_is_midnight():
    assert shop_day(datetime(2026, 9, 26, 0, 0), 0) == date(2026, 9, 26)


def test_range_spans_one_shop_day():
    assert shop_day_range(datetime(2026, 9, 26, 1, 30), 5) == (datetime(2026, 9, 25, 5), datetime(2026, 9, 26, 5))
    assert day_start(date(2026, 9, 25), 5) == datetime(2026, 9, 25, 5)


def test_tickets_count_up_within_a_shop_day_and_restart_the_next():
    t = TicketCounter(hour=5)
    assert [t.next(datetime(2026, 9, 25, 10)), t.next(datetime(2026, 9, 26, 1))] == [1, 2]   # 01:00 is still the 25th
    assert t.next(datetime(2026, 9, 26, 6)) == 1


def test_tickets_continue_from_a_seed():
    t = TicketCounter(hour=5, seed={date(2026, 9, 25): 7})
    assert t.next(datetime(2026, 9, 25, 12)) == 8


def test_resume_raises_a_days_count_to_at_least_n():
    t = TicketCounter(hour=5)
    t.next(datetime(2026, 9, 25, 10))                        # 1
    t.resume(date(2026, 9, 25), 5)
    assert t.next(datetime(2026, 9, 25, 11)) == 6


def test_resume_never_lowers_the_count():
    t = TicketCounter(hour=5, seed={date(2026, 9, 25): 7})
    t.resume(date(2026, 9, 25), 3)
    assert t.next(datetime(2026, 9, 25, 12)) == 8
