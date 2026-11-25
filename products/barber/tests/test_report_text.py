from report_text import TEXT, clock, day_name, label, number, percent, timer


def test_both_languages_have_every_key():
    assert set(TEXT["en"]) == set(TEXT["ar"])
    assert label("ar", "ch.noData") == "لا توجد بيانات في هذه الفترة"     # the dashboard's wording


def test_formatting_matches_the_dashboard():
    assert timer(None) == "—" and timer(75) == "01:15" and timer(3725) == "1:02:05"
    assert clock("2026-10-11T20:05:31") == "20:05" and clock(None) == "—"
    assert number(1234) == "1,234" and percent({"count": 3, "percent": 12.5}) == "3 (12.5 %)"
    assert percent({"count": 0, "percent": None}) == "0"


def test_dates_keep_latin_digits_in_arabic():
    assert day_name("en", "2026-10-11") == "Sun 11 Oct"
    ar = day_name("ar", "2026-10-11")
    assert "11" in ar and "أكتوبر" in ar
