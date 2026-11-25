from app import plate_reads
from app.truck_feed import TruckFeedBridge


def _rows(*r):
    return [{"cam": c, "text": t, "ocr_conf": q} for c, t, q in r]


def test_best_read_is_the_surest_one_with_a_usable_number():
    rows = _rows(("cam-01", "431", 0.72), ("cam-02", "5978", 0.96), ("cam-03", "12", 0.99), ("cam-04", "7192", 0.4))
    assert plate_reads.best_read(rows)["text"] == "5978"       # 12 has too few digits, 7192 is unsure
    assert plate_reads.best_read(_rows(("cam-01", "ب", 0.9))) is None


def test_the_feed_sends_the_reading_with_the_identity_plate(tmp_path):
    (tmp_path / "7_reads.json").write_text('[{"cam": "cam-02", "text": "5978", "ocr_conf": 0.96}]', encoding="utf-8")
    bridge = TruckFeedBridge({"plates_dir": str(tmp_path)}, "redis://x")
    entry = bridge._plate_entry(7, "2026-09-13T11:50:00Z")
    assert entry["plate"].endswith("0007") and entry["read_plate"] == "5978" and entry["confidence"] == 0.96
    assert "read_plate" not in bridge._plate_entry(8, "2026-09-13T11:50:00Z")     # nothing read yet


def test_letters_are_put_in_reading_order():
    assert plate_reads.logical("5356روس") == "س و ر 5356"
    assert plate_reads.logical("431") == "431"


def test_a_later_shorter_reading_does_not_replace_the_fuller_one():
    reads = _rows(("cam-02", "5356رس", 0.86), ("cam-02", "5356رس", 0.85),
                  ("cam-02", "5356روس", 0.80), ("cam-02", "5356وس", 0.84))
    assert plate_reads.consensus(reads)["text"] == "5356روس"


def test_a_one_off_invented_letter_is_not_added():
    reads = _rows(("cam-02", "5356روس", 0.8), ("cam-03", "5356رودس", 0.7), ("cam-02", "5356روس", 0.85))
    assert plate_reads.consensus(reads)["text"] == "5356روس"


def test_the_digits_with_the_most_confidence_behind_them_win():
    reads = _rows(("cam-01", "1421", 0.54), ("cam-02", "2593يفد", 0.82), ("cam-03", "2593", 0.7), ("cam-09", "1", 0.3))
    assert plate_reads.consensus(reads)["text"] == "2593يفد"


def test_a_reading_that_improves_after_the_last_sighting_is_sent_again(tmp_path):
    import json
    bridge = TruckFeedBridge({"plates_dir": str(tmp_path), "camera_ips": {"cam-01": "10.0.0.1"}}, "redis://x")
    bridge._sent[("cam-01", 1)] = 1000.0                       # the sighting went out; nothing new is due
    (tmp_path / "1_reads.json").write_text(json.dumps([{"cam": "cam-01", "text": "1234ط", "ocr_conf": 0.55}]))
    first = bridge._reading_updates(set())
    assert [e["plates"][0]["read_plate"] for e in first] == ["ط 1234"]
    bridge._told.update(bridge._told_marks)
    assert bridge._reading_updates(set()) == []                # told once, nothing changed
    (tmp_path / "1_reads.json").write_text(json.dumps(
        [{"cam": "cam-01", "text": "1234ط", "ocr_conf": 0.55}, {"cam": "cam-01", "text": "1421طدو", "ocr_conf": 0.6}]))
    import os; os.utime(tmp_path / "1_reads.json", (2e9, 2e9))
    later = bridge._reading_updates(set())
    assert [e["plates"][0]["read_plate"] for e in later] == ["و د ط 1421"]
    assert later[0]["ip"] == "10.0.0.1"


def test_a_doubled_letter_stays_doubled():
    reads = _rows(("cam-01", "1421ننط", 0.6), ("cam-01", "1421نط", 0.55), ("cam-01", "1421ننط", 0.62))
    assert plate_reads.consensus(reads)["text"] == "1421ننط"
    assert plate_reads.logical("1421ننط") == "ط ن ن 1421"


def test_a_fairly_unsure_reading_is_still_used_and_found_after_a_restart(tmp_path):
    import json
    assert plate_reads.consensus(_rows(("cam-01", "1461دو", 0.497)))["text"] == "1461دو"
    assert plate_reads.consensus(_rows(("cam-01", "1461دو", 0.3))) is None
    (tmp_path / "1_reads.json").write_text(json.dumps([{"cam": "cam-01", "text": "1461دو", "ocr_conf": 0.497}]))
    (tmp_path / "1.json").write_text(json.dumps([{"cam": "cam-01", "text": "1461دو", "ocr_conf": 0.497, "at": 1790000000.0}]))
    bridge = TruckFeedBridge({"plates_dir": str(tmp_path), "camera_ips": {"cam-01": "10.0.0.1"}}, "redis://x")
    assert bridge._sent == {}                                  # as after a restart of the app
    ups = bridge._reading_updates(set())
    assert [e["plates"][0]["read_plate"] for e in ups] == ["و د 1461"] and ups[0]["ip"] == "10.0.0.1"


def test_consensus_keeps_a_letter_that_no_single_read_has_with_the_others():
    # six real reads of one plate: no read holds all three voted letters (ر و س), the order comes from all of them
    reads = [{"text": t, "ocr_conf": c} for t, c in [
        ("5356وس", 0.758), ("5356رس", 0.875), ("5356وس", 0.815),
        ("5356وس", 0.851), ("5356روص", 0.736), ("5356ردس", 0.835)]]
    assert plate_reads.consensus(reads)["text"] == "5356روس"


def test_a_cameras_shown_reading_is_its_surest_not_its_latest(tmp_path):
    import time
    import numpy as np
    from perception.modules.mtmc.plate_reader import PlateBook

    class Reader:
        def __init__(self):
            self.out = iter([("5356روس", 0.88), ("5356ردس", 0.79)])

        def read(self, crop):
            return next(self.out)

    book = PlateBook(Reader(), str(tmp_path))
    crop = np.zeros((20, 60, 3), np.uint8)
    book.submit(3, "cam-02", crop, 0.8, 1)
    book.submit(3, "cam-02", crop, 0.8, 1)
    for _ in range(100):
        if book._q.empty() and (tmp_path / "3_reads.json").exists() and len(book._reads.get(3, [])) == 2:
            break
        time.sleep(0.05)
    assert book._rows[3]["cam-02"]["text"] == "5356روس"
    assert len(book._reads[3]) == 2                      # both readings still count in the vote


def test_two_readings_that_differ_in_one_letter_are_one_two_letter_plate_not_a_three_letter_one():
    # truck 5356 on a later run: three readings, all two letters; the old vote showed س و ر then س و
    reads = _rows(("cam-02", "5356رس", 0.876), ("cam-02", "5356وس", 0.885), ("cam-02", "5356وس", 0.708))
    assert plate_reads.consensus(reads)["text"] == "5356وس"
    shown = [plate_reads.consensus(reads[:i])["text"] for i in (1, 2, 3)]
    assert all(len(t) == len("5356") + 2 for t in shown)            # never a third letter from two disagreeing reads


def test_a_dropped_letter_does_not_shift_the_others_out_of_their_slots():
    # one three-letter reading is enough to say the plate has three letters; the two-letter ones then fill the
    # slots they agree with ("وس" is slots 2 and 3), so the middle letter is not outvoted by a misread one
    reads = _rows(("cam-02", "5356ردس", 0.835), ("cam-02", "5356روص", 0.736), ("cam-02", "5356وس", 0.851),
                  ("cam-02", "5356وس", 0.815), ("cam-02", "5356وس", 0.758))
    assert plate_reads.consensus(reads)["text"] == "5356روس"
