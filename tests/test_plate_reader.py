import numpy as np

from perception.modules.mtmc.plate_reader import PlateReader


def _reader():
    r = PlateReader.__new__(PlateReader)          # decoding only: no model needed
    r._chars = ["blank", "0", "1", "2", "ا"]
    return r


def _probs(seq):
    p = np.full((len(seq), 5), 0.01)
    for t, i in enumerate(seq):
        p[t, i] = 0.96
    return p


def test_ctc_read_collapses_repeats_and_drops_blanks():
    text, conf = _reader().decode(_probs([0, 2, 2, 0, 2, 3, 3, 0, 4]))
    assert text == "11 2ا".replace(" ", "")
    assert abs(conf - 0.96) < 1e-6


def test_nothing_read_has_zero_confidence():
    assert _reader().decode(_probs([0, 0, 0])) == ("", 0.0)


def test_an_earlier_runs_plates_are_cleared_when_a_new_run_starts(tmp_path):
    from perception.modules.mtmc.plate_reader import PlateBook
    for name in ("0.json", "0.png", "0_cam-01.jpg", "12.jpg", "notes.txt"):
        (tmp_path / name).write_text("x")
    book = PlateBook.__new__(PlateBook)
    book._dir = str(tmp_path)
    book._clear_old()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["notes.txt"]


def test_the_canvas_is_padded_with_zero_after_normalising():
    """Black padding made the real model drop every letter (checked on real plate crops)."""
    import pytest
    ort = pytest.importorskip("onnxruntime")
    seen = {}

    class Sess:
        def run(self, _out, feed):
            seen["x"] = feed["x"]
            return [np.zeros((1, 40, 28), np.float32)]

    r = PlateReader.__new__(PlateReader)
    r._enhance = False
    r._sess, r._chars = Sess(), ["blank"] + list("0123456789ابسدعفجهلمنقرصطوي")
    r.read(np.full((20, 40, 3), 255, np.uint8))
    x = seen["x"]
    assert x.shape == (1, 3, 48, 320)
    assert float(x[0, 0, 0, 0]) == 1.0 and float(x[0, 0, 0, 319]) == 0.0      # plate: white; padding: 0

def test_enhancement_enlarges_three_times_and_keeps_a_blurry_crop_untouched_otherwise():
    from perception.modules.mtmc.plate_reader import enhance
    flat = np.full((20, 40, 3), 120, np.uint8)            # no detail: metrics 0 -> upscale only
    out = enhance(flat)
    assert out.shape == (60, 120, 3) and abs(int(out.mean()) - 120) <= 1


def test_a_sharp_crop_gets_the_gentle_contrast_pass():
    from perception.modules.mtmc.plate_reader import enhance
    rng = np.random.default_rng(1)
    sharp = (rng.integers(0, 2, (20, 40, 1)) * 255).astype(np.uint8).repeat(3, axis=2)
    plain = __import__("cv2").resize(sharp, None, fx=3, fy=3, interpolation=__import__("cv2").INTER_CUBIC)
    assert enhance(sharp).shape == plain.shape and not np.array_equal(enhance(sharp), plain)


def test_a_crop_below_the_minimum_size_is_never_queued_for_reading():
    import numpy as np
    from perception.modules.mtmc.plate_reader import PlateBook
    book = PlateBook.__new__(PlateBook)
    book._min_size, book.too_small = (28, 12), 0
    import queue
    book._q = queue.Queue()
    assert book.submit(1, "cam-01", np.zeros((6, 6, 3), np.uint8), 0.9, 3) is False     # the 6 x 6 crop
    assert book.too_small == 1 and book._q.qsize() == 0
    assert book.submit(1, "cam-01", np.zeros((20, 40, 3), np.uint8), 0.9, 3) is True
    assert book._q.qsize() == 1


def test_a_restart_with_the_files_kept_keeps_what_was_read_before(tmp_path, monkeypatch):
    import json
    import cv2
    import numpy as np
    from perception.modules.mtmc.plate_reader import PlateBook
    monkeypatch.setenv("SCALE_KEEP_RUN_FILES", "1")
    (tmp_path / "7_reads.json").write_text(json.dumps([{"cam": "cam-01", "text": "1234", "ocr_conf": 0.9}]))
    (tmp_path / "7.json").write_text(json.dumps([{"cam": "cam-01", "local_id": 3, "text": "1234", "ocr_conf": 0.9,
                                                    "det_conf": 0.8, "at": 1.0}]))
    cv2.imwrite(str(tmp_path / "7_cam-01.jpg"), np.zeros((20, 40, 3), np.uint8))
    book = PlateBook.__new__(PlateBook)
    book._dir, book._reads, book._rows = str(tmp_path), {}, {}
    book._restore(7)
    assert book._reads[7][0]["text"] == "1234" and book._rows[7]["cam-01"]["crop"].shape[:2] == (20, 40)
    book._clear_old()                                              # and nothing is deleted at start
    assert (tmp_path / "7.json").exists() and (tmp_path / "7_cam-01.jpg").exists()
    monkeypatch.delenv("SCALE_KEEP_RUN_FILES")
    book._clear_old()                                              # a fresh run still clears the old ids
    assert not (tmp_path / "7.json").exists()
