from perception.modules.plates.hold import PlateDiag, PlateHold, one_plate_per_vehicle


def car(x, oid=1):
    return (x, 100.0, 200.0, 100.0, "vehicle", 0.9, oid)


def test_plate_is_held_between_detections_and_moves_with_its_vehicle():
    h = PlateHold(hold_s=1.0)
    out = h.update([(150.0, 170.0, 40.0, 20.0, 0.8, 1)], [car(100.0)], 0.0)
    assert out[0][:4] == (150.0, 170.0, 40.0, 20.0)
    out = h.update([], [car(120.0)], 0.3)           # no detection this frame; vehicle moved 20 px
    assert out[0][:4] == (170.0, 170.0, 40.0, 20.0)


def test_plate_expires_after_hold_and_with_its_vehicle():
    h = PlateHold(hold_s=1.0)
    h.update([(150.0, 170.0, 40.0, 20.0, 0.8, 1)], [car(100.0)], 0.0)
    assert h.update([], [car(100.0)], 1.2) == []
    h.update([(150.0, 170.0, 40.0, 20.0, 0.8, 1)], [car(100.0)], 2.0)
    assert h.update([], [], 2.1) == []


def test_new_detections_are_smoothed():
    h = PlateHold(smooth=0.5)
    h.update([(150.0, 170.0, 40.0, 20.0, 0.8, 1)], [car(100.0)], 0.0)
    out = h.update([(160.0, 170.0, 40.0, 20.0, 0.8, 1)], [car(100.0)], 0.1)
    assert out[0][0] == 155.0


def test_two_plates_on_one_vehicle_keep_the_one_nearest_its_bottom_centre():
    v = car(100.0)                                  # box 100..300 x 100..200
    low_centre = (170.0, 170.0, 60.0, 20.0, 0.5, 1)
    high_side = (105.0, 105.0, 40.0, 15.0, 0.9, 1)  # surer, but near a top corner
    assert one_plate_per_vehicle([high_side, low_centre], [v]) == [low_centre]


def test_each_vehicle_keeps_its_own_plate():
    a, b = car(0.0, 1), car(400.0, 2)
    pa, pb = (70.0, 170.0, 60.0, 20.0, 0.5, 1), (470.0, 170.0, 60.0, 20.0, 0.5, 2)
    assert sorted(one_plate_per_vehicle([pa, pb], [a, b])) == [pa, pb]


def test_a_plate_that_is_not_on_its_vehicle_is_dropped():
    v = car(100.0)
    assert one_plate_per_vehicle([(320.0, 170.0, 60.0, 20.0, 0.9, 1)], [v]) == []   # beside it
    assert one_plate_per_vehicle([(150.0, 150.0, 190.0, 40.0, 0.9, 1)], [v]) == []  # as big as the car
    assert one_plate_per_vehicle([(150.0, 170.0, 60.0, 20.0, 0.9, 5)], [v]) == []   # unknown parent


def test_the_plate_already_shown_is_kept_over_a_second_candidate():
    v = car(100.0)                                   # box 100..300 x 100..200
    shown = (170.0, 150.0, 40.0, 20.0, 0.7, 1)       # centre (190, 160) -> relative (0.45, 0.6)
    other = (170.0, 175.0, 40.0, 20.0, 0.9, 1)       # nearer the bottom-centre
    assert one_plate_per_vehicle([shown, other], [v]) == [other]
    assert one_plate_per_vehicle([shown, other], [v], remembered={1: (0.45, 0.6)}) == [shown]


def test_held_plate_centres_are_reported_relative_to_the_vehicle():
    h = PlateHold()
    h.update([(150.0, 170.0, 40.0, 20.0, 0.8, 1)], [car(100.0)], 0.0)
    cx, cy = h.centres()[1]
    assert abs(cx - 0.35) < 1e-9 and abs(cy - 0.8) < 1e-9


def test_a_plate_too_small_or_too_oddly_shaped_to_read_is_dropped():
    from perception.modules.plates.hold import plate_size_ok
    assert plate_size_ok(6, 6, (28, 12), (1.0, 6.0)) == "too small to read"      # the 6 x 6 box of truck 5356
    assert plate_size_ok(40, 20, (28, 12), (1.0, 6.0)) == ""
    assert plate_size_ok(30, 60, (28, 12), (1.0, 6.0)) == "odd shape for a plate"  # taller than wide
    assert plate_size_ok(120, 10, (28, 8), (1.0, 6.0)) == "odd shape for a plate"  # a sliver
    assert plate_size_ok(6, 6) == ""                                               # no limits given: unchanged


def test_a_tiny_plate_does_not_win_over_a_real_one_by_sitting_lower():
    tiny = (197.0, 194.0, 6.0, 6.0, 0.9, 1)            # right at the bottom-centre of the vehicle box (100,100 200x100)
    real = (150.0, 160.0, 40.0, 20.0, 0.8, 1)
    dropped = []
    out = one_plate_per_vehicle([tiny, real], [car(100.0)], dropped=dropped, min_size=(28, 12), aspect=(1.0, 6.0))
    assert out == [real]
    assert [d[2] for d in dropped] == ["too small to read"]
    # without the limits the tiny one wins, which is the behaviour this guards against
    assert one_plate_per_vehicle([tiny, real], [car(100.0)]) == [tiny]


def test_plate_diag_says_per_vehicle_whether_a_plate_was_found_kept_or_dropped():
    lines = []
    d = PlateDiag(quiet_s=2.0, emit=lines.append)
    box = (100.0, 100.0, 200.0, 100.0, "vehicle", 0.9, 7)
    other = (400.0, 100.0, 200.0, 100.0, "vehicle", 0.9, 8)
    small = (150.0, 170.0, 20.0, 9.0, 0.8, 7)
    for i in range(5):                                  # vehicle 7: found in 2 frames, both dropped; vehicle 8: never
        t = i * 0.3
        raw = [(7, 0.8)] if i in (1, 3) else []
        lost = [(small, box, "too small to read")] if i in (1, 3) else []
        d.frame("cam-01", t, {7, 8}, raw, [], lost)
    assert lines == []                                  # still in view
    d.frame("cam-01", 10.0, set(), [], [], [])          # long gone: both summarised
    by = {l.split("#")[1].split(":")[0]: l for l in lines}
    assert "in 5 analysed frames" in by["7"] and "detected in 2 (best conf 0.80)" in by["7"]
    assert "kept in 0" in by["7"] and "too small to read x2" in by["7"]
    assert "detected in 0, kept in 0, dropped: none" in by["8"]


def test_a_plate_without_a_parent_is_counted_on_the_vehicle_that_holds_it():
    from perception.modules.plates.hold import resolve_parent
    big, small = car(100.0, 1), (150.0, 150.0, 80.0, 60.0, "vehicle", 0.9, 2)
    assert resolve_parent(1, (0, 0, 5, 5), [big, small]) == 1          # a real parent wins
    assert resolve_parent(-1, (170.0, 170.0, 20.0, 10.0), [big, small]) == 2    # the smallest box holding it
    assert resolve_parent(-1, (900.0, 900.0, 20.0, 10.0), [big, small]) == -1


def test_a_plate_cut_by_the_frame_edge_is_dropped_but_a_whole_one_is_kept():
    veh = [(100.0, 300.0, 600.0, 420.0, "vehicle", 0.9, 1)]            # a vehicle driving out of a 1280x720 frame
    half = (400.0, 709.0, 60.0, 11.0, 0.83, 1)                           # its plate ends on the frame's last row
    whole = (400.0, 640.0, 60.0, 25.0, 0.80, 1)
    lost = []
    assert one_plate_per_vehicle([half], veh, dropped=lost, frame=(1280, 720)) == []
    assert lost[0][2] == "cut by the frame edge"
    assert one_plate_per_vehicle([half, whole], veh, frame=(1280, 720)) == [whole]
    assert one_plate_per_vehicle([half], veh) == [half]                  # no frame given: as before
