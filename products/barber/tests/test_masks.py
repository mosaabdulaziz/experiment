import numpy as np

from masks import GRID, Raster, raster

SQ = [[0.1, 0.1], [0.2, 0.1], [0.2, 0.2], [0.1, 0.2]]


def test_square_area_is_exact_on_the_grid():
    r = raster(SQ)
    assert (r.x0, r.y0) == (96, 54)
    assert r.area == 96 * 54


def test_flat_points_are_accepted():
    flat = [v for p in SQ for v in p]
    assert raster(flat).area == raster(SQ).area


def test_triangle_is_about_half_its_box():
    r = raster([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]])
    box = 480 * 270
    assert abs(r.area - box / 2) / box < 0.01


def test_degenerate_polygons_are_none():
    assert raster(None) is None
    assert raster([[0.1, 0.1], [0.2, 0.2]]) is None
    assert raster([[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]]) is None or raster([[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]]).area == 0


def test_lower_fraction_keeps_the_bottom_rows():
    low = raster(SQ).lower(0.35)
    assert low.area == 96 * 19
    rows = np.flatnonzero(low.mask.any(axis=1))
    assert rows[-1] == 53 and rows[0] == 53 - 18


def test_lower_one_is_the_whole_mask():
    r = raster(SQ)
    assert r.lower(1.0).area == r.area


def test_overlap_of_shifted_squares():
    a = raster(SQ)
    b = raster([[0.15, 0.1], [0.25, 0.1], [0.25, 0.2], [0.15, 0.2]])
    assert a.overlap(b) == 48 * 54
    assert a.overlap(raster([[0.5, 0.5], [0.6, 0.5], [0.6, 0.6]])) == 0


def test_points_outside_the_frame_are_clipped():
    r = raster([[-0.5, -0.5], [0.1, -0.5], [0.1, 0.1], [-0.5, 0.1]])
    assert (r.x0, r.y0) == (0, 0) and r.area == 96 * 54
    assert GRID == (960, 540)
