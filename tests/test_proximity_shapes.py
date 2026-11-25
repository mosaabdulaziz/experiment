"""proximity.shapes: an instance-mask grid -> foot, head, lower outline, reference point; the rider test."""
import numpy as np
import pytest

from perception.engine.nvds import InstanceMask
from perception.modules.proximity.shapes import (
    Blob, blob_of, gap_to_outline, is_rider, outline_polygon, person_shape, person_shape_from_box, vehicle_shape,
    vehicle_shape_from_box)


def _mask(grid, threshold=0.5):
    g = np.asarray(grid, np.float32)
    return InstanceMask(g.reshape(-1), g.shape[1], g.shape[0], threshold)


def test_a_box_sized_grid_maps_cells_to_mux_pixels():
    grid = np.zeros((10, 10))
    grid[2:9, 4:6] = 1                                     # a thin upright person
    shape = person_shape(blob_of(_mask(grid), (100, 200, 10, 10)), frame_h=720)
    assert shape.foot == pytest.approx((105.0, 208.5))     # lowest row 8, columns 4-5
    assert shape.head == pytest.approx((105.0, 202.5))
    assert not shape.touches_bottom


def test_a_large_grid_over_a_small_box_is_thinned_to_about_a_cell_per_pixel():
    grid = np.zeros((184, 320))                            # the parser's grid for a 736x1280 input
    grid[40:180, 100:220] = 1
    blob = blob_of(_mask(grid), (0, 0, 32, 18))            # a far person: 32 x 18 px
    assert blob.grid.shape == (19, 32)
    assert (blob.cell_w, blob.cell_h) == pytest.approx((1.0, 18 * 10 / 184))


def test_only_the_largest_blob_counts():
    grid = np.zeros((20, 10))
    grid[0:12, 3:7] = 1                                    # the body
    grid[17:19, 0:2] = 1                                   # a stray fragment lower down
    shape = person_shape(blob_of(_mask(grid), (0, 0, 10, 20)), frame_h=720)
    assert shape.foot[1] == pytest.approx(11.5)


def test_no_mask_or_an_unusable_one_gives_none():
    assert blob_of(None, (0, 0, 10, 10)) is None
    assert blob_of(_mask(np.zeros((8, 8))), (0, 0, 10, 10)) is None                     # empty
    assert blob_of(_mask(np.ones((8, 8))), (0, 0, 0.5, 10)) is None                     # degenerate box
    assert blob_of(InstanceMask(np.ones(5, np.float32), 4, 8, 0.5), (0, 0, 10, 10)) is None   # wrong length


def test_a_mask_reaching_the_frame_bottom_is_cut_off():
    shape = person_shape(blob_of(_mask(np.ones((10, 4))), (0, 710, 4, 10)), frame_h=720)
    assert shape.touches_bottom
    assert person_shape_from_box((0, 600, 10, 119), 720).touches_bottom
    assert not person_shape_from_box((0, 600, 10, 100), 720).touches_bottom


def test_a_vehicle_outline_is_the_lowest_mask_point_of_each_column():
    grid = np.zeros((10, 6))
    grid[2:8, :] = 1                                       # the body
    grid[8:10, 1] = 1                                      # two wheels
    grid[8:10, 4] = 1
    v = vehicle_shape(blob_of(_mask(grid), (0, 0, 6, 10)))
    assert v.outline[:, 0].tolist() == pytest.approx([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    assert v.outline[:, 1].tolist() == pytest.approx([7.5, 9.5, 7.5, 7.5, 9.5, 7.5])
    assert v.ref == pytest.approx((3.0, 9.5))              # median of the lowest third: the wheels
    assert v.area == pytest.approx(40.0)


def test_a_wide_vehicle_keeps_at_most_64_outline_points():
    v = vehicle_shape(blob_of(_mask(np.ones((20, 300))), (0, 0, 300, 20)))
    assert len(v.outline) <= 64


def test_without_a_mask_the_box_bottom_is_the_outline():
    v = vehicle_shape_from_box((100, 50, 200, 100))
    assert v.ref == (200.0, 150.0) and set(v.outline[:, 1]) == {150.0} and v.blob is None


def test_the_gap_is_to_the_nearest_outline_point():
    d, near = gap_to_outline((10.0, 130.0), np.array([[0.0, 100.0], [10.0, 100.0], [20.0, 90.0]]))
    assert d == pytest.approx(30.0) and near == (10.0, 100.0)


def _flatbed():
    """A 200 x 150 truck at (100, 50) whose mask is only its lowest 50 rows (bed and wheels)."""
    grid = np.zeros((150, 200))
    grid[100:150, :] = 1
    box = (100, 50, 200, 150)
    return box, vehicle_shape(blob_of(_mask(grid), box))


def test_a_driver_inside_the_vehicle_is_a_rider_and_a_person_beside_it_is_not():
    box = (100, 100, 200, 100)
    truck = vehicle_shape(blob_of(_mask(np.ones((100, 200))), box))
    assert is_rider((150, 110, 30, 60), (165, 169), box, truck, ph=100, feet_hidden=False)
    assert not is_rider((310, 120, 30, 80), (325, 199), box, truck, ph=100, feet_hidden=False)


def test_a_person_standing_on_a_flatbed_is_a_rider_but_one_hidden_behind_it_is_not():
    box, truck = _flatbed()
    person, foot = (150, 60, 30, 80), (165, 139.5)          # feet above the bed's lower outline (y ~ 199.5)
    assert is_rider(person, foot, box, truck, ph=80, feet_hidden=False)
    assert not is_rider(person, foot, box, truck, ph=80, feet_hidden=True)


def _cab(hole=True):
    """A 200 x 100 px vehicle mask over box (100, 100, 200, 100); with `hole`, a seated driver's pixels (x 150-180,
    y 120-190) are cut out of it: an instance mask gives each pixel to one object."""
    grid = np.ones((100, 200))
    if hole:
        grid[20:90, 50:80] = 0
    box = (100, 100, 200, 100)
    return box, vehicle_shape(blob_of(_mask(grid), box))


def test_a_driver_whose_pixels_cut_a_hole_in_the_vehicles_mask_is_on_it():
    box, cab = _cab()
    # feet hidden in the cab: his lowest visible point lies in the hole his own pixels cut, inside the vehicle
    assert is_rider((150, 120, 30, 70), (165, 189), box, cab, ph=100, feet_hidden=True)


def test_a_pedestrian_with_feet_in_view_is_not_on_a_vehicle_by_the_hole_they_cut_in_its_mask():
    box, cab = _cab()
    # standing in front of it: the same hole, but the feet are in view and level with its lower edge
    assert not is_rider((150, 120, 30, 70), (165, 189), box, cab, ph=100, feet_hidden=False)


def test_a_blob_outline_is_its_contour_through_the_cell_centres_in_mux_pixels():
    grid = np.zeros((10, 8), bool)
    grid[2:8, 1:7] = True
    poly = outline_polygon(Blob(grid, left=100.0, top=200.0, cell_w=4.0, cell_h=5.0))
    xs, ys = zip(*poly)
    assert len(poly) == 4 and (min(xs), max(xs), min(ys), max(ys)) == (106.0, 126.0, 212.5, 237.5)


def test_a_blob_too_small_to_outline_has_none():
    grid = np.zeros((3, 3), bool)
    grid[1, 1] = True
    assert outline_polygon(Blob(grid, 0.0, 0.0, 4.0, 4.0)) == []
