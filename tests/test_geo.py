import math

import numpy as np
import pyproj
import pytest

from brmap.geo import Extent

CAPITOL = (-91.1868, 30.4560)
TIGER_STADIUM = (-91.1855, 30.4115)
I10_BRIDGE_EAST = (-91.1935, 30.4395)


def test_center_maps_to_origin(extent):
    x, y = extent.to_level(-91.160, 30.430)
    assert abs(x) < 0.01 and abs(y) < 0.01


def test_round_trip(extent):
    lon, lat = extent.to_lonlat(*extent.to_level(-91.1874, 30.4570))
    assert abs(lon + 91.1874) < 1e-7 and abs(lat - 30.4570) < 1e-7


def test_capitol_north_of_tiger_stadium(extent):
    _, y_cap = extent.to_level(*CAPITOL)
    _, y_tig = extent.to_level(*TIGER_STADIUM)
    assert y_cap > y_tig  # +y is north


def test_west_is_negative_x(extent):
    x, _ = extent.to_level(-91.20, 30.430)
    assert x < 0  # +x is east


def test_level_distance_matches_ground(extent):
    a = extent.to_level(*CAPITOL)
    b = extent.to_level(*TIGER_STADIUM)
    ground = pyproj.Geod(ellps="GRS80").inv(*CAPITOL, *TIGER_STADIUM)[2]
    assert abs(math.dist(a, b) - ground) / ground < 0.001  # 1:1 within 0.1 %


def test_spawn_points_inside_box(extent):
    for lon, lat in [TIGER_STADIUM, CAPITOL, I10_BRIDGE_EAST]:
        assert extent.contains(*extent.to_level(lon, lat))


def test_contains_rejects_outside(extent):
    assert not extent.contains(8192.5, 0.0)
    assert not extent.contains(0.0, -9000.0)


def test_to_level_accepts_arrays(extent):
    xs, ys = extent.to_level(np.array([-91.16, -91.17]), np.array([30.43, 30.44]))
    assert xs.shape == (2,) and abs(xs[0]) < 0.01 and ys[1] > ys[0]


def test_pixel_size_default(extent):
    assert extent.pixel_size(4096) == 4.0


def test_bbox_utm(extent):
    w, s, e, n = extent.bbox_utm(500)
    assert e - w == pytest.approx(2 * (8192 + 500))
    assert (w + e) / 2 == pytest.approx(extent.center_e)
    assert (s + n) / 2 == pytest.approx(extent.center_n)


def test_bbox_lonlat_covers_box(extent):
    w, s, e, n = extent.bbox_lonlat(500)
    assert w < -91.243 and e > -91.077 and s < 30.358 and n > 30.502
    for x, y in [(-8692, -8692), (8692, 8692), (-8692, 8692), (8692, -8692)]:
        lon, lat = extent.to_lonlat(x, y)
        assert w <= lon <= e and s <= lat <= n


def test_transform_origin_is_north_west(extent):
    t = extent.transform(4096)
    assert t.a == 4.0 and t.e == -4.0
    assert t.c == pytest.approx(extent.center_e - 8192)
    assert t.f == pytest.approx(extent.center_n + 8192)


def test_level_transform(extent):
    t = extent.level_transform(4096)
    assert (t.a, t.e, t.c, t.f) == (4.0, -4.0, -8192.0, 8192.0)
    x, y = t @ (0.5, 0.5)  # center of the north-west pixel
    assert (x, y) == (-8190.0, 8190.0)


def test_pixel_centers_and_to_pixel(extent):
    xs, ys = extent.pixel_centers(4096)
    assert xs[0] == pytest.approx(-8190.0) and ys[0] == pytest.approx(8190.0)
    assert xs[-1] == pytest.approx(8190.0) and ys[-1] == pytest.approx(-8190.0)
    col, row = extent.to_pixel(xs[10], ys[20], 4096)
    assert col == pytest.approx(10.0) and row == pytest.approx(20.0)


def test_from_config_uses_full_size_by_default():
    from brmap.config import Config

    ext = Extent.from_config(Config())
    assert ext.half_size == 8192.0 and ext.center_lat == 30.430
