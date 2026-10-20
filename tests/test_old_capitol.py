"""The Old State Capitol's model script (blender/landmarks/old_capitol.py): its plain-Python geometry on
the real params, built without Blender."""
import math

import pytest
from shapely.geometry import Point, Polygon

from landmark_scripts import check_parts, fixture_params, load_script

import brlm  # noqa: E402  (landmark_scripts puts blender/lib on the path)

PARAMS = fixture_params("old_capitol")   # landmarks.resolve on the full map (data/out), 2026-10-19
LODS = (400, 100, 25)


@pytest.fixture(scope="module")
def old():
    return load_script("old_capitol")


@pytest.fixture(scope="module")
def dims(old):
    return old.resolve(PARAMS)


@pytest.fixture(scope="module")
def built(old, dims):
    return old.parts(dims)


def corners_of(b):
    return [p for pts, _, _ in b.faces for p in pts]


def test_parts_and_levels(built):
    assert set(built) == {"main"}
    check_parts(built, LODS)


def test_heights(dims, built):
    # the lantern over the rotunda is the top (towers_m, the lidar's highest return); the towers' and the
    # entrance block's battlements reach height_m
    g = PARAMS["ground_dz"]
    for px, _, b in built["main"]["lods"]:
        assert max(p[2] for p in corners_of(b)) == pytest.approx(g + PARAMS["towers_m"], abs=0.01), px
    high = built["main"]["lods"][0][2]
    for cx, cy, r, top in dims["towers"]:
        near = [p for p in corners_of(high) if math.hypot(p[0] - cx, p[1] - cy) <= r + 0.01]
        assert max(p[2] for p in near) == pytest.approx(g + PARAMS["height_m"], abs=0.01)


def test_eight_crenellated_octagonal_towers(old, dims, built):
    foot = Polygon(PARAMS["footprint_local"])
    assert len(dims["towers"]) == 8
    for cx, cy, r, _ in dims["towers"]:
        assert foot.buffer(0.5).contains(Point(cx, cy).buffer(r * 0.9))       # on the footprint's tower outlines
    high = built["main"]["lods"][0][2]
    cx, cy, r, top = dims["towers"][0]
    sides = {round(math.degrees(math.atan2(n[1], n[0])) % 360.0, 1)          # its walls, up past the wings
             for pts, _, _ in high.faces for n in [brlm.normal(pts)] if abs(n[2]) < 1e-6
             and all(math.hypot(p[0] - cx, p[1] - cy) <= r + 1e-6 for p in pts)
             and max(p[2] for p in pts) > PARAMS["ground_dz"] + old.WING_M}
    assert len(sides) == 8
    merlons = [pts for pts, _, _ in high.faces if brlm.normal(pts)[2] > 0.999 and min(p[2] for p in pts) > top - 1e-6
               and all(math.hypot(p[0] - cx, p[1] - cy) <= r + 1e-6 for p in pts)]
    assert len(merlons) == old.MERLONS_PER_TOWER                       # battlements: merlons and gaps


def test_pointed_windows_are_decals_on_the_walls(old, dims, built):
    for px, _, b in built["main"]["lods"][:2]:
        windows = [pts for pts, _, mat in b.faces if mat == old.WINDOW]
        assert len(windows) >= 40, px
        for pts in windows:
            assert brlm.normal(pts)[2] == pytest.approx(0.0, abs=1e-6)     # upright
    assert old.WINDOW not in built["main"]["lods"][2][2].materials


def test_entrances_north_and_south(old, dims, built):
    b = built["main"]["lods"][0][2]
    doors = [pts for pts, _, mat in b.faces if mat == old.WINDOW and max(p[2] for p in pts) - min(p[2] for p in pts) > 5]
    sides = {round(brlm.normal(pts)[1]) for pts in doors}
    assert sides == {-1, 1}
    for pts in doors:
        u = sum(p[0] for p in pts) / len(pts)
        assert dims["bar"][0] < u < dims["bar"][1]                          # between the entrance towers


def test_white_stucco_at_every_level(old, built):
    for px, _, b in built["main"]["lods"]:
        assert old.STUCCO in b.materials, px


def test_window_decals_keep_the_textures_proportions(old):
    from brmap import texgen

    assert old.GOTHIC_ASPECT == texgen.GOTHIC_ASPECT    # the script can't import brmap: the two must agree
