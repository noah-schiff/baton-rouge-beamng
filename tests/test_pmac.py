"""The PMAC's model script (blender/landmarks/pmac.py): its plain-Python geometry on the real LSU
params, built without Blender."""
import math

import pytest
from shapely.geometry import Point, Polygon

from landmark_scripts import check_parts, close, fixture_params, load_script

import brlm  # noqa: E402  (landmark_scripts puts blender/lib on the path)

PARAMS = fixture_params("pmac")      # out_lsu/landmarks.json, 2026-10-19
LODS = (600, 150, 40)


@pytest.fixture(scope="module")
def pmac():
    return load_script("pmac")


@pytest.fixture(scope="module")
def dims(pmac):
    return pmac.resolve(PARAMS)


@pytest.fixture(scope="module")
def built(pmac, dims):
    return pmac.parts(dims)


def radius(p):
    return math.hypot(p[0], p[1])


def bearing(p):
    """The compass bearing of a model point (+u east, +v north)."""
    return math.degrees(math.atan2(p[0], p[1])) % 360.0


def test_parts_and_levels(built):
    assert set(built) == {"main", "annex"}
    check_parts(built, LODS)


def test_ribs_follow_the_aerial(pmac, dims):
    ribs = pmac.rib_angles(dims)
    assert len(ribs) == PARAMS["ribs"] == 45
    assert (90.0 - ribs[0]) % 360.0 == pytest.approx(PARAMS["rib_az_deg"])         # a rib at bearing 4.6
    assert all((b - a) % 360.0 == pytest.approx(8.0) for a, b in zip(ribs, ribs[1:] + ribs[:1]))


def test_ring_wall_and_crown(pmac, dims, built):
    b = built["main"]["lods"][0][2]
    g = PARAMS["ground_dz"]
    pts = [p for pts, _, _ in b.faces for p in pts]
    assert max(p[2] for p in pts) == pytest.approx(g + PARAMS["crown_m"], abs=0.01)
    rim = [p for p in pts if abs(radius(p) - PARAMS["diameter_m"] / 2) < 1e-4]
    tops = {round(bearing(p), 3) for p in rim if abs(p[2] - (g + PARAMS["wall_m"])) < 1e-6}
    assert len(tops) == 45                                     # the ring wall's top under every rib
    assert min(p[2] for p in rim) >= g + PARAMS["wall_m"] - pmac.FOLD - pmac.SKIRT_DROP - 1e-6


def test_the_dome_profile_follows_the_lidar(pmac, dims):
    # the lidar's median dome heights above the ground at r 20, 30, 40 and 50 m (out_lsu ndsm)
    g, r = PARAMS["ground_dz"], PARAMS["diameter_m"] / 2
    for at, lidar in ((20, 29.1), (30, 27.1), (40, 24.1), (50, 19.8)):
        assert pmac.dome_z(dims, at / r) - g == pytest.approx(lidar, abs=1.0), at


def test_the_dome_is_folded(dims, built):
    b = built["main"]["lods"][0][2]
    r = PARAMS["diameter_m"] / 2 * 0.6
    ring = {round(p[2], 3) for pts, _, _ in b.faces for p in pts if abs(radius(p) - r) < 0.01}
    assert len(ring) == 2                                      # ridges and valleys
    assert max(ring) - min(ring) == pytest.approx(0.6 * 1.5, abs=0.01)


def fins(pmac, dims, b):
    """The buttress fins' sloped tops: faces facing up and out between the ring wall and the skirt's foot."""
    out = []
    for (pts, _, mat), col in zip(b.faces, b.cols):
        if col is not None:                                    # the skirt's glazing between them
            continue
        n = brlm.normal(pts)
        mid = [sum(c) / len(pts) for c in zip(*pts)]
        rad = (mid[0] / radius(mid), mid[1] / radius(mid))
        if mat == pmac.CONCRETE and n[2] > 0.2 and n[0] * rad[0] + n[1] * rad[1] > 0.2 and \
                dims["R"] < radius(mid) < dims["R"] + pmac.BUTTRESS_OUT and len(pts) == 4:
            out.append(mid)
    return out


def test_a_buttress_under_every_rib(pmac, dims, built):
    tops = fins(pmac, dims, built["main"]["lods"][0][2])
    assert len(tops) == 45
    ribs = sorted((90.0 - a) % 360.0 for a in pmac.rib_angles(dims))
    assert sorted(bearing(m) for m in tops) == pytest.approx(ribs, abs=0.01)


def test_entrance_canopies_at_their_bearings(pmac, dims, built):
    b = built["main"]["lods"][0][2]
    posts = [p for pts, _, mat in b.faces if mat == pmac.STEEL for p in pts]
    for want in PARAMS["entrances_az_deg"]:
        near = [p for p in posts if abs((bearing(p) - want + 180.0) % 360.0 - 180.0) < 10.0]
        assert near and all(pmac.PODIUM_OUT < radius(p) < pmac.PODIUM_OUT + pmac.ENTRANCE_D + 0.5 for p in near), want
    assert len({round(bearing(p) / 30) for p in posts}) == 3


def test_windows_are_dark_bands(pmac, built):
    b = built["main"]["lods"][0][2]
    assert any(close(c, pmac.DARK) for c in b.cols) and any(close(c, pmac.GLAZING) for c in b.cols)


def test_outside_a_circle(pmac):
    square = [(-10.0, -10.0), (30.0, -10.0), (30.0, 30.0), (-10.0, 30.0)]
    for r in (20.0, 35.0):                                     # the circle crosses the ring
        want = Polygon(square).difference(Point(0.0, 0.0).buffer(r, 256))
        got = pmac.outside(square, r, step_deg=1.0)
        assert sum(Polygon(g).area for g in got) == pytest.approx(want.area, rel=2e-3), r
        assert all(Polygon(g).exterior.is_ccw for g in got)
    assert pmac.outside(square, 50.0) == []                   # all inside it
    far = [(100.0, 100.0), (110.0, 100.0), (110.0, 110.0), (100.0, 110.0)]
    assert pmac.outside(far, 5.0) == [far]                     # all outside it
    with pytest.raises(ValueError, match="inside"):
        pmac.outside(square, 5.0)                              # a hole: not a case the PMAC has


def test_the_annex_is_the_footprint_outside_the_podium(pmac, dims, built):
    foot = Polygon(PARAMS["footprint_local"])
    want = foot.difference(Point(0.0, 0.0).buffer(pmac.PODIUM_OUT, 256))
    b = built["annex"]["lods"][0][2]
    g = PARAMS["ground_dz"]
    roof = [pts for pts, _, _ in b.faces if all(abs(p[2] - (g + PARAMS["annex_m"])) < 1e-6 for p in pts)
            and brlm.normal(pts)[2] > 0.99]
    assert sum(brlm._area(pts) for pts in roof) == pytest.approx(want.area, rel=0.01)
    feet = [p for pts, _, _ in b.faces for p in pts if abs(p[2]) < 1e-9]
    assert all(foot.buffer(0.05).contains(Point(p[:2])) and radius(p) > pmac.PODIUM_OUT - 0.05 for p in feet)


def test_collision_is_the_podium_and_the_annex(pmac, dims, built):
    col = built["main"]["collision"][1]
    feet = [p for pts, _, _ in col.faces for p in pts if abs(p[2]) < 1e-9]
    assert feet and all(abs(radius(p) - pmac.PODIUM_OUT) < 0.01 for p in feet)     # the podium's outer wall
    assert built["annex"]["collision"][1].materials == {pmac.CONCRETE}
