"""Alex Box Stadium's model script (blender/landmarks/alex_box.py): its plain-Python geometry on the
real LSU params, built without Blender."""
import math

import pytest
from shapely.geometry import LineString, Point, Polygon

from brmap import landmarks, materials
from landmark_scripts import check_parts, close, fixture_params, load_script

import brlm  # noqa: E402  (landmark_scripts puts blender/lib on the path)

PARAMS = fixture_params("alex_box")      # out_lsu/landmarks.json, 2026-10-19
LODS = (600, 150, 40)
PURPLE, GOLD = (0x46 / 255, 0x1D / 255, 0x7C / 255), (0xFD / 255, 0xD0 / 255, 0x23 / 255)


@pytest.fixture(scope="module")
def box():
    return load_script("alex_box")


@pytest.fixture(scope="module")
def dims(box):
    return box.resolve(PARAMS)


@pytest.fixture(scope="module")
def built(box, dims):
    return box.parts(dims)


def bearing(p, origin=(0.0, 0.0)):
    """The compass bearing of a model point (+u east, +v north) from origin."""
    return math.degrees(math.atan2(p[0] - origin[0], p[1] - origin[1])) % 360.0


def turn(a, b):
    """b - a in degrees, in [-180, 180)."""
    return (b - a + 180.0) % 360.0 - 180.0


def test_parts_and_levels(built):
    assert set(built) == {"grandstand", "bleachers", "field", "wall"}
    check_parts(built, LODS)


def test_home_plate_and_the_bases(dims):
    holes = [r for poly in PARAMS["field_local"] for r in poly[1:]]
    mound = Polygon(min(holes, key=lambda r: Polygon(r).area)).centroid.coords[0]
    assert dims["mound"] == pytest.approx(mound, abs=0.01)                    # the infield grass's hole
    home, first, second, third = dims["home"], dims["first"], dims["second"], dims["third"]
    assert math.dist(mound, home) == pytest.approx(PARAMS["mound_m"])
    assert bearing(home, mound) == pytest.approx(PARAMS["home_az_deg"])
    for a, b in ((home, first), (first, second), (second, third), (third, home)):
        assert math.dist(a, b) == pytest.approx(PARAMS["bases_m"])
    assert bearing(first, home) == pytest.approx(PARAMS["home_az_deg"] - 180 + 45)    # right of the batter
    assert Polygon(PARAMS["field_local"][0][0]).contains(Point(home))


def test_the_grandstand_arc(box, dims, built):
    g = PARAMS["ground_dz"]
    for px, _, b in built["grandstand"]["lods"]:
        pts = [p for pts, _, _ in b.faces for p in pts]
        assert all(150.0 - 1e-6 <= bearing(p) <= 255.0 + 1e-6 for p in pts), px
        assert max(p[2] for p in pts) == pytest.approx(g + PARAMS["rim_m"]), px
    b = built["grandstand"]["lods"][0][2]
    tops = [p for pts, _, _ in b.faces for p in pts if abs(p[2] - (g + PARAMS["rim_m"])) < 1e-6]
    radii = [math.hypot(p[0], p[1]) for p in tops]                          # the roof's front edge
    assert min(radii) == pytest.approx(52.0, abs=0.05) and max(radii) == pytest.approx(58.0, abs=0.05)
    assert any(close(c, PURPLE) for c in b.cols) and any(close(c, GOLD) for c in b.cols)


def test_seats_rise_from_the_field_wall(box, dims):
    for theta in (200.0, 247.5, 295.0):
        c = box.grandstand_column(dims, theta)
        assert c["zf"] == pytest.approx(dims["field_z"] + box.FRONT_WALL)
        assert c["r_f"] < c["r_c"] < c["r_top"] < c["r_b"] < c["r_e"] <= c["r_out"]
        assert c["zf"] < c["z_top"] < c["z_rb"] - box.ROOF_T                  # the top rows under the roof
        rake = math.degrees(math.atan2(c["z_top"] - c["zf"], c["r_top"] - c["r_f"]))
        assert 15.0 < rake <= box.RAKE + 1e-6


def fair(dims, p):
    return abs(turn(bearing(dims["second"], dims["home"]), bearing(p, dims["home"])))


def test_the_outfield_wall_runs_along_the_fair_part_of_the_field(box, dims, built):
    outline = LineString(PARAMS["field_local"][0][0] + PARAMS["field_local"][0][0][:1])
    b = built["wall"]["lods"][0][2]
    fronts = [p for (pts, _, mat), c in zip(b.faces, b.cols) for p in pts if close(c, box.WALL_GREEN)]
    on_line = [p for p in fronts if outline.distance(Point(p[:2])) < 0.01]
    assert len(on_line) > 20 and all(fair(dims, p) <= 45.0 + 1e-6 for p in on_line)
    angles = sorted(fair(dims, p) for p in on_line)
    assert angles[-1] == pytest.approx(45.0)                                  # out to the foul poles
    poles = [p for (pts, _, _), c in zip(b.faces, b.cols) for p in pts if close(c, GOLD)]
    assert max(p[2] for p in poles) == pytest.approx(dims["field_z"] + box.POLE_H)


def test_the_field(box, dims, built):
    registry = materials.load_registry(landmarks.MATERIAL_REGISTRY)
    b = built["field"]["lods"][0][2]
    turf = [p for pts, _, mat in b.faces if mat == box.TURF and brlm.normal(pts)[2] > 0.99 for p in pts]
    assert {round(p[2], 6) for p in turf} >= {round(dims["field_z"], 6)}
    assert dims["field_z"] == pytest.approx(max(PARAMS["ground_dz"], PARAMS["field_dz"]) + box.FIELD_LIFT)
    clay = [pts for pts, _, mat in b.faces if mat == box.CLAY]
    assert clay and all(dims["field_z"] < p[2] for pts in clay for p in pts)
    assert registry[box.CLAY]["flags"]["groundType"] == "DIRT"
    assert built["field"]["collision"][1].materials == {box.TURF}
    assert registry[box.TURF]["flags"]["groundType"] == "GRASS"
    whites = [pts for pts, _, mat in b.faces if mat == box.LINES]
    assert len(whites) >= 5                                                    # the bases, home, foul lines


def test_the_outfield_bleachers_stand_on_their_outlines(box, built):
    b = built["bleachers"]["lods"][0][2]
    feet = [Point(p[:2]) for pts, _, _ in b.faces for p in pts if abs(p[2]) < 1e-9]
    for ring in PARAMS["outlines_local"]:
        outline = Polygon(ring)
        assert sum(outline.buffer(0.05).contains(f) for f in feet) >= 8
