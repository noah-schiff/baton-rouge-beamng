"""Mike the Tiger's habitat's model script (blender/landmarks/mike_habitat.py): its plain-Python
geometry on the real LSU params (the traced enclosure), built without Blender."""
import math

import pytest
from shapely.geometry import LineString, Point, Polygon

from brmap import landmarks, materials
from landmark_scripts import check_parts, fixture_params, load_script

import brlm  # noqa: E402  (landmark_scripts puts blender/lib on the path)

PARAMS = fixture_params("mike_habitat")      # out_lsu/landmarks.json, 2026-10-19
LODS = (300, 80, 20)
MATERIALS = {"br_lm_stucco_tan", "br_lm_roof_tile", "br_lm_trim_stone", "br_lm_rock", "br_lm_water", "br_lm_glass",
             "br_lm_steel", "br_lm_concrete"}


@pytest.fixture(scope="module")
def habitat():
    return load_script("mike_habitat")


@pytest.fixture(scope="module")
def dims(habitat):
    return habitat.resolve(PARAMS)


@pytest.fixture(scope="module")
def built(habitat, dims):
    return habitat.parts(dims)


def full(built):
    return built["main"]["lods"][0][2]


def faces(b, mat):
    return [pts for pts, _, m in b.faces if m == mat]


def test_parts_and_levels(built):
    assert set(built) == {"main"}
    check_parts(built, LODS)


def test_no_animal_only_the_habitat(built):
    for px, _, b in built["main"]["lods"]:
        assert b.materials <= MATERIALS, px


def test_the_night_house_tower(habitat, built):
    b = full(built)
    g = PARAMS["ground_dz"]
    roofs = [p for pts in faces(b, habitat.TILE_ROOF) for p in pts]
    apex = max(roofs, key=lambda p: p[2])
    assert apex[2] == pytest.approx(g + PARAMS["tower_m"])
    assert apex[:2] == pytest.approx(tuple(PARAMS["night_house_m"]), abs=0.01)
    stucco = [p for pts in faces(b, habitat.STUCCO) for p in pts]
    near = [p for p in stucco if math.dist(p[:2], PARAMS["night_house_m"]) < habitat.TOWER_W]
    assert max(p[2] for p in near) > g + PARAMS["tower_m"] - 3.0              # a tower, not just a roof
    assert Polygon(PARAMS["footprint_local"]).buffer(0.5).contains(Point(apex[:2]))


def test_the_pool_and_the_stream(habitat, dims, built):
    b = full(built)
    g = PARAMS["ground_dz"]
    water = faces(b, habitat.WATER)
    pool = Polygon(PARAMS["pool_m"])
    in_pool = [pts for pts in water if all(abs(p[2] - dims["z_pool"]) < 1e-6 for p in pts)
               and pool.buffer(0.01).contains(Point(pts[0][:2]))]
    assert sum(brlm._area(pts) for pts in in_pool) == pytest.approx(pool.area, rel=1e-3)
    assert dims["z_pool"] == pytest.approx(g + habitat.POOL_LIFT)
    stream = LineString(PARAMS["stream_m"])
    ribbon = [p for pts in water for p in pts if stream.distance(Point(p[:2])) <= habitat.STREAM_W / 2 + 0.05
              and not pool.contains(Point(p[:2]))]
    head = min(ribbon, key=lambda p: math.dist(p[:2], PARAMS["stream_m"][0]))
    foot = min(ribbon, key=lambda p: math.dist(p[:2], PARAMS["stream_m"][-1]))
    assert head[2] > foot[2] and foot[2] == pytest.approx(dims["z_pool"], abs=0.01)   # it runs down to the pool
    rock = [p for pts in faces(b, habitat.ROCK) for p in pts]
    assert max(p[2] for p in rock if pool.buffer(2.0).contains(Point(p[:2]))) > dims["z_pool"]   # the pool's rim


def test_rock_work_where_it_was_traced(habitat, built):
    b = full(built)
    g = PARAMS["ground_dz"]
    rock = [p for pts in faces(b, habitat.ROCK) for p in pts]
    for x, y, r, h in PARAMS["rocks_m"]:
        mine = [p for p in rock if math.dist(p[:2], (x, y)) <= r * 1.2]
        assert mine and max(p[2] for p in mine) == pytest.approx(g + h, abs=0.3), (x, y)


def test_glass_along_the_viewing_edge_and_a_fence_round_the_rest(habitat, built):
    registry = materials.load_registry(landmarks.MATERIAL_REGISTRY)
    b = full(built)
    view = LineString(PARAMS["viewing_m"])
    glass = [p for pts in faces(b, habitat.GLASS) for p in pts]
    assert glass and all(view.distance(Point(p[:2])) < 0.5 for p in glass)
    widths = [max(math.dist(p[:2], q[:2]) for p in pts for q in pts) for pts in faces(b, habitat.GLASS)]
    assert sum(widths) == pytest.approx(view.length, rel=0.1)                # panes all along it
    assert registry[habitat.GLASS]["flags"]["translucent"] is True
    outline = Polygon(PARAMS["footprint_local"]).exterior
    steel = [p for pts in faces(b, habitat.STEEL) for p in pts if view.distance(Point(p[:2])) > 1.0]
    assert steel and all(outline.distance(Point(p[:2])) < 0.5 for p in steel)
    west = [p for p in steel if p[0] < -16.0]                                  # the fence on the west side
    assert max(p[1] for p in west) - min(p[1] for p in west) > 25.0


def test_collision(habitat, built):
    col = built["main"]["collision"][1]
    assert col.materials <= {habitat.CONCRETE, habitat.STUCCO, habitat.ROCK, habitat.STEEL}
    view = LineString(PARAMS["viewing_m"])
    assert any(view.distance(Point(p[:2])) < 0.5 for pts, _, _ in col.faces for p in pts)
