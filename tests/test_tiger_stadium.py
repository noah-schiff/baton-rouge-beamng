"""Tiger Stadium's model script (blender/landmarks/tiger_stadium.py): its plain-Python geometry, built
without Blender. (The script is loaded as a module with brexport stubbed out; the Blender run itself
is covered by the models stage and the in-game step.)"""
import importlib.util
import json
import math
import re
import sys
import types
from pathlib import Path

import pytest
from shapely.geometry import Point, Polygon

from brmap import landmarks, logos, materials, validate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "blender" / "lib"))
import brlm  # noqa: E402

# the params the models stage hands the script (out_lsu/landmarks.json, 2026-10-19)
PARAMS = json.loads((ROOT / "tests" / "fixtures" / "tiger_stadium_params.json").read_text(encoding="utf-8"))
BOWLS = ("bowl_ne", "bowl_nw", "bowl_se", "bowl_sw")
PARTS = (*BOWLS, "field", "lights", "board")
LODS = (600, 150, 40)
PURPLE, GOLD = (0x46 / 255, 0x1D / 255, 0x7C / 255), (0xFD / 255, 0xD0 / 255, 0x23 / 255)


class Stop(Exception):
    """Raised by the stub brexport.args, to end the script's main() before it needs Blender."""


@pytest.fixture(scope="module")
def stadium():
    stub = types.ModuleType("brexport")

    def args():
        raise Stop

    stub.args = args
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "brexport", stub)
        spec = importlib.util.spec_from_file_location("tiger_stadium", ROOT / "blender/landmarks/tiger_stadium.py")
        module = importlib.util.module_from_spec(spec)
        with pytest.raises(Stop):
            spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def dims(stadium):
    return stadium.resolve(PARAMS)


@pytest.fixture(scope="module")
def built(stadium, dims):
    return stadium.parts(dims)


def levels(part):
    """(name, Builder) of every level of detail of a part and its collision mesh."""
    return [(f"a{px}", b) for px, _, b in part["lods"]] + [("Colmesh-1", part["collision"][1])]


def close(c, want, tol=1.5 / 255):
    return c is not None and all(abs(a - b) <= tol for a, b in zip(c, want))


# ------------------------------------------------------------------ parts and budgets

def test_parts_and_levels(built):
    assert set(built) == set(PARTS)
    for name, part in built.items():
        assert [px for px, _, _ in part["lods"]] == list(LODS), name
        assert part["collision"][1].faces, name


def test_every_level_stays_under_the_vertex_cap(built):
    for name, part in built.items():
        for level, b in levels(part):
            assert b.corners <= validate.DAE_MAX_VERTS, (name, level, b.corners)


def test_levels_get_cheaper_and_stay_in_budget(built):
    for name, part in built.items():
        tris = [b.tris for _, _, b in part["lods"]]
        assert tris == sorted(tris, reverse=True) and tris[0] > tris[-1], (name, tris)
        for px, budget, b in part["lods"]:
            assert b.tris <= budget, (name, px, b.tris, budget)
        budget, col = part["collision"]
        assert col.tris <= budget, (name, col.tris)


def test_node_names_have_no_underscore_before_a_number(built):
    # lod_nodes names the levels <part>_a<px>; brexport refuses "_<n>" at the end (a minus sign in game)
    for name in built:
        for node in [f"{name}_a{px}" for px in LODS]:
            assert re.fullmatch(r"[a-z]+(_[a-z]+)*_a\d+", node) and not re.search(r"_\d+$", node), node


def test_every_material_it_binds_is_in_the_registry(built):
    registry = materials.load_registry(landmarks.MATERIAL_REGISTRY)
    for name, part in built.items():
        for level, b in levels(part):
            assert b.materials <= set(registry), (name, level, b.materials - set(registry))


def test_field_collision_is_grass_and_the_bowls_concrete(built):
    registry = materials.load_registry(landmarks.MATERIAL_REGISTRY)
    assert built["field"]["collision"][1].materials == {"br_lm_turf"}
    assert registry["br_lm_turf"]["flags"]["groundType"] == "GRASS"
    for name in BOWLS:
        assert built[name]["collision"][1].materials == {"br_lm_concrete_tan"}
    assert registry["br_lm_seats"]["flags"]["vertColor"] is True


# ------------------------------------------------------------------ the bowl

def test_rim_height_follows_the_profile_held_to_rim_m(stadium, dims):
    g = PARAMS["ground_dz"]
    for angle, h in PARAMS["rim_profile"]:
        assert stadium.rim_height(dims, angle) == pytest.approx(g + min(max(h, 20), 55), abs=1e-6)
    low = stadium.resolve({**PARAMS, "rim_profile": [[0.0, 5.0], [180.0, 80.0]]})
    assert stadium.rim_height(low, 0.0) == pytest.approx(g + 20)
    assert stadium.rim_height(low, 180.0) == pytest.approx(g + 55)


def test_bowl_sections_have_two_decks_and_a_concourse(stadium, dims):
    for angle in range(0, 360, 5):
        c = stadium.column(dims, float(angle))
        height = c["H"] - dims["g"]
        assert c["zc"] - dims["g"] == pytest.approx(0.45 * height)                     # the concourse break
        assert c["s2"] - c["lip"] == pytest.approx(4.0)                                 # the overhang
        assert 0 < c["s1"] < c["lip"] < c["s2"] < c["top"] < c["L"]
        assert c["zf"] == pytest.approx(dims["field_z"] + 1.0)                          # the field wall
        assert c["zf"] < c["zc"] < c["zs"] < c["zu"] < c["H"]


def seats_near(b, dims, angle, tol=2.0):
    cx, cy = dims["centre"]
    return [p for (pts, _, mat) in b.faces if mat == "br_lm_seats" for p in pts
            if abs((brlm.angle_of((cx, cy), p) - angle + 180) % 360 - 180) <= tol]


def test_the_top_row_is_at_the_rim(stadium, dims, built):
    bowl = {name: part["lods"][0][2] for name, part in built.items() if name in BOWLS}
    for angle, name in ((45, "bowl_nw"), (135, "bowl_sw"), (225, "bowl_se"), (315, "bowl_ne")):
        top = max(p[2] for p in seats_near(bowl[name], dims, angle))
        assert top == pytest.approx(stadium.rim_height(dims, angle), abs=0.6), angle
    north = max(p[2] for b in (bowl["bowl_ne"], bowl["bowl_nw"]) for p in seats_near(b, dims, 0.0))
    west = max(p[2] for p in seats_near(bowl["bowl_nw"], dims, 90.0))
    assert north < west - 15                       # the low north end, the tall west upper deck


def test_seats_are_purple_and_gold(built):
    for name in BOWLS:
        for px, _, b in built[name]["lods"]:
            cols = [c for (_, _, mat), c in zip(b.faces, b.cols) if mat == "br_lm_seats"]
            assert any(close(c, PURPLE) for c in cols) and any(close(c, GOLD) for c in cols), (name, px)


def test_quarters_meet(stadium, dims, built):
    # neighbouring parts share their boundary column, so the bowl closes without a gap
    def boundary(name, angle):
        b = built[name]["lods"][0][2]
        cx, cy = dims["centre"]
        return {(round(p[0], 3), round(p[1], 3), round(p[2], 3)) for pts, _, _ in b.faces for p in pts
                if abs((brlm.angle_of((cx, cy), p) - angle + 180) % 360 - 180) < 1e-6}
    for a, b, angle in (("bowl_nw", "bowl_ne", 0.0), ("bowl_nw", "bowl_sw", 90.0), ("bowl_sw", "bowl_se", 180.0),
                        ("bowl_se", "bowl_ne", 270.0)):
        shared = boundary(a, angle) & boundary(b, angle)
        assert len(shared) > 50, (a, b)


def test_the_collision_wall_stands_on_the_facade(built):
    outline = Polygon(PARAMS["footprint_local"])
    for name in BOWLS:
        feet = [p for pts, _, _ in built[name]["collision"][1].faces for p in pts if abs(p[2]) < 1e-9]
        assert all(outline.buffer(0.01).contains(Point(p[:2])) for p in feet), name
        on_ring = [p for p in feet if outline.exterior.distance(Point(p[:2])) < 0.01]
        assert len(on_ring) >= len(feet) / 2, name                 # the outer face runs along the outline


def test_the_facade_has_arches_and_faces_out(stadium, dims, built):
    b = built["bowl_nw"]["lods"][0][2]
    recesses = [pts for (pts, _, mat), c in zip(b.faces, b.cols)
                if mat == "br_lm_concrete_tan" and close(c, stadium.RECESS)]
    assert len(recesses) >= 15                       # a blind arch every bay or so round the quarter
    cx, cy = dims["centre"]
    for pts in recesses:
        n = brlm.normal(pts)
        mid = [sum(c) / len(pts) for c in zip(*pts)]
        assert n[0] * (mid[0] - cx) + n[1] * (mid[1] - cy) > 0


# ------------------------------------------------------------------ the field

def test_atlas_regions_are_the_logos_ones(stadium):
    for name, uv in stadium.ATLAS.items():
        assert uv == pytest.approx(logos.atlas_uv(name)), name


def faces_with_uv(b, box):
    """The atlas decals showing exactly that uv box."""
    u0, v0, u1, v1 = box
    return [pts for pts, uv, mat in b.faces if mat == "br_lm_field_marks" and
            all(min(abs(u - u0), abs(u - u1)) < 1e-9 and min(abs(v - v0), abs(v - v1)) < 1e-9 for u, v in uv) and
            {min(u0, u1), max(u0, u1)} == {min(u for u, _ in uv), max(u for u, _ in uv)} and
            {v0, v1} == {min(v for _, v in uv), max(v for _, v in uv)}]


def test_field_markings(stadium, dims, built):
    b = built["field"]["lods"][0][2]
    zf = dims["field_z"]
    assert zf == pytest.approx(max(PARAMS["ground_dz"], PARAMS["field_dz"]) + 0.03)
    turf = [pts for pts, _, mat in b.faces if mat == "br_lm_turf"]
    assert turf and all(p[2] == pytest.approx(zf) for pts in turf for p in pts)
    others = [p for pts, _, mat in b.faces if mat != "br_lm_turf" for p in pts]
    assert all(zf < p[2] <= zf + 0.05 for p in others)                     # just above the turf
    lines = [pts for pts, _, mat in b.faces if mat == "br_lm_field_paint" and
             sorted(round(math.dist(pts[0], q), 2) for q in pts[1:])[0] == pytest.approx(0.1, abs=0.01)]
    across = [pts for pts in lines if max(math.dist(pts[0], q) for q in pts[1:]) > 48.0]
    assert len(across) == 21                                                # every 5 yd, goal line to goal line
    assert len(faces_with_uv(b, stadium.ATLAS["end_zone_a"])) == len(faces_with_uv(b, stadium.ATLAS["end_zone_b"])) == 1
    assert len(faces_with_uv(b, stadium.ATLAS["tiger"])) == 1
    numerals = sum(len(faces_with_uv(b, stadium.ATLAS[f"numeral_{d}"])) for d in range(6))
    assert numerals == 36                                                    # 10 .. 50 .. 10, both sides
    assert len(faces_with_uv(b, stadium.ATLAS["arrow_up"])) == 16
    stripes = {c for (_, _, mat), c in zip(b.faces, b.cols) if mat == "br_lm_turf"}
    assert len(stripes) == 2                                                  # mowed light and dark
    (plane,) = [pts for pts, _, _ in built["field"]["collision"][1].faces]
    assert all(p[2] == pytest.approx(zf) for p in plane)


def test_end_zones_lie_at_the_ends_of_the_field(stadium, dims, built):
    b = built["field"]["lods"][0][2]
    (pu, pv), length = PARAMS["pitch_local"], PARAMS["field_m"][0]
    for name, sign in (("end_zone_a", 1), ("end_zone_b", -1)):
        (pts,) = faces_with_uv(b, stadium.ATLAS[name])
        u = sorted(p[0] - pu for p in pts)
        assert sign * u[0 if sign > 0 else -1] == pytest.approx(length / 2 - 9.144, abs=0.2)
        assert brlm.normal(pts)[2] == pytest.approx(1.0)


# ------------------------------------------------------------------ the lights and the board

def test_four_light_towers_at_the_high_corners(stadium, dims, built):
    b = built["lights"]["lods"][0][2]
    lamps = [pts for pts, _, mat in b.faces if mat == "br_lm_lamp"]
    cx, cy = dims["centre"]
    corners = {round(brlm.angle_of((cx, cy), [sum(c) / len(pts) for c in zip(*pts)]) / 90 - 0.5) % 4 for pts in lamps}
    assert corners == {0, 1, 2, 3}                                           # one tower per quarter
    top = max(p[2] for pts, _, _ in b.faces for p in pts)
    assert top == pytest.approx(dims["g"] + stadium.LIGHT_TOP + stadium.BANK_H, abs=1.0)
    for angle in stadium.LIGHT_ANGLES:
        assert stadium.rim_height(dims, angle) - dims["g"] > 40                # on the tall decks


def test_board_faces_the_field_from_the_north_end(stadium, dims, built):
    b = built["board"]["lods"][0][2]
    (face,) = [(pts, uv) for pts, uv, mat in b.faces if mat == "br_lm_board"]
    pts, uv = face
    assert brlm.normal(pts)[0] == pytest.approx(-1.0, abs=0.01)             # towards -u, the field
    assert min(p[0] for p in pts) > PARAMS["pitch_local"][0] + 60             # beyond the north end zone
    assert sorted({u for u, _ in uv}) == [0.0, 1.0] and sorted({v for _, v in uv}) == [0.0, 1.0]
    assert max(p[2] for p in pts) == pytest.approx(dims["g"] + stadium.BOARD_TOP - stadium.BOARD_FRAME, abs=0.01)
