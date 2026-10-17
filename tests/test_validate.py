import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from shapely.geometry import box

from brmap import validate
from brmap.buildings import Building, save_buildings
from brmap.config import Config
from brmap.geo import Extent
from brmap.markings import Marking, save_markings
from brmap.roads import Road, save_roads
from brmap.trees import save_trees
from brmap.water import WaterBody, save_water

N = 256


def cfg():
    return Config(debug_half_size_m=512.0, terrain_size=N)


def road(xs, ys, zs, oid=1, bridge=False, name="Test Road"):
    xs, ys, zs = (np.asarray(v, float) for v in (xs, ys, zs))
    return Road([oid], "primary", np.column_stack([xs, ys, zs]), 7.1, 2, False, bridge,
                1 if bridge else 0, name)


def good_roads():
    xs = np.linspace(-300, 300, 31)
    ground = road(xs, np.zeros(31), np.full(31, 5.0))
    deck = road(xs, np.full(31, 100.0), 5 + 4 * np.sin(np.linspace(0, np.pi, 31)), oid=2, bridge=True)
    return [ground, deck]


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "data" / "out"
    out.mkdir(parents=True)
    np.save(out / "heights.npy", np.full((N, N), 5.0, np.float32))
    np.save(out / "layers.npy", np.zeros((N, N), np.uint8))
    Image.fromarray(np.full((N, N, 3), 90, np.uint8)).save(out / "basecolor.png")
    save_roads(good_roads(), out / "roads.json")
    save_water([WaterBody("Pond", box(-400, -400, -300, -300), 4.0, 2.0)], out / "water.json")
    return out


def test_check_clean_on_good_fixture(data):
    assert validate.check(cfg()) == []


def test_check_flags_low_bridge(data):
    xs = np.linspace(-300, 300, 31)
    zs = np.full(31, 8.0)
    zs[15] = 3.0  # 2 m below the 5 m ground, mid-span
    save_roads([road(xs, np.zeros(31), zs, oid=77, bridge=True, name="Low Bridge")], data / "roads.json")
    problems = validate.check(cfg())
    assert len(problems) == 1 and "77" in problems[0] and "Low Bridge" in problems[0]


def test_small_deck_dip_left_to_the_terrain_cut(data):
    xs = np.linspace(-300, 300, 31)
    zs = np.full(31, 8.0)
    zs[15] = 4.7  # 0.5 m short of the margin: the level writer cuts the terrain there
    save_roads([road(xs, np.zeros(31), zs, oid=78, bridge=True)], data / "roads.json")
    assert validate.check(cfg()) == []


def test_low_bridge_ends_not_flagged(data):
    xs = np.linspace(-300, 300, 31)
    zs = np.full(31, 8.0)
    zs[0] = zs[-1] = 4.0  # abutments may meet the ground
    save_roads([road(xs, np.zeros(31), zs, bridge=True)], data / "roads.json")
    assert validate.check(cfg()) == []


def test_check_flags_a_dip_at_a_deck_joint(data):
    xs = np.linspace(-300, 0, 16)
    zs = np.full(16, 15.0)
    zs[0] = zs[-1] = 5.0  # both pieces come down to the ground where they meet
    west = road(xs, np.zeros(16), zs, oid=41, bridge=True, name="West Approach")
    east = road(-xs[::-1], np.zeros(16), zs[::-1], oid=42, bridge=True, name="River Span")
    save_roads([west, east], data / "roads.json")
    problems = validate.check(cfg())
    assert len(problems) == 1 and "dip" in problems[0]
    assert "41" in problems[0] and "42" in problems[0]


def test_check_flags_a_step_where_a_ramp_joins_a_deck(data):
    xs = np.linspace(-300, 300, 31)
    main = road(xs, np.zeros(31), np.full(31, 15.0), oid=51, bridge=True, name="Main")
    ramp = road(np.linspace(0, 200, 11), np.linspace(0, -50, 11), np.full(11, 5.0), oid=52,
                bridge=True, name="Ramp")  # starts on the mainline at its middle node
    save_roads([main, ramp], data / "roads.json")
    problems = validate.check(cfg())
    assert len(problems) == 1 and "52" in problems[0] and "51" in problems[0]
    assert "10.0 m below" in problems[0]


def test_ramp_merging_at_a_shallow_angle_is_a_joint_not_a_crossing(data):
    xs = np.linspace(-300, 300, 31)
    main = road(xs, np.zeros(31), np.full(31, 15.0), oid=71, bridge=True, name="Main")
    rx = np.linspace(-200, 0, 11)
    zs = np.r_[np.linspace(5.0, 15.0, 9), 15.0, 15.0]  # level with the mainline once on its pavement
    ramp = road(rx, -30 + 30.4 * (rx + 200) / 200, zs, oid=72, bridge=True,
                name="Ramp")  # ends 0.4 m past the mainline's centerline, so the lines cross 2.7 m back
    save_roads([main, ramp], data / "roads.json")
    assert validate.check(cfg()) == []


def test_check_flags_decks_overlapping_with_a_step(data):
    xs = np.linspace(-300, 300, 31)
    main = road(xs, np.zeros(31), np.full(31, 15.0), oid=81, bridge=True, name="Main")
    rx = np.linspace(0, 300, 16)  # leaves at the middle node, 6 degrees off, 1.5 m too high at once
    ramp = road(rx, -rx * math.tan(math.radians(6)), np.r_[15.0, np.full(15, 16.5)], oid=82,
                bridge=True, name="Ramp")
    save_roads([main, ramp], data / "roads.json")
    problems = validate.check(cfg())
    assert len(problems) == 1 and "step" in problems[0]
    assert "81" in problems[0] and "82" in problems[0]


def test_check_flags_decks_crossing_too_close(data):
    xs = np.linspace(-300, 300, 31)
    low = road(xs, np.zeros(31), np.full(31, 12.0), oid=61, bridge=True, name="Lower")
    high = road(np.zeros(31), xs, np.full(31, 13.2), oid=62, bridge=True, name="Upper")
    save_roads([low, high], data / "roads.json")
    problems = validate.check(cfg())
    assert len(problems) == 1 and "1.2 m" in problems[0] and "61" in problems[0]
    high.nodes[:, 2] = 18.0  # 6 m apart is fine
    save_roads([low, high], data / "roads.json")
    assert validate.check(cfg()) == []


def test_check_flags_out_of_box(data):
    save_roads([road([0, 522], [0, 0], [5, 5], oid=31)], data / "roads.json")
    problems = validate.check(cfg())
    assert any("outside" in p and "31" in p for p in problems)


def test_check_flags_nan(data):
    save_roads([road([0, 100], [0, 0], [5, np.nan], oid=41)], data / "roads.json")
    assert any("NaN" in p and "41" in p for p in validate.check(cfg()))


def test_check_flags_wrong_heightmap_size(data):
    np.save(data / "heights.npy", np.full((128, 128), 5.0, np.float32))
    assert any("128" in p for p in validate.check(cfg()))


def test_landmarks_round_trip_full_box():
    assert validate.landmark_errors(Config()) == []


def test_overlay_written(data):
    path = validate.render_overlay(cfg(), data / "overlay.png")
    with Image.open(path) as im:
        assert im.size == (2048, 2048) and im.mode == "RGB"
        px = np.asarray(im)
    assert (px[..., 0] > 200).any() and (px[..., 1] < 80).any()  # a red bridge is drawn


def test_overlay_crop(data):
    path = validate.render_overlay(cfg(), data / "crop.png", crop=(0.0, 0.0, 200.0), size=1024)
    with Image.open(path) as im:
        assert im.size == (1024, 1024)


def test_run_writes_overview_and_spawn_crops(data):
    validate.run(cfg())
    assert (data / "overlay.png").exists()
    assert list(data.glob("overlay_spawn_*.png"))
    assert (data / "problems.txt").read_text(encoding="utf-8").strip() == "no problems found"


# ------------------------------------------------------------------ buildings, trees, paint, shapes
# The fixture's ground road is way 1 along y = 0 (corridor |y| <= 3.55) and its deck way 2
# along y = 100 at z 5 to 9; the terrain is flat at 5 m.

def bld(bid, x0, y0, x1, y1, base_z=4.7):
    return Building(id=bid, source="osm", kind="house",
                    ring=np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float), holes=[], height=6.0,
                    base_z=base_z, roof="flat", wall="res", colour=(1.0, 1.0, 1.0), height_source="default")


def tree(x, y, z=4.8, item="br_holm_oak_city_tall"):
    return {"x": float(x), "y": float(y), "z": float(z), "item": item, "scale": 1.0, "yaw": 0.0, "height": 12.0}


def mark(name, nodes):
    return Marking(name, "centre_double", "br_line_yellow_double", 0.4, 2, 6.0, False, np.asarray(nodes, float))


def test_buildings_clear_of_the_roads_pass(data):
    save_buildings([bld("w10", -50, 20, -30, 40),
                    bld("w11", 50, 3.3, 70, 20)],  # 0.25 m on the corridor: inside the 0.5 m shrink
                   data / "buildings.json")
    assert validate.check(cfg()) == []


def test_building_on_a_road_or_a_deck_flagged(data):
    save_buildings([bld("w10", -50, 20, -30, 40), bld("w12", 50, -10, 70, 10), bld("r13", -50, 95, -30, 98)],
                   data / "buildings.json")
    problems = validate.check(cfg())
    assert len(problems) == 2
    assert "w12" in problems[0] and "way 1" in problems[0] and "Test Road" in problems[0]
    assert "r13" in problems[1] and "way 2" in problems[1]


def test_building_rules_match_the_buildings_stage():
    # the stage keeps a big building with a service drive ending in it, a building over a
    # covered passage (the State Capitol), and a tower part standing on its podium; the
    # validator must agree with it, and still flag a street through a building
    ext = Extent.from_config(cfg())
    terrain = np.full((64, 64), 5.0, np.float32)

    def rd(xy, kind="service", width=4.0, covered=False):
        xy = np.asarray(xy, float)
        return Road([9], kind, np.column_stack([xy, np.full(len(xy), 5.0)]), width, 1, False, False, 0,
                    None, covered=covered)

    big, passage, tower = bld("w20", 0, 0, 60, 60), bld("w21", 100, 0, 160, 60), bld("w20/t0", 20, 20, 40, 40)
    roads = [rd([(30, -40), (30, 10)]), rd([(80, 30), (180, 30)], covered=True)]
    out = validate.building_problems([big, passage, tower], roads, terrain, ext)
    assert [p for p in out if "corridor" in p] == []
    street = [rd([(30, -40), (30, 10)], kind="residential", width=6.5)]
    assert any("w20 " in p and "corridor" in p for p in validate.building_problems([big], street, terrain, ext))


def test_tower_part_may_sit_below_ground_inside_its_podium():
    # a tower part stands on its podium's base (the podium's lowest corner), so on a slope it
    # starts below the ground under its own corners, hidden in the podium; above it would float
    ext = Extent.from_config(cfg())
    terrain = np.full((64, 64), 5.0, np.float32)
    sunk, floating = bld("w30/t0", 0, 0, 20, 20, base_z=2.0), bld("w31/t0", 40, 0, 60, 20, base_z=8.0)
    out = validate.building_problems([sunk, floating], [], terrain, ext)
    assert len(out) == 1 and "w31/t0" in out[0] and "above" in out[0]


def test_base_checked_on_the_dry_ground_along_the_walls(data):
    # the validator finds a building's ground as the stage does: the lowest terrain along its
    # walls, sampled every 2 m, leaving out the points in a water body or at its carved edge
    h = np.full((N, N), 5.0, np.float32)
    X, Y = np.meshgrid(*Extent.from_config(cfg()).pixel_centers(N))
    h[(np.abs(X + 350) < 50) & (np.abs(Y + 350) < 50)] = -10.0  # the pond's carved bed
    h[(np.abs(X - 50) < 8) & (np.abs(Y - 20) < 8)] = 2.0        # a dip under the middle of a wall
    np.save(data / "heights.npy", h)
    save_buildings([bld("w40", -315, -320, -275, -300),            # 15 of its 40 m over the pond
                    bld("w41", 0, 20, 99, 40, base_z=1.7)],         # a 99 m wall over the dip
                   data / "buildings.json")
    assert validate.check(cfg()) == []


def test_trees_clear_of_roads_decks_and_buildings_pass(data):
    save_buildings([bld("w10", -50, 20, -30, 40)], data / "buildings.json")
    save_trees([tree(0, 6.0),     # 2.45 m off the road's corridor (needs 1.5)
                tree(0, 107.0),   # 3.45 m off the deck's corridor (needs 3)
                tree(-25, 30)],   # beside the house
               data / "trees.json")
    assert validate.check(cfg()) == []


def test_trees_on_roads_decks_and_in_buildings_flagged(data):
    save_buildings([bld("w10", -50, 20, -30, 40)], data / "buildings.json")
    save_trees([tree(0, 4.5), tree(0, 105.0), tree(-40, 30, item="br_tree_oak_large_a"), tree(200, 200)],
               data / "trees.json")
    problems = validate.check(cfg())
    assert len(problems) == 3
    assert "tree 0" in problems[0] and "way 1" in problems[0]
    assert "tree 1" in problems[1] and "way 2" in problems[1] and "deck" in problems[1]
    assert "tree 2" in problems[2] and "br_tree_oak_large_a" in problems[2] and "w10" in problems[2]


def test_trees_without_buildings_still_checked(data):
    save_trees([tree(0, 0)], data / "trees.json")
    problems = validate.check(cfg())
    assert len(problems) == 1 and "tree 0" in problems[0]


def test_features_on_the_terrain_pass(data):
    save_buildings([bld("w20", -50, 20, -30, 40, base_z=4.7), bld("w21", 50, 20, 70, 40, base_z=4.2)],
                   data / "buildings.json")
    save_trees([tree(-200, -200, z=4.8), tree(200, -200, z=5.9)], data / "trees.json")
    assert validate.check(cfg()) == []


def test_floating_or_sunk_features_flagged(data):
    save_buildings([bld("w20", -50, 20, -30, 40, base_z=7.0), bld("w21", 50, 20, 70, 40, base_z=2.0)],
                   data / "buildings.json")
    save_trees([tree(-200, -200, z=6.5), tree(200, -200, z=4.5)], data / "trees.json")
    problems = validate.check(cfg())
    assert len(problems) == 3
    assert "w20" in problems[0] and "2.0 m above" in problems[0]
    assert "w21" in problems[1] and "3.0 m below" in problems[1]
    assert "tree 0" in problems[2] and "1.5 m above" in problems[2]


def test_heights_checked_against_the_terrain_the_level_uses(data, monkeypatch):
    def cut(heights, extent, roads):  # stands in for a deep cut under a deck
        return np.full_like(heights, 2.0), {"Test Road": 3.0}

    monkeypatch.setattr(validate, "clear_under_decks", cut)
    save_buildings([bld("w20", -50, 20, -30, 40, base_z=4.7)], data / "buildings.json")
    problems = validate.check(cfg())
    assert len(problems) == 1 and "w20" in problems[0] and "above" in problems[0]


def test_markings_inside_the_box_pass(data):
    save_markings([mark("12_centre_0", [[-100, 0, 5], [100, 0, 5]])], data / "markings.json")
    assert validate.check(cfg()) == []


def test_marking_outside_the_box_flagged(data):
    save_markings([mark("12_centre_0", [[-100, 0, 5], [100, 0, 5]]),
                   mark("13_divider_1", [[400, 0, 5], [530, 0, 5]])], data / "markings.json")
    problems = validate.check(cfg())
    assert len(problems) == 1 and "13_divider_1" in problems[0] and "outside" in problems[0]


def dae_text(counts: dict[str, int]) -> str:
    """A tile in dae.py's layout: one mesh per level of detail, every input at offset 0."""
    geoms, nodes = [], []
    for lod, n in counts.items():
        g = f"{lod}-mesh"
        pos = " ".join(["0 0 0"] * n)
        tris = " ".join(str(i % n) for i in range(3 * max(1, n // 3)))
        geoms.append(f'<geometry id="{g}" name="{g}"><mesh>'
                     f'<source id="{g}-positions"><float_array id="{g}-positions-array" count="{3 * n}">{pos}'
                     f'</float_array><technique_common><accessor source="#{g}-positions-array" count="{n}" '
                     f'stride="3"><param name="X" type="float"/><param name="Y" type="float"/>'
                     f'<param name="Z" type="float"/></accessor></technique_common></source>'
                     f'<vertices id="{g}-vertices"><input semantic="POSITION" source="#{g}-positions"/></vertices>'
                     f'<triangles material="m-material" count="{len(tris.split()) // 3}">'
                     f'<input semantic="VERTEX" source="#{g}-vertices" offset="0"/><p>{tris}</p></triangles>'
                     f'</mesh></geometry>')
        nodes.append(f'<node id="{lod}" name="{lod}"><instance_geometry url="#{g}"/></node>')
    return ('<?xml version="1.0" encoding="utf-8"?>\n'
            '<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">'
            '<library_geometries>' + "".join(geoms) + '</library_geometries>'
            '<library_visual_scenes><visual_scene id="Scene" name="Scene">'
            '<node id="base00" name="base00"><node id="start01" name="start01">' + "".join(nodes)
            + '<node id="nulldetail15" name="nulldetail15"/></node><node id="collision-1" name="collision-1"/>'
            '</node></visual_scene></library_visual_scenes></COLLADA>\n')


def shapes_dir():
    """The building shapes folder of the built level (the data fixture's working folder)."""
    path = Path("dist") / "level" / "levels" / "baton_rouge" / "art" / "shapes" / "buildings"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_dae_vertex_counts(tmp_path):
    path = tmp_path / "t.dae"
    path.write_text(dae_text({"bld_L300": 30, "bld_L60": 12, "Colmesh-1": 9}), encoding="utf-8")
    assert validate.dae_vertex_counts(path) == {"bld_L300": 30, "bld_L60": 12, "Colmesh-1": 9}


def test_dae_vertex_counts_with_split_indices(tmp_path):
    """Inputs at different offsets: the engine makes a vertex per distinct index tuple."""
    text = dae_text({"bld_L300": 4}).replace(
        '<input semantic="VERTEX" source="#bld_L300-mesh-vertices" offset="0"/><p>0 1 2</p>',
        '<input semantic="VERTEX" source="#bld_L300-mesh-vertices" offset="0"/>'
        '<input semantic="NORMAL" source="#n" offset="1"/><p>0 0 1 0 2 0 0 1 2 1 3 1</p>')
    assert 'offset="1"' in text
    (tmp_path / "t.dae").write_text(text, encoding="utf-8")
    assert validate.dae_vertex_counts(tmp_path / "t.dae") == {"bld_L300": 6}


def test_dae_tiles_under_the_cap_pass(data):
    (shapes_dir() / "br_bld_0_0.dae").write_text(dae_text({"bld_L300": 60000, "bld_L60": 300, "Colmesh-1": 300}),
                                                 encoding="utf-8")
    assert validate.check(cfg()) == []


def test_dae_tile_over_the_cap_flagged(data):
    (shapes_dir() / "br_bld_0_0.dae").write_text(dae_text({"bld_L300": 300, "bld_L60": 300, "Colmesh-1": 300}),
                                                 encoding="utf-8")
    (shapes_dir() / "br_bld_0_1.dae").write_text(dae_text({"bld_L300": 60001, "bld_L60": 300, "Colmesh-1": 300}),
                                                 encoding="utf-8")
    problems = validate.check(cfg())
    assert len(problems) == 1 and "br_bld_0_1.dae" in problems[0] and "bld_L300" in problems[0]
    assert "60001" in problems[0]


def test_overlay_draws_footprints_and_trees(data):
    save_buildings([bld("w30", -260, 140, -200, 200)], data / "buildings.json")
    save_trees([tree(250, 250)], data / "trees.json")
    path = validate.render_overlay(cfg(), data / "overlay.png")
    with Image.open(path) as im:
        px = np.asarray(im)

    def at(x, y):  # 2048 px over 1024 m
        return tuple(int(v) for v in px[int((512 - y) * 2), int((x + 512) * 2)])

    assert at(-230, 170) == validate.FOOTPRINT_COLOUR
    assert at(250, 250) == validate.TREE_COLOUR
    assert at(250, -250) not in (validate.FOOTPRINT_COLOUR, validate.TREE_COLOUR)


def test_overlay_thins_the_trees():
    idx = validate.thin(1000, 200)
    assert len(idx) == 200 and len(set(idx.tolist())) == 200 and idx.min() >= 0 and idx.max() < 1000
    assert (validate.thin(50, 200) == np.arange(50)).all()
