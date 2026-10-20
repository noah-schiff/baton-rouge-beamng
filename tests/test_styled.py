"""Styled buildings: the LSU campus core in tan stucco and red tile (styled.py, and its hooks in
buildings, trees, validate, materials, textures and level)."""
import json
import logging
import re
from pathlib import Path

import numpy as np
import pytest
import shapely
from shapely import affinity
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from brmap import beamng_format as bf
from brmap import buildings, canopy, level, materials, styled, texgen, textures, trees, validate
from brmap.buildings import Building
from brmap.config import Config
from brmap.geo import Extent
from brmap.roads import Road, save_roads

EXT = Extent(-91.160, 30.430, 512.0)   # the small_extent box
N = 256
FLAT = np.full((N, N), 5.0)
STYLE = {"wall": "br_sty_lsu_wall", "floor_m": 3.6, "bay_m": 3.0, "roof": "hip_tile",
         "roof_material": "br_lm_roof_tile", "cornice": "br_lm_trim_stone"}
QUAD_EDGE = ["w155026535", "w156100215", "w155026552", "w155026533", "w156100171", "w156100178", "w156100189",
             "w156100201", "w156100211", "w425641654", "w727209494"]
CORE_HULL = ["w155026537", "w155026560", "w155026551", "w155037359", "w155037365", "w155037357", "w155026561",
             "w425641653", "w155026536", "w155026773"]
NEVER = ["w155026610", "w1064486330"]   # St Albans, Christ the King


def bld(id="w1", ring=((0, 0), (20, 0), (20, 10), (0, 10)), height=12.0, base_z=4.7, holes=()):
    return Building(id=id, source="osm", kind="university", ring=np.asarray(ring, float),
                    holes=[np.asarray(h, float) for h in holes], height=height, base_z=base_z, roof="flat",
                    wall="brick", colour=(1.0, 1.0, 1.0), height_source="lidar")


def corners(poly) -> int:
    return len(poly.exterior.coords) - 1


L_SHAPE = Polygon([(0, 0), (30, 0), (30, 10), (10, 10), (10, 25), (0, 25)])           # 450 m2
U_SHAPE = Polygon([(0, 0), (30, 0), (30, 25), (20, 25), (20, 10), (10, 10), (10, 25), (0, 25)])  # 600 m2


# ------------------------------------------------------------------ rectangles

@pytest.mark.parametrize("angle", [0.0, 25.0])
def test_rectangles_l_shape(angle):
    poly = affinity.rotate(L_SHAPE, angle, origin=(0, 0))
    rects = styled.rectangles(poly)
    assert len(rects) == 2
    assert unary_union(rects).area == pytest.approx(poly.area, rel=0.01)
    for r in rects:
        assert corners(r) == 4 and r.area == pytest.approx(r.minimum_rotated_rectangle.area, rel=1e-6)
        assert poly.buffer(0.01).contains(r)


@pytest.mark.parametrize("angle", [0.0, -40.0])
def test_rectangles_u_shape(angle):
    poly = affinity.rotate(U_SHAPE, angle, origin=(5, 5))
    rects = styled.rectangles(poly)
    assert len(rects) == 3
    assert unary_union(rects).area == pytest.approx(poly.area, rel=0.01)


def test_rectangles_a_rectangle_is_one():
    assert len(styled.rectangles(box(0, 0, 20, 10))) == 1


def test_rectangles_stop_at_eight_and_cover_most_of_the_area():
    comb = unary_union([box(0, 0, 100, 10)] + [box(4 + 10 * k, 10, 10 + 10 * k, 30) for k in range(10)])
    rects = styled.rectangles(comb)   # a bar with ten teeth: 11 rectangles would cover it all
    assert len(rects) == styled.MAX_RECTS
    rects = styled.rectangles(unary_union([box(0, 0, 40, 20), box(40, 0, 40.6, 2)]))  # a 0.6 m stub
    assert len(rects) == 1                  # the main block alone covers over 97 % of the area


# ------------------------------------------------------------------ the split

def test_split_styled_moves_parts_too():
    blds = [bld("w1"), bld("w1/t0"), bld("w2"), bld("w3/1"), bld("w3/1/t0"), bld("w31")]
    groups = {"core": {"style": "lsu", "ids": ["w1", "w3"]}}
    rest, moved = buildings.split_styled(blds, groups)
    assert [b.id for b in rest] == ["w2", "w31"]
    assert [m["building"]["id"] for m in moved] == ["w1", "w1/t0", "w3/1", "w3/1/t0"]
    assert all(m["style"] == "lsu" and m["group"] == "core" for m in moved)
    assert moved[0]["building"] == bld("w1").to_dict()


def test_split_styled_refuses_an_id_in_two_groups():
    with pytest.raises(ValueError, match="w1"):
        buildings.split_styled([bld("w1")], {"a": {"style": "lsu", "ids": ["w1"]},
                                             "b": {"style": "lsu", "ids": ["w1"]}})


# ------------------------------------------------------------------ the meshes

def tri_corners(mesh, material):
    return mesh.pos[mesh.tris[material].ravel()]


def test_hip_roof_top():
    b = bld(height=12.0, base_z=4.7)
    meshes = styled.mesh([b], STYLE, (0.0, 0.0))
    assert set(meshes) == {"sty_a400", "sty_a60", "Colmesh-1"}
    full = meshes["sty_a400"]
    assert full.pos[:, 2].max() == pytest.approx(4.7 + 12.0, abs=0.01)                 # the roof peak
    assert tri_corners(full, STYLE["wall"])[:, 2].max() == pytest.approx(4.7 + 12.0 - 3.5, abs=0.01)
    roof = tri_corners(full, STYLE["roof_material"])
    assert roof[:, :2].min(axis=0) == pytest.approx([-0.6, -0.6], abs=0.01)             # 0.6 m eaves
    assert roof[:, :2].max(axis=0) == pytest.approx([20.6, 10.6], abs=0.01)
    assert set(full.tris) == {STYLE["wall"], STYLE["roof_material"], STYLE["cornice"]}


def test_hip_rise_is_capped():
    full = styled.mesh([bld(ring=((0, 0), (60, 0), (60, 30), (0, 30)), height=15.0, base_z=0.0)],
                       STYLE, (0.0, 0.0))["sty_a400"]
    assert tri_corners(full, STYLE["wall"])[:, 2].max() == pytest.approx(15.0 - 4.0, abs=0.01)   # at most 4 m
    low = styled.mesh([bld(ring=((0, 0), (14, 0), (14, 14), (0, 14)), height=5.8, base_z=0.0)],
                      STYLE, (0.0, 0.0))["sty_a400"]
    assert tri_corners(low, STYLE["wall"])[:, 2].max() == pytest.approx(2.5, abs=0.01)  # 2.5 m of wall stays


def test_cornice_band_is_04_m_deep():
    full = styled.mesh([bld(height=12.0, base_z=0.0)], STYLE, (0.0, 0.0))["sty_a400"]
    band = tri_corners(full, STYLE["cornice"])
    assert band[:, :2].min(axis=0) == pytest.approx([-0.4, -0.4], abs=0.01)
    assert band[:, :2].max(axis=0) == pytest.approx([20.4, 10.4], abs=0.01)
    assert band[:, 2].max() == pytest.approx(8.5, abs=0.01)      # up to the eaves


def test_wall_uv_per_floor():
    # 20 x 10 m, 10.7 m to the peak of a 3.5 m roof: 7.2 m of wall
    full = styled.mesh([bld(height=10.7, base_z=3.0)], STYLE, (0.0, 0.0))["sty_a400"]
    idx = np.unique(full.tris[STYLE["wall"]])
    v, z = full.uv[idx, 1], full.pos[idx, 2]
    assert v.min() == pytest.approx(0.0, abs=1e-3) and v.max() == pytest.approx(2.0, abs=1e-3)
    assert v == pytest.approx((z - 3.0) / 3.6, abs=1e-3)                       # one repeat per floor
    south = idx[np.abs(full.nrm[idx, 1] + 1.0) < 1e-6]                         # the 20 m wall facing -y
    assert np.ptp(full.uv[south, 0]) == pytest.approx(20.0 / 3.0, abs=1e-3)    # one repeat per bay


def test_wall_bays_are_centred_on_each_wall():
    # each wall starts and ends in the pier between two windows (u near a whole number)
    ring = ((0, 0), (13.5, 0), (13.5, 7.4), (0, 7.4))
    full = styled.mesh([bld(ring=ring, height=12.0, base_z=0.0)], STYLE, (0.0, 0.0))["sty_a400"]
    idx = np.unique(full.tris[STYLE["wall"]])
    u = full.uv[idx, 0]
    assert np.abs(u - np.round(u)).max() <= 0.25 + 1e-6


def test_flat_parapet():
    style = {**STYLE, "roof": "flat_parapet"}
    full = styled.mesh([bld(height=12.0, base_z=0.0)], style, (0.0, 0.0))["sty_a400"]
    assert tri_corners(full, style["wall"])[:, 2].max() == pytest.approx(12.0, abs=0.01)
    deck = full.pos[np.unique(full.tris[style["roof_material"]])]
    assert deck[:, 2] == pytest.approx(11.0, abs=0.01)                         # the roof 1 m below the top


def test_far_detail_and_collision_are_flat_topped_prisms():
    meshes = styled.mesh([bld(height=12.0, base_z=4.7)], STYLE, (100.0, 50.0))
    for node in ("sty_a60", "Colmesh-1"):
        pos = meshes[node].pos
        assert pos[:, 2].max() == pytest.approx(4.7 + 8.5 + 3.5 / 2, abs=0.01), node
        assert pos[:, 2].min() == pytest.approx(4.7, abs=0.01), node
        assert pos[:, :2].min(axis=0) == pytest.approx([-100.0, -50.0], abs=0.01), node   # relative to origin
    assert len(meshes["sty_a60"].pos) < len(meshes["sty_a400"].pos)


def test_l_building_gets_two_hips_and_no_holes_in_the_roof():
    b = bld(ring=list(L_SHAPE.exterior.coords)[:-1], height=14.0, base_z=0.0)
    full = styled.mesh([b], STYLE, (0.0, 0.0))["sty_a400"]
    roof = tri_corners(full, STYLE["roof_material"]).reshape(-1, 3, 3)
    up = roof[np.cross(roof[:, 1] - roof[:, 0], roof[:, 2] - roof[:, 0])[:, 2] > 1e-9]
    # seen from above, the roof's upward faces cover the whole footprint
    cover = unary_union([Polygon(t[:, :2]) for t in up if Polygon(t[:, :2]).area > 1e-9])
    assert cover.buffer(0.01).contains(L_SHAPE)
    assert full.pos[:, 2].max() == pytest.approx(14.0, abs=0.01)
    sloped = up[np.abs(np.cross(up[:, 1] - up[:, 0], up[:, 2] - up[:, 0])[:, 2]) <
                np.linalg.norm(np.cross(up[:, 1] - up[:, 0], up[:, 2] - up[:, 0]), axis=1) - 1e-6]
    assert len(sloped) >= 2 * 6          # two hipped roofs: 2 trapezoids (2 triangles each) + 2 hips


# ------------------------------------------------------------------ the registry and the texture

def test_registry_style_and_campus_core():
    reg = styled.load_registry()
    lsu = reg["styles"]["lsu"]
    assert (lsu["wall"], lsu["floor_m"], lsu["bay_m"], lsu["roof"], lsu["roof_material"]) == (
        "br_sty_lsu_wall", 3.6, 3.0, "hip_tile", "br_lm_roof_tile")
    core = reg["groups"]["campus_core"]
    assert core["style"] == "lsu"
    ids = [str(i) for i in core["ids"]]
    assert set(QUAD_EDGE + CORE_HULL) <= set(ids) and len(ids) == len(set(ids))
    assert not set(NEVER) & set(ids)
    text = styled.REGISTRY.read_text(encoding="utf-8")
    for checked in ("w155026562", "w155037367", "w155026554", "w155026546", "w863392085"):
        assert checked in text, f"styled.yaml records no decision on {checked}"
    mats = materials.load_registry(styled.MATERIAL_REGISTRY)
    assert mats["br_sty_lsu_wall"]["maps"] == {"b": "tex:lsu_wall", "nm": "tex:lsu_wall", "r": "tex:lsu_wall"}
    for style in reg["styles"].values():
        assert {style["wall"], style["roof_material"], style["cornice"]} <= set(mats)


def test_lsu_wall_texture():
    maps = texgen.lsu_wall(256)
    assert set(maps) == {"b", "nm", "r"} and all(m.size == (256, 256) for m in maps.values())
    b = np.asarray(maps["b"].convert("RGB"), float)

    def at(x0, x1, z0, z1):     # the mean colour over x0..x1 m along the bay, z0..z1 m above the floor
        return b[round(256 * (1 - z1 / 3.6)):round(256 * (1 - z0 / 3.6)),
                 round(256 * x0 / 3.0):round(256 * x1 / 3.0)].mean(axis=(0, 1))

    stucco = at(0.05, 0.4, 3.3, 3.55)                          # by the top corner: plain stucco
    assert stucco[0] > stucco[2] + 25 and stucco.min() > 110   # tan
    assert at(0.05, 0.4, 0.2, 3.5) == pytest.approx(stucco, abs=12)   # the pier beside the window
    glass = at(1.0, 1.4, 1.2, 2.2)                             # the window, left of the mullion
    assert glass.sum() < stucco.sum() - 150                    # dark
    fan = at(1.25, 1.4, 2.6, 2.75)                             # under the arch, above the springing
    assert fan.sum() < stucco.sum() - 100
    nm = np.asarray(maps["nm"], float)
    assert np.allclose(nm[4:12, 4:12].mean(axis=(0, 1)), [128, 128, 255], atol=3)   # flat stucco
    assert (texgen.LSU_BAY_M, texgen.LSU_FLOOR_M) == (styled.load_registry()["styles"]["lsu"]["bay_m"],
                                                      styled.load_registry()["styles"]["lsu"]["floor_m"])
    assert texgen.lsu_wall(256)["b"].tobytes() == maps["b"].tobytes()     # deterministic
    assert "lsu_wall" in textures.GENERATED


DOWNTOWN = {"one_american_place": ("ribbon_dark", ["w295246614"]),     # its tower part w1546031101 goes with it
            "tower_451_florida": ("precast_punched", ["w452000173"]),
            "tower_north_laurel": ("precast_punched", ["w515634563"]),
            "hotel_heidelberg": ("brick_hotel", ["w295246621"]),
            "river_center_arena": ("precast_arena", ["w452086109"]),
            "river_center_library": ("glass_blue", ["w1065992025"])}


def test_registry_downtown_standouts():
    # one group per standout, named by the plan's neutral id (group names go into file names), each in
    # its own style; the floor height from building:levels where OSM tags it, else the styles' 3.8 m
    reg = styled.load_registry()
    for group, (style, ids) in DOWNTOWN.items():
        g = reg["groups"][group]
        assert (g["style"], [str(i) for i in g["ids"]]) == (style, ids), group
        assert not re.search(r"chase|hilton|cane|marriott", group)
    floors = {group: styled.group_style(reg, group)["floor_m"] for group in DOWNTOWN}
    assert floors == {"one_american_place": 3.92, "tower_451_florida": 4.0, "tower_north_laurel": 3.33,
                      "hotel_heidelberg": 4.29, "river_center_arena": 3.8, "river_center_library": 3.8}
    mats = materials.load_registry(styled.MATERIAL_REGISTRY)
    for style in {s for s, _ in DOWNTOWN.values()}:
        st, drawn = reg["styles"][style], texgen.FACADES[style]
        assert (st["roof"], st["floor_m"]) == ("flat_parapet", 3.8), style
        assert (st["floors_per_tile"], st["bays_per_tile"], st["bay_m"], st["floor_m"]) == (
            drawn["floors"], drawn["bays"], drawn["bay_m"], drawn["floor_m"]), style
        wall = mats[st["wall"]]
        assert wall["maps"] == {"b": f"tex:facade_{style}", "r": f"tex:facade_{style}"}, style
        assert (wall.get("factors") or {}).get("metallicFactor", 0.0) == texgen.facade(style, 64).get("m", 0.0)
        assert f"facade_{style}" in textures.GENERATED


def test_group_style_takes_the_groups_floor_height():
    reg = {"styles": {"s": {**STYLE, "floor_m": 3.8}}, "groups": {"a": {"style": "s", "ids": [], "floor_m": 4.2},
                                                                 "b": {"style": "s", "ids": []}}}
    assert styled.group_style(reg, "a") == {**STYLE, "floor_m": 4.2}
    assert styled.group_style(reg, "b") == {**STYLE, "floor_m": 3.8}


def test_registry_refuses_an_unknown_group_key(tmp_path):
    path = tmp_path / "styled.yaml"
    path.write_text("styles:\n  lsu: {wall: w, floor_m: 3.6, bay_m: 3.0, roof: hip_tile, roof_material: r, cornice: c}\n"
                    "groups:\n  core: {style: lsu, ids: [w1], height_m: 4}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="height_m"):
        styled.load_registry(path)


def test_a_texture_repeat_of_several_floors_and_bays():
    # 20 x 10 m, 12 m high, flat: a repeat of 4 floors (14.4 m) and 2 bays (6 m)
    style = {**STYLE, "roof": "flat_parapet", "floors_per_tile": 4, "bays_per_tile": 2}
    full = styled.mesh([bld(height=12.0, base_z=3.0)], style, (0.0, 0.0))["sty_a400"]
    idx = np.unique(full.tris[style["wall"]])
    outer = idx[np.abs(full.nrm[idx, 2]) < 1e-6]
    v, z = full.uv[outer, 1], full.pos[outer, 2]
    assert v == pytest.approx((z - 3.0) / (3.6 * 4), abs=1e-3)
    south = outer[(np.abs(full.nrm[outer, 1] + 1.0) < 1e-6) & (full.pos[outer, 1] < 1e-6)]   # not the parapet's back
    assert np.ptp(full.uv[south, 0]) == pytest.approx(20.0 / 6.0, abs=1e-3)
    u = full.uv[outer, 0] * 2                                                  # in bays
    assert np.abs(u - np.round(u)).max() <= 0.25 + 1e-6                       # each wall still ends in a pier


# ------------------------------------------------------------------ level content

def styled_entry(id, x, y, group="campus_core", style="lsu", side=20.0, height=12.0):
    h = side / 2
    ring = ((x - h, y - h), (x + h, y - h), (x + h, y + h), (x - h, y + h))
    return {"style": style, "group": group, "building": bld(id, ring, height=height, base_z=4.7).to_dict()}


def fake_textures(data_dir):
    (data_dir / "textures").mkdir(parents=True)
    index = {}
    for tex, keys in (("lsu_wall", "b nm r"), ("bevel", "nm")):
        files = {k: f"{tex}_{textures.SUFFIX[k]}.dds" for k in keys.split()}
        for f in files.values():
            (data_dir / "textures" / f).write_bytes(b"dds")
        index[tex] = {"source": "generated", "files": files}
    (data_dir / "textures" / "index.json").write_text(json.dumps(index), encoding="utf-8")


def test_level_content_tiles(tmp_path):
    cfg = Config(debug_half_size_m=512.0, terrain_size=N, building_tile_m=256.0)
    data_dir, root = tmp_path / "out", tmp_path / "level"
    data_dir.mkdir()
    fake_textures(data_dir)
    (data_dir / "styled.json").write_text(json.dumps(
        [styled_entry("w1", 100.0, 100.0), styled_entry("w1/t0", 100.0, 100.0, side=8.0, height=30.0),
         styled_entry("w2", -100.0, 30.0)]), encoding="utf-8")
    groups = styled.level_content(cfg, data_dir, root, FLAT, EXT)
    objs = groups["styled"]
    assert len(objs) == 2                                          # one TSStatic per tile
    keys = ["w01n00", "e00n00"]
    folder = root / "art/shapes/styled"
    assert sorted(p.name for p in folder.glob("*.dae")) == sorted(f"br_sty_campus_core_{k}.dae" for k in keys)
    for o, key in zip(sorted(objs, key=lambda o: o["shapeName"]), sorted(keys)):
        assert o["shapeName"] == f"/levels/baton_rouge/art/shapes/styled/br_sty_campus_core_{key}.dae"
        assert (o["class"], o["persistentId"]) == ("TSStatic", bf.persistent_id("sty", "campus_core", key))
        assert (o["collisionType"], o["decalType"], o["useInstanceRenderData"]) == ("Collision Mesh", "None", True)
        assert o["position"][2] == 0.0
    mats = json.loads((folder / "main.materials.json").read_text(encoding="utf-8"))
    bound = set()
    for f in folder.glob("*.dae"):
        bound |= set(re.findall(r'<material id="[^"]*" name="([^"]+)"', f.read_text(encoding="utf-8")))
        assert set(validate.dae_vertex_counts(f)) == {"sty_a400", "sty_a60", "Colmesh-1"}
    assert bound == set(mats) == {"br_sty_lsu_wall", "br_lm_roof_tile", "br_lm_trim_stone"}
    assert mats["br_sty_lsu_wall"]["Stages"][0]["baseColorMap"] == (
        "/levels/baton_rouge/art/textures/br/lsu_wall_b.color.png")
    assert (root / "art/textures/br/lsu_wall_nm.normal.dds").read_bytes() == b"dds"
    assert not list((root / "art/textures/br").glob("*.png"))


def test_level_content_without_styled_buildings(tmp_path):
    data_dir = tmp_path / "out"
    data_dir.mkdir()
    (data_dir / "styled.json").write_text("[]", encoding="utf-8")
    assert styled.level_content(Config(), data_dir, tmp_path / "level", FLAT, EXT) == {}


def off(a, b) -> float:
    """The Hausdorff distance between two outlines, their edges sampled every 0.25 m."""
    return shapely.hausdorff_distance(shapely.segmentize(a, 0.25), shapely.segmentize(b, 0.25))


def test_drawn_keeps_the_osm_outline(tmp_path):
    # the walls follow the orthogonalized footprint, never its bounding rectangle: a step at one end
    # leaving a notch of 13 % stays (Prescott grew by 198 m2 when it did not), and a 0.4 m jog is
    # straightened. The checks follow the same outline.
    ring = ((0, 0), (60, 0), (60, 20), (52, 20), (52, 17), (30, 17), (30, 16.6), (0, 16.6))
    b = bld("w1/t0", ring, height=12.0, base_z=4.7)
    d = styled.drawn(b)
    assert len(d.ring) == 6 and off(Polygon(d.ring), Polygon(ring)) < 0.4
    assert Polygon(d.ring).area == pytest.approx(Polygon(ring).area, rel=0.01)
    assert (d.id, d.height, d.base_z) == (b.id, b.height, b.base_z)
    assert Polygon(d.ring).exterior.is_ccw
    path = tmp_path / "styled.json"
    path.write_text(json.dumps([{"style": "lsu", "group": "g", "building": b.to_dict()}]), encoding="utf-8")
    (e,) = styled.drawn_buildings(path)
    assert np.allclose(e.ring, d.ring)


def test_drawn_keeps_a_diagonal_wall():
    # a 7 m chamfer at 45 degrees would move more than OUTLINE_TOL squared off, so the walls keep
    # the footprint's own outline; the roof's rectangles stay inside it (test_hip_roofs_stay_over_the_walls)
    ring = ((0, 0), (40, 0), (40, 15), (35, 20), (0, 20))
    d = styled.drawn(bld("w1", ring))
    assert Polygon(d.ring).equals(Polygon(ring)) and Polygon(d.ring).exterior.is_ccw
    rects = styled.rectangles(Polygon(ring))
    assert rects and all(corners(r) == 4 and r.area == pytest.approx(r.minimum_rotated_rectangle.area)
                         for r in rects)
    full = styled.mesh([bld("w1", ring)], STYLE, (0.0, 0.0))["sty_a400"]
    wall = np.unique(full.tris[STYLE["wall"]])
    assert np.isclose(full.nrm[wall, :2], [2 ** -0.5, 2 ** -0.5], atol=1e-6).all(axis=1).any()  # the chamfer


def test_outline_runs_along_the_walls():
    # three 30 x 10 m bars in a staircase: the minimum rotated rectangle turns 26.6 degrees, and the
    # outline along it would be one tilted rectangle 5.6 m off the walls; along the walls (0 degrees)
    # it keeps all twelve corners, and the roof's rectangles run along x and y
    stairs = unary_union([box(0, 0, 30, 10), box(20, 10, 50, 20), box(40, 20, 70, 30)])
    d = styled.drawn(bld("w1", np.asarray(stairs.exterior.coords)[:-1]))
    assert len(d.ring) == 12 and off(Polygon(d.ring), stairs) < 1e-6
    rects = styled.rectangles(stairs)
    assert len(rects) == 3 and unary_union(rects).area == pytest.approx(stairs.area, rel=0.01)
    for r in rects:
        x0, y0, x1, y1 = r.bounds
        assert r.area == pytest.approx((x1 - x0) * (y1 - y0))                 # along x and y


# the eaves reach EAVES past each side of a hip's rectangle, so EAVES x sqrt 2 (0.85 m) past a corner
REACH = styled.EAVES * 2 ** 0.5 + 0.01


def roof_over_walls(b):
    """(walls, roof, hips) of b seen from above: the outline its walls follow (drawn), its
    roof-material faces, and the sloped ones among them (the hipped roofs)."""
    d = styled.drawn(b)
    walls = Polygon(d.ring, d.holes)
    tris = tri_corners(styled.mesh([b], STYLE, (0.0, 0.0))["sty_a400"], STYLE["roof_material"]).reshape(-1, 3, 3)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    sloped = np.abs(n[:, 2]) < np.linalg.norm(n, axis=1) - 1e-6
    faces = [(Polygon(t[:, :2]), s) for t, s in zip(tris, sloped) if Polygon(t[:, :2]).area > 1e-9]
    return walls, unary_union([p for p, _ in faces]), unary_union([p for p, s in faces if s])


def reach(roof, walls) -> float:
    """How far roof reaches past walls (its outline sampled every 0.25 m)."""
    pts = shapely.points(shapely.get_coordinates(shapely.segmentize(roof.boundary, 0.25)))
    return float(shapely.distance(walls, pts).max())


def check_roof_over_walls(b, cover):
    # no part of the roof reaches more than the eaves past the walls (REACH), and the hipped roofs
    # still cover at least `cover` of what the walls enclose
    walls, roof, hips = roof_over_walls(b)
    assert roof.difference(walls.buffer(REACH, quad_segs=32)).area < 1e-4, (b.id, reach(roof, walls))
    share = hips.intersection(walls).area / walls.area
    assert share >= cover, (b.id, share)


APSE = unary_union([box(0, 0, 30, 20),                                 # a 30 x 20 m nave with a six-sided apse
                    shapely.Point(30, 10).buffer(10, quad_segs=3).intersection(box(30, 0, 40, 20))])


@pytest.mark.parametrize("angle", [0.0, 25.0])
@pytest.mark.parametrize("poly", [Polygon([(0, 0), (40, 0), (40, 15), (35, 20), (0, 20)]), APSE],
                         ids=["chamfer", "apse"])
def test_hip_roofs_stay_over_the_walls(poly, angle):
    # walls that keep the footprint's diagonal (a chamfer, a polygonal apse): the hipped roofs'
    # rectangles stay inside them, so no roof hangs past a wall by more than its eaves
    poly = affinity.rotate(poly, angle, origin=(0, 0))
    b = bld("w1", np.asarray(poly.exterior.coords)[:-1], height=14.0, base_z=0.0)
    assert off(Polygon(styled.drawn(b).ring), poly) < 1e-6             # the walls keep the diagonal
    for r in styled.rectangles(poly):
        assert poly.buffer(0.01).covers(r)
    check_roof_over_walls(b, 0.85)


LSU_STYLED = Path(__file__).resolve().parents[1] / "data" / "out_lsu" / "styled.json"


@pytest.mark.skipif(not LSU_STYLED.exists(), reason="no data/out_lsu/styled.json (the LSU map's buildings stage)")
def test_drawn_outlines_stay_on_the_osm_rings():
    # every styled building of the LSU map: the walls within 1.0 m of the OSM outline (Hausdorff) and
    # the area within 5 % of its
    entries = styled.load_styled(LSU_STYLED)
    assert entries
    for e in entries:
        b = e["building"]
        ring, d = Polygon(b.ring, b.holes), styled.drawn(b)
        poly = Polygon(d.ring, d.holes)
        assert off(poly, ring) <= 1.0, b.id
        assert poly.area == pytest.approx(ring.area, rel=0.05), b.id


@pytest.mark.skipif(not LSU_STYLED.exists(), reason="no data/out_lsu/styled.json (the LSU map's buildings stage)")
def test_lsu_hip_roofs_stay_over_the_walls():
    # every styled building of the LSU map, those whose walls keep a chamfer or an apse among them:
    # the roof reaches at most its eaves past the walls, and the hips still cover most of the building
    entries = styled.load_styled(LSU_STYLED)
    assert entries
    for e in entries:
        check_roof_over_walls(e["building"], 0.8)


# ------------------------------------------------------------------ the stage and its hooks

def ll(xy):
    xy = np.asarray(xy, float)
    lon, lat = EXT.to_lonlat(xy[:, 0], xy[:, 1])
    return [[float(a), float(b)] for a, b in zip(lon, lat)]


def square_osm(id, cx, cy, side=20.0):
    h = side / 2
    ring = [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h), (cx - h, cy - h)]
    return {"id": id, "tags": {"building": "university"}, "polygons": [{"outer": ll(ring), "inner": []}]}


def small_cfg(tmp_path, **kw):
    return Config(debug_half_size_m=512.0, terrain_size=N, raw_dir=str(tmp_path / "raw"),
                  out_dir=str(tmp_path / "out"), **kw)


def registry(tmp_path, ids):
    path = tmp_path / "styled.yaml"
    path.write_text("styles:\n  lsu: {wall: br_sty_lsu_wall, floor_m: 3.6, bay_m: 3.0, roof: hip_tile,\n"
                    "        roof_material: br_lm_roof_tile, cornice: br_lm_trim_stone}\n"
                    f"groups:\n  core: {{style: lsu, ids: [{', '.join(ids)}]}}\n", encoding="utf-8")
    return path


def test_buildings_run_writes_styled_json(tmp_path, monkeypatch, caplog):
    raw, out = tmp_path / "raw", tmp_path / "out"
    raw.mkdir()
    out.mkdir()
    osm = {"buildings": [square_osm("w1", 100.0, 100.0), square_osm("w2", -200.0, 0.0)], "areas": []}
    (raw / "osm_features.json").write_text(json.dumps(osm), encoding="utf-8")
    np.save(out / "heights.npy", np.full((N, N), 6.0, np.float32))
    save_roads([], out / "roads.json")
    monkeypatch.setattr(styled, "REGISTRY", registry(tmp_path, ["w1", "w99"]))
    with caplog.at_level(logging.INFO, logger="brmap.buildings"):
        buildings.run(small_cfg(tmp_path, buildings_microsoft=False))
    assert [b.id for b in buildings.load_buildings(out / "buildings.json")] == ["w2"]
    (entry,) = styled.load_styled(out / "styled.json")
    assert (entry["style"], entry["group"], entry["building"].id) == ("lsu", "core", "w1")
    assert "1 styled" in caplog.text and "w99" in caplog.text           # the id not in this box is named


def test_trees_run_keeps_trees_out_of_styled_buildings(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (tmp_path / "raw").mkdir()
    np.save(out / "heights.npy", np.full((N, N), 6.0, np.float32))
    n, px = canopy.analysis_grid(EXT)
    xs, ys = EXT.pixel_centers(n)
    X, Y = np.meshgrid(xs, ys)
    ndsm = (12.0 * np.exp(-((X - 201.0) ** 2 + (Y - 5.0) ** 2) / 18.0)).astype(np.float32)
    canopy.save_canopy(out, ndsm, np.full((n, n), 0.6, np.float32), px)
    cfg = small_cfg(tmp_path)
    trees.run(cfg)
    assert len(trees.load_trees(out / "trees.json")) == 1
    (out / "styled.json").write_text(json.dumps([styled_entry("w1", 200.0, 0.0)]), encoding="utf-8")
    trees.run(cfg)
    assert trees.load_trees(out / "trees.json") == []


def ground_road(xy):
    xy = np.asarray(xy, float)
    return Road([1], "residential", np.column_stack([xy, np.full(len(xy), 5.0)]), 6.5, 2, False, False, 0,
                "Test Road")


def test_validate_checks_the_styled_buildings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "styled.json").write_text(json.dumps([styled_entry("w1", 0.0, 0.0)]), encoding="utf-8")
    probs = validate.feature_problems(Config(debug_half_size_m=512.0, terrain_size=N), out, EXT, FLAT,
                                      [ground_road([(-50, 0), (50, 0)])])
    assert len(probs) == 1 and probs[0].startswith("building w1 (osm): footprint on the corridor of way 1")
    assert len(validate.overlay_features(out)["rings"]) == 1


def test_materials_are_written_once_per_level(tmp_path):
    a = materials.material("br_a", {})
    b = materials.material("br_b", {})
    assert materials.write_materials(tmp_path, "art/shapes/one/main.materials.json", {"br_a": a}) == {"br_a": a}
    written = materials.write_materials(tmp_path, "art/shapes/two/main.materials.json", {"br_a": a, "br_b": b})
    assert written == {"br_b": b}
    assert set(json.loads((tmp_path / "art/shapes/two/main.materials.json").read_text())) == {"br_b"}


def test_styled_level_hook_registered():
    mods = level.FEATURE_MODULES
    assert mods.index("buildings") < mods.index("styled") < mods.index("trees") < mods.index("landmarks")
    assert styled.__doc__
