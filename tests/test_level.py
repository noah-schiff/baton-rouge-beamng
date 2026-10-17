import json
import math
import struct
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from shapely.geometry import box

from brmap import beamng_format as bf
from brmap import level
from brmap.config import Config
from brmap.roads import Road, save_roads
from brmap.water import WaterBody, save_water

FIXTURES = Path(__file__).parent / "fixtures"
# Microsoft's footprints supply most of the buildings and are under the ODbL too
CREDITS = ("Map data © OpenStreetMap contributors (ODbL). Building footprints © Microsoft (ODbL). "
           "Elevation and surface: USGS 3DEP. Imagery: USDA NAIP. Land cover: MRLC NLCD.")


def small_cfg():
    return Config(debug_half_size_m=512.0, terrain_size=256)


def make_inputs(data: Path):
    """Synthetic stage outputs for a 1,024 m map: 256 x 256 grid at 4 m."""
    data.mkdir(parents=True, exist_ok=True)
    h = np.full((256, 256), 6.0, np.float32)
    h[0, :] = 30.0  # a ridge along the north edge, to check the row order on disk
    np.save(data / "heights.npy", h)
    np.save(data / "heights_raw.npy", h)
    layers = np.zeros((256, 256), np.uint8)
    layers[:, 128:] = 2
    np.save(data / "layers.npy", layers)
    img = np.zeros((256, 256, 3), np.uint8)
    img[:128] = (200, 50, 50)  # north half red
    Image.fromarray(img).save(data / "basecolor.png")
    xs = np.linspace(-300, 300, 31)
    ground = Road([1, 2], "residential", np.column_stack([xs, np.zeros(31), np.full(31, 6.0)]),
                  6.5, 2, False, False, 0, "Test Avenue")
    oneway = Road([3], "primary", np.column_stack([np.zeros(11), np.linspace(-200, 200, 11),
                                                   np.full(11, 6.0)]), 7.1, 2, True, False, 0, None)
    deck = Road([4], "motorway", np.column_stack([xs, np.full(31, 100.0), 6 + 4 * np.sin(np.linspace(0, np.pi, 31))]),
                13.8, 3, True, True, 1, "Test Bridge")
    save_roads([ground, oneway, deck], data / "roads.json")
    save_water([WaterBody("Test Lake", box(-400, -400, -250, -250), 5.0, 2.0),
                WaterBody("Long River", box(300, -500, 380, 500), 3.0, 15.0)], data / "water.json")


@pytest.fixture
def level_dir(tmp_path):
    make_inputs(tmp_path / "data" / "out")
    return level.write_level(small_cfg(), out_dir=tmp_path / "dist" / "level",
                             data_dir=tmp_path / "data" / "out")


def items(level_dir):
    """Every object line in every items.level.json, with its file's group folder."""
    out = []
    for f in sorted(level_dir.rglob("items.level.json")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append((f, json.loads(line)))
    return out


def by_class(level_dir, cls):
    return [o for _, o in items(level_dir) if o["class"] == cls]


def test_level_dir_layout(level_dir):
    assert level_dir.as_posix().endswith("levels/baton_rouge")
    for rel in ["info.json", "theTerrain.ter", "theTerrain.terrain.json", "main/items.level.json",
                "main/MissionGroup/items.level.json", "art/terrains/main.materials.json",
                "art/terrains/br_basecolor.png", "art/road/main.materials.json",
                "baton_rouge_preview.jpg"]:
        assert (level_dir / rel).exists(), rel


def test_items_every_line_is_json_with_parent(level_dir):
    objs = items(level_dir)
    groups = {o["name"] for _, o in objs if o["class"] == "SimGroup"}
    assert "MissionGroup" in groups
    for f, o in objs:
        assert "class" in o
        if o.get("name") == "MissionGroup":
            assert "__parent" not in o
            continue
        assert o["__parent"] in groups, o
        # an object lives in the folder named after its parent group
        assert f.parent.name == o["__parent"], (f, o["__parent"])


def test_persistent_ids_unique_and_stable(tmp_path):
    make_inputs(tmp_path / "data" / "out")
    a = level.write_level(small_cfg(), tmp_path / "d1", tmp_path / "data" / "out")
    b = level.write_level(small_cfg(), tmp_path / "d2", tmp_path / "data" / "out")
    ids_a = [o.get("persistentId") for _, o in items(a)]
    assert all(ids_a) and len(set(ids_a)) == len(ids_a)
    assert ids_a == [o.get("persistentId") for _, o in items(b)]


def test_road_count_matches(level_dir):
    roads = [o for _, o in items(level_dir) if o.get("__parent") == "roads"]
    assert len(roads) == 2 and {o["class"] for o in roads} == {"DecalRoad"}
    assert len(by_class(level_dir, "MeshRoad")) == 1


def test_decal_road_fields(level_dir):
    roads = {o["material"]: o for o in by_class(level_dir, "DecalRoad")}
    res = roads[bf.ROAD_MATERIAL["residential"]]
    assert len(res["nodes"]) == 31 and len(res["nodes"][0]) == 4
    assert res["nodes"][0] == [-300.0, 0.0, 6.0, 6.5]
    assert res["position"] == res["nodes"][0][:3]
    assert res["drivability"] > 0 and res["improvedSpline"] is True
    assert roads[bf.ROAD_MATERIAL["primary"]]["oneWay"] is True
    # AI lanes and speed: two-way residential 1+1 at 25 mph, one-way primary 0+2 at 45 mph
    assert (res["autoLanes"], res["lanesLeft"], res["lanesRight"], res["speedLimit"]) == (False, 1, 1, "11.18")
    prim = roads[bf.ROAD_MATERIAL["primary"]]
    assert (prim["lanesLeft"], prim["lanesRight"], prim["speedLimit"]) == (0, 2, "20.12")
    # a covered passage (tunnel=building_passage, covered=yes) runs through a building the
    # buildings stage keeps: still drawn, but out of the AI network (map.lua takes only
    # drivability > 0), so it carries no AI lanes or speed either
    for kind in ("service", "residential"):
        open_road, passage = bf.decal_road(_road(kind), 3), bf.decal_road(_road(kind, covered=True), 3)
        assert open_road["drivability"] > 0 and passage["drivability"] == 0
        assert not {"autoLanes", "lanesLeft", "lanesRight", "speedLimit"} & set(passage)
        assert {k: v for k, v in passage.items() if k != "drivability"} == {
            k: v for k, v in open_road.items()
            if k not in ("drivability", "autoLanes", "lanesLeft", "lanesRight", "speedLimit")}
    covered_deck = bf.bridge_connector(_road("service", is_bridge=True, covered=True), 0)
    assert covered_deck["drivability"] == 0 and "speedLimit" not in covered_deck


def test_mesh_road_fields(level_dir):
    (mr,) = by_class(level_dir, "MeshRoad")
    assert mr["topMaterial"] == bf.BRIDGE_MATERIALS[0]
    assert len(mr["nodes"][0]) == 8
    x, y, z, w, depth, nx, ny, nz = mr["nodes"][0]
    assert (w, depth, nx, ny, nz) == (13.8, bf.DECK_DEPTH, 0, 0, 1)


def test_terrain_block_geometry(level_dir):
    (tb,) = by_class(level_dir, "TerrainBlock")
    # vertices sit on the grid's cell centers: half a 4 m cell in from the -512 m edge
    assert tb["position"] == [-510.0, -510.0, -5.0]
    assert tb["squareSize"] == 4.0 and tb["maxHeight"] == 45.0
    assert tb["terrainFile"] == "/levels/baton_rouge/theTerrain.ter"


def test_ter_rows_run_south_to_north(level_dir):
    data = (level_dir / "theTerrain.ter").read_bytes()
    version, size = data[0], struct.unpack_from("<I", data, 1)[0]
    h = np.frombuffer(data, "<u2", count=size * size, offset=5).reshape(size, size)
    assert (version, size) == (9, 256)
    assert h[-1].min() > h[0].max()  # the north ridge is the LAST row on disk
    lay = np.frombuffer(data, np.uint8, count=size * size, offset=5 + 2 * size * size).reshape(size, size)
    assert (lay[:, 0] == 0).all() and (lay[:, -1] == 2).all()  # west grass, east asphalt


def test_ter_cut_under_a_low_deck(tmp_path):
    from brmap.bridges import TERRAIN_GAP
    from brmap.geo import Extent

    data = tmp_path / "data" / "out"
    make_inputs(data)  # 6 m ground
    xs = np.linspace(-300, 300, 31)
    low = Road([9], "primary", np.column_stack([xs, np.full(31, -100.0), np.full(31, 5.0)]),
               8.0, 2, False, True, 1, "Low Deck")  # 1 m under the ground
    save_roads([low], data / "roads.json")
    cfg = small_cfg()
    root = level.write_level(cfg, tmp_path / "dist", data)
    u = bf.read_ter(root / bf.TERRAIN_FILE)["heights"].astype(float)
    z = cfg.z_min + u / 65535 * (cfg.z_max - cfg.z_min)
    col, row = (round(v) for v in Extent.from_config(cfg).to_pixel(0.0, -100.0, 256))
    assert z[row, col] <= 5.0 - TERRAIN_GAP + 0.01
    assert z[row - 10, col] == pytest.approx(6.0, abs=0.01)  # 40 m north of the deck: untouched
    assert np.load(data / "heights.npy").min() == 6.0      # the stage output is not modified


def test_ter_material_names_match_terrain_materials(level_dir):
    names = bf.read_ter(level_dir / "theTerrain.ter")["materials"]
    mats = json.loads((level_dir / "art/terrains/main.materials.json").read_text(encoding="utf-8"))
    internal = {v["internalName"] for v in mats.values() if v.get("class") == "TerrainMaterial"}
    assert set(names) <= internal
    (tb,) = by_class(level_dir, "TerrainBlock")
    sets = [v for v in mats.values() if v.get("class") == "TerrainMaterialTextureSet"]
    assert [s["name"] for s in sets] == [tb["materialTextureSet"]]
    for v in mats.values():
        if v.get("class") == "TerrainMaterial":
            assert v["baseColorBaseTex"] == "/levels/baton_rouge/art/terrains/br_basecolor.png"
            # one copy of the aerial spans the map; the size is in terrain cells, not meters
            # (seen in game: 16384 on the 4 m grid stretched the aerial four times over)
            assert v["baseColorBaseTexSize"] == 256.0


def test_terrain_base_textures_match_the_texture_set(level_dir):
    """BeamNG drops any base texture whose size differs from the set's baseTexSize."""
    mats = json.loads((level_dir / "art/terrains/main.materials.json").read_text(encoding="utf-8"))
    (tset,) = [v for v in mats.values() if v.get("class") == "TerrainMaterialTextureSet"]
    with Image.open(level_dir / bf.BASECOLOR_FILE) as im:
        assert tset["baseTexSize"] == [im.width, im.height]
    for v in mats.values():
        if v.get("class") != "TerrainMaterial":
            continue
        for key in ("baseColorBaseTex", "normalBaseTex", "roughnessBaseTex", "aoBaseTex", "heightBaseTex"):
            path = v[key]
            assert path.startswith("/levels/baton_rouge/"), path  # nothing borrowed from other levels
            with Image.open(level_dir / path.removeprefix("/levels/baton_rouge/")) as im:
                assert list(im.size) == tset["baseTexSize"], key
            assert v[key + "Size"] == 256.0  # each spans the whole map (in terrain cells)


def test_ter_matches_stock_header_and_table(tmp_path):
    head = (FIXTURES / "stock_template_ter_head.bin").read_bytes()
    tail = (FIXTURES / "stock_template_ter_tail.bin").read_bytes()
    p = tmp_path / "t.ter"
    bf.write_ter(p, np.zeros((1024, 1024), np.uint16), np.zeros((1024, 1024), np.uint8),
                 ["Grass", "BeachSand", "dirt_rocky_large", "rock"])
    data = p.read_bytes()
    assert data[:5] == head[:5]
    assert data[-len(tail):] == tail
    assert len(data) == 3145775  # same length as the stock template's theTerrain.ter


def test_ter_roundtrip(tmp_path):
    u = np.arange(16, dtype=np.uint16).reshape(4, 4) * 1000
    lay = np.arange(16, dtype=np.uint8).reshape(4, 4) % 3
    bf.write_ter(tmp_path / "t.ter", u, lay, ["a", "bb", "ccc"])
    back = bf.read_ter(tmp_path / "t.ter")
    assert back["version"] == 9 and back["size"] == 4 and back["materials"] == ["a", "bb", "ccc"]
    np.testing.assert_array_equal(back["heights"], u)
    np.testing.assert_array_equal(back["layers"], lay)


def test_road_materials_defined(level_dir):
    mats = json.loads((level_dir / "art/road/main.materials.json").read_text(encoding="utf-8"))
    used = {o["material"] for o in by_class(level_dir, "DecalRoad")} - {bf.CONNECTOR_MATERIAL}
    used |= {m for o in by_class(level_dir, "MeshRoad")
             for m in (o["topMaterial"], o["bottomMaterial"], o["sideMaterial"])}
    assert used <= set(mats)
    assert bf.CONNECTOR_MATERIAL not in mats  # a global game material, not redefined here
    for v in mats.values():
        assert v["class"] == "Material" and v["Stages"][0]["baseColorMap"].startswith("/assets/")


def test_water_blocks_cover_bodies(level_dir):
    blocks = by_class(level_dir, "WaterBlock")
    lake = [b for b in blocks if b["position"][2] == 5.0]
    assert len(lake) == 1
    (cx, cy, z), (sx, sy, sz) = lake[0]["position"], lake[0]["scale"]
    assert cx - sx / 2 <= -400 and cx + sx / 2 >= -250 and sz >= 2.0
    river = [b for b in blocks if b["position"][2] == 3.0]
    assert len(river) >= 2  # a 1 km long river is split into strips that follow it


def test_spawns_inside_box(level_dir):
    spawns = by_class(level_dir, "SpawnSphere")
    assert spawns, "at least one spawn point"
    for s in spawns:
        x, y, z = s["position"]
        assert abs(x) <= 512 and abs(y) <= 512 and s["dataBlock"] == "SpawnSphereMarker"


def test_center_spawn_lands_on_the_nearest_road():
    from brmap.geo import Extent

    cfg = small_cfg()  # none of the configured spawns is inside this 1 km box
    xs = np.linspace(-300, 300, 31)
    side = Road([1], "residential", np.column_stack([xs, np.full(31, 40.0), np.full(31, 6.0)]),
                6.5, 2, False, False, 0, "Side Street")
    far = Road([2], "residential", np.column_stack([xs, np.full(31, -90.0), np.full(31, 6.0)]),
               6.5, 2, False, False, 0, "Far Street")
    (s,) = level.place_spawns(cfg, Extent.from_config(cfg), np.full((256, 256), 6.0, np.float32), [far, side])
    assert s["name"] == "spawn_center" and (s["x"], s["y"]) == (0.0, 40.0)
    assert s["z"] == pytest.approx(6.5)


def test_spawn_snaps_onto_the_nearest_deck_centerline():
    from brmap.config import SpawnPoint
    from brmap.geo import Extent

    base = small_cfg()
    extent = Extent.from_config(base)
    lon, lat = extent.to_lonlat(5.0, 0.0)
    cfg = Config(debug_half_size_m=512.0, terrain_size=256, spawns=[SpawnPoint("s", "On the deck", lat, lon)])
    xs = np.linspace(-200, 200, 21)  # nodes 20 m apart, 20 m north of the spawn, heading east
    deck = Road([7], "motorway", np.column_stack([xs, np.full(21, 20.0), np.full(21, 30.0)]),
                14.0, 3, True, True, 1, "Deck")
    (s,) = level.place_spawns(cfg, extent, np.full((256, 256), 6.0, np.float32), [deck])
    assert (s["x"], s["y"], s["z"]) == (pytest.approx(5.0), pytest.approx(20.0), pytest.approx(31.0))
    assert s["heading"] == pytest.approx(0.0)  # along the deck, the way traffic runs
    lon, lat = extent.to_lonlat(5.0, -40.0)  # 60 m away: stays on the ground
    cfg = Config(debug_half_size_m=512.0, terrain_size=256, spawns=[SpawnPoint("g", "Ground", lat, lon)])
    (g,) = level.place_spawns(cfg, extent, np.full((256, 256), 6.0, np.float32), [deck])
    assert g["z"] == pytest.approx(6.5) and g["heading"] is None
    assert math.isclose(g["y"], -40.0, abs_tol=1e-6)


def test_spawn_sphere_heading():
    # Measured in game (0.39.4): an unrotated sphere spawns the vehicle facing south, and
    # rotationMatrix [c, -s, 0, s, c, 0, 0, 0, 1] turns it to face 270 degrees - a.
    south = bf.spawn_sphere("a", 0, 0, 0, heading=3 * math.pi / 2)
    assert south["rotationMatrix"] == pytest.approx([1, 0, 0, 0, 1, 0, 0, 0, 1], abs=1e-9)
    north = bf.spawn_sphere("b", 0, 0, 0, heading=math.pi / 2)
    assert north["rotationMatrix"] == pytest.approx([-1, 0, 0, 0, -1, 0, 0, 0, 1], abs=1e-9)
    west = bf.spawn_sphere("c", 0, 0, 0, heading=math.radians(177.1))  # the I-10 bridge spawn
    c, s = west["rotationMatrix"][0], west["rotationMatrix"][3]
    assert math.degrees(math.atan2(s, c)) == pytest.approx(92.9, abs=1e-3)  # stored to 6 decimals
    assert "rotationMatrix" not in bf.spawn_sphere("d", 0, 0, 0)


def test_info_json(level_dir):
    info = json.loads((level_dir / "info.json").read_text(encoding="utf-8"))
    names = {s["name"] for s in by_class(level_dir, "SpawnSphere")}
    assert info["defaultSpawnPointName"] in names
    assert {p["objectname"] for p in info["spawnPoints"]} == names
    assert level.CREDITS == CREDITS and info["description"].endswith(" " + CREDITS)
    assert info["size"] == [1024, 1024] and info["previews"] == ["baton_rouge_preview.jpg"]
    assert info["supportsTraffic"] is True and info["roadRules"] == {"turnOnRed": True}
    assert "\u2014" not in json.dumps(info, ensure_ascii=False)


def test_full_size_spawns_from_config(tmp_path):
    cfg = Config(terrain_size=256)  # the real 16 km box: the three spawn points from the spec
    make_inputs(tmp_path / "data" / "out")
    d = level.write_level(cfg, tmp_path / "dist", tmp_path / "data" / "out")
    info = json.loads((d / "info.json").read_text(encoding="utf-8"))
    assert info["defaultSpawnPointName"] == "spawn_lsu"
    assert [p["objectname"] for p in info["spawnPoints"]] == ["spawn_lsu", "spawn_capitol",
                                                               "spawn_i10_bridge"]


def test_env_objects(level_dir):
    classes = {o["class"] for o in by_class(level_dir, "LevelInfo") + by_class(level_dir, "ScatterSky")
               + by_class(level_dir, "TimeOfDay") + by_class(level_dir, "CloudLayer")}
    assert classes == {"LevelInfo", "ScatterSky", "TimeOfDay", "CloudLayer"}


def test_preview(level_dir):
    with Image.open(level_dir / "baton_rouge_preview.jpg") as im:
        assert im.size == (1024, 576) and im.format == "JPEG"


def test_zip_root_is_levels(level_dir, tmp_path):
    z = level.zip_mod(level_dir.parents[1], tmp_path / "out.zip", "baton_rouge")
    with zipfile.ZipFile(z) as zf:
        names = zf.namelist()
    assert names and all(n.startswith("levels/baton_rouge/") for n in names)
    assert "levels/baton_rouge/info.json" in names


def test_zip_leaves_out_other_levels_in_the_stage_folder(level_dir, tmp_path):
    stale = level_dir.parent / "old_name"  # left over from a build under another level name
    stale.mkdir()
    (stale / "info.json").write_text("{}", encoding="utf-8")
    z = level.zip_mod(level_dir.parents[1], tmp_path / "out.zip", "baton_rouge")
    with zipfile.ZipFile(z) as zf:
        names = zf.namelist()
    assert "levels/baton_rouge/info.json" in names
    assert not any(n.startswith("levels/old_name/") for n in names)


def _road(kind, lanes=2, oneway=False, maxspeed=None, is_bridge=False, covered=False):
    xs = np.linspace(0, 100, 6)
    return Road([11, 12], kind, np.column_stack([xs, np.zeros(6), 2 + xs / 10]), 7.0, lanes, oneway,
                is_bridge, 1 if is_bridge else 0, None, maxspeed=maxspeed, covered=covered)


def test_render_priority_major_roads_on_top():
    """A lower renderPriority draws on top (checked in game), so major roads get the low numbers."""
    def p(kind):
        return bf.decal_road(_road(kind), 0)["renderPriority"]

    order = ["motorway", "trunk", "primary", "secondary", "tertiary", "residential", "service"]
    assert [p(k) for k in order] == sorted(p(k) for k in order)
    assert len({p(k) for k in order}) == len(order)
    assert (p("motorway"), p("trunk"), p("primary"), p("secondary"), p("tertiary")) == (6, 7, 8, 9, 10)
    assert p("unclassified") == p("residential") == p("living_street") == 11 and p("service") == 12
    assert p("motorway_link") == p("motorway") and p("tertiary_link") == p("tertiary")
    assert p("track") == 12  # anything unlisted draws under the rest


def test_lane_fields_oneway_and_twoway():
    f = bf.lane_fields(_road("motorway", lanes=3, oneway=True))
    assert (f["autoLanes"], f["lanesLeft"], f["lanesRight"]) == (False, 0, 3)
    assert f["speedLimit"] == "29.06"  # 65 mph
    four = bf.lane_fields(_road("primary", lanes=4))
    assert (four["lanesLeft"], four["lanesRight"]) == (2, 2)
    one = bf.lane_fields(_road("residential", lanes=1))
    assert (one["lanesLeft"], one["lanesRight"], one["speedLimit"]) == (1, 1, "11.18")
    assert bf.lane_fields(_road("primary_link", lanes=1, oneway=True))["speedLimit"] == "20.12"
    assert bf.lane_fields(_road("tertiary", maxspeed=13.8889))["speedLimit"] == "13.89"
    speeds = {k: float(bf.lane_fields(_road(k))["speedLimit"])
              for k in ("motorway", "trunk", "primary", "secondary", "tertiary", "service")}
    assert speeds == pytest.approx({"motorway": 29.06, "trunk": 24.59, "primary": 20.12,
                                    "secondary": 17.88, "tertiary": 15.65, "service": 11.18}, abs=0.006)
    dr = bf.decal_road(_road("motorway", lanes=3, oneway=True), 0)
    assert {k: dr[k] for k in f} == f  # decal roads carry the lane fields


def test_bridge_connector_fields():
    deck = _road("motorway", lanes=3, oneway=True, is_bridge=True)
    c = bf.bridge_connector(deck, 5)
    m = bf.mesh_road(deck, 5)
    assert c["class"] == "DecalRoad" and c["material"] == bf.CONNECTOR_MATERIAL == "road_invisible"
    assert c["overObjects"] is True and c["drivability"] > 0 and c["oneWay"] is True
    assert c["drivability"] == bf.decal_road(_road("motorway"), 0)["drivability"]
    assert c["renderPriority"] == 30
    assert (c["autoLanes"], c["lanesLeft"], c["lanesRight"]) == (False, 0, 3)
    assert len(c["nodes"]) == len(m["nodes"])
    assert [n[:4] for n in c["nodes"]] == [n[:4] for n in m["nodes"]]  # same x, y, z and width
    assert c["position"] == c["nodes"][0][:3]
    assert c["persistentId"] == bf.persistent_id("conn", 5, 11, 12) != m["persistentId"]
    assert bf.bridge_connector(_road("service", is_bridge=True), 0)["drivability"] > 0
    # lua/ge/map.lua builds the AI path from the engine's edge (subdivision) positions unless
    # useSubdivisions is false; projected onto a 60 m high deck those miss ("centerline to
    # edge distance exceeding 20m"), so the connector gives its exact node positions and
    # widths instead, as WCUSA's 78 invisible overObjects decals do
    assert c["useSubdivisions"] is False


def test_level_writes_connectors_and_traffic_info(level_dir):
    groups = [o["name"] for _, o in items(level_dir) if o["class"] == "SimGroup"]
    assert bf.GROUPS["connectors"] == "ai_bridges" and "ai_bridges" in groups
    conns = [o for _, o in items(level_dir) if o.get("__parent") == "ai_bridges"]
    (deck,) = by_class(level_dir, "MeshRoad")
    assert len(conns) == 1  # one per bridge deck
    (c,) = conns
    assert c["class"] == "DecalRoad" and c["material"] == "road_invisible" and c["overObjects"] is True
    assert [n[:3] for n in c["nodes"]] == [n[:3] for n in deck["nodes"]]
    assert all(o["useSubdivisions"] is False for o in conns)  # the AI follows the exact nodes
    # the ground roads keep the engine's subdivisions (the field's default)
    assert all("useSubdivisions" not in o for _, o in items(level_dir) if o.get("__parent") == "roads")
    info = json.loads((level_dir / "info.json").read_text(encoding="utf-8"))
    assert info["supportsTraffic"] is True and info["roadRules"] == {"turnOnRed": True}


def test_traffic_can_be_turned_off(tmp_path):
    make_inputs(tmp_path / "data" / "out")
    cfg = Config(debug_half_size_m=512.0, terrain_size=256, traffic_enabled=False)
    root = level.write_level(cfg, tmp_path / "dist", tmp_path / "data" / "out")
    assert json.loads((root / "info.json").read_text(encoding="utf-8"))["supportsTraffic"] is False


def _fake_feature(monkeypatch, name, content):
    """Stand in a module brmap.<name> whose level_content records its calls and returns content."""
    calls = []

    def level_content(cfg, data_dir, root, heights, extent):
        calls.append({"cfg": cfg, "data_dir": Path(data_dir), "root": Path(root),
                      "heights": heights.copy(), "extent": extent})
        folder = Path(root) / "art" / "shapes" / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "tile.dae").write_text("<COLLADA/>", encoding="utf-8")
        return content

    mod = types.ModuleType(f"brmap.{name}")
    mod.level_content = level_content
    monkeypatch.setitem(sys.modules, f"brmap.{name}", mod)
    return calls


def test_level_feature_hook(tmp_path, monkeypatch):
    from brmap.bridges import clear_under_decks
    from brmap.geo import Extent
    from brmap.roads import load_roads

    assert level.FEATURE_MODULES == ("buildings", "trees", "markings")
    data = tmp_path / "data" / "out"
    make_inputs(data)
    xs = np.linspace(-300, 300, 31)
    low = Road([9], "primary", np.column_stack([xs, np.full(31, -100.0), np.full(31, 5.0)]),
               8.0, 2, False, True, 1, "Low Deck")  # 1 m under the 6 m ground: the level cuts there
    save_roads(load_roads(data / "roads.json") + [low], data / "roads.json")
    obj = {"class": "TSStatic", "persistentId": bf.persistent_id("bld", "0_0"),
           "shapeName": "/levels/baton_rouge/art/shapes/buildings/tile.dae", "position": [0, 0, 0]}
    calls = _fake_feature(monkeypatch, "buildings", {"buildings": [obj]})

    root = level.write_level(small_cfg(), tmp_path / "d0", data)  # no out/buildings.json: not called
    assert calls == []
    assert not (root / "main" / "MissionGroup" / "buildings").exists()

    (data / "buildings.json").write_text("[]", encoding="utf-8")
    cfg = small_cfg()
    root = level.write_level(cfg, tmp_path / "d1", data)
    (call,) = calls
    assert call["cfg"] is cfg and call["data_dir"] == data and call["root"] == root
    extent = Extent.from_config(cfg)
    assert call["extent"].half_size == extent.half_size
    raw = np.load(data / "heights.npy")
    expected, cuts = clear_under_decks(raw, extent, load_roads(data / "roads.json"))
    assert cuts and (expected != raw).any()
    np.testing.assert_array_equal(call["heights"], expected)  # the terrain after the deck cuts

    listing = [json.loads(line) for line in
               (root / "main/MissionGroup/items.level.json").read_text(encoding="utf-8").splitlines()]
    assert {"name": "buildings", "class": "SimGroup", "__parent": "MissionGroup",
            "persistentId": bf.persistent_id("group", "baton_rouge", "buildings")} in listing
    lines = (root / "main/MissionGroup/buildings/items.level.json").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [
        {"class": "TSStatic", "persistentId": obj["persistentId"], "__parent": "buildings",
         "shapeName": obj["shapeName"], "position": [0, 0, 0]}]
    assert (root / "art/shapes/buildings/tile.dae").exists()  # its own files go in the level folder


def test_level_ships_collected_compiled_meshes(tmp_path, monkeypatch):
    # a .cdae the game compiled from this exact .dae (out/cdae, see brmap.meshcache) goes
    # beside it, so the game doesn't compile 3,000 tiles on every first load
    import hashlib
    data = tmp_path / "data" / "out"
    make_inputs(data)
    (data / "buildings.json").write_text("[]", encoding="utf-8")
    _fake_feature(monkeypatch, "buildings", {"buildings": []})  # writes art/shapes/buildings/tile.dae
    cache = data / "cdae" / "art" / "shapes" / "buildings"
    cache.mkdir(parents=True)
    (cache / "tile.cdae").write_bytes(b"compiled")
    (data / "cdae" / "manifest.json").write_text(json.dumps(
        {"art/shapes/buildings/tile.dae": hashlib.sha1(b"<COLLADA/>").hexdigest()}), encoding="utf-8")
    root = level.write_level(small_cfg(), tmp_path / "stage", data)
    assert (root / "art/shapes/buildings/tile.cdae").read_bytes() == b"compiled"
    z = level.zip_mod(tmp_path / "stage", tmp_path / "m.zip", "baton_rouge")
    names = zipfile.ZipFile(z).namelist()
    assert "levels/baton_rouge/art/shapes/buildings/tile.cdae" in names
    assert "levels/baton_rouge/art/shapes/buildings/tile.dae" in names


def test_level_feature_hook_rejects_bad_groups(tmp_path, monkeypatch):
    data = tmp_path / "data" / "out"
    make_inputs(data)
    (data / "markings.json").write_text("[]", encoding="utf-8")
    _fake_feature(monkeypatch, "markings", {"roads": [{"class": "DecalRoad", "persistentId": "x"}]})
    with pytest.raises(ValueError, match="roads"):  # would clash with a built-in group
        level.write_level(small_cfg(), tmp_path / "d0", data)
    _fake_feature(monkeypatch, "markings", {"markings": [{"class": "DecalRoad"}]})
    with pytest.raises(ValueError, match="persistentId"):
        level.write_level(small_cfg(), tmp_path / "d1", data)
