import importlib.util
import json
import math
import re
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Polygon, box

from brmap import beamng_format as bf
from brmap import canopy, level, materials, trees, validate
from brmap.bayous import save_bayous
from brmap.buildings import Building, save_buildings
from brmap.config import Config
from brmap.geo import Extent
from brmap.roads import Road, save_roads
from brmap.terrain import sample_height
from brmap.water import WaterBody, save_water

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "src" / "brmap" / "data"
SWAMP_MIX = {"br_cypress_a", "br_cypress_b", "br_tupelo_a", "br_tree_oak_large_c"}


def small_cfg(tmp_path=None, **kw):
    if tmp_path is not None:
        kw.update(raw_dir=str(tmp_path / "raw"), out_dir=str(tmp_path / "out"))
    return Config(debug_half_size_m=512.0, terrain_size=256, **kw)


def species_items(context, swamp=True):
    return {item for item, _, _ in trees.species(swamp)[context]}


def grid_tops(n, h=18.0, spacing=10.0, around=None, spread=None):
    """n tree tops of height h on a square grid, spacing m apart from (0, 0) up; with around and
    spread, the grid fills the square spread m each side of around instead."""
    side = math.ceil(math.sqrt(n))
    x0, y0 = 0.0, 0.0
    if spread is not None:
        spacing = 2 * spread / max(side - 1, 1)
        x0, y0 = around[0] - spread, around[1] - spread
    return np.array([[x0 + (i % side) * spacing, y0 + (i // side) * spacing, h] for i in range(n)], float)


def tops_at(points):
    """Tree tops at these (x, y, height)."""
    return np.array(points, float).reshape(-1, 3)


def contract_npz(path, nodes, materials=("br_tree_cypress_bark", "br_tree_cypress_leaves"), billboard=""):
    """A shape part as Blender writes it (the npz contract): one triangle per material per node, none
    in an empty nulldetail node."""
    k = len(materials)
    pos = np.array([[2.0 * i, 0.0, 0.0] for i in range(k) for _ in range(3)]) + np.tile(
        [[0.0, 0, 0], [1, 0, 0], [0, 1, 1]], (k, 1))
    arrays = {"meta/nodes": np.array(list(nodes)), "meta/billboard": np.array(billboard),
              "meta/blender": np.array("test"), "meta/script_sha1": np.array("0" * 40)}
    for name in nodes:
        if name.startswith("nulldetail"):
            continue
        arrays |= {f"{name}/pos": pos, f"{name}/nrm": np.tile([0.0, 0.0, 1.0], (3 * k, 1)),
                   f"{name}/uv": pos[:, :2], f"{name}/col": np.ones((3 * k, 4)),
                   f"{name}/mat": np.arange(k, dtype=np.int32), f"{name}/materials": np.array(list(materials))}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


CYPRESS_NODES = ("br_cypress_a_a600", "br_cypress_a_a250", "br_cypress_a_a80", "Colmesh-1")
EXT = Extent(-91.160, 30.430, 512.0)   # the small_extent box
FLAT = np.full((256, 256), 6.0, np.float32)


def bumps(extent, crowns, sigma=3.0):
    """ndsm on the 2 m analysis grid: a gaussian crown of height h at each (x, y, h)."""
    n, _ = canopy.analysis_grid(extent)
    xs, ys = extent.pixel_centers(n)
    X, Y = np.meshgrid(xs, ys)
    ndsm = np.zeros((n, n), np.float32)
    for x, y, h in crowns:
        ndsm += (h * np.exp(-((X - x) ** 2 + (Y - y) ** 2) / (2 * sigma ** 2))).astype(np.float32)
    return ndsm


def house(x0, y0, x1, y1):
    return Building(id="w1", source="osm", kind="house",
                    ring=np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float), holes=[], height=6.0,
                    base_z=5.0, roof="flat", wall="res", colour=(1.0, 1.0, 1.0), height_source="default")


def street(y=0.0):
    xs = np.linspace(-400, 400, 41)
    return Road([1], "residential", np.column_stack([xs, np.full(41, y), np.full(41, 6.0)]),
                6.5, 2, False, False, 0, "Test Street")


def deck(y=-200.0):
    xs = np.linspace(-300, 300, 31)
    return Road([2], "motorway", np.column_stack([xs, np.full(31, y), np.full(31, 12.0)]),
                13.8, 3, True, True, 1, "Test Bridge")


# ------------------------------------------------------------------ stock tree data

def test_stock_trees_json():
    data = trees.stock_trees()
    # extracted from 0.40.0.0 (the whole of integrity.json's version), or a later game on re-extraction
    assert re.fullmatch(r"\d+(\.\d+){3}", data["game_version"])
    assert tuple(map(int, data["game_version"].split("."))) >= (0, 40, 0, 0)
    assert len(data["items"]) == 14
    for key, it in data["items"].items():
        assert key.startswith("br_")
        assert it["shapeFile"].startswith("/levels/") and it["shapeFile"].endswith(".dae")
        assert 3 <= it["native_height"] <= 45, key
        assert not {"name", "internalName", "persistentId", "class", "shapeFile"} & set(it["item"])
        assert it["materials"] and set(it["materials"]) <= set(data["materials"])
        col = it["collision"]  # the stock collision mesh's reach from the trunk, per height band
        assert col["dz"] == 0.5 and col["r"] and all(0 <= r <= 10 for r in col["r"]), key
        assert col["z0"] <= 0.0 and col["r"][int(-col["z0"] / col["dz"])] > 0, key  # from the trunk's base
    paths = [p for m in data["materials"].values() for st in m["Stages"] for k, p in st.items()
             if isinstance(p, str) and k.endswith("Map")]
    assert paths and all(p.startswith("/") for p in paths)
    for context in ("evergreen", "mixed", "wetland", "developed", "other", "small"):
        assert trees.SPECIES[context] and species_items(context, swamp=False) <= set(data["items"])
        assert species_items(context) <= set(trees.tree_items())


# ------------------------------------------------------------------ custom trees (Blender models)

def test_tree_items_include_custom():
    items = trees.tree_items("baton_rouge")
    c = items["br_cypress_a"]
    assert c["shapeFile"] == "/levels/baton_rouge/art/shapes/trees/br_swamp/br_cypress_a.dae"
    assert c["native_height"] == 22 and "br_holm_oak" in items


def test_tree_items_follow_the_level_name_and_keep_the_stock_items():
    stock = trees.stock_trees()["items"]
    items = trees.tree_items("baton_rouge_lsu")
    assert items["br_cypress_a"]["shapeFile"] == "/levels/baton_rouge_lsu/art/shapes/trees/br_swamp/br_cypress_a.dae"
    assert {k: items[k] for k in stock} == stock
    assert set(items) - set(stock) == set(trees.custom_trees())
    c = items["br_cypress_a"]
    assert c["materials"] == ["br_tree_cypress_bark", "br_tree_cypress_leaves"]
    assert c["item"] == stock["br_holm_oak"]["item"]       # the stock holm oak's item fields
    # the trunk prism (0.7 m to 8 m) in the stock collision form: 0.5 m bands from the origin up
    col = c["collision"]
    assert (col["z0"], col["dz"]) == (0.0, 0.5) and len(col["r"]) == 44   # 22 m tall
    assert col["r"][:16] == [0.7] * 16 and col["r"][16:] == [0.0] * 28


def test_custom_tree_registry():
    stock = trees.stock_trees()["items"]
    tree_mats = materials.load_registry(trees.TREE_MATERIALS)
    for key, e in trees.custom_trees().items():
        # a key may end in a digit (br_moss_cork_oak_large_1): its nodes end in _a<px> all the same
        assert re.fullmatch(r"br_[a-z0-9_]*[a-z0-9]", key) and key not in stock, key
        assert (ROOT / e["script"]).is_file(), key
        assert set(e) == {"script", "params", "native_height", "collision", "item", "materials", "billboard"}, key
        assert set(e["collision"]) == {"trunk_radius_m", "height_m"} and isinstance(e["billboard"], bool), key
        assert not {"name", "internalName", "persistentId", "class", "shapeFile"} & set(e["item"]), key
        assert e["materials"] and set(e["materials"]) <= set(tree_mats), key
    cypress = trees.custom_trees()["br_cypress_a"]
    assert cypress["script"] == "blender/trees/bald_cypress.py"
    assert cypress["params"] == {"height_m": 22, "seed": 11, "knees": 8}
    assert cypress["billboard"] is True and cypress["native_height"] == cypress["params"]["height_m"]


def test_swamp_tree_registry():
    custom = trees.custom_trees()
    b = custom["br_cypress_b"]       # the cypress script again, hung with moss
    assert b["script"] == "blender/trees/bald_cypress.py"
    assert {k: b["params"][k] for k in ("moss", "seed", "height_m")} == {"moss": True, "seed": 12, "height_m": 20}
    assert b["native_height"] == 20 and b["billboard"] is True
    assert b["materials"] == ["br_tree_cypress_bark", "br_tree_cypress_leaves", "br_tree_moss"]
    assert b["item"] == custom["br_cypress_a"]["item"] and b["collision"] == custom["br_cypress_a"]["collision"]
    t = custom["br_tupelo_a"]
    assert t["script"] == "blender/trees/water_tupelo.py" and t["native_height"] == t["params"]["height_m"] == 18
    assert t["billboard"] is True and t["materials"] == ["br_tree_tupelo_bark", "br_tree_tupelo_leaves"]


def test_moss_overlay_registry():
    custom, stock = trees.custom_trees(), trees.stock_trees()["items"]
    assert trees.LIVE_OAKS == ("br_oak_dry_a", "br_cork_oak_large_1", "br_tree_oak_large_b", "br_holm_oak")
    for oak in trees.LIVE_OAKS:
        key = "br_moss_" + oak[3:]
        e = custom[key]
        assert trees.moss_item(oak) == key
        assert e["script"] == "blender/trees/moss_drape.py" and e["params"] == {"hang_points_from": oak}, key
        assert e["materials"] == ["br_tree_moss"] and e["billboard"] is False, key
        assert e["collision"] == {"trunk_radius_m": 0, "height_m": 0}, key           # no collision mesh
        assert e["native_height"] == stock[oak]["native_height"], key              # drawn at the oak's scale
        assert e["item"] == stock[oak]["item"], key                                 # it bends with the oak
        assert trees.collision_reach(key, 1.0) == 0.0


def test_swamp_tree_scale_limits():
    # short wetland trees keep their lidar height down to 6.6 m (0.3 x the 22 m cypress), as do the others
    assert trees.scale_limits("br_cypress_a") == (0.3, 1.6)
    items = trees.tree_items()
    for key in ("br_cypress_b", "br_tupelo_a"):
        lo, hi = trees.scale_limits(key)
        assert lo * items[key]["native_height"] == pytest.approx(6.6, abs=0.01), key
        assert hi * items[key]["native_height"] == pytest.approx(35.2, abs=0.02), key
    placed = trees.assign(tops_at([(0.0, 0.0, 7.0)]), np.array([90]))[0]
    assert placed["scale"] * items[placed["item"]]["native_height"] == pytest.approx(7.0, abs=0.05)


def test_swamp_materials():
    reg = materials.load_registry(trees.TREE_MATERIALS)
    index = {"moss": {"files": {"b": "moss_b.color.dds", "nm": "moss_nm.normal.dds", "o": "moss_o.data.dds"}}}
    mats, files = materials.resolve(reg, ["br_tree_moss", "br_tree_tupelo_bark", "br_tree_tupelo_leaves"],
                                    "baton_rouge", index)
    moss = mats["br_tree_moss"]
    assert {k: v for k, v in moss["Stages"][0].items() if k.endswith("Map")} == {
        "baseColorMap": "/levels/baton_rouge/art/textures/br/moss_b.color.png",
        "normalMap": "/levels/baton_rouge/art/textures/br/moss_nm.normal.png",
        "opacityMap": "/levels/baton_rouge/art/textures/br/moss_o.data.png"}
    assert {k: moss[k] for k in ("alphaTest", "alphaRef", "doubleSided", "invertBackFaceNormals", "subSurface",
                                 "subSurfaceIntensity", "groundType")} == {
        "alphaTest": True, "alphaRef": 67, "doubleSided": True, "invertBackFaceNormals": True, "subSurface": True,
        "subSurfaceIntensity": 0.05, "groundType": "LEAVES_THIN"}
    assert sorted(f.name for f in files) == ["moss_b.color.dds", "moss_nm.normal.dds", "moss_o.data.dds"]
    poplar = "/assets/materials/tree/poplar/t_poplar_bark/t_poplar_bark_"
    assert {k: v for k, v in mats["br_tree_tupelo_bark"]["Stages"][0].items() if k.endswith("Map")} == {
        "baseColorMap": poplar + "b.color.png", "normalMap": poplar + "nm.normal.png",
        "roughnessMap": poplar + "r.data.png", "ambientOcclusionMap": poplar + "ao.data.png"}
    leaves = mats["br_tree_tupelo_leaves"]
    assert leaves["Stages"][0]["opacityMap"] == "/assets/materials/tree/beech/t_beech_leaves/t_beech_leaves_o.data.png"
    assert leaves["Stages"][0]["baseColorFactor"] != [1, 1, 1, 1]          # tinted
    assert leaves["alphaTest"] and leaves["doubleSided"] and leaves["groundType"] == "LEAVES_THIN"
    for m in mats.values():
        assert not {"vertColor", "instanceDiffuse"} & (set(m) | set(m["Stages"][0]))


def test_cypress_materials():
    reg = materials.load_registry(trees.TREE_MATERIALS)
    assert all(name.startswith("br_") for name in reg)
    mats, files = materials.resolve(reg, ["br_tree_cypress_bark", "br_tree_cypress_leaves"], "baton_rouge", {})
    assert files == []                                          # stock textures only
    bark, leaves = mats["br_tree_cypress_bark"], mats["br_tree_cypress_leaves"]
    fir = "/assets/materials/tree/fir/t_fir_bark/t_fir_bark_"   # the stock fir_bark's maps
    assert {k: v for k, v in bark["Stages"][0].items() if k.endswith("Map")} == {
        "baseColorMap": fir + "b.color.png", "normalMap": fir + "nm.normal.png",
        "roughnessMap": fir + "r.data.png", "ambientOcclusionMap": fir + "ao.data.png"}
    leaf = "/assets/materials/tree/cypress/t_cypress_leaves/t_cypress_leaves_"
    stage = leaves["Stages"][0]
    assert {k: v for k, v in stage.items() if k.endswith("Map")} == {
        "baseColorMap": leaf + "b.color.png", "normalMap": leaf + "nm.normal.png",
        "roughnessMap": leaf + "r.data.png", "opacityMap": leaf + "o.data.png",
        "ambientOcclusionMap": leaf + "ao.data.png"}
    assert stage["baseColorFactor"] == [0.78, 0.92, 0.62, 1]
    assert {k: leaves[k] for k in ("alphaTest", "alphaRef", "doubleSided", "invertBackFaceNormals", "subSurface",
                                   "subSurfaceIntensity", "groundType")} == {
        "alphaTest": True, "alphaRef": 90, "doubleSided": True, "invertBackFaceNormals": True,
        "subSurface": True, "subSurfaceIntensity": 1, "groundType": "LEAVES_THIN"}
    # forest leaves leave the vertex colours to the wind: no vertColor, no instanceDiffuse
    for m in mats.values():
        assert not {"vertColor", "instanceDiffuse"} & (set(m) | set(m["Stages"][0]))


def test_species_swamp_switch():
    assert trees.species(False) == trees.SPECIES and "bank" not in trees.SPECIES
    swamp = trees.species(True)
    assert swamp["wetland"] == [("br_cypress_a", .35, None), ("br_cypress_b", .15, None), ("br_tupelo_a", .35, None),
                                ("br_tree_oak_large_c", .15, None)]
    assert swamp["bank"] == [("br_cypress_a", .30, None), ("br_cypress_b", .15, None), ("br_tupelo_a", .30, None),
                             ("br_tree_oak_large_c", .25, None)]
    assert {k: v for k, v in swamp.items() if k not in ("wetland", "bank")} == {
        k: v for k, v in trees.SPECIES.items() if k != "wetland"}
    assert trees.SPECIES["wetland"] == [("br_redwood_broad", 0.6, None), ("br_tree_oak_large_c", 0.4, None)]
    assert trees.BANK_DIST == 30.0


def test_wetland_becomes_cypress():
    placed = trees.assign(grid_tops(400, h=18), lambda x, y: np.full(len(x), 90), swamp=True)
    assert {t["item"] for t in placed} == SWAMP_MIX


def test_bank_context():
    placed = trees.assign(tops_at([(0, 10, 15), (500, 500, 15)]), lambda x, y: np.full(len(x), 22),
                          swamp=True, bayou_lines=[LineString([(-100, 0), (100, 0)])])
    assert placed[0]["item"] in SWAMP_MIX


def test_bank_context_spares_small_trees_and_the_rest_of_town():
    # a whole developed grid beside a bayou: the trees within 30 m of it take the bank mix, the rest the
    # street mix; a small (under 8 m) street tree stays a small one even on the bank
    line = LineString([(-1000, 0), (1000, 0)])
    tops = grid_tops(400, h=15, around=(0, 0), spread=95)
    placed = trees.assign(tops, lambda x, y: np.full(len(x), 22), swamp=True, bayou_lines=[line])
    near = np.abs(tops[:, 1]) <= trees.BANK_DIST
    assert near.any() and (~near).any()
    assert {t["item"] for t, n in zip(placed, near) if n} == SWAMP_MIX
    assert not {t["item"] for t, n in zip(placed, near) if not n} & {"br_cypress_a", "br_cypress_b", "br_tupelo_a"}
    small = trees.assign(tops_at([(0, 5, 6)]), np.array([22]), swamp=True, bayou_lines=[line])[0]
    assert small["item"] in species_items("small")
    # with the swamp trees off there is no bank mix: v0.1's trees
    v01 = trees.assign(tops, lambda x, y: np.full(len(x), 22), swamp=False, bayou_lines=[line])
    assert v01 == trees.assign(tops, lambda x, y: np.full(len(x), 22), swamp=False)


def test_osm_wetland_context():
    placed = trees.assign(grid_tops(200, h=15, around=(0, 0), spread=40), lambda x, y: np.full(len(x), 22),
                          swamp=True, wetland_polys=[box(-50, -50, 50, 50)])
    assert {t["item"] for t in placed} <= SWAMP_MIX


def test_osm_wetland_takes_small_trees_too():
    # a 6 m tree in an OSM swamp is a swamp tree whatever NLCD says (the small mix is for town)
    placed = trees.assign(tops_at([(0, 0, 6), (200, 0, 6)]), np.array([22, 22]), swamp=True,
                          wetland_polys=[box(-50, -50, 50, 50)])
    assert placed[0]["item"] in SWAMP_MIX and placed[1]["item"] in species_items("small")


def test_moss_share_is_deterministic():
    tops = grid_tops(10000, h=14)
    a = trees.assign(tops, lambda x, y: np.full(len(x), 22), swamp=True, moss_share=0.30)
    b = trees.assign(tops, lambda x, y: np.full(len(x), 22), swamp=True, moss_share=0.30)
    oaks = [t for t in a if t["item"] in trees.LIVE_OAKS]
    assert a == b and 0.28 <= sum(t.get("moss", False) for t in oaks) / len(oaks) <= 0.32


def test_moss_only_on_live_oaks_in_town():
    tops = grid_tops(2000, h=14)
    town = trees.assign(tops, np.full(len(tops), 22), moss_share=1.0)
    assert all(t.get("moss") is True for t in town if t["item"] in trees.LIVE_OAKS)
    assert not any("moss" in t for t in town if t["item"] not in trees.LIVE_OAKS)
    mixed = trees.assign(tops, np.full(len(tops), 43), moss_share=1.0)      # mixed forest: live oaks, no moss
    assert any(t["item"] in trees.LIVE_OAKS for t in mixed) and not any("moss" in t for t in mixed)
    # no moss_share, no moss: the trees are the same as before it
    assert trees.assign(tops, np.full(len(tops), 22)) == [{k: v for k, v in t.items() if k != "moss"} for t in town]


def test_swamp_off_keeps_v01_mix():
    placed = trees.assign(grid_tops(400, h=18), lambda x, y: np.full(len(x), 90), swamp=False)
    assert {t["item"] for t in placed} == {"br_redwood_broad", "br_tree_oak_large_c"}


def test_cypress_keeps_its_measured_height():
    out = trees.assign(grid_tops(100, h=18), np.full(100, 90))      # swamp trees by default
    cyp = [t for t in out if t["item"] == "br_cypress_a"]
    assert cyp and all(t["scale"] == pytest.approx(18 / 22, abs=1e-3) for t in cyp)


def test_custom_collision_reach():
    assert trees.collision_reach("br_cypress_a", 1.0, 0.0, 4.0) == pytest.approx(0.7)
    assert trees.collision_reach("br_cypress_a", 1.0, 9.0, 12.0) == 0.0


def test_custom_tree_collision_kept_off_road():
    def tree(x, y, scale):
        return {"x": x, "y": y, "z": 0.0, "item": "br_cypress_a", "scale": scale, "yaw": 0.0, "height": 18.0}

    # 6.5 m street along y = 0: its corridor ends at 3.25 m; a 0.7 m trunk at 3.8 m reaches 3.1 m
    placed = [tree(0.0, 3.8, 1.0), tree(10.0, 4.1, 1.0), tree(20.0, 4.1, 1.6)]
    kept, dropped = trees.clear_collisions(placed, [street()])
    assert [t["x"] for t in kept] == [10.0] and dropped == {"road": 2, "deck": 0}


def write_cypress_inputs(data_dir, billboard="bb_autobillboard25", items=("br_cypress_a",)):
    trees.save_trees([{"x": 10.0 * i, "y": 20.0, "z": 0.0, "item": item, "scale": 0.9, "yaw": 0.0,
                       "height": 19.8} for i, item in enumerate(items)], data_dir / "trees.json")
    return contract_npz(data_dir / "models" / "br_cypress_a" / "main.npz", CYPRESS_NODES, billboard=billboard)


def test_level_content_writes_custom_shape(tmp_path):
    cfg, data_dir, root = small_cfg(), tmp_path / "out", tmp_path / "level"
    write_cypress_inputs(data_dir)
    trees.level_content(cfg, data_dir, root, FLAT, EXT)
    assert (root / "art/shapes/trees/br_swamp/br_cypress_a.dae").exists()
    meta = json.loads((root / "art/shapes/trees/br_swamp/br_cypress_a.dae.asset.json").read_text())
    assert meta["imposters"][0]["size"] == 25
    assert "br_tree_cypress_leaves" in json.loads((root / trees.MATERIALS_FILE).read_text())


def test_level_content_custom_and_stock_items(tmp_path):
    cfg, data_dir, root = small_cfg(), tmp_path / "out", tmp_path / "level"
    write_cypress_inputs(data_dir, items=("br_cypress_a", "br_tree_oak_large_c"))
    groups = trees.level_content(cfg, data_dir, root, FLAT, EXT)
    assert {o["class"] for o in groups["vegetation"]} == {"Forest", "ForestWindEmitter"}

    dae = (root / "art/shapes/trees/br_swamp/br_cypress_a.dae").read_text(encoding="utf-8")
    assert 'name="bb_autobillboard25"' in dae and 'name="br_cypress_a_a600"' in dae
    assert validate.dae_vertex_counts(root / "art/shapes/trees/br_swamp/br_cypress_a.dae") == {
        name: 6 for name in CYPRESS_NODES}
    meta = json.loads((root / "art/shapes/trees/br_swamp/br_cypress_a.dae.asset.json").read_text())
    assert meta == {"imposters": [{"detailLevel": 1, "dimension": 256, "equatorSteps": 16, "includePoles": False,
                                   "polarAngle": 0, "polarSteps": 0, "size": 25}]}

    mid = json.loads((root / trees.ITEM_DATA_FILE).read_text(encoding="utf-8"))
    assert set(mid) == {"br_cypress_a", "br_tree_oak_large_c"}
    c = mid["br_cypress_a"]
    assert c["shapeFile"] == "/levels/baton_rouge/art/shapes/trees/br_swamp/br_cypress_a.dae"
    assert (c["name"], c["internalName"], c["class"]) == ("br_cypress_a", "br_cypress_a", "TSForestItemData")
    assert c["persistentId"] == bf.persistent_id("fid", "br_cypress_a")
    assert {k: c[k] for k in trees.stock_trees()["items"]["br_holm_oak"]["item"]} == (
        trees.stock_trees()["items"]["br_holm_oak"]["item"])
    lines = [json.loads(s) for s in (root / "forest/br_cypress_a.forest4.json").read_text().splitlines()]
    assert len(lines) == 1 and lines[0]["type"] == "br_cypress_a" and lines[0]["scale"] == 0.9
    assert lines[0]["pos"] == [0.0, 20.0, pytest.approx(5.8)]

    mats = json.loads((root / trees.MATERIALS_FILE).read_text(encoding="utf-8"))
    stock_names = set(trees.stock_trees()["items"]["br_tree_oak_large_c"]["materials"])
    assert set(mats) == stock_names | {"br_tree_cypress_bark", "br_tree_cypress_leaves"}
    assert list(mats) == sorted(mats)
    assert all(m["mapTo"] == name and m["persistentId"] for name, m in mats.items())
    assert mats["br_tree_cypress_leaves"]["alphaRef"] == 90


def test_level_content_wants_the_built_model(tmp_path):
    cfg, data_dir = small_cfg(), tmp_path / "out"
    write_cypress_inputs(data_dir).unlink()
    with pytest.raises(FileNotFoundError, match="models"):
        trees.level_content(cfg, data_dir, tmp_path / "level", FLAT, EXT)


def test_level_content_checks_the_model_against_the_registry(tmp_path):
    cfg, data_dir = small_cfg(), tmp_path / "out"
    write_cypress_inputs(data_dir, billboard="")                     # the registry asks for a billboard
    with pytest.raises(ValueError, match="billboard"):
        trees.level_content(cfg, data_dir, tmp_path / "level", FLAT, EXT)
    write_cypress_inputs(data_dir)
    contract_npz(data_dir / "models" / "br_cypress_a" / "main.npz", CYPRESS_NODES,
                 materials=("br_tree_cypress_bark", "br_no_such_material"), billboard="bb_autobillboard25")
    with pytest.raises(ValueError, match="br_no_such_material"):
        trees.level_content(cfg, data_dir, tmp_path / "level", FLAT, EXT)


MOSS_NODES = ("br_moss_oak_dry_a_a300", "br_moss_oak_dry_a_a100", "nulldetail30")


def write_moss_texture(data_dir):
    """textures/index.json and the files the moss material names, as the textures stage writes them."""
    folder = data_dir / "textures"
    folder.mkdir(parents=True, exist_ok=True)
    files = {"b": "moss_b.color.dds", "nm": "moss_nm.normal.dds", "o": "moss_o.data.dds"}
    for name in files.values():
        (folder / name).write_bytes(b"DDS ")
    (folder / "index.json").write_text(json.dumps({"moss": {"source": "generated", "files": files}}),
                                       encoding="utf-8")


def test_moss_overlay_lines_match_their_tree(tmp_path):
    # one br_oak_dry_a with moss -> br_moss_oak_dry_a.forest4.json has one line with the same pos,
    # rotationMatrix, scale
    cfg, data_dir, root = small_cfg(), tmp_path / "out", tmp_path / "level"
    mossy = {"x": 12.5, "y": -40.0, "z": 0.0, "item": "br_oak_dry_a", "scale": 1.2, "yaw": 0.7, "height": 10.6,
             "moss": True}
    bare = {k: v for k, v in mossy.items() if k != "moss"} | {"x": 60.0, "yaw": 2.0}
    trees.save_trees([mossy, bare], data_dir / "trees.json")
    contract_npz(data_dir / "models" / "br_moss_oak_dry_a" / "main.npz", MOSS_NODES, materials=("br_tree_moss",))
    write_moss_texture(data_dir)
    trees.level_content(cfg, data_dir, root, FLAT, EXT)

    def lines(item):
        return [json.loads(s) for s in (root / f"forest/{item}.forest4.json").read_text().splitlines()]

    oaks, moss = lines("br_oak_dry_a"), lines("br_moss_oak_dry_a")
    assert len(oaks) == 2 and len(moss) == 1 and moss[0]["type"] == "br_moss_oak_dry_a"
    oak = next(o for o in oaks if o["pos"][0] == 12.5)
    assert {k: moss[0][k] for k in ("ctxid", "pos", "rotationMatrix", "scale")} == {
        k: oak[k] for k in ("ctxid", "pos", "rotationMatrix", "scale")}

    mid = json.loads((root / trees.ITEM_DATA_FILE).read_text(encoding="utf-8"))
    assert set(mid) == {"br_oak_dry_a", "br_moss_oak_dry_a"}
    m = mid["br_moss_oak_dry_a"]
    assert m["shapeFile"] == "/levels/baton_rouge/art/shapes/trees/br_swamp/br_moss_oak_dry_a.dae"
    assert {k: m[k] for k in trees.stock_trees()["items"]["br_oak_dry_a"]["item"]} == (
        trees.stock_trees()["items"]["br_oak_dry_a"]["item"])                    # it sways with its oak
    dae = root / "art/shapes/trees/br_swamp/br_moss_oak_dry_a.dae"
    assert dae.exists() and not dae.with_name(dae.name + ".asset.json").exists()   # no billboard, no imposters
    assert validate.dae_vertex_counts(dae) == {"br_moss_oak_dry_a_a300": 3, "br_moss_oak_dry_a_a100": 3}
    mats = json.loads((root / trees.MATERIALS_FILE).read_text(encoding="utf-8"))
    assert mats["br_tree_moss"]["Stages"][0]["opacityMap"] == "/levels/baton_rouge/art/textures/br/moss_o.data.png"
    assert (root / "art/textures/br/moss_o.data.dds").exists()


def test_no_moss_overlay_without_mossy_trees(tmp_path):
    cfg, data_dir, root = small_cfg(), tmp_path / "out", tmp_path / "level"
    trees.save_trees([{"x": 1.0, "y": 2.0, "z": 0.0, "item": "br_oak_dry_a", "scale": 1.0, "yaw": 0.0,
                       "height": 9.0}], data_dir / "trees.json")
    trees.level_content(cfg, data_dir, root, FLAT, EXT)          # needs neither the moss model nor its texture
    assert sorted(p.name for p in (root / "forest").iterdir()) == ["br_oak_dry_a.forest4.json"]


def test_written_moss_material_is_tinted_down(tmp_path):
    # the moss card's albedo read near white-blue under the 0.40 sky; the in-game A/B picked this factor
    # (0.75 too pale, 0.4 dark blocks), with the card's own colours kept
    cfg, data_dir, root = small_cfg(), tmp_path / "out", tmp_path / "level"
    trees.save_trees([{"x": 1.0, "y": 2.0, "z": 0.0, "item": "br_oak_dry_a", "scale": 1.0, "yaw": 0.0,
                       "height": 9.0, "moss": True}], data_dir / "trees.json")
    contract_npz(data_dir / "models" / "br_moss_oak_dry_a" / "main.npz", MOSS_NODES, materials=("br_tree_moss",))
    write_moss_texture(data_dir)
    trees.level_content(cfg, data_dir, root, FLAT, EXT)
    mats = json.loads((root / trees.MATERIALS_FILE).read_text(encoding="utf-8"))
    assert mats["br_tree_moss"]["Stages"][0]["baseColorFactor"] == [0.65, 0.65, 0.5, 1.0]
    assert "baseColorFactor" not in mats["br_tree_moss"]          # a stage field, not a top-level flag


def test_moss_points_sit_in_the_crown():
    pts = json.loads((DATA / "moss_points.json").read_text())
    for item in trees.LIVE_OAKS:
        z = np.array(pts[item])[:, 2]; h = trees.tree_items()[item]["native_height"]
        assert 40 <= len(z) <= 120 and z.min() >= 0.35 * h - 1e-6 and z.max() <= 0.9 * h + 1e-6
    assert set(pts) == set(trees.LIVE_OAKS)


def test_run_passes_moss_bayous_and_wetlands(tmp_path, small_extent, monkeypatch):
    """run() hands assign() the config's moss share, the bayou runs' centrelines and the OSM wetlands in
    level coordinates."""
    cfg = small_cfg(tmp_path, moss_share=0.45)
    raw, out = tmp_path / "raw", tmp_path / "out"
    raw.mkdir()
    out.mkdir()
    np.save(out / "heights.npy", _ramp(extent=small_extent))
    n, px = canopy.analysis_grid(small_extent)
    canopy.save_canopy(out, bumps(small_extent, [(201.0, 101.0, 12.0)]), np.full((n, n), 0.6, np.float32), px)
    save_bayous({"runs": [{"id": "w1", "name": None, "kind": "stream",
                           "nodes": [[-300.0, 0.0, 4.0, 3.0], [0.0, 10.0, 4.0, 3.0], [300.0, 0.0, 4.0, 3.0]]}],
                 "pools": []}, out / "bayous.json")
    ring = [(-400.0, -400.0), (-300.0, -400.0), (-300.0, -300.0), (-400.0, -300.0), (-400.0, -400.0)]
    lonlat = [list(map(float, small_extent.to_lonlat(x, y))) for x, y in ring]
    (raw / "osm_features.json").write_text(json.dumps({"buildings": [], "nodes": [], "trees": [], "wetlands": [
        {"id": "w2", "wetland": "swamp", "polygons": [{"outer": lonlat, "inner": []}]}]}), encoding="utf-8")
    seen, assign = {}, trees.assign
    monkeypatch.setattr(trees, "assign", lambda *a, **kw: seen.update(kw) or assign(*a, **kw))
    trees.run(cfg)
    assert seen["moss_share"] == 0.45 and seen["swamp"] is True
    (line,) = seen["bayou_lines"]
    assert list(line.coords) == [(-300.0, 0.0), (0.0, 10.0), (300.0, 0.0)]
    (poly,) = seen["wetland_polys"]
    assert poly.area == pytest.approx(100.0 * 100.0, rel=1e-3) and poly.contains(Polygon(ring).centroid)


def test_moss_survives_into_trees_json(tmp_path, small_extent, monkeypatch):
    cfg = small_cfg(tmp_path, moss_share=1.0)
    out = tmp_path / "out"
    out.mkdir(parents=True)
    np.save(out / "heights.npy", _ramp(extent=small_extent))
    n, px = canopy.analysis_grid(small_extent)
    canopy.save_canopy(out, bumps(small_extent, [(201.0, 101.0, 12.0)]), np.full((n, n), 0.6, np.float32), px)
    monkeypatch.setattr(trees, "NLCD_CONTEXT", {0: "developed"})         # no nlcd.tif: class 0 everywhere
    monkeypatch.setitem(trees.SPECIES, "developed", [("br_holm_oak", 1.0, None)])
    trees.run(cfg)
    (t,) = trees.load_trees(out / "trees.json")
    assert t["item"] == "br_holm_oak" and t["moss"] is True


# ------------------------------------------------------------------ tree tops

def test_filters_match_scipy():
    from scipy import ndimage

    a = np.random.default_rng(3).normal(5, 3, (37, 53)).astype(np.float32)
    a[4:9, 10:20] = 7.0  # a plateau
    np.testing.assert_allclose(trees._gaussian(a, 1.0), ndimage.gaussian_filter(a, 1.0), atol=1e-5)
    for size in (5, 7):
        np.testing.assert_array_equal(trees._max_filter(a, size),
                                      ndimage.maximum_filter(a, size=size, mode="constant", cval=-np.inf))


def test_detect_tops_finds_crowns(small_extent):
    n, px = canopy.analysis_grid(small_extent)
    crowns = [(-301.0, 201.0, 12.0), (-1.0, -99.0, 18.0), (199.0, 101.0, 25.0)]
    tops = trees.detect_tops(bumps(small_extent, crowns), np.full((n, n), 0.6, np.float32), px, small_extent)
    assert tops.shape == (3, 3)
    for x, y, h in crowns:
        d = np.hypot(tops[:, 0] - x, tops[:, 1] - y)
        i = int(np.argmin(d))
        assert d[i] <= 2.0 and abs(tops[i, 2] - h) <= 1.0


def test_detect_tops_ignores_buildings_like_flat(small_extent):
    n, px = canopy.analysis_grid(small_extent)
    ndsm = np.zeros((n, n), np.float32)
    ndsm[200:210, 300:310] = 10.0  # a flat 20 x 20 m roof
    ndvi = np.full((n, n), 0.1, np.float32)
    assert len(trees.detect_tops(ndsm, ndvi, px, small_extent)) == 0
    # the same block green (a hedge, a clipped canopy): one top for the plateau, not dozens
    tops = trees.detect_tops(ndsm, np.full((n, n), 0.6, np.float32), px, small_extent)
    assert len(tops) == 1
    x, y = -512 + 305 * px, 512 - 205 * px
    assert abs(tops[0, 0] - x) <= 10 and abs(tops[0, 1] - y) <= 10 and tops[0, 2] == pytest.approx(10.0)


def test_no_trees_where_canopy_nan(small_extent):
    n, px = canopy.analysis_grid(small_extent)
    xs, ys = small_extent.pixel_centers(n)
    ndsm = bumps(small_extent, [(-301.0, 201.0, 12.0), (-1.0, -99.0, 18.0), (199.0, 101.0, 25.0)])
    ndvi = np.full((n, n), 0.6, np.float32)
    ndsm[:, xs > 100] = np.nan   # the lidar gap
    ndvi[ys < -50, :] = np.nan   # no imagery
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tops = trees.detect_tops(ndsm, ndvi, px, small_extent)
        assert len(trees.detect_tops(np.full((n, n), np.nan, np.float32), ndvi, px, small_extent)) == 0
    assert tops.shape == (1, 3) and np.hypot(tops[0, 0] + 301, tops[0, 1] - 201) <= 2.0


def test_tree_not_on_road_or_deck():
    tops = np.array([[0.0, 4.0, 10], [0.0, 6.0, 10],            # 4 m off a 6.5 m street: on it; 6 m: clear
                     [0.0, -191.0, 10], [0.0, -189.0, 10],      # 9 m off a 13.8 m deck: too close; 11 m: clear
                     [106.0, 104.0, 10], [112.6, 104.0, 10], [114.0, 104.0, 10],  # in, 0.6 m and 2 m off a house
                     [-300.0, -300.0, 10]])                     # in a lake
    kept, dropped = trees.exclude(tops, [house(100, 100, 112, 109)], [street(), deck()],
                                  [WaterBody("Lake", box(-400, -400, -250, -250), 5.0, 2.0)])
    assert sorted(map(tuple, kept[:, :2])) == [(0.0, -189.0), (0.0, 6.0), (114.0, 104.0)]
    assert dropped == {"road": 1, "deck": 1, "building": 2, "water": 1, "landmark": 0, "bayou": 0}
    kept, dropped = trees.exclude(np.empty((0, 3)), [], [], [])
    assert kept.shape == (0, 3) and sum(dropped.values()) == 0


def test_tree_collision_kept_off_road_and_deck():
    # The stock collision meshes (stock_trees.json "collision", read from the game) are wider than
    # the trunks: italy's holm_oak has branches 3.3 m out between 2 and 3 m up, WCUSA's fanpalm_tall a
    # 0.4 m trunk, ECA's tree_oak_large_b a trunk and low limbs 1.7 m out at most to 3.5 m up under a
    # crown reaching 2.8 m out from 4 m up. (0.40 took the crowns out of the other oaks' collision
    # meshes, leaving their trunks.)
    road = (-trees.Z_OFFSET, trees.HEADROOM - trees.Z_OFFSET)          # the band clear_collisions keeps off a road
    assert trees.collision_reach("br_holm_oak", 1.0, *road) >= 3.2
    assert trees.collision_reach("br_fanpalm_tall", 1.0, *road) < 0.5
    assert trees.collision_reach("br_tree_oak_large_b", 1.0, *road) < 1.75
    assert trees.collision_reach("br_tree_oak_large_b", 1.5) == pytest.approx(1.5 * 2.84, abs=0.05)

    def tree(x, y, item, scale=1.0):
        return {"x": x, "y": y, "z": 0.0, "item": item, "scale": scale, "yaw": 0.0, "height": 12.0}

    placed = [tree(0.0, 5.0, "br_holm_oak"),                   # 1.75 m off the 6.5 m street: its mesh is on it
              tree(10.0, 5.0, "br_fanpalm_tall"),              # as close, a slim trunk: kept
              tree(20.0, 5.25, "br_tree_oak_large_b"),         # 2 m off, its crown above the headroom: kept
              tree(30.0, 9.0, "br_holm_oak"),                  # 5.75 m off: clear
              tree(0.0, -189.6, "br_tree_oak_large_b", 1.5),   # 3.5 m off the deck, its crown 4.3 m out: dropped
              tree(40.0, -189.6, "br_tree_oak_large_b")]       # crown 2.8 m out: kept
    kept, dropped = trees.clear_collisions(placed, [street(), deck()])
    assert [(t["x"], t["y"]) for t in kept] == [(10.0, 5.0), (20.0, 5.25), (30.0, 9.0), (40.0, -189.6)]
    assert dropped == {"road": 1, "deck": 1}
    assert trees.clear_collisions([], [street()]) == ([], {"road": 0, "deck": 0})
    assert trees.clear_collisions(placed, [])[0] == placed


# ------------------------------------------------------------------ species, scale, cap

def test_species_and_scale():
    stock = trees.tree_items()
    tops = np.array([[0.0, 0.0, 30.0], [10.0, 0.0, 20.0], [20.0, 0.0, 5.0], [30.0, 0.0, 60.0],
                     [40.0, 0.0, 4.1], [50.0, 0.0, 14.0]])
    codes = np.array([90, 42, 22, 22, 22, 0])
    out = trees.assign(tops, codes)
    assert [set(t) for t in out] == [{"x", "y", "z", "item", "scale", "yaw", "height"}] * 6
    assert out[0]["item"] in species_items("wetland")       # cypress
    assert out[1]["item"] in species_items("evergreen")     # pine
    assert out[2]["item"] in species_items("small")         # under 8 m
    assert out[3]["item"] in species_items("developed")
    assert out[5]["item"] in species_items("other")
    for t, (x, y, h) in zip(out, tops):
        assert (t["x"], t["y"], t["height"]) == (x, y, h)
        native = stock[t["item"]]["native_height"]
        lo, hi = trees.scale_limits(t["item"])
        assert t["scale"] == pytest.approx(min(max(h / native, lo), hi), abs=1e-3)
        assert 0 <= t["yaw"] < 2 * math.pi
    assert out[3]["scale"] == trees.scale_limits(out[3]["item"])[1]  # clamped
    assert out[4]["scale"] == pytest.approx(max(trees.scale_limits(out[4]["item"])[0],
                                                4.1 / stock[out[4]["item"]]["native_height"]), abs=1e-3)
    # the same input gives the same trees; nlcd_at may also be a function of x, y
    assert trees.assign(tops, lambda x, y: codes) == out


def test_species_picked_from_the_in_game_lineup():
    # the 2026-10-16 lineup: holm_oak_city_tall is too spindly for a Baton Rouge street tree
    used = {i for opts in trees.SPECIES.values() for i, _, _ in opts}
    assert "br_holm_oak_city_tall" not in used
    assert used <= set(trees.stock_trees()["items"])
    assert {i for i, _, _ in trees.SPECIES["evergreen"]} == {"br_pine_radiata_large"}
    assert abs(sum(w for _, w, _ in trees.SPECIES["developed"]) - 1.0) < 1e-9
    palms = sum(w for i, w, _ in trees.SPECIES["developed"] if "palm" in i)
    assert 0 < palms <= 0.02


def test_cypress_scaled_to_swamp_height():
    # without the swamp trees, the redwood stand-in (38 m natively) must keep a wetland tree's height
    tops = np.array([[float(x), 0.0, 18.0] for x in range(0, 400, 10)])
    out = trees.assign(tops, np.full(len(tops), 90), swamp=False)
    cyp = [t for t in out if t["item"] == "br_redwood_broad"]
    assert cyp, "no cypress in a wetland"
    native = trees.stock_trees()["items"]["br_redwood_broad"]["native_height"]
    assert all(t["scale"] * native == pytest.approx(18.0, abs=0.05) for t in cyp)


def test_small_mix_only_in_town():
    # a 6 m tree in a swamp is still a swamp tree; in town it is a small street tree
    swamp = trees.assign(np.array([[0.0, 0.0, 6.0]]), np.array([90]))[0]["item"]
    town = trees.assign(np.array([[0.0, 0.0, 6.0]]), np.array([22]))[0]["item"]
    assert swamp in species_items("wetland") and town in species_items("small")


def test_species_mix_follows_weights():
    rng = np.random.default_rng(1)
    n = 4000
    tops = np.column_stack([rng.uniform(-3000, 3000, n), rng.uniform(-3000, 3000, n), np.full(n, 12.0)])
    out = trees.assign(tops, np.full(n, 22))
    weights = {}
    for item, w, max_h in trees.SPECIES["developed"]:
        if max_h is None or 12.0 <= max_h:
            weights[item] = weights.get(item, 0) + w
    total = sum(weights.values())
    counts = {}
    for t in out:
        counts[t["item"]] = counts.get(t["item"], 0) + 1
    assert set(counts) <= set(weights)
    for item, w in weights.items():
        assert counts.get(item, 0) / n == pytest.approx(w / total, abs=0.03), item
    assert len({round(t["yaw"], 2) for t in out}) > 500  # yaws spread


def test_cap_keeps_spread():
    rng = np.random.default_rng(0)
    xy = rng.uniform(-200, 200, (1000, 2))  # 8 x 8 cells of 50 m, all occupied
    h = rng.uniform(4, 30, 1000)
    tt = [{"x": float(x), "y": float(y), "z": 0.0, "item": "br_holm_oak", "scale": 1.0, "yaw": 0.0,
           "height": float(v)} for (x, y), v in zip(xy, h)]
    kept = trees.cap(tt, 100)
    assert len(kept) == 100
    cells = {}
    for t in tt:
        key = (math.floor(t["x"] / 50), math.floor(t["y"] / 50))
        cells[key] = max(cells.get(key, 0.0), t["height"])
    assert len(cells) == 64
    kept_ids = {id(t) for t in kept}
    tallest = {key: next(t for t in tt if (math.floor(t["x"] / 50), math.floor(t["y"] / 50)) == key
                         and t["height"] == hmax) for key, hmax in cells.items()}
    assert all(id(t) in kept_ids for t in tallest.values())  # every cell keeps its tallest
    rest_kept = [t["height"] for t in kept if id(t) not in {id(v) for v in tallest.values()}]
    rest_dropped = [t["height"] for t in tt if id(t) not in kept_ids]
    assert len(rest_kept) == 36 and min(rest_kept) >= max(rest_dropped)  # then the rest by height
    assert trees.cap(tt, 5000) == tt
    few = trees.cap(tt, 10)  # fewer than the cells: the 10 tallest cell winners
    assert sorted(t["height"] for t in few) == sorted(cells.values())[-10:]
    assert trees.cap(tt, 0) == []


# ------------------------------------------------------------------ stage and level files

def _ramp(n=256, extent=None):
    """Terrain rising 1 m per 100 m eastward, around 6 m."""
    xs, _ = extent.pixel_centers(n)
    return np.tile((6.0 + xs / 100.0).astype(np.float32), (n, 1))


def test_level_content_files(tmp_path, small_extent):
    cfg = small_cfg()
    data = tmp_path / "out"
    def tree(x, y, item, scale, yaw, height):
        return {"x": x, "y": y, "z": 0.0, "item": item, "scale": scale, "yaw": yaw, "height": height}

    trees.save_trees([tree(10.0, 20.0, "br_holm_oak_city_tall", 1.2, 0.0, 11.5),
                      tree(-50.0, 5.0, "br_holm_oak_city_tall", 0.8, math.pi / 2, 7.7),
                      tree(300.0, -40.0, "br_fanpalm_tall", 1.0, 1.0, 17.0)], data / "trees.json")
    heights = _ramp(extent=small_extent)
    root = tmp_path / "level"
    groups = trees.level_content(cfg, data, root, heights, small_extent)
    stock = trees.stock_trees()

    (veg,) = groups.values()
    assert list(groups) == ["vegetation"] and {o["class"] for o in veg} == {"Forest", "ForestWindEmitter"}
    assert all(o["persistentId"] for o in veg) and len({o["persistentId"] for o in veg}) == 2

    mid = json.loads((root / "art/forest/managedItemData.json").read_text(encoding="utf-8"))
    assert set(mid) == {"br_holm_oak_city_tall", "br_fanpalm_tall"}  # only the used items
    for key, d in mid.items():
        assert d["name"] == d["internalName"] == key and d["class"] == "TSForestItemData"
        assert d["persistentId"] == bf.persistent_id("fid", key)
        assert d["shapeFile"] == stock["items"][key]["shapeFile"]
        assert {k: d[k] for k in stock["items"][key]["item"]} == stock["items"][key]["item"]

    assert sorted(p.name for p in (root / "forest").iterdir()) == ["br_fanpalm_tall.forest4.json",
                                                                   "br_holm_oak_city_tall.forest4.json"]
    lines = [json.loads(s) for s in
             (root / "forest/br_holm_oak_city_tall.forest4.json").read_text(encoding="utf-8").splitlines()]
    assert [list(o) for o in lines] == [["ctxid", "pos", "rotationMatrix", "scale", "type"]] * 2
    assert all(o["ctxid"] == 0 and o["type"] == "br_holm_oak_city_tall" for o in lines)
    assert lines[0]["pos"][:2] == [10.0, 20.0] and lines[0]["scale"] == 1.2
    assert lines[0]["pos"][2] == pytest.approx(sample_height(heights, small_extent, 10.0, 20.0) - 0.2, abs=1e-3)
    assert lines[1]["pos"][2] == pytest.approx(sample_height(heights, small_extent, -50.0, 5.0) - 0.2, abs=1e-3)
    assert lines[0]["rotationMatrix"] == [1, 0, 0, 0, 1, 0, 0, 0, 1]
    assert lines[1]["rotationMatrix"] == [0, 1, 0, -1, 0, 0, 0, 0, 1]  # counter-clockwise 90 degrees

    mats = json.loads((root / "art/shapes/trees/br_trees.materials.json").read_text(encoding="utf-8"))
    used = ("br_holm_oak_city_tall", "br_fanpalm_tall")
    assert set(mats) == {m for key in used for m in stock["items"][key]["materials"]}
    for name, m in mats.items():
        assert m["class"] == "Material" and m["mapTo"] == name and m["persistentId"]

    trees.save_trees([{"x": 0.0, "y": 0.0, "z": 0.0, "item": "br_no_such_tree", "scale": 1.0, "yaw": 0.0,
                       "height": 9.0}], data / "trees.json")
    with pytest.raises(ValueError, match="br_no_such_tree"):
        trees.level_content(cfg, data, tmp_path / "level2", heights, small_extent)


def test_level_writes_the_vegetation_group(tmp_path):
    from test_level import make_inputs

    data = tmp_path / "data" / "out"
    make_inputs(data)
    trees.save_trees([{"x": 50.0, "y": 50.0, "z": 0.0, "item": "br_tree_oak_large_a", "scale": 1.0, "yaw": 0.5,
                       "height": 13.0}], data / "trees.json")
    root = level.write_level(small_cfg(), tmp_path / "dist" / "level", data)
    listing = (root / "main/MissionGroup/items.level.json").read_text(encoding="utf-8")
    assert '"name": "vegetation"' in listing
    objs = [json.loads(s) for s in
            (root / "main/MissionGroup/vegetation/items.level.json").read_text(encoding="utf-8").splitlines()]
    assert sorted(o["class"] for o in objs) == ["Forest", "ForestWindEmitter"]
    assert all(o["__parent"] == "vegetation" for o in objs)
    item_data = root / "art/forest/managedItemData.json"
    assert re.search(r"/[A-Za-z]+Data\.json$", item_data.as_posix())  # the loader's /%a+Data%.json$ pattern
    assert (root / "forest/br_tree_oak_large_a.forest4.json").exists()
    assert (root / "art/shapes/trees/br_trees.materials.json").name.endswith("materials.json")


def _nlcd(path, extent, west: int, east: int):
    """A 30 m NLCD raster in UTM: one class west of the centre, another east of it."""
    w, s, e, n = extent.bbox_utm(300.0)
    width, height = int((e - w) / 30), int((n - s) / 30)
    cols = w + (np.arange(width) + 0.5) * 30
    codes = np.where(cols < extent.center_e, west, east).astype(np.uint8)
    with rasterio.open(path, "w", driver="GTiff", width=width, height=height, count=1, dtype="uint8",
                       crs="EPSG:26915", transform=from_origin(w, n, 30, 30), nodata=0) as dst:
        dst.write(np.tile(codes, (height, 1)), 1)


def test_run_writes_trees_json(tmp_path, small_extent, monkeypatch):
    cfg = small_cfg(tmp_path, trees_max=100)
    raw, out = tmp_path / "raw", tmp_path / "out"
    raw.mkdir()
    heights = _ramp(extent=small_extent)
    out.mkdir()
    np.save(out / "heights.npy", heights)
    n, px = canopy.analysis_grid(small_extent)
    crowns = [(-201.0, 201.0, 20.0),   # west: wetland
              (201.0, 201.0, 15.0),    # east: developed
              (101.0, 1.0, 12.0),      # on the street
              (301.0, -199.0, 10.0)]   # inside a house
    canopy.save_canopy(out, bumps(small_extent, crowns), np.full((n, n), 0.6, np.float32), px)
    save_roads([street()], out / "roads.json")
    save_buildings([house(290, -210, 312, -188)], out / "buildings.json")
    save_water([WaterBody("Lake", box(-480, -480, -420, -420), 5.0, 2.0)], out / "water.json")
    _nlcd(raw / "nlcd.tif", small_extent, west=90, east=22)

    def node(i, x, y, tags):
        lon, lat = small_extent.to_lonlat(x, y)
        return {"id": f"n{i}", "lon": lon, "lat": lat, "tags": {"natural": "tree", **tags}}

    (raw / "osm_features.json").write_text(json.dumps({"buildings": [], "nodes": [], "trees": [
        node(1, -199.0, 201.0, {}),                # 2 m from a detected top: the same tree
        node(2, -301.0, -301.0, {"height": "7 m"}),
        node(3, 600.0, 0.0, {})]}), encoding="utf-8")  # outside the box

    trees.run(cfg)
    got = trees.load_trees(out / "trees.json")
    assert sorted((round(t["x"]), round(t["y"])) for t in got) == [(-301, -301), (-201, 201), (201, 201)]
    by_xy = {(round(t["x"]), round(t["y"])): t for t in got}
    assert by_xy[(-201, 201)]["item"] in species_items("wetland")
    assert by_xy[(201, 201)]["item"] in species_items("developed")
    # 7 m but in the west swamp: still a swamp tree (the small mix is for town)
    assert by_xy[(-301, -301)]["item"] in species_items("wetland") and by_xy[(-301, -301)]["height"] == 7.0
    for t in got:
        assert t["z"] == pytest.approx(sample_height(heights, small_extent, t["x"], t["y"]) - 0.2, abs=1e-3)
        assert t["item"] in trees.tree_items()

    # config trees.swamp reaches assign()
    swamp, assign = [], trees.assign
    monkeypatch.setattr(trees, "assign", lambda *a, **kw: swamp.append(kw["swamp"]) or assign(*a, **kw))
    trees.run(cfg)
    trees.run(small_cfg(tmp_path, trees_max=100, swamp_trees=False))
    assert swamp == [True, False]
    by_xy = {(round(t["x"]), round(t["y"])): t for t in trees.load_trees(out / "trees.json")}
    assert by_xy[(-201, 201)]["item"] in species_items("wetland", swamp=False)

    trees.run(small_cfg(tmp_path, trees_max=2))  # the cap keeps the tallest
    kept = sorted(t["height"] for t in trees.load_trees(out / "trees.json"))
    assert kept == pytest.approx([15.0, 20.0], abs=1.0)


def test_run_keeps_tree_collision_off_roads(tmp_path, small_extent, monkeypatch):
    """A crown 1.75 m off the street's edge passes exclude(); run() then keeps it only as a
    species whose collision mesh stays off the road."""
    cfg = small_cfg(tmp_path)
    raw, out = tmp_path / "raw", tmp_path / "out"
    raw.mkdir()
    out.mkdir()
    np.save(out / "heights.npy", _ramp(extent=small_extent))
    n, px = canopy.analysis_grid(small_extent)
    canopy.save_canopy(out, bumps(small_extent, [(201.0, 5.0, 12.0)]), np.full((n, n), 0.6, np.float32), px)
    save_roads([street()], out / "roads.json")
    for item, count in (("br_tree_oak_large_a", 0), ("br_fanpalm_tall", 1)):
        monkeypatch.setitem(trees.SPECIES, "other", [(item, 1.0, None)])  # no nlcd.tif: every tree is "other"
        trees.run(cfg)
        assert len(trees.load_trees(out / "trees.json")) == count, item


def _tool():
    spec = importlib.util.spec_from_file_location("extract_stock_trees", ROOT / "tools" / "extract_stock_trees.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _dae(lods: dict[str, tuple[float, float, float, str]], colmesh_material: str) -> str:
    """A minimal stock-like tree DAE: base00 > start01 > one node per LOD (name -> z range,
    scale, material) plus a Colmesh. Each LOD is a single triangle spanning the z range."""
    geoms, nodes, mats = [], [], {colmesh_material}
    for i, (name, (z0, z1, scale, mat)) in enumerate(lods.items()):
        mats.add(mat)
        pos = f"0 0 {z0} 1 0 {z1} 0 1 {(z0 + z1) / 2}"
        geoms.append(f"""
    <geometry id="g{i}-mesh"><mesh>
      <source id="g{i}-pos"><float_array id="g{i}-pos-array" count="9">{pos}</float_array>
        <technique_common><accessor source="#g{i}-pos-array" count="3" stride="3"/></technique_common></source>
      <vertices id="g{i}-verts"><input semantic="POSITION" source="#g{i}-pos"/></vertices>
      <triangles material="{mat}-material" count="1"><input semantic="VERTEX" source="#g{i}-verts" offset="0"/>
        <p>0 1 2</p></triangles>
    </mesh></geometry>""")
        nodes.append(f"""
        <node id="{name}" name="{name}" type="NODE">
          <matrix sid="transform">{scale} 0 0 0 0 {scale} 0 0 0 0 {scale} 0 0 0 0 1</matrix>
          <instance_geometry url="#g{i}-mesh" name="{name}"><bind_material><technique_common>
            <instance_material symbol="{mat}-material" target="#{mat}-material"/>
          </technique_common></bind_material></instance_geometry>
        </node>""")
    nodes.append(f"""
        <node id="Colmesh_x-1" name="Colmesh_x-1" type="NODE">
          <instance_geometry url="#g0-mesh" name="Colmesh_x-1"><bind_material><technique_common>
            <instance_material symbol="{colmesh_material}-material" target="#{colmesh_material}-material"/>
          </technique_common></bind_material></instance_geometry>
        </node>""")
    library = "".join(f'<material id="{m}-material" name="{m}"><instance_effect url="#{m}-effect"/></material>'
                      for m in sorted(mats))
    return f"""<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset><unit name="meter" meter="1"/><up_axis>Z_UP</up_axis></asset>
  <library_materials>{library}</library_materials>
  <library_geometries>{"".join(geoms)}</library_geometries>
  <library_visual_scenes><visual_scene id="Scene" name="Scene">
    <node id="base00" name="base00" type="NODE">
      <node id="bb_autobillboard25" name="bb_autobillboard25" type="NODE"/>
      <node id="start01" name="start01" type="NODE">
        <matrix sid="transform">1 0 0 0 0 1 0 0 0 0 1 3 0 0 0 1</matrix>{"".join(nodes)}
      </node>
    </node>
  </visual_scene></library_visual_scenes>
  <scene><instance_visual_scene url="#Scene"/></scene>
</COLLADA>"""


def _material(name: str, colour: str) -> dict:
    return {"name": name, "mapTo": name, "class": "Material", "persistentId": f"stock-{name}",
            "Stages": [{"baseColorMap": colour, "normalMap": "/assets/materials/tree/oak/t_nm.normal.png"},
                       {}, {}, {}], "alphaRef": 80}


def _fake_game(tmp_path) -> Path:
    """content/levels/{italy,east_coast_usa}.zip laid out like the stock levels."""
    game = tmp_path / "game"
    (game / "content" / "levels").mkdir(parents=True)
    (game / "integrity.json").write_text(json.dumps({"version": "0.40.0.0", "format": 1}), encoding="utf-8")
    with zipfile.ZipFile(game / "content" / "levels" / "italy.zip", "w") as z:
        z.writestr("levels/italy/art/forest/managedItemData.json", json.dumps({
            # the item for holm_oak.dae is called holm_oak_test, as in the real italy level
            "holm_oak_test": {"name": "holm_oak_test", "internalName": "holm_oak_test", "class": "forestItemData",
                              "__parent": "DataBlockGroup", "persistentId": "stock-pid", "annotation": "NATURE",
                              "shapeFile": "levels/italy/art/shapes/trees/trees_italy/holm_oak.dae",
                              "windScale": 0.5, "radius": 0.7}}))
        z.writestr("levels/italy/art/shapes/trees/trees_italy/holm_oak.dae", _dae(
            {"holm_oak_a250": (0.0, 20.0, 1.0, "oak_leaves"),      # lower detail: ignored
             "holm_oak_a900": (-0.5, 9.5, 1.5, "oak_bark")}, "leaves_strong"))
        z.writestr("levels/italy/art/shapes/trees/trees_italy/main.materials.json", json.dumps({
            "oak_bark": _material("oak_bark", "t_bark_b.color.png"),  # relative to this file's folder
            "oak_leaves": _material("oak_leaves", "/assets/materials/tree/oak/t_leaves_b.color.png")}))
        z.writestr("levels/italy/art/shapes/trees/main.materials.json", json.dumps({
            "leaves_strong": {"name": "leaves_strong", "mapTo": "leaves_strong", "class": "Material",
                              "Stages": [{}, {}, {}, {}], "groundType": "leaves_strong"}}))
        z.writestr("levels/italy/art/shapes/rocks/main.materials.json", "{ not json")  # stock has broken files
    with zipfile.ZipFile(game / "content" / "levels" / "east_coast_usa.zip", "w") as z:
        z.writestr("levels/east_coast_usa/art/forest/managedItemData.json", json.dumps({
            "tree_oak_large_a": {"name": "tree_oak_large_a", "class": "TSForestItemData",
                                 "shapeFile": "/levels/east_coast_usa/art/shapes/trees/trees_oak/tree_oak_large_a.dae",
                                 "mass": 100}}))
        z.writestr("levels/east_coast_usa/art/shapes/trees/trees_oak/tree_oak_large_a.dae", _dae(
            {"tree_oak_large_a_a1000": (0.0, 18.0, 1.0, "oak_leaves")}, "leaves_strong"))
        z.writestr("levels/east_coast_usa/art/shapes/trees/trees_oak/main.materials.json", json.dumps({
            # the same name as italy's, defined differently: italy's (listed first) is kept
            "oak_leaves": _material("oak_leaves", "/assets/materials/tree/oak/t_leaves_02_b.color.png")}))
        z.writestr("levels/east_coast_usa/art/shapes/trees/main.materials.json", json.dumps({
            "leaves_strong": {"name": "leaves_strong", "mapTo": "leaves_strong", "class": "Material",
                              "Stages": [{}, {}, {}, {}], "groundType": "leaves_strong"}}))
    return game


def test_extract_stock_trees_from_level_zips(tmp_path):
    tool = _tool()
    game = _fake_game(tmp_path)
    data = tool.extract(game, [("br_holm_oak", "italy", "holm_oak"),
                               ("br_tree_oak_large_a", "east_coast_usa", "tree_oak_large_a")])
    assert data["game_version"] == "0.40.0.0"                 # integrity.json's, all of it
    assert list(data["items"]) == ["br_holm_oak", "br_tree_oak_large_a"]
    oak = data["items"]["br_holm_oak"]
    assert oak["source"] == "holm_oak_test" and oak["level"] == "italy"  # found by its shape's name
    assert oak["shapeFile"] == "/levels/italy/art/shapes/trees/trees_italy/holm_oak.dae"
    assert oak["native_height"] == pytest.approx(15.0)  # the a900 mesh, scaled 1.5 by its node
    assert oak["item"] == {"annotation": "NATURE", "windScale": 0.5, "radius": 0.7}
    assert oak["materials"] == ["leaves_strong", "oak_bark", "oak_leaves"]  # every LOD's and the Colmesh's
    # the Colmesh is the a250 triangle (0,0,3) (1,0,23) (0,1,13) after start01's +3 m: in 0.5 m bands
    # from z 3 it reaches 0.05 m from the trunk axis in the lowest and 1 m at most
    col = oak["collision"]
    assert col["z0"] == 3.0 and col["dz"] == 0.5 and len(col["r"]) == 40
    assert col["r"][0] == pytest.approx(0.05, abs=0.011) and max(col["r"]) == pytest.approx(1.0)
    assert data["items"]["br_tree_oak_large_a"]["native_height"] == pytest.approx(18.0)
    assert data["items"]["br_tree_oak_large_a"]["materials"] == ["leaves_strong", "oak_leaves"]

    mats = data["materials"]
    assert set(mats) == {"leaves_strong", "oak_bark", "oak_leaves"}
    bark = mats["oak_bark"]["Stages"][0]
    assert bark["baseColorMap"] == "/levels/italy/art/shapes/trees/trees_italy/t_bark_b.color.png"
    assert bark["normalMap"] == "/assets/materials/tree/oak/t_nm.normal.png"
    assert all("persistentId" not in m for m in mats.values())
    assert mats["oak_leaves"]["Stages"][0]["baseColorMap"].endswith("t_leaves_b.color.png")
    assert data["material_conflicts"] == {"oak_leaves": ["italy", "east_coast_usa"]}
    assert all(p.startswith("/") for m in mats.values() for p in tool.texture_paths(m))


def test_extract_reports_a_missing_candidate(tmp_path):
    tool = _tool()
    with pytest.raises(LookupError, match="no_such_tree"):
        tool.extract(_fake_game(tmp_path), [("br_x", "italy", "no_such_tree")])


# ------------------------------------------------------------------ tools/extract_moss_points.py

def _moss_tool():
    spec = importlib.util.spec_from_file_location("extract_moss_points", ROOT / "tools" / "extract_moss_points.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _branches(n=10, facing=-1.0, z0=1.0, dz=0.9):
    """(2n, 3, 3) triangles: n level slabs 0.3 m wide from 0.5 m to 4.5 m out from the axis, z0 + k dz
    high, each two triangles wound to face down (facing -1) or up (+1)."""
    out = []
    for k in range(n):
        a, z = 2.4 * k, z0 + dz * k
        d, s = np.array([math.cos(a), math.sin(a), 0.0]), np.array([-math.sin(a), math.cos(a), 0.0])
        p = [0.5 * d - 0.15 * s, 4.5 * d - 0.15 * s, 4.5 * d + 0.15 * s, 0.5 * d + 0.15 * s]
        p = [q + (0.0, 0.0, z) for q in p]
        quad = [(p[0], p[1], p[2]), (p[0], p[2], p[3])]
        out += [t if facing > 0 else (t[0], t[2], t[1]) for t in quad]
    return np.array(out, float)


def test_hang_points_sit_under_the_branches():
    tool = _moss_tool()
    down, up = _branches(), _branches(facing=1.0, z0=1.05)
    pts = tool.hang_points(np.vstack([down, up]), 10.0)
    assert 40 <= len(pts) <= 120 and pts.shape[1] == 3
    assert pts[:, 2].min() >= 3.5 and pts[:, 2].max() <= 9.0                 # 35 % to 90 % of the height
    slabs = 1.0 + 0.9 * np.arange(10)
    assert np.abs(pts[:, 2, None] - slabs[None, :]).min(axis=1).max() < 1e-9   # only on the undersides
    assert np.hypot(pts[:, 0], pts[:, 1]).min() >= tool.MIN_REACH
    from scipy.spatial import cKDTree
    gaps = cKDTree(pts).query(pts, k=2)[0][:, 1]
    assert gaps.min() >= 0.5 * np.median(gaps)                                 # spread out (Poisson disc)
    assert np.array_equal(tool.hang_points(np.vstack([down, up]), 10.0), pts)  # the same every run
    assert len(tool.hang_points(down, 16.0)) > len(tool.hang_points(down, 10.0))   # bigger oaks, more moss


def _mesh_dae(lods: dict) -> str:
    """A stock-like tree DAE: base00 > start01 (lifted 0.5 m) > one node per LOD, its mesh a <triangles>
    block per material: lods = {node name: {material: (T, 3, 3) triangles}}."""
    geoms, nodes, mats = [], [], set()
    for i, (name, by_mat) in enumerate(lods.items()):
        pos = np.vstack(list(by_mat.values())).reshape(-1, 3)
        blocks, base = [], 0
        for mat, tris in by_mat.items():
            mats.add(mat)
            idx = " ".join(str(base + j) for j in range(3 * len(tris)))
            blocks.append(f'<triangles material="{mat}-material" count="{len(tris)}">'
                          f'<input semantic="VERTEX" source="#g{i}-verts" offset="0"/><p>{idx}</p></triangles>')
            base += 3 * len(tris)
        geoms.append(f"""<geometry id="g{i}-mesh"><mesh>
      <source id="g{i}-pos"><float_array id="g{i}-pos-array" count="{pos.size}">{" ".join(map(str, pos.ravel()))}</float_array>
        <technique_common><accessor source="#g{i}-pos-array" count="{len(pos)}" stride="3"/></technique_common></source>
      <vertices id="g{i}-verts"><input semantic="POSITION" source="#g{i}-pos"/></vertices>{"".join(blocks)}
    </mesh></geometry>""")
        binds = "".join(f'<instance_material symbol="{m}-material" target="#{m}-material"/>' for m in by_mat)
        nodes.append(f'<node id="{name}" name="{name}" type="NODE"><instance_geometry url="#g{i}-mesh">'
                     f'<bind_material><technique_common>{binds}</technique_common></bind_material>'
                     f'</instance_geometry></node>')
    library = "".join(f'<material id="{m}-material" name="{m}"><instance_effect url="#{m}-effect"/></material>'
                      for m in sorted(mats))
    return f"""<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset><unit name="meter" meter="1"/><up_axis>Z_UP</up_axis></asset>
  <library_materials>{library}</library_materials>
  <library_geometries>{"".join(geoms)}</library_geometries>
  <library_visual_scenes><visual_scene id="Scene" name="Scene">
    <node id="base00" name="base00" type="NODE">
      <node id="start01" name="start01" type="NODE">
        <matrix sid="transform">1 0 0 0 0 1 0 0 0 0 1 0.5 0 0 0 1</matrix>{"".join(nodes)}
      </node>
    </node>
  </visual_scene></library_visual_scenes>
  <scene><instance_visual_scene url="#Scene"/></scene>
</COLLADA>"""


def test_extract_moss_points_from_level_zips(tmp_path):
    tool = _moss_tool()
    game = tmp_path / "game"
    (game / "content" / "levels").mkdir(parents=True)
    shape = "levels/italy/art/shapes/trees/trees_italy/holm_oak.dae"
    lifted = _branches(z0=0.5)                                    # 1.0 .. 8.6 m after start01's 0.5 m
    with zipfile.ZipFile(game / "content" / "levels" / "italy.zip", "w") as z:
        z.writestr(shape, _mesh_dae({
            "holm_oak_a250": {"oak_bark": _branches(n=3, z0=6.0)},                # a lower LOD: ignored
            "holm_oak_a900": {"oak_bark": lifted, "oak_leaves": _branches(z0=0.7)}}))
    stock = {"items": {"br_holm_oak": {"level": "italy", "shapeFile": "/" + shape, "native_height": 10.0}},
             "materials": {"oak_bark": {"groundType": "WOOD", "Stages": [{}]},
                           "oak_leaves": {"groundType": "LEAVES_THIN", "alphaTest": True, "Stages": [{}]},
                           "oak_branch": {"groundType": "WOOD", "alphaTest": True, "Stages": [{}]}}}
    assert tool.bark_materials(stock["materials"]) == {"oak_bark"}           # cut-out branch cards are not bark
    pts = tool.extract(game, stock, ["br_holm_oak"])
    assert list(pts) == ["br_holm_oak"]
    p = np.array(pts["br_holm_oak"])
    assert 40 <= len(p) <= 120 and 3.5 <= p[:, 2].min() and p[:, 2].max() <= 9.0
    slabs = 1.0 + 0.9 * np.arange(10)                                         # the a900 bark, not the leaves
    assert np.abs(p[:, 2, None] - slabs[None, :]).min(axis=1).max() < 1e-6
