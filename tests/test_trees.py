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
from shapely.geometry import box

from brmap import beamng_format as bf
from brmap import canopy, level, trees
from brmap.buildings import Building, save_buildings
from brmap.config import Config
from brmap.geo import Extent
from brmap.roads import Road, save_roads
from brmap.terrain import sample_height
from brmap.water import WaterBody, save_water

ROOT = Path(__file__).resolve().parents[1]


def small_cfg(tmp_path=None, **kw):
    if tmp_path is not None:
        kw.update(raw_dir=str(tmp_path / "raw"), out_dir=str(tmp_path / "out"))
    return Config(debug_half_size_m=512.0, terrain_size=256, **kw)


def species_items(context):
    return {item for item, _, _ in trees.SPECIES[context]}


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
    assert data["game_version"].startswith("0.39")
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
        assert trees.SPECIES[context] and species_items(context) <= set(data["items"])


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
    assert dropped == {"road": 1, "deck": 1, "building": 2, "water": 1}
    kept, dropped = trees.exclude(np.empty((0, 3)), [], [], [])
    assert kept.shape == (0, 3) and sum(dropped.values()) == 0


def test_tree_collision_kept_off_road_and_deck():
    # The stock collision meshes (stock_trees.json "collision", read from the game) are wider than
    # the trunks: ECA's tree_oak_large_a has a root flare 1.9 m out and branches 2.3 m out at 2 m
    # and 4.3 m out at 3 m, WCUSA's fanpalm_tall a 0.4 m trunk, italy's holm_oak_city_tall a 0.3 m
    # trunk under a crown reaching 2.9 m out from 4 m up.
    assert trees.collision_reach("br_tree_oak_large_a", 1.0, 0.2, 3.2) >= 2.0
    assert trees.collision_reach("br_fanpalm_tall", 1.0, 0.2, 3.2) < 0.5
    assert trees.collision_reach("br_holm_oak_city_tall", 1.0, 0.2, 3.2) < 0.5
    assert trees.collision_reach("br_holm_oak_city_tall", 1.5) == pytest.approx(1.5 * 2.91, abs=0.05)

    def tree(x, y, item, scale=1.0):
        return {"x": x, "y": y, "z": 0.0, "item": item, "scale": scale, "yaw": 0.0, "height": 12.0}

    placed = [tree(0.0, 5.0, "br_tree_oak_large_a"),           # 1.75 m off the 6.5 m street: its mesh is on it
              tree(10.0, 5.0, "br_fanpalm_tall"),              # as close, a slim trunk: kept
              tree(20.0, 5.0, "br_holm_oak_city_tall", 1.5),   # as close, its crown above the headroom: kept
              tree(30.0, 9.0, "br_tree_oak_large_a"),          # 5.75 m off: clear
              tree(0.0, -189.6, "br_holm_oak_city_tall", 1.5),  # 3.5 m off the deck, its crown 4.4 m out: dropped
              tree(40.0, -189.6, "br_holm_oak_city_tall")]      # crown 2.9 m out: kept
    kept, dropped = trees.clear_collisions(placed, [street(), deck()])
    assert [(t["x"], t["y"]) for t in kept] == [(10.0, 5.0), (20.0, 5.0), (30.0, 9.0), (40.0, -189.6)]
    assert dropped == {"road": 1, "deck": 1}
    assert trees.clear_collisions([], [street()]) == ([], {"road": 0, "deck": 0})
    assert trees.clear_collisions(placed, [])[0] == placed


# ------------------------------------------------------------------ species, scale, cap

def test_species_and_scale():
    stock = trees.stock_trees()["items"]
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
    # the redwood stand-in is 38 m natively: a wetland tree must keep its measured height
    tops = np.array([[float(x), 0.0, 18.0] for x in range(0, 400, 10)])
    out = trees.assign(tops, np.full(len(tops), 90))
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


def test_run_writes_trees_json(tmp_path, small_extent):
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
        assert t["item"] in trees.stock_trees()["items"]

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
    (game / "integrity.json").write_text(json.dumps({"version": "0.39.4.0", "format": 1}), encoding="utf-8")
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
    assert data["game_version"] == "0.39.4"
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
