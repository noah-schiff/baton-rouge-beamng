import json

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from brmap import water
from brmap.config import Config
from brmap.water import WaterBody, build_water, carve, load_water, save_water

N = 256  # small_extent (1,024 m) at 4 m per cell


def ring(extent, xy):
    """Level-coordinate ring -> Overpass-style geometry list."""
    out = []
    for x, y in xy:
        lon, lat = extent.to_lonlat(x, y)
        out.append({"lat": lat, "lon": lon})
    return out


def square(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]


def lake_way(extent, xy, name="Test Lake", wid=1, tags=None):
    return {"type": "way", "id": wid, "nodes": list(range(len(xy))),
            "tags": tags or {"natural": "water", "name": name}, "geometry": ring(extent, xy)}


def heights_with(extent, inside_poly, inside=3.0, outside=10.0):
    xs, ys = extent.pixel_centers(N)
    X, Y = np.meshgrid(xs, ys)
    from shapely import contains_xy

    return np.where(contains_xy(inside_poly, X, Y), inside, outside).astype(np.float32)


def test_level_z_is_median_inside(small_extent):
    xy = square(-200, -200, 200, 200)
    h = heights_with(small_extent, Polygon(xy))
    (body,) = build_water({"elements": [lake_way(small_extent, xy)]}, small_extent, h)
    assert body.level_z == pytest.approx(3.0) and body.depth == 2.0 and body.name == "Test Lake"
    assert body.polygon.area == pytest.approx(400 * 400, rel=1e-3)


def test_small_pond_dropped(small_extent):
    xy = square(0, 0, 30, 30)  # 900 m^2
    assert build_water({"elements": [lake_way(small_extent, xy)]}, small_extent,
                       np.zeros((N, N), np.float32)) == []


def test_unclosed_way_ignored(small_extent):
    xy = square(-200, -200, 200, 200)[:-1]
    assert build_water({"elements": [lake_way(small_extent, xy)]}, small_extent,
                       np.zeros((N, N), np.float32)) == []


def test_mississippi_is_deep(small_extent):
    xy = square(-300, -300, 300, 300)
    (body,) = build_water({"elements": [lake_way(small_extent, xy, name="Mississippi River")]},
                          small_extent, np.zeros((N, N), np.float32))
    assert body.depth == 15.0


def test_unnamed_big_river_area_is_deep(extent):
    # In OSM the Mississippi's water area carries water=river but no name.
    big = square(-1500, -500, 1500, 500)  # 3 km^2
    small = square(3000, 3000, 3300, 3300)
    els = [lake_way(extent, big, wid=1, tags={"natural": "water", "water": "river"}),
           lake_way(extent, small, wid=2, tags={"natural": "water", "water": "river"})]
    bodies = sorted(build_water({"elements": els}, extent, np.zeros((N, N), np.float32)),
                    key=lambda b: -b.polygon.area)
    assert [b.depth for b in bodies] == [15.0, 2.0]


def test_carve_map_edge_is_not_shore(small_extent):
    # a river running off the east edge: the cut edge must stay deep, not slope up
    body = WaterBody("R", Polygon(square(200, -100, 512, 100)), level_z=3.0, depth=15.0)
    out = carve(np.full((N, N), 3.0, np.float32), small_extent, [body])
    assert out[N // 2, -1] == pytest.approx(3.0 - 15.0 * min(1.0, 102.0 / 60.0), abs=0.5)


def test_carve_window_matches_reference(small_extent):
    rng = np.random.default_rng(3)
    h = rng.uniform(2, 9, (N, N)).astype(np.float32)
    bodies = [WaterBody("A", Polygon(square(-300, -250, 50, -40)), 4.0, 2.0),
              WaterBody("B", Polygon([(100, 100), (400, 150), (250, 400), (100, 100)]), 5.0, 15.0)]
    np.testing.assert_allclose(carve(h, small_extent, bodies), water._carve_reference(h, small_extent, bodies))


def test_riverbank_way_counts(small_extent):
    xy = square(-200, -200, 200, 200)
    way = lake_way(small_extent, xy, tags={"waterway": "riverbank"})
    assert len(build_water({"elements": [way]}, small_extent, np.zeros((N, N), np.float32))) == 1


def test_polygon_clipped_to_box(small_extent):
    xy = square(300, -100, 900, 100)  # pokes 388 m out of the 512 m half box
    (body,) = build_water({"elements": [lake_way(small_extent, xy)]}, small_extent,
                          np.zeros((N, N), np.float32))
    assert body.polygon.bounds[2] == pytest.approx(512.0)


def test_carve_depth_center_and_bank(small_extent):
    body = WaterBody("L", Polygon(square(-200, -200, 200, 200)), level_z=3.0, depth=2.0)
    h = np.full((N, N), 3.0, np.float32)
    out = carve(h, small_extent, [body])
    assert out[N // 2, N // 2] == pytest.approx(1.0)        # center: level - depth
    xs, ys = small_extent.pixel_centers(N)
    shore_col = int(np.argmin(np.abs(xs - (-198.0))))       # first cell inside the west shore
    assert abs(out[N // 2, shore_col] - 3.0) < 0.5


def test_carve_outside_untouched(small_extent):
    body = WaterBody("L", Polygon(square(-200, -200, 200, 200)), level_z=3.0, depth=2.0)
    h = np.random.default_rng(1).uniform(4, 9, (N, N)).astype(np.float32)
    out = carve(h, small_extent, [body])
    xs, ys = small_extent.pixel_centers(N)
    X, Y = np.meshgrid(xs, ys)
    outside = (np.abs(X) > 202) | (np.abs(Y) > 202)
    np.testing.assert_array_equal(out[outside], h[outside])
    assert (out <= h).all()   # carving only lowers


def test_multipolygon_relation(small_extent):
    outer = square(-300, -300, 300, 300)
    inner = square(-50, -50, 50, 50)  # an island
    rel = {"type": "relation", "id": 9, "tags": {"type": "multipolygon", "natural": "water",
           "name": "Island Lake"}, "members": [
        {"type": "way", "ref": 1, "role": "outer", "geometry": ring(small_extent, outer[:3])},
        {"type": "way", "ref": 2, "role": "outer", "geometry": ring(small_extent, outer[2:])},
        {"type": "way", "ref": 3, "role": "inner", "geometry": ring(small_extent, inner)}]}
    (body,) = build_water({"elements": [rel]}, small_extent, np.full((N, N), 3.0, np.float32))
    assert body.polygon.area == pytest.approx(600 * 600 - 100 * 100, rel=1e-3)
    out = carve(np.full((N, N), 3.0, np.float32), small_extent, [body])
    assert out[N // 2, N // 2] == pytest.approx(3.0)  # the island stays dry


def test_relation_member_way_not_counted_twice(small_extent):
    outer = square(-300, -300, 300, 300)
    member = lake_way(small_extent, outer, wid=1)
    rel = {"type": "relation", "id": 9, "tags": {"natural": "water", "type": "multipolygon"},
           "members": [{"type": "way", "ref": 1, "role": "outer", "geometry": ring(small_extent, outer)}]}
    assert len(build_water({"elements": [member, rel]}, small_extent,
                           np.zeros((N, N), np.float32))) == 1


def test_water_json_roundtrip(tmp_path, small_extent):
    body = WaterBody("L", box(-100, -100, 100, 100), level_z=2.5, depth=2.0)
    save_water([body], tmp_path / "water.json")
    (back,) = load_water(tmp_path / "water.json")
    assert back.name == "L" and back.level_z == 2.5 and back.polygon.equals(body.polygon)


def test_rerun_level_z_stable(tmp_path, monkeypatch, small_extent):
    from brmap import terrain
    from rasters import write_dem

    monkeypatch.chdir(tmp_path)
    xy = square(-200, -200, 200, 200)
    raw = tmp_path / "data" / "raw"
    (raw / "dem").mkdir(parents=True)
    poly = Polygon([(x + small_extent.center_e, y + small_extent.center_n) for x, y in xy])
    from shapely import contains_xy

    write_dem(raw / "dem" / "2023-01-01_a.tif", small_extent,
              lambda E, N_: np.where(contains_xy(poly, E, N_), 3.0, 10.0))
    (raw / "osm.json").write_text(json.dumps({"elements": [lake_way(small_extent, xy)]}),
                                  encoding="utf-8")
    cfg = Config(debug_half_size_m=512.0, terrain_size=256)
    terrain.run(cfg)
    water.run(cfg)
    first = load_water(tmp_path / "data" / "out" / "water.json")[0].level_z
    terrain.run(cfg)  # now carves
    carved = np.load(tmp_path / "data" / "out" / "heights.npy")
    assert carved.min() < 2.0
    water.run(cfg)
    assert load_water(tmp_path / "data" / "out" / "water.json")[0].level_z == pytest.approx(first)
    assert first == pytest.approx(3.0, abs=0.01)
