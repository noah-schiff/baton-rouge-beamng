"""Buildings: footprints, conflicts, heights, roofs and level content (buildings.py)."""
import json
import logging
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from brmap import beamng_format as bf
from brmap import buildings, canopy, dae, level
from brmap.buildings import Building
from brmap.config import Config
from brmap.roads import Road, save_roads
from brmap.water import WaterBody, save_water

C = "http://www.collada.org/2005/11/COLLADASchema"
NS = {"c": C}


def ll(extent, xy):
    """[[lon, lat], ...] for level x, y points."""
    xy = np.asarray(xy, float)
    lon, lat = extent.to_lonlat(xy[:, 0], xy[:, 1])
    return [[float(a), float(b)] for a, b in zip(lon, lat)]


def rect(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def osm_building(extent, id, outer, inner=(), **tags):
    return {"id": id, "tags": {"building": "yes", **tags},
            "polygons": [{"outer": ll(extent, outer), "inner": [ll(extent, h) for h in inner]}]}


def fp(id, ring, kind="house", source="osm", tags=None, ms_height=None):
    """A footprint tuple as footprints() returns it."""
    return (id, source, kind, Polygon(ring), tags or {}, ms_height)


def road(xy, kind="residential", width=6.5, bridge=False, z=5.0):
    xy = np.asarray(xy, float)
    return Road([1], kind, np.column_stack([xy, np.full(len(xy), z)]), width, 2, False, bridge,
                1 if bridge else 0, None)


# ------------------------------------------------------------------ footprints

def test_footprints_level_coordinates_and_exclusions(small_extent):
    osm = {"buildings": [
        osm_building(small_extent, "w1", rect(10, 10, 22, 19), building="house"),
        osm_building(small_extent, "w2", rect(40, 10, 60, 30), building="roof"),
        osm_building(small_extent, "w3", rect(70, 10, 90, 30), building="carport"),
        osm_building(small_extent, "w4", rect(100, 10, 120, 30), **{"building:part": "yes"}),
        osm_building(small_extent, "w5", rect(130, 10, 133, 14)),        # 12 m2
        osm_building(small_extent, "w6", rect(-50, -50, -30, -30), building="ruins"),
    ], "trees": [], "nodes": []}
    out = buildings.footprints(osm, [], small_extent)
    assert [f[0] for f in out] == ["w1"]
    id, source, kind, poly, tags, ms_height = out[0]
    assert (source, kind, ms_height) == ("osm", "house", None) and tags["building"] == "house"
    assert poly.is_valid and poly.exterior.is_ccw
    assert poly.area == pytest.approx(12 * 9, rel=1e-6)
    assert poly.bounds == pytest.approx((10, 10, 22, 19), abs=1e-4)


def test_invalid_footprints_repaired_or_skipped(small_extent, caplog):
    bow_tie = [(0, 0), (10, 10), (10, 0), (0, 10)]
    osm = {"buildings": [
        osm_building(small_extent, "w1", bow_tie),
        {"id": "w2", "tags": {"building": "yes"},
         "polygons": [{"outer": ll(small_extent, [(50, 50), (60, 60)]), "inner": []}]},
        osm_building(small_extent, "w3", [(100, 0), (110, 0), (120, 0)]),  # zero area
        {"id": "w4", "tags": {"building": "yes"}, "polygons": []},
        {"id": "w5", "tags": {"building": "yes"}, "polygons": [{"outer": [], "inner": []}]},
        osm_building(small_extent, "w6", rect(200, 0, 210, 10) + [(200, 0)]),  # closed ring
    ], "trees": [], "nodes": []}
    ms = [{"ring": ll(small_extent, [(300, 0), (301, 1)]), "height": 4.0},
          {"ring": ll(small_extent, bow_tie[::-1] + [bow_tie[-1]]), "height": 4.0}]
    with caplog.at_level(logging.INFO, logger="brmap.buildings"):
        out = buildings.footprints(osm, ms, small_extent)
    ids = [f[0] for f in out]
    assert {i.split("/")[0] for i in ids} == {"w1", "w6"}  # the bow tie's MS copy overlaps it
    bow = [f[3] for f in out if f[0].startswith("w1")]
    assert len(bow) == 2 and all(p.is_valid and p.geom_type == "Polygon" for p in bow)
    assert sum(p.area for p in bow) == pytest.approx(50.0, rel=1e-6)
    assert len(set(ids)) == len(ids)
    assert "skipped" in caplog.text


def test_non_finite_footprints_skipped(small_extent, caplog):
    """A NaN or infinite coordinate (a corrupt record) skips that ring, not the stage."""
    nan, inf = float("nan"), float("inf")
    good = ll(small_extent, rect(0, 0, 20, 20))
    osm = {"buildings": [
        {"id": "w1", "tags": {"building": "yes"},
         "polygons": [{"outer": good[:2] + [[nan, nan]] + good[3:], "inner": []}]},
        {"id": "w2", "tags": {"building": "yes"},
         "polygons": [{"outer": [[inf, 30.43]] + ll(small_extent, [(50, 0), (60, 10)]), "inner": []}]},
        {"id": "w3", "tags": {"building": "yes"},   # a broken hole is left out, the building kept
         "polygons": [{"outer": ll(small_extent, rect(100, 0, 130, 30)),
                       "inner": [ll(small_extent, rect(110, 10, 120, 20))[:3] + [[nan, 30.43]]]}]},
        osm_building(small_extent, "w4", rect(200, 0, 210, 10)),
    ], "trees": [], "nodes": []}
    ms = [{"ring": good[:3] + [[nan, inf]], "height": 4.0},
          {"ring": ll(small_extent, rect(300, 0, 310, 10)), "height": 4.0}]
    with caplog.at_level(logging.INFO, logger="brmap.buildings"):
        out = buildings.footprints(osm, ms, small_extent)
    assert [f[0] for f in out] == ["w3", "w4", "ms1"]
    assert out[0][3].area == pytest.approx(900, rel=1e-6) and not out[0][3].interiors
    assert "skipped 3" in caplog.text


def test_multipolygon_with_hole(small_extent):
    osm = {"buildings": [
        {"id": "r7", "tags": {"building": "school"}, "polygons": [
            {"outer": ll(small_extent, rect(0, 0, 30, 30)), "inner": [ll(small_extent, rect(10, 10, 20, 20))]},
            {"outer": ll(small_extent, rect(50, 0, 60, 10)), "inner": []}]},
    ], "trees": [], "nodes": []}
    out = buildings.footprints(osm, [], small_extent)
    assert len(out) == 2 and len({f[0] for f in out}) == 2
    court = next(f[3] for f in out if f[3].area > 500)
    assert len(court.interiors) == 1 and court.area == pytest.approx(800, rel=1e-6)
    b = buildings.make_building(out[0], None, None, small_extent, 2.0, np.zeros((64, 64)))
    assert len(b.holes) == 1 and b.holes[0].shape == (4, 2)
    xml, _ = buildings.dae.tile_dae([b], (0.0, 0.0))
    root = ET.fromstring(xml)
    walls = [t for t in root.iter(f"{{{C}}}triangles") if "wall" in t.get("material")]
    # 4 outer walls and 4 courtyard walls, two triangles each, in the full LOD
    assert int(walls[0].get("count")) == 16


def test_building_across_edge_dropped(extent):
    h = extent.half_size
    osm = {"buildings": [
        osm_building(extent, "w1", rect(h - 20, 0, h + 5, 20)),       # crosses x = half
        osm_building(extent, "w2", rect(h - 20.5, 0, h - 0.5, 20)),  # inside, but within 1 m
        osm_building(extent, "w3", rect(h - 21.5, 0, h - 1.5, 20)),  # inside
        osm_building(extent, "w4", rect(0, -h - 5, 20, -h + 15)),     # crosses y = -half
    ], "trees": [], "nodes": []}
    ms = [{"ring": ll(extent, rect(-h - 3, 0, -h + 10, 10)), "height": 3.0}]
    assert [f[0] for f in buildings.footprints(osm, ms, extent)] == ["w3"]


def test_ms_overlap_dedup(small_extent):
    osm = {"buildings": [osm_building(small_extent, "w1", rect(0, 0, 10, 10))], "trees": [], "nodes": []}
    ms = [{"ring": ll(small_extent, rect(5, 0, 15, 10)), "height": 4.0},    # 50 % under w1
          {"ring": ll(small_extent, rect(9, 20, 19, 30)), "height": 5.0},   # not under it
          {"ring": ll(small_extent, rect(9, 0, 19, 10)), "height": 6.0}]    # 10 % under w1
    out = buildings.footprints(osm, ms, small_extent)
    assert [(f[0], f[1], f[2], f[5]) for f in out] == [
        ("w1", "osm", "yes", None), ("ms1", "ms", "yes", 5.0), ("ms2", "ms", "yes", 6.0)]


def test_osm_duplicates_kept_once(small_extent, caplog):
    # two OSM footprints drawn over each other (two ways, or a way and a multipolygon) both
    # became buildings, their walls and roofs z-fighting: when two overlap by 90 % or more of
    # the smaller one's area only the larger is kept, on a tie the first in id order
    assert buildings.OSM_DUP_OVERLAP == 0.9
    osm = {"buildings": [
        osm_building(small_extent, "w20", rect(0, 0, 20, 10)),            # the same outline twice
        osm_building(small_extent, "w9", rect(0, 0, 20, 10)),
        osm_building(small_extent, "r31", rect(100.2, 0.1, 120, 10)),     # inside a slightly larger copy
        osm_building(small_extent, "w30", rect(100, 0, 120.5, 10.2)),
        osm_building(small_extent, "w40", rect(200, 0, 220, 10)),         # half over its neighbour: both stay
        osm_building(small_extent, "w41", rect(210, 0, 230, 10)),
        osm_building(small_extent, "w50", rect(300, 0, 320, 10)),         # 85 % over a larger one: both stay
        osm_building(small_extent, "w51", rect(303, 0, 320, 12)),
    ], "trees": [], "nodes": []}
    with caplog.at_level(logging.INFO, logger="brmap.buildings"):
        out = buildings.footprints(osm, [], small_extent)
    assert [f[0] for f in out] == ["w9", "w30", "w40", "w41", "w50", "w51"]
    assert "6 OpenStreetMap" in caplog.text and "2 OpenStreetMap duplicates" in caplog.text


def test_ms_under_excluded_osm_dropped(small_extent):
    """A Microsoft footprint of a structure OSM leaves out (a fuel-station canopy tagged
    building=roof, a carport) stays out too: it would come back as a solid block."""
    osm = {"buildings": [osm_building(small_extent, "w1", rect(0, 0, 20, 10), building="roof"),
                         osm_building(small_extent, "w2", rect(50, 0, 60, 10), building="carport")],
           "trees": [], "nodes": []}
    ms = [{"ring": ll(small_extent, rect(1, 1, 19, 9)), "height": 5.0},      # the canopy
          {"ring": ll(small_extent, rect(50, 0, 60, 12)), "height": 3.0},    # the carport
          {"ring": ll(small_extent, rect(100, 0, 110, 10)), "height": 4.0}]  # a house nearby
    assert [f[0] for f in buildings.footprints(osm, ms, small_extent)] == ["ms2"]


def test_footprints_skip_ms_when_list_empty(small_extent):
    osm = {"buildings": [], "trees": [], "nodes": []}
    assert buildings.footprints(osm, [], small_extent) == []


# ------------------------------------------------------------------ conflicts

def test_building_on_road_dropped():
    roads = [road([(-100, 0), (100, 0)])]  # residential, 6.5 m: corridor |y| <= 3.25
    polys = [fp("across", rect(-5, -5, 5, 5)),
             fp("clear", rect(20, 5.25, 30, 15.25)),          # 2 m clear of the corridor
             fp("touching", rect(40, 2.95, 50, 12.95)),       # 0.3 m in: within the 0.5 m shrink
             fp("service", rect(-5, 60, 5, 70))]
    roads.append(road([(-100, 65), (100, 65)], kind="service", width=4.0))
    kept, dropped = buildings.drop_conflicts(polys, roads, [])
    assert [k[0] for k in kept] == ["clear", "touching"]
    assert dropped == {"road": ["across", "service"]}


def test_service_drive_into_large_building_kept():
    # the State Capitol has a service drive running 200 m2 into its 3,400 m2 outline (a ramp
    # under the building); a big building is kept when only a service road clips a small share
    # of it, but any street, or a service road through a small building, still drops it
    big = rect(0, 0, 60, 60)
    roads = [road([(30, -40), (30, 10)], kind="service", width=4.0)]  # 10 m into the building: 40 m2
    kept, dropped = buildings.drop_conflicts([fp("capitol", big)], roads, [])
    assert [k[0] for k in kept] == ["capitol"] and dropped == {}
    street = [road([(30, -40), (30, 10)], kind="residential", width=6.5)]
    assert buildings.drop_conflicts([fp("capitol", big)], street, [])[1] == {"road": ["capitol"]}
    through = [road([(-20, 30), (80, 30)], kind="service", width=4.0)]  # 240 m2: 6.7 % but all the way through
    assert buildings.drop_conflicts([fp("capitol", big)], through, [])[1] == {"road": ["capitol"]}
    # a drive that starts or ends right at the wall (where a passage comes out) only touches it
    # (real coordinates land a hair inside: the piece in the outline is a zero-length line)
    at_wall = [road([(30, 59.9999999), (30, 90)], kind="service", width=4.0),
               road([(90, 20), (60.0000001, 20)], kind="service", width=4.0)]
    assert buildings.drop_conflicts([fp("capitol", big)], at_wall, [])[1] == {}


def test_service_road_mostly_inside_a_building_drops_it():
    # a drive only ends in a building when at most SERVICE_MAX_INSIDE m of it is inside:
    # Microsoft footprint ms45051 (10,224 m2) kept the 249 m open-air loop 751606144 that runs
    # wholly inside it, since the loop covers under 15 % of the footprint and never leaves it
    assert buildings.SERVICE_MAX_INSIDE == 30.0
    side = 249.0 / 4
    loop = road(np.array(rect(20, 20, 20 + side, 20 + side) + [(20, 20)]), kind="service", width=4.0)
    drive = road([(230, -40), (230, 10)], kind="service", width=4.0)     # 10 m into its building
    deep = road([(430, -40), (430, 35)], kind="service", width=4.0)      # 35 m into its building
    polys = [fp("ms45051", rect(0, 0, 101, 101), source="ms"), fp("capitol", rect(200, 0, 260, 60)),
             fp("deep", rect(400, 0, 460, 60))]
    kept, dropped = buildings.drop_conflicts(polys, [loop, drive, deep], [])
    assert [k[0] for k in kept] == ["capitol"] and dropped == {"road": ["ms45051", "deep"]}


def test_covered_road_keeps_its_building():
    # OSM maps a drive passing under a building (tunnel=building_passage, covered=yes): it is
    # the building's own passage, so it does not knock the building out
    passage = road([(-20, 30), (80, 30)], kind="service", width=4.0)
    passage.covered = True
    kept, dropped = buildings.drop_conflicts([fp("capitol", rect(0, 0, 60, 60))], [passage], [])
    assert [k[0] for k in kept] == ["capitol"] and dropped == {}


def test_building_on_bridge_deck_dropped():
    deck = road([(-100, 0), (100, 0)], kind="motorway", width=13.8, bridge=True, z=20.0)
    polys = [fp("under", rect(-10, -4, 10, 4)), fp("beside", rect(-10, 9, 10, 19))]
    kept, dropped = buildings.drop_conflicts(polys, [deck], [])
    assert [k[0] for k in kept] == ["beside"] and dropped == {"bridge": ["under"]}


def test_building_on_service_bridge_deck_dropped():
    # every deck drops what it touches, service ones too (w1464127183 stood on the service
    # bridge way 1464127172): a deck never "ends in" a big building the way a drive can
    small = road([(-100, 0), (100, 0)], kind="service", width=4.0, bridge=True, z=12.0)
    into_big = road([(130, -40), (130, 10)], kind="service", width=4.0, bridge=True, z=12.0)  # 10 m in
    polys = [fp("under", rect(-10, -3, 10, 3)), fp("beside", rect(-10, 9, 10, 19)),
             fp("big", rect(100, 0, 160, 60))]
    kept, dropped = buildings.drop_conflicts(polys, [small, into_big], [])
    assert [k[0] for k in kept] == ["beside"] and dropped == {"bridge": ["under", "big"]}


def test_building_in_water_dropped():
    lake = WaterBody("Lake", box(0, 0, 100, 100), 3.0, 2.0)
    polys = [fp("barge", rect(10, 10, 30, 20)), fp("shore", rect(95, 10, 110, 20)),
             fp("dry", rect(200, 10, 220, 20))]
    kept, dropped = buildings.drop_conflicts(polys, [], [lake])
    assert [k[0] for k in kept] == ["shore", "dry"] and dropped == {"water": ["barge"]}


# ------------------------------------------------------------------ heights and roofs

def canopy_grid(n=512, ndsm_value=np.nan, ndvi_value=0.1):
    return np.full((n, n), ndsm_value, np.float32), np.full((n, n), ndvi_value, np.float32)


def cells(small_extent, poly, n=512):
    """Row, column masks of the analysis cells whose centres are inside poly."""
    xs, ys = small_extent.pixel_centers(n)
    X, Y = np.meshgrid(xs, ys)
    import shapely
    return shapely.contains_xy(poly, X, Y)


def test_height_order(small_extent):
    poly = Polygon(rect(100, 100, 120, 120))
    ndsm, ndvi = canopy_grid()
    inside = cells(small_extent, poly)
    vals = np.where(np.arange(inside.sum()) % 2, 10.0, 14.0)
    ndsm[inside] = vals
    tree = inside & (np.cumsum(inside, axis=None).reshape(inside.shape) % 7 == 0)
    ndsm[tree], ndvi[tree] = 30.0, 0.6  # an overhanging tree: vegetation, not roof
    args = (small_extent, 2.0)
    h, src = buildings.building_height(poly, {"height": "50"}, 7.0, "house", ndsm, ndvi, *args)
    assert src == "lidar"
    assert h == pytest.approx(np.percentile(ndsm[inside & ~tree], 75))
    empty, _ = canopy_grid()
    assert buildings.building_height(poly, {"height": "21 m", "building:levels": "4"}, 7.0, "house",
                                     empty, ndvi, *args) == (pytest.approx(21.0), "osm_height")
    assert buildings.building_height(poly, {"building:levels": "4"}, 7.0, "house",
                                     empty, ndvi, *args) == (pytest.approx(13.8), "levels")
    assert buildings.building_height(poly, {}, 7.0, "house", empty, ndvi, *args) == (7.0, "ms")
    assert buildings.building_height(poly, {}, -1.0, "house", empty, ndvi, *args) == (6.0, "default")
    assert buildings.building_height(poly, {}, None, "apartments", empty, ndvi, *args) == (12.0, "default")
    assert buildings.building_height(poly, {}, None, "shed", empty, ndvi, *args) == (3.0, "default")
    assert buildings.building_height(poly, {}, None, "yes", empty, ndvi, *args) == (5.0, "default")


def test_tower_part_from_lidar(small_extent):
    # a 60 x 60 m podium at 20 m with a 20 x 20 m tower at 100 m (the Capitol's shape): the
    # tower must come out as its own part, not be averaged into the podium
    poly = Polygon(rect(0, 0, 60, 60))
    tower = Polygon(rect(20, 20, 40, 40))
    ndsm, ndvi = canopy_grid()
    ndsm[cells(small_extent, poly)] = 20.0
    ndsm[cells(small_extent, tower)] = 100.0
    parts = buildings.tower_parts(poly, ndsm, ndvi, small_extent, 2.0, 20.0)
    assert len(parts) == 1
    part, h = parts[0]
    assert h == pytest.approx(100.0, abs=0.5)
    assert part.area == pytest.approx(400.0, rel=0.25) and part.difference(poly.buffer(0.01)).is_empty
    assert part.centroid.distance(tower.centroid) < 1.0


def test_no_tower_on_flat_roof_or_small_building(small_extent):
    flat = Polygon(rect(0, 0, 60, 60))
    ndsm, ndvi = canopy_grid()
    ndsm[cells(small_extent, flat)] = 12.0
    ndsm[cells(small_extent, Polygon(rect(10, 10, 16, 16)))] = 15.0  # rooftop plant 3 m up: not a tower
    assert buildings.tower_parts(flat, ndsm, ndvi, small_extent, 2.0, 12.0) == []
    house = Polygon(rect(100, 100, 115, 112))
    ndsm[cells(small_extent, house)] = 6.0
    ndsm[cells(small_extent, Polygon(rect(104, 104, 110, 110)))] = 30.0
    assert buildings.tower_parts(house, ndsm, ndvi, small_extent, 2.0, 6.0) == []  # too small to have one
    empty, _ = canopy_grid()
    assert buildings.tower_parts(flat, empty, ndvi, small_extent, 2.0, 12.0) == []


def test_make_buildings_adds_tower(small_extent):
    poly = Polygon(rect(0, 0, 60, 60))
    ndsm, ndvi = canopy_grid()
    ndsm[cells(small_extent, poly)] = 20.0
    ndsm[cells(small_extent, Polygon(rect(20, 20, 40, 40)))] = 100.0
    heights = np.full((128, 128), 5.0, np.float32)
    base, tower = buildings.make_buildings(fp("w1", rect(0, 0, 60, 60), kind="office"), ndsm, ndvi,
                                           small_extent, 2.0, heights)
    assert base.id == "w1" and base.height == pytest.approx(20.0, abs=0.5)
    assert tower.id == "w1/t0" and tower.height == pytest.approx(100.0, abs=0.5)
    assert tower.base_z == base.base_z and tower.roof == "flat" and tower.wall == "glass"
    assert tower.height_source == "lidar" and tower.source == base.source and tower.kind == base.kind


# ------------------------------------------------------------------ bases

RIVER = WaterBody("Mississippi River", box(-512, -512, 0, 512), 4.0, 15.0)


def river_terrain():
    """256 x 256 cells of 4 m over small_extent: the river (x < 0) carved to -11 m, the bank
    at 15.6 m."""
    h = np.full((256, 256), 15.6, np.float32)
    h[:, :128] = -11.0
    return h


def test_base_from_the_dry_part_of_a_footprint_over_the_river(small_extent):
    # Hollywood Casino (w545765932) has corners in the channel, carved to -11 m, and on the
    # bank at 15.6 m: its base came from the river bed and the whole building sat under the
    # 4 m surface. The points in a water body, or so near it that their terrain sample takes
    # in the carved bed (x = 1 here), are left out, unless all of them are in it.
    heights = river_terrain()
    wet = buildings.wet_areas([RIVER], small_extent, heights.shape[0])
    casino = fp("w545765932", rect(-29, 100, 31, 120), kind="commercial")
    args = (None, None, small_extent, 2.0, heights)
    assert buildings.make_building(casino, *args, wet).base_z == pytest.approx(15.6 - 0.3)
    assert buildings.make_building(casino, *args).base_z == pytest.approx(-11.0 - 0.3)  # no water given
    barge = fp("w1", rect(-100, 100, -80, 120))  # wholly in the water: its base is from all of it
    assert buildings.make_building(barge, *args, wet).base_z == pytest.approx(-11.0 - 0.3)


def test_base_at_or_below_a_dip_between_the_corners(small_extent):
    # w515634563 floated 3 m over the ground under its 99 m wall, which dips between the
    # corners: the outline is sampled every 2 m or less, its holes' walls too
    heights = np.full((256, 256), 10.0, np.float32)
    X, Y = np.meshgrid(*small_extent.pixel_centers(256))
    heights[(np.abs(X - 50) < 8) & (np.abs(Y) < 8)] = 7.0          # under the middle of the south wall
    heights[(np.abs(X - 140) < 8) & (np.abs(Y - 10) < 4)] = 6.0    # under a courtyard's west wall
    args = (None, None, small_extent, 2.0, heights)
    wall = buildings.make_building(fp("w515634563", rect(0, 0, 99, 20), kind="warehouse"), *args)
    assert wall.base_z <= 7.0 - 0.3 + 1e-6
    court = ("w2", "osm", "school", Polygon(rect(110, -20, 190, 40), [rect(140, 0, 160, 20)]), {}, None)
    assert buildings.make_building(court, *args).base_z <= 6.0 - 0.3 + 1e-6


def test_run_bases_off_the_carved_lake_bed(tmp_path, small_extent):
    cfg, raw, out = stage_inputs(tmp_path, small_extent)
    osm = json.loads((raw / "osm_features.json").read_text(encoding="utf-8"))
    osm["buildings"].append(osm_building(small_extent, "w5", rect(-262, -300, -230, -290)))  # 3/8 in the lake
    (raw / "osm_features.json").write_text(json.dumps(osm), encoding="utf-8")
    h = np.load(out / "heights.npy")
    X, Y = np.meshgrid(*small_extent.pixel_centers(256))
    h[(np.abs(X + 300) < 50) & (np.abs(Y + 300) < 50)] = -20.0  # the lake's bed
    np.save(out / "heights.npy", h)
    buildings.run(cfg)
    blds = {b.id: b for b in buildings.load_buildings(out / "buildings.json")}
    assert blds["w5"].base_z == pytest.approx(6.0 - 0.3)


def test_wall_class_large_untyped_is_not_a_house():
    # Microsoft footprints are all building=yes: a big low box is a warehouse or a store
    assert buildings.wall_class("yes", 6.0, 2000.0) == "concrete"
    assert buildings.wall_class("yes", 6.0, 150.0) == "res"
    assert buildings.wall_class("yes", 6.0) == "res"
    assert buildings.wall_class("house", 6.0, 2000.0) == "res"
    # stone civic towers (the Capitol is limestone) are not glass curtain walls
    assert buildings.wall_class("public", 110.0, 600.0) == "concrete"
    assert buildings.wall_class("office", 90.0, 600.0) == "glass"


def test_height_lidar_rejected_when_out_of_range_or_too_few_cells(small_extent):
    poly = Polygon(rect(100, 100, 120, 120))
    inside = cells(small_extent, poly)
    low, ndvi = canopy_grid()
    low[inside] = 1.0  # a lawn, not a building
    assert buildings.building_height(poly, {}, None, "house", low, ndvi, small_extent, 2.0)[1] == "default"
    few, _ = canopy_grid()
    few[tuple(np.argwhere(inside)[:2].T)] = 9.0
    assert buildings.building_height(poly, {}, None, "house", few, ndvi, small_extent, 2.0)[1] == "default"
    veg, green = canopy_grid(ndsm_value=12.0, ndvi_value=0.7)  # all vegetation
    assert buildings.building_height(poly, {}, None, "house", veg, green, small_extent, 2.0)[1] == "default"


def test_height_fallback_when_lidar_missing(small_extent):
    ndsm, ndvi = canopy_grid(ndvi_value=np.nan)
    for poly in (Polygon(rect(100, 100, 120, 120)), Polygon(rect(-511, -511, -500, -500)),
                 Polygon(rect(500, 500, 600, 600))):  # inside, on the edge, partly off the grid
        for tags, ms_height, kind in [({}, None, "yes"), ({"height": "0"}, 0.4, "house"),
                                      ({"height": "999"}, None, "yes"), ({"building:levels": "x"}, None, "shed"),
                                      ({"height": "12'"}, None, "yes")]:
            h, src = buildings.building_height(poly, tags, ms_height, kind, ndsm, ndvi, small_extent, 2.0)
            assert 2.5 <= h <= 150 and src in buildings.HEIGHT_SOURCES
    assert buildings.building_height(Polygon(rect(0, 0, 10, 10)), {}, None, "yes", None, None,
                                     small_extent, 2.0) == (5.0, "default")


def test_hip_roof_rule():
    house = Polygon(rect(0, 0, 12, 9))
    assert buildings.roof_type("house", "osm", house, 6.0) == "hip"
    assert buildings.roof_type("detached", "osm", house, 6.0) == "hip"
    assert buildings.roof_type("yes", "ms", house, 6.0) == "hip"
    ell = Polygon([(0, 0), (10, 0), (10, 4.5), (4.5, 4.5), (4.5, 10), (0, 10)])
    assert ell.area / ell.minimum_rotated_rectangle.area == pytest.approx(0.70, abs=0.01)
    assert buildings.roof_type("house", "osm", ell, 6.0) == "flat"
    assert buildings.roof_type("office", "osm", Polygon(rect(0, 0, 30, 20)), 30.0) == "flat"
    assert buildings.roof_type("yes", "osm", house, 6.0) == "flat"            # not residential
    assert buildings.roof_type("yes", "ms", Polygon(rect(0, 0, 20, 20)), 6.0) == "flat"  # 400 m2
    assert buildings.roof_type("house", "osm", house, 12.0) == "flat"         # 9 m or taller
    assert buildings.roof_type("house", "osm", house, 3.0) == "flat"          # no room for walls
    tilted = Polygon([(0, 0), (8, 6), (2.6, 13.2), (-5.4, 7.2)])             # a rotated rectangle
    assert buildings.roof_type("house", "osm", tilted, 6.0) == "hip"
    courtyard = Polygon(rect(0, 0, 12, 9), [rect(4, 3, 6, 5)])
    assert buildings.roof_type("house", "osm", courtyard, 6.0) == "flat"


def test_wall_class():
    assert buildings.wall_class("house", 6.0) == "res"
    assert buildings.wall_class("yes", 5.0) == "res"
    assert buildings.wall_class("church", 12.0) == "brick"
    assert buildings.wall_class("warehouse", 9.0) == "concrete"
    assert buildings.wall_class("yes", 12.0) == "concrete"
    assert buildings.wall_class("apartments", 25.0) == "glass"
    assert buildings.wall_class("office", 84.0) == "glass"
    assert set(buildings.WALLS) == {"res", "brick", "concrete", "glass"}


def test_materials_use_only_shared_maps():
    names = {"br_bld_wall_res", "br_bld_wall_brick", "br_bld_wall_concrete", "br_bld_wall_glass",
             "br_bld_roof_shingle", "br_bld_roof_flat"}
    assert set(buildings.MATERIALS) == names
    # every material the meshes bind is defined
    assert {*dae.WALL_MATERIALS.values(), *dae.ROOF_MATERIALS.values(), dae.COLLISION_MATERIAL} == names
    assert set(buildings.WALLS) == set(dae.WALL_MATERIALS)
    for name, m in buildings.MATERIALS.items():
        assert m["name"] == m["mapTo"] == name and m["class"] == "Material"
        assert m["persistentId"] == bf.persistent_id("material", name)
        stage = m["Stages"][0]
        assert stage["vertColor"] is True and len(m["Stages"]) == 4
        maps = [v for k, v in stage.items() if k.endswith("Map")]
        assert maps and all(v.startswith("/assets/materials/") and v.endswith(".png") for v in maps)


def test_building_textures_exist_in_the_game():
    test_game_assets = pytest.importorskip("test_game_assets")
    if not (test_game_assets.CONTENT / "assets").is_dir():
        pytest.skip("BeamNG.drive is not installed here")
    missing = [(n, p) for n, p in test_game_assets.stage_textures(buildings.MATERIALS)
               if not test_game_assets.find(p)]
    assert missing == []


# ------------------------------------------------------------------ the stage

def stage_inputs(tmp_path, small_extent, ms=True):
    raw, out = tmp_path / "raw", tmp_path / "out"
    raw.mkdir()
    osm = {"buildings": [
        osm_building(small_extent, "w1", rect(100, 100, 112, 109), building="house"),
        osm_building(small_extent, "w2", rect(-5, -20, 5, 20)),                   # on the road
        osm_building(small_extent, "w3", rect(200, -200, 230, -170), building="office",
                     **{"building:levels": "10"}),
        osm_building(small_extent, "w4", rect(-300, -300, -280, -280)),           # in the lake
    ], "trees": [], "nodes": []}
    (raw / "osm_features.json").write_text(json.dumps(osm), encoding="utf-8")
    (raw / "ms_buildings.json").write_text(json.dumps([
        {"ring": ll(small_extent, rect(150, 150, 160, 158)), "height": 4.5}]), encoding="utf-8")
    out.mkdir()
    save_roads([road([(0, -400), (0, 400)])], out / "roads.json")
    save_water([WaterBody("Lake", box(-350, -350, -250, -250), 3.0, 2.0)], out / "water.json")
    h = np.full((256, 256), 6.0, np.float32)
    h[:, 128:] = 8.0  # the east half is 2 m higher: x >= 0
    np.save(out / "heights.npy", h)
    ndsm, ndvi = canopy_grid()
    ndsm[cells(small_extent, Polygon(rect(100, 100, 112, 109)))] = 7.5
    canopy.save_canopy(out, ndsm, ndvi, 2.0)
    cfg = Config(debug_half_size_m=512.0, terrain_size=256, raw_dir=str(raw), out_dir=str(out),
                 buildings_microsoft=ms)
    return cfg, raw, out


def test_run_writes_buildings(tmp_path, small_extent, caplog):
    cfg, raw, out = stage_inputs(tmp_path, small_extent)
    with caplog.at_level(logging.INFO, logger="brmap.buildings"):
        buildings.run(cfg)
    blds = {b.id: b for b in buildings.load_buildings(out / "buildings.json")}
    assert set(blds) == {"w1", "w3", "ms0"}
    house, office, ms = blds["w1"], blds["w3"], blds["ms0"]
    assert (house.height, house.height_source, house.roof, house.wall) == (pytest.approx(7.5), "lidar", "hip", "res")
    assert (office.height, office.height_source, office.roof) == (pytest.approx(33.0), "levels", "flat")
    assert office.wall == "glass"
    assert (ms.source, ms.height, ms.height_source) == ("ms", 4.5, "ms")
    assert house.base_z == pytest.approx(8.0 - 0.3) and office.base_z == pytest.approx(8.0 - 0.3)
    assert len(house.ring) == 4 and Polygon(house.ring).exterior.is_ccw
    assert all(0 <= c <= 1 for c in house.colour)
    assert "road 1" in caplog.text and "water 1" in caplog.text
    assert "lidar 1" in caplog.text


def test_run_without_microsoft_or_canopy(tmp_path, small_extent):
    cfg, raw, out = stage_inputs(tmp_path, small_extent, ms=False)
    for f in ("ndsm.npy", "ndvi.npy", "canopy.json"):
        (out / f).unlink()
    buildings.run(cfg, raw_dir=raw, out_dir=out)
    blds = {b.id: b for b in buildings.load_buildings(out / "buildings.json")}
    assert set(blds) == {"w1", "w3"} and blds["w1"].height_source == "default"


def test_colour_is_stable_per_building():
    a = buildings.building_colour("w123", "res")
    assert a == buildings.building_colour("w123", "res") and len(a) == 3
    assert len({buildings.building_colour(f"w{i}", "res") for i in range(50)}) > 1


# ------------------------------------------------------------------ level content

def level_buildings():
    def b(id, x, y, roof="flat", wall="concrete", height=9.0):
        return Building(id=id, source="osm", kind="yes", ring=np.array(rect(x, y, x + 12, y + 9), float),
                        holes=[], height=height, base_z=5.7, roof=roof, wall=wall,
                        colour=(0.9, 0.9, 0.9), height_source="default")
    return [b("w1", 100, 100, roof="hip", wall="res", height=6.0), b("w2", 130, 100, wall="brick"),
            b("w3", -200, -150, wall="glass", height=40.0)]


def test_level_content_writes_tiles(tmp_path, small_extent):
    data, root = tmp_path / "out", tmp_path / "levels" / "baton_rouge"
    buildings.save_buildings(level_buildings(), data / "buildings.json")
    cfg = Config(debug_half_size_m=512.0, terrain_size=256)
    heights = np.full((256, 256), 6.0, np.float32)
    content = buildings.level_content(cfg, data, root, heights, small_extent)
    assert list(content) == ["buildings"]
    objs = content["buildings"]
    assert len(objs) == 2
    folder = root / "art" / "shapes" / "buildings"
    materials = json.loads((folder / "main.materials.json").read_text(encoding="utf-8"))
    used = set()
    for o in objs:
        assert o["class"] == "TSStatic" and "rotationMatrix" not in o and "__parent" not in o
        assert o["useInstanceRenderData"] is True and o["collisionType"] == "Collision Mesh"
        assert o["decalType"] == "None"
        prefix = "/levels/baton_rouge/art/shapes/buildings/br_bld_"
        assert o["shapeName"].startswith(prefix) and o["shapeName"].endswith(".dae")
        key = o["shapeName"][len(prefix):-len(".dae")]
        assert o["persistentId"] == bf.persistent_id("bld", key)
        path = folder / f"br_bld_{key}.dae"
        assert path.exists()
        tree = ET.fromstring(path.read_text(encoding="utf-8"))
        used |= {m.get("name") for m in tree.findall("c:library_materials/c:material", NS)}
        assert o["position"][2] == 0.0
    assert sorted(o["position"][:2] for o in objs) == [[-128.0, -128.0], [128.0, 128.0]]
    assert used and used <= set(materials)
    assert all(materials[n]["mapTo"] == n for n in used)


def test_level_content_lowers_base_to_cut_terrain(tmp_path, small_extent):
    data, root = tmp_path / "out", tmp_path / "lvl"
    buildings.save_buildings(level_buildings()[:1], data / "buildings.json")
    heights = np.full((256, 256), 6.0, np.float32)
    heights[100:130, 150:160] = 2.0  # terrain cut under a deck next to the house at x 100-112
    cfg = Config(debug_half_size_m=512.0, terrain_size=256)
    (obj,) = buildings.level_content(cfg, data, root, heights, small_extent)["buildings"]
    xml = (root / "art/shapes/buildings" / obj["shapeName"].rsplit("/", 1)[1]).read_text(encoding="utf-8")
    tree = ET.fromstring(xml)
    pos = np.array(tree.find(".//c:float_array", NS).text.split(), float).reshape(-1, 3)
    assert pos[:, 2].min() == pytest.approx(2.0 - 0.3, abs=0.02)
    assert pos[:, 2].max() == pytest.approx(5.7 + 6.0)  # the roof stays where it was


def test_level_content_keeps_bases_off_the_river_bed(tmp_path, small_extent):
    # the level settles each building on its terrain with the stage's rule, water left out
    data, root = tmp_path / "out", tmp_path / "lvl"
    casino = Building(id="w545765932", source="osm", kind="commercial", ring=np.array(rect(-29, 100, 31, 120), float),
                      holes=[], height=12.0, base_z=15.3, roof="flat", wall="concrete", colour=(0.9, 0.9, 0.9),
                      height_source="lidar")
    buildings.save_buildings([casino], data / "buildings.json")
    save_water([RIVER], data / "water.json")
    cfg = Config(debug_half_size_m=512.0, terrain_size=256)
    (obj,) = buildings.level_content(cfg, data, root, river_terrain(), small_extent)["buildings"]
    xml = (root / "art/shapes/buildings" / obj["shapeName"].rsplit("/", 1)[1]).read_text(encoding="utf-8")
    pos = np.array(ET.fromstring(xml).find(".//c:float_array", NS).text.split(), float).reshape(-1, 3)
    assert pos[:, 2].min() == pytest.approx(15.3, abs=0.02)


def test_settle_tower_part_keeps_its_podiums_base(small_extent):
    # a tower part stands on its podium's base: where the level lowers the podium (terrain cut
    # under a deck along its south wall) the tower goes down with it, its roof staying put
    def part(id, ring, height):
        return Building(id=id, source="osm", kind="office", ring=np.array(ring, float), holes=[], height=height,
                        base_z=5.7, roof="flat", wall="glass", colour=(0.9, 0.9, 0.9), height_source="lidar")

    heights = np.full((256, 256), 6.0, np.float32)
    X, Y = np.meshgrid(*small_extent.pixel_centers(256))
    heights[(np.abs(Y) < 6) & (np.abs(X - 30) < 20)] = 2.0
    podium, tower = buildings.settle([part("w1", rect(0, 0, 60, 60), 20.0), part("w1/t0", rect(20, 20, 40, 40), 100.0)],
                                     heights, small_extent)
    assert podium.base_z == pytest.approx(2.0 - 0.3) and podium.height == pytest.approx(24.0)
    assert tower.base_z == podium.base_z and tower.height == pytest.approx(104.0)


def test_level_writes_buildings_group(tmp_path):
    from test_level import items, make_inputs, small_cfg

    data = tmp_path / "data" / "out"
    make_inputs(data)
    buildings.save_buildings(level_buildings(), data / "buildings.json")
    root = level.write_level(small_cfg(), out_dir=tmp_path / "dist", data_dir=data)
    objs = [o for _, o in items(root) if o.get("__parent") == "buildings"]
    assert len(objs) == 2 and {o["class"] for o in objs} == {"TSStatic"}
    groups = [o["name"] for _, o in items(root) if o["class"] == "SimGroup"]
    assert "buildings" in groups
    for o in objs:
        assert (root.parent.parent / o["shapeName"].lstrip("/")).exists()
