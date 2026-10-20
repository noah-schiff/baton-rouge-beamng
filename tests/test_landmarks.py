"""Landmarks: the registry, placement on the terrain, clear zones, level objects and the checks
(landmarks.py, and its hooks in buildings, trees, validate, level and the command line)."""
import ast
import dataclasses
import json
import logging
import math
import re
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import LineString, Polygon, box

from brmap import __main__ as cli
from brmap import beamng_format as bf
from brmap import buildings, canopy, dressing, landmarks, level, trees, validate
from brmap.buildings import Building
from brmap.config import Config
from brmap.geo import Extent
from brmap.roads import Road, load_roads, save_roads
from rasters import slope_east

EXT = Extent(-91.160, 30.430, 512.0)   # the small_extent box
N = 256
FLAT = np.full((N, N), 5.0)
FLAT0 = np.zeros((N, N))
OSM = {"buildings": [], "areas": []}
CONTRACT_KEYS = {"name", "model", "x", "y", "yaw", "base_z", "ground_z", "footprint", "clear", "replaces", "params",
                 "lane_cut"}
NO_CUT = {"m2": 0.0, "share": 0.0, "pieces": 1, "holes": 0, "roads": [], "through": []}   # lane_cut: nothing cut


def ll(extent, xy):
    """[[lon, lat], ...] for level x, y points."""
    xy = np.asarray(xy, float)
    lon, lat = extent.to_lonlat(xy[:, 0], xy[:, 1])
    return [[float(a), float(b)] for a, b in zip(lon, lat)]


def square(side, cx=0.0, cy=0.0):
    h = side / 2
    return [[cx - h, cy - h], [cx + h, cy - h], [cx + h, cy + h], [cx - h, cy + h]]


def osm_shape(id, extent, ring, **tags):
    """An osm_features.json entry (closed outer ring in lon/lat, as the features stage writes it)."""
    ring = [list(p) for p in ring]
    return {"id": id, "tags": tags, "polygons": [{"outer": ll(extent, ring + ring[:1]), "inner": []}]}


def square_building(id, extent, centre=(0.0, 0.0), side=40.0):
    return osm_shape(id, extent, square(side, *centre), building="yes")


def rect_building(angle_deg, id="w1", length=60.0, width=20.0):
    """A length x width rectangle round the origin, its long side angle_deg counter-clockwise from +x."""
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    corners = [(-length / 2, -width / 2), (length / 2, -width / 2), (length / 2, width / 2), (-length / 2, width / 2)]
    return osm_shape(id, EXT, [(x * c - y * s, x * s + y * c) for x, y in corners], building="yes")


def entry(**kw):
    """A registry entry anchored on, turned with and footprinted by OSM way w1."""
    return {"model": "x.py", "anchor": {"osm": "w1"}, "yaw": {"from": "w1"}, "footprint": {"osm": ["w1"]},
            "replaces": ["w1"], "clear_pad_m": 2.0, "params": {}, "sources": {}, **kw}


def resolve_one(building=None, azimuth=None):
    e = entry() if azimuth is None else entry(yaw={"azimuth_deg": azimuth})
    osm = {"buildings": [building or square_building("w1", EXT)], "areas": []}
    (lm,) = landmarks.resolve({"t": e}, osm, EXT, FLAT)
    return lm


def reg_at(lat, lon, d=1e-4):
    ring = [[lat - d, lon - d], [lat - d, lon + d], [lat + d, lon + d], [lat + d, lon - d]]
    return {"t": entry(anchor={"lat": lat, "lon": lon}, yaw={"azimuth_deg": 0.0}, footprint={"ring": ring},
                       replaces=[])}


CLEAR = square(100)  # box(-50, -50, 50, 50)


def bld(id, inside_share=0.0):
    """A 10 x 10 m building with inside_share of its area inside CLEAR (crossing its east edge)."""
    x0 = 50.0 - 10.0 * inside_share
    return Building(id=id, source="osm", kind="yes", ring=np.array(square(10, x0 + 5, 0.0), float), holes=[],
                    height=8.0, base_z=4.7, roof="flat", wall="concrete", colour=(1.0, 1.0, 1.0),
                    height_source="default")


def lm(**kw):
    """A landmarks.json entry."""
    return {"name": "t", "model": "x.py", "x": 0.0, "y": 0.0, "yaw": 0.0, "base_z": 4.7, "ground_z": 5.0,
            "footprint": square(40), "clear": square(44), "replaces": [], "params": {}, **kw}


def ground_road(xy, oid=1, kind="residential", bridge=False, covered=False):
    xy = np.asarray(xy, float)
    return Road([oid], kind, np.column_stack([xy, np.full(len(xy), 5.0)]), 6.5, 2, False, bridge,
                1 if bridge else 0, "Test Road", covered=covered)


def tops_at(points):
    return np.asarray(points, float).reshape(-1, 3)


def contract_npz(path, nodes=("t_a100", "Colmesh-1"), material="br_lm_stucco_tan"):
    """A shape part as Blender writes it (the npz contract): one triangle per node."""
    pos = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    arrays = {"meta/nodes": np.array(list(nodes)), "meta/billboard": np.array(""),
              "meta/blender": np.array("test"), "meta/script_sha1": np.array("0" * 40)}
    for name in nodes:
        arrays |= {f"{name}/pos": pos, f"{name}/nrm": np.tile([0.0, 0.0, 1.0], (3, 1)), f"{name}/uv": pos[:, :2],
                   f"{name}/col": np.ones((3, 4)), f"{name}/mat": np.zeros(1, np.int32),
                   f"{name}/materials": np.array([material])}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


BOX_FACES = [[0, 2, 3], [0, 3, 1], [4, 5, 7], [4, 7, 6], [0, 1, 5], [0, 5, 4],
             [2, 6, 7], [2, 7, 3], [0, 4, 6], [0, 6, 2], [1, 3, 7], [1, 7, 5]]


def box_corners(lo, hi):
    """The 36 triangle corners of the box from lo to hi, (u, v, z) in model coordinates."""
    (u0, v0, z0), (u1, v1, z1) = lo, hi
    c = np.array([[u, v, z] for z in (z0, z1) for v in (v0, v1) for u in (u0, u1)], float)
    return c[np.ravel(BOX_FACES)]


def wall_corners(points, z0, z1, closed=True):
    """The triangle corners of vertical walls (two triangles each) along the (u, v) polyline, from z0
    to z1: a shell with no floor or roof, like the landmarks' collision meshes."""
    pts = [tuple(p) for p in points]
    legs = list(zip(pts, pts[1:] + pts[:1] if closed else pts[1:]))
    return np.array([c for (a, b) in legs for c in ((*a, z0), (*b, z0), (*b, z1), (*a, z0), (*b, z1), (*a, z1))], float)


def mesh_npz(path, collision, material="br_lm_stucco_tan"):
    """A shape part (the npz contract) whose collision mesh (Colmesh-1) has the given triangle
    corners, a (3T, 3) array of (u, v, z) in model coordinates, and whose visible level is one
    triangle."""
    shapes = {"t_a100": np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]), "Colmesh-1": np.asarray(collision, float)}
    arrays = {"meta/nodes": np.array(list(shapes)), "meta/billboard": np.array(""),
              "meta/blender": np.array("test"), "meta/script_sha1": np.array("0" * 40)}
    for name, pos in shapes.items():
        n = len(pos)
        arrays |= {f"{name}/pos": pos, f"{name}/nrm": np.tile([0.0, 0.0, 1.0], (n, 1)), f"{name}/uv": pos[:, :2],
                   f"{name}/col": np.ones((n, 4)), f"{name}/mat": np.zeros(n // 3, np.int32),
                   f"{name}/materials": np.array([material])}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def box_npz(path, lo, hi, material="br_lm_stucco_tan"):
    """A shape part whose collision mesh is the box from lo to hi, (u, v, z) in model coordinates."""
    return mesh_npz(path, box_corners(lo, hi), material)


def small_cfg(tmp_path, **kw):
    return Config(debug_half_size_m=512.0, terrain_size=N, raw_dir=str(tmp_path / "raw"),
                  out_dir=str(tmp_path / "out"), **kw)


# ------------------------------------------------------------------ resolve

def test_resolve_places_on_lowest_ground(small_extent):
    heights = slope_east(small_extent, 10.0, 0.05)        # z = 10 + 0.05 * (x + half)
    osm = {"buildings": [square_building("w1", small_extent, centre=(0, 0), side=40)], "areas": []}
    reg = {"t": {"model": "x.py", "anchor": {"osm": "w1"}, "yaw": {"from": "w1"},
                 "footprint": {"osm": ["w1"]}, "replaces": ["w1"], "clear_pad_m": 2.0, "params": {}, "sources": {}}}
    (lm,) = landmarks.resolve(reg, osm, small_extent, heights)
    lowest = 10 + 0.05 * (-20 + small_extent.half_size)
    assert lm["base_z"] == pytest.approx(lowest - 0.3, abs=0.05)
    assert lm["ground_z"] == pytest.approx(10 + 0.05 * small_extent.half_size, abs=0.1)
    assert lm["params"]["ground_dz"] == pytest.approx(lm["ground_z"] - lm["base_z"])


def test_yaw_from_osm_rectangle(small_extent):   # a 60 x 20 m rectangle rotated 30 degrees
    assert resolve_one(rect_building(30.0))["yaw"] == pytest.approx(math.radians(30.0), abs=1e-3)


def test_yaw_from_osm_rectangle_is_taken_modulo_pi():
    assert resolve_one(rect_building(-40.0))["yaw"] == pytest.approx(math.radians(140.0), abs=1e-3)
    assert resolve_one(rect_building(100.0, length=20.0, width=60.0))["yaw"] == pytest.approx(
        math.radians(10.0), abs=1e-3)   # the long side, not the first edge


def test_yaw_from_azimuth(small_extent):
    assert resolve_one(azimuth=156.0)["yaw"] == pytest.approx(math.radians(90 - 156.0) % math.pi, abs=1e-6)


def test_anchor_outside_box_is_skipped(small_extent, caplog):
    with caplog.at_level(logging.INFO, logger="brmap.landmarks"):
        assert landmarks.resolve(reg_at(lat=31.0, lon=-91.0), OSM, small_extent, FLAT) == []
    assert "t" in caplog.text and "outside" in caplog.text


def test_unknown_osm_id_is_skipped_with_a_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="brmap.landmarks"):
        assert landmarks.resolve({"t": entry()}, OSM, EXT, FLAT) == []
    assert "w1" in caplog.text


def test_resolve_gives_the_contract_entry():
    building = rect_building(30.0)
    building["tags"]["height"] = "12"
    osm = {"buildings": [], "areas": [building]}    # an area (building:part, man_made=tower) works too
    reg = {"t": entry(params={"height_m": 12.0}, clear_pad_m=3.0)}
    (e,) = landmarks.resolve(reg, osm, EXT, FLAT)
    assert set(e) == CONTRACT_KEYS
    assert (e["name"], e["model"], e["replaces"]) == ("t", "x.py", ["w1"])
    assert (e["x"], e["y"]) == (pytest.approx(0.0, abs=0.01), pytest.approx(0.0, abs=0.01))
    assert (e["base_z"], e["ground_z"]) == (pytest.approx(4.7), pytest.approx(5.0))
    foot, clear = Polygon(e["footprint"]), Polygon(e["clear"])
    assert foot.area == pytest.approx(1200.0, rel=1e-3) and foot.exterior.is_ccw
    assert clear.contains(foot.buffer(2.9)) and not clear.contains(foot.buffer(3.1))
    params = e["params"]
    assert set(params) == {"height_m", "footprint_local", "ground_dz"} and params["height_m"] == 12.0
    local = np.array(params["footprint_local"])     # model coordinates: +u along the yaw, +v left of it
    np.testing.assert_allclose(np.abs(local), np.tile([30.0, 10.0], (len(local), 1)), atol=0.02)
    assert Polygon(local).exterior.is_ccw
    assert params["ground_dz"] == pytest.approx(0.3)
    assert e["lane_cut"] == NO_CUT                  # no roads: nothing cut (and lane_cut is no model param)
    json.dumps(e)                                   # plain json types


def test_footprint_from_a_ring_and_anchor_from_lat_lon():
    lon, lat = EXT.to_lonlat(100.0, -50.0)
    ring = [[float(b), float(a)] for a, b in ll(EXT, square(30, 100.0, -50.0))]   # [lat, lon]
    reg = {"t": entry(anchor={"lat": lat, "lon": lon}, yaw={"azimuth_deg": 90.0}, footprint={"ring": ring},
                      replaces=[])}
    (e,) = landmarks.resolve(reg, OSM, EXT, FLAT)
    assert (e["x"], e["y"]) == (pytest.approx(100.0, abs=0.01), pytest.approx(-50.0, abs=0.01))
    assert e["yaw"] == pytest.approx(0.0, abs=1e-9)  # azimuth 90 (east) is +x
    assert Polygon(e["footprint"]).area == pytest.approx(900.0, rel=1e-3)
    np.testing.assert_allclose(np.abs(e["params"]["footprint_local"]), 15.0, atol=0.02)


# ------------------------------------------------------------------ the rim from the lidar, the pitch

def circle(r, cx=0.0, cy=0.0, n=72):
    a = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    return [[cx + r * math.cos(t), cy + r * math.sin(t)] for t in a]


def rect(angle_deg, length, width, centre=(0.0, 0.0)):
    """A length x width rectangle round centre, its long side angle_deg counter-clockwise from +x."""
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    half = [(-length / 2, -width / 2), (length / 2, -width / 2), (length / 2, width / 2), (-length / 2, width / 2)]
    return [(centre[0] + x * c - y * s, centre[1] + x * s + y * c) for x, y in half]


def osm_holed(id, extent, ring, hole, **tags):
    """An osm_features.json entry with one hole (a bowl round its field)."""
    e = osm_shape(id, extent, ring, **tags)
    hole = [list(p) for p in hole]
    e["polygons"][0]["inner"] = [ll(extent, hole + hole[:1])]
    return e


def ring_ndsm(extent, centre, r_in, r_out, north, south, field=0.0, outside=0.0):
    """The 2 m analysis grid (canopy.analysis_grid) with `north` m on the ring between r_in and
    r_out north of centre and `south` m south of it; `field` inside r_in, `outside` beyond r_out."""
    n, px = canopy.analysis_grid(extent)
    xs, ys = extent.pixel_centers(n)
    X, Y = np.meshgrid(xs, ys)
    r = np.hypot(X - centre[0], Y - centre[1])
    ndsm = np.where(Y > centre[1], north, south).astype(np.float32)
    ndsm[r <= r_in], ndsm[r >= r_out] = field, outside
    return ndsm, px


def test_rim_profile_reads_sectors():
    ndsm, px = ring_ndsm(EXT, (20.0, -10.0), 60.0, 120.0, north=20.0, south=48.0)
    outer, inner = Polygon(circle(120.0, 20.0, -10.0)), Polygon(circle(60.0, 20.0, -10.0))
    profile = landmarks.rim_profile(outer, inner, ndsm, EXT, px)
    assert [a for a, _ in profile] == pytest.approx(list(range(0, 360, 15)))   # one per 15 deg sector
    h = dict((round(a), h) for a, h in profile)
    assert h[0] == pytest.approx(20.0, abs=2.0) and h[180] == pytest.approx(48.0, abs=2.0)
    assert 20.0 < h[90] < 48.0 and 20.0 < h[270] < 48.0      # east and west straddle the two halves
    json.dumps(profile)                                       # plain floats


def test_rim_profile_reads_only_the_seats_between_the_rings():
    ndsm, px = ring_ndsm(EXT, (0.0, 0.0), 60.0, 120.0, north=30.0, south=30.0, field=90.0, outside=90.0)
    profile = landmarks.rim_profile(Polygon(circle(120.0)), Polygon(circle(60.0)), ndsm, EXT, px)
    assert [h for _, h in profile] == pytest.approx([30.0] * 24)


def test_rim_profile_fills_a_sector_the_lidar_missed():
    ndsm, px = ring_ndsm(EXT, (0.0, 0.0), 60.0, 120.0, north=30.0, south=30.0)
    xs, ys = EXT.pixel_centers(ndsm.shape[0])
    X, Y = np.meshgrid(xs, ys)
    az = np.degrees(np.arctan2(X, Y)) % 360
    ndsm[(az > 75) & (az < 105)] = np.nan                     # the lidar gap: no cells round east
    profile = landmarks.rim_profile(Polygon(circle(120.0)), Polygon(circle(60.0)), ndsm, EXT, px)
    assert all(math.isfinite(h) for _, h in profile)
    assert dict((round(a), h) for a, h in profile)[90] == pytest.approx(30.0)
    with pytest.raises(ValueError, match="no lidar"):
        landmarks.rim_profile(Polygon(circle(120.0)), Polygon(circle(60.0)), np.full_like(ndsm, np.nan), EXT, px)


def stadium(params, heights=FLAT, ndsm=None, px=None, pitch=None):
    """resolve() of one bowl: a 300 x 260 m outline round a 130 x 80 m field hole, turned to
    azimuth 156 like Tiger Stadium, with the pitch (an area) when given."""
    bowl = osm_holed("w1", EXT, square(300.0), rect(0.0, 130.0, 80.0), building="stadium")
    osm = {"buildings": [bowl], "areas": [osm_shape("w2", EXT, pitch, leisure="pitch")] if pitch else []}
    reg = {"t": entry(yaw={"azimuth_deg": 156.0}, params=params)}
    return landmarks.resolve(reg, osm, EXT, heights, ndsm, px)


def test_pitch_params():
    centre = (10.0, -20.0)
    (e,) = stadium({"pitch": "w2"}, pitch=rect(90.0 - 156.0, 111.0, 50.0, centre))
    p = e["params"]
    assert p["pitch_yaw"] == pytest.approx(0.0, abs=1e-3)      # the model is aligned to the field
    want = landmarks.to_local([centre], e["x"], e["y"], e["yaw"])[0]
    assert math.dist(p["pitch_local"], want) < 1.0
    json.dumps(e)


def test_pitch_yaw_is_the_field_axis_in_model_coordinates():
    (e,) = stadium({"pitch": "w2"}, pitch=rect(90.0 - 156.0 + 10.0, 111.0, 50.0))
    assert e["params"]["pitch_yaw"] == pytest.approx(math.radians(10.0), abs=1e-3)
    (e,) = stadium({"pitch": "w2"}, pitch=rect(90.0 - 156.0 - 10.0 + 180.0, 111.0, 50.0))
    assert e["params"]["pitch_yaw"] == pytest.approx(math.radians(-10.0), abs=1e-3)   # modulo pi, near 0


def test_field_dz_is_the_highest_ground_on_the_field():
    heights = slope_east(EXT, 10.0, 0.05)                 # the field hole spans x -65 .. 65
    (e,) = stadium({"pitch": "w2"}, heights=heights, pitch=rect(-66.0, 111.0, 50.0))
    top = 10 + 0.05 * (65.0 + EXT.half_size)
    cell = 2 * EXT.half_size / N
    assert e["params"]["field_dz"] + e["base_z"] == pytest.approx(top, abs=0.05 * 1.5 * cell)
    assert e["params"]["field_dz"] > e["params"]["ground_dz"]


def test_rim_lidar_puts_the_profile_in_model_coordinates():
    ndsm, px = ring_ndsm(EXT, (0.0, 0.0), 40.0, 140.0, north=20.0, south=48.0)
    (e,) = stadium({"rim": "lidar", "rim_m": [20, 55]}, ndsm=ndsm, px=px)
    p = e["params"]
    angles = [a for a, _ in p["rim_profile"]]
    assert len(angles) == 24 and angles == sorted(angles) and 0 <= angles[0] and angles[-1] < 360
    # +u points to azimuth 336 (yaw 114 deg): north is 24 deg right of +u (angle 336, counter-clockwise
    # from +u) and south 156
    rim = {round(a): h for a, h in p["rim_profile"]}
    assert rim[336] == pytest.approx(20.0, abs=2.0) and rim[156] == pytest.approx(48.0, abs=2.0)
    hole = Polygon(p["hole_local"])
    assert hole.area == pytest.approx(130.0 * 80.0, rel=1e-3) and hole.exterior.is_ccw
    assert Polygon(p["footprint_local"]).contains(hole)
    assert p["rim_m"] == [20, 55]
    json.dumps(e)


def test_rim_lidar_without_the_canopy_is_skipped(caplog):
    with caplog.at_level(logging.WARNING, logger="brmap.landmarks"):
        assert stadium({"rim": "lidar"}) == []
    assert "t" in caplog.text and "ndsm" in caplog.text


def osm_parts(id, extent, parts, **tags):
    """An osm_features.json entry of several polygons, each (outer ring, [holes])."""
    def closed(ring):
        ring = [list(p) for p in ring]
        return ll(extent, ring + ring[:1])
    return {"id": id, "tags": tags,
            "polygons": [{"outer": closed(outer), "inner": [closed(h) for h in holes]} for outer, holes in parts]}


def test_field_params():
    # a ball field (leisure=pitch): the playing field round the infield hole, and the infield
    # grass inside that hole round the mound's hole; heights slope up to the east
    heights = slope_east(EXT, 10.0, 0.05)
    field = osm_parts("w2", EXT, [(square(100.0, 30.0), [square(40.0, 30.0)]),
                                  (square(30.0, 30.0), [square(4.0, 30.0)])], leisure="pitch")
    osm = {"buildings": [square_building("w1", EXT, side=160.0)], "areas": [field]}
    (e,) = landmarks.resolve({"t": entry(yaw={"azimuth_deg": 90.0}, params={"field": "w2"})}, osm, EXT, heights)
    p = e["params"]
    big, small = p["field_local"]                                  # the larger polygon first
    assert len(big) == 2 and len(small) == 2                       # [outer ring, hole] each
    outer, hole = Polygon(big[0]), Polygon(big[1])
    assert outer.area == pytest.approx(100.0 ** 2, rel=1e-3) and outer.exterior.is_ccw
    assert hole.area == pytest.approx(40.0 ** 2, rel=1e-3) and not hole.exterior.is_ccw
    assert Polygon(small[1]).area == pytest.approx(16.0, rel=1e-3)
    # model coordinates: yaw azimuth 90 puts +u east, so the field's centre is 30 m along +u
    assert outer.centroid.coords[0] == pytest.approx((30.0, 0.0), abs=0.05)
    top = 10 + 0.05 * (80.0 + EXT.half_size)                       # the field's east edge
    cell = 2 * EXT.half_size / N
    assert p["field_dz"] + e["base_z"] == pytest.approx(top, abs=0.05 * 1.5 * cell)
    json.dumps(e)


def test_outlines_params():
    bleachers = [osm_shape("w2", EXT, rect(30.0, 20.0, 8.0, (60.0, 0.0)), building="stadium"),
                 osm_shape("w3", EXT, square(10.0, -60.0, 20.0)[::-1], building="stadium")]
    osm = {"buildings": [square_building("w1", EXT, side=40.0)] + bleachers, "areas": []}
    (e,) = landmarks.resolve({"t": entry(yaw={"azimuth_deg": 0.0}, params={"outlines": ["w2", "w3"]})}, osm, EXT,
                             FLAT)
    a, b = e["params"]["outlines_local"]
    assert Polygon(a).area == pytest.approx(160.0, rel=1e-3) and Polygon(a).exterior.is_ccw
    assert Polygon(b).area == pytest.approx(100.0, rel=1e-3) and Polygon(b).exterior.is_ccw   # turned round
    # yaw azimuth 0 puts +u north and +v west: the bleacher 60 m east of the anchor is at v -60
    assert Polygon(a).centroid.coords[0] == pytest.approx((0.0, -60.0), abs=0.05)
    json.dumps(e)


def test_outlines_name_a_missing_id():
    osm = {"buildings": [square_building("w1", EXT)], "areas": []}
    reg = {"t": entry(params={"outlines": ["w404"]})}
    assert landmarks.resolve(reg, osm, EXT, FLAT) == []


def deck_road(xy, oid, z, width=13.8):
    xy = np.asarray(xy, float)
    return Road([oid], "motorway", np.column_stack([xy, np.broadcast_to(z, len(xy))]), width, 3, True, True, 2,
                "Test Bridge")


def lon_at(x, y=0.0):
    return float(EXT.to_lonlat(x, y)[0])


def truss_entry(decks=("w7", "w8")):
    """A registry entry over two decks: anchored on (0, 0), +u east, from x -110 to 170, peaks at 0 and 60."""
    ring = [[float(lat), float(lon)] for lon, lat in (EXT.to_lonlat(x, y) for x, y in square(400, 25, 7))]
    lon, lat = EXT.to_lonlat(0.0, 0.0)
    return {"model": "x.py", "anchor": {"lat": float(lat), "lon": float(lon)}, "yaw": {"azimuth_deg": 90},
            "footprint": {"ring": ring}, "replaces": [], "clear_pad_m": 0, "sources": {},
            "params": {"decks": list(decks), "from_lon": lon_at(-110), "to_lon": lon_at(170),
                       "peaks_lon": [lon_at(0), lon_at(60)], "top_m": [16, 24]}}


def test_deck_local_samples_between_ends():
    xs = np.arange(-300.0, 301.0, 20.0)
    east = deck_road(np.column_stack([xs, np.zeros_like(xs)]), 7, 20 + xs / 100)
    west = deck_road(np.column_stack([xs[::-1], np.full(len(xs), 15.0)]), 8, 21 + xs[::-1] / 100, width=12.0)
    other = deck_road(np.column_stack([xs, np.full(len(xs), -40.0)]), 9, 20.0)
    (e,) = landmarks.resolve({"truss": truss_entry()}, OSM, EXT, FLAT, roads=[other, west, east])
    p = e["params"]
    assert [d["id"] for d in p["deck_local"]] == ["w7", "w8"]
    assert [d["width"] for d in p["deck_local"]] == [13.8, 12.0]
    for d, v, z0 in zip(p["deck_local"], (0.0, 15.0), (20.0, 21.0)):
        u, vv, z = np.asarray(d["nodes"], float).T
        assert u.tolist() == sorted(u.tolist())                                  # in u order, whichever way drawn
        np.testing.assert_allclose(u, np.arange(-100.0, 161.0, 20.0), atol=0.01)   # the nodes between the ends
        np.testing.assert_allclose(vv, v, atol=0.01)
        np.testing.assert_allclose(z, z0 + u / 100 - e["base_z"], atol=0.01)      # up from base_z
    assert (p["from_u"], p["to_u"]) == (pytest.approx(-110.0, abs=0.01), pytest.approx(170.0, abs=0.01))
    assert p["peaks_u"] == [pytest.approx(0.0, abs=0.01), pytest.approx(60.0, abs=0.01)]
    json.dumps(e)


def test_deck_local_needs_its_decks(caplog):
    xs = np.arange(-300.0, 301.0, 20.0)
    east = deck_road(np.column_stack([xs, np.zeros_like(xs)]), 7, 20.0)
    with caplog.at_level(logging.WARNING, logger="brmap.landmarks"):
        assert landmarks.resolve({"truss": truss_entry()}, OSM, EXT, FLAT, roads=[east]) == []
    assert "truss" in caplog.text and "w8" in caplog.text


def test_footprint_keeps_off_the_lanes_beside_it(caplog):
    # a 40 m square (+-20 m) beside a 6.5 m road along y = 21.85 whose corridor (width / 2) clips it by 1.4 m:
    # the footprint keeps LANE_CLEAR off that corridor and loses only the 1.9 m sliver. A road farther off, a
    # bridge deck and a covered passage (drivability 0) across it cut nothing
    beside, far = ground_road([(-100, 21.85), (100, 21.85)], oid=5), ground_road([(-100, 60), (100, 60)], oid=6)
    deck = ground_road([(-100, 0), (100, 0)], oid=7, bridge=True)
    passage = ground_road([(0, -100), (0, 100)], oid=8, covered=True)
    osm = {"buildings": [square_building("w1", EXT)], "areas": []}
    with caplog.at_level(logging.INFO, logger="brmap.landmarks"):
        (e,) = landmarks.resolve({"t": entry(yaw={"azimuth_deg": 90.0})}, osm, EXT, FLAT,
                                 roads=[beside, far, deck, passage])
    foot = Polygon(e["footprint"])
    assert foot.distance(LineString(beside.nodes[:, :2])) == pytest.approx(3.25 + landmarks.LANE_CLEAR, abs=0.01)
    assert foot.bounds == pytest.approx((-20.0, -20.0, 20.0, 18.1), abs=0.01)
    assert foot.area == pytest.approx(40.0 * 38.1, rel=1e-4) and foot.exterior.is_ccw
    assert (e["x"], e["y"]) == (pytest.approx(0.0, abs=0.01), pytest.approx(0.0, abs=0.01))   # the OSM anchor
    local = Polygon(e["params"]["footprint_local"])                  # yaw 0: model u, v are level x, y
    assert local.equals_exact(foot, 0.01) and local.exterior.is_ccw
    clear = Polygon(e["clear"])                                      # clear_pad_m 2 round the cut footprint
    assert clear.contains(foot.buffer(1.9)) and not clear.contains(foot.buffer(2.1))
    assert "t" in caplog.text and "way 5 (Test Road) 76.0 m2" in caplog.text
    assert "way 6" not in caplog.text and "way 7" not in caplog.text and "way 8" not in caplog.text
    assert e["lane_cut"] == {"m2": pytest.approx(76.0), "share": pytest.approx(76.0 / 1600.0), "pieces": 1,
                             "holes": 0, "roads": [{"road": "way 5 (Test Road)", "m2": pytest.approx(76.0)}],
                             "through": []}


def test_footprint_cut_in_two_keeps_the_larger_piece():
    # a road across the footprint along x = -14: its corridor (+-3.75 m) leaves a 2.25 m strip to the west and
    # the 30.25 m rest to the east, which is kept
    across = ground_road([(-14, -100), (-14, 100)])
    osm = {"buildings": [square_building("w1", EXT)], "areas": []}
    (e,) = landmarks.resolve({"t": entry(yaw={"azimuth_deg": 90.0})}, osm, EXT, FLAT, roads=[across])
    foot = Polygon(e["footprint"])
    assert foot.bounds == pytest.approx((-10.25, -20.0, 20.0, 20.0), abs=0.01) and foot.exterior.is_ccw
    # the corridor takes 7.5 m x 40 m, the 2.25 m strip west of it goes with the piece not kept
    assert e["lane_cut"] == {"m2": pytest.approx(390.0), "share": pytest.approx(390.0 / 1600.0), "pieces": 2,
                             "holes": 0, "roads": [{"road": "way 1 (Test Road)", "m2": pytest.approx(300.0)}],
                             "through": ["way 1 (Test Road)"]}     # its centreline crosses the uncut footprint


def test_the_cut_log_counts_what_was_really_removed(caplog):
    # a short road wholly inside the footprint leaves a hole, which the kept outline fills again: it removes nothing
    inside = ground_road([(-5, 0), (5, 0)], oid=6)
    osm = {"buildings": [square_building("w1", EXT)], "areas": []}
    with caplog.at_level(logging.INFO, logger="brmap.landmarks"):
        (e,) = landmarks.resolve({"t": entry(yaw={"azimuth_deg": 90.0})}, osm, EXT, FLAT, roads=[inside])
    assert Polygon(e["footprint"]).area == pytest.approx(1600.0, rel=1e-4)
    assert "way 6 (Test Road) 0.0 m2" in caplog.text and "0.0 of 1600.0 m2 (0.00 %) cut" in caplog.text
    assert "1 hole" in caplog.text
    assert e["lane_cut"] == {"m2": pytest.approx(0.0, abs=1e-3), "share": pytest.approx(0.0, abs=1e-6), "pieces": 1,
                             "holes": 1, "roads": [{"road": "way 6 (Test Road)", "m2": pytest.approx(0.0, abs=1e-3)}],
                             "through": ["way 6 (Test Road)"]}


def test_the_cut_log_counts_an_overlap_of_corridors_once(caplog):
    # the corridors of two roads at right angles beside the footprint overlap in its north-east corner
    # (1.9 x 1.9 m): the first road takes it, the second only the rest of its 76 m2
    north, east = ground_road([(-100, 21.85), (100, 21.85)], oid=5), ground_road([(21.85, -100), (21.85, 100)], oid=6)
    osm = {"buildings": [square_building("w1", EXT)], "areas": []}
    with caplog.at_level(logging.INFO, logger="brmap.landmarks"):
        (e,) = landmarks.resolve({"t": entry(yaw={"azimuth_deg": 90.0})}, osm, EXT, FLAT, roads=[north, east])
    assert "way 5 (Test Road) 76.0 m2, way 6 (Test Road) 72.4 m2" in caplog.text
    assert "148.4 of 1600.0 m2 (9.27 %) cut" in caplog.text
    assert e["lane_cut"]["m2"] == pytest.approx(76.0 + 76.0 - 1.9 ** 2)
    assert [r["m2"] for r in e["lane_cut"]["roads"]] == [pytest.approx(76.0), pytest.approx(76.0 - 1.9 ** 2)]


def test_corridors_that_cover_the_whole_footprint_are_an_error():
    wide = dataclasses.replace(ground_road([(-100, 0), (100, 0)]), width=50.0)   # 25.5 m each side with LANE_CLEAR
    osm = {"buildings": [square_building("w1", EXT)], "areas": []}
    with pytest.raises(ValueError, match="landmark t: the roads' corridors cover its whole footprint"):
        landmarks.resolve({"t": entry(yaw={"azimuth_deg": 90.0})}, osm, EXT, FLAT, roads=[wide])


def test_a_landmark_can_keep_its_whole_footprint():
    # params.keep_off_roads false (for a landmark the cut would cost over 2 % of its footprint): resolve leaves the
    # footprint whole and validate leaves the lanes beside it alone, not a road through it
    beside, through = ground_road([(-100, 21.85), (100, 21.85)], oid=5), ground_road([(-100, 0), (100, 0)], oid=6)
    osm = {"buildings": [square_building("w1", EXT)], "areas": []}
    reg = {"t": entry(yaw={"azimuth_deg": 90.0}, params={"keep_off_roads": False})}
    (e,) = landmarks.resolve(reg, osm, EXT, FLAT, roads=[beside])
    assert Polygon(e["footprint"]).area == pytest.approx(1600.0, rel=1e-4)
    assert e["params"]["keep_off_roads"] is False and e["lane_cut"] == NO_CUT
    assert validate.landmark_problems([e], [beside], FLAT, EXT) == []
    assert validate.landmark_problems([e], [beside, through], FLAT, EXT) == [
        "t: road way 6 (Test Road) passes through it"]


def test_a_landmark_keeps_off_every_road_but_the_ways_it_names(caplog):
    # params.keep_off_roads {except: [w5]} (the Capitol: two service drives end at its walls, where the round ends
    # of their corridors would notch them): resolve cuts the footprint by every other road and validate checks
    # their lanes; it leaves w5's lanes alone, but not w5 passing through it
    north, east = ground_road([(-100, 21.85), (100, 21.85)], oid=5), ground_road([(21.85, -100), (21.85, 100)], oid=6)
    osm = {"buildings": [square_building("w1", EXT)], "areas": []}
    rule = {"except": ["w5"]}
    reg = {"t": entry(yaw={"azimuth_deg": 90.0}, params={"keep_off_roads": rule})}
    with caplog.at_level(logging.INFO, logger="brmap.landmarks"):
        (e,) = landmarks.resolve(reg, osm, EXT, FLAT, roads=[north, east])
    assert Polygon(e["footprint"]).bounds == pytest.approx((-20.0, -20.0, 18.1, 20.0), abs=0.01)
    assert "way 6 (Test Road) 76.0 m2" in caplog.text                     # what east cut
    assert "t leaves its footprint whole by way 5 (Test Road) 76.0 m2" in caplog.text
    assert "WARNING" not in caplog.text
    assert validate.landmark_problems([e], [north, east], FLAT, EXT) == [      # only what east cut (4.8 %), not north
        "t: the lane cut removed 4.8 % of its footprint (76 m2), by way 6 (Test Road) 76.0 m2"]
    whole = lm(footprint=square(40), params={"keep_off_roads": rule})
    assert validate.landmark_problems([whole], [north, east], FLAT, EXT) == [
        "t: the corridor of road way 6 (Test Road) reaches into it near (20, 0)"]
    through = ground_road([(-100, 0), (100, 0)], oid=5)
    assert validate.landmark_problems([whole], [through], FLAT, EXT) == ["t: road way 5 (Test Road) passes through it"]


def test_keep_off_roads_warns_of_a_named_way_that_does_not_reach_it(caplog):
    # a typo in the registry, or a way that has a new id since the last fetch: its lanes would be cut after all
    far = ground_road([(-100, 60), (100, 60)], oid=6)
    osm = {"buildings": [square_building("w1", EXT)], "areas": []}
    reg = {"t": entry(yaw={"azimuth_deg": 90.0}, params={"keep_off_roads": {"except": ["w6", "w9"]}})}
    with caplog.at_level(logging.WARNING, logger="brmap.landmarks"):
        (e,) = landmarks.resolve(reg, osm, EXT, FLAT, roads=[far])
    assert Polygon(e["footprint"]).area == pytest.approx(1600.0, rel=1e-4)
    warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warned) == 2 and all(m.startswith("landmark t: keep_off_roads names w") for m in warned)
    assert "w6" in warned[0] and "w9" in warned[1]


@pytest.mark.parametrize("rule", ["false", {"only": ["w5"]}, {"except": "w5"}, {"except": ["w5"], "also": []}])
def test_keep_off_roads_is_true_false_or_named_ways(rule):
    beside = ground_road([(-100, 21.85), (100, 21.85)], oid=5)
    osm = {"buildings": [square_building("w1", EXT)], "areas": []}
    with pytest.raises(ValueError, match="landmark t: keep_off_roads is true, false or"):
        landmarks.resolve({"t": entry(params={"keep_off_roads": rule})}, osm, EXT, FLAT, roads=[beside])
    with pytest.raises(ValueError, match="landmark t: keep_off_roads is true, false or"):
        validate.landmark_problems([lm(params={"keep_off_roads": rule})], [beside], FLAT, EXT)


def test_a_landmark_over_decks_is_not_cut():
    xs = np.arange(-300.0, 301.0, 20.0)
    east = deck_road(np.column_stack([xs, np.zeros_like(xs)]), 7, 20.0)
    west = deck_road(np.column_stack([xs[::-1], np.full(len(xs), 15.0)]), 8, 21.0)
    under = ground_road([(-300, 100), (300, 100)], oid=9)          # a road under the truss, across its footprint
    (e,) = landmarks.resolve({"truss": truss_entry()}, OSM, EXT, FLAT, roads=[east, west, under])
    assert Polygon(e["footprint"]).area == pytest.approx(400.0 ** 2, rel=1e-3) and e["lane_cut"] == NO_CUT


def test_rim_lidar_needs_a_field_hole():
    ndsm, px = ring_ndsm(EXT, (0.0, 0.0), 40.0, 140.0, north=20.0, south=48.0)
    reg = {"t": entry(params={"rim": "lidar"})}
    with pytest.raises(ValueError, match="t.*hole"):
        landmarks.resolve(reg, {"buildings": [square_building("w1", EXT, side=300.0)], "areas": []}, EXT, FLAT,
                          ndsm, px)


# ------------------------------------------------------------------ removed buildings, clear zones

def test_removed_ids_parts_and_share():
    blds = [bld("w1"), bld("w1/t0"), bld("w2", inside_share=0.6), bld("w3", inside_share=0.3)]
    assert landmarks.removed(blds, [lm(replaces=["w1"], clear=CLEAR)]) == {"w1", "w1/t0", "w2"}


def test_removed_by_names_each_landmark(caplog):
    blds = [bld("w1"), bld("w2", inside_share=0.6), bld("w2/t0"), bld("w9", inside_share=0.0)]
    with caplog.at_level(logging.WARNING, logger="brmap.landmarks"):
        got = landmarks.removed_by(blds, [lm(name="a", replaces=["w1", "w404"], clear=square(4, 500, 500)),
                                          lm(name="b", clear=CLEAR)])
    assert got == {"a": {"w1"}, "b": {"w2", "w2/t0"}}    # a replaced id that isn't there is no removal, but a warning
    assert any("w404" in m and "matches no building" in m for m in caplog.messages)
    assert landmarks.removed([], [lm(replaces=["w1"])]) == set() and landmarks.removed(blds, []) == set()


def test_removed_by_matches_replaces_by_osm_element():
    # make_valid splits an OSM way into parts '<id>/<k>': replaces [w5] takes them all, not w50
    blds = [bld("w5/0", inside_share=0.2), bld("w5/1", inside_share=0.2), bld("w50", inside_share=0.2)]
    assert landmarks.removed_by(blds, [lm(replaces=["w5"], clear=CLEAR)]) == {"t": {"w5/0", "w5/1"}}


def test_removed_by_warns_of_a_replaced_building_outside_the_clear_zone(caplog):
    # a typo in the registry, or an id that names another building after a new fetch
    away = dataclasses.replace(bld("w2"), ring=np.array(square(10, 300.0, 0.0), float))
    with caplog.at_level(logging.WARNING, logger="brmap.landmarks"):
        got = landmarks.removed_by([bld("w1", inside_share=0.3), away], [lm(name="hall", replaces=["w1", "w2"],
                                                                             clear=CLEAR)])
    assert got == {"hall": {"w1", "w2"}}
    (rec,) = caplog.records
    assert "hall" in rec.getMessage() and "w2" in rec.getMessage() and "w1" not in rec.getMessage()


def test_removed_by_warns_of_a_replaced_id_that_matches_no_building(caplog):
    # a typo in the registry, or an OSM element that is no building (here or any more): w5/0 and w5/1 count as w5
    blds = [bld("w5/0", inside_share=0.3), bld("w5/1", inside_share=0.3)]
    with caplog.at_level(logging.WARNING, logger="brmap.landmarks"):
        got = landmarks.removed_by(blds, [lm(name="hall", replaces=["w5", "w404", "w50"], clear=CLEAR)])
    assert got == {"hall": {"w5/0", "w5/1"}}
    (w404, w50) = [r.getMessage() for r in caplog.records]       # one warning per id, in replaces order
    assert w404.startswith("landmark hall replaces w404, which matches no building")
    assert w50.startswith("landmark hall replaces w50, which matches no building")
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="brmap.landmarks"):
        landmarks.removed_by(blds, [lm(name="hall", replaces=["w5"], clear=CLEAR), lm(name="empty", replaces=[])])
    assert caplog.records == []


def test_trees_cleared():
    kept, dropped = trees.exclude(tops_at([(0, 0, 10), (100, 0, 10)]), [], [], [], clear=[box(-5, -5, 5, 5)])
    assert len(kept) == 1 and dropped["landmark"] == 1


def test_clear_polygons(tmp_path):
    path = tmp_path / "landmarks.json"
    assert landmarks.clear_polygons(path) == []      # the landmarks stage has not run
    path.write_text(json.dumps([lm(clear=CLEAR), lm(name="u", clear=square(10, 200, 0))]), encoding="utf-8")
    polys = landmarks.clear_polygons(path)
    assert [p.bounds for p in polys] == [(-50, -50, 50, 50), (195, -5, 205, 5)]


# ------------------------------------------------------------------ what the models stand on

def test_ground_outline_is_the_footprint_without_a_model(tmp_path):
    assert landmarks.ground_outline(lm(footprint=square(40)), tmp_path / "models").equals(Polygon(square(40)))


def test_ground_outline_from_the_collision_mesh(tmp_path):
    # model coordinates: +u along the yaw (here 90 degrees: north), +v left of it (west); ground_dz 0.5,
    # so the collision vertices under 3.5 m stand on the ground
    models = tmp_path / "models"
    box_npz(models / "t" / "main.npz", (-20, -20, 0.0), (30, 10, 12.0))       # on the ground
    box_npz(models / "t" / "roof.npz", (-50, -50, 3.6), (50, 50, 6.0))        # from 3.1 m above it: left out
    g = landmarks.ground_outline(lm(x=100.0, y=50.0, yaw=math.pi / 2, params={"ground_dz": 0.5}), models)
    assert g.geom_type == "Polygon" and g.area == pytest.approx(1500.0)
    assert g.bounds == pytest.approx((90.0, 30.0, 120.0, 80.0))            # v -20..10 -> x 110..80, u -> y


def test_ground_outline_follows_a_concave_collision_mesh(tmp_path):
    # a U of three wall boxes: 40 x 40 m outside, the inside 20 x 30 m open towards +v. The convex hull of
    # its vertices is the whole 40 x 40 m square; the collision triangles cover only the U (1000 m2)
    models = tmp_path / "models"
    walls = [((-20, -20, 0.0), (-10, 20, 10.0)), ((10, -20, 0.0), (20, 20, 10.0)), ((-10, -20, 0.0), (10, -10, 10.0))]
    mesh_npz(models / "t" / "main.npz", np.vstack([box_corners(lo, hi) for lo, hi in walls]))
    g = landmarks.ground_outline(lm(params={"ground_dz": 0.3}), models)
    assert g.area == pytest.approx(1000.0, rel=0.02)
    assert not g.intersects(box(-9, -9, 9, 19))                   # the U's inside (its open end included)
    assert g.contains(box(-19, -19, 19, -11))                     # the back wall
    assert g.bounds == pytest.approx((-20.0, -20.0, 20.0, 20.0))


def test_ground_outline_fills_what_collision_walls_enclose(tmp_path):
    # the landmarks' collision meshes are floorless shells: vertical walls with no triangle that has an area
    # at the ground. A U-shaped wall ring (outside 40 x 40 m, the notch 20 x 30 m open towards +v) encloses
    # 1000 m2; a fence standing apart, which encloses nothing, adds nothing
    models = tmp_path / "models"
    u_ring = [(-20, -20), (20, -20), (20, 20), (10, 20), (10, -10), (-10, -10), (-10, 20), (-20, 20)]
    mesh_npz(models / "t" / "main.npz", wall_corners(u_ring, 0.0, 10.0))
    mesh_npz(models / "t" / "fence.npz", wall_corners([(60, -20), (60, 20)], 0.0, 2.0, closed=False))
    g = landmarks.ground_outline(lm(params={"ground_dz": 0.3}), models)
    assert g.geom_type == "Polygon" and g.area == pytest.approx(1000.0, rel=0.02)
    assert g.equals(Polygon(u_ring))
    assert not g.intersects(box(-9, -9, 9, 19))


def test_ground_outline_is_the_footprint_when_the_walls_enclose_nothing(tmp_path):
    models = tmp_path / "models"
    mesh_npz(models / "t" / "main.npz", wall_corners([(60, -20), (60, 20)], 0.0, 2.0, closed=False))
    assert landmarks.ground_outline(lm(footprint=square(40)), models).equals(Polygon(square(40)))


def test_ground_outline_takes_a_whole_triangle_that_touches_the_ground(tmp_path):
    # the ramp's triangle has one corner at the ground and two 10 m up: it counts, with its whole shadow;
    # a triangle that stays above 3.3 m, and a lone vertical one (it encloses nothing), do not
    models = tmp_path / "models"
    ramp = [[0, 0, 0.0], [10, 0, 10.0], [0, 10, 10.0]]
    high = [[50, 50, 4.0], [60, 50, 4.0], [50, 60, 4.0]]
    wall = [[-30, 0, 0.0], [-20, 0, 0.0], [-20, 0, 5.0]]
    mesh_npz(models / "t" / "main.npz", ramp + high + wall)
    g = landmarks.ground_outline(lm(params={"ground_dz": 0.3}), models)
    assert g.geom_type == "Polygon" and g.area == pytest.approx(50.0)
    assert g.equals(Polygon([(0, 0), (10, 0), (0, 10)]))


def test_clear_zones_cover_what_the_model_stands_on(tmp_path):
    # the collision box reaches 18 m past the clear zone (+-22 m) to the south
    models = tmp_path / "models"
    box_npz(models / "t" / "main.npz", (-20, -40, 0.0), (20, 20, 10.0))
    e = lm(clear=square(44), params={"ground_dz": 0.3})
    under = dataclasses.replace(bld("w5"), ring=np.array(square(10, 0.0, -32.0), float))
    assert landmarks.removed([under], [e]) == set()
    assert landmarks.removed([under], [e], models) == {"w5"}
    assert landmarks.removed_by([under], [e], models) == {"t": {"w5"}}
    path = tmp_path / "landmarks.json"
    path.write_text(json.dumps([e]), encoding="utf-8")
    assert landmarks.clear_polygons(path)[0].bounds == (-22, -22, 22, 22)
    assert landmarks.clear_polygons(path, models)[0].bounds == pytest.approx((-22, -40, 22, 22))


# ------------------------------------------------------------------ checks

def test_landmark_problems_road_through():
    # a 6.5 m road along y = 23 keeps its corridor grown to width / 2 - 0.5 m (2.75 m) off the footprint (+-20 m)
    road_in = ground_road([(-50, 0), (50, 0)]); road_by = ground_road([(-50, 23.0), (50, 23.0)])
    probs = validate.landmark_problems([lm(footprint=square(40))], [road_in, road_by], FLAT, EXT)
    assert len(probs) == 1 and "passes through it" in probs[0]


def test_landmark_problems_leave_bridges_and_covered_roads_alone():
    deck = ground_road([(-50, 0), (50, 0)], oid=2, bridge=True)
    passage = ground_road([(0, -50), (0, 50)], oid=3, covered=True)
    edge = ground_road([(-50, 22.85), (50, 22.85)], oid=4)    # its corridor grown to width / 2 - 0.5 m 0.1 m off
    assert validate.landmark_problems([lm(footprint=square(40))], [deck, passage, edge], FLAT, EXT) == []
    (msg,) = validate.landmark_problems([lm(name="tower", footprint=square(40))],
                                        [ground_road([(0, 0), (0, 50)], oid=7)], FLAT, EXT)
    assert msg == "tower: road way 7 (Test Road) passes through it"


def test_landmark_problems_report_lanes_that_reach_into_it():
    # the road along y = 21.85 misses the footprint (+-20 m) shrunk 0.5 m, but its corridor (6.5 m wide) clips
    # the footprint by 1.4 m: grown to width / 2 - 0.5 m (building_problems' rule) it still reaches 0.9 m in.
    # Cut by resolve (the corridor grown LANE_CLEAR), the footprint passes the lanes check: all that is left to
    # report is the 4.8 % the cut took
    beside = ground_road([(-100, 21.85), (100, 21.85)], oid=5)
    (msg,) = validate.landmark_problems([lm(footprint=square(40))], [beside], FLAT, EXT)
    assert msg == "t: the corridor of road way 5 (Test Road) reaches into it near (0, 20)"
    osm = {"buildings": [square_building("w1", EXT)], "areas": []}
    (cut,) = landmarks.resolve({"t": entry(yaw={"azimuth_deg": 90.0})}, osm, EXT, FLAT, roads=[beside])
    took = ["t: the lane cut removed 4.8 % of its footprint (76 m2), by way 5 (Test Road) 76.0 m2"]
    assert validate.landmark_problems([cut], [beside], FLAT, EXT) == took
    assert validate.landmark_problems([cut], [beside], FLAT, EXT, Path("no-models")) == took   # the footprint again


def cut_by(*roads, params=None, side=40.0):
    """The side x side footprint (+-20 m, 1,600 m2, by default) of landmark t, as resolve cuts it by the given
    roads."""
    osm = {"buildings": [square_building("w1", EXT, side=side)], "areas": []}
    (e,) = landmarks.resolve({"t": entry(yaw={"azimuth_deg": 90.0}, params=params or {})}, osm, EXT, FLAT,
                             roads=list(roads))
    return e


def test_landmark_problems_report_a_lane_cut_over_two_percent():
    # a road along y = 22.55: its corridor grown LANE_CLEAR (3.75 m) bites 1.2 m x 40 m = 48 m2 (3.0 %) off the
    # footprint; along y = 23.35 only 0.4 m x 40 m = 16 m2 (1.0 %)
    big, small = ground_road([(-100, 22.55), (100, 22.55)], oid=5), ground_road([(-100, 23.35), (100, 23.35)], oid=6)
    e = cut_by(big)
    assert e["lane_cut"]["share"] == pytest.approx(0.03)
    assert validate.landmark_problems([e], [big], FLAT, EXT) == [
        "t: the lane cut removed 3.0 % of its footprint (48 m2), by way 5 (Test Road) 48.0 m2"]
    e = cut_by(small)
    assert e["lane_cut"]["share"] == pytest.approx(0.01)
    assert validate.landmark_problems([e], [small], FLAT, EXT) == []
    both = cut_by(big, small)                      # the roads' bites overlap: 48 m2, the second adds nothing
    assert validate.landmark_problems([both], [big, small], FLAT, EXT) == [
        "t: the lane cut removed 3.0 % of its footprint (48 m2), by way 5 (Test Road) 48.0 m2, "
        "way 6 (Test Road) 0.0 m2"]


def test_landmark_problems_report_a_road_that_splits_the_footprint():
    # a road across it along x = -14: the corridor leaves a 2.25 m strip to the west, dropped, and the east
    # 30.25 m, kept; the centreline is no longer in the cut footprint, but it crossed it before the cut (lane_cut's
    # through)
    across = ground_road([(-14, -100), (-14, 100)], oid=5)
    e = cut_by(across)
    assert validate.landmark_problems([e], [across], FLAT, EXT) == [
        "t: road way 5 (Test Road) passes through it",
        "t: the lane cut removed 24.4 % of its footprint (390 m2) and split it into 2 pieces, "
        "by way 5 (Test Road) 300.0 m2"]
    sliver = lm(lane_cut={"m2": 5.0, "share": 0.003, "pieces": 2, "holes": 0,      # no through: written by 22f
                          "roads": [{"road": "way 5 (Test Road)", "m2": 5.0}]})
    assert validate.landmark_problems([sliver], [], FLAT, EXT) == [      # however small
        "t: the lane cut split it into 2 pieces, by way 5 (Test Road) 5.0 m2"]


def test_landmark_problems_report_a_road_wholly_inside_the_footprint():
    # a corridor wholly inside the footprint leaves a hole, which the kept outline fills again, so the cut removes
    # nothing; the road's centreline is in the footprint too (reported as passing through it)
    inside = ground_road([(-5, 0), (5, 0)], oid=6)
    e = cut_by(inside)
    assert validate.landmark_problems([e], [inside], FLAT, EXT) == [
        "t: road way 6 (Test Road) passes through it",
        "t: the lane cut left 1 hole in it, by way 6 (Test Road) 0.0 m2"]


def test_landmark_problems_report_every_reason_in_one_line():
    cut = {"m2": 300.0, "share": 0.1875, "pieces": 3, "holes": 2,
           "roads": [{"road": "way 5 (A)", "m2": 200.0}, {"road": "way 6", "m2": 100.0}]}
    assert validate.landmark_problems([lm(lane_cut=cut)], [], FLAT, EXT) == [
        "t: the lane cut removed 18.8 % of its footprint (300 m2), split it into 3 pieces and left 2 holes in it, "
        "by way 5 (A) 200.0 m2, way 6 100.0 m2"]


def test_landmark_problems_pass_a_lane_cut_up_to_two_percent():
    ok = {"m2": 32.0, "share": validate.LANE_CUT_SHARE, "pieces": 1, "holes": 0,
          "roads": [{"road": "way 5 (Test Road)", "m2": 32.0}]}
    assert validate.landmark_problems([lm(lane_cut=ok)], [], FLAT, EXT) == []
    assert validate.landmark_problems([lm(lane_cut=NO_CUT)], [], FLAT, EXT) == []
    assert validate.landmark_problems([lm(lane_cut=ok | {"share": 0.0201})], [], FLAT, EXT) == [
        "t: the lane cut removed 2.0 % of its footprint (32 m2), by way 5 (Test Road) 32.0 m2"]
    assert validate.landmark_problems([lm(lane_cut=ok | {"share": 0.5, "roads": []})], [], FLAT, EXT) == [
        "t: the lane cut removed 50.0 % of its footprint (32 m2)"]       # no road named: none to list


def test_landmark_problems_do_not_report_the_ways_a_landmark_is_exempt_for():
    # the 3 % bite of the road along y = 22.55 is the cut's to make, unless the landmark keeps its footprint whole
    # (keep_off_roads false) or names the way
    big = ground_road([(-100, 22.55), (100, 22.55)], oid=5)
    for rule in (False, {"except": ["w5"]}):
        e = cut_by(big, params={"keep_off_roads": rule})
        assert e["lane_cut"] == NO_CUT and Polygon(e["footprint"]).area == pytest.approx(1600.0)
        assert validate.landmark_problems([e], [big], FLAT, EXT) == []
    other = cut_by(big, params={"keep_off_roads": {"except": ["w9"]}})      # another way: the cut happens
    assert validate.landmark_problems([other], [big], FLAT, EXT) == [
        "t: the lane cut removed 3.0 % of its footprint (48 m2), by way 5 (Test Road) 48.0 m2"]


DEAD_END = [(0, -150), (0, -30)]      # a road that ends 70 m inside the 200 m square (+-100 m)


def test_landmark_problems_report_a_road_into_the_footprint():
    # a 6.5 m dead-end road 70 m into a 200 m square: its corridor grown LANE_CLEAR (3.75 m) notches the footprint
    # by 7.5 m x 70 m and a half circle (547 m2, 1.37 %), under LANE_CUT_SHARE, and the cut footprint no longer
    # meets its centreline. But the centreline crossed the footprint before the cut: a road into it, not beside it
    into = ground_road(DEAD_END, oid=5)
    e = cut_by(into, side=200.0)
    assert e["lane_cut"]["share"] == pytest.approx((7.5 * 70 + math.pi * 3.75 ** 2 / 2) / 200 ** 2, rel=1e-3)
    assert e["lane_cut"]["share"] < validate.LANE_CUT_SHARE
    assert not Polygon(e["footprint"]).intersects(LineString(into.nodes[:, :2]))
    assert e["lane_cut"]["through"] == ["way 5 (Test Road)"]
    assert validate.landmark_problems([e], [into], FLAT, EXT) == ["t: road way 5 (Test Road) passes through it"]


def test_landmark_problems_pass_a_road_beside_the_footprint():
    # a road along y = 101.85 beside the 200 m square: its corridor (width / 2) clips the footprint by 1.4 m, and
    # grown LANE_CLEAR it cuts a 1.9 m x 200 m sliver (0.95 %); its centreline never enters the footprint
    beside = ground_road([(-150, 101.85), (150, 101.85)], oid=5)
    e = cut_by(beside, side=200.0)
    assert e["lane_cut"]["share"] == pytest.approx(1.9 * 200 / 200 ** 2) and e["lane_cut"]["through"] == []
    assert validate.landmark_problems([e], [beside], FLAT, EXT) == []


def test_lane_cut_through_is_a_centreline_across_the_footprint_shrunk_half_a_metre():
    # validate's "passes through" test (the footprint shrunk validate.SHRINK) on the footprint before the cut: a
    # centreline 0.4 m inside the edge runs beside it, one 0.6 m inside through it
    edge, inside = ground_road([(-150, 99.6), (150, 99.6)], oid=5), ground_road([(-150, 99.4), (150, 99.4)], oid=6)
    assert cut_by(edge, side=200.0)["lane_cut"]["through"] == []
    assert cut_by(inside, side=200.0)["lane_cut"]["through"] == ["way 6 (Test Road)"]


def test_lane_cut_does_not_record_an_exempt_way():
    # a drive that ends at the wall, 0.3 m inside it (the Capitol's two end on theirs), exempt by keep_off_roads:
    # nothing is cut or reported
    drive = ground_road([(0, -150), (0, -99.7)], oid=5)
    e = cut_by(drive, params={"keep_off_roads": {"except": ["w5"]}}, side=200.0)
    assert e["lane_cut"] == NO_CUT and validate.landmark_problems([e], [drive], FLAT, EXT) == []
    # the dead-end road 70 m in, exempt: resolve leaves the footprint whole by it and records nothing, so it is
    # validate's own check of the footprint that reports it, once
    into = ground_road(DEAD_END, oid=5)
    for rule in (False, {"except": ["w5"]}):
        e = cut_by(into, params={"keep_off_roads": rule}, side=200.0)
        assert e["lane_cut"] == NO_CUT
        assert validate.landmark_problems([e], [into], FLAT, EXT) == ["t: road way 5 (Test Road) passes through it"]


def test_landmark_problems_report_the_roads_the_lane_cut_records_through():
    # the record names the roads (validate's roads need not hold them), each once, after those validate finds
    # across the footprint as it is
    cut = NO_CUT | {"through": ["way 9 (Gone Road)", "way 7 (Test Road)"]}
    road = ground_road([(-50, 0), (50, 0)], oid=7)
    assert validate.landmark_problems([lm(lane_cut=cut)], [road], FLAT, EXT) == [
        "t: road way 7 (Test Road) passes through it", "t: road way 9 (Gone Road) passes through it"]
    assert validate.landmark_problems([lm(lane_cut=cut)], [], FLAT, EXT) == [
        "t: road way 9 (Gone Road) passes through it", "t: road way 7 (Test Road) passes through it"]


def test_a_road_the_lane_cut_records_through_is_reported_once(tmp_path):
    # the road along x = -14 the cut took out of the footprint, under a model that still stands on all of it: it
    # passes through the landmark, which says it (not again as "past the footprint")
    models = tmp_path / "models"
    box_npz(models / "t" / "main.npz", (-20, -20, 0.0), (20, 20, 10.0))
    across = ground_road([(-14, -100), (-14, 100)], oid=5)
    e = cut_by(across)
    assert validate.landmark_problems([e], [across], FLAT, EXT, models) == [
        "t: road way 5 (Test Road) passes through it",
        "t: the lane cut removed 24.4 % of its footprint (390 m2) and split it into 2 pieces, "
        "by way 5 (Test Road) 300.0 m2"]


def test_landmark_problems_check_what_the_model_stands_on(tmp_path):
    # the footprint is +-20 m; the model's collision box reaches 10 m past it to the south, over a road
    # along y = -25 that misses the footprint
    models = tmp_path / "models"
    box_npz(models / "t" / "main.npz", (-20, -30, 0.0), (20, 20, 10.0))
    e = lm(footprint=square(40), params={"ground_dz": 0.3})
    past, through = ground_road([(-50, -25), (50, -25)]), ground_road([(-50, 0), (50, 0)], oid=2)
    assert validate.landmark_problems([e], [past], FLAT, EXT) == []          # the footprint alone
    assert validate.landmark_problems([e], [past, through], FLAT, EXT, models) == [
        "t: road way 2 (Test Road) passes through it",
        "t: road way 1 (Test Road) passes through its model's ground outline, past the footprint"]
    edge = ground_road([(-50, -29.8), (50, -29.8)], oid=3)    # inside the outline, outside it shrunk 0.5 m
    beside = ground_road([(-50, -32.0), (50, -32.0)], oid=4)  # outside it, its corridor 0.75 m in
    clear = ground_road([(-50, -32.85), (50, -32.85)], oid=6)  # its corridor (width / 2 - 0.5 m) 0.1 m off
    assert validate.landmark_problems([e], [edge, beside, clear], FLAT, EXT) == []
    assert validate.landmark_problems([e], [edge, beside, clear], FLAT, EXT, models) == [
        "t: the corridor of road way 3 (Test Road) reaches into its model's ground outline, past the footprint "
        "near (0, -29)",
        "t: the corridor of road way 4 (Test Road) reaches into its model's ground outline, past the footprint "
        "near (0, -30)"]


def test_landmark_problems_check_only_what_a_truss_stands_on(tmp_path):
    # a landmark over bridge decks (params.decks) is in the air over its footprint: the roads under it
    # count only where its model stands (its piers)
    models = tmp_path / "models"
    truss = lm(footprint=square(40), params={"decks": ["w7"], "ground_dz": 0.3})
    under, on = ground_road([(-50, 0), (50, 0)]), ground_road([(-50, 15), (50, 15)], oid=2)
    assert validate.landmark_problems([truss], [under], FLAT, EXT) == []
    box_npz(models / "t" / "main.npz", (-5, 12, 0.0), (5, 18, 30.0))          # a pier on the road at y = 15
    assert validate.landmark_problems([truss], [under, on], FLAT, EXT, models) == [
        "t: road way 2 (Test Road) passes through what its model stands on"]
    beside = ground_road([(-50, 20), (50, 20)], oid=3)          # its corridor 0.75 m into the pier (to y 18)
    assert validate.landmark_problems([truss], [under, beside], FLAT, EXT, models) == [
        "t: the corridor of road way 3 (Test Road) reaches into what its model stands on near (0, 18)"]


def test_landmark_problems_floating():
    assert "floats" in validate.landmark_problems([lm(footprint=square(40), base_z=1.5)], [], FLAT0, EXT)[0]


def test_landmark_base_on_the_lowest_ground_passes():
    heights = slope_east(EXT, 10.0, 0.05)
    lowest = 10 + 0.05 * (-20 + EXT.half_size)
    assert validate.landmark_problems([lm(footprint=square(40), base_z=lowest + 0.9)], [], heights, EXT) == []
    (msg,) = validate.landmark_problems([lm(footprint=square(40), base_z=lowest + 1.2)], [], heights, EXT)
    assert msg == "t: base floats 1.2 m"


def test_feature_problems_check_what_the_models_stand_on(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "landmarks.json").write_text(json.dumps([lm(name="tower", params={"ground_dz": 0.3})]),
                                        encoding="utf-8")
    box_npz(out / "models" / "tower" / "main.npz", (-20, -30, 0.0), (20, 20, 10.0))
    probs = validate.feature_problems(Config(debug_half_size_m=512.0, terrain_size=N), out, EXT, FLAT,
                                      [ground_road([(-50, -25), (50, -25)])])
    assert probs == ["tower: road way 1 (Test Road) passes through its model's ground outline, past the footprint"]


def test_feature_problems_check_the_landmarks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "landmarks.json").write_text(json.dumps([lm(name="tower")]), encoding="utf-8")
    probs = validate.feature_problems(Config(debug_half_size_m=512.0, terrain_size=N), out, EXT, FLAT,
                                      [ground_road([(-50, 0), (50, 0)])])
    assert probs == ["tower: road way 1 (Test Road) passes through it"]


# ------------------------------------------------------------------ level objects

def test_level_content_writes_shape_static_and_materials(tmp_path):
    # data_dir/landmarks.json with one entry, data_dir/models/t/main.npz (contract npz, material br_lm_stucco_tan)
    cfg, data_dir, root = Config(debug_half_size_m=512.0, terrain_size=N), tmp_path / "out", tmp_path / "level"
    lm_x, lm_y, lm_base_z, lm_yaw = 10.5, -20.25, 4.7, 0.5
    data_dir.mkdir()
    (data_dir / "landmarks.json").write_text(json.dumps([lm(x=lm_x, y=lm_y, base_z=lm_base_z, yaw=lm_yaw)]),
                                             encoding="utf-8")
    contract_npz(data_dir / "models" / "t" / "main.npz")
    groups = landmarks.level_content(cfg, data_dir, root, FLAT, EXT)
    (st,) = groups["landmarks"]
    assert st["shapeName"] == "/levels/baton_rouge/art/shapes/landmarks/t.dae"
    assert st["position"] == [lm_x, lm_y, lm_base_z] and st["rotationMatrix"] == trees.rotation_matrix(lm_yaw)
    assert (st["class"], st["persistentId"]) == ("TSStatic", bf.persistent_id("lm", "t", "main"))
    assert (st["collisionType"], st["decalType"], st["useInstanceRenderData"]) == (
        "Collision Mesh", "Collision Mesh", True)
    mats = json.loads((root / "art/shapes/landmarks/main.materials.json").read_text())
    assert set(mats) == {"br_lm_stucco_tan"}
    assert mats["br_lm_stucco_tan"]["Stages"][0]["baseColorFactor"] == [0.86, 0.76, 0.60, 1]
    counts = validate.dae_vertex_counts(root / "art/shapes/landmarks/t.dae")
    assert counts == {"t_a100": 3, "Colmesh-1": 3}


def test_level_content_names_parts_and_copies_textures(tmp_path, monkeypatch):
    registry = tmp_path / "landmark_materials.yaml"
    registry.write_text("br_lm_clock:\n  maps: {b: 'tex:clock_face'}\n"
                        "br_lm_unused:\n  maps: {b: 'tex:other'}\n", encoding="utf-8")
    monkeypatch.setattr(landmarks, "MATERIAL_REGISTRY", registry)
    cfg, data_dir, root = Config(debug_half_size_m=512.0, terrain_size=N), tmp_path / "out", tmp_path / "level"
    (data_dir / "textures").mkdir(parents=True)
    (data_dir / "textures" / "clock_face_b.color.dds").write_bytes(b"dds")
    (data_dir / "textures" / "index.json").write_text(json.dumps(
        {"clock_face": {"source": "generated", "files": {"b": "clock_face_b.color.dds"}}}), encoding="utf-8")
    (data_dir / "landmarks.json").write_text(json.dumps([lm(name="hall"), lm(name="dome")]), encoding="utf-8")
    contract_npz(data_dir / "models" / "hall" / "main.npz", nodes=("hall_a100",), material="br_lm_clock")
    contract_npz(data_dir / "models" / "hall" / "roof.npz", nodes=("roof_a100",), material="br_lm_clock")
    groups = landmarks.level_content(cfg, data_dir, root, FLAT, EXT)
    assert [o["shapeName"].rsplit("/", 1)[1] for o in groups["landmarks"]] == ["hall.dae", "hall_roof.dae"]
    assert [o["persistentId"] for o in groups["landmarks"]] == [bf.persistent_id("lm", "hall", "main"),
                                                                bf.persistent_id("lm", "hall", "roof")]
    mats = json.loads((root / "art/shapes/landmarks/main.materials.json").read_text())
    assert set(mats) == {"br_lm_clock"}
    assert mats["br_lm_clock"]["Stages"][0]["baseColorMap"] == (
        "/levels/baton_rouge/art/textures/br/clock_face_b.color.png")
    assert (root / "art/textures/br/clock_face_b.color.dds").read_bytes() == b"dds"
    assert not list((root / "art/textures/br").glob("*.png"))      # the .dds ships, under its .png name


def test_level_content_leaves_out_a_landmark_without_a_model(tmp_path, caplog):
    data_dir = tmp_path / "out"
    data_dir.mkdir()
    (data_dir / "landmarks.json").write_text(json.dumps([lm(name="tower")]), encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="brmap.landmarks"):
        assert landmarks.level_content(Config(), data_dir, tmp_path / "level", FLAT, EXT) == {}
    assert "tower" in caplog.text and "models" in caplog.text


# ------------------------------------------------------------------ the registry

def test_registry_cites_its_heights():
    for name, e in landmarks.load_registry().items():
        assert set(e.get("params", {})) & landmarks.CITED <= set(e.get("sources", {})), name


def test_memorial_tower_entry():
    e = landmarks.load_registry()["memorial_tower"]
    assert e["model"] == "blender/landmarks/memorial_tower.py"
    assert (e["anchor"], e["yaw"]) == ({"osm": "w550888707"}, {"from": "w550888707"})
    assert (e["footprint"], e["replaces"], e["clear_pad_m"]) == ({"osm": ["w155026538"]}, ["w155026538"], 2.0)
    assert e["params"] == {"height_m": 53.3, "shaft_m": [8.0, 6.6]}
    assert e["sources"]["height_m"] == "OSM w550888707 height=53; published 175 ft"


def test_tiger_stadium_entry():
    e = landmarks.load_registry()["tiger_stadium"]
    assert e["model"] == "blender/landmarks/tiger_stadium.py"
    assert (e["anchor"], e["yaw"]) == ({"osm": "r1640828"}, {"azimuth_deg": 156})
    # replaces nothing by id: the bowl r1640828 is dropped for Powerhouse Lane's corridor before the landmarks look,
    # the leisure=stadium area w515632965 is no building (each would be a "matches no building" warning), and the
    # clear zone takes the Microsoft footprints
    assert (e["footprint"], e["replaces"], e["clear_pad_m"]) == ({"osm": ["r1640828"]}, [], 3)
    assert e["params"] == {"rim": "lidar", "rim_m": [20, 55], "pitch": "w119361456", "field_m": [109.73, 48.77]}
    assert {"rim_m", "field_m"} <= set(e["sources"])


def test_pmac_entry():
    e = landmarks.load_registry()["pmac"]
    assert e["model"] == "blender/landmarks/pmac.py"
    assert (e["anchor"], e["yaw"]) == ({"lat": 30.414255, "lon": -91.184495}, {"azimuth_deg": 90})
    assert (e["footprint"], e["replaces"]) == ({"osm": ["w133751599"]}, ["w133751599"])
    p = e["params"]
    assert (p["diameter_m"], p["crown_m"], p["wall_m"], p["ribs"]) == (106, 30.6, 16, 45)
    assert {"diameter_m", "crown_m", "wall_m", "ribs", "annex_m", "entrances_az_deg"} <= set(e["sources"])


def test_alex_box_entry():
    e = landmarks.load_registry()["alex_box"]
    assert e["model"] == "blender/landmarks/alex_box.py"
    assert (e["anchor"], e["yaw"]) == ({"osm": "w133888808"}, {"azimuth_deg": 90})
    assert e["footprint"] == {"osm": ["w133888808"]}
    assert e["replaces"] == ["w1148083778", "w1148083777", "w1148083776"]
    p = e["params"]
    assert {k: p[k] for k in ("rim_m", "home_az_deg", "field", "bases_m", "mound_m")} == {
        "rim_m": 19, "home_az_deg": 202, "field": "r17708790", "bases_m": 27.43, "mound_m": 18.44}
    assert p["outlines"] == ["w1148083777", "w1148083776"]          # the outfield bleachers it replaces
    assert {"rim_m", "home_az_deg", "bases_m", "mound_m"} <= set(e["sources"])


def test_mike_habitat_entry():
    e = landmarks.load_registry()["mike_habitat"]
    assert e["model"] == "blender/landmarks/mike_habitat.py"
    assert (e["anchor"], e["yaw"]) == ({"lat": 30.4134234, "lon": -91.1850659}, {"azimuth_deg": 90})
    assert e["replaces"] == [] and len(e["footprint"]["ring"]) >= 8
    assert e["sources"]["footprint"] == "traced from NAIP 2023, data/raw/naip"
    assert {"tower_m", "night_house_m", "pool_m", "stream_m", "rocks_m", "viewing_m"} <= set(e["params"])


def test_capitol_entry():
    e = landmarks.load_registry()["capitol"]
    assert e["model"] == "blender/landmarks/capitol.py"
    assert (e["anchor"], e["yaw"], e["footprint"]) == ({"osm": "w97757524"}, {"from": "w97757524"},
                                                       {"osm": ["w97757524"]})
    assert e["replaces"] == ["w97757524"]                  # its lidar tower part (/t0) goes with it
    assert e["params"] == {"height_m": 137, "wings_m": 21, "centre_m": 30,
                           "setbacks": [[60, 24, 22], [100, 14, 14], [110, 10, 10], [118, 6, 6]],
                           # the two service drives that end at its wings' north walls; every other road it keeps off
                           "keep_off_roads": {"except": ["w97757425", "w824026179"]}}
    assert {"height_m", "wings_m", "centre_m", "setbacks"} <= set(e["sources"])
    assert "450 ft" in e["sources"]["height_m"]


def test_old_capitol_entry():
    e = landmarks.load_registry()["old_capitol"]
    assert e["model"] == "blender/landmarks/old_capitol.py"
    assert (e["anchor"], e["yaw"], e["footprint"]) == ({"osm": "w515632945"}, {"from": "w515632945"},
                                                       {"osm": ["w515632945"]})
    assert e["replaces"] == ["w515632945"]
    assert e["params"] == {"height_m": 26.3, "towers_m": 35.9}
    assert {"height_m", "towers_m"} <= set(e["sources"])


def test_wilkinson_truss_entry():
    e = landmarks.load_registry()["wilkinson_truss"]
    assert e["model"] == "blender/landmarks/wilkinson_truss.py"
    assert e["anchor"] == {"lat": 30.4396461, "lon": -91.198343}     # on w181653662 at full-map x -3700
    assert e["yaw"] == {"azimuth_deg": 273.16}                        # along the deck
    assert (e["replaces"], e["clear_pad_m"], len(e["footprint"]["ring"])) == ([], 0, 4)
    assert e["params"] == {"decks": ["w181653662", "w682521449"], "from_lon": -91.2029294, "to_lon": -91.1937876,
                           "peaks_lon": [-91.1965543, -91.200423], "top_m": [16, 24], "clear_m": 6.0, "panel_m": 15}
    assert {"from_lon", "to_lon", "peaks_lon", "top_m"} <= set(e["sources"])


ROOT = Path(__file__).resolve().parents[1]
REAL_OSM = ROOT / "data" / "raw" / "osm_features.json"
REAL_ROADS = ROOT / "data" / "out" / "roads.json"
FACT_AREAS = {"memorial_tower": 548.7, "tiger_stadium": 50316.0, "pmac": 13963.0, "alex_box": 24988.0,
              "mike_habitat": 1291.0, "capitol": 3411.0, "old_capitol": 1687.0,
              "wilkinson_truss": 31692.0}   # m2 (the habitat: its traced ring; the truss: 880 x 36 m round its decks)
CAMPUS = {"memorial_tower", "tiger_stadium", "pmac", "alex_box", "mike_habitat"}   # the LSU test map's


@pytest.mark.skipif(not REAL_OSM.exists() or not REAL_ROADS.exists(),
                    reason="no data/raw/osm_features.json or data/out/roads.json")
def test_registry_resolves_on_real_osm(caplog):
    from brmap.config import load_config
    extent = Extent.from_config(load_config(ROOT / "config.yaml"))
    registry = landmarks.load_registry()
    assert set(registry) <= set(FACT_AREAS)            # every entry has its area in the plan's Facts
    assert CAMPUS <= set(registry)
    n = 1024
    with caplog.at_level(logging.INFO, logger="brmap.landmarks"):
        got = landmarks.resolve(registry, json.loads(REAL_OSM.read_text(encoding="utf-8")), extent,
                                np.zeros((256, 256)), np.zeros((n, n), np.float32), 2 * extent.half_size / n,
                                load_roads(REAL_ROADS))
    assert [e["name"] for e in got] == list(registry)
    # the Capitol's keep_off_roads still names the two service drives that end at its walls (a re-fetch that
    # renumbers them warns), and only those leave its footprint whole
    spared = [r.getMessage() for r in caplog.records if "leaves its footprint whole" in r.getMessage()]
    assert len(spared) == 1 and spared[0].startswith("landmark capitol leaves its footprint whole by way ")
    assert sorted(re.findall(r"way (\d+)", spared[0])) == ["824026179", "97757425"]     # whichever order
    assert "keep_off_roads names" not in caplog.text
    for e in got:
        assert Polygon(e["footprint"]).area == pytest.approx(FACT_AREAS[e["name"]], rel=0.15), e["name"]
    # the downtown scripts take +v for north: a re-traced outline whose long axis comes out flipped (yaw near
    # pi, +u west) turns them round
    yaw = {e["name"]: e["yaw"] for e in got}
    for name in ("capitol", "old_capitol"):
        off_east = abs((yaw[name] + math.pi) % (2 * math.pi) - math.pi)
        assert off_east < math.radians(45.0), (name, math.degrees(yaw[name]))
    # cut by the roads beside them (the Capitol by all but its two service drives), the footprints keep off the
    # lanes (the full map's roads; no models), and no cut is a problem: Tiger Stadium loses Powerhouse Lane's 147 m2
    # (0.29 %), the Capitol nothing (its drives would have cost 0.57 %), the others nothing
    assert validate.landmark_problems(got, load_roads(REAL_ROADS), np.zeros((256, 256)), extent) == []
    cuts = {e["name"]: e["lane_cut"] for e in got}
    assert cuts["capitol"] == NO_CUT and 0.001 < cuts["tiger_stadium"]["share"] < 0.01
    assert all(c["pieces"] == 1 and c["holes"] == 0 for c in cuts.values())
    # no road's centreline crosses a footprint before the cut either (Powerhouse Lane's runs 1.86 m outside Tiger
    # Stadium's; the Capitol's drives are exempt)
    assert all(c["through"] == [] for c in cuts.values())


@pytest.mark.skipif(not REAL_OSM.exists() or not REAL_ROADS.exists(),
                    reason="no data/raw/osm_features.json or data/out/roads.json")
def test_wilkinson_truss_resolves_on_the_real_decks():
    from brmap.config import load_config
    extent = Extent.from_config(load_config(ROOT / "config.yaml"))
    roads = load_roads(REAL_ROADS)
    (e,) = landmarks.resolve({"wilkinson_truss": landmarks.load_registry()["wilkinson_truss"]}, OSM, extent,
                             np.zeros((256, 256)), roads=roads)
    p = e["params"]
    assert (e["x"], e["y"]) == (pytest.approx(-3700.0, abs=0.5), pytest.approx(1009.9, abs=0.5))
    along = 1 / math.cos(math.radians(3.16))                         # the deck runs 3.16 deg off the x axis
    assert abs(p["from_u"] - p["to_u"]) == pytest.approx((4141 - 3262) * along, abs=1.0)       # 880 m of truss
    assert abs(p["peaks_u"][0] - p["peaks_u"][1]) == pytest.approx((3900 - 3528) * along, abs=1.0)   # the main span
    by_id = {f"w{r.osm_ids[0]}": r for r in roads if r.is_bridge and len(r.osm_ids) == 1}
    lo, hi = sorted((p["from_u"], p["to_u"]))
    for d in p["deck_local"]:
        road = by_id[d["id"]]
        u, v, z = np.asarray(d["nodes"], float).T
        assert d["width"] == road.width == 13.8 and len(u) >= 40
        assert (np.diff(u) > 0).all() and lo <= u[0] and u[-1] <= hi
        assert abs(v).max() < 30.0                                    # both decks along +u
        assert np.abs((z + e["base_z"])[:, None] - road.nodes[None, :, 2]).min(axis=1).max() < 0.01


@pytest.mark.skipif(not REAL_ROADS.exists(), reason="no data/out/roads.json")
def test_wilkinson_truss_footprint_ends_at_the_truss_ends():
    # the footprint's two short ends cross w181653662 within 1 m (along the deck) of where it crosses from_lon
    # and to_lon, the truss's ends, where the model centres its end piers
    from brmap.config import load_config
    extent = Extent.from_config(load_config(ROOT / "config.yaml"))
    roads = load_roads(REAL_ROADS)
    (e,) = landmarks.resolve({"wilkinson_truss": landmarks.load_registry()["wilkinson_truss"]}, OSM, extent,
                             np.zeros((256, 256)), roads=roads)
    p = e["params"]
    (deck,) = [r for r in roads if r.is_bridge and r.osm_ids == [181653662]]
    line = LineString(deck.nodes[:, :2])
    ring = np.asarray(e["footprint"], float)
    edges = sorted(zip(ring, np.roll(ring, -1, axis=0)), key=lambda ab: math.dist(*ab))[:2]
    ends = [LineString(ab).intersection(line) for ab in edges]
    assert [g.geom_type for g in ends] == ["Point", "Point"]
    u = sorted(float(landmarks.to_local([(g.x, g.y)], e["x"], e["y"], e["yaw"])[0, 0]) for g in ends)
    lo, hi = sorted((p["from_u"], p["to_u"]))
    assert u == [pytest.approx(lo, abs=1.0), pytest.approx(hi, abs=1.0)]


def blender_constants(script: Path) -> dict:
    """The module-level constants a Blender script assigns (NAME = literal, or A, B = literal, literal), read
    from its source: importing the script needs Blender."""
    out = {}
    for node in ast.parse(script.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
            tuples = isinstance(target, ast.Tuple) and isinstance(value, ast.Tuple)
            for name, v in (zip(target.elts, value.elts) if tuples else [(target, value)]):
                if isinstance(name, ast.Name):
                    try:
                        out[name.id] = ast.literal_eval(v)
                    except ValueError:      # not a literal (a call, a name)
                        pass
    return out


def test_span_pad_covers_the_truss_end_piers():
    # the deck dressing keeps its piers SPAN_PAD off the truss's footprint, whose ends the end piers straddle
    end_pier = blender_constants(ROOT / "blender" / "landmarks" / "wilkinson_truss.py")["END_PIER"]
    assert isinstance(end_pier, float) and end_pier > 0
    assert dressing.SPAN_PAD >= end_pier / 2


def test_load_registry_names_an_entry_without_its_keys(tmp_path):
    path = tmp_path / "landmarks.yaml"
    path.write_text("broken:\n  model: x.py\n", encoding="utf-8")
    with pytest.raises(ValueError, match="broken"):
        landmarks.load_registry(path)


def test_constants():
    assert landmarks.CITED == {"height_m", "diameter_m", "crown_m", "rim_m"}
    assert (landmarks.REMOVE_SHARE, landmarks.SHAPES_DIR, landmarks.TEX_DIR, landmarks.GROUP) == (
        0.5, "art/shapes/landmarks", "art/textures/br", "landmarks")
    assert landmarks.THROUGH_SHRINK == validate.SHRINK == 0.5   # lane_cut's through is validate's "passes through"


# ------------------------------------------------------------------ the stage and its hooks

def stage_inputs(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    raw.mkdir()
    out.mkdir()
    osm = {"buildings": [square_building("w1", EXT, centre=(100.0, 100.0), side=20.0),
                         square_building("w2", EXT, centre=(-200.0, 0.0), side=20.0)], "areas": []}
    (raw / "osm_features.json").write_text(json.dumps(osm), encoding="utf-8")
    np.save(out / "heights.npy", np.full((N, N), 6.0, np.float32))
    reg = tmp_path / "landmarks.yaml"
    reg.write_text("hall:\n  model: x.py\n  anchor: {osm: w1}\n  yaw: {azimuth_deg: 90}\n"
                   "  footprint: {osm: [w1]}\n  replaces: [w1]\n  clear_pad_m: 2.0\n"
                   "  params: {height_m: 9.0}\n  sources: {height_m: test}\n", encoding="utf-8")
    return raw, out, reg


def test_run_writes_landmarks_json(tmp_path, monkeypatch, caplog):
    raw, out, reg = stage_inputs(tmp_path)
    monkeypatch.setattr(landmarks, "REGISTRY", reg)
    with caplog.at_level(logging.INFO, logger="brmap.landmarks"):
        landmarks.run(small_cfg(tmp_path))
    (e,) = landmarks.load_landmarks(out / "landmarks.json")
    assert set(e) == CONTRACT_KEYS and e["name"] == "hall"
    assert (e["x"], e["y"], e["base_z"]) == (pytest.approx(100.0, abs=0.01), pytest.approx(100.0, abs=0.01),
                                             pytest.approx(5.7))
    assert "hall" in caplog.text and "(100.0, 100.0)" in caplog.text


def test_run_writes_an_empty_list_when_landmarks_are_off(tmp_path, monkeypatch):
    raw, out, reg = stage_inputs(tmp_path)
    monkeypatch.setattr(landmarks, "REGISTRY", reg)
    landmarks.run(small_cfg(tmp_path, landmarks_enabled=False))
    assert json.loads((out / "landmarks.json").read_text(encoding="utf-8")) == []


def test_buildings_run_drops_what_landmarks_remove(tmp_path, caplog):
    raw, out, _ = stage_inputs(tmp_path)
    save_roads([], out / "roads.json")
    cfg = small_cfg(tmp_path, buildings_microsoft=False)
    buildings.run(cfg)
    assert {b.id for b in buildings.load_buildings(out / "buildings.json")} == {"w1", "w2"}
    (out / "landmarks.json").write_text(json.dumps([lm(name="hall", replaces=["w1"])]), encoding="utf-8")
    with caplog.at_level(logging.INFO, logger="brmap.buildings"):
        buildings.run(cfg)
    assert {b.id for b in buildings.load_buildings(out / "buildings.json")} == {"w2"}
    assert "hall 1 (w1)" in caplog.text


def test_buildings_run_drops_what_landmark_models_stand_on(tmp_path):
    # w2 (a 20 m square at (-200, 0)) is far from the landmark's clear zone, under its model's collision box
    raw, out, _ = stage_inputs(tmp_path)
    save_roads([], out / "roads.json")
    (out / "landmarks.json").write_text(json.dumps([lm(name="hall", x=-150.0, y=0.0, clear=square(20, -150.0, 0.0),
                                                       params={"ground_dz": 0.3})]), encoding="utf-8")
    box_npz(out / "models" / "hall" / "main.npz", (-62, -12, 0.0), (10, 12, 10.0))
    buildings.run(small_cfg(tmp_path, buildings_microsoft=False))
    assert {b.id for b in buildings.load_buildings(out / "buildings.json")} == {"w1"}


def test_trees_run_keeps_trees_off_what_landmark_models_stand_on(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (tmp_path / "raw").mkdir()
    np.save(out / "heights.npy", np.full((N, N), 6.0, np.float32))
    n, px = canopy.analysis_grid(EXT)
    xs, ys = EXT.pixel_centers(n)
    X, Y = np.meshgrid(xs, ys)
    ndsm = (12.0 * np.exp(-((X - 201.0) ** 2 + (Y - 5.0) ** 2) / 18.0)).astype(np.float32)
    canopy.save_canopy(out, ndsm, np.full((n, n), 0.6, np.float32), px)
    (out / "landmarks.json").write_text(json.dumps([lm(x=150.0, y=0.0, clear=square(20, 150.0, 0.0),
                                                       params={"ground_dz": 0.3})]), encoding="utf-8")
    cfg = small_cfg(tmp_path)
    trees.run(cfg)
    assert len(trees.load_trees(out / "trees.json")) == 1           # the tree at (201, 5) is outside the zone
    box_npz(out / "models" / "t" / "main.npz", (-10, -10, 0.0), (60, 15, 10.0))   # x 140..210, y -10..15
    trees.run(cfg)
    assert trees.load_trees(out / "trees.json") == []


def test_trees_run_keeps_trees_off_the_landmarks(tmp_path):
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
    (out / "landmarks.json").write_text(json.dumps([lm(clear=square(20, 200.0, 0.0))]), encoding="utf-8")
    trees.run(cfg)
    assert trees.load_trees(out / "trees.json") == []


def test_landmarks_stage_and_level_hook_registered():
    stages = cli.FEATURE_STAGES
    assert stages.index("canopy") < stages.index("landmarks") < stages.index("models") < stages.index("buildings")
    assert "landmarks" in cli.STAGES and "landmarks" in cli.ALL_ORDER
    assert "landmarks" in level.FEATURE_MODULES
    assert landmarks.__doc__
