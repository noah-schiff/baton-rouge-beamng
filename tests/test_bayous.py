"""Bayous: OSM waterways drawn as murky River objects in their lidar channels, and swamp pools as
WaterBlocks (bayous.py, and its hooks in trees, validate, level and the command line)."""
import json
import math
from pathlib import Path

import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, box

from brmap import __main__ as cli
from brmap import bayous, canopy, level, trees, validate
from brmap import beamng_format as bf
from brmap.config import Config
from brmap.geo import Extent
from brmap.roads import Road, save_roads
from brmap.terrain import sample_height
from brmap.water import WaterBody, save_water
from rasters import slope_east

EXT = Extent(-91.160, 30.430, 512.0)   # the small_extent box
N = 256
FLAT = np.full((N, N), 5.0)
GROUND = 6.0                           # the channel fixtures' ground
A, B, C, D = (-300.0, 0.0), (-100.0, 0.0), (100.0, 0.0), (300.0, 0.0)


def ll(extent, xy):
    """[[lon, lat], ...] for level x, y points."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    lon, lat = extent.to_lonlat(xy[:, 0], xy[:, 1])
    return [[float(a), float(b)] for a, b in zip(np.atleast_1d(lon), np.atleast_1d(lat))]


def way(id, *points, kind="stream", name=None, culvert=False, extent=EXT):
    """An osm_features.json waterway through the level points."""
    return {"id": id, "kind": kind, "name": name, "culvert": culvert, "coords": ll(extent, points)}


def ground_road(xy, width=6.5, oid=1, kind="residential", bridge=False, covered=False):
    xy = np.asarray(xy, float)
    return Road([oid], kind, np.column_stack([xy, np.full(len(xy), 5.0)]), width, 2, False, bridge,
                1 if bridge else 0, "Test Road", covered=covered)


def square(side, cx=0.0, cy=0.0):
    h = side / 2
    return [[cx - h, cy - h], [cx + h, cy - h], [cx + h, cy + h], [cx - h, cy + h]]


def swamp_ll(extent, ring, wetland="swamp", id="w9"):
    """An osm_features.json wetland with one polygon (closed outer ring in lon/lat)."""
    ring = [list(p) for p in ring]
    return {"id": id, "wetland": wetland, "polygons": [{"outer": ll(extent, ring + ring[:1]), "inner": []}]}


def ground_level(extent):
    """The ground of the channel fixtures (flat, so the channel bottom is GROUND - depth)."""
    return GROUND


def channel(extent, depth, top_width, n=1024):
    """Flat ground with a trapezoid channel along y = 0: full depth over the middle half of the
    top width, the sides sloping up to the ground at |y| = top_width / 2 (1 m cells by default)."""
    _, ys = extent.pixel_centers(n)
    f = np.clip((top_width / 2 - np.abs(ys)) / (top_width / 4), 0.0, 1.0)
    return np.tile((GROUND - depth * f)[:, None], (1, n))


def channel_with_flat_middle(extent, n=1024):
    """channel(), then 100 m of flat ground (|x| < 50), then the channel again."""
    h = channel(extent, 1.7, 8.0, n)
    xs, _ = extent.pixel_centers(n)
    h[:, np.abs(xs) < 50] = GROUND
    return h


def bumpy(extent, base, amp, wavelength=50.0, band=8.0, n=N):
    """Bumps base + amp * (cos(kx) + cos(ky')) / 2 with a level band |y| <= band along y = 0
    (y' = max(|y| - band, 0)), where the terrain only rises and falls along x."""
    xs, ys = extent.pixel_centers(n)
    k = 2 * math.pi / wavelength
    X, Y = np.meshgrid(xs, ys)
    return base + amp * (np.cos(k * X) + np.cos(k * np.maximum(np.abs(Y) - band, 0.0))) / 2


def corridor_samples(road, step=0.5):
    """x, y arrays of points every step m covering the road's corridor (centreline buffered by
    width / 2, its edge included)."""
    corridor = LineString(road.nodes[:, :2]).buffer(road.width / 2)
    x0, y0, x1, y1 = corridor.bounds
    X, Y = np.meshgrid(np.arange(x0, x1 + 1e-9, step), np.arange(y0, y1 + 1e-9, step))
    inside = shapely.intersects_xy(corridor, X, Y)
    return X[inside], Y[inside]


def tops_at(points):
    return np.asarray(points, float).reshape(-1, 3)


def run_along_x(width, x0=-100.0, x1=100.0, y=0.0, z=5.0, id="w1"):
    """A bayous.json run along y from x0 to x1."""
    return {"id": id, "name": None, "kind": "stream", "nodes": [[x0, y, z, width], [x1, y, z, width]]}


def small_cfg(tmp_path, **kw):
    return Config(debug_half_size_m=512.0, terrain_size=1024, raw_dir=str(tmp_path / "raw"),
                  out_dir=str(tmp_path / "out"), **kw)


# ------------------------------------------------------------------ constants

def test_constants():
    assert (bayous.FILL, bayous.MIN_DEPTH, bayous.SEARCH_HALF, bayous.BOTTOM_HALF) == (0.3, 0.4, 15.0, 4.0)
    assert bayous.WIDTH == {"river": (6, 30), "canal": (3, 20), "stream": (2, 12), "ditch": (1.5, 5),
                            "drain": (1.5, 6)}
    assert (bayous.ROAD_GAP, bayous.SMOOTH, bayous.RIVER_DEPTH, bayous.GROUP) == (2.0, 5, 1.5, "bayous")
    assert (bayous.WIDTH_SMOOTH, bayous.RIBBON_OVERLAP) == (5, 0.05)
    assert (bayous.POOL_KINDS, bayous.POOL_PCT, bayous.POOL_FILL, bayous.ROAD_FREEBOARD) == (
        ("swamp", "marsh"), 25, 0.15, 0.3)
    assert (bayous.POOL_MIN_SIDE, bayous.POOL_FLOOD_CELLS, bayous.POOL_FLOOD_DEPTH, bayous.POOL_LEVEL_TOL) == (
        16.0, 2, 0.05, 0.01)
    assert (bayous.POOL_SHOW_DEPTH, bayous.POOL_LEVEL_CELLS) == (0.05, 10)


# ------------------------------------------------------------------ chains

def test_chains_merge_and_split_at_culverts(small_extent):
    ww = [way("w1", A, B), way("w2", B, C, culvert=True), way("w3", C, D)]
    out = bayous.chains(ww, small_extent, ("stream",), 0.0, [], [])
    assert sorted(c["id"] for c in out) == ["w1", "w3"]


def test_chains_merge_same_kind_and_name_end_to_end(small_extent):
    ww = [way("w1", A, B, name="Bayou Fountain"),
          way("w2", C, (-100.0, 0.4), name="Bayou Fountain"),   # drawn the other way, its end 0.4 m off
          way("w3", C, D, name="Ward Creek"),                   # another name: not joined
          way("w4", C, D, kind="ditch", name="Bayou Fountain"),  # another kind: not joined
          way("w5", D, (300.0, 200.0))]                          # unnamed
    out = bayous.chains(ww, small_extent, ("stream", "ditch"), 0.0, [], [])
    by_id = {c["id"]: c for c in out}
    assert set(by_id) == {"w1+w2", "w3", "w4", "w5"}
    merged = by_id["w1+w2"]
    assert (merged["name"], merged["kind"]) == ("Bayou Fountain", "stream")
    assert merged["xy"].shape[1] == 2
    assert merged["xy"][:, 0].min() == pytest.approx(-300.0, abs=0.01)
    assert merged["xy"][:, 0].max() == pytest.approx(100.0, abs=0.01)
    assert LineString(merged["xy"]).length == pytest.approx(400.0, abs=0.1)
    assert (by_id["w4"]["kind"], by_id["w5"]["name"]) == ("ditch", None)


def test_chains_three_ways_at_one_node_stay_apart(small_extent):
    ww = [way("w1", A, B), way("w2", B, C), way("w3", B, (-100.0, 200.0))]   # a confluence
    assert sorted(c["id"] for c in bayous.chains(ww, small_extent, ("stream",), 0.0, [], [])) == ["w1", "w2", "w3"]


def test_chains_kinds_box_and_length(small_extent):
    ww = [way("w1", A, B, kind="river"),                  # not a kind asked for
          way("w2", (400.0, 0.0), (700.0, 0.0)),            # leaves the box at x = 512
          way("w3", (0.0, 100.0), (15.0, 100.0)),           # 15 m: shorter than 20 m
          way("w4", (0.0, 200.0), (50.0, 200.0))]           # 50 m
    out = bayous.chains(ww, small_extent, ("stream",), 0.0, [], [])
    assert sorted(c["id"] for c in out) == ["w2", "w4"]
    (w2,) = [c for c in out if c["id"] == "w2"]
    assert w2["xy"][:, 0].max() == pytest.approx(512.0, abs=1e-6)
    assert [c["id"] for c in bayous.chains(ww, small_extent, ("stream",), 60.0, [], [])] == ["w2"]


def test_chain_split_at_road(small_extent):
    out = bayous.chains([way("w1", (-200, 0), (200, 0))], small_extent, ("stream",), 0.0, [],
                        [ground_road([(0, -100), (0, 100)], width=8.0)])
    assert len(out) == 2
    assert min(abs(c["xy"][:, 0]).min() for c in out) >= 4.0 + 2.0 - 1e-6


def test_chain_not_split_by_a_bridge_or_a_covered_passage(small_extent):
    roads = [ground_road([(0, -100), (0, 100)], width=8.0, bridge=True),
             ground_road([(50, -100), (50, 100)], oid=2, covered=True)]
    (c,) = bayous.chains([way("w1", (-200, 0), (200, 0))], small_extent, ("stream",), 0.0, [], roads)
    assert LineString(c["xy"]).length == pytest.approx(400.0, abs=0.1)


def test_chain_inside_lake_dropped(small_extent):
    assert bayous.chains([way("w1", (-10, 0), (10, 0))], small_extent, ("stream",), 0.0, [box(-50, -50, 50, 50)], []) == []


def test_chain_cut_where_it_enters_a_lake(small_extent):
    (c,) = bayous.chains([way("w1", (-200, 0), (0, 0))], small_extent, ("stream",), 0.0, [box(-50, -50, 50, 50)], [])
    assert c["xy"][:, 0].max() == pytest.approx(-50.0, abs=1e-6)


# ------------------------------------------------------------------ profile

def test_profile_finds_the_channel(small_extent):
    heights = channel(small_extent, depth=1.7, top_width=8.0)   # trapezoid along y = 0
    (run,) = bayous.profile(np.array([[-200, 0], [200, 0]]), "stream", heights, small_extent, 10.0)
    bottom = ground_level(small_extent) - 1.7
    assert run.shape == (41, 4)
    assert np.allclose(run[:, 2], bottom + 0.3, atol=0.1)
    assert np.all((run[:, 3] >= 2) & (run[:, 3] <= 12))
    assert np.allclose(run[:, 1], 0.0) and np.allclose(np.diff(run[:, 0]), 10.0)


def test_profile_width_is_the_span_under_the_surface(small_extent):
    heights = channel(small_extent, depth=1.7, top_width=20.0)  # under the surface for |y| < 5.9
    (run,) = bayous.profile(np.array([[-200, 0], [200, 0]]), "stream", heights, small_extent, 10.0)
    assert np.allclose(run[:, 3], 2 * (10 - 5 * 1.4 / 1.7), atol=0.3)
    (narrow,) = bayous.profile(np.array([[-200, 0], [200, 0]]), "ditch", heights, small_extent, 10.0)
    assert np.allclose(narrow[:, 3], 5.0)                      # clamped to the ditch's 1.5 to 5 m


def test_profile_finds_a_channel_off_the_centreline(small_extent):
    heights = channel(small_extent, depth=1.7, top_width=8.0)
    (run,) = bayous.profile(np.array([[-200, 3.0], [200, 3.0]]), "stream", heights, small_extent, 10.0)
    assert np.allclose(run[:, 2], ground_level(small_extent) - 1.7 + 0.3, atol=0.1)


def channel_at(extent, y0, depth=1.7, top_width=8.0, n=1024):
    """channel() with its middle along y = y0. On a pixel centre (y0 = k + 0.5) every kink of its
    sides lies on one, so the terrain sampled there is the trapezoid exactly."""
    _, ys = extent.pixel_centers(n)
    f = np.clip((top_width / 2 - np.abs(ys - y0)) / (top_width / 4), 0.0, 1.0)
    return np.tile((GROUND - depth * f)[:, None], (1, n))


HALF = 4 - 2 * 1.4 / 1.7   # half the span of channel(1.7, 8.0) under its bottom + FILL


def edge_terrain(run, extent, heights):
    """The terrain under both edges of a straight run along x (edges at y -+ width / 2)."""
    x, y, w = run[:, 0], run[:, 1], run[:, 3]
    return (sample_height(heights, extent, x, y - w / 2), sample_height(heights, extent, x, y + w / 2))


def test_profile_moves_the_nodes_onto_the_channel(small_extent):
    heights = channel_at(small_extent, 0.5)
    (run,) = bayous.profile(np.array([[-200, 3.5], [200, 3.5]]), "stream", heights, small_extent, 10.0)
    assert np.allclose(run[:, 1], 0.5, atol=0.01)              # off the OSM line, onto the channel
    assert np.allclose(run[:, 3], 2 * HALF, atol=0.01)
    for side in edge_terrain(run, small_extent, heights):      # both edges on the banks, none mid-channel
        assert np.allclose(side, run[:, 2], atol=0.01)


def test_profile_smooths_the_nodes_and_reaches_both_banks(small_extent):
    heights = channel_at(small_extent, 0.5)
    xs, _ = small_extent.pixel_centers(1024)
    jog = np.abs(xs) < 3                                        # the channel 3 m to the side at one node
    heights[:, jog] = channel_at(small_extent, 3.5)[:, jog]
    (run,) = bayous.profile(np.array([[-200, 0.5], [200, 0.5]]), "stream", heights, small_extent, 10.0)
    assert np.allclose(run[:, 1], 0.5, atol=0.01)              # a running median: the node stays in line
    i = int(np.argmin(np.abs(run[:, 0])))
    assert run[i, 3] == pytest.approx(2 * (3.0 + HALF), abs=0.01)   # wide enough to reach its far bank
    raw = np.full(len(run), 2 * HALF)
    raw[i] = 2 * (3.0 + HALF)
    assert np.allclose(run[:, 3], smoothed_widths(raw), atol=0.01)   # its neighbours are widened to ramp down from it
    assert np.all(run[:, 3] >= raw - 1e-9)


def test_profile_width_is_under_the_water_drawn(small_extent):
    heights = channel_at(small_extent, 0.5)
    xs, _ = small_extent.pixel_centers(1024)
    heights[:, np.abs(xs) < 3] -= 1.0                           # a pit in the bed at one node
    (run,) = bayous.profile(np.array([[-200, 0.5], [200, 0.5]]), "stream", heights, small_extent, 10.0)
    i = int(np.argmin(np.abs(run[:, 0])))
    assert run[i, 2] == pytest.approx(GROUND - 1.7 + 0.3)      # the median's surface, not the pit's
    assert run[i, 3] == pytest.approx(2 * (4 - 2 * 0.4 / 1.7), abs=0.01)   # the span under that surface


def test_profile_node_dry_under_the_smoothed_surface(small_extent):
    heights = channel(small_extent, depth=1.7, top_width=8.0)
    xs, _ = small_extent.pixel_centers(1024)
    heights[:, np.abs(xs) < 3] += 1.0                           # a sill in the bed at one node, over the water
    (run,) = bayous.profile(np.array([[-200, 0], [200, 0]]), "stream", heights, small_extent, 10.0)
    assert run.shape == (41, 4) and np.all(np.isfinite(run))
    assert np.allclose(run[:, 1], 0.0, atol=0.01) and np.all((run[:, 3] >= 2) & (run[:, 3] <= 12))


def test_profile_leaves_a_node_whose_water_runs_past_its_section(small_extent):
    # a bank along y = -3.5 and low ground under the water from y = -1.5 on, past the section's end:
    # the span has no far side, so no middle to move to; the node stays and its width is its span
    _, ys = small_extent.pixel_centers(1024)
    f = np.clip((ys + 3.5) / 2, 0.0, 1.0)
    heights = np.tile((GROUND - 1.7 * f)[:, None], (1, 1024))
    (run,) = bayous.profile(np.array([[-200, 0.5], [200, 0.5]]), "stream", heights, small_extent, 10.0)
    assert np.allclose(run[:, 1], 0.5) and np.allclose(run[:, 3], 12.0)   # 15 + 2.35 m, clamped


def test_profile_keeps_the_nodes_in_the_box(small_extent):
    # the OSM line leaves the channel as it runs to the box's edge (x = 512): its last node, moved
    # back onto the channel along its normal, would land 0.2 m past the edge
    heights = channel_at(small_extent, 0.5, top_width=20.0)
    (run,) = bayous.profile(np.array([[300, 3.0], [512, 3.0 + 8 * 212 / 300]]), "stream", heights, small_extent, 10.0)
    assert np.allclose(run[:, 1], 0.5, atol=0.01)
    assert run[-1, 0] == pytest.approx(512.0) and run[:, 0].max() <= 512.0


def test_profile_skips_flat_ground(small_extent):
    heights = channel_with_flat_middle(small_extent)            # channel, then 100 m of flat ground, then channel
    runs = bayous.profile(np.array([[-300, 0], [300, 0]]), "stream", heights, small_extent, 10.0)
    assert len(runs) == 2
    assert runs[0][:, 0].max() < -40 and runs[1][:, 0].min() > 40


def test_profile_smooths_z_with_a_running_median(small_extent):
    heights = channel(small_extent, depth=1.7, top_width=8.0)
    xs, _ = small_extent.pixel_centers(1024)
    heights[:, np.abs(xs) < 3] -= 1.0                           # a pit in the bed at one node
    (run,) = bayous.profile(np.array([[-200, 0], [200, 0]]), "stream", heights, small_extent, 10.0)
    assert np.allclose(run[:, 2], GROUND - 1.7 + 0.3, atol=0.1)


def test_profile_of_flat_ground_is_empty(small_extent):
    assert bayous.profile(np.array([[-200, 0], [200, 0]]), "stream", FLAT, small_extent, 10.0) == []


def alternating_channel(extent, narrow=4.0, wide=9.0, n=1024):
    """channel() whose top width alternates narrow / wide every 10 m along x (blocks [-200, -190),
    [-190, -180), ... so that nodes 10 m apart starting at x = -195 sit in the middle of each)."""
    xs, ys = extent.pixel_centers(n)
    top = np.where(np.floor((xs + 200.0) / 10.0) % 2 == 0, narrow, wide)
    f = np.clip((top[None, :] / 2 - np.abs(ys)[:, None]) / (top[None, :] / 4), 0.0, 1.0)
    return GROUND - 1.7 * f


def smoothed_widths(values, window=5):
    """The widths as they should come out, written out: the mean over the window centred on each node
    (shorter at both ends) of the widest width within the window centred on each of its nodes."""
    values = list(values)
    half = window // 2
    widest = [max(values[max(j - half, 0):j + half + 1]) for j in range(len(values))]
    return np.array([np.mean(widest[max(i - half, 0):i + half + 1]) for i in range(len(values))])


def span_width(extent, top_width):
    """The width profile gives along a channel of one top width (nothing to smooth): the span under the water."""
    (run,) = bayous.profile(np.array([[-200, 0], [200, 0]]), "stream", channel(extent, 1.7, top_width), extent, 10.0)
    return float(run[0, 3])


def test_widths_are_smoothed(small_extent):
    # the channel's top width alternates 4 m / 9 m every 10 m, so the span under the water alternates
    # from node to node: raw, a staircase whose steps are the whole difference between the two spans
    heights = alternating_channel(small_extent)
    narrow, wide = span_width(small_extent, 4.0), span_width(small_extent, 9.0)
    assert wide - narrow > 2.5
    (run,) = bayous.profile(np.array([[-195, 0], [205, 0]]), "stream", heights, small_extent, 10.0)
    width = run[:, 3]
    lo, hi = bayous.WIDTH["stream"]
    assert len(width) == 41 and np.all((width >= lo) & (width <= hi))
    assert np.abs(np.diff(width)).max() < 1.5
    assert np.allclose(run[:, 1], 0.0, atol=0.01)             # smoothing the widths leaves the nodes where they were
    raw = np.where(np.arange(41) % 2 == 0, narrow, wide)       # node 0, at x = -195, is in a narrow block
    assert np.all(width >= raw - 1e-9)                         # none narrower than the span reaching both its banks


def test_widths_smoothing_follows_a_wide_stretch_into_its_neighbours(small_extent):
    # a channel with one wide block among narrow ones: the node in it is as wide as its banks need, the
    # four beside it are no narrower than their own banks and the widths ramp down to the narrow span
    heights = channel(small_extent, 1.7, 4.0)
    xs, _ = small_extent.pixel_centers(1024)
    block = (xs >= -100.0) & (xs < -90.0)
    heights[:, block] = channel(small_extent, 1.7, 9.0)[:, block]
    narrow, wide = span_width(small_extent, 4.0), span_width(small_extent, 9.0)
    (run,) = bayous.profile(np.array([[-195, 0], [205, 0]]), "stream", heights, small_extent, 10.0)
    raw = np.full(41, narrow)
    raw[10] = wide                                              # x = -95
    assert np.allclose(run[:, 3], smoothed_widths(raw), atol=0.01)
    assert run[10, 3] == pytest.approx(wide, abs=0.01) and run[0, 3] == pytest.approx(narrow, abs=0.01)
    assert np.abs(np.diff(run[:, 3])).max() < 1.5
    assert np.all(run[:, 3] >= raw - 1e-9)


def test_smooth_widths_average_the_widest_within_the_window():
    w = np.array([3.0, 8.0, 4.0, 4.0, 6.0, 3.0, 3.0, 3.0, 9.0, 4.0, 4.0])
    out = bayous._smooth_widths(w, 2.0, 12.0)
    assert np.allclose(out, smoothed_widths(w))
    assert out[0] == pytest.approx((8 + 8 + 8) / 3)             # a window of three nodes at the run's end
    assert np.all(out >= w - 1e-9)                              # never narrower than the span that reaches the banks


def test_smooth_widths_stay_within_the_kind_s_range():
    w = np.array([1.5, 1.5, 12.0, 12.0, 1.5, 12.0, 1.5])
    out = bayous._smooth_widths(w, 2.0, 12.0)
    assert out.shape == w.shape and np.all((out >= 2.0) & (out <= 12.0))
    assert np.allclose(bayous._smooth_widths(np.full(4, 1.5), 2.0, 12.0), 2.0)   # clamped from below
    assert np.allclose(bayous._smooth_widths(np.full(4, 13.0), 2.0, 12.0), 12.0)  # and from above
    assert np.allclose(bayous._smooth_widths(np.full(2, 7.0), 2.0, 12.0), 7.0)   # a two-node run has a window too
    assert np.allclose(bayous._smooth_widths(np.array([4.0, 9.0]), 2.0, 12.0), 9.0)


def test_nodes_follow_the_channel(small_extent):
    # a straight trapezoid channel along y = 3.5, the OSM line 3 m to its side at y = 0.5
    heights = channel_at(small_extent, 3.5)
    (run,) = bayous.profile(np.array([[-200, 0.5], [200, 0.5]]), "stream", heights, small_extent, 10.0)
    assert np.allclose(run[:, 1], 3.5, atol=0.5)
    assert np.allclose(run[:, 2], GROUND - 1.7 + bayous.FILL, atol=0.1)
    for side in edge_terrain(run, small_extent, heights):     # the smoothed width still has both edges on the banks
        assert np.allclose(side, run[:, 2], atol=0.05)


# ------------------------------------------------------------------ keep clear

def oblique_run(ts, angle_deg=20.0, width=8.0, z=5.0):
    """A run of nodes at distances ts along a line through the origin, angle_deg to the x axis."""
    a = math.radians(angle_deg)
    ts = np.asarray(ts, float)
    return np.column_stack([ts * math.cos(a), ts * math.sin(a), np.full(len(ts), z), np.full(len(ts), width)])


def worst_overlap(run, corridors):
    """The most any ribbon segment polygon of run (the segment between two nodes buffered by the
    larger of their half-widths, flat caps) shares with the corridors, in m2."""
    worst = 0.0
    for a, b in zip(run[:-1], run[1:]):
        ribbon = LineString([a[:2], b[:2]]).buffer(max(a[3], b[3]) / 2, cap_style="flat")
        worst = max(worst, max(ribbon.intersection(c).area for c in corridors))
    return worst


def test_ribbon_segment_over_a_road_is_cut(small_extent):
    # a bayou crossing a ground road at 20 degrees. The terrain is above its water everywhere, so each
    # node passes the check for water on the road, yet the 8 m wide ribbon between nodes covers the corridor
    road = ground_road([(-150, 0), (150, 0)])                  # corridor |y| <= 3.25
    corridors = bayous._drivable_corridors([road], 0.0)
    run = oblique_run(np.arange(-60.0, 60.1, 10.0))
    assert worst_overlap(run, corridors) > 10.0                # the raw run covers the road
    away = run + np.array([0, 100.0, 0, 0])                      # the same run, 100 m from the road
    pieces = bayous._keep_clear([run, away], corridors, [], np.full((N, N), 7.0), small_extent)
    assert len(pieces) == 3                                     # the run, split at the road; the other, whole
    for piece in pieces:
        assert len(piece) >= 2 and worst_overlap(piece, corridors) <= bayous.RIBBON_OVERLAP
    assert pieces[0][0, 0] == run[0, 0] and pieces[1][-1, 0] == run[-1, 0]   # both ends of the run kept
    assert pieces[0][-1, 1] < 0 < pieces[1][0, 1]                              # one piece each side of the road
    assert np.array_equal(pieces[2], away)


def test_ribbon_segment_between_nodes_clear_of_a_road_is_cut(small_extent):
    # wet ground, and nodes 47 m apart each side of the road with their cross-sections clear of it:
    # the node checks pass, the ribbon between them runs over the road. Neither side keeps 2 nodes.
    road = ground_road([(-150, 0), (150, 0)])
    corridors = bayous._drivable_corridors([road], 0.0)
    run = oblique_run([-25.0, 25.0], width=6.0)
    assert bayous._keep_clear([run], corridors, [], np.full((N, N), 3.0), small_extent) == []


def test_ribbon_segment_beside_a_road_is_kept(small_extent):
    road = ground_road([(-150, 0), (150, 0)])
    corridors = bayous._drivable_corridors([road], 0.0)
    run = oblique_run([-60.0, -50.0, -40.0], angle_deg=0.0, width=4.0)
    run[:, 1] = 6.0                                             # its edge 0.75 m off the corridor's
    (kept,) = bayous._keep_clear([run], corridors, [], np.full((N, N), 3.0), small_extent)
    assert np.array_equal(kept, run)


# ------------------------------------------------------------------ swamp pools

def test_pool_stays_below_roads(small_extent):
    heights = bumpy(small_extent, base=5.0, amp=0.5)              # p25 inside the swamp is about 4.9
    road = ground_road([(-100, 0), (100, 0)], width=8)             # along the level band, the terrain about 5.0
    (p,) = bayous.pools([swamp_ll(small_extent, square(150))], heights, small_extent, [road])
    under = sample_height(heights, small_extent, *corridor_samples(road)).min()
    assert p["level_z"] <= under - 0.3 + 1e-6


def test_pool_dropped_when_nothing_shows(small_extent):
    road = ground_road([(-100, 0), (100, 0)], width=8)
    assert bayous.pools([swamp_ll(small_extent, square(150))], slope_east(small_extent, 10, 0.2), small_extent, [road]) == []


def test_pool_level_rects_and_kinds(small_extent):
    heights = bumpy(small_extent, base=5.0, amp=0.5)
    wetlands = [swamp_ll(small_extent, square(150), id="w1"), swamp_ll(small_extent, square(60, 300, 0), "marsh", "w2"),
                swamp_ll(small_extent, square(60, -300, 0), "tidalflat", "w3"),
                swamp_ll(small_extent, square(60, 0, 300), None, "w4")]
    out = bayous.pools(wetlands, heights, small_extent, [])
    assert [p["id"] for p in out] == ["w1", "w2"]
    poly = shapely.Polygon(square(150))
    xs, ys = small_extent.pixel_centers(N)
    X, Y = np.meshgrid(xs, ys)
    inside = heights[shapely.contains_xy(poly, X, Y)]
    assert out[0]["level_z"] == pytest.approx(np.percentile(inside, 25) + 0.15, abs=1e-3)
    assert out[0]["rects"] == [pytest.approx(list(r), abs=1e-3) for r in level.water_rects(poly)]


def test_pool_ignores_bridges_and_covered_roads(small_extent):
    heights = bumpy(small_extent, base=5.0, amp=0.5)
    roads = [ground_road([(-100, 0), (100, 0)], width=8, bridge=True),
             ground_road([(-100, 0), (100, 0)], width=8, oid=2, covered=True)]
    (free,) = bayous.pools([swamp_ll(small_extent, square(150))], heights, small_extent, [])
    (p,) = bayous.pools([swamp_ll(small_extent, square(150))], heights, small_extent, roads)
    assert p["level_z"] == free["level_z"]


def in_rects(rects, extent, n=N):
    """Which cells of an n x n grid have their centres in one of the (cx, cy, sx, sy) rects."""
    xs, ys = extent.pixel_centers(n)
    X, Y = np.meshgrid(xs, ys)
    out = np.zeros((n, n), bool)
    for cx, cy, sx, sy in rects:
        out |= (np.abs(X - cx) <= sx / 2) & (np.abs(Y - cy) <= sy / 2)
    return out


def cells_in(geom, extent, n=N):
    xs, ys = extent.pixel_centers(n)
    X, Y = np.meshgrid(xs, ys)
    return shapely.intersects_xy(geom, X, Y)


def covered(pool, poly) -> float:
    """The m2 of poly under the pool's rects."""
    return shapely.union_all([box(cx - sx / 2, cy - sy / 2, cx + sx / 2, cy + sy / 2)
                              for cx, cy, sx, sy in pool["rects"]]).intersection(poly).area


def written(rects) -> np.ndarray:
    """x0, y0, x1, y1 (a (K, 4) array) of the WaterBlocks bf.water_block writes for (cx, cy, sx, sy) rects:
    what the game gets, the centre and size to the centimetre."""
    out = []
    for r in rects:
        b = bf.water_block(*r, 0.0, bayous.RIVER_DEPTH, "test")
        (x, y, _), (w, d, _) = b["position"], b["scale"]
        out.append((x - w / 2, y - d / 2, x + w / 2, y + d / 2))
    return np.asarray(out, float).reshape(-1, 4)


def block_cells(rect, heights, extent):
    """The cell centres xs, ys and the terrain of the cells (centres) inside rect's WaterBlock (as written)."""
    (x0, y0, x1, y1), = written([rect])
    xs, ys = extent.pixel_centers(heights.shape[0])
    cols, rows = np.flatnonzero((xs >= x0) & (xs <= x1)), np.flatnonzero((ys >= y0) & (ys <= y1))
    return xs[cols], ys[rows], np.asarray(heights[np.ix_(rows, cols)], float)


def wrong_cells(rect, level_z, others, heights, extent):
    """block_cells' xs, ys and terrain h for rect's WaterBlock at level_z, and which of those cells lie inside a
    block of another water (others: [(written boxes, level_z)]) with their terrain under the lower of the two
    levels + 0.05 m: where the lower water shows under the other's surface (a raised slab)."""
    xs, ys, h = block_cells(rect, heights, extent)
    bad = np.zeros(h.shape, bool)
    if not h.size:
        return xs, ys, h, bad
    for edges, z in others:
        near = (edges[:, 0] <= xs[-1]) & (edges[:, 2] >= xs[0]) & (edges[:, 1] <= ys[0]) & (edges[:, 3] >= ys[-1])
        for x0, y0, x1, y1 in edges[near]:
            inside = ((xs >= x0) & (xs <= x1))[None, :] & ((ys >= y0) & (ys <= y1))[:, None]
            bad |= inside & (h < min(level_z, z) + 0.05)
    return xs, ys, h, bad


def wrong_water(rect, level_z, others, heights, extent) -> int:
    """How many cells (centres) inside rect's WaterBlock at level_z show another water's (wrong_cells)."""
    return int(wrong_cells(rect, level_z, others, heights, extent)[3].sum())


def draws_water(rect, level_z, others, swamp, heights, extent) -> bool:
    """Whether rect's WaterBlock (as written) at level_z draws the swamp's water: covers a cell (centre) inside
    the swamp polygon whose terrain is under level_z and where no other water (others, as wrong_cells') shows."""
    xs, ys, h, bad = wrong_cells(rect, level_z, others, heights, extent)
    X, Y = np.meshgrid(xs, ys)
    return bool((shapely.contains_xy(swamp, X, Y) & (h < level_z) & ~bad).any())


def lake_of(poly, heights, extent, level_z=3.0, bed=True) -> WaterBody:
    """A water.json lake at level_z (2 m under the bumpy swamps' ground), its bed (unless bed is False) cut into
    heights 1 m under that."""
    if bed:
        heights[cells_in(poly, extent)] = level_z - 1.0
    return WaterBody("Lake", poly, level_z, 2.0)


# an L-shaped lake: its one water_rects box covers the ground in the corner it wraps round
LAKE_L = shapely.union_all([box(100, -200, 200, 200), box(-200, -200, 200, -100)])
BANK = square(150, -180, 50)      # a swamp on the ground in that corner, partly inside the lake's box


def test_pool_covers_ground_above_a_lower_waters_level_inside_its_box(small_extent):
    # the lake's water is under the ground there, so the swamp keeps its water over all of it
    heights = bumpy(small_extent, base=5.0, amp=0.5)
    lake = lake_of(LAKE_L, heights, small_extent)
    (lake_box,) = written(level.water_rects(LAKE_L))
    swamp = shapely.Polygon(BANK)
    assert swamp.intersection(box(*lake_box)).area > 0.6 * swamp.area
    (p,) = bayous.pools([swamp_ll(small_extent, BANK)], heights, small_extent, [], [lake])
    low = cells_in(swamp.intersection(box(*lake_box)), small_extent) & (heights < p["level_z"])
    assert low.sum() > 100 and (in_rects(p["rects"], small_extent) & low).sum() == low.sum()


def test_pool_keeps_off_a_lower_waters_visible_surface(small_extent):
    # the slab case: a lake inside the swamp, its surface 2 m under the swamp's ground; the swamp's box
    # covers the lake's water, which a pool box over it would raise to the swamp's level
    heights = bumpy(small_extent, base=5.0, amp=0.5)
    lake = lake_of(box(-20, -20, 20, 20), heights, small_extent)
    swamp = shapely.Polygon(square(200))
    (p,) = bayous.pools([swamp_ll(small_extent, square(200))], heights, small_extent, [], [lake])
    lakes = [(written(level.water_rects(lake.polygon)), lake.level_z)]
    assert max(wrong_water(r, p["level_z"], lakes, heights, small_extent) for r in p["rects"]) == 0
    # the rest of the swamp keeps its water, the ground beside the lake inside the lake's box too
    clear = cells_in(swamp, small_extent) & ~cells_in(lake.polygon.buffer(level.WATER_MARGIN), small_extent)
    low = clear & (heights < p["level_z"])
    assert (in_rects(p["rects"], small_extent) & low).sum() >= 0.98 * low.sum()


def test_a_lake_cutting_a_swamp_in_two(small_extent):
    # a lake crossing the swamp, widening north of it so its one box covers all of the swamp: the swamp
    # less the lake is a MultiPolygon, its one start box covers the lake, and both halves keep their water
    heights = bumpy(small_extent, base=5.0, amp=0.5)
    lake = lake_of(shapely.union_all([box(-10, -150, 10, 150), box(-150, 130, 150, 200)]), heights, small_extent)
    swamp = shapely.Polygon(square(200))
    halves = swamp.difference(lake.polygon.buffer(level.WATER_MARGIN))
    assert halves.geom_type == "MultiPolygon" and len(level.water_rects(halves)) == 1
    (lake_box,) = written(level.water_rects(lake.polygon))
    assert box(*lake_box).contains(swamp)
    (p,) = bayous.pools([swamp_ll(small_extent, square(200))], heights, small_extent, [], [lake])
    lakes = [(written(level.water_rects(lake.polygon)), lake.level_z)]
    assert max(wrong_water(r, p["level_z"], lakes, heights, small_extent) for r in p["rects"]) == 0
    wet = in_rects(p["rects"], small_extent)
    for half in halves.geoms:     # less the 9 m beside the lake (27 m pieces holding the lake), a trough of the bumps
        low = cells_in(half, small_extent) & (heights < p["level_z"])
        assert low.sum() > 100 and (wet & low).sum() >= 0.7 * low.sum()


STRIPS = [(-300, -50), (300, -50), (300, 50), (-300, 50)]   # a 600 m swamp: two water_rects strips of 512 m


def high_east(extent):
    """The bumpy ground with 8 m high ground east of x = 200, under all of the STRIPS swamp's east box (x from 204)."""
    heights = bumpy(extent, base=5.0, amp=0.5)
    xs, _ = extent.pixel_centers(N)
    heights[:, xs > 200] = 8.0
    return heights


def test_pools_drop_boxes_that_draw_no_water(small_extent):
    # a 600 m swamp has two boxes (level.water_rects strips of 512 m); the east one is all on high ground
    heights = high_east(small_extent)
    (west, east) = level.water_rects(shapely.Polygon(STRIPS))
    (p,) = bayous.pools([swamp_ll(small_extent, STRIPS)], heights, small_extent, [])
    assert not (block_cells(east, heights, small_extent)[2] < p["level_z"]).any()
    assert p["rects"] == [pytest.approx(list(west), abs=1e-3)]


def test_pools_drop_boxes_whose_only_water_shows_over_a_lake(small_extent):
    # the east box's only cells under the level are a pond's bed, 2 cells inside the pond's box under its level:
    # rule (a) lets a box show that many, but the only water the east box would draw is a slab over the pond's
    heights = high_east(small_extent)
    pond = lake_of(box(248, -4, 256, 0), heights, small_extent)        # its bed: the cells at x 250 and 254, y -2
    (west, east) = level.water_rects(shapely.Polygon(STRIPS))
    (p,) = bayous.pools([swamp_ll(small_extent, STRIPS)], heights, small_extent, [], [pond])
    _, _, h, shown = wrong_cells(east, p["level_z"], [(written(level.water_rects(pond.polygon)), pond.level_z)],
                                 heights, small_extent)
    assert shown.sum() == 2 and (shown == (h < p["level_z"])).all()
    assert p["rects"] == [pytest.approx(list(west), abs=1e-3)]


def test_pools_drop_boxes_whose_only_water_is_off_the_swamp(small_extent):
    # the east box's only cells under the level lie 4 m north of the swamp, in the WATER_MARGIN its box reaches
    # past the swamp's edge: no rule forbids that water there, but the box draws none of the swamp's
    heights = high_east(small_extent)
    xs, ys = small_extent.pixel_centers(N)
    heights[np.ix_(ys == 54, xs > 260)] = 4.0
    swamp = shapely.Polygon(STRIPS)
    (west, east) = level.water_rects(swamp)
    (p,) = bayous.pools([swamp_ll(small_extent, STRIPS)], heights, small_extent, [])
    assert (block_cells(east, heights, small_extent)[2] < p["level_z"]).sum() == 12
    assert not draws_water(east, p["level_z"], [], swamp, heights, small_extent)
    assert p["rects"] == [pytest.approx(list(west), abs=1e-3)]


def test_pool_left_with_no_box_that_draws_water_is_dropped(small_extent):
    # the swamp's low part lies in the L-shaped lake's box under the lake's level, so the lake's water shows
    # there and no pool box may cover it; the rest of the swamp is high ground
    heights = bumpy(small_extent, base=5.0, amp=0.5)
    lake = lake_of(LAKE_L, heights, small_extent)
    (lake_box,) = written(level.water_rects(LAKE_L))
    swamp = shapely.Polygon(BANK)
    heights[cells_in(swamp, small_extent)] = 8.0
    heights[cells_in(swamp.intersection(box(*lake_box)), small_extent)] = lake.level_z - 0.5
    assert bayous.pools([swamp_ll(small_extent, BANK)], heights, small_extent, [], [lake]) == []


def test_pool_level_ignores_cells_under_other_water(small_extent):
    # the south half of the swamp lies in a lake, its bed 3 m under the swamp's ground
    heights = bumpy(small_extent, base=5.0, amp=0.5)
    lake = lake_of(box(-200, -200, 200, 0), heights, small_extent)
    swamp = shapely.Polygon(square(200))
    (p,) = bayous.pools([swamp_ll(small_extent, square(200))], heights, small_extent, [], [lake])
    clear = cells_in(swamp, small_extent) & ~cells_in(lake.polygon.buffer(level.WATER_MARGIN), small_extent)
    assert p["level_z"] == pytest.approx(np.percentile(heights[clear], 25) + 0.15, abs=1e-3)


def test_pool_level_falls_back_to_the_whole_swamp(small_extent):
    # a lake's polygon over all of the swamp but its north 9 m, on ground above its level (as the
    # Mississippi's over the batture): no cell of the swamp is clear of the lake grown by WATER_MARGIN
    heights = bumpy(small_extent, base=5.0, amp=0.5)
    lake = lake_of(box(-200, -200, 200, 39), heights, small_extent, bed=False)
    swamp = shapely.Polygon(square(96))
    assert not (cells_in(swamp, small_extent) & ~cells_in(lake.polygon.buffer(level.WATER_MARGIN), small_extent)).any()
    (p,) = bayous.pools([swamp_ll(small_extent, square(96))], heights, small_extent, [], [lake])
    assert p["level_z"] == pytest.approx(np.percentile(heights[cells_in(swamp, small_extent)], 25) + 0.15, abs=1e-3)


def test_pool_shows_water_by_its_whole_swamp(small_extent):
    # the largest batture swamp's case: a road caps the level under the SHOW_PCT percentile of the cells the pool
    # can draw (clear of the lake), though not under that of the whole swamp, whose lake bed lies far lower. A dip
    # in the drawable ground lies under the level, so water shows: the pool is kept, and its boxes draw that water
    heights = np.full((N, N), 7.0)
    lake = lake_of(box(-100, -100, 100, -72), heights, small_extent)    # its bed at 2 m: 14 % of the swamp's cells
    xs, ys = small_extent.pixel_centers(N)
    dip = (np.abs(xs) < 20)[None, :] & ((ys > 0) & (ys < 16))[:, None]  # 40 cells at 4 m, 2 % of the drawable ones
    heights[dip] = 4.0
    road = ground_road([(-150, 60), (150, 60)])                          # on the 7 m ground: the level 6.7 at most
    swamp = shapely.Polygon(square(200))
    whole = cells_in(swamp, small_extent)
    drawable = whole & ~cells_in(lake.polygon.buffer(level.WATER_MARGIN), small_extent)
    (p,) = bayous.pools([swamp_ll(small_extent, square(200))], heights, small_extent, [road], [lake])
    assert p["level_z"] == pytest.approx(7.0 - bayous.ROAD_FREEBOARD, abs=1e-3)
    assert np.percentile(heights[whole], bayous.SHOW_PCT) <= p["level_z"]
    assert p["level_z"] < np.percentile(heights[drawable], bayous.SHOW_PCT)
    assert (in_rects(p["rects"], small_extent) & dip).sum() == dip.sum() == 40


def test_pool_boxes_are_checked_on_the_blocks_the_game_gets():
    rects = [(10.004, -3.996, 20.006, 7.0049), (0.0051, 0.0, 33.333, 16.0)]
    edges, z = bayous._written(rects, 4.80049)
    assert edges.tolist() == written(rects).tolist()
    assert z == bf.water_block(*rects[0], 4.80049, 1.5, "x")["position"][2] == 4.8


L_SWAMP = [(-100, -100), (100, -100), (100, 0), (0, 0), (0, 100), (-100, 100)]


def test_pool_rects_do_not_flood_outside_the_swamp(small_extent):
    # an L-shaped swamp whose bounding box covers its notch, where the ground lies 2 m lower
    heights = bumpy(small_extent, base=5.0, amp=0.5)
    xs, ys = small_extent.pixel_centers(N)
    notch = ((xs > 0) & (xs < 100))[None, :] & ((ys > 0) & (ys < 100))[:, None]
    heights[notch] = 3.0
    poly = shapely.Polygon(L_SWAMP)
    assert in_rects(level.water_rects(poly), small_extent)[notch].all()
    (p,) = bayous.pools([swamp_ll(small_extent, L_SWAMP)], heights, small_extent, [])
    wet = in_rects(p["rects"], small_extent)
    outside = ~cells_in(poly.buffer(level.WATER_MARGIN), small_extent)
    assert (wet & outside & (heights < p["level_z"] - 0.05)).sum() <= 2
    low = cells_in(poly, small_extent) & (heights < p["level_z"])
    assert low.sum() > 100 and (wet & low).sum() >= 0.9 * low.sum()


def test_pools_at_different_levels_keep_off_each_others_water(small_extent):
    # two swamps side by side, the east one's ground 1 m higher: each one's box reaches WATER_MARGIN over the
    # other, and the east pool must not raise the west one's water where it shows
    heights = bumpy(small_extent, base=5.0, amp=0.5)
    xs, _ = small_extent.pixel_centers(N)
    heights[:, xs > 0] += 1.0
    wetlands = [swamp_ll(small_extent, square(98, 50, 0), id="w2"), swamp_ll(small_extent, square(98, -50, 0), id="w1")]
    (east_box,) = level.water_rects(shapely.Polygon(square(98, 50, 0)))
    (alone,) = bayous.pools(wetlands[1:], heights, small_extent, [])                 # the west swamp's pool
    assert wrong_water(east_box, alone["level_z"] + 1.0, [(written(alone["rects"]), alone["level_z"])], heights,
                       small_extent) > bayous.POOL_FLOOD_CELLS
    out = bayous.pools(wetlands, heights, small_extent, [])
    assert [p["id"] for p in out] == ["w1", "w2"]                           # in the order of their ids
    west, east = out
    assert east["level_z"] - west["level_z"] > 0.5
    assert max(wrong_water(r, east["level_z"], [(written(west["rects"]), west["level_z"])], heights, small_extent)
               for r in east["rects"]) <= bayous.POOL_FLOOD_CELLS
    assert covered(east, shapely.Polygon(square(98, 50, 0))) >= 0.7 * 98 ** 2
    assert bayous.pools(wetlands[::-1], heights, small_extent, []) == out  # whichever comes first in the file


def test_pool_kept_below_a_road_beside_it_under_its_water(small_extent):
    """A water block covers the polygon's bounding box and 8 m round it, so a road just outside
    the polygon is under its water too."""
    heights = bumpy(small_extent, base=5.0, amp=0.5)
    _, ys = small_extent.pixel_centers(N)
    heights[np.abs(ys) >= 70] = 5.1                                # a level strip along the swamp's north and south edges
    road = ground_road([(-100, 79), (100, 79)], width=4)          # 2 m outside the swamp's north edge
    (free,) = bayous.pools([swamp_ll(small_extent, square(150))], heights, small_extent, [])
    (p,) = bayous.pools([swamp_ll(small_extent, square(150))], heights, small_extent, [road])
    under = sample_height(heights, small_extent, *corridor_samples(road)).min()
    assert p["level_z"] < free["level_z"] and p["level_z"] <= under - 0.3 + 1e-3


# ------------------------------------------------------------------ level objects

def test_river_object_fields():
    r = bayous.river_object({"id": "w1", "nodes": [[0, 0, 5, 6], [10, 0, 5, 6]]}, 0)
    assert r["class"] == "River" and r["flowMagnitudePhysics"] == 0
    assert r["nodes"][0] == [0, 0, 5, 6, 1.5, 0, 0, 1]


def test_river_object_is_the_stock_muddy_river_made_murky():
    r = bayous.river_object({"id": "w1+w2", "nodes": [[1.23456, -2, 5.5, 6.25], [10, 0, 5, 6]]}, 3)
    assert set(bayous.STOCK_RIVER) <= set(r)
    assert r["persistentId"] == bf.persistent_id("bayou", "w1+w2", 3)
    assert r["position"] == [1.235, -2.0, 5.5] and r["nodes"][0] == [1.235, -2.0, 5.5, 6.25, 1.5, 0, 0, 1]
    assert r["baseColor"][:3] == [40, 46, 24]
    assert (r["depthGradientTex"], r["depthGradientMax"]) == ("/core/art/water/mud_depthcolor_ramp_b.png", 1.0)
    assert (r["waterFogDensity"], r["waterFogDensityOffset"], r["reflectivity"], r["fullReflect"]) == (8, 0.2, 0.2, False)
    # tuned in game (spike S3): no foam on water this shallow, calm ripples
    assert (r["overallWaveMagnitude"], r["overallRippleMagnitude"]) == (0.02, 0.5)
    assert (r["overallFoamOpacity"], r["foamMaxDepth"]) == (0, 0)
    assert r["clarity"] == 0.05  # near opaque: the default shows the channel bed through the water (spike S3)
    assert bayous.STOCK_RIVER["depthGradientTex"].endswith("depthcolor_ramp_italy_muddy_b.png")
    assert "__parent" not in r and "persistentId" not in bayous.STOCK_RIVER


def test_pool_block_has_the_bayous_still_water_look():
    pool = {"id": "w9", "level_z": 4.8, "rects": [[0, 0, 50, 40], [60, 0, 50, 40]]}
    b = bayous.pool_block(pool, 2, 1)
    plain = bf.water_block(60, 0, 50, 40, 4.8, bayous.RIVER_DEPTH, "x")
    assert b["class"] == "WaterBlock" and b["position"] == plain["position"] and b["scale"] == plain["scale"]
    river = bayous.river_object({"id": "w1", "nodes": [[0, 0, 5, 6], [10, 0, 5, 6]]}, 0)
    for key, value in bayous.MURKY.items():
        assert b[key] == river[key] == value, key
    # foam on water this shallow reads as grey gravel (spike S3); the rivers' mud depth ramp
    assert (b["overallFoamOpacity"], b["foamMaxDepth"]) == (0, 0)
    assert (b["depthGradientTex"], b["depthGradientMax"]) == ("/core/art/water/mud_depthcolor_ramp_b.png", 1.0)
    assert "flowMagnitudePhysics" not in b                                 # the River's own physics
    assert b["persistentId"] != bayous.pool_block(pool, 2, 0)["persistentId"]


def test_level_content(tmp_path):
    data = tmp_path / "out"
    runs = [{"id": "w1+w2", "name": "Bayou Fountain", "kind": "stream", "nodes": [[0, 0, 5, 6], [10, 0, 5, 6]]},
            {"id": "w3", "name": None, "kind": "ditch", "nodes": [[0, 50, 5, 2], [10, 50, 5, 2]]}]
    pools = [{"id": "w9", "level_z": 4.8, "rects": [[0, 0, 50, 40], [60, 0, 50, 40]]}]
    bayous.save_bayous({"runs": runs, "pools": pools}, data / "bayous.json")
    groups = bayous.level_content(Config(), data, tmp_path / "level", FLAT, EXT)
    objs = groups["bayous"]
    assert [o["class"] for o in objs] == ["River", "River", "WaterBlock", "WaterBlock"]
    assert [o["persistentId"] for o in objs[:2]] == [bf.persistent_id("bayou", "w1+w2", 0),
                                                     bf.persistent_id("bayou", "w3", 1)]
    assert len({o["persistentId"] for o in objs}) == 4
    bayous.save_bayous({"runs": [], "pools": []}, data / "bayous.json")
    assert bayous.level_content(Config(), data, tmp_path / "level", FLAT, EXT) == {}


# ------------------------------------------------------------------ the stage

def stage_inputs(tmp_path, roads=(), ways=None, heights=None, water=()):
    raw, out = tmp_path / "raw", tmp_path / "out"
    raw.mkdir()
    out.mkdir()
    if ways is None:
        ways = [way("w1", (-200, 0), (200, 0), name="Bayou Fountain"), way("w2", (0, 300), (100, 300))]
    osm = {"waterways": ways, "wetlands": [swamp_ll(EXT, square(100, 0, -250))]}
    (raw / "osm_features.json").write_text(json.dumps(osm), encoding="utf-8")
    np.save(out / "heights.npy", (channel(EXT, 1.7, 8.0) if heights is None else heights).astype(np.float32))
    save_roads(list(roads), out / "roads.json")
    save_water(list(water), out / "water.json")
    return raw, out


def test_run_writes_bayous_json(tmp_path):
    _, out = stage_inputs(tmp_path)
    bayous.run(small_cfg(tmp_path))
    got = bayous.load_bayous(out / "bayous.json")
    assert set(got) == {"runs", "pools"}
    (r,) = got["runs"]                     # w2 lies on flat ground: no water there
    assert (r["id"], r["name"], r["kind"]) == ("w1", "Bayou Fountain", "stream")
    assert len(r["nodes"]) == 41 and all(len(n) == 4 for n in r["nodes"])
    (p,) = got["pools"]
    assert set(p) == {"id", "level_z", "rects"} and p["id"] == "w9"


def test_run_cuts_the_bayou_at_a_road(tmp_path):
    _, out = stage_inputs(tmp_path, [ground_road([(0, -100), (0, 100)], width=8.0)])
    bayous.run(small_cfg(tmp_path))
    assert len(bayous.load_bayous(out / "bayous.json")["runs"]) == 2


def test_run_keeps_the_water_off_a_road_it_would_flood(tmp_path):
    # the channel along y = 0.5 and its OSM line 6 m off it, so the nodes move onto the channel; a
    # road (not cut by chains: its corridor + ROAD_GAP stops at y = 2) lies in the water for 50 < x < 150
    road = ground_road([(50, -1), (150, -1)], width=2.0)
    _, out = stage_inputs(tmp_path, [road], ways=[way("w1", (-200, 6.5), (200, 6.5))], heights=channel_at(EXT, 0.5))
    bayous.run(small_cfg(tmp_path))
    runs = [np.asarray(r["nodes"]) for r in bayous.load_bayous(out / "bayous.json")["runs"]]
    assert [(r[0, 0], r[-1, 0]) for r in runs] == [(-200, 40), (160, 200)]
    assert np.allclose(np.concatenate(runs)[:, 1], 0.5, atol=0.01)


def test_run_keeps_the_water_where_a_road_stands_over_it(tmp_path):
    # a river's 6 m minimum width puts the ribbon's edge at y = -2.5, over its bank and 0.1 m short of the
    # road's corridor (y < -2.6); the terrain there is above the water, so the run stays whole
    road = ground_road([(50, -3.6), (150, -3.6)], width=2.0)
    _, out = stage_inputs(tmp_path, [road], ways=[way("w1", (-200, 0.5), (200, 0.5), kind="river")],
                          heights=channel_at(EXT, 0.5))
    bayous.run(small_cfg(tmp_path))
    (r,) = bayous.load_bayous(out / "bayous.json")["runs"]
    nodes = np.asarray(r["nodes"])
    assert len(nodes) == 41 and np.allclose(nodes[:, 3], 6.0) and np.allclose(nodes[:, 1], 0.5, atol=0.01)


def test_run_cuts_the_water_whose_ribbon_covers_a_road(tmp_path):
    # the same river with the road 1.1 m nearer: its 6 m ribbon (y > -2.5) covers 0.5 m of the corridor
    # (y < -2). The terrain there is above the water, but the rule is the ribbon's, not the water's height:
    # the segments over the road go, with their end nodes (the corridor's rounded ends reach x = 49 and 151)
    road = ground_road([(50, -3.0), (150, -3.0)], width=2.0)
    _, out = stage_inputs(tmp_path, [road], ways=[way("w1", (-200, 0.5), (200, 0.5), kind="river")],
                          heights=channel_at(EXT, 0.5))
    bayous.run(small_cfg(tmp_path))
    runs = [np.asarray(r["nodes"]) for r in bayous.load_bayous(out / "bayous.json")["runs"]]
    assert [(r[0, 0], r[-1, 0]) for r in runs] == [(-200, 40), (170, 200)]


def test_run_drops_nodes_moved_into_a_lake(tmp_path):
    lake = WaterBody(None, box(45, -50, 300, 2), 4.0, 2.0)     # beside the OSM line, over the channel
    _, out = stage_inputs(tmp_path, ways=[way("w1", (-200, 6.5), (200, 6.5))], heights=channel_at(EXT, 0.5),
                          water=[lake])
    bayous.run(small_cfg(tmp_path))
    (r,) = bayous.load_bayous(out / "bayous.json")["runs"]
    assert (r["nodes"][0][0], r["nodes"][-1][0]) == (-200, 40)


def test_run_writes_nothing_to_draw_when_off(tmp_path):
    _, out = stage_inputs(tmp_path)
    bayous.run(small_cfg(tmp_path, bayous_enabled=False))
    assert bayous.load_bayous(out / "bayous.json") == {"runs": [], "pools": []}


def test_run_keeps_the_pools_off_the_lakes(tmp_path):
    heights = FLAT.copy()
    lake = lake_of(box(-20, -270, 20, -230), heights, EXT)          # a lake inside the stage's swamp
    _, out = stage_inputs(tmp_path, water=[lake], heights=heights)
    bayous.run(small_cfg(tmp_path))
    (p,) = bayous.load_bayous(out / "bayous.json")["pools"]
    lakes = [(written(level.water_rects(lake.polygon)), lake.level_z)]
    assert max(wrong_water(r, p["level_z"], lakes, heights, EXT) for r in p["rects"]) == 0


ROOT = Path(__file__).resolve().parents[1]
REAL_OUT, REAL_OSM = ROOT / "data" / "out", ROOT / "data" / "raw" / "osm_features.json"
REAL_INPUTS = [REAL_OSM] + [REAL_OUT / f for f in ("heights.npy", "water.json", "roads.json")]


def swamp_polygons(wetlands, extent) -> dict:
    """The swamp polygons bayous.pools() draws pools in (level coordinates, clipped to the box), by pool id."""
    h = extent.half_size
    out = {}
    for w in wetlands:
        if w.get("wetland") in bayous.POOL_KINDS:
            polygons = w.get("polygons") or []
            for k, p in enumerate(polygons):
                poly = bayous._wetland_polygon(p, extent).intersection(box(-h, -h, h, h))
                out[str(w["id"]) if len(polygons) == 1 else f"{w['id']}/{k}"] = bayous._polygonal(poly)
    return out


@pytest.mark.skipif(not all(f.exists() for f in REAL_INPUTS),
                    reason="no full-map data/raw/osm_features.json, data/out/heights.npy, water.json or roads.json")
def test_real_pools_keep_off_the_water():
    # on the full map the swamps' boxes stood as slabs over the Mississippi and the batture's lakes
    from brmap.config import load_config
    from brmap.roads import load_roads
    from brmap.water import load_water

    extent = Extent.from_config(load_config(ROOT / "config.yaml"))
    water = load_water(REAL_OUT / "water.json")
    wetlands = json.loads(REAL_OSM.read_text(encoding="utf-8")).get("wetlands") or []
    heights = np.load(REAL_OUT / "heights.npy")
    got = bayous.pools(wetlands, heights, extent, load_roads(REAL_OUT / "roads.json"), water)
    assert got and sum(len(p["rects"]) for p in got) < 400
    # the largest batture swamp keeps its water: a road caps its level under the SHOW_PCT percentile of the cells
    # it can draw, but not under that of its whole swamp
    assert "w1462559887" in [p["id"] for p in got]
    swamps = swamp_polygons(wetlands, extent)
    lakes = [(written(level.water_rects(b.polygon)), b.level_z) for b in water]
    for i, p in enumerate(got):
        # the water.json bodies and the pools before it (in id order) at another level
        others = lakes + [(written(q["rects"]), q["level_z"]) for q in got[:i]
                          if abs(q["level_z"] - p["level_z"]) > bayous.POOL_LEVEL_TOL]
        for r in p["rects"]:
            assert wrong_water(r, p["level_z"], others, heights, extent) <= bayous.POOL_FLOOD_CELLS, (p["id"], r)
            # each block draws some of its swamp's own water, not only the wrong water the rules tolerate
            assert draws_water(r, p["level_z"], others, swamps[p["id"]], heights, extent), (p["id"], r)


def test_trees_not_in_channel():
    kept, dropped = trees.exclude(tops_at([(0, 0.5, 10), (0, 10, 10)]), [], [], [], bayous=[run_along_x(width=4)])
    assert len(kept) == 1 and dropped["bayou"] == 1


def test_trees_kept_off_a_run_by_its_half_width_and_half_a_metre():
    runs = [{"id": "w1", "name": None, "kind": "stream", "nodes": [[-100, 0, 5, 4], [0, 0, 5, 4], [100, 0, 5, 10]]}]
    kept, dropped = trees.exclude(tops_at([(-50, 2.4, 10), (-50, 2.6, 10), (99, 5.4, 10), (99, 5.6, 10)]), [], [],
                                  [], bayous=runs)
    assert sorted(map(tuple, kept[:, :2])) == [(-50.0, 2.6), (99.0, 5.6)] and dropped["bayou"] == 2


def test_trees_run_keeps_trees_out_of_the_bayous(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (tmp_path / "raw").mkdir()
    np.save(out / "heights.npy", np.full((N, N), 6.0, np.float32))
    n, px = canopy.analysis_grid(EXT)
    xs, ys = EXT.pixel_centers(n)
    X, Y = np.meshgrid(xs, ys)
    ndsm = (12.0 * np.exp(-((X - 201.0) ** 2 + (Y - 5.0) ** 2) / 18.0)).astype(np.float32)
    canopy.save_canopy(out, ndsm, np.full((n, n), 0.6, np.float32), px)
    cfg = Config(debug_half_size_m=512.0, terrain_size=N, raw_dir=str(tmp_path / "raw"), out_dir=str(out))
    trees.run(cfg)
    assert len(trees.load_trees(out / "trees.json")) == 1
    bayous.save_bayous({"runs": [run_along_x(6.0, 150.0, 250.0, y=5.0)], "pools": []}, out / "bayous.json")
    trees.run(cfg)
    assert trees.load_trees(out / "trees.json") == []


# ------------------------------------------------------------------ checks

def test_bayou_problems():
    good = run_along_x(4.0)
    runs = [good, run_along_x(4.0, x1=600.0, id="w2"), run_along_x(0.0, id="w3"),
            {"id": "w4", "name": "Ward Creek", "kind": "stream", "nodes": [[0, 0, 5, 4]]}]
    probs = validate.bayou_problems({"runs": runs, "pools": []}, EXT)
    assert len(probs) == 3
    assert "w2" in probs[0] and "outside the box" in probs[0]
    assert "w3" in probs[1] and "width" in probs[1]
    assert "w4" in probs[2] and "1 node" in probs[2]
    assert validate.bayou_problems({"runs": [good], "pools": []}, EXT) == []


def test_feature_problems_check_the_bayous(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out"
    bayous.save_bayous({"runs": [run_along_x(4.0, x1=600.0, id="w2")], "pools": []}, out / "bayous.json")
    (msg,) = validate.feature_problems(Config(debug_half_size_m=512.0, terrain_size=N), out, EXT, FLAT, [])
    assert "w2" in msg and "outside the box" in msg


def test_bayous_stage_and_level_hook_registered():
    stages = cli.FEATURE_STAGES
    assert stages.index("models") < stages.index("bayous") < stages.index("buildings")
    assert "bayous" in cli.STAGES and "bayous" in cli.ALL_ORDER
    assert level.FEATURE_MODULES.index("landmarks") < level.FEATURE_MODULES.index("bayous")
    assert bayous.__doc__
