"""Lane markings (plan 2026-10-16, Task 5): lane lines, junction trims, stop bars and crosswalks."""
import json
import math
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import LineString, Point

from brmap import beamng_format as bf
from brmap import markings
from brmap.config import Config
from brmap.markings import Marking
from brmap.roads import Road, save_roads

GAME = Path(os.environ.get("BEAMNG_GAME_DIR",
                           r"C:\Program Files (x86)\Steam\steamapps\common\BeamNG.drive"))
DECALROAD_ZIP = GAME / "content" / "assets" / "materials" / "decalroad.zip"


def road(kind, xy, width, lanes, oneway=False, bridge=False, z=5.0, osm=1):
    xy = np.asarray(xy, float)
    zz = np.full(len(xy), z) if np.ndim(z) == 0 else np.asarray(z, float)
    return Road([osm], kind, np.column_stack([xy, zz]), width, lanes, oneway, bridge,
                1 if bridge else 0, None)


def line(x0, y0, x1, y1, n=41):
    return np.column_stack([np.linspace(x0, x1, n), np.linspace(y0, y1, n)])


def flat(n=64, z=5.0):
    return np.full((n, n), z, np.float32)


def node(nid, x, y, **tags):
    return {"id": nid, "x": float(x), "y": float(y), "tags": tags}


def by_offset(lines):
    return sorted(lines, key=lambda t: t[1])


# ------------------------------------------------------------------ lane lines

def test_two_way_tertiary_lines():
    r = road("tertiary", line(0, 0, 100, 0), 6.5, 2)
    assert markings.lane_lines(r) == [("centre_double", 0.0, "br_line_yellow_double", 0.4)]


def test_four_lane_primary_lines():
    r = road("primary", line(0, 0, 100, 0), 13.7, 4)  # lane_w (13.7 - 0.5) / 4 = 3.3, band 13.2
    got = by_offset(markings.lane_lines(r))
    assert [(k, m) for k, _, m, _ in got] == [
        ("edge_white", "br_line_white"), ("divider", "br_line_white_dashed"),
        ("centre_double", "br_line_yellow_double"), ("divider", "br_line_white_dashed"),
        ("edge_white", "br_line_white")]
    assert [o for _, o, _, _ in got] == pytest.approx([-6.4, -3.3, 0.0, 3.3, 6.4])
    assert got[2][3] == 0.4 and all(w < 0.4 for k, _, _, w in got if k != "centre_double")
    # secondary: no edge lines, and a 6-lane road gets two dividers a side
    six = road("secondary", line(0, 0, 100, 0), 6 * 3.3 + 0.5, 6)
    got = by_offset(markings.lane_lines(six))
    assert [k for k, *_ in got] == ["divider", "divider", "centre_double", "divider", "divider"]
    assert [o for _, o, _, _ in got] == pytest.approx([-6.6, -3.3, 0.0, 3.3, 6.6])


def test_oneway_motorway_lines():
    r = road("motorway", line(0, 0, 100, 0), 13.8, 3, oneway=True)  # lane_w (13.8 - 3) / 3 = 3.6
    got = by_offset(markings.lane_lines(r))
    assert [(k, m) for k, _, m, _ in got] == [
        ("edge_yellow", "br_line_yellow"), ("divider", "br_line_white_dashed"),
        ("divider", "br_line_white_dashed"), ("edge_white", "br_line_white")]
    assert [o for _, o, _, _ in got] == pytest.approx([-5.2, -1.8, 1.8, 5.2])
    # a one-lane ramp keeps its edges; a city one-way gets dividers only
    ramp = road("motorway_link", line(0, 0, 100, 0), 6.6, 1, oneway=True)
    assert [k for k, *_ in by_offset(markings.lane_lines(ramp))] == ["edge_yellow", "edge_white"]
    city = road("primary", line(0, 0, 100, 0), 7.1, 2, oneway=True)
    assert [(k, o) for k, o, _, _ in markings.lane_lines(city)] == [("divider", pytest.approx(0.0))]


def test_odd_lane_dividers_on_lane_boundaries():
    # an odd two-way count has a centre lane, so the lane edges sit at +-(k + 0.5) * lane_w;
    # a divider at +-k * lane_w would run down the middle of a lane
    five = road("secondary", line(0, 0, 100, 0), 5 * 3.3 + 0.5, 5)  # lane_w 3.3
    got = by_offset(markings.lane_lines(five))
    assert [o for k, o, _, _ in got if k == "divider"] == pytest.approx([-4.95, 4.95])
    seven = road("primary", line(0, 0, 100, 0), 7 * 3.3 + 0.5, 7)
    got = by_offset(markings.lane_lines(seven))
    assert [o for k, o, _, _ in got if k == "divider"] == pytest.approx([-8.25, -4.95, 4.95, 8.25])
    assert [o for k, o, _, _ in got if k == "edge_white"] == pytest.approx([-11.35, 11.35])
    three = road("tertiary", line(0, 0, 100, 0), 3 * 3.0 + 0.5, 3)
    assert [(k, o) for k, o, _, _ in by_offset(markings.lane_lines(three))] == pytest.approx(
        [("centre_turn", -1.5), ("centre_turn", 1.5)])


def test_centre_turn_lane_edges():
    # a two-way left-turn lane: solid-and-dashed yellow on both of its edges, no double yellow
    # down the middle of it
    five = road("secondary", line(0, 0, 100, 0), 5 * 3.3 + 0.5, 5)
    got = by_offset(markings.lane_lines(five))
    assert [k for k, *_ in got] == ["divider", "centre_turn", "centre_turn", "divider"]
    assert [o for _, o, _, _ in got] == pytest.approx([-4.95, -1.65, 1.65, 4.95])
    assert {m for k, _, m, _ in got if k == "centre_turn"} == {"br_line_yellow_turn"}
    # the two edges mirror each other: the left one runs against the road's node order
    marks = markings.road_lines(five, 0, [])
    left = next(m for m in marks if m.kind == "centre_turn" and m.nodes[0, 1] > 0)
    right = next(m for m in marks if m.kind == "centre_turn" and m.nodes[0, 1] < 0)
    assert left.nodes[0, 0] > left.nodes[-1, 0] and right.nodes[0, 0] < right.nodes[-1, 0]
    assert left.texture_length == right.texture_length == markings.TEX_DASHED


def test_single_lane_two_way_unpainted():
    # a one-lane two-way road has no centre line in the US
    for kind in ("tertiary", "primary"):
        assert markings.lane_lines(road(kind, line(0, 0, 100, 0), 4.0, 1)) == []


def test_residential_unmarked():
    for kind in ("residential", "service", "unclassified", "living_street"):
        assert markings.lane_lines(road(kind, line(0, 0, 100, 0), 6.5, 2)) == []
    assert set(markings.MARKED_KINDS) == {"motorway", "trunk", "primary", "secondary", "tertiary",
                                          "motorway_link", "trunk_link", "primary_link",
                                          "secondary_link", "tertiary_link"}


def test_offset_polyline_keeps_nodes():
    xy = np.array([[0, 0], [5, 0], [10, 0], [15, 0], [20, 0], [20, 5], [20, 10], [20, 15], [20, 20]], float)
    original = LineString(xy)
    for d in (3.0, -3.0):  # right of travel (the outside of this left turn), then the inside
        out = markings.offset_polyline(xy, d)
        assert out.shape == xy.shape
        dist = np.array([original.distance(Point(p)) for p in out])
        corner = 4
        assert np.delete(dist, corner) == pytest.approx(3.0, abs=1e-6)
        assert abs(dist[corner] - 3.0) < 0.5
        along = [original.project(Point(p)) for p in out]
        assert along == sorted(along)  # same order
    # positive is right of the direction of travel
    assert markings.offset_polyline(np.array([[0.0, 0.0], [10.0, 0.0]]), 2.0) == pytest.approx(
        np.array([[0.0, -2.0], [10.0, -2.0]]))
    # a repeated node does not break it
    dup = markings.offset_polyline(np.array([[0.0, 0.0], [5.0, 0.0], [5.0, 0.0], [10.0, 0.0]]), 1.0)
    assert np.isfinite(dup).all() and dup[:, 1] == pytest.approx(-1.0)


# ------------------------------------------------------------------ junctions and trims

def test_junction_trim():
    m = road("primary", line(-100, 0, 100, 0), 13.7, 4)
    cross = road("residential", line(10, -50, 10, 50, 11), 6.5, 2, osm=2)
    juncs = markings.junctions([m, cross])
    assert juncs[0] == [pytest.approx((10.0, 0.0, 6.5 / 2 + 2))]
    assert 1 not in juncs  # only marked roads get junctions
    pieces = markings.trim(m.nodes[:, :2], juncs[0])
    assert len(pieces) == 2
    assert pieces[0][0] == pytest.approx([-100, 0]) and pieces[1][-1] == pytest.approx([100, 0])
    assert pieces[0][-1] == pytest.approx([4.75, 0.0]) and pieces[1][0] == pytest.approx([15.25, 0.0])
    gap = pieces[1][0][0] - pieces[0][-1][0]
    assert gap == pytest.approx(2 * (6.5 / 2 + 2))
    # z (any extra column) is carried along
    with_z = markings.trim(m.nodes, juncs[0])
    assert with_z[0].shape[1] == 3 and with_z[0][-1] == pytest.approx([4.75, 0.0, 5.0])


def test_t_junction_and_three_ends():
    m = road("primary", line(-100, 0, 100, 0), 13.7, 4)
    stem = road("residential", line(-40, 0, -40, -80, 9), 6.5, 2, osm=2)  # starts on m
    a = road("primary", line(-100, 50, 0, 50, 21), 13.7, 4, osm=3)
    b = road("primary", line(0, 50, 100, 50, 21), 13.7, 4, osm=4)
    c = road("residential", line(0, 50, 0, 120, 8), 6.5, 2, osm=5)  # third road ending at (0, 50)
    juncs = markings.junctions([m, stem, a, b, c])
    assert juncs[0] == [pytest.approx((-40.0, 0.0, 5.25))]
    # the radius comes from the road crossing, not from a's own continuation b
    assert juncs[2] == [pytest.approx((0.0, 50.0, 6.5 / 2 + 2))]
    assert juncs[3] == [pytest.approx((0.0, 50.0, 6.5 / 2 + 2))]
    # both streets split at the crossing: four ends, each trimmed by the other street's width
    w = road("primary", line(-100, 0, 0, 0, 21), 13.7, 4)
    e = road("primary", line(0, 0, 100, 0, 21), 13.7, 4, osm=2)
    n = road("tertiary", line(0, 0, 0, 80, 9), 6.5, 2, osm=3)
    s = road("tertiary", line(0, -80, 0, 0, 9), 6.5, 2, osm=4)
    juncs = markings.junctions([w, e, n, s])
    assert [juncs[i] for i in range(4)] == [[pytest.approx((0.0, 0.0, 5.25))]] * 2 + [
        [pytest.approx((0.0, 0.0, 8.85))]] * 2
    # a narrow driveway is not a third road
    drive = road("service", line(0, 50, 0, 80, 4), 4.0, 1, osm=6)
    juncs = markings.junctions([a, b, drive])
    assert juncs.get(0, []) == [] and juncs.get(1, []) == []


def test_bridge_crossing_not_a_junction():
    m = road("primary", line(-100, 0, 100, 0), 13.7, 4)
    deck = road("motorway", line(10, -60, 10, 60, 13), 13.8, 3, oneway=True, bridge=True, z=12.0, osm=2)
    juncs = markings.junctions([m, deck])
    assert juncs.get(0, []) == [] and juncs.get(1, []) == []
    pieces = markings.trim(m.nodes[:, :2], juncs.get(0, []))
    assert len(pieces) == 1 and pieces[0] == pytest.approx(m.nodes[:, :2])
    # a ground road 7 m below a level crossing is not a junction either
    low = road("residential", line(10, -50, 10, 50, 11), 6.5, 2, z=-2.0, osm=3)
    assert markings.junctions([m, low]).get(0, []) == []


def test_gore_not_a_junction():
    # a motorway diverge: the mainline carries straight on and a ramp leaves at 15 degrees.
    # The mainline's paint must run through the gore (no 10-14 m gap at every ramp).
    a = road("motorway", line(-100, 0, 0, 0, 21), 13.8, 3, oneway=True)
    b = road("motorway", line(0, 0, 100, 0, 21), 13.8, 3, oneway=True, osm=2)
    ramp = road("motorway_link", line(0, 0, 96.6, -25.9, 21), 6.6, 1, oneway=True, osm=3)
    juncs = markings.junctions([a, b, ramp])
    assert juncs.get(0, []) == [] and juncs.get(1, []) == []


def test_continuation_not_a_junction():
    a = road("primary", line(-100, 0, 0, 0, 21), 13.7, 4)
    b = road("primary", line(0, 0, 100, 0, 21), 13.7, 4, osm=2)
    juncs = markings.junctions([a, b])
    assert juncs.get(0, []) == [] and juncs.get(1, []) == []
    assert len(markings.trim(a.nodes[:, :2], juncs.get(0, []))) == 1


def test_trim_drops_short_pieces():
    xy = line(0, 0, 100, 0, 11)
    pieces = markings.trim(xy, [(10.0, 0.0, 9.5), (50.0, 0.0, 1.0)])  # leaves 0..0.5 (too short)
    assert [round(p[0][0], 6) for p in pieces] == [19.5, 51.0]
    assert markings.trim(xy, [(50.0, 0.0, 60.0)]) == []


# ------------------------------------------------------------------ merges and overlapping carriageways

def _on(marks, prefix):
    return [m for m in marks if m.name.startswith(prefix)]


def _inside(m, road_):
    """Length of a marking's line on the road's carriageway (its centreline buffered by width / 2)."""
    carriageway = LineString(road_.nodes[:, :2]).buffer(road_.width / 2, cap_style="flat")
    return LineString(m.nodes[:, :2]).intersection(carriageway).length


def test_merging_ramp_paint_stays_off_the_mainline(small_extent):
    # a one-lane ramp joins a motorway at 11 degrees and ends on it mid-way (the motorway is not
    # split there). The motorway's paint runs through the merge, and the ramp's paint stops
    # where it would run across the motorway's lanes.
    main = road("motorway", line(-300, 0, 300, 0, 31), 13.8, 3, oneway=True)
    ramp = road("motorway_link", line(-200, -40, 0, 0, 21), 6.6, 1, oneway=True, osm=2)
    assert markings.junctions([main, ramp]).get(0, []) == []
    marks = markings.build_markings([main, ramp], [], flat(64), small_extent)
    on_main = _on(marks, "r0_")
    assert sorted(m.kind for m in on_main) == ["divider", "divider", "edge_white", "edge_yellow"]
    assert all(m.nodes[:, 0].min() == pytest.approx(-300) and m.nodes[:, 0].max() == pytest.approx(300)
               for m in on_main)
    on_ramp = _on(marks, "r1_")
    assert {m.kind for m in on_ramp} == {"edge_white", "edge_yellow"}  # its paint before the merge stays
    assert all(_inside(m, main) < 0.05 for m in on_ramp)
    # the yellow (left) edge is nearer the motorway, so it stops first
    stop = {m.kind: m.nodes[:, 0].max() for m in on_ramp}
    assert stop["edge_yellow"] < stop["edge_white"] < -20


def test_diverging_ramp_paint_stays_off_the_mainline(small_extent):
    # the motorway split at the gore (a ends, b and the ramp start there): the ramp's paint keeps
    # off b's lanes, and b's paint is not cut by the ramp (through roads win over links)
    a = road("motorway", line(-300, 0, 0, 0, 16), 13.8, 3, oneway=True)
    b = road("motorway", line(0, 0, 300, 0, 16), 13.8, 3, oneway=True, osm=2)
    ramp = road("motorway_link", line(0, 0, 200, -40, 21), 6.6, 1, oneway=True, osm=3)
    marks = markings.build_markings([a, b, ramp], [], flat(64), small_extent)
    on_ramp = _on(marks, "r2_")
    assert {m.kind for m in on_ramp} == {"edge_white", "edge_yellow"}
    assert all(_inside(m, b) < 0.05 for m in on_ramp)
    on_b = _on(marks, "r1_")
    assert len(on_b) == 4 and all(m.nodes[:, 0].max() == pytest.approx(300) for m in on_b)
    assert len({round(float(m.nodes[:, 0].min()), 3) for m in on_b}) == 1  # every line starts together


def test_converging_carriageways_keep_off_each_other(small_extent):
    # a divided trunk ends in a Y: two one-way carriageways (same class) meet a two-way trunk.
    # Where they overlap, neither paints across the other's lanes.
    east = road("trunk", line(-200, -15, 0, 0, 21), 10.2, 2, oneway=True)
    west = road("trunk", line(0, 0, -200, 15, 21), 10.2, 2, oneway=True, osm=2)
    on = road("trunk", line(0, 0, 200, 0, 21), 13.8, 2, osm=3)
    marks = markings.build_markings([east, west, on], [], flat(64), small_extent)
    assert all(_inside(m, west) < 0.05 for m in _on(marks, "r0_"))
    assert all(_inside(m, east) < 0.05 for m in _on(marks, "r1_"))
    assert {m.kind for m in _on(marks, "r0_")} == {"divider", "edge_yellow", "edge_white"}


def test_bend_between_continuations_keeps_lines_whole(small_extent):
    # two pieces of one primary meet end to end with a 20 degree bend: their carriageways
    # overlap a little on the inside of the bend, but each one's lines run right to the joint
    a = road("primary", line(-100, 0, 0, 0, 11), 13.7, 4)
    b = road("primary", line(0, 0, 100 * math.cos(0.35), 100 * math.sin(0.35), 11), 13.7, 4, osm=2)
    marks = markings.build_markings([a, b], [], flat(64), small_extent)
    on_a, on_b = _on(marks, "r0_"), _on(marks, "r1_")
    assert len(on_a) == len(on_b) == 5
    assert all(m.nodes[-1, 0] == pytest.approx(0.0, abs=1e-6) for m in on_a)
    b_line = LineString(b.nodes[:, :2])
    assert all(b_line.project(Point(m.nodes[0, :2])) == pytest.approx(0.0, abs=1e-6) for m in on_b)


def test_paint_kept_off_only_on_the_same_level(small_extent):
    # a motorway deck runs 7 m above a primary, along it: the primary's paint stays whole
    street = road("primary", line(-200, 0, 200, 0, 21), 13.7, 4)
    deck = road("motorway", line(-150, 3, 150, 3, 16), 13.8, 3, oneway=True, bridge=True, z=12.0, osm=2)
    marks = markings.build_markings([street, deck], [], flat(64), small_extent)
    on_street = _on(marks, "r0_")
    assert len(on_street) == 5 and all(m.nodes[:, 0].max() == pytest.approx(200) for m in on_street)


# ------------------------------------------------------------------ paint on bridge decks
#
# 19 of the 439 deck paint decals on the full map came out with infinite vertices in game
# ("packed position exceeded supported local range ... by inf m"); none of the ground ones
# did. Deck paint used the deck's own node stations, so every paint vertex sat on a seam of
# the deck MeshRoad. It now sits at 1/3 and 2/3 of each deck segment, and its ends are inset
# 0.5 m (Road Architect keeps its paint 0.45 m short of the ends).

def _stations(deck):
    s = markings._cum(deck.nodes)
    return s, (s[:-1] + s[1:]) / 2  # the deck's nodes and the middles of its segments


def _arc_deck(osm, stations, radius, z0, rise, kind, width, lanes, x0=0.0, y0=0.0, oneway=False):
    """A bridge deck bending left along a circle, nodes at the given distances along it,
    rising and falling over a vertical curve."""
    s = np.asarray(stations, float)
    a = s / radius
    xy = np.column_stack([x0 + radius * np.sin(a), y0 + radius * (1 - np.cos(a))])
    z = z0 + rise * np.sin(np.pi * s / s[-1])
    return road(kind, xy, width, lanes, oneway=oneway, bridge=True, z=z, osm=osm)


def test_deck_line_resamples_off_the_seams():
    # deck nodes every 10 m along x, z rising 0.1 m per m; a piece of a line 3.3 m to the left
    deck = road("primary", line(0, 0, 100, 0, 11), 13.7, 4, bridge=True, z=10 + np.linspace(0, 10, 11))
    piece = np.array([[x, 3.3, 10 + x / 10] for x in (9.6, 10.0, 20.0, 30.0, 40.0, 45.7)])
    got = markings.deck_line(piece, deck)
    # ends inset 0.5 m: 10.1 is within 0.25 m of the node at 10, so it moves on to 10.25;
    # 45.2 is within 0.25 m of the middle of 40..50, so it moves back to 44.75. In between,
    # 1/3 and 2/3 of each deck segment (13.33 is far enough from the end at 10.25).
    want_x = [10.25, 40 / 3, 50 / 3, 70 / 3, 80 / 3, 100 / 3, 110 / 3, 130 / 3, 44.75]
    assert got[:, 0] == pytest.approx(want_x, abs=1e-6)
    assert got[:, 1] == pytest.approx(3.3, abs=1e-9)
    assert got[:, 2] == pytest.approx(10 + got[:, 0] / 10, abs=1e-6)  # z along the deck
    # a line running against the deck's node order comes out the same, reversed
    back = markings.deck_line(piece[::-1], deck)
    assert back == pytest.approx(got[::-1], abs=1e-6)
    # a piece under MIN_LINE once both ends are inset is dropped
    assert markings.deck_line(np.array([[60.0, 3.3, 16.0], [62.9, 3.3, 16.29]]), deck) is None
    assert markings.deck_line(np.array([[60.0, 3.3, 16.0], [63.1, 3.3, 16.31]]), deck) is not None


def test_deck_paint_keeps_off_the_deck_seams(small_extent):
    # two curved decks with uneven node spacing (one with a centre turn lane, whose left edge
    # runs against the node order), a deck ramp ending on the first one mid-way (a junction
    # that cuts its lines), and a ground road
    main = _arc_deck(1, [0, 3, 12, 30, 48, 51, 70, 90, 110, 113.5, 130, 150], 400, 10.0, 6.0,
                     "primary", 13.7, 4, x0=-200, y0=-150)
    five = _arc_deck(2, [0, 20, 23, 41, 60, 64, 80, 100], 250, 12.0, 3.0, "secondary",
                     5 * 3.3 + 0.5, 5, x0=-200, y0=100)
    at = 70.0  # the ramp ends on main 70 m along it, at main's height there
    px, py = np.interp(at, markings._cum(main.nodes), main.nodes[:, 0]), np.interp(
        at, markings._cum(main.nodes), main.nodes[:, 1])
    pz = float(np.interp(at, markings._cum(main.nodes), main.nodes[:, 2]))
    ramp = road("tertiary", np.column_stack([np.full(7, px), py + np.linspace(-60, 0, 7)]), 6.5, 2,
                bridge=True, z=pz, osm=3)
    ground = road("primary", line(-200, -300, 200, -300, 21), 13.7, 4, osm=4)
    roads = [main, five, ramp, ground]
    juncs = markings.junctions(roads)
    assert len(juncs[0]) == 1  # the ramp cuts main's lines
    marks = markings.build_markings(roads, [], flat(64), small_extent)
    before = {m.name: m for i, r in enumerate(roads) for m in markings.road_lines(r, i, juncs.get(i, []))}
    deck_marks = [m for m in marks if m.over_objects]
    assert {m.name.split("_")[0] for m in deck_marks} == {"r0", "r1", "r2"}
    assert len({m.name for m in deck_marks if m.name.startswith("r0_")}) == 2 * 5  # two pieces, 5 lines
    for m in deck_marks:
        deck = roads[int(m.name.split("_")[0][1:])]
        centre = LineString(deck.nodes[:, :2])
        nodes_s, halves = _stations(deck)
        s = np.array([centre.project(Point(p)) for p in m.nodes[:, :2]])
        # no paint vertex on (or within 0.25 m of) a seam of the deck MeshRoad
        assert np.abs(s[:, None] - nodes_s[None, :]).min() >= 0.25 - 1e-6, m.name
        assert np.abs(s[:, None] - halves[None, :]).min() >= 0.25 - 1e-6, m.name
        # at least 0.5 m inside the ends of the piece it was cut from, along the line
        piece = LineString(before[m.name].nodes[:, :2])
        along = np.array([piece.project(Point(p)) for p in m.nodes[:, :2]])
        assert along.min() >= 0.5 - 1e-6 and along.max() <= piece.length - 0.5 + 1e-6, m.name
        assert along[0] < along[-1]  # same direction as before
        # on the line it was, z along the deck
        assert max(piece.distance(Point(p)) for p in m.nodes[:, :2]) < 0.05
        assert m.nodes[:, 2] == pytest.approx(np.interp(s, nodes_s, deck.nodes[:, 2]), abs=0.02)
        # 1/3 and 2/3 of every deck segment the line spans are among its nodes
        inner = s[1:-1]
        lo, hi = sorted((s[0], s[-1]))
        seg = np.diff(nodes_s)
        thirds = np.r_[nodes_s[:-1] + seg / 3, nodes_s[:-1] + 2 * seg / 3]
        expect = thirds[(thirds > lo + 0.25) & (thirds < hi - 0.25)]
        assert np.sort(inner) == pytest.approx(np.sort(expect), abs=0.02), m.name
    # ground paint keeps its nodes: the road's own, offset sideways
    on_ground = [m for m in marks if m.name.startswith("r3_")]
    assert len(on_ground) == 5 and not any(m.over_objects for m in on_ground)
    for m in on_ground:
        assert m.nodes[:, :2] == pytest.approx(before[m.name].nodes[:, :2])
        assert m.nodes[:, 2] == pytest.approx(5.0)


def _cfg(tmp_path, **kw):
    return Config(debug_half_size_m=512.0, terrain_size=256, raw_dir=str(tmp_path / "raw"),
                  out_dir=str(tmp_path / "out"), **kw)


def _inputs(tmp_path, small_extent, roads, nodes=()):
    out, raw = tmp_path / "out", tmp_path / "raw"
    out.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    np.save(out / "heights.npy", np.full((256, 256), 6.0, np.float32))
    save_roads(roads, out / "roads.json")
    osm_nodes = []
    for n in nodes:
        lon, lat = small_extent.to_lonlat(n["x"], n["y"])
        osm_nodes.append({"id": n["id"], "lon": lon, "lat": lat, "tags": n["tags"]})
    (raw / "osm_features.json").write_text(json.dumps({"buildings": [], "trees": [], "nodes": osm_nodes}),
                                           encoding="utf-8")


def _read(tmp_path):
    return json.loads((tmp_path / "out" / "markings.json").read_text(encoding="utf-8"))


def test_short_road_gets_no_paint(tmp_path, small_extent):
    short = road("primary", line(0, 0, 6, 0, 3), 13.7, 4)  # 6 m between two cross streets
    west = road("residential", line(0, -60, 0, 60, 13), 6.5, 2, osm=2)
    east = road("residential", line(6, -60, 6, 60, 13), 6.5, 2, osm=3)
    _inputs(tmp_path, small_extent, [short, west, east], [node("n1", 3, 1, highway="traffic_signals")])
    markings.run(_cfg(tmp_path))
    assert _read(tmp_path) == []


def test_line_clipped_to_box(extent):
    half = extent.half_size
    east = road("primary", line(half - 192, 100, half + 208, 100, 21), 13.7, 4)  # runs out of the box
    north = road("primary", line(-100, half - 2, 100, half - 2, 21), 13.7, 4, osm=2)  # along the edge
    marks = markings.build_markings([east, north], [], flat(16), extent)
    assert marks
    pts = np.vstack([m.nodes for m in marks])
    assert extent.contains(pts[:, 0], pts[:, 1]).all()
    assert pts[:, 0].max() == pytest.approx(half)
    assert {m.name.split("_")[0] for m in marks} == {"r0", "r1"}
    assert all(len(m.nodes) >= 2 and np.hypot(*np.diff(m.nodes[:, :2], axis=0).T).sum() >= 2 for m in marks)


# ------------------------------------------------------------------ stop bars and crosswalks

def _crossroads():
    p = road("primary", line(-200, 0, 200, 0), 13.7, 4)               # lane_w 3.3, band 13.2
    s = road("secondary", line(0, -200, 0, 200), 7.1, 2, osm=2)        # lane_w 3.3, band 6.6
    return [p, s]


def _span(m):
    (x0, y0, _), (x1, y1, _) = m.nodes
    return math.hypot(x1 - x0, y1 - y0)


def test_stop_bar_at_signal():
    roads = _crossroads()
    juncs = markings.junctions(roads)
    assert markings.stop_bars(roads, juncs, []) == []
    bars = markings.stop_bars(roads, juncs, [node("n1", 4, 4, highway="traffic_signals")])
    assert len(bars) == 4
    assert all(b.kind == "stop_bar" and b.width == 0.5 and b.priority == 1 and len(b.nodes) == 2
               for b in bars)
    p_bars = sorted((b for b in bars if b.name.startswith("r0_")), key=lambda b: b.nodes[0][0])
    s_bars = sorted((b for b in bars if b.name.startswith("r1_")), key=lambda b: b.nodes[0][1])
    assert [_span(b) for b in p_bars] == pytest.approx([6.6, 6.6])  # the approach band, 2 lanes
    assert [_span(b) for b in s_bars] == pytest.approx([3.3, 3.3])
    # eastbound bar on the south half, just short of the cross street's trim (7.1 / 2 + 2 m)
    west, east = p_bars
    assert west.nodes[:, 0] == pytest.approx([-5.8, -5.8])
    assert sorted(west.nodes[:, 1]) == pytest.approx([-6.6, 0.0])
    assert east.nodes[:, 0] == pytest.approx([5.8, 5.8]) and sorted(east.nodes[:, 1]) == pytest.approx([0.0, 6.6])
    south, north = s_bars
    assert south.nodes[:, 1] == pytest.approx([-9.1, -9.1]) and sorted(south.nodes[:, 0]) == pytest.approx([0, 3.3])
    assert north.nodes[:, 1] == pytest.approx([9.1, 9.1]) and sorted(north.nodes[:, 0]) == pytest.approx([-3.3, 0])
    # a one-way approach gets one bar, across the whole band
    oneway = [road("primary", line(-200, 0, 200, 0), 7.1, 2, oneway=True), _crossroads()[1]]
    bars = [b for b in markings.stop_bars(oneway, markings.junctions(oneway),
                                          [node("n1", 4, 4, highway="traffic_signals")])
            if b.name.startswith("r0_")]
    assert len(bars) == 1 and _span(bars[0]) == pytest.approx(6.6) and bars[0].nodes[0][0] < 0


def test_stop_sign_only_on_its_approach():
    p = road("primary", line(-200, 0, 200, 0), 13.7, 4)
    t = road("tertiary", line(0, 200, 0, 0), 6.5, 2, osm=2)           # ends on p, heading south
    r = road("residential", line(50, -150, 50, 0, 16), 6.5, 2, osm=3)  # ends on p
    stops = [node("n1", 0, 12, highway="stop"), node("n2", 50, -10, highway="stop")]
    bars = markings.stop_bars([p, t, r], markings.junctions([p, t, r]), stops)
    assert [b.name for b in bars] == ["r1_0_bar_end"]  # the main road does not stop
    (bar,) = bars
    assert bar.nodes[:, 1] == pytest.approx([8.85 + 0.25] * 2)
    assert sorted(bar.nodes[:, 0]) == pytest.approx([-3.0, 0.0])  # southbound lanes are on the west
    # stop=all stops the main road too
    every = markings.stop_bars([p, t, r], markings.junctions([p, t, r]),
                               [node("n3", 0, 0, highway="stop", stop="all")])
    assert len(every) == 3


def test_crosswalk_at_marked_crossing():
    p = road("primary", line(-200, 0, 200, 0), 13.7, 4)
    side = road("residential", line(-100, 0, -100, 100, 11), 6.5, 2, osm=2)
    nodes = [node("n1", 20, 0.4, highway="crossing", crossing="marked"),
             node("n2", 60, 0, highway="crossing", crossing="unmarked"),
             node("n3", 100, -0.2, highway="crossing", crossing="uncontrolled",
                  **{"crossing:markings": "zebra"}),
             node("n4", 140, 0, highway="crossing", crossing="marked", **{"crossing:markings": "no"}),
             node("n5", -50, 10, highway="crossing", crossing="marked"),            # 10 m off the road
             node("n6", -100, 30, highway="crossing", crossing="zebra"),            # on a residential street
             node("n7", -150, 0, highway="traffic_signals", crossing="traffic_signals")]
    walks = markings.crosswalks([p, side], nodes)
    assert [w.name for w in walks] == ["x_n1", "x_n3"]
    w = walks[0]
    assert w.kind == "crosswalk" and w.material == "br_crosswalk" and w.width == 3.0
    assert w.priority == 1 and w.texture_length == 3.0 and len(w.nodes) == 2
    assert w.nodes[:, 0] == pytest.approx([20, 20])               # perpendicular to the road
    assert sorted(w.nodes[:, 1]) == pytest.approx([-6.85, 6.85])  # curb to curb


# ------------------------------------------------------------------ run, files and level content

def test_markings_json_roundtrip(tmp_path):
    m = Marking("r0_0_0_centre_double", "centre_double", "br_line_yellow_double", 0.4, 2, 6.0, False,
                np.array([[0.0, 0.0, 5.0], [10.0, 0.0, 5.25]]))
    path = markings.save_markings([m], tmp_path / "out" / "markings.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw == [{"name": "r0_0_0_centre_double", "kind": "centre_double",
                    "material": "br_line_yellow_double", "width": 0.4, "priority": 2,
                    "texture_length": 6.0, "over_objects": False,
                    "nodes": [[0.0, 0.0, 5.0], [10.0, 0.0, 5.25]]}]
    (back,) = markings.load_markings(path)
    assert back.name == m.name and back.nodes.shape == (2, 3) and back.over_objects is False


def test_run_writes_markings(tmp_path, small_extent):
    roads = _crossroads() + [road("motorway", line(-300, 300, 300, 300, 31), 13.8, 3, oneway=True,
                                  bridge=True, z=14.0, osm=3)]
    _inputs(tmp_path, small_extent, roads, [node("n1", 4, 4, highway="traffic_signals"),
                                            node("n2", 60, 0, highway="crossing", crossing="zebra")])
    markings.run(_cfg(tmp_path))
    rows = _read(tmp_path)
    kinds = {r["kind"] for r in rows}
    assert {"centre_double", "divider", "edge_white", "edge_yellow", "stop_bar", "crosswalk"} <= kinds
    assert len({r["name"] for r in rows}) == len(rows)
    for r in rows:
        z = np.array(r["nodes"])[:, 2]
        assert r["over_objects"] == r["name"].startswith("r2_")
        assert z == pytest.approx(14.0 if r["over_objects"] else 6.0)
        assert r["priority"] == (2 if r["kind"] in ("centre_double", "divider", "edge_white", "edge_yellow")
                                 else 1)
        dashed = r["material"] == "br_line_white_dashed"
        assert r["texture_length"] == (3.0 if r["kind"] == "crosswalk" else 12.0 if dashed else 6.0)
    pts = np.vstack([np.array(r["nodes"]) for r in rows])
    assert small_extent.contains(pts[:, 0], pts[:, 1]).all()


def test_run_disabled_writes_empty(tmp_path, small_extent):
    _inputs(tmp_path, small_extent, _crossroads())
    cfg = _cfg(tmp_path, markings_enabled=False)
    markings.run(cfg)
    assert _read(tmp_path) == []
    assert markings.level_content(cfg, tmp_path / "out", tmp_path / "level", flat(256, 6.0),
                                  small_extent) == {}


def test_level_content_decals(tmp_path, small_extent):
    roads = _crossroads() + [road("motorway", line(-300, 300, 300, 300, 31), 13.8, 3, oneway=True,
                                  bridge=True, z=14.0, osm=3)]
    nodes = [node("n1", 4, 4, highway="traffic_signals"), node("n2", 60, 0, highway="crossing", crossing="zebra")]
    marks = markings.build_markings(roads, nodes, flat(64, 6.0), small_extent)
    data = tmp_path / "out"
    markings.save_markings(marks, data / "markings.json")
    root = tmp_path / "levels" / "x"
    content = markings.level_content(_cfg(tmp_path), data, root, flat(256, 6.0), small_extent)
    assert list(content) == ["markings"]
    objs = content["markings"]
    assert len(objs) == len(marks)
    for o, m in zip(objs, marks):
        json.dumps(o)  # plain JSON types only
        assert o["class"] == "DecalRoad" and o["persistentId"] == bf.persistent_id("mark", m.name)
        assert o["drivability"] == -1 and o["breakAngle"] == 1 and o["improvedSpline"] is True
        assert o["distanceFade"] == [200, 50] and o["startEndFade"] == [0.1, 0.1]
        assert o["renderPriority"] == (2 if m.kind in markings.LINE_KINDS else 1)
        assert o["renderPriority"] < min(bf.RENDER_PRIORITY.values())  # paint draws over every road
        assert o["textureLength"] == m.texture_length and o["material"] == m.material
        assert o["overObjects"] is m.over_objects
        assert all(len(p) == 4 and p[3] == pytest.approx(m.width) for p in o["nodes"])
        lift = markings.DECK_LIFT if m.over_objects else 0.0
        assert [p[2] for p in o["nodes"]] == pytest.approx(list(m.nodes[:, 2] + lift), abs=1e-3)
        assert o["position"] == o["nodes"][0][:3]
    assert {o["renderPriority"] for o in objs} == {1, 2}
    assert len({o["persistentId"] for o in objs}) == len(objs)
    mats = json.loads((root / "art/road/br_markings.materials.json").read_text(encoding="utf-8"))
    assert set(mats) == {m.material for m in marks}
    assert all(mats[k] == markings.MATERIALS[k] for k in mats)


def test_decals_are_named_after_their_marking():
    # the engine's decal errors name the road ("road '<name>'"), so every line carries its name
    marks = [Marking("r1_0_0_divider", "divider", "br_line_white_dashed", 0.15, 2, 12.0, False,
                     np.array([[0.0, 0.0, 5.0], [10.0, 0.0, 5.0]])),
             Marking("x_7", "crosswalk", "br_crosswalk", 3.0, 1, 3.0, False,
                     np.array([[0.0, -3.0, 5.0], [0.0, 3.0, 5.0]]))]
    assert [markings.decal_road(m)["name"] for m in marks] == ["br_mark_r1_0_0_divider", "br_mark_x_7"]


def test_materials_point_at_shared_line_textures():
    mats = markings.MATERIALS
    assert set(mats) == {"br_line_white", "br_line_yellow", "br_line_yellow_double",
                         "br_line_white_dashed", "br_crosswalk", "br_line_yellow_turn"}
    for name, m in mats.items():
        assert m["name"] == m["mapTo"] == name and m["class"] == "Material"
        assert m["persistentId"] == bf.persistent_id("material", name)
        maps = {k: v for k, v in m["Stages"][0].items() if k.endswith("Map")}
        assert maps and all(v.startswith("/assets/materials/decalroad/lines/") for v in maps.values())
        assert m["translucent"] is True and m["castShadows"] is False
    white = mats["br_line_white"]["Stages"][0]
    assert set(k for k in white if k.endswith("Map")) == {"baseColorMap", "normalMap", "opacityMap"}
    assert "/line_white/t_line_white_b.color.png" in white["baseColorMap"]
    yellow = mats["br_line_yellow"]["Stages"][0]
    assert yellow["baseColorMap"] == white["baseColorMap"]
    assert yellow["baseColorFactor"] == [0.94, 0.79, 0.27, 0.9]
    assert "/line_white_discontinue/" in mats["br_line_white_dashed"]["Stages"][0]["baseColorMap"]
    assert "t_line_crossing_white" in mats["br_crosswalk"]["Stages"][0]["baseColorMap"]


@pytest.mark.skipif(not DECALROAD_ZIP.is_file(), reason="BeamNG.drive is not installed here")
def test_line_textures_exist_in_the_game():
    with zipfile.ZipFile(DECALROAD_ZIP) as z:
        shipped = {n.lower().rsplit(".", 1)[0] for n in z.namelist()}
    refs = [(name, v) for name, m in markings.MATERIALS.items() for st in m["Stages"]
            for k, v in st.items() if k.endswith("Map")]
    missing = [(n, v) for n, v in refs if v.lstrip("/").lower().rsplit(".", 1)[0] not in shipped]
    assert refs and missing == []
