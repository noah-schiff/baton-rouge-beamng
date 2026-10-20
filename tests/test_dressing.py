"""Deck dressing (plan Task 19): concrete barriers along the bridge decks' edges and piers under them
(dressing.py, its stage, its level hook and validate.dressing_problems)."""
import itertools
import json
import logging
import math

import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, Point, box

from brmap import __main__ as cli
from brmap import beamng_format as bf
from brmap import dressing, landmarks, level, validate
from brmap.config import Config
from brmap.geo import Extent
from brmap.roads import Road, save_roads
from brmap.water import WaterBody, save_water

EXT = Extent(-91.160, 30.430, 512.0)   # the small_extent box
N = 256
FLAT10 = np.full((N, N), 10.0)
_ids = itertools.count(1)


def deck(xyz, width=12.0, oid=None, oneway=True, name=None):
    """A bridge deck through the given (x, y, z) nodes."""
    return Road([next(_ids) if oid is None else oid], "primary", np.asarray(xyz, float).reshape(-1, 3), width, 2,
                oneway, True, 1, name)


def ground_road(xy, width=8.0, z=10.0, oid=None, kind="residential"):
    xy = np.asarray(xy, float).reshape(-1, 2)
    return Road([next(_ids) if oid is None else oid], kind, np.column_stack([xy, np.full(len(xy), z)]), width, 2,
                False, False, 0, None)


def corridor(road):
    return LineString(road.nodes[:, :2]).buffer(road.width / 2)


def length(points):
    xy = np.asarray(points, float)[:, :2]
    return float(np.hypot(*np.diff(xy, axis=0).T).sum())


def small_cfg(tmp_path, **kw):
    return Config(debug_half_size_m=512.0, terrain_size=N, raw_dir=str(tmp_path / "raw"),
                  out_dir=str(tmp_path / "out"), dist_dir=str(tmp_path / "dist"), **kw)


# ------------------------------------------------------------------ constants and registration

def test_constants():
    assert dressing.BARRIER == (0.60, 0.15, 0.81)
    assert (dressing.INSET, dressing.END_TRIM) == (0.30, 2.0)
    assert (dressing.JOINT_TOL, dressing.JOINT_REACH, dressing.TAPER) == (0.5, 10.0, 5.0)
    assert math.degrees(math.acos(dressing.STRAIGHT_COS)) == pytest.approx(30.0, abs=0.01)
    assert (dressing.PIER_SPACING, dressing.PIER_MIN_GAP, dressing.PIER_THICK, dressing.PIER_ROAD_CLEAR) == (
        30.0, 2.5, 1.2, 1.0)
    assert dressing.DECK_DEPTH == bf.DECK_DEPTH
    assert (dressing.GROUP, dressing.MATERIAL, dressing.SHAPES_DIR) == ("dressing", "concrete_plain",
                                                                       "art/shapes/dressing")
    assert (dressing.FULL, dressing.LOW, dressing.COLLISION) == ("dr_a300", "dr_a60", "Colmesh-1")
    assert dressing.TILE_M == 256.0


def test_stage_and_level_hook_registered():
    assert cli.FEATURE_STAGES[-1] == "dressing"
    assert "dressing" in cli.STAGES and "dressing" in cli.ALL_ORDER
    assert level.FEATURE_MODULES[-1] == "dressing"
    assert dressing.__doc__


# ------------------------------------------------------------------ the deck's curve

def test_deck_curve_passes_through_the_nodes_and_keeps_a_straight_deck_straight():
    nodes = np.array([(0, 0, 10), (20, 0, 10), (25, 0, 10), (60, 0, 12)], float)
    curve = dressing.deck_curve(nodes)
    for p in nodes:
        assert np.min(np.linalg.norm(curve - p, axis=1)) < 1e-9
    assert np.allclose(curve[:, 1], 0.0)
    assert (np.diff(curve[:, 0]) > 0).all()


def test_deck_curve_follows_a_bend_closer_than_the_chords():
    r = 60.0                                          # a loop ramp: nodes every 20 m round a 60 m circle
    a = np.arange(0.0, math.pi, 20.0 / r)
    nodes = np.column_stack([r * np.cos(a), r * np.sin(a), np.full(len(a), 10.0)])
    curve = dressing.deck_curve(nodes)
    off = np.abs(np.hypot(curve[:, 0], curve[:, 1]) - r)
    chord_sagitta = r * (1 - math.cos(10.0 / r))      # 0.83 m
    angle = np.arctan2(curve[:, 1], curve[:, 0])
    inner = (angle > a[1] + 1e-9) & (angle < a[-2] - 1e-9)   # the end segments' tangents come from their chords
    assert off[inner].max() < 0.05 * chord_sagitta and off.max() < 0.6 * chord_sagitta
    assert len(curve) > len(nodes)


# ------------------------------------------------------------------ barriers

def test_barriers_follow_edges():
    (l, r) = dressing.barriers([deck([(0, 0, 10), (100, 0, 10)], width=13.8)], [])
    assert {round(abs(p[1]), 2) for p in l["points"] + r["points"]} == {6.6}
    assert min(p[0] for p in l["points"]) == pytest.approx(2.0) and max(p[0] for p in l["points"]) == pytest.approx(98.0)
    assert (l["side"], r["side"]) == ("l", "r")
    assert all(p[1] > 0 for p in l["points"]) and all(p[1] < 0 for p in r["points"])   # left of the travel
    assert {p[2] for p in l["points"] + r["points"]} == {10.0}


def test_barriers_on_a_narrow_deck_stand_just_outside_its_carriageway():
    # a deck narrower than NARROW_DECK (the one-lane ramps and the narrow two-lane street bridges): each
    # barrier's centre line NARROW_OUT (0.25 m) past the slab's edge, so its 0.6 m base reaches only 0.05 m
    # onto the carriageway and a 3.5 m deck keeps 3.4 m between the bases (2.3 m at width / 2 - INSET);
    # 7 m and wider decks keep width / 2 - INSET
    half_base = dressing.BARRIER[0] / 2
    for width, centre in ((3.5, 2.0), (6.99, 3.745), (7.0, 3.2), (13.8, 6.6)):
        (l, r) = dressing.barriers([deck([(0, 0, 10), (100, 0, 10)], width=width)], [])
        assert {round(abs(p[1]), 3) for p in l["points"] + r["points"]} == {centre}, width
        ends = (min(p[0] for p in l["points"]), max(p[0] for p in r["points"]))
        assert ends == pytest.approx((2.0, 98.0))
    (l, r) = dressing.barriers([deck([(0, 0, 10), (100, 0, 10)], width=3.5)], [])
    inner = min(p[1] for p in l["points"]) - max(p[1] for p in r["points"])     # centre line to centre line
    assert inner - 2 * half_base >= 3.4 - 1e-9              # inner base edge to edge: all but 5 cm a side
    (wide_l, wide_r) = dressing.barriers([deck([(0, 0, 10), (100, 0, 10)], width=13.8)], [])
    assert min(p[1] for p in wide_l["points"]) - max(p[1] for p in wide_r["points"]) == pytest.approx(13.2)
    assert (dressing.NARROW_DECK, dressing.NARROW_OUT) == (7.0, 0.25)


def test_barrier_dropped_in_merge():
    a = deck([(0, 0, 10), (200, 0, 10)], width=12)
    b = deck([(100, 3, 10), (200, 30, 10)], width=8)
    bars = dressing.barriers([a, b], [a, b])
    for bar in bars:
        other = b if bar["deck"] == dressing.deck_id(a) else a
        assert not any(corridor(other).contains(Point(p[:2])) for p in bar["points"])
    on_a = [bar for bar in bars if bar["deck"] == dressing.deck_id(a)]
    right = [bar for bar in on_a if bar["side"] == "r"]
    assert len(right) == 1 and length(right[0]["points"]) == pytest.approx(196.0)   # b joins on a's left
    left = [bar for bar in on_a if bar["side"] == "l"]
    # cut where b's corridor covers it, from its round end (x 97.0) to where b has parted (x 125.4)
    assert len(left) == 2 and sum(length(bar["points"]) for bar in left) == pytest.approx(196.0 - 28.35, abs=0.3)


def test_barriers_run_on_over_a_road_below():
    d = deck([(-150, 0, 22), (150, 0, 22)], width=12)
    under = ground_road([(0, -50), (0, 50)], width=8, z=10.0)
    bars = dressing.barriers([d], [d, under])
    assert len(bars) == 2 and all(length(bar["points"]) == pytest.approx(296.0) for bar in bars)


def of(bars, road, side=None):
    return [bar for bar in bars if bar["deck"] == dressing.deck_id(road) and side in (None, bar["side"])]


def joint_gap(bars, a, side_a, b, side_b, at):
    """The distance between the ends nearest the joint `at` of deck a's side_a barriers and deck b's
    side_b barriers."""
    def end(road, side):
        ends = [np.asarray(p, float) for bar in of(bars, road, side) for p in (bar["points"][0], bar["points"][-1])]
        return min(ends, key=lambda p: math.dist(p[:2], at))
    return float(np.linalg.norm(end(a, side_a) - end(b, side_b)))


def surface(*decks):
    """The decks' slabs as drawn: their centre lines buffered by width / 2, flat at the ends (tests grow
    them 1 mm, dressing.json's rounding)."""
    return shapely.union_all([LineString(d.nodes[:, :2]).buffer(d.width / 2, cap_style="flat") for d in decks])


def test_barriers_run_on_across_a_straight_joint():
    # a deck chained to the next at its level: no END_TRIM at the joint and no cut by the other's corridor,
    # so each side's barrier carries on across the joint; the free ends keep END_TRIM. Sides swap when the
    # next deck runs the other way
    a = deck([(0, 0, 10), (100, 0, 10)], width=13.8)
    for nodes, sides in (([(100, 0, 10), (200, 0, 10)], {"l": "l", "r": "r"}),
                         ([(200, 0, 10), (100, 0, 10)], {"l": "r", "r": "l"})):
        b = deck(nodes, width=13.8)
        bars = dressing.barriers([a, b], [a, b])
        assert len(bars) == 4
        for side in "lr":
            assert joint_gap(bars, a, side, b, sides[side], (100, 0)) <= 0.3
        xs = [p[0] for bar in bars for p in bar["points"]]
        assert (min(xs), max(xs)) == pytest.approx((2.0, 198.0))
        assert sum(length(bar["points"]) for bar in bars) == pytest.approx(2 * 196.0, abs=0.3)
        assert all(abs(abs(p[1]) - 6.6) < 1e-6 for bar in bars for p in bar["points"])


def test_barriers_meet_round_a_bend_at_a_straight_joint():
    # 20 degrees between the decks (a straight joint takes up to 30): the barrier on the inside of the bend
    # is cut back to where the two cross, the one on the outside carried over the wedge between the slabs'
    # ends; each vertex stays on a slab
    turn = math.radians(20)
    a = deck([(0, 0, 10), (100, 0, 10)], width=13.8)
    b = deck([(100, 0, 10), (100 + 100 * math.cos(turn), 100 * math.sin(turn), 10)], width=13.8)
    bars = dressing.barriers([a, b], [a, b])
    assert len(bars) == 4
    slabs = surface(a, b).buffer(1e-3)
    for side in "lr":
        assert joint_gap(bars, a, side, b, side, (100, 0)) <= 0.3
    assert all(slabs.covers(Point(p[:2])) for bar in bars for p in bar["points"])


def test_barriers_still_stop_at_an_angled_joint():
    # 45 degrees is no straight joint: each deck's barriers keep END_TRIM and stay out of the other's corridor,
    # as either deck's alone
    a = deck([(0, 0, 10), (100, 0, 10)], width=13.8)
    c = deck([(100, 0, 10), (170.7, 70.7, 10)], width=13.8)
    bars = dressing.barriers([a, c], [a, c])
    assert bars == dressing.barriers([a], [a, c]) + dressing.barriers([c], [a, c])
    for bar in bars:
        other = c if bar["deck"] == dressing.deck_id(a) else a
        assert not any(corridor(other).contains(Point(p[:2])) for p in bar["points"])
    assert max(p[0] for bar in of(bars, a) for p in bar["points"]) <= 98.0 + 1e-6


def test_barriers_still_stop_at_a_ramp_merge():
    # a ramp forking off a straight joint at 8 degrees: the two decks carrying straight on join their
    # barriers; the ramp keeps END_TRIM and the cut by both, and theirs stay out of the ramp's corridor
    a = deck([(0, 0, 10), (100, 0, 10)], width=13.8)
    b = deck([(100, 0, 10), (200, 0, 10)], width=13.8)
    ramp = deck([(100, 0, 10), (200, 100 * math.tan(math.radians(8)), 10)], width=6.6)
    bars = dressing.barriers([a, b, ramp], [a, b, ramp])
    assert of(bars, ramp) == dressing.barriers([ramp], [a, b, ramp])
    for bar in of(bars, ramp):
        assert not any(corridor(a).contains(Point(p[:2])) or corridor(b).contains(Point(p[:2]))
                       for p in bar["points"])
    for bar in of(bars, a) + of(bars, b):
        assert not any(corridor(ramp).contains(Point(p[:2])) for p in bar["points"])
    for side in "lr":
        assert joint_gap(bars, a, side, b, side, (100, 0)) <= 0.3


def fork(right_deg, left_deg, reverse=False):
    """A 13.8 m deck forking at (100, 0) into two 300 m 10.2 m ones, right_deg to the right and left_deg to
    the left (negative: the other way), as decks running into the fork when reversed (a merge)."""
    def leg(deg):
        t = math.radians(deg)
        return [(100, 0, 10), (100 + 300 * math.cos(t), 300 * math.sin(t), 10)]
    a, b, c = [(0, 0, 10), (100, 0, 10)], leg(-right_deg), leg(left_deg)
    return [deck(n[::-1] if reverse else n, width=w) for n, w in ((a, 13.8), (b, 10.2), (c, 10.2))]


def test_straight_joints_join_each_side_to_its_outermost_deck():
    # I-10 at (-4454.8, 1053.2): a 13.8 m deck forks into two 10.2 m ones within 30 degrees of it. Each side
    # joins the deck whose barrier is outermost there (JOINT_REACH on), so the trunk's left runs on onto the
    # left fork's left and its right onto the right fork's right; the forks' sides facing each other join
    # nothing. Reversed (a merge), the sides swap with the node order
    for reverse in (False, True):
        a, b, c = fork(1.3, 2.7, reverse)
        got = dressing.straight_joints([a, b, c])
        e, f = (0, -1) if reverse else (-1, 0)
        r, l = ("l", "r") if reverse else ("r", "l")
        want = {(0, e, r): (1, f, r), (0, e, l): (2, f, l)}
        assert got == {**want, **{v: k for k, v in want.items()}}
    # a 6.6 m ramp forking off 8 degrees, beside a 13.8 m deck carrying straight on: that one is outermost on
    # both sides 10 m on, so it joins both
    a = deck([(0, 0, 10), (100, 0, 10)], width=13.8)
    b = deck([(100, 0, 10), (200, 0, 10)], width=13.8)
    ramp = deck([(100, 0, 10), (200, 100 * math.tan(math.radians(8)), 10)], width=6.6)
    assert dressing.straight_joints([a, b, ramp]) == {(0, -1, s): (1, 0, s) for s in "lr"} | {
        (1, 0, s): (0, -1, s) for s in "lr"}


@pytest.mark.parametrize("right_deg, left_deg, reverse", [(1.3, 2.7, False), (1.3, 2.7, True), (25.4, 19.3, False)])
def test_barriers_run_on_round_both_outer_edges_of_a_fork(right_deg, left_deg, reverse):
    # the I-10 fork (1.3 and 2.7 degrees), the same as a merge, and the Y split at (-1812.6, 1375.5) (25.4 and
    # 19.3): both outer edges are continuous, the trunk's barriers meeting each fork's outer one, on the slabs;
    # the forks' barriers facing each other (the gore) keep END_TRIM and stay out of the other decks' corridors,
    # as at any ramp merge, and dressing_problems is clean
    a, b, c = roads = fork(right_deg, left_deg, reverse)
    bars = dressing.barriers(roads, roads)
    r, l = ("l", "r") if reverse else ("r", "l")
    assert joint_gap(bars, a, r, b, r, (100, 0)) <= 0.3
    assert joint_gap(bars, a, l, c, l, (100, 0)) <= 0.3
    slabs = surface(*roads).buffer(1e-3)
    outer = of(bars, a) + of(bars, b, r) + of(bars, c, l)
    assert all(slabs.covers(Point(p[:2])) for bar in outer for p in bar["points"])
    gore = of(bars, b, l) + of(bars, c, r)
    assert gore                                                    # the forks part well within 300 m
    for bar in gore:
        others = [d for d in roads if dressing.deck_id(d) != bar["deck"]]
        assert not any(corridor(d).contains(Point(p[:2])) for d in others for p in bar["points"])
        assert min(math.dist(p[:2], (100, 0)) for p in bar["points"]) >= dressing.END_TRIM
    assert validate.dressing_problems({"barriers": bars, "piers": []}, roads) == []


def test_dressing_problems_still_name_a_gore_barrier_in_the_other_fork_near_the_joint():
    # a fork's barrier on the side facing the other fork joins nothing, so that fork's corridor counts there
    a, b, c = roads = fork(1.3, 2.7)
    gore = [{"deck": dressing.deck_id(b), "side": "l", "points": [[106, 4.0, 10], [110, 4.0, 10]]}]
    (problem,) = validate.dressing_problems({"barriers": gore, "piers": []}, roads)
    assert f"corridor of {dressing.deck_id(c)} (" in problem and "2 vertices" in problem


@pytest.mark.parametrize("widths", [(13.8, 17.4), (17.4, 13.8)])
def test_barriers_taper_across_a_straight_joint_to_a_narrower_deck(widths):
    # the wider deck's barriers taper in to the narrower one's line before the joint, on its own slab
    a = deck([(0, 0, 10), (100, 0, 10)], width=widths[0])
    b = deck([(100, 0, 10), (200, 0, 10)], width=widths[1])
    bars = dressing.barriers([a, b], [a, b])
    assert len(bars) == 4
    slabs = surface(a, b).buffer(1e-3)
    assert all(slabs.covers(Point(p[:2])) for bar in bars for p in bar["points"])
    for side in "lr":
        assert joint_gap(bars, a, side, b, side, (100, 0)) <= 0.3
    wide = a if widths[0] > widths[1] else b
    across = {round(abs(p[1]), 3) for bar in of(bars, wide) for p in bar["points"]}
    assert min(across) == pytest.approx(6.6) and max(across) == pytest.approx(8.4)


def test_deck_id_names_every_way():
    assert dressing.deck_id(Road([12, 34], "primary", np.zeros((2, 3)), 8.0, 2, True, True, 1, None)) == "w12+w34"


# ------------------------------------------------------------------ piers

def test_piers_spacing_and_clearance(small_extent):
    d = deck([(-150, 0, 22), (150, 0, 22)], width=12)               # 12 m over flat ground at 10 m
    under = ground_road([(0, -50), (0, 50)], width=8)
    ps = dressing.piers([d], [d, under], FLAT10, small_extent, [])
    xs = sorted(p["x"] for p in ps)
    assert all(abs(x) > 4 + 1.0 + 0.6 for x in xs) and np.allclose(np.diff(xs)[np.diff(xs) < 40], 30, atol=0.5)
    assert len(xs) == 8                                  # every 30 m from the deck's start, less the one on the road


def test_pier_fields(small_extent):
    d = deck([(-150, 0, 22), (150, 0, 22)], width=12)
    (p, *_) = dressing.piers([d], [d], FLAT10, small_extent, [])
    assert set(p) == {"deck", "x", "y", "yaw", "width", "z_top", "z_bottom"}
    assert p["deck"] == dressing.deck_id(d) and p["y"] == pytest.approx(0.0)
    assert p["width"] == pytest.approx(6.0)                          # half the deck
    assert p["z_top"] == pytest.approx(22.0 - bf.DECK_DEPTH)        # the deck's underside
    assert p["z_bottom"] < 10.0                                      # into the ground
    assert math.cos(p["yaw"]) == pytest.approx(1.0)                  # along the deck
    wide = deck([(-150, 0, 22), (150, 0, 22)], width=20)
    assert {q["width"] for q in dressing.piers([wide], [wide], FLAT10, small_extent, [])} == {8.0}


def test_pier_footprint_keeps_clear_of_a_road(small_extent):
    # the pier at x = 0 is 1.2 m thick along the deck: a road whose corridor + 1 m reaches x = 0 takes it
    d = deck([(-150, 0, 22), (150, 0, 22)], width=12)
    for x, kept in ((5.0, False), (5.7, True)):
        road = ground_road([(x, -50), (x, 50)], width=8)
        ps = dressing.piers([d], [d, road], FLAT10, small_extent, [])
        assert any(abs(p["x"]) < 1e-6 for p in ps) == kept, x


def test_piers_stand_under_a_deck_passing_over(small_extent):
    d = deck([(-150, 0, 22), (150, 0, 22)], width=12)
    over = deck([(0, -100, 40), (0, 100, 40)], width=12)
    ps = dressing.piers([d], [d, over], FLAT10, small_extent, [])
    assert any(abs(p["x"]) < 1e-6 for p in ps)


def test_no_piers_when_low_or_under_a_span(small_extent):
    assert dressing.piers([deck([(-50, 0, 11.5), (50, 0, 11.5)], width=12)], [], FLAT10, small_extent, []) == []
    assert dressing.piers([deck([(-50, 0, 30), (50, 0, 30)], width=12)], [], FLAT10, small_extent,
                          [box(-100, -100, 100, 100)]) == []


def test_piers_keep_clear_of_a_span_and_its_end_piers(small_extent):
    # a landmark over the decks (the Wilkinson truss) brings its own piers, its end piers reaching SPAN_PAD
    # past its footprint: a pier (PIER_THICK along the deck) coming that close is left out
    span = box(-100, -20, 40, 20)
    for x0, kept in ((12.5, False), (12.7, True)):     # the first pier at x0 + 30: 1.9 and 2.1 m off the span
        d = deck([(x0, 0, 22), (x0 + 300, 0, 22)], width=12)
        ps = dressing.piers([d], [d], FLAT10, small_extent, [span])
        assert any(abs(p["x"] - (x0 + 30)) < 1e-6 for p in ps) == kept, x0
        assert len(ps) == 9 - (not kept)
    assert dressing.SPAN_PAD == 2.0


# ------------------------------------------------------------------ the checks

def test_dressing_problems_clean_on_what_the_stage_makes(small_extent):
    a = deck([(-200, 0, 22), (200, 0, 22)], width=12)
    b = deck([(0, 3, 22), (200, 40, 22)], width=8)
    under = ground_road([(50, -80), (50, 80)], width=8)
    roads = [a, b, under]
    data = {"barriers": dressing.barriers([a, b], roads),
            "piers": dressing.piers([a, b], roads, FLAT10, small_extent, [])}
    assert data["barriers"] and data["piers"]
    assert validate.dressing_problems(data, roads) == []


def test_dressing_problems_names_a_barrier_on_another_road_and_a_pier_on_a_road():
    a = deck([(-200, 0, 22), (200, 0, 22)], width=12)
    b = deck([(0, 3, 22), (200, 40, 22)], width=8)
    under = ground_road([(50, -80), (50, 80)], width=8)
    bad = {"barriers": [{"deck": dressing.deck_id(a), "side": "l", "points": [[10, 5.7, 22], [40, 5.7, 22]]}],
           "piers": [{"deck": dressing.deck_id(a), "x": 50.0, "y": 0.0, "yaw": 0.0, "width": 6.0, "z_top": 20.5,
                      "z_bottom": 9.5}]}
    problems = validate.dressing_problems(bad, [a, b, under])
    assert len(problems) == 2
    assert dressing.deck_id(a) in problems[0] and dressing.deck_id(b) in problems[0]
    assert "pier" in problems[1] and dressing.deck_id(under) in problems[1]


def test_dressing_problems_let_barriers_run_on_across_a_straight_joint():
    # the joined barriers reach into the next deck's corridor near the joint: not a problem there (within
    # JOINT_REACH of it, and only against that deck), so what the stage makes is clean
    turn = math.radians(20)
    a = deck([(0, 0, 10), (100, 0, 10)], width=13.8)
    for b in (deck([(100, 0, 10), (200, 0, 10)], width=13.8), deck([(200, 0, 10), (100, 0, 10)], width=17.4),
              deck([(100, 0, 10), (100 + 100 * math.cos(turn), 100 * math.sin(turn), 10)], width=13.8)):
        data = {"barriers": dressing.barriers([a, b], [a, b]), "piers": []}
        assert validate.dressing_problems(data, [a, b]) == []


def test_dressing_problems_still_name_a_barrier_in_a_crossing_deck_near_a_straight_joint():
    # beside the joint a deck crossing at its level still counts, and so does the joined deck further on
    a = deck([(0, 0, 10), (100, 0, 10)], width=13.8)
    b = deck([(100, 0, 10), (200, 0, 10)], width=13.8)
    c = deck([(95, -50, 10), (95, 50, 10)], width=8.0)
    near = {"barriers": [{"deck": dressing.deck_id(a), "side": "l", "points": [[91, 6.6, 10], [95, 6.6, 10],
                                                                               [99.5, 6.6, 10]]}], "piers": []}
    (problem,) = validate.dressing_problems(near, [a, b, c])
    assert f"corridor of {dressing.deck_id(c)} (" in problem and "1 vertices" in problem
    far = {"barriers": [{"deck": dressing.deck_id(a), "side": "l", "points": [[80, 6.0, 10], [115, 6.0, 10]]}],
           "piers": []}
    (problem,) = validate.dressing_problems(far, [a, b])
    assert f"corridor of {dressing.deck_id(b)} (" in problem and "1 vertices" in problem


def test_feature_problems_check_the_dressing(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    a = deck([(-200, 0, 22), (200, 0, 22)], width=12)
    under = ground_road([(50, -80), (50, 80)], width=8)
    (out / "dressing.json").write_text(json.dumps({"barriers": [], "piers": [
        {"deck": dressing.deck_id(a), "x": 50.0, "y": 0.0, "yaw": 0.0, "width": 6.0, "z_top": 20.5,
         "z_bottom": 9.5}]}), encoding="utf-8")
    got = validate.feature_problems(small_cfg(tmp_path), out, EXT, FLAT10, [a, under])
    assert len(got) == 1 and "pier" in got[0]


# ------------------------------------------------------------------ the stage

def stage_inputs(tmp_path, roads, river=None):
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    (tmp_path / "raw").mkdir(exist_ok=True)
    np.save(out / "heights.npy", FLAT10.astype(np.float32))
    save_roads(roads, out / "roads.json")
    bodies = [WaterBody("Mississippi River", river, 5.0, 15.0)] if river is not None else []
    save_water(bodies, out / "water.json")
    return out


def test_run_writes_dressing_json(tmp_path):
    d = deck([(-150, 0, 22), (150, 0, 22)], width=12)
    out = stage_inputs(tmp_path, [d, ground_road([(0, -50), (0, 50)])])
    dressing.run(small_cfg(tmp_path))
    data = json.loads((out / "dressing.json").read_text(encoding="utf-8"))
    assert set(data) == {"barriers", "piers"}
    assert len(data["barriers"]) == 2 and len(data["piers"]) == 8
    assert set(data["barriers"][0]) == {"deck", "side", "points"}
    assert dressing.load_dressing(out / "dressing.json") == data


def test_run_puts_piers_in_the_river_clear_of_the_truss(tmp_path):
    # the truss (a landmark over the decks) spans the west half of the river: the deck's east half over the
    # water gets piers, the stretch under the truss and SPAN_PAD past it none
    d = deck([(-150, 0, 22), (150, 0, 22)], width=12)
    river = box(-130, -100, 130, 100)
    out = stage_inputs(tmp_path, [d], river=river)
    truss = {"name": "t", "footprint": [[-130, -20], [0, -20], [0, 20], [-130, 20]], "params": {"decks": ["w1"]}}
    (out / "landmarks.json").write_text(json.dumps([truss]), encoding="utf-8")
    dressing.run(small_cfg(tmp_path))
    ps = dressing.load_dressing(out / "dressing.json")["piers"]
    assert sorted(p["x"] for p in ps) == [30.0, 60.0, 90.0, 120.0]
    assert all(river.contains(Point(p["x"], p["y"])) for p in ps)          # over the water


def test_run_leaves_a_truss_span_to_the_truss(tmp_path):
    # a landmark over the decks (params.decks) brings its own piers, on the bank as over the river
    d = deck([(-150, 0, 22), (150, 0, 22)], width=12)
    out = stage_inputs(tmp_path, [d])
    truss = {"name": "t", "footprint": [[-100, -20], [40, -20], [40, 20], [-100, 20]], "params": {"decks": ["w1"]}}
    other = {"name": "o", "footprint": [[50, -20], [140, -20], [140, 20], [50, 20]], "params": {}}
    text = json.dumps([truss, other])
    (out / "landmarks.json").write_text(text, encoding="utf-8")
    dressing.run(small_cfg(tmp_path))
    xs = sorted(p["x"] for p in dressing.load_dressing(out / "dressing.json")["piers"])
    assert xs == [-120.0, 60.0, 90.0, 120.0]
    assert (out / "landmarks.json").read_text(encoding="utf-8") == text          # read, never written


def test_run_warns_when_landmarks_json_lacks_a_landmark_over_its_decks(tmp_path, monkeypatch, caplog):
    # a registry landmark over decks (the Wilkinson truss) whose deck is in roads.json but not in landmarks.json
    # (the landmarks stage not re-run since): piers would stand in the river under it, so the stage says so; a
    # landmark over decks this map lacks (another box's) is no news
    spec = "  model: m.py\n  anchor: {lat: 30.43, lon: -91.16}\n  yaw: {azimuth_deg: 90}\n  footprint: {ring: []}\n"
    registry = tmp_path / "landmarks.yaml"
    registry.write_text(f"my_truss:\n{spec}  params: {{decks: [w9001]}}\n"
                        f"far_truss:\n{spec}  params: {{decks: [w9002]}}\n"
                        f"tower:\n{spec}  params: {{}}\n", encoding="utf-8")
    monkeypatch.setattr(landmarks, "REGISTRY", registry)
    d = deck([(-150, 0, 22), (150, 0, 22)], width=12, oid=9001)
    out = stage_inputs(tmp_path, [d])
    tower = {"name": "tower", "footprint": [[200, 200], [210, 200], [210, 210]], "params": {}}
    truss = {"name": "my_truss", "footprint": [[-130, -20], [0, -20], [0, 20], [-130, 20]],
             "params": {"decks": ["w9001"]}}
    for placed, warned in (([tower], True), ([tower, truss], False)):
        (out / "landmarks.json").write_text(json.dumps(placed), encoding="utf-8")
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="brmap.dressing"):
            dressing.run(small_cfg(tmp_path))
        warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == int(warned), warnings
        if warned:
            assert "my_truss" in warnings[0] and "far_truss" not in warnings[0] and "w9001" in warnings[0]
            assert "landmarks" in warnings[0]


def test_run_writes_empty_lists_when_dressing_is_off(tmp_path):
    out = stage_inputs(tmp_path, [deck([(-150, 0, 22), (150, 0, 22)])])
    dressing.run(small_cfg(tmp_path, deck_dressing=False))
    assert dressing.load_dressing(out / "dressing.json") == {"barriers": [], "piers": []}


# ------------------------------------------------------------------ the level

def network(n=4):
    """About 3 km of decks: a long curving viaduct, a ramp merging into it and short overpasses."""
    a = np.linspace(-0.6 * math.pi, 0.6 * math.pi, 60)
    loop = deck(np.column_stack([400 * np.sin(a), 300 * np.cos(a) - 200, 18 + 4 * np.cos(a)]), width=13.8)
    ramp = deck([(-350, -380, 12), (-200, -330, 16), (-60, -310, 18.5)], width=6.6)
    bridges = [deck([(x, -460, 17), (x, -140, 17)], width=8) for x in np.linspace(-300, 300, n)]
    return [loop, ramp, *bridges]


def dressed(tmp_path, roads):
    out = stage_inputs(tmp_path, roads)
    cfg = small_cfg(tmp_path)
    dressing.run(cfg)
    root = tmp_path / "level"
    return out, root, dressing.level_content(cfg, out, root, FLAT10, EXT)


def test_dressing_tiles_under_the_cap(tmp_path):
    out, root, content = dressed(tmp_path, network())
    data = dressing.load_dressing(out / "dressing.json")
    assert sum(length(b["points"]) for b in data["barriers"]) > 3000.0 and data["piers"]
    shapes = sorted((root / dressing.SHAPES_DIR).glob("*.dae"))
    (objs,) = content.values()
    assert list(content) == [dressing.GROUP] and len(objs) == len(shapes) > 1     # one TSStatic per tile
    assert {o["shapeName"].rsplit("/", 1)[1] for o in objs} == {s.name for s in shapes}
    for s in shapes:
        counts = validate.dae_vertex_counts(s)
        assert counts and max(counts.values()) <= validate.DAE_MAX_VERTS, s.name
        assert set(counts) <= {dressing.FULL, dressing.LOW, dressing.COLLISION}


def test_dressing_tiles_split_over_the_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(dressing, "MAX_VERTS", 600)
    _, root, content = dressed(tmp_path, network())
    shapes = sorted((root / dressing.SHAPES_DIR).glob("*.dae"))
    assert any("_" in s.stem[len("br_dr_"):] for s in shapes)                   # quartered tiles
    for s in shapes:
        assert max(validate.dae_vertex_counts(s).values()) <= 600, s.name
    assert len(content[dressing.GROUP]) == len(shapes)


def test_level_content_places_tiles_with_the_stock_concrete(tmp_path):
    d = deck([(-150, 0, 22), (150, 0, 22)], width=12)
    _, root, content = dressed(tmp_path, [d])
    objs = content[dressing.GROUP]
    for o in objs:
        assert o["class"] == "TSStatic" and o["position"][2] == 0.0 and "rotationMatrix" not in o
        assert o["collisionType"] == "Collision Mesh" and o["useInstanceRenderData"] is True
        assert o["shapeName"].startswith(f"/levels/{small_cfg(tmp_path).level_name}/{dressing.SHAPES_DIR}/br_dr_")
    assert len({o["persistentId"] for o in objs}) == len(objs)
    text = "".join(s.read_text(encoding="utf-8") for s in (root / dressing.SHAPES_DIR).glob("*.dae"))
    assert 'material="concrete_plain-material"' in text and "br_" not in text.split("<library_geometries>")[0]
    assert not list((root / dressing.SHAPES_DIR).glob("*.json"))                 # a stock global material


def test_level_content_far_level_keeps_only_the_piers(tmp_path):
    d = deck([(-150, 0, 22), (150, 0, 22)], width=12)
    out = stage_inputs(tmp_path, [d])
    cfg = small_cfg(tmp_path)
    dressing.run(cfg)
    data = dressing.load_dressing(out / "dressing.json")
    tiles = dressing.tiles(data, dressing.TILE_M)
    for key, origin, meshes in tiles:
        full, low, col = (meshes[n] for n in (dressing.FULL, dressing.LOW, dressing.COLLISION))
        assert col is not None and full is not None
        if low is not None:
            assert len(low.pos) < len(full.pos)
    piers_only = sum(len(m[dressing.LOW].pos) for _, _, m in tiles if m[dressing.LOW] is not None)
    assert piers_only == 16 * len(data["piers"])         # a box's four sides, four vertices each


def test_level_content_is_empty_when_off_or_bare(tmp_path):
    out = stage_inputs(tmp_path, [ground_road([(0, -50), (0, 50)])])
    cfg = small_cfg(tmp_path)
    dressing.run(cfg)
    assert dressing.level_content(cfg, out, tmp_path / "level", FLAT10, EXT) == {}
    out = stage_inputs(tmp_path, [deck([(-150, 0, 22), (150, 0, 22)])])
    dressing.run(cfg)
    assert dressing.level_content(small_cfg(tmp_path, deck_dressing=False), out, tmp_path / "level", FLAT10, EXT) == {}


def test_barrier_mesh_is_a_jersey_profile_facing_out(tmp_path):
    d = deck([(10, 100, 22), (210, 100, 22)], width=12)           # all of it in one 1024 m tile
    data = {"barriers": dressing.barriers([d], [d]), "piers": []}
    (key, origin, meshes), = dressing.tiles(data, 1024.0)
    full = meshes[dressing.FULL]
    pos = full.pos + [*origin, 0.0]
    across = np.abs(pos[:, 1] - 100.0)
    top = np.isclose(pos[:, 2], 22.0 + 0.81)
    assert pos[:, 2].max() == pytest.approx(22.0 + 0.81)
    assert across[top].min() == pytest.approx(5.7 - 0.075) and across[top].max() == pytest.approx(5.7 + 0.075)
    base = np.abs(pos[:, 2] - 22.0) < 0.2
    assert across[base].min() == pytest.approx(5.7 - 0.30) and across[base].max() == pytest.approx(5.7 + 0.30)
    (tris,) = full.tris.values()
    a, b, c = (full.pos[tris[:, k]] for k in range(3))
    face = np.cross(b - a, c - a)
    assert (np.einsum("ij,ij->i", face, full.nrm[tris[:, 0]]) > 0).all()   # counter-clockwise from outside
