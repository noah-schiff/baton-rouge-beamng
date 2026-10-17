import math

import numpy as np
import pytest

from brmap.bridges import (MARGIN, MAX_GRADE, TERRAIN_GAP, apply_bridges, clear_under_decks,
                           deck_profile, merge_bridges)
from brmap.config import Config
from brmap.roads import Road
from brmap.terrain import sample_height


def grid(extent, fn, n=256):
    xs, ys = extent.pixel_centers(n)
    X, Y = np.meshgrid(xs, ys)
    return np.asarray(fn(X, Y), dtype=np.float32) * np.ones((n, n), np.float32)


def line(x0, x1, y=0.0, n=21):
    xs = np.linspace(x0, x1, n)
    return np.column_stack([xs, np.full(n, y)])


def bridge(xy, name="X", kind="primary", oid=1, layer=1, oneway=False):
    nodes = np.column_stack([xy, np.full(len(xy), np.nan)])
    return Road([oid], kind, nodes, 8.0, 2, oneway, True, layer, name)


def ground_road(xy, heights, extent, oid=99):
    z = sample_height(heights, extent, xy[:, 0], xy[:, 1])
    return Road([oid], "residential", np.column_stack([xy, z]), 6.5, 2, False, False, 0, "Under St")


def lin_of(z_start, z_end, xy):
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    return z_start + (z_end - z_start) * d / d[-1]


def test_deck_ends_meet_terrain(small_extent):
    h = grid(small_extent, lambda X, Y: 5 + X / 100)
    xy = line(-200, 200)
    z = deck_profile(xy, h, small_extent, clearance=5.5)
    assert z[0] == pytest.approx(sample_height(h, small_extent, -200, 0), abs=0.01)
    assert z[-1] == pytest.approx(sample_height(h, small_extent, 200, 0), abs=0.01)


def test_deck_stays_above_terrain_bump(small_extent):
    h = grid(small_extent, lambda X, Y: np.where(np.abs(X) < 30, 8.0, 5.0))
    xy = line(-200, 200, n=81)
    z = deck_profile(xy, h, small_extent, clearance=5.5)
    t = np.linspace(0, 1, 81)
    ground = sample_height(h, small_extent, xy[:, 0], xy[:, 1])
    mid = (t >= 0.1) & (t <= 0.9)
    assert (z[mid] >= ground[mid] + MARGIN - 0.01).all()


def test_deck_clears_crossing_road(small_extent):
    h = grid(small_extent, lambda X, Y: 5.0 + 0 * X)
    xy = line(-200, 200)
    z = deck_profile(xy, h, small_extent, clearance=5.5, obstacles=[(0.5, 5.0)])
    assert z[10] >= 5.0 + 5.5 - 0.01  # node 10 of 21 is mid-span


def test_flat_deck_when_no_obstacle(small_extent):
    h = grid(small_extent, lambda X, Y: np.where(np.abs(X) > 190, 10.0, 2.0))  # a valley
    xy = line(-200, 200)
    z = deck_profile(xy, h, small_extent, clearance=5.5)
    np.testing.assert_allclose(z, 10.0, atol=0.01)


def test_peak_override_used(small_extent):
    h = grid(small_extent, lambda X, Y: 3 + X / 200)
    xy = line(-400, 400)
    z = deck_profile(xy, h, small_extent, clearance=5.5, peak_override=15.0)
    ground = sample_height(h, small_extent, xy[:, 0], xy[:, 1])
    assert z[10] - lin_of(ground[0], ground[-1], xy)[10] == pytest.approx(15.0, abs=0.05)
    slopes = np.abs(np.diff(z)) / np.hypot(*np.diff(xy, axis=0).T)
    assert slopes.max() <= MAX_GRADE + 1e-6


def test_peak_override_out_of_reach_is_capped_by_max_grade(small_extent):
    h = grid(small_extent, lambda X, Y: 5.0 + 0 * X)
    xy = line(-200, 200)  # 52 m up and down again in 400 m would be a 26 % climb
    z = deck_profile(xy, h, small_extent, clearance=5.5, peak_override=52.0)
    slopes = np.abs(np.diff(z)) / np.hypot(*np.diff(xy, axis=0).T)
    assert slopes.max() <= MAX_GRADE + 1e-6
    assert 5.0 + 10.0 < z.max() < 5.0 + 52.0


def test_hump_capped_by_max_grade(small_extent):
    h = grid(small_extent, lambda X, Y: 5.0 + 0 * X)
    xy = line(-30, 30)  # a 60 m span cannot rise 5.5 m without a steep hump
    z = deck_profile(xy, h, small_extent, clearance=5.5, obstacles=[(0.5, 5.0)])
    slopes = np.abs(np.diff(z)) / np.hypot(*np.diff(xy, axis=0).T)
    assert slopes.max() <= MAX_GRADE + 1e-6


def test_long_viaduct_is_a_plateau_not_a_hump(small_extent):
    h = grid(small_extent, lambda X, Y: 5.0 + 0 * X)
    xy = line(-450, 450, n=91)  # 900 m elevated road, 10 m between nodes
    z = deck_profile(xy, h, small_extent, clearance=5.5,
                     obstacles=[(0.2, 5.0), (0.5, 5.0), (0.8, 5.0)])
    for node in (18, 45, 72):  # the three crossings
        assert z[node] >= 10.5 - 0.01
    assert z.max() <= 10.5 + 0.05  # flat on top instead of a tall parabola
    slopes = np.abs(np.diff(z)) / np.hypot(*np.diff(xy, axis=0).T)
    assert slopes.max() <= MAX_GRADE + 1e-6


def test_crossing_near_an_end_does_not_lift_the_whole_deck(small_extent):
    h = grid(small_extent, lambda X, Y: 5.0 + 0 * X)
    xy = line(-350, 350, n=71)
    z = deck_profile(xy, h, small_extent, clearance=5.5, obstacles=[(0.01, 5.0), (0.5, 5.0)])
    assert z[35] >= 10.5 - 0.01   # the mid-span crossing is still cleared
    assert z.max() <= 10.5 + 0.05  # the unreachable one 7 m from the end is given up


def test_deck_leaves_the_ground_gently(small_extent):
    h = grid(small_extent, lambda X, Y: 5.0 + 0 * X)
    xy = line(-200, 200)
    z = deck_profile(xy, h, small_extent, clearance=5.5, obstacles=[(0.5, 5.0)])
    assert (z[1] - z[0]) / 20.0 < 0.02  # ramps start tangent to the approach road


def test_clear_under_decks_cuts_terrain_poking_through(small_extent):
    h = grid(small_extent, lambda X, Y: np.where(np.abs(X) < 40, 9.0, 5.0))  # a 9 m mound
    deck = bridge(line(-200, 200, n=41))
    deck.nodes[:, 2] = 6.0  # flat deck running into the mound
    out, cuts = clear_under_decks(h, small_extent, [deck])
    assert sample_height(out, small_extent, 0.0, 0.0) <= 6.0 - TERRAIN_GAP + 0.01
    assert sample_height(out, small_extent, 0.0, 60.0) == pytest.approx(9.0)  # beside the deck
    assert sample_height(out, small_extent, -100.0, 0.0) == pytest.approx(5.0)  # under, but clear
    assert cuts == {"X": pytest.approx(9.0 - (6.0 - TERRAIN_GAP), abs=0.01)}
    assert h.max() == 9.0  # the input is not modified


def test_clear_under_decks_leaves_banks_beside_the_deck(small_extent):
    h = grid(small_extent, lambda X, Y: np.where((Y > 5) & (Y < 11), 6.5, 5.0))  # bank 2 m off the edge
    deck = bridge(line(-200, 200, n=41))  # 8 m wide, so its edge is at y = 4
    deck.nodes[:, 2] = 5.5
    out, cuts = clear_under_decks(h, small_extent, [deck])
    assert cuts == {} and np.abs(out - h).max() < 1e-6


def test_clear_under_decks_leaves_good_decks_and_ground_roads(small_extent):
    h = grid(small_extent, lambda X, Y: 5.0 + X / 400)
    xy = line(-200, 200)
    deck = bridge(xy)
    deck.nodes[:, 2] = deck_profile(xy, h, small_extent, clearance=5.5, obstacles=[(0.5, 5.0)])
    road = ground_road(np.array([[-300.0, 0.0], [-200.0, 0.0]]), h, small_extent)
    out, cuts = clear_under_decks(h, small_extent, [deck, road])
    assert cuts == {} and np.abs(out - h).max() < 1e-6


def test_merge_split_bridge():
    parts = [bridge(line(0, 100), oid=1), bridge(line(100, 200), oid=2), bridge(line(200, 300), oid=3)]
    merged = merge_bridges([parts[1], parts[2], parts[0]])
    assert len(merged) == 1 and merged[0].osm_ids == [1, 2, 3]
    assert (np.diff(merged[0].nodes[:, 0]) > 0).all()  # continuous, no repeated joint


def test_merge_ignores_different_names():
    assert len(merge_bridges([bridge(line(0, 100), name="X"), bridge(line(100, 200), name="Y")])) == 2


def test_merge_unnamed_by_kind():
    a = bridge(line(0, 100), name=None, kind="service")
    b = bridge(line(100, 200), name=None, kind="service", oid=2)
    c = bridge(line(200, 300), name=None, kind="primary", oid=3)
    assert sorted(len(r.osm_ids) for r in merge_bridges([a, b, c])) == [1, 2]


def test_merge_reversed_two_way_piece():
    a = bridge(line(0, 100), oid=1)
    b = bridge(line(200, 100), oid=2)  # drawn backwards
    merged = merge_bridges([a, b])
    assert len(merged) == 1 and (np.diff(merged[0].nodes[:, 0]) > 0).all()


def test_merge_keeps_one_way_directions():
    a = bridge(line(0, 100), oid=1, oneway=True)
    b = bridge(line(200, 100), oid=2, oneway=True)  # opposite carriageway direction
    assert len(merge_bridges([a, b])) == 2


def test_bridge_clipped_end(small_extent):
    from brmap.roads import build_roads

    h = grid(small_extent, lambda X, Y: 4 + X / 100)
    half = small_extent.half_size
    a = small_extent.to_lonlat(half - 300, 0.0)
    b = small_extent.to_lonlat(half + 300, 0.0)
    osm = {"elements": [{"type": "way", "id": 5, "tags": {"highway": "primary", "bridge": "yes",
            "layer": "1", "name": "Edge Bridge"},
            "geometry": [{"lon": a[0], "lat": a[1]}, {"lon": b[0], "lat": b[1]}]}]}
    roads = apply_bridges(build_roads(osm, small_extent, h), h, small_extent, Config())
    (br,) = roads
    assert not np.isnan(br.nodes[:, 2]).any()
    assert br.nodes[-1, 0] == pytest.approx(half, abs=0.01)
    assert br.nodes[-1, 2] == pytest.approx(sample_height(h, small_extent, half, 0.0), abs=0.01)


def test_apply_bridges_clears_road_underneath(small_extent):
    h = grid(small_extent, lambda X, Y: 5.0 + 0 * X)
    over = bridge(line(-200, 200), layer=1)
    under = ground_road(np.array([[0.0, -100.0], [0.0, 100.0]]), h, small_extent)
    out = apply_bridges([over, under], h, small_extent, Config())
    deck = next(r for r in out if r.is_bridge)
    assert deck.nodes[10, 2] >= 5.0 + 5.5 - 0.01
    assert any(not r.is_bridge for r in out)


def test_apply_bridges_ignores_approach_roads(small_extent):
    h = grid(small_extent, lambda X, Y: 5.0 + 0 * X)
    approach = ground_road(np.array([[-300.0, 0.0], [-200.0, 0.0]]), h, small_extent)
    alone = apply_bridges([bridge(line(-200, 200), layer=1)], h, small_extent, Config())[0]
    with_road = next(r for r in apply_bridges([bridge(line(-200, 200), layer=1), approach], h,
                                              small_extent, Config()) if r.is_bridge)
    np.testing.assert_allclose(with_road.nodes[:, 2], alone.nodes[:, 2])  # touching the end adds nothing
    assert with_road.nodes[:, 2].max() <= 5.0 + MARGIN / 0.36 + 0.01      # only the small terrain margin


def test_apply_bridges_stacks_higher_layer_over_lower(small_extent):
    h = grid(small_extent, lambda X, Y: 5.0 + 0 * X)
    low = bridge(line(-300, 300), name="Low", layer=1, oid=1)
    high = bridge(np.column_stack([np.zeros(21), np.linspace(-300, 300, 21)]), name="High", layer=2, oid=2)
    low_road = ground_road(np.array([[-50.0, -40.0], [-50.0, 40.0]]), h, small_extent)
    out = apply_bridges([high, low, low_road], h, small_extent, Config())
    lo = next(r for r in out if r.name == "Low")
    hi = next(r for r in out if r.name == "High")
    z_low_at_cross = np.interp(0.0, lo.nodes[:, 0], lo.nodes[:, 2])
    z_high_at_cross = np.interp(0.0, hi.nodes[:, 1], hi.nodes[:, 2])
    assert z_high_at_cross >= z_low_at_cross + 5.5 - 0.01


def test_apply_bridges_uses_override_by_name(small_extent):
    h = grid(small_extent, lambda X, Y: 5.0 + 0 * X)
    br = bridge(line(-400, 400), name="Big Bridge")
    cfg = Config(bridge_overrides={"Big Bridge": 20.0})
    (deck,) = apply_bridges([br], h, small_extent, cfg)
    assert deck.nodes[:, 2].max() == pytest.approx(25.0, abs=0.01)
    assert not math.isnan(deck.nodes[0, 2])


def by_id(roads):
    return {r.osm_ids[0]: r for r in roads if r.is_bridge}


def crossing_roads(h, extent, xs, oid=90):
    return [ground_road(np.array([[x, -50.0], [x, 50.0]]), h, extent, oid=oid + i)
            for i, x in enumerate(xs)]


def test_touching_pieces_share_their_joint_above_the_ground(small_extent):
    # "I 10" on layer 1 runs into "X Bridge" on layer 2, with roads under both near the joint
    h = grid(small_extent, lambda X, Y: 5.0 + 0 * X)
    a = bridge(line(-300, 0, n=16), name="I 10", oid=1, layer=1, oneway=True)
    b = bridge(line(0, 300, n=16), name="X Bridge", oid=2, layer=2, oneway=True)
    d = by_id(apply_bridges([a, b] + crossing_roads(h, small_extent, (-60.0, 60.0)), h,
                            small_extent, Config()))
    assert d[1].nodes[-1, 2] == pytest.approx(d[2].nodes[0, 2], abs=1e-6)
    assert d[1].nodes[-1, 2] >= 5.0 + 5.5 - 0.05  # level between the crossings, not down at the ground
    assert d[1].nodes[0, 2] == pytest.approx(5.0, abs=0.01)
    assert d[2].nodes[-1, 2] == pytest.approx(5.0, abs=0.01)


def test_ramp_joining_a_deck_mid_span_starts_at_its_height(small_extent):
    h = grid(small_extent, lambda X, Y: 5.0 + 0 * X)
    main = bridge(line(-400, 400, n=41), name="Main", oid=1)
    ramp_xy = np.column_stack([np.linspace(100, 400, 16), np.linspace(0, -60, 16)])  # leaves at x = 100
    ramp = bridge(ramp_xy, name=None, kind="motorway_link", oid=2, oneway=True)
    d = by_id(apply_bridges([main, ramp] + crossing_roads(h, small_extent, (0.0,)), h,
                            small_extent, Config()))
    z_main = np.interp(100.0, d[1].nodes[:, 0], d[1].nodes[:, 2])
    assert z_main > 7.0  # the mainline is still climbing down from the road it crosses
    assert d[2].nodes[0, 2] == pytest.approx(z_main, abs=0.01)
    assert d[2].nodes[-1, 2] == pytest.approx(5.0, abs=0.01)


def test_ramp_follows_the_mainline_until_their_decks_part(small_extent):
    h = grid(small_extent, lambda X, Y: 5.0 + 0 * X)
    main = bridge(line(-400, 400, n=41), name="Main", oid=1)
    xs = np.linspace(0, 300, 16)  # leaves the mainline at x = 0, 3 degrees off, down to the ground
    ramp = bridge(np.column_stack([xs, -xs * math.tan(math.radians(3))]), name=None,
                  kind="motorway_link", oid=2, oneway=True)
    d = by_id(apply_bridges([main, ramp] + crossing_roads(h, small_extent, (40.0,)), h,
                            small_extent, Config()))
    m, r = d[1], d[2]
    together = np.abs(r.nodes[:, 1]) < m.width / 2  # its centerline still on the mainline's pavement
    assert together.sum() >= 3
    z_main = np.interp(r.nodes[:, 0], m.nodes[:, 0], m.nodes[:, 2])
    np.testing.assert_allclose(r.nodes[together, 2], z_main[together], atol=0.05)
    assert r.nodes[-1, 2] == pytest.approx(5.0, abs=0.01)


def test_fork_continues_straight_and_the_branch_starts_at_the_joint(small_extent):
    h = grid(small_extent, lambda X, Y: 5.0 + 0 * X)
    a = bridge(line(-300, 0, n=16), name="I 10", oid=1, oneway=True)
    b = bridge(line(0, 300, n=16), name="I 10", oid=2, layer=2, oneway=True)
    branch_xy = np.column_stack([np.linspace(0, 260, 14), np.linspace(0, 150, 14)])  # 30 degrees off
    c = bridge(branch_xy, name=None, kind="motorway_link", oid=3, oneway=True)
    d = by_id(apply_bridges([a, b, c] + crossing_roads(h, small_extent, (-60.0, 60.0)), h,
                            small_extent, Config()))
    joint = d[1].nodes[-1, 2]
    assert joint >= 5.0 + 5.5 - 0.05
    assert d[2].nodes[0, 2] == pytest.approx(joint, abs=1e-6)
    assert d[3].nodes[0, 2] == pytest.approx(joint, abs=0.01)
    assert d[3].nodes[-1, 2] == pytest.approx(5.0, abs=0.01)


def test_override_crest_runs_on_into_the_approaches(small_extent):
    h = grid(small_extent, lambda X, Y: 5.0 + 0 * X)
    west = bridge(line(-500, -200, n=16), name="I 10", oid=1, layer=2, oneway=True)
    span = bridge(line(-200, 200, n=21), name="Big Bridge", oid=2, layer=2, oneway=True)
    east = bridge(line(200, 500, n=16), name="I 10", oid=3, layer=2, oneway=True)
    d = by_id(apply_bridges([west, span, east], h, small_extent,
                            Config(bridge_overrides={"Big Bridge": 25.0})))
    x = np.concatenate([d[1].nodes[:, 0], d[2].nodes[1:, 0], d[3].nodes[1:, 0]])
    z = np.concatenate([d[1].nodes[:, 2], d[2].nodes[1:, 2], d[3].nodes[1:, 2]])
    assert np.interp(0.0, x, z) == pytest.approx(30.0, abs=0.05)  # 25 m over the 5 m ground line
    assert d[1].nodes[-1, 2] == pytest.approx(d[2].nodes[0, 2], abs=1e-6)
    assert d[2].nodes[-1, 2] == pytest.approx(d[3].nodes[0, 2], abs=1e-6)
    assert d[2].nodes[0, 2] > 15.0  # the climb starts on the approach, not at a joint on the ground
    assert (np.abs(np.diff(z)) / np.diff(x)).max() <= MAX_GRADE + 1e-6
    assert z[0] == pytest.approx(5.0, abs=0.01) and z[-1] == pytest.approx(5.0, abs=0.01)
