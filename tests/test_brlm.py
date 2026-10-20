"""The landmark model helpers (blender/lib/brlm.py): their plain-Python geometry, without Blender."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "blender" / "lib"))
import brlm  # noqa: E402

# the Memorial Tower's footprint in model coordinates (out/landmarks.json, full map, 2026-10-19)
TOWER_FOOTPRINT = [(6.735, 9.211), (-9.725, 9.11), (-9.706, 5.63), (-22.206, 5.548), (-22.143, -6.66),
                   (-9.631, -6.579), (-9.625, -6.752), (6.835, -6.651), (6.828, -6.487), (18.097, -6.412),
                   (18.033, 5.797), (6.764, 5.731)]
SHAFT = (-4.0, -3.3, 4.0, 3.3)


def area(rects):
    return sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in rects)


def overlaps(rects):
    return any(min(a[2], b[2]) - max(a[0], b[0]) > 1e-9 and min(a[3], b[3]) - max(a[1], b[1]) > 1e-9
               for i, a in enumerate(rects) for b in rects[i + 1:])


def polygon_area(ring):
    return abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1]))) / 2


def test_split_rectangles_covers_the_tower_footprint():
    rects = brlm.split_rectangles(TOWER_FOOTPRINT)
    assert len(rects) == 2 and not overlaps(rects)   # the band through the wings, the strip beside it
    assert area(rects) == pytest.approx(polygon_area(TOWER_FOOTPRINT), rel=0.02)


def test_split_rectangles_leaves_out_the_holes():
    rects = brlm.split_rectangles(TOWER_FOOTPRINT, holes=[SHAFT])
    assert len(rects) == 5 and not overlaps(rects)
    assert area(rects) == pytest.approx(polygon_area(TOWER_FOOTPRINT) - 8.0 * 6.6, rel=0.02)
    for x0, y0, x1, y1 in rects:                 # none reaches into the shaft
        assert x1 <= SHAFT[0] + 1e-9 or x0 >= SHAFT[2] - 1e-9 or y1 <= SHAFT[1] + 1e-9 or y0 >= SHAFT[3] - 1e-9
    assert {-4.0, 4.0} <= {x for r in rects for x in (r[0], r[2])}   # the holes' own lines, exactly


def test_split_rectangles_of_an_l():
    rects = brlm.split_rectangles([(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)])
    assert sorted(rects) == [(0, 0, 10, 4), (0, 4, 4, 10)]


def face_normals(b):
    return [brlm.normal(pts) for pts, _, _ in b.faces]


def test_box_faces_outwards_and_skips():
    b = brlm.box(brlm.Builder(), (0, 0, 0), (2, 3, 4), "m", skip=("-z",), mats={"+z": "top"})
    normals = face_normals(b)
    assert len(normals) == 5 and b.tris == 10 and b.materials == {"m", "top"}
    centre = (1, 1.5, 2)
    for pts, _, _ in b.faces:
        mid = [sum(c) / len(pts) for c in zip(*pts)]
        assert brlm.dot(brlm.normal(pts), brlm.sub(mid, centre)) > 0
    with pytest.raises(ValueError, match="top"):
        brlm.box(brlm.Builder(), (0, 0, 0), (1, 1, 1), "m", skip=("top",))


def test_wall_uv_is_in_metres():
    b = brlm.Builder()
    b.face([(0, 0, 0), (4, 0, 0), (4, 0, 2), (0, 0, 2)], "m", tile=2.0)      # faces -y
    (_, uv, _), = b.faces
    us, vs = [u for u, _ in uv], [v for _, v in uv]
    assert max(us) - min(us) == pytest.approx(2.0) and vs == pytest.approx([0, 0, 1, 1])
    assert uv[1][0] > uv[0][0]                                                # u runs left to right


def test_hipped_roof_slopes_face_up_and_out():
    b = brlm.hipped_roof(brlm.Builder(), 0, 0, 10, 6, 5.0, 30, "roof", overhang=0.5, soffit="eave")
    roof = [pts for pts, _, m in b.faces if m == "roof"]
    assert len(roof) == 4 and b.tris == 6 + 2
    top = max(p[2] for pts in roof for p in pts)
    assert top == pytest.approx(5.0 + 3.5 * math.tan(math.radians(30)))
    for pts in roof:
        n = brlm.normal(pts)
        mid = [sum(c) / len(pts) for c in zip(*pts)]
        assert n[2] > 0 and n[0] * (mid[0] - 5) + n[1] * (mid[1] - 3) > 0
    (eave,) = [pts for pts, _, m in b.faces if m == "eave"]
    assert brlm.normal(eave)[2] == pytest.approx(-1)
    assert brlm.hipped_roof(brlm.Builder(), 0, 0, 4, 4, 0.0, 30, "roof").tris == 4    # square: a pyramid


def test_decal_faces_where_it_is_told():
    b = brlm.decal(brlm.Builder(), (0, -3, 40), brlm.cross((0, 0, 1), (0, -1, 0)), 2.6, 2.6, "clock")
    (pts, uv, _), = b.faces
    assert brlm.normal(pts) == pytest.approx((0, -1, 0))
    top = [u for p, u in zip(pts, uv) if p[2] > 40]
    assert sorted(v for _, v in top) == [1.0, 1.0]                            # the image's top is up


def test_arched_wall_cuts_round_openings():
    out = (0.0, -1.0, 0.0)                                                    # the wall faces -y
    right = brlm.cross((0, 0, 1), out)
    b = brlm.arched_wall(brlm.Builder(), (0, 0, 0), right, 8.0, 0.0, 5.0, [(2.5, 2.0, 1.0, 2.5), (5.5, 2.0, 1.0, 2.5)],
                         0.5, "wall", segments=8, double_inner=True)
    outer = [pts for pts in (f[0] for f in b.faces) if brlm.normal(pts) == pytest.approx(out) and
             all(abs(p[1]) < 1e-9 for p in pts)]
    wall_area = sum(polygon_area([(p[0], p[2]) for p in pts]) for pts in outer)
    hole = 2 * (2.0 * 1.5 + 8 * 0.5 * 1.0 ** 2 * math.sin(math.pi / 8))      # rectangle + the arch polygon
    assert wall_area == pytest.approx(8.0 * 5.0 - hole)
    inner = [pts for pts in (f[0] for f in b.faces) if all(abs(p[1] - 0.5) < 1e-9 for p in pts)]
    facing = {tuple(round(c, 6) for c in brlm.normal(pts)) for pts in inner}
    assert facing == {(0.0, 1.0, 0.0), (0.0, -1.0, 0.0)}                     # double-sided
    with pytest.raises(ValueError, match="doesn't fit"):
        brlm.arched_wall(brlm.Builder(), (0, 0, 0), right, 8.0, 0.0, 5.0, [(2.5, 2.0, 1.0, 4.5)], 0.5, "wall")


# ------------------------------------------------------------------ vertex colours, rings, lofts, steps, towers

def test_face_colours_ride_along_with_the_faces():
    b = brlm.Builder()
    b.face([(0, 0, 0), (1, 0, 0), (0, 1, 0)], "m")
    b.face([(0, 0, 1), (1, 0, 1), (0, 1, 1)], "m", col=(0.2, 0.4, 0.6), double=True)
    assert b.cols == [None, (0.2, 0.4, 0.6, 1.0), (0.2, 0.4, 0.6, 1.0)]
    other = brlm.Builder()
    other.face([(0, 0, 2), (1, 0, 2), (0, 1, 2)], "m", col=(1, 0, 0, 0.5))
    b.extend(other)
    assert len(b.cols) == len(b.faces) == 4 and b.cols[-1] == (1.0, 0.0, 0.0, 0.5)
    assert b.corners == 12                                 # the most vertices the welded mesh can have
    with pytest.raises(ValueError, match="0..1"):
        b.face([(0, 0, 0), (1, 0, 0), (0, 1, 0)], "m", col=(70, 29, 124))


SQUARE = [(5.0, -5.0), (5.0, 5.0), (-5.0, 5.0), (-5.0, -5.0)]


def test_ring_at_is_where_a_ray_leaves_the_ring():
    assert brlm.ring_at(SQUARE, (0.0, 0.0), 0.0) == pytest.approx((5.0, 0.0))
    assert brlm.ring_at(SQUARE, (0.0, 0.0), 45.0) == pytest.approx((5.0, 5.0))
    assert brlm.ring_at(SQUARE, (1.0, 0.0), 180.0) == pytest.approx((-5.0, 0.0))
    with pytest.raises(ValueError, match="outside"):
        brlm.ring_at(SQUARE, (20.0, 0.0), 0.0)


def test_centroid_and_angles_of_a_ring():
    assert brlm.centroid([(0, 0), (4, 0), (4, 2), (0, 2)]) == pytest.approx((2.0, 1.0))
    assert brlm.angle_of((0.0, 0.0), (0.0, 3.0)) == pytest.approx(90.0)
    assert brlm.angle_of((0.0, 0.0), (0.0, -3.0)) == pytest.approx(270.0)        # 0 <= angle < 360


def ring3(r, z, n=8):
    return [(r * math.cos(2 * math.pi * k / n), r * math.sin(2 * math.pi * k / n), z) for k in range(n)]


def test_loft_joins_rings_with_quads_facing_up_or_out():
    b = brlm.loft(brlm.Builder(), [ring3(1.0, 0.0), ring3(2.0, 0.0)], "m", closed=True)
    assert len(b.faces) == 8 and all(len(pts) == 4 for pts, _, _ in b.faces)
    assert all(brlm.normal(pts)[2] == pytest.approx(1.0) for pts, _, _ in b.faces)   # inner to outer: up
    wall = brlm.loft(brlm.Builder(), [ring3(2.0, 0.0), ring3(2.0, 3.0)], "m", closed=False)
    assert len(wall.faces) == 7                                                    # an open strip
    for pts, _, _ in wall.faces:                                                   # bottom to top: inwards
        mid = [sum(c) / 4 for c in zip(*pts)]
        assert brlm.dot(brlm.normal(pts), (mid[0], mid[1], 0.0)) < 0
    out = brlm.loft(brlm.Builder(), [ring3(2.0, 0.0), ring3(2.0, 3.0)], "m", flip=True)
    assert all(brlm.dot(brlm.normal(p), (sum(c[0] for c in p), sum(c[1] for c in p), 0.0)) > 0 for p, _, _ in out.faces)


def test_loft_splits_a_twisted_quad_and_drops_a_flat_one():
    lower = [(4.0, 0.0, 0.0), (0.0, 0.0, 0.0)]                 # counter-clockwise round (2, 2), inner to outer
    upper = [(4.0, 4.0, 2.0), (0.0, 4.0, 0.0)]                 # one corner 2 m up: not planar
    b = brlm.loft(brlm.Builder(), [lower, upper], "m", col=(0.5, 0.5, 0.5))
    assert [len(pts) for pts, _, _ in b.faces] == [3, 3] and b.cols == [(0.5, 0.5, 0.5, 1.0)] * 2
    assert all(brlm.normal(pts)[2] > 0 for pts, _, _ in b.faces)
    same = brlm.loft(brlm.Builder(), [lower, lower], "m")      # no area: nothing
    assert same.faces == []


def test_step_rings_rise_in_level_treads_and_upright_risers():
    front = [(0.0, 0.0, 1.0), (0.0, 10.0, 1.0)]
    back = [(8.0, 0.0, 5.0), (8.0, 10.0, 5.0)]
    rings = brlm.step_rings(front, back, 4)
    assert len(rings) == 9 and rings[0] == front and rings[-1] == back
    for j in range(4):
        tread_from, tread_to, riser_top = rings[2 * j], rings[2 * j + 1], rings[2 * j + 2]
        assert [p[2] for p in tread_to] == pytest.approx([p[2] for p in tread_from])          # level
        assert [p[:2] for p in riser_top] == pytest.approx([p[:2] for p in tread_to])          # upright
        assert tread_to[0][0] == pytest.approx(2.0 * (j + 1)) and riser_top[0][2] == pytest.approx(1.0 + j + 1)


def test_seat_steps_face_up_and_towards_the_front():
    # rising towards -x, the field at +x: the rows run counter-clockwise round it (the field on their left)
    front = [(0.0, 0.0, 1.0), (0.0, -10.0, 1.0), (0.0, -20.0, 1.0)]
    back = [(-8.0, 0.0, 5.0), (-8.0, -10.0, 5.0), (-8.0, -20.0, 5.0)]
    b = brlm.seat_steps(brlm.Builder(), front, back, 4, "seats", tread=(1, 0, 0), riser=(0, 0, 1))
    assert len(b.faces) == 2 * 4 * 2
    treads = [pts for (pts, _, _), c in zip(b.faces, b.cols) if c == (1.0, 0.0, 0.0, 1.0)]
    risers = [pts for (pts, _, _), c in zip(b.faces, b.cols) if c == (0.0, 0.0, 1.0, 1.0)]
    assert len(treads) == len(risers) == 8
    assert all(brlm.normal(p) == pytest.approx((0.0, 0.0, 1.0)) for p in treads)
    assert all(brlm.normal(p) == pytest.approx((1.0, 0.0, 0.0)) for p in risers)        # towards the field


def test_lattice_tower_stands_in_its_footprint():
    b = brlm.lattice_tower(brlm.Builder(), (10.0, 20.0), 3.0, 40.0, 52.0, "steel", panels=4, top_width=2.0)
    pts = [p for f, _, _ in b.faces for p in f]
    # (the slanting members' corners reach a few cm past the ends)
    assert min(p[2] for p in pts) == pytest.approx(40.0, abs=0.1)
    assert max(p[2] for p in pts) == pytest.approx(52.0, abs=0.1)
    assert max(abs(p[0] - 10.0) for p in pts) <= 1.5 + 0.25 and max(abs(p[1] - 20.0) for p in pts) <= 1.5 + 0.25
    top = [p for p in pts if p[2] > 51.95]
    assert max(abs(p[0] - 10.0) for p in top) <= 1.0 + 0.25                                # tapered
    legs_only = brlm.lattice_tower(brlm.Builder(), (0.0, 0.0), 3.0, 0.0, 12.0, "steel", panels=4, braces=False)
    assert 0 < len(legs_only.faces) < len(b.faces)
    with pytest.raises(ValueError, match="panel"):
        brlm.lattice_tower(brlm.Builder(), (0.0, 0.0), 3.0, 0.0, 12.0, "steel", panels=0)


def test_beam_is_a_square_tube_between_two_points():
    b = brlm.beam(brlm.Builder(), (0.0, 0.0, 0.0), (0.0, 0.0, 5.0), 0.4, "steel")
    assert len(b.faces) == 4
    xs = [p[0] for f, _, _ in b.faces for p in f]
    assert min(xs) == pytest.approx(-0.2) and max(xs) == pytest.approx(0.2)
    for pts, _, _ in b.faces:
        mid = [sum(c) / 4 for c in zip(*pts)]
        assert brlm.dot(brlm.normal(pts), (mid[0], mid[1], 0.0)) > 0                     # outwards


def test_blind_arches_are_recessed_not_cut_through():
    out = (0.0, -1.0, 0.0)
    right = brlm.cross((0, 0, 1), out)
    b = brlm.blind_arches(brlm.Builder(), (0, 0, 0), right, 8.0, 0.0, 5.0, [(2.5, 2.0, 1.0, 2.5), (5.5, 2.0, 1.0, 2.5)],
                          0.5, "wall", recess=(0.3, 0.3, 0.3), segments=8)
    front = [pts for pts, _, _ in b.faces if all(abs(p[1]) < 1e-9 for p in pts)]
    hole = 2 * (2.0 * 1.5 + 8 * 0.5 * 1.0 ** 2 * math.sin(math.pi / 8))
    assert sum(polygon_area([(p[0], p[2]) for p in pts]) for pts in front) == pytest.approx(8.0 * 5.0 - hole)
    backs = [(pts, c) for (pts, _, _), c in zip(b.faces, b.cols) if all(abs(p[1] - 0.5) < 1e-9 for p in pts)]
    assert len(backs) == 2 and all(c == (0.3, 0.3, 0.3, 1.0) for _, c in backs)
    assert all(brlm.normal(pts) == pytest.approx(out) for pts, _ in backs)                # seen from outside
    assert not any(brlm.normal(pts)[1] > 0.5 for pts, _, _ in b.faces)                     # no inner face


def test_oriented_box_faces_out_along_its_axes():
    a = (math.cos(0.3), math.sin(0.3), 0.0)
    axes = (a, brlm.cross((0, 0, 1), a), (0.0, 0.0, 1.0))
    b = brlm.oriented_box(brlm.Builder(), (5.0, 5.0, 2.0), axes, (1.0, 2.0, 0.5), "m")
    assert len(b.faces) == 6
    for pts, _, _ in b.faces:
        mid = [sum(c) / 4 for c in zip(*pts)]
        assert brlm.dot(brlm.normal(pts), brlm.sub(mid, (5.0, 5.0, 2.0))) > 0
    with pytest.raises(ValueError, match="right-handed"):
        brlm.oriented_box(brlm.Builder(), (0, 0, 0), (axes[1], axes[0], axes[2]), (1, 1, 1), "m")


def test_light_tower_points_its_lamps_at_the_field():
    b = brlm.light_tower(brlm.Builder(), (0.0, 0.0), 2.4, 40.0, 50.0, (1.0, 1.0, 0.0), "steel", "lamp",
                         bank=(8.0, 4.0, 0.5), lamps=(4, 6), tilt_deg=20.0, panels=3)
    lamps = [pts for pts, _, mat in b.faces if mat == "lamp"]
    assert len(lamps) == 24
    for pts in lamps:
        n = brlm.normal(pts)
        assert n[0] > 0.6 and n[1] > 0.6 and n[2] == pytest.approx(-math.sin(math.radians(20.0)), abs=1e-6)
        assert min(p[2] for p in pts) > 50.0                                  # the bank sits on the mast
    plain = brlm.light_tower(brlm.Builder(), (0.0, 0.0), 2.4, 40.0, 50.0, (1.0, 0.0, 0.0), "steel", "lamp",
                             bank=(8.0, 4.0, 0.5), lamps=None, lattice=False)
    assert plain.materials == {"steel"} and len(plain.faces) < len(b.faces)


# ------------------------------------------------------------------ fills, offsets, walls along a line

def signed_area(ring):
    return sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1])) / 2


ARROW = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (5.0, 4.0), (0.0, 10.0)]    # concave at (5, 4)


def test_triangulate_covers_a_concave_ring():
    for ring in (ARROW, ARROW[::-1], ARROW[:1] + [(5.0, 0.0)] + ARROW[1:]):   # either way round, a corner on an edge
        tris = brlm.triangulate(ring)
        assert len(tris) == len(ring) - 2
        pts = [[ring[i] for i in t] for t in tris]
        assert all(signed_area(p) > 1e-9 for p in pts)                       # counter-clockwise, none flat
        assert sum(signed_area(p) for p in pts) == pytest.approx(polygon_area(ring))


def test_triangulate_refuses_a_ring_with_no_area():
    with pytest.raises(ValueError, match="area"):
        brlm.triangulate([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])


def test_fill_faces_up_or_down():
    up = brlm.fill(brlm.Builder(), ARROW, 2.0, "m", col=(0.5, 0.5, 0.5))
    assert up.tris == 3 and all(p[2] == 2.0 for pts, _, _ in up.faces for p in pts)
    assert all(brlm.normal(pts) == pytest.approx((0.0, 0.0, 1.0)) for pts, _, _ in up.faces)
    assert up.cols == [(0.5, 0.5, 0.5, 1.0)] * 3
    down = brlm.fill(brlm.Builder(), ARROW[::-1], 0.0, "m", down=True)
    assert all(brlm.normal(pts) == pytest.approx((0.0, 0.0, -1.0)) for pts, _, _ in down.faces)
    assert sum(brlm._area(pts) for pts, _, _ in down.faces) == pytest.approx(polygon_area(ARROW))


def test_offset_ring_moves_every_edge_out():
    square = [(-5.0, -5.0), (5.0, -5.0), (5.0, 5.0), (-5.0, 5.0)]
    assert brlm.offset_ring(square, 1.0) == pytest.approx([(-6.0, -6.0), (6.0, -6.0), (6.0, 6.0), (-6.0, 6.0)])
    assert brlm.offset_ring(square[::-1], 1.0) == pytest.approx([(-6.0, 6.0), (6.0, 6.0), (6.0, -6.0), (-6.0, -6.0)])
    assert brlm.offset_ring(square, -1.0) == pytest.approx([(-4.0, -4.0), (4.0, -4.0), (4.0, 4.0), (-4.0, 4.0)])


def test_offset_line_keeps_its_ends_square():
    assert brlm.offset_line([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)], 1.0) == pytest.approx(
        [(0.0, 1.0), (9.0, 1.0), (9.0, 10.0)])                               # to the left, mitred at the bend


def test_wall_along_a_bent_line():
    b = brlm.wall_along(brlm.Builder(), [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)], 0.0, 3.0, 0.5, "front",
                        back="back", top="top", col=(0.2, 0.4, 0.2))
    normals = {mat: {tuple(round(c, 6) + 0.0 for c in brlm.normal(pts)) for pts, _, m in b.faces if m == mat}
               for mat in ("front", "back", "top")}
    assert normals["front"] == {(0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)}             # the left of the way along
    assert normals["back"] == {(0.0, -1.0, 0.0), (1.0, 0.0, 0.0)}
    assert normals["top"] == {(0.0, 0.0, 1.0)}
    backs = [p for pts, _, m in b.faces if m == "back" for p in pts]
    assert min(p[1] for p in backs) == pytest.approx(-0.5) and max(p[0] for p in backs) == pytest.approx(10.5)
    (start,) = [pts for pts, _, _ in b.faces if all(abs(p[0]) < 1e-9 for p in pts)]
    (end,) = [pts for pts, _, _ in b.faces if all(abs(p[1] - 10.0) < 1e-9 for p in pts)]
    assert brlm.normal(start) == pytest.approx((-1.0, 0.0, 0.0)) and brlm.normal(end) == pytest.approx((0.0, 1.0, 0.0))
    assert all(c == (0.2, 0.4, 0.2, 1.0) for (_, _, m), c in zip(b.faces, b.cols) if m == "front")
