"""Deck dressing: concrete Jersey barriers along both edges of every bridge deck and piers under the
decks, drawn as tiles of meshes in the game's own concrete.

out/dressing.json (run) holds {"barriers": [{"deck", "side", "points": [[x, y, z], ...]}], "piers":
[{"deck", "x", "y", "yaw", "width", "z_top", "z_bottom"}]} in level coordinates: "deck" is the deck's
deck_id, "side" "l" or "r" of its direction (its node order), "points" the barrier's centre line on
the deck's surface; a pier's yaw is the deck's direction there (radians counter-clockwise from +x),
its width across the deck, z_top the deck's underside.

* deck_curve(nodes) -> (M, 3): the deck as its MeshRoad is drawn, a Catmull-Rom spline through its
  nodes: at a node the tangent is (next - previous) over their distance along the nodes, times the
  segment's length (on evenly spaced nodes Torque's CatmullRom, the end nodes repeated; on a short
  segment between long ones it does not loop back), each segment cut into pieces at most MAX_STEP
  long, and into enough of them that the spline strays at most CURVE_TOL from their chords (the
  chords of a loop ramp's 20 m segments miss it by 0.8 m).
* deck_id(road) -> "w<id>+w<id>...": the deck's OSM ways.
* barriers(decks, roads) -> [{"deck", "side", "points"}]: both edges of every deck, its deck_curve
  offset width / 2 - INSET to each side (markings.offset_polyline), or on a deck narrower than
  NARROW_DECK (the one-lane ramps and the narrow two-lane street bridges) width / 2 + NARROW_OUT,
  just outside the carriageway: the base overhangs the slab's edge by 0.55 m and reaches 0.05 m
  onto it, not 0.6, so a 3.5 m deck keeps 3.4 m between the bases; END_TRIM taken off each end,
  less the stretches inside any other road's corridor (its centre line buffered by width / 2, and
  CUT_TOL so the cut ends stay outside it) on the barrier's level: where that road's surface is
  within SAME_LEVEL of the barrier's base, the barrier would stand on it (a ramp merging, a deck
  meeting at an angle) or reach into its slab. A road passing under the deck, or a deck over it,
  leaves the barrier be. The level is read every DZ_STEP along each stretch, and a stretch is cut
  to the first point past where the level changes. Pieces shorter than MIN_PIECE are dropped; per
  deck, the left pieces, then the right.
  Where a barrier side is joined at a straight joint (straight_joints) it runs on from one deck to
  the next instead: no END_TRIM at that end, and within JOINT_REACH of the joint no cut by the deck
  joined there, nor by another deck ending there whose barrier on that side lies inside the joined
  pair's (joint_decks); one lying outside them cuts them as at any ramp merge. A wider deck's
  barrier tapers in to the narrower one's line by the joint, TAPER m along per m across, on its
  own slab. Then the two lines meet: cut back to where they cross within JOINT_REACH of the joint
  (on the inside of a bend), or else the first carried on to the second's start (over the wedge
  between the two slabs' square ends on the outside), so they end at the same point.
* straight_joints(decks) -> {(deck, end, side): (deck, end, side)}: the barrier sides joined at
  the straight joints, both ways round: two decks' ends (indices into decks, end 0 or -1) within
  JOINT_TOL (bridges.JOIN_TOL) of each other, on one level (|dz| < SAME_LEVEL), one deck leaving
  the joint within 30 degrees of straight on from the other (STRAIGHT_COS, markings.CONTINUE_COS;
  their directions taken over markings.LEAVE_LEN), and the side, "l" or "r", its own on each deck.
  Where more ends meet (a deck forking in two, or two merging into one), each side of the travel
  joins the pair whose barriers there lie outermost JOINT_REACH from the joint (_outer), each the
  outermost of the ends straight on from the other: at the I-10 fork at the Wilkinson bridge's
  west end the trunk's north barrier runs on onto the ramp's and its south one onto I-10's. The
  branches' sides facing each other (the gore) join nothing and are left to the cut, as are both
  sides of a narrow ramp whose slab lies inside the deck carrying straight on there.
* joint_decks(decks, joints=None) -> {deck: [(other deck, joint (x, y), side)]}: for each of the
  deck's joined barrier sides, the decks whose corridors leave that barrier be within JOINT_REACH
  of the joint: the deck joined there, and any other deck ending there whose barrier on that side
  lies no further out than the joined deck's on its own side of the joint;
  validate.dressing_problems exempts the same vertices.
* piers(decks, roads, heights, extent, spans) -> [pier]: along each deck's curve every
  PIER_SPACING from its start (its ends are the abutments) where its underside (DECK_DEPTH under the
  surface) is PIER_MIN_GAP or more above the highest terrain under the pier (its corners and its
  centre: over the river, its bed). A pier is PIER_THICK along the deck and min(PIER_SHARE x the
  deck's width, PIER_MAX_WIDTH) across it (pier_footprint), from z_bottom, PIER_SINK under the lowest
  terrain there, to z_top, the underside. It is left out when its footprint comes within
  PIER_ROAD_CLEAR of the corridor of another road running under its top (a ground road, a lower
  deck: road_z there below z_top; a deck passing over, or one beside its own at its level, stands
  on it too), and when it comes within SPAN_PAD of one of spans, the footprints of the landmarks
  over decks: the Wilkinson truss brings its own piers, its end piers reaching SPAN_PAD past its
  footprint. Piers stand in the river too, as under the I-10 bridge's east approach past the truss.
* run(cfg) writes dressing.json from out/roads.json, out/heights.npy and out/landmarks.json: the
  footprints of the landmarks over decks (params.decks: the Wilkinson truss, whose west anchor arm
  reaches past the river onto the bank) are piers()'s spans. It warns of every registry landmark
  over decks (landmarks.REGISTRY) that has one of its decks in roads.json but is missing from
  landmarks.json (missing_spans: the landmarks stage not run since), as piers would stand under it.
  Empty lists when cfg.deck_dressing is off, so the level places none.
* tiles(dressing, tile_m=TILE_M) -> [(key, origin, {FULL, LOW, COLLISION: collada.Mesh or None})]:
  each barrier segment goes to the tile holding its middle (a barrier is drawn as runs of segments,
  capped only at its own ends) and each pier to the tile holding its centre, on the grid of
  buildings.tiles (keys e<col>n<row>); positions are relative to the tile's centre (origin), z
  world. A tile with a level over MAX_VERTS vertices is split in four (buildings.QUARTERS, the key
  plus "_<quarter>"), down to buildings.MIN_TILE.
  - FULL (dr_a300): the barriers swept along their points in the Jersey profile (JERSEY: a TOE high
    vertical toe, the 55 degree slope to BREAK, the 84 degree face to the top; BARRIER_SINK into the
    deck, hard edges), with end caps, and the piers' four sides.
  - LOW (dr_a60): the piers only; None in a tile without piers (its barriers vanish there).
  - COLLISION (Colmesh-1): the barriers as a prism of BARRIER's trapezoid (base to top width), capped,
    and the piers' four sides.
  uv in metres / TEX_M; every vertex colour white; the one material MATERIAL.
* level_content(cfg, data_dir, root, heights, extent) (the level.FEATURE_MODULES hook) writes
  SHAPES_DIR/br_dr_<key>.dae for each tile (collada.TILE decimals), each placed by one TSStatic at the
  tile's centre, z 0, unrotated, with persistentId bf.persistent_id("dr", key), in group GROUP.
  MATERIAL is the game's own concrete_plain (art/shapes/objects/main.materials.json in
  content/art_shapes.zip, loaded for every level), so no materials file is written. {} when
  cfg.deck_dressing is off or there is nothing to place.
"""
from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon

from . import beamng_format as bf
from . import collada, dae, landmarks
from .bridges import JOIN_TOL
from .buildings import MIN_TILE, QUARTERS, _tile_key
from .geo import Extent
from .markings import (CONTINUE_COS, DUP_TOL, MIN_LINE, _between, _cum, _dedupe, _leaving, _line_parts, _sub,
                       offset_polyline)
from .roads import MAX_SPACING, load_roads
from .terrain import load_heights, sample_height

log = logging.getLogger(__name__)

DRESSING_FILE = "dressing.json"
GROUP = "dressing"
SHAPES_DIR = "art/shapes/dressing"
MATERIAL = "concrete_plain"    # the game's global concrete (art/shapes/objects/main.materials.json)
FULL, LOW, COLLISION = "dr_a300", "dr_a60", "Colmesh-1"
NODES = (FULL, LOW, COLLISION)
TILE_M = 256.0                 # m, a tile's side (as the buildings')
MAX_VERTS = 60000              # per level of detail and tile (validate.DAE_MAX_VERTS)

# barriers
BARRIER = (0.60, 0.15, 0.81)   # m: base width, top width and height above the deck (the Jersey profile)
INSET = 0.30                   # m from the deck's edge to the barrier's centre line
NARROW_DECK = 7.0              # m: on a deck narrower than this (one-lane ramps, narrow two-lane bridges) ...
NARROW_OUT = 0.25              # ... the barrier's centre line runs this far outside the deck's edge
END_TRIM = 2.0                 # m off each end of a deck's barriers, but where one is joined at a straight joint
TOE = 0.075                    # m: the profile's vertical toe (3 in)
BREAK = (0.127, 0.33)          # m: half width and height of its slope break (55 degrees below, 84 above)
BARRIER_SINK = 0.10            # m the barrier reaches down into the deck
DECK_DEPTH = bf.DECK_DEPTH     # m, a deck MeshRoad's thickness
SAME_LEVEL = DECK_DEPTH + BARRIER[2]   # m: another road's surface this close to a barrier's base is in its way
CUT_TOL = 0.01                 # m beyond another road's corridor a cut ends
DZ_STEP = 0.5                  # m between the points a cut stretch's level is read at
MIN_PIECE = MIN_LINE           # m; shorter barrier pieces are dropped
JOINT_TOL = JOIN_TOL           # m between two decks' ends at a joint (bridges.JOIN_TOL)
STRAIGHT_COS = CONTINUE_COS    # a deck leaving a joint within 30 degrees of straight on continues the other
JOINT_REACH = 10.0             # m round a straight joint in which the decks there leave the joined barriers
                               # be, and from it at which _outer compares the decks' barriers
TAPER = 5.0                    # m along per m across: a wider deck's barriers taper to a narrower one's line

# piers
PIER_SPACING = 30.0            # m along a deck
PIER_MIN_GAP = 2.5             # m from the deck's underside down to the terrain, at least
PIER_THICK = 1.2               # m along the deck
PIER_ROAD_CLEAR = 1.0          # m a pier keeps off another road's corridor
PIER_SHARE = 0.5               # of the deck's width, across it ...
PIER_MAX_WIDTH = 8.0           # ... and at most this, m
PIER_SINK = 0.5                # m a pier reaches into the ground
SPAN_PAD = 2.0                 # m a landmark over decks reaches past its footprint (the truss's 4 m end piers)

# the deck's curve
CURVE_TOL = 0.02               # m the spline may stray from the chords of the curve drawn
MAX_STEP = MAX_SPACING         # m, the longest piece of a deck's curve (roads.MAX_SPACING)

TEX_M = 2.0                    # m per texture repeat
WHITE = (1.0, 1.0, 1.0, 1.0)

_A0, _A2, _H = BARRIER[0] / 2, BARRIER[1] / 2, BARRIER[2]
# (across, up) round the barrier from its left foot over the top to its right foot
JERSEY = ((-_A0, -BARRIER_SINK), (-_A0, TOE), (-BREAK[0], BREAK[1]), (-_A2, _H),
          (_A2, _H), (BREAK[0], BREAK[1]), (_A0, TOE), (_A0, -BARRIER_SINK))
TRAPEZOID = ((-_A0, -BARRIER_SINK), (-_A2, _H), (_A2, _H), (_A0, -BARRIER_SINK))


# ------------------------------------------------------------------ the file

def save_dressing(data: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"barriers": list(data["barriers"]), "piers": list(data["piers"])}),
                    encoding="utf-8")
    return path


def load_dressing(path) -> dict:
    """dressing.json as {"barriers": [...], "piers": [...]}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {"barriers": list(data.get("barriers") or []), "piers": list(data.get("piers") or [])}


# ------------------------------------------------------------------ decks

def deck_id(road) -> str:
    return "+".join(f"w{int(i)}" for i in road.osm_ids)


def _usable(road) -> bool:
    return len(road.nodes) >= 2 and bool(np.isfinite(road.nodes).all())


def deck_curve(nodes) -> np.ndarray:
    """The deck's spline through its nodes, (M, 3); see the module docstring."""
    p = np.asarray(nodes, float).reshape(-1, 3)
    p = p[np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-9]]
    if len(p) < 2:
        return p
    chord = np.linalg.norm(np.diff(p, axis=0), axis=1)
    ends = np.vstack([p[:1], p, p[-1:]])                             # the end nodes repeated
    span = np.r_[chord[0], chord] + np.r_[chord, chord[-1]]          # previous to next node, an end counting twice
    velocity = (ends[2:] - ends[:-2]) / span[:, None]               # per metre at each node
    a, b = p[:-1], p[1:]
    ma, mb = velocity[:-1] * chord[:, None], velocity[1:] * chord[:, None]
    stray = 0.125 * np.linalg.norm(ma - mb, axis=1)                 # the spline's middle off the chord
    pieces = np.maximum(np.ceil(chord / MAX_STEP), np.ceil(np.sqrt(stray / CURVE_TOL))).astype(int)
    out = []
    for i, k in enumerate(np.maximum(pieces, 1)):
        t = (np.arange(k) / k)[:, None]
        t2, t3 = t * t, t * t * t
        out.append((2 * t3 - 3 * t2 + 1) * a[i] + (t3 - 2 * t2 + t) * ma[i] + (3 * t2 - 2 * t3) * b[i]
                   + (t3 - t2) * mb[i])
    out.append(p[-1:])
    return np.vstack(out)


def road_z(road, line: LineString, xy) -> np.ndarray:
    """The road's surface z where the points (N, 2) project onto its centre line `line`."""
    along = shapely.line_locate_point(line, shapely.points(np.asarray(xy, float).reshape(-1, 2)))
    return np.interp(along, _cum(road.nodes), road.nodes[:, 2])


def _true_runs(mask) -> list[slice]:
    edges = np.flatnonzero(np.diff(np.r_[0, np.asarray(mask, np.int8), 0]))
    return [slice(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def _rounded(points) -> list[list[float]]:
    return [[round(float(v), 3) + 0.0 for v in p] for p in points]


# ------------------------------------------------------------------ barriers

def _on_level(line: np.ndarray, cum: np.ndarray, s0: float, s1: float, road, centre: LineString) -> list:
    """The parts of the stretch s0..s1 of the barrier line where `road` is on its level, as spans
    (s0, s1, None), each reaching out to the first point read past it."""
    s = np.unique(np.r_[np.arange(s0, s1, DZ_STEP), s1])
    xy = np.column_stack([np.interp(s, cum, line[:, 0]), np.interp(s, cum, line[:, 1])])
    same = np.abs(np.interp(s, cum, line[:, 2]) - road_z(road, centre, xy)) < SAME_LEVEL
    return [(s[run.start - 1] if run.start else s0, s[run.stop] if run.stop < len(s) else s1, None)
            for run in _true_runs(same)]


def _cut(line: np.ndarray, deck, others, centres, areas, tree, trim=(True, True), exempt=()) -> list[np.ndarray]:
    """The barrier line's pieces once END_TRIM is off its ends (the ones `trim` says, start and end)
    and the stretches on another road's level inside its corridor are out, but for the corridor of a
    deck in `exempt` [(deck, (x, y))] within JOINT_REACH of that straight joint (joint_decks)."""
    line = _dedupe(line)
    if len(line) < 2:
        return []
    cum = _cum(line)
    ends = ((0.0, END_TRIM, None), (cum[-1] - END_TRIM, cum[-1], None))
    spans = [span for span, cut in zip(ends, trim) if cut]
    geom = LineString(line[:, :2])
    for j in (tree.query(geom, predicate="intersects") if tree is not None else []):
        road = others[int(j)]
        if road is deck:
            continue
        area = areas[int(j)]
        near = [Point(p).buffer(JOINT_REACH) for other, p in exempt if other is road]
        if near:
            area = area.difference(shapely.union_all(near))
        for part in _line_parts(geom.intersection(area)):
            s0, s1 = sorted(geom.project(Point(c)) for c in (part.coords[0], part.coords[-1]))
            spans += _on_level(line, cum, s0, s1, road, centres[int(j)])
    return [piece for piece, _, _ in _between(line, cum, spans)]


def _offset(d) -> float:
    """The barriers' centre lines' distance from the deck's."""
    return d.width / 2 + NARROW_OUT if d.width < NARROW_DECK else d.width / 2 - INSET


def _swap(side: str) -> str:
    return "rl"[side == "r"]


def _outer(d, e: int, ahead: np.ndarray, side: str) -> float:
    """How far out on `side` ("l" or "r") of the travel `ahead` (a unit vector, through the joint at
    the deck's end e) the deck's barrier line on that side lies JOINT_REACH from the joint (or at its
    far end, if shorter), the deck running away from the joint or into it."""
    xy = d.nodes[:, :2] if e == 0 else d.nodes[::-1, :2]
    cum = _cum(xy)
    p = np.array([np.interp(min(JOINT_REACH, cum[-1]), cum, xy[:, k]) for k in (0, 1)]) - xy[0]
    along = p / max(math.hypot(*p), 1e-12)
    if along @ ahead < 0:                                                # it runs into the joint
        along = -along
    sign = 1.0 if side == "r" else -1.0                                  # right: clockwise of the travel
    return float((p + sign * _offset(d) * np.array([along[1], -along[0]])) @ (sign * np.array([ahead[1], -ahead[0]])))


def straight_joints(decks) -> dict[tuple[int, int, str], tuple[int, int, str]]:
    """{(deck, end, side): (deck, end, side)}, both ways round, for each barrier side joined at a
    straight joint between the decks (an index into decks, its end, 0 or -1, and its own side, "l" or
    "r"); see the module docstring."""
    ends = [(i, e) for i, d in enumerate(decks) if _usable(d) for e in (0, -1)]
    if not ends:
        return {}
    xyz = np.array([decks[i].nodes[e] for i, e in ends], float)
    into = [_leaving(decks[i], e == 0) for i, e in ends]               # from the joint into the deck
    pts = shapely.points(xyz[:, :2])
    partners: dict[int, list[int]] = defaultdict(list)                 # end -> the ends straight on from it
    near = shapely.STRtree(pts).query(pts, predicate="dwithin", distance=JOINT_TOL)
    for a, b in zip(*(k.tolist() for k in near)):
        if (a != b and ends[a][0] != ends[b][0] and abs(xyz[a, 2] - xyz[b, 2]) < SAME_LEVEL
                and -float(into[a] @ into[b]) >= STRAIGHT_COS):        # cos of the turn through the joint
            partners[a].append(b)

    def outermost(a: int, ahead: np.ndarray, side: str) -> int:
        """Of the ends straight on from end a, the one whose barrier is furthest out on that side."""
        return max(sorted(partners[a]), key=lambda c: _outer(decks[ends[c][0]], ends[c][1], ahead, side))

    out: dict[tuple[int, int, str], tuple[int, int, str]] = {}
    for a in sorted(partners):
        (i, e), ahead = ends[a], -into[a]                               # travelling out of deck i into the joint
        for side in "lr":                                               # of the travel
            b = outermost(a, ahead, side)
            if b < a or outermost(b, into[b], side) != a:
                continue                                                # once per pair, outermost both ways
            j, f = ends[b]
            mine, theirs = (side if e == -1 else _swap(side)), (side if f == 0 else _swap(side))
            out[(i, e, mine)], out[(j, f, theirs)] = (j, f, theirs), (i, e, mine)
    return out


def _joint_ends(decks, i: int, e: int) -> list[tuple[int, int]]:
    """The other decks' ends (deck, end) at deck i's end e: within JOINT_TOL of it, on its level."""
    p = decks[i].nodes[e]
    return [(k, g) for k, d in enumerate(decks) if k != i and _usable(d) for g in (0, -1)
            if math.dist(d.nodes[g, :2], p[:2]) <= JOINT_TOL and abs(d.nodes[g, 2] - p[2]) < SAME_LEVEL]


def joint_decks(decks, joints=None) -> dict[int, list[tuple[int, np.ndarray, str]]]:
    """{deck: [(other deck, the joint's (x, y), side)]}: at each of the deck's barrier sides joined at a
    straight joint (straight_joints(decks) unless given), the decks whose corridors leave that barrier
    be within JOINT_REACH of the joint: the deck joined there, and any other deck ending there whose
    barrier on that side lies no further out (_outer) than the joined pair's on its own side of the
    joint (a deck forking off on the far side, or a ramp inside the deck carrying straight on)."""
    joints = straight_joints(decks) if joints is None else joints
    out: dict[int, list] = defaultdict(list)
    for (i, e, side), (j, f, _) in sorted(joints.items()):
        d = decks[i]
        p = d.nodes[e, :2]
        ahead = -_leaving(d, e == 0)                                    # travelling out of deck i into the joint
        travel = side if e == -1 else _swap(side)                       # the side of that travel
        out[i].append((j, p, side))
        mine, theirs = _outer(d, e, ahead, travel), _outer(decks[j], f, ahead, travel)
        for k, g in _joint_ends(decks, i, e):
            if k != j:
                beyond = theirs if _leaving(decks[k], g == 0) @ ahead > 0 else mine   # on j's side of the joint, or i's
                if _outer(decks[k], g, ahead, travel) <= beyond:
                    out[i].append((k, p, side))
    return dict(out)


def _tapered(curve: np.ndarray, offset: float, start: float, end: float) -> tuple[np.ndarray, np.ndarray]:
    """The curve with a vertex added where a taper begins, and the barriers' offset at each vertex:
    `offset`, coming in to `start` at the curve's start and `end` at its end over TAPER m per m."""
    cum = _cum(curve)
    begins = [s for s in ((offset - start) * TAPER, cum[-1] - (offset - end) * TAPER) if 0.0 < s < cum[-1]]
    if begins:
        s = np.union1d(cum, begins)
        curve, cum = np.column_stack([np.interp(s, cum, curve[:, k]) for k in range(3)]), s
    return curve, np.minimum.reduce([np.full(len(cum), offset), start + cum / TAPER, end + (cum[-1] - cum) / TAPER])


def _meet(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One side's barrier lines at a straight joint, a ending there and b starting there, made to meet:
    cut back to where they cross within JOINT_REACH of their ends (on the inside of a bend), or else a
    carried on to b's start (over the wedge between the slabs' ends on the outside of a bend)."""
    ca, cb = _cum(a), _cum(b)
    a0 = max(0.0, ca[-1] - JOINT_REACH)
    tail = LineString(_sub(a, ca, a0, ca[-1])[:, :2])
    head = LineString(_sub(b, cb, 0.0, min(JOINT_REACH, cb[-1]))[:, :2])
    hits = shapely.get_coordinates(tail.intersection(head))
    if not len(hits):                       # (a's end so close to b's start would be deduped onto it)
        return np.vstack([a[:-1] if math.dist(a[-1, :2], b[0, :2]) < DUP_TOL else a, b[:1]]), b
    at = Point(hits[np.argmin(np.hypot(*(hits - (a[-1, :2] + b[0, :2]) / 2).T))])
    return _sub(a, ca, 0.0, a0 + tail.project(at)), _sub(b, cb, head.project(at), cb[-1])


def barriers(decks, roads) -> list[dict]:
    """The barriers along the decks' edges; see the module docstring."""
    others = [r for r in roads if _usable(r)]
    centres = [LineString(r.nodes[:, :2]) for r in others]
    areas = [c.buffer(r.width / 2 + CUT_TOL) for c, r in zip(centres, others)]
    tree = shapely.STRtree(areas) if areas else None
    decks = [d for d in decks if _usable(d)]
    joints = straight_joints(decks)
    offsets = [_offset(d) for d in decks]
    lines = {}
    for i, d in enumerate(decks):
        drawn = deck_curve(d.nodes)
        for side, sign in (("l", -1.0), ("r", 1.0)):                # offset_polyline: positive is right
            start, end = (min(offsets[i], offsets[joints[(i, e, side)][0]]) if (i, e, side) in joints
                          else offsets[i] for e in (0, -1))
            curve, offset = _tapered(drawn, offsets[i], start, end)
            lines[(i, side)] = np.column_stack([offset_polyline(curve[:, :2], sign * offset), curve[:, 2]])
    for (i, e, side), (j, f, other) in sorted(joints.items()):
        if (i, e, side) > (j, f, other):
            continue                                                # each joined side once
        a, b = lines[(i, side)], lines[(j, other)]
        a, b = _meet(a if e == -1 else a[::-1], b if f == 0 else b[::-1])
        lines[(i, side)], lines[(j, other)] = (a if e == -1 else a[::-1]), (b if f == 0 else b[::-1])
    near = joint_decks(decks, joints)
    out = []
    for i, d in enumerate(decks):
        for side in "lr":
            trim = tuple((i, e, side) not in joints for e in (0, -1))
            exempt = [(decks[k], p) for k, p, sides in near.get(i, []) if side in sides]
            out += [{"deck": deck_id(d), "side": side, "points": _rounded(piece)}
                    for piece in _cut(lines[(i, side)], d, others, centres, areas, tree, trim, exempt)]
    return out


# ------------------------------------------------------------------ piers

def pier_footprint(pier: dict) -> Polygon:
    """A pier's rectangle: PIER_THICK along its yaw, its width across it, counter-clockwise."""
    c, s = math.cos(pier["yaw"]), math.sin(pier["yaw"])
    t, w = PIER_THICK / 2, pier["width"] / 2
    return Polygon([(pier["x"] + du * c - dv * s, pier["y"] + du * s + dv * c)
                    for du, dv in ((-t, -w), (t, -w), (t, w), (-t, w))])


def piers(decks, roads, heights: np.ndarray, extent: Extent, spans) -> list[dict]:
    """The piers under the decks; see the module docstring."""
    others = [r for r in roads if _usable(r)]
    centres = [LineString(r.nodes[:, :2]) for r in others]
    areas = [c.buffer(r.width / 2 + PIER_ROAD_CLEAR) for c, r in zip(centres, others)]
    tree = shapely.STRtree(areas) if areas else None
    spans = [g for g in spans if not g.is_empty]
    spans = shapely.union_all(spans) if spans else None
    out = []
    for d in decks:
        if not _usable(d):
            continue
        curve = deck_curve(d.nodes)
        cum = _cum(curve)
        width = round(min(PIER_SHARE * d.width, PIER_MAX_WIDTH), 3)
        for s in np.arange(PIER_SPACING, cum[-1] - 1e-6, PIER_SPACING):
            k = int(np.clip(np.searchsorted(cum, s) - 1, 0, len(curve) - 2))
            dx, dy = curve[k + 1, :2] - curve[k, :2]
            x, y, z = (float(np.interp(s, cum, curve[:, i])) for i in range(3))
            pier = {"deck": deck_id(d), "x": round(x, 3) + 0.0, "y": round(y, 3) + 0.0,
                    "yaw": round(math.atan2(dy, dx), 6) + 0.0, "width": width}
            foot = pier_footprint(pier)
            corners = np.asarray(foot.exterior.coords)[:4]
            ground = np.atleast_1d(sample_height(heights, extent, np.r_[corners[:, 0], x], np.r_[corners[:, 1], y]))
            z_top = z - DECK_DEPTH
            if z_top - float(ground.max()) < PIER_MIN_GAP:
                continue
            if spans is not None and shapely.dwithin(spans, foot, SPAN_PAD):
                continue
            hits = [int(j) for j in tree.query(foot, predicate="intersects")] if tree is not None else []
            if any(others[j] is not d and road_z(others[j], centres[j], [(x, y)])[0] < z_top for j in hits):
                continue
            out.append({**pier, "z_top": round(z_top, 3) + 0.0,
                        "z_bottom": round(float(ground.min()) - PIER_SINK, 3) + 0.0})
    return out


# ------------------------------------------------------------------ the stage

def _length(points) -> float:
    xy = np.asarray(points, float).reshape(-1, 3)[:, :2]
    return float(np.hypot(*np.diff(xy, axis=0).T).sum())


def missing_spans(registry: dict, over, decks) -> dict[str, list[str]]:
    """{name: its decks among `decks`} for every registry landmark over decks (params.decks) that has
    a deck among them but is none of `over`, the landmarks.json entries piers() keeps off."""
    here = {f"w{int(i)}" for d in decks for i in d.osm_ids}
    have = {lm.get("name") for lm in over}
    out = {}
    for name, e in registry.items():
        ways = [w for w in map(str, (e.get("params") or {}).get("decks") or []) if w in here]
        if ways and name not in have:
            out[name] = ways
    return out


def run(cfg, raw_dir=None, out_dir=None) -> None:
    """The stage (see the module docstring); it reads only out/ (and the landmark registry), so
    raw_dir goes unused."""
    out_dir = Path(out_dir or cfg.out_dir)
    path = out_dir / DRESSING_FILE
    if not cfg.deck_dressing:
        save_dressing({"barriers": [], "piers": []}, path)
        log.info("deck dressing is off (bridges.dressing in the config): %s lists none", path)
        return
    roads = load_roads(out_dir / "roads.json")
    decks = [r for r in roads if r.is_bridge]
    found = out_dir / landmarks.LANDMARKS_FILE
    over = [lm for lm in (landmarks.load_landmarks(found) if found.exists() else [])
            if (lm.get("params") or {}).get("decks") and len(lm["footprint"]) >= 3]
    spans = [Polygon(lm["footprint"]) for lm in over]
    missing = missing_spans(landmarks.load_registry(), over, decks)
    if missing:
        log.warning("%s lacks %s: run the landmarks stage before dressing, or piers stand under %s", found,
                    "; ".join(f"{name} (over decks {', '.join(ways)} of roads.json)" for name, ways in missing.items()),
                    "it" if len(missing) == 1 else "them")
    data = {"barriers": barriers(decks, roads),
            "piers": piers(decks, roads, load_heights(out_dir=out_dir), Extent.from_config(cfg), spans)}
    save_dressing(data, path)
    log.info("%d decks (%.1f km): %d barriers, %.1f km; %d piers (none within %g m of %d landmarks over decks)",
             len(decks), sum(_length(r.nodes) for r in decks) / 1000, len(data["barriers"]),
             sum(_length(b["points"]) for b in data["barriers"]) / 1000, len(data["piers"]), SPAN_PAD, len(spans))


# ------------------------------------------------------------------ meshes

def _frames(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit (along, right) vectors in plan at each point of a polyline: along averages the
    segments either side, right is along turned clockwise."""
    d = np.diff(points[:, :2], axis=0)
    d /= np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-12)[:, None]
    along = np.vstack([d[:1], d[:-1] + d[1:], d[-1:]])
    along /= np.maximum(np.hypot(along[:, 0], along[:, 1]), 1e-12)[:, None]
    return along, np.column_stack([along[:, 1], -along[:, 0]])


def _cap(mesh: dae._Mesh, at: np.ndarray, right: np.ndarray, facing: np.ndarray, profile, origin) -> None:
    """A barrier's end: the profile's polygon at point `at`, facing `facing` (plan), in quads
    pairing its left and right sides."""
    prof = np.asarray(profile, float)
    pos = np.column_stack([at[0] - origin[0] + prof[:, 0] * right[0], at[1] - origin[1] + prof[:, 0] * right[1],
                           at[2] + prof[:, 1]])
    n = len(prof)
    tris = np.array([t for j in range(n // 2 - 1) for t in ((j, j + 1, n - 2 - j), (j, n - 2 - j, n - 1 - j))])
    face = np.cross(pos[tris[:, 1]] - pos[tris[:, 0]], pos[tris[:, 2]] - pos[tris[:, 0]])
    want = np.array([facing[0], facing[1], 0.0])
    tris[face @ want < 0] = tris[face @ want < 0][:, ::-1]
    mesh.add(pos, np.tile(want, (n, 1)), prof / TEX_M, np.tile(WHITE, (n, 1)), tris, np.zeros(len(tris), int))


def _sweep(mesh: dae._Mesh, points: np.ndarray, along: np.ndarray, right: np.ndarray, profile, origin,
           start: bool, end: bool) -> None:
    """The profile swept along the points (a strip of quads per profile edge, hard edges between
    them), capped at the start and the end when asked."""
    prof = np.asarray(profile, float)
    n = len(points)
    s = _cum(points) / TEX_M
    rel = points - [origin[0], origin[1], 0.0]
    edge = np.r_[0.0, np.cumsum(np.hypot(*np.diff(prof, axis=0).T))] / TEX_M
    quad = np.array([[0, 1, 3], [0, 3, 2]])                  # station i: (a, b), station i + 1: (a', b')
    for q in range(len(prof) - 1):
        (a0, b0), (a1, b1) = prof[q], prof[q + 1]
        d = np.array([a1 - a0, b1 - b0]) / math.hypot(a1 - a0, b1 - b0)
        n_across, n_up = -d[1], d[0]                            # the outward normal of an edge walked clockwise
        pos = np.empty((n, 2, 3))
        for k, (a, b) in enumerate(((a0, b0), (a1, b1))):
            pos[:, k, :2] = rel[:, :2] + a * right
            pos[:, k, 2] = rel[:, 2] + b
        nrm = np.column_stack([n_across * right, np.full(n, n_up)])
        uv = np.stack([np.column_stack([s, np.full(n, edge[q])]), np.column_stack([s, np.full(n, edge[q + 1])])],
                      axis=1)
        tris = (np.arange(n - 1)[:, None, None] * 2 + quad).reshape(-1, 3)
        mesh.add(pos.reshape(-1, 3), np.repeat(nrm, 2, axis=0), uv.reshape(-1, 2), np.tile(WHITE, (2 * n, 1)), tris,
                 np.zeros(len(tris), int))
    if start:
        _cap(mesh, points[0], right[0], -along[0], profile, origin)
    if end:
        _cap(mesh, points[-1], right[-1], along[-1], profile, origin)


def _pier(mesh: dae._Mesh, pier: dict, origin) -> None:
    """A pier's four sides, from z_bottom to z_top, facing out."""
    corners = np.asarray(pier_footprint(pier).exterior.coords)[:4] - origin
    z0, z1 = pier["z_bottom"], pier["z_top"]
    for k in range(4):
        a, b = corners[k], corners[(k + 1) % 4]
        d = b - a
        width = math.hypot(*d)
        pos = np.array([[*a, z0], [*b, z0], [*b, z1], [*a, z1]])
        uv = np.array([[0.0, z0], [width, z0], [width, z1], [0.0, z1]]) / TEX_M
        nrm = np.tile([d[1] / width, -d[0] / width, 0.0], (4, 1))
        mesh.add(pos, nrm, uv, np.tile(WHITE, (4, 1)), [[0, 1, 2], [0, 2, 3]], [0, 0])


def _meshes(segs, piers_here, bars, frames, origin) -> dict:
    meshes = {node: dae._Mesh() for node in NODES}
    for m in meshes.values():
        m.code(MATERIAL)
    by_bar = defaultdict(list)
    for b, k, _, _ in segs:
        by_bar[b].append(k)
    for b, ks in sorted(by_bar.items()):
        ks = np.sort(ks)
        for run in np.split(ks, np.flatnonzero(np.diff(ks) > 1) + 1):
            k0, k1 = int(run[0]), int(run[-1]) + 1                 # the run's first and last point
            pts, (along, right) = bars[b][k0:k1 + 1], (f[k0:k1 + 1] for f in frames[b])
            start, end = k0 == 0, k1 == len(bars[b]) - 1
            _sweep(meshes[FULL], pts, along, right, JERSEY, origin, start, end)
            _sweep(meshes[COLLISION], pts, along, right, TRAPEZOID, origin, start, end)
    for p in piers_here:
        for node in NODES:
            _pier(meshes[node], p, origin)
    return {node: (m.to_mesh() if m.n else None) for node, m in meshes.items()}


def _split(key: str, origin, size: float, segs, piers_here, bars, frames, max_verts: int) -> list[tuple]:
    meshes = _meshes(segs, piers_here, bars, frames, origin)
    most = max(len(m.pos) for m in meshes.values() if m is not None)
    if most <= max_verts or size / 2 < MIN_TILE or len(segs) + len(piers_here) <= 1:
        if most > max_verts:
            log.warning("dressing tile %s keeps %d vertices in one level of detail (cap %d)", key, most, max_verts)
        return [(key, origin, meshes)]
    out = []
    for name, sx, sy in QUARTERS:
        def here(x, y):
            return (x >= origin[0]) == (sx > 0) and (y >= origin[1]) == (sy > 0)
        sub_segs = [g for g in segs if here(g[2], g[3])]
        sub_piers = [p for p in piers_here if here(p["x"], p["y"])]
        if sub_segs or sub_piers:
            quarter = (origin[0] + sx * size / 4, origin[1] + sy * size / 4)
            out += _split(f"{key}_{name}", quarter, size / 2, sub_segs, sub_piers, bars, frames, max_verts)
    return out


def tiles(data: dict, tile_m: float = TILE_M) -> list[tuple]:
    """(key, origin, {node: Mesh or None}) per tile; see the module docstring."""
    bars = [np.asarray(b["points"], float).reshape(-1, 3) for b in data["barriers"]]
    bars = [b for b in bars if len(b) >= 2]
    frames = [_frames(b) for b in bars]
    cells: dict[tuple[int, int], tuple[list, list]] = defaultdict(lambda: ([], []))
    for b, pts in enumerate(bars):
        mid = (pts[:-1, :2] + pts[1:, :2]) / 2
        for k, ((mx, my), (ix, iy)) in enumerate(zip(mid.tolist(), np.floor(mid / tile_m).astype(int).tolist())):
            cells[(ix, iy)][0].append((b, k, mx, my))
    for p in data["piers"]:
        cells[(math.floor(p["x"] / tile_m), math.floor(p["y"] / tile_m))][1].append(p)
    out = []
    for ix, iy in sorted(cells):
        segs, piers_here = cells[(ix, iy)]
        origin = ((ix + 0.5) * tile_m, (iy + 0.5) * tile_m)
        out += _split(_tile_key(ix, iy), origin, tile_m, segs, piers_here, bars, frames, MAX_VERTS)
    return out


# ------------------------------------------------------------------ level files

def ts_static(level_name: str, key: str, origin) -> dict:
    """The TSStatic placing one tile's shape at the tile's centre, unrotated."""
    return {"class": "TSStatic", "persistentId": bf.persistent_id("dr", key),
            "shapeName": f"/levels/{level_name}/{SHAPES_DIR}/br_dr_{key}.dae",
            "position": [round(float(origin[0]), 2), round(float(origin[1]), 2), 0.0],
            "useInstanceRenderData": True, "collisionType": "Collision Mesh", "decalType": "None"}


def level_content(cfg, data_dir, root, heights, extent) -> dict[str, list[dict]]:
    """The level.FEATURE_MODULES hook; see the module docstring. heights and extent are not needed:
    run set every z."""
    path = Path(data_dir) / DRESSING_FILE
    if not cfg.deck_dressing or not path.exists():
        return {}
    data = load_dressing(path)
    if not data["barriers"] and not data["piers"]:
        return {}
    folder = Path(root) / SHAPES_DIR
    folder.mkdir(parents=True, exist_ok=True)
    objs, most, size = [], dict.fromkeys(NODES, 0), 0
    for key, origin, meshes in tiles(data):
        dest = folder / f"br_dr_{key}.dae"
        dest.write_text(collada.shape_dae([(node, meshes[node]) for node in NODES], decimals=collada.TILE),
                        encoding="utf-8")
        size += dest.stat().st_size
        most = {node: max(most[node], len(meshes[node].pos) if meshes[node] is not None else 0) for node in NODES}
        objs.append(ts_static(cfg.level_name, key, origin))
    log.info("dressing: %d barriers and %d piers in %d tiles (%.1f MB of .dae); most vertices in a tile: %s",
             len(data["barriers"]), len(data["piers"]), len(objs), size / 1e6,
             ", ".join(f"{k} {v}" for k, v in most.items()))
    return {GROUP: objs}
