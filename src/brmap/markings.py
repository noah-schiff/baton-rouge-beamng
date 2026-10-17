"""Stage: US lane markings on the major roads: lane lines, junction trims, stop bars and crosswalks.

Reads out/roads.json (Road: lanes, oneway, width, kind, is_bridge, node z; one-way
roads' nodes run with the traffic), raw/osm_features.json "nodes" (signals, stops and
crossings, from features.py) and the terrain (load_heights, sample_height).

out/markings.json: [{"name", "kind", "material", "width", "priority", "texture_length",
"over_objects", "nodes": [[x, y, z], ...]}] (Marking.to_dict), name unique and stable (it
seeds the persistentId). z comes from load_heights on the ground, or from the deck with
over_objects true on bridges. With cfg.markings_enabled false, run writes an empty
list, so a file left by an earlier build is not painted.

* MARKED_KINDS: motorway, trunk, primary, secondary, tertiary and their _links; the
  other roads get no paint.
* lane_lines(road) -> [(kind, offset_m, material, width)], positive offset = right of
  the travel direction. lane_w = (road.width - shoulder) / lanes (shoulder as in
  roads.road_width: 3.0 for motorway/trunk and their links, else 0.5); band = lanes * lane_w.
  - Two-way: centre_double at 0 (0.4 m) and divider at +-k * lane_w for
    k = 1 .. lanes_per_dir - 1 (lanes_per_dir = max(1, lanes // 2)), or at
    +-(k + 0.5) * lane_w for an odd lane count, whose middle lane is a centre turn lane
    (so the dividers stay on lane edges); trunk and primary also edge_white at
    +-(band / 2 - 0.2).
  - One-way: divider at -band / 2 + k * lane_w; motorway, trunk and their links also
    edge_yellow at -band / 2 + 0.2 and edge_white at +band / 2 - 0.2.
* offset_polyline(nodes_xy, d): per-node offset along averaged segment normals, same
  node count and order (shapely offset_curve lines failed to show in game).
* junctions(roads) -> {marked road index: [(x, y, radius), ...]}: ends shared by 3 or
  more roads (on a 0.5 m grid), and crossings or touches with any drivable road except
  service roads under 5 m wide, on the same level (|dz| < 2 m, neither a bridge over the
  other); radius = the other road's width / 2 + 2 (at a shared end: the widest road
  crossing there, not the road's own straight-on continuation). Two roads end to end are
  not a junction, and neither is a road ending on another mid-way alongside it (within 30
  degrees: a ramp merging or forking), through which the other's paint runs. Kept per
  road, because the radius depends on which road is trimmed and a deck passing over a
  junction must not be trimmed by it.
* carriageways(roads) and keep_off(lines, others): where two marked carriageways overlap on
  the same level (a ramp merging or forking, two carriageways converging), the paint of the
  lower-ranked road stops at the other's edge, each line on its own; through roads rank over
  links, then by class; equal ranks both give way. Roads carrying straight on from each
  other's ends don't cut each other.
* trim(line, junctions) -> [ndarray]: the line without the parts within each radius;
  pieces under 2 m are dropped. Lines are trimmed as the centreline and then offset, so
  every line of a road stops at the same cross-section.
* stop_bars(roads, junctions, nodes), junctions being the junctions() mapping: one 0.5 m
  bar across the approach lanes at the trimmed line end of every approach to a junction
  with a traffic signal within 20 m, or with a stop node within 20 m that lies on that
  approach (stop=all: every approach; a stop on the cross street does not stop the main
  road). Only where the lines next to the junction survive the trims.
* crosswalks(roads, nodes): highway=crossing nodes with crossing in {marked, zebra,
  traffic_signals}, or crossing:markings present and not "no" (which wins over crossing):
  a 2-node 3 m wide decal across the nearest road (within 3 m) if it is marked,
  perpendicular, curb to curb.
* In both, nodes are the OSM nodes in level coordinates ({"id", "x", "y", "tags"}, from
  level_nodes).
* deck_line(nodes, deck): a lane line on a bridge deck moved off the deck MeshRoad's seams:
  its vertices where the cross-sections at 1/3 and 2/3 of each deck segment cut it, its
  ends inset 0.5 m, every vertex at least 0.25 m along the line from the cross-sections at
  the deck's nodes and segment middles, z the deck's; None when under 2 m once inset. Ground
  lines keep their nodes, and deck stop bars and crosswalks are left as they are.
* run(cfg, raw_dir=None, out_dir=None) writes out/markings.json, clipped to the box (paint
  shorter than 2 m after that is dropped; deck lines then go through deck_line), and logs counts.
* level_content (the level.FEATURE_MODULES hook) writes art/road/br_markings.materials.json
  with the materials used and returns {"markings": [DecalRoad, ...]} (see decal_road), or
  {} when there is no paint.
"""
from __future__ import annotations

import copy
import json
import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from shapely import STRtree
from shapely.geometry import LineString, Point

from . import beamng_format as bf
from .geo import Extent
from .roads import MAIN_CLASSES, SHOULDER, base_kind, load_roads
from .terrain import load_heights, sample_height

log = logging.getLogger(__name__)

OUT_DIR = Path("data/out")
GROUP = "markings"
MATERIALS_FILE = "art/road/br_markings.materials.json"

MARKED_KINDS = frozenset(MAIN_CLASSES) | frozenset(f"{k}_link" for k in MAIN_CLASSES)
HIGHWAY = ("motorway", "trunk")      # base kinds with edge lines when one-way
EDGE_TWO_WAY = ("trunk", "primary")  # two-way kinds with white edge lines
DEFAULT_SHOULDER = 0.5               # m, the shoulder of the kinds not in roads.SHOULDER
LINE_KINDS = ("centre_double", "centre_turn", "divider", "edge_white", "edge_yellow")
MARKED_CROSSINGS = ("marked", "zebra", "traffic_signals")

CENTRE_W = 0.4     # m, the double yellow (the stock levels use 0.4)
LINE_W = 0.15      # m, single lines; the line textures paint about 2/3 of the decal width
EDGE_INSET = 0.2   # m, edge lines sit this far inside the lane band
BAR_W = 0.5        # m along the road
CROSSWALK_W = 3.0  # m along the road
MIN_LINE = 2.0     # m; shorter lines are not painted
MIN_ACROSS = 0.5   # m; shorter stop bars and crosswalks (cut by the box edge) are not painted
LINE_PRIORITY = 2  # lower draws on top: paint goes over every road (bf.RENDER_PRIORITY 6-12)
MARK_PRIORITY = 1  # stop bars and crosswalks over the lines
TEX_DASHED = 12.0  # m per texture repeat: about 6 m dash and 6 m gap
TEX_SOLID = 6.0
TEX_CROSSWALK = 3.0  # four bars per repeat

JUNCTION_MARGIN = 2.0  # m; trim radius = the other road's width / 2 + this
NARROW_SERVICE = 5.0   # m; narrower service roads (driveways) make no junctions
SAME_LEVEL = 2.0       # m; |dz| under this where two roads meet = the same level
TOUCH_TOL = 1.0        # m; an end this close to a centreline touches it (roads are simplified at 0.5 m)
END_TOL = 1.0          # m; a meeting point this close to a road's end is at that end
END_GRID = 0.5         # m; road ends are matched on this grid
LEAVE_LEN = 5.0        # m; a road's direction at an end is taken over this length
CONTINUE_COS = 0.866   # another road leaving a shared end within 30 degrees of straight on continues this one
MERGE_TOL = 1.0        # m; junction points on one road this close together are one
SIGNAL_REACH = 20.0    # m; a signal or stop node this close to a junction controls it
STOP_ON_ROAD = 1.0     # m beyond the half width: a stop node this close to a centreline is on that road
ON_JUNCTION = 1.0      # m along the road; a stop node this close to the junction can't tell its approach
CROSSWALK_SNAP = 3.0   # m; a crossing node this close to a centreline is on that road
MITER_MIN_COS = 0.5    # inside corners are mitred up to 2x (turns up to 120 degrees)
DUP_TOL = 0.05         # m, as roads.DUPLICATE_TOL
# Deck paint nodes go this far above the deck. overObjects projects the decal down onto
# the objects below, and a node right on the surface can miss it where the deck's spline
# runs above the chord; Road Architect lifts all its decal nodes 4.2 m
# (lua/ge/extensions/editor/tech/roadArchitect/decals.lua:23). At 0.5 m two deck lines
# on the debug map got infinite vertices in 0.39.4 ("packed position exceeded supported
# local range ... by inf m") where the spline dips under the deck's vertical curve. A deck
# above another clears it by 5.5 m less its 1.5 m thickness, so 2 m stays under it.
DECK_LIFT = 2.0
# On the full map 19 of 439 deck paint decals (and none of the ground ones) got infinite
# vertices in 0.39.4 with DECK_LIFT at 0.5 m and at 2 m. Deck paint used the deck's own
# node stations, so every one of its vertices sat on a seam of the deck MeshRoad; the
# working hypothesis (to be checked in game) is that this is the cause. So deck_line puts
# the vertices where DECK_STATIONS of each deck segment cut the line, never on a node or a
# segment's middle, insets the ends DECK_END_INSET (Road Architect keeps its paint 0.45 m
# short of the ends) and keeps them DECK_SEAM_GAP from the nodes' and middles' cross-sections.
DECK_STATIONS = (1 / 3, 2 / 3)
DECK_END_INSET = 0.5  # m along the line
DECK_SEAM_GAP = 0.25  # m along the line

# ------------------------------------------------------------------ materials

WHITE, YELLOW, YELLOW_DOUBLE = "br_line_white", "br_line_yellow", "br_line_yellow_double"
DASHED, CROSSWALK = "br_line_white_dashed", "br_crosswalk"
TURN = "br_line_yellow_turn"  # solid + dashed yellow: the edges of a two-way left-turn lane
LINES_DIR = "/assets/materials/decalroad/lines/"
_MAPS = {"b": ("baseColorMap", "_b.color.png"), "n": ("normalMap", "_nm.normal.png"),
         "o": ("opacityMap", "_o.data.png"), "r": ("roughnessMap", "_r.data.png"),
         "a": ("ambientOcclusionMap", "_ao.data.png")}
_WHITE_FACTOR = [0.992, 0.992, 0.992, 0.8]
_YELLOW_FACTOR = [0.94, 0.79, 0.27, 0.9]
# fields of the global m_line_* materials (content/art_shapes.zip,
# art/shapes/common/decalroads/main.materials.json, 0.39.4)
_LINE_FIELDS = {"alphaRef": 255, "annotation": "SOLID_LINE", "castShadows": False, "groundType": "ASPHALT",
                "materialTag0": "RoadAndPath", "materialTag1": "beamng", "specularStrength0": "0",
                "translucent": True, "translucentZWrite": True, "version": 1.5}
# name: (folder under lines/ in content/assets/materials/decalroad.zip, texture stem, the maps
# that folder has, stage fields, material fields), copied from m_line_white, m_line_yellow,
# m_line_yellow_double, m_line_white_discontinue and crossing_white. line_white has only the
# b, nm and o maps; the three crossing sets in the zip are the same texture, and
# m_marking_zebra is the one with a roughness map.
MATERIAL_DEFS = {
    WHITE: ("line_white", "t_line_white", "bno",
            {"baseColorFactor": _WHITE_FACTOR, "opacityFactor": 0.7}, {}),
    YELLOW: ("line_white", "t_line_white", "bno",
             {"baseColorFactor": _YELLOW_FACTOR, "opacityFactor": 0.7}, {}),
    YELLOW_DOUBLE: ("line_yellow_double", "t_line_yellow_double", "bnora",
                    {"baseColorFactor": _YELLOW_FACTOR, "opacityFactor": 0.7,
                     "specular": [0.992, 0.992, 0.992, 1.0]}, {"alphaRef": 0, "alphaTest": True}),
    DASHED: ("line_white_discontinue", "t_line_white_discontinue", "bnora",
             {"baseColorFactor": _WHITE_FACTOR, "opacityFactor": 0.7}, {}),
    TURN: ("line_yellow_double_discontinue", "t_line_yellow_double_discontinue", "bnora",
           {"baseColorFactor": _YELLOW_FACTOR, "opacityFactor": 0.7,
            "specular": [0.992, 0.992, 0.992, 1.0]},
           {"alphaRef": 0, "alphaTest": True, "annotation": "DASHED_LINE"}),
    CROSSWALK: ("m_marking_zebra", "t_line_crossing_white", "bnor", {"roughnessFactor": 0.292},
                {"annotation": "ZEBRA_CROSSING", "groundType": None}),
}


def _line_material(name: str, folder: str, stem: str, maps: str, stage: dict, fields: dict) -> dict:
    st = {key: f"{LINES_DIR}{folder}/{stem}{suffix}" for c, (key, suffix) in _MAPS.items() if c in maps}
    st.update(copy.deepcopy(stage))
    m = {"name": name, "mapTo": name, "class": "Material", "persistentId": bf.persistent_id("material", name),
         "Stages": [st, {}, {}, {}]}
    m.update((k, v) for k, v in {**_LINE_FIELDS, **fields}.items() if v is not None)
    return m


MATERIALS = {name: _line_material(name, *d) for name, d in MATERIAL_DEFS.items()}

# ------------------------------------------------------------------ records


@dataclass
class Marking:
    name: str
    kind: str            # a LINE_KINDS entry, "stop_bar" or "crosswalk"
    material: str        # a MATERIALS key
    width: float         # decal width, m
    priority: int        # renderPriority
    texture_length: float
    over_objects: bool   # on a bridge deck: nodes at the deck's z
    nodes: np.ndarray    # (N, 3) level x, y, z

    def to_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind, "material": self.material, "width": float(self.width),
                "priority": int(self.priority), "texture_length": float(self.texture_length),
                "over_objects": bool(self.over_objects),
                "nodes": [[round(float(v), 3) for v in p] for p in self.nodes]}

    @classmethod
    def from_dict(cls, d: dict) -> "Marking":
        return cls(name=d["name"], kind=d["kind"], material=d["material"], width=float(d["width"]),
                   priority=int(d["priority"]), texture_length=float(d["texture_length"]),
                   over_objects=bool(d["over_objects"]),
                   nodes=np.array(d["nodes"], dtype=float).reshape(-1, 3))


def save_markings(marks: list[Marking], path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([m.to_dict() for m in marks]), encoding="utf-8")
    return path


def load_markings(path=OUT_DIR / "markings.json") -> list[Marking]:
    return [Marking.from_dict(d) for d in json.loads(Path(path).read_text(encoding="utf-8"))]


# ------------------------------------------------------------------ lane lines

def _lanes(road) -> tuple[int, float]:
    """(lane count, lane width) of the painted band: the road width less its shoulders."""
    lanes = max(1, int(road.lanes))
    return lanes, (road.width - SHOULDER.get(base_kind(road.kind), DEFAULT_SHOULDER)) / lanes


def lane_lines(road) -> list[tuple[str, float, str, float]]:
    """(kind, offset m, material, width m) of each line along the road; positive offsets are
    right of the direction of travel (the node order). [] for roads that get no paint."""
    if road.kind not in MARKED_KINDS:
        return []
    lanes, lane_w = _lanes(road)
    if lane_w <= 0:
        return []
    half = lanes * lane_w / 2
    if road.oneway:
        out = [("divider", -half + k * lane_w, DASHED, LINE_W) for k in range(1, lanes)]
        if base_kind(road.kind) in HIGHWAY:
            out += [("edge_yellow", -half + EDGE_INSET, YELLOW, LINE_W),
                    ("edge_white", half - EDGE_INSET, WHITE, LINE_W)]
        return out
    if lanes == 1:  # a one-lane two-way road has no centre line
        return []
    centre = (lanes % 2) / 2  # an odd count has a centre (turn) lane: the lane edges are half a lane out
    if centre:
        out = [("centre_turn", -centre * lane_w, TURN, CENTRE_W), ("centre_turn", centre * lane_w, TURN, CENTRE_W)]
    else:
        out = [("centre_double", 0.0, YELLOW_DOUBLE, CENTRE_W)]
    for k in range(1, max(1, lanes // 2)):
        off = (k + centre) * lane_w
        out += [("divider", -off, DASHED, LINE_W), ("divider", off, DASHED, LINE_W)]
    if road.kind in EDGE_TWO_WAY:
        out += [("edge_white", -(half - EDGE_INSET), WHITE, LINE_W),
                ("edge_white", half - EDGE_INSET, WHITE, LINE_W)]
    return out


def offset_polyline(nodes_xy, d: float) -> np.ndarray:
    """The polyline moved d m to the right of its direction (left when negative), node by
    node along the averaged normals of the segments on either side, so it keeps the node
    count and order. On the inside of a bend the step is mitred (up to 2x), so the node
    stays d from both segments; on the outside it stays d from the corner."""
    xy = np.asarray(nodes_xy, dtype=float)[:, :2]
    if len(xy) < 2:
        return xy.copy()
    seg = np.diff(xy, axis=0)
    length = np.hypot(seg[:, 0], seg[:, 1])
    ok = length > 1e-9
    if not ok.any():
        return xy.copy()
    # zero-length segments (repeated nodes) borrow the direction of the last real one before them
    src = np.maximum.accumulate(np.where(ok, np.arange(len(seg)), -1))
    src[src < 0] = int(np.argmax(ok))
    u = seg[src] / length[src, None]
    before = np.vstack([u[:1], u])  # direction into each node (node 0: its first segment)
    after = np.vstack([u, u[-1:]])  # direction out of each node
    nb = np.column_stack([before[:, 1], -before[:, 0]])  # right normals
    na = np.column_stack([after[:, 1], -after[:, 0]])
    m = nb + na
    mlen = np.hypot(m[:, 0], m[:, 1])
    m = np.where(mlen[:, None] > 1e-9, m / np.maximum(mlen, 1e-12)[:, None], nb)  # a U-turn keeps nb
    cos_half = np.clip((m * nb).sum(axis=1), MITER_MIN_COS, 1.0)
    turn = before[:, 0] * after[:, 1] - before[:, 1] * after[:, 0]  # > 0: turning left
    inside = turn * d < 0  # the left side is inside a left turn
    return xy + m * (d * np.where(inside, 1.0 / cos_half, 1.0))[:, None]


# ------------------------------------------------------------------ polyline pieces

def _cum(xy: np.ndarray) -> np.ndarray:
    return np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy[:, :2], axis=0).T))]


def _dedupe(line: np.ndarray) -> np.ndarray:
    keep = [0]
    for i in range(1, len(line)):
        if math.dist(line[i, :2], line[keep[-1], :2]) >= DUP_TOL:
            keep.append(i)
    return line[keep]


def _at(line: np.ndarray, cum: np.ndarray, s: float) -> np.ndarray:
    return np.array([np.interp(s, cum, col) for col in line.T])


def _sub(line: np.ndarray, cum: np.ndarray, s0: float, s1: float) -> np.ndarray:
    """The stretch of line between arc lengths s0 and s1, every column interpolated."""
    inner = (cum > s0 + DUP_TOL) & (cum < s1 - DUP_TOL)
    return np.vstack([_at(line, cum, s0), line[inner], _at(line, cum, s1)])


def _circle_spans(xy: np.ndarray, cum: np.ndarray, x: float, y: float, r: float) -> np.ndarray:
    """(s0, s1) arc-length spans of the polyline within r of (x, y), one per segment."""
    a, d, length = xy[:-1], np.diff(xy, axis=0), np.diff(cum)
    u = d / np.maximum(length, 1e-12)[:, None]
    f = a - (x, y)
    b = (f * u).sum(axis=1)
    disc = b * b - ((f * f).sum(axis=1) - r * r)
    root = np.sqrt(np.maximum(disc, 0.0))
    t0, t1 = np.clip(-b - root, 0.0, length), np.clip(-b + root, 0.0, length)
    hit = (disc > 0) & (t1 > t0)
    return np.column_stack([cum[:-1][hit] + t0[hit], cum[:-1][hit] + t1[hit]])


def _cut(line, juncs) -> list[tuple[np.ndarray, int | None, int | None]]:
    """The pieces of a polyline (N x 2 or more columns) outside every junction circle and at
    least MIN_LINE long, each with the index into juncs of the junction that cut its start
    and its end (None at the line's own ends)."""
    line = _dedupe(np.asarray(line, dtype=float))
    if len(line) < 2:
        return []
    cum = _cum(line)
    return _between(line, cum, [(s0, s1, j) for j, (x, y, r) in enumerate(juncs)
                                for s0, s1 in _circle_spans(line[:, :2], cum, x, y, r)])


def _between(line: np.ndarray, cum: np.ndarray, spans) -> list[tuple[np.ndarray, object, object]]:
    """The pieces of a polyline outside the arc-length spans [(s0, s1, tag), ...] and at least
    MIN_LINE long, each with the tag of the span before and after it (None at the line's ends)."""
    gaps: list[list] = []  # merged spans: [s0, s1, tag at s0, tag at s1]
    for s0, s1, j in sorted(spans, key=lambda t: (t[0], t[1])):
        if gaps and s0 <= gaps[-1][1] + 1e-9:
            if s1 > gaps[-1][1]:
                gaps[-1][1], gaps[-1][3] = s1, j
        else:
            gaps.append([s0, s1, j, j])
    pieces, start, start_j = [], 0.0, None
    for s0, s1, j0, j1 in gaps + [[cum[-1], cum[-1], None, None]]:
        if s0 - start >= MIN_LINE:
            pieces.append((_sub(line, cum, start, s0), start_j, j0))
        start, start_j = s1, j1
    return pieces


def trim(line_xy, junctions) -> list[np.ndarray]:
    """The parts of a line outside every junction's radius ([(x, y, radius), ...]), in order;
    pieces shorter than 2 m are dropped. Extra columns (z) are interpolated along."""
    return [piece for piece, _, _ in _cut(line_xy, junctions)]


def _box_span(a: np.ndarray, b: np.ndarray, half: float):
    """(t0, t1) of the segment a -> b inside the square |x|, |y| <= half (Liang-Barsky), or None."""
    t0, t1 = 0.0, 1.0
    d = b - a
    for p, q in ((-d[0], a[0] + half), (d[0], half - a[0]), (-d[1], a[1] + half), (d[1], half - a[1])):
        if p == 0:
            if q < 0:
                return None
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
    return (t0, t1) if t0 < t1 else None


def _clip_box(line: np.ndarray, half: float) -> list[np.ndarray]:
    """The parts of a polyline inside the map square, in order."""
    if (np.abs(line[:, :2]) <= half).all():
        return [line]
    line = _dedupe(line)
    cum = _cum(line)
    spans: list[list[float]] = []
    for k in range(len(line) - 1):
        t = _box_span(line[k, :2], line[k + 1, :2], half)
        if t is None:
            continue
        s0, s1 = cum[k] + t[0] * (cum[k + 1] - cum[k]), cum[k] + t[1] * (cum[k + 1] - cum[k])
        if spans and s0 <= spans[-1][1] + 1e-9:
            spans[-1][1] = max(spans[-1][1], s1)
        else:
            spans.append([s0, s1])
    out = []
    for s0, s1 in spans:
        part = _sub(line, cum, s0, s1)
        part[:, :2] = np.clip(part[:, :2], -half, half)
        out.append(part)
    return out


# ------------------------------------------------------------------ junctions

def _makes_junctions(road) -> bool:
    """Every drivable road can make a junction, except narrow service roads (driveways)."""
    return len(road.nodes) >= 2 and not (road.kind == "service" and road.width < NARROW_SERVICE)


def _z_at(road, s: float) -> float:
    return float(np.interp(s, _cum(road.nodes), road.nodes[:, 2]))


def _points(g) -> list[tuple[float, float]]:
    if g.is_empty:
        return []
    if g.geom_type == "Point":
        return [(g.x, g.y)]
    if g.geom_type == "LineString":  # an overlapping stretch: where the two meet and part
        return [tuple(g.coords[0][:2]), tuple(g.coords[-1][:2])]
    return [p for sub in getattr(g, "geoms", []) for p in _points(sub)]


def _heading(line: LineString, s: float) -> np.ndarray:
    """Unit direction of a line at arc length s (over +-1 m)."""
    a = np.array(line.interpolate(max(s - 1.0, 0.0)).coords[0][:2])
    b = np.array(line.interpolate(min(s + 1.0, line.length)).coords[0][:2])
    n = math.hypot(*(b - a))
    return (b - a) / n if n > 0 else b - a


def _meetings(m, lm: LineString, r, lr: LineString) -> list[tuple[float, float]]:
    """Points where road m's centreline crosses or touches road r's on the same level, except
    where they only meet end to end (a continuation), and where r ends on m running alongside
    it (within 30 degrees): a ramp merging into m or forking off it, through which m's paint
    runs (keep_off keeps r's paint off m's lanes)."""
    pts = _points(lm.intersection(lr))
    for e in (lr.coords[0], lr.coords[-1]):  # r ends on m (simplified lines can miss by < 0.5 m)
        if lm.distance(Point(e)) <= TOUCH_TOL:
            pts.append(lm.interpolate(lm.project(Point(e))).coords[0][:2])
    for e in (lm.coords[0], lm.coords[-1]):  # m ends on r
        if lr.distance(Point(e)) <= TOUCH_TOL:
            pts.append(tuple(e[:2]))
    out = []
    for x, y in pts:
        p = Point(x, y)
        sm, sr = lm.project(p), lr.project(p)
        at_m_end = sm <= END_TOL or sm >= lm.length - END_TOL
        at_r_end = sr <= END_TOL or sr >= lr.length - END_TOL
        if at_m_end and at_r_end:
            continue  # end to end; three or more ends are found by _shared_ends
        if (at_r_end and not at_m_end
                and abs(float(_heading(lm, sm) @ _leaving(r, sr <= END_TOL))) >= CONTINUE_COS):
            continue  # r merges into m or forks off it
        if m.is_bridge != r.is_bridge and not (at_m_end if m.is_bridge else at_r_end):
            continue  # the bridge passes over the other road (it joins it only at its ends)
        if not abs(_z_at(m, sm) - _z_at(r, sr)) < SAME_LEVEL:
            continue
        out.append((float(x), float(y)))
    return out


def _end_key(x: float, y: float) -> tuple[int, int]:
    return round(x / END_GRID), round(y / END_GRID)


def _leaving(road, start: bool) -> np.ndarray:
    """Unit direction in which a road leaves its start (or its end), over its first LEAVE_LEN m."""
    xy = road.nodes[:, :2] if start else road.nodes[::-1, :2]
    cum = _cum(xy)
    d = np.array([np.interp(min(LEAVE_LEN, cum[-1]), cum, xy[:, k]) for k in (0, 1)]) - xy[0]
    n = math.hypot(*d)
    return d / n if n > 0 else d


def _shared_ends(roads, i: int, ends: dict) -> list[tuple[float, float, float]]:
    """Junctions at road i's ends where two or more other roads end too. The radius comes from
    the roads that cross there; a road carrying straight on is road i's own continuation."""
    out = []
    for start in (True, False):
        x, y, z = roads[i].nodes[0 if start else -1]
        mine = _leaving(roads[i], start)
        others: dict[int, list[np.ndarray]] = defaultdict(list)
        for j, zj, dj in ends[_end_key(x, y)]:
            if j != i and abs(zj - z) < SAME_LEVEL:
                others[j].append(dj)
        if len(others) < 2:
            continue
        # a road leaving straight on continues this one, and one leaving alongside it is a fork
        # (a gore); only the roads leaving at an angle cross it
        cross = [j for j, ds in others.items() if any(abs(float(mine @ d)) < CONTINUE_COS for d in ds)]
        if not cross:  # every other road carries straight on or forks off: paint runs through
            continue
        widest = max(roads[j].width for j in cross)
        out.append((float(x), float(y), widest / 2 + JUNCTION_MARGIN))
    return out


def _merge_points(found) -> list[tuple[float, float, float]]:
    out: list[tuple[float, float, float]] = []
    for x, y, r in found:
        for k, (ox, oy, orad) in enumerate(out):
            if math.hypot(x - ox, y - oy) <= MERGE_TOL:
                out[k] = (ox, oy, max(orad, float(r)))
                break
        else:
            out.append((float(x), float(y), float(r)))
    return out


def junctions(roads) -> dict[int, list[tuple[float, float, float]]]:
    """{index of each marked road: [(x, y, radius), ...]}: where its paint stops, with the
    radius its lines are trimmed by there (the other road's width / 2 + 2 m)."""
    lines = {j: LineString(r.nodes[:, :2]) for j, r in enumerate(roads) if _makes_junctions(r)}
    keys = list(lines)
    tree = STRtree([lines[j] for j in keys])
    ends = defaultdict(list)  # end on the 0.5 m grid -> [(road index, z, direction leaving it)]
    for j in keys:
        for start in (True, False):
            x, y, z = roads[j].nodes[0 if start else -1]
            ends[_end_key(x, y)].append((j, z, _leaving(roads[j], start)))
    out = {}
    for i in keys:
        m = roads[i]
        if m.kind not in MARKED_KINDS:
            continue
        found = []
        for k in tree.query(lines[i], predicate="dwithin", distance=TOUCH_TOL):
            j = keys[int(k)]
            if j != i:
                r = roads[j].width / 2 + JUNCTION_MARGIN
                found += [(x, y, r) for x, y in _meetings(m, lines[i], roads[j], lines[j])]
        out[i] = _merge_points(found + _shared_ends(roads, i, ends))
    return out


# ------------------------------------------------------------------ overlapping carriageways

def _rank(kind: str) -> tuple[bool, int]:
    """Whose paint wins where two carriageways overlap: through roads over links, then the
    classes in draw order (bf.RENDER_PRIORITY, lower on top). Equal ranks both give way."""
    return (not kind.endswith("_link"), -bf.RENDER_PRIORITY.get(base_kind(kind), bf.DEFAULT_RENDER_PRIORITY))


def _carriageway(road):
    return LineString(road.nodes[:, :2]).buffer(road.width / 2, cap_style="flat")


def carriageways(roads) -> dict[int, list[tuple]]:
    """{index of each marked road: [(road, centreline, carriageway), ...] of the marked roads
    whose lanes its paint keeps off}: those of at least its rank (_rank) whose carriageway
    overlaps its own (a ramp merging or forking, two carriageways converging), except the
    roads carrying straight on from its ends, which only meet it end to end."""
    idx = [j for j, r in enumerate(roads) if r.kind in MARKED_KINDS and len(r.nodes) >= 2]
    lines = [LineString(roads[j].nodes[:, :2]) for j in idx]
    areas = [_carriageway(roads[j]) for j in idx]
    ends = defaultdict(list)  # end on the 0.5 m grid -> [(position in idx, direction leaving it)]
    for k, j in enumerate(idx):
        for start in (True, False):
            x, y, _ = roads[j].nodes[0 if start else -1]
            ends[_end_key(x, y)].append((k, _leaving(roads[j], start)))
    straight = defaultdict(set)
    for there in ends.values():
        for a, da in there:
            straight[a].update(b for b, db in there if float(da @ db) < -CONTINUE_COS)
    tree = STRtree(areas)
    out = {}
    for k, i in enumerate(idx):
        rank = _rank(roads[i].kind)
        out[i] = [(roads[idx[q]], lines[q], areas[q])
                  for q in sorted(int(q) for q in tree.query(areas[k], predicate="intersects"))
                  if q != k and q not in straight[k] and _rank(roads[idx[q]].kind) >= rank]
    return out


def _line_parts(g) -> list[LineString]:
    if g.is_empty:
        return []
    if g.geom_type == "LineString":
        return [g]
    return [p for sub in getattr(g, "geoms", []) for p in _line_parts(sub)]


def keep_off(marks: list[Marking], others) -> list[Marking]:
    """One road's lines without the stretches on the lanes of `others` (from carriageways) at
    the same level, each line cut on its own; pieces under MIN_LINE are dropped, and a line
    cut in several keeps its name with "_o<k>" added."""
    if not others:
        return marks
    out = []
    for m in marks:
        nodes = _dedupe(m.nodes)
        if len(nodes) < 2:
            continue
        line, cum = LineString(nodes[:, :2]), _cum(nodes)
        spans = []
        for road, centre, area in others:
            for part in _line_parts(line.intersection(area)):
                s0, s1 = sorted(line.project(Point(c)) for c in (part.coords[0], part.coords[-1]))
                mid = (s0 + s1) / 2
                dz = np.interp(mid, cum, nodes[:, 2]) - _z_at(road, centre.project(line.interpolate(mid)))
                if abs(dz) < SAME_LEVEL:  # a deck above or below is another level
                    spans.append((s0, s1, None))
        if not spans:
            out.append(m)
            continue
        pieces = [piece for piece, _, _ in _between(nodes, cum, spans)]
        out += [replace(m, name=m.name if len(pieces) == 1 else f"{m.name}_o{k}", nodes=piece)
                for k, piece in enumerate(pieces)]
    return out


# ------------------------------------------------------------------ stop bars and crosswalks

def _tag(tags: dict, key: str) -> str | None:
    v = tags.get(key)
    return None if v is None else str(v).strip().lower()


def level_nodes(osm_nodes, extent: Extent) -> list[dict]:
    """osm_features.json nodes ({"id", "lon", "lat", "tags"}) as {"id", "x", "y", "tags"}."""
    if not osm_nodes:
        return []
    x, y = extent.to_level(np.array([n["lon"] for n in osm_nodes], dtype=float),
                           np.array([n["lat"] for n in osm_nodes], dtype=float))
    return [{"id": str(n["id"]), "x": float(xi), "y": float(yi), "tags": n.get("tags") or {}}
            for n, xi, yi in zip(osm_nodes, np.atleast_1d(x), np.atleast_1d(y))]


def _controls(road, juncs, signals, stops, stop_tree) -> dict[int, set[str]]:
    """{junction index: {"before", "after"}}: which approaches of the road stop at each junction
    ("before" = the traffic running with the node order, "after" = against it)."""
    line = LineString(road.nodes[:, :2])
    out: dict[int, set[str]] = {}
    for j, (x, y, _) in enumerate(juncs):
        if signals is not None and signals.query_ball_point((x, y), SIGNAL_REACH):
            out[j] = {"before", "after"}
            continue
        if stop_tree is None:
            continue
        sj = line.project(Point(x, y))
        for k in sorted(stop_tree.query_ball_point((x, y), SIGNAL_REACH)):
            p = Point(stops[k]["x"], stops[k]["y"])
            if line.distance(p) > road.width / 2 + STOP_ON_ROAD:
                continue  # a stop on the cross street
            if _tag(stops[k]["tags"], "stop") == "all":
                out[j] = {"before", "after"}
                break
            s = line.project(p)
            if abs(s - sj) >= ON_JUNCTION:  # on the junction node itself it could be either road's
                out.setdefault(j, set()).add("before" if s < sj else "after")
    return out


def _bar(road, i: int, p: int, piece: np.ndarray, at_end: bool) -> Marking:
    """A stop bar across the approach lanes, just short of where the piece's lines stop."""
    lanes, lane_w = _lanes(road)
    half = lanes * lane_w / 2
    if at_end:  # the traffic running with the node order stops at the piece's end
        end, prev = piece[-1], piece[-2]
        t = (end[:2] - prev[:2]) / math.dist(end[:2], prev[:2])
        centre = end[:2] - t * BAR_W / 2
        a, b = (-half, half) if road.oneway else (0.0, half)
    else:  # two-way: the traffic against the node order stops at the piece's start
        end, nxt = piece[0], piece[1]
        t = (nxt[:2] - end[:2]) / math.dist(end[:2], nxt[:2])
        centre = end[:2] + t * BAR_W / 2
        a, b = -half, 0.0
    right = np.array([t[1], -t[0]])
    nodes = np.array([[*(centre + a * right), end[2]], [*(centre + b * right), end[2]]])
    return Marking(f"r{i}_{p}_bar_{'end' if at_end else 'start'}", "stop_bar", WHITE, BAR_W,
                   MARK_PRIORITY, TEX_SOLID, bool(road.is_bridge), nodes)


def stop_bars(roads, junctions, nodes) -> list[Marking]:
    """Stop bars on the marked roads' approaches to signalled or stop-controlled junctions.
    junctions is the junctions() mapping; nodes are level_nodes."""
    sig = [(n["x"], n["y"]) for n in nodes if _tag(n["tags"], "highway") == "traffic_signals"]
    stops = [n for n in nodes if _tag(n["tags"], "highway") == "stop"]
    if not sig and not stops:
        return []
    signals = cKDTree(sig) if sig else None
    stop_tree = cKDTree([(n["x"], n["y"]) for n in stops]) if stops else None
    out = []
    for i, juncs in junctions.items():
        road = roads[i]
        control = _controls(road, juncs, signals, stops, stop_tree) if juncs else {}
        if not control:
            continue
        for p, (piece, j0, j1) in enumerate(_cut(road.nodes, juncs)):
            if j1 is not None and "before" in control.get(j1, ()):
                out.append(_bar(road, i, p, piece, at_end=True))
            if j0 is not None and not road.oneway and "after" in control.get(j0, ()):
                out.append(_bar(road, i, p, piece, at_end=False))
    return out


def _marked_crossing(tags: dict) -> bool:
    if _tag(tags, "highway") != "crossing":
        return False
    painted = _tag(tags, "crossing:markings")
    if painted is not None:
        return painted != "no"
    return _tag(tags, "crossing") in MARKED_CROSSINGS


def crosswalks(roads, nodes) -> list[Marking]:
    """A crosswalk across the marked road under each marked crossing node (level_nodes)."""
    marked = [n for n in nodes if _marked_crossing(n["tags"])]
    idx = [j for j, r in enumerate(roads) if len(r.nodes) >= 2]
    if not marked or not idx:
        return []
    lines = [LineString(roads[j].nodes[:, :2]) for j in idx]
    tree = STRtree(lines)
    out, names = [], Counter()
    for n in marked:
        p = Point(n["x"], n["y"])
        near = tree.query_nearest(p, max_distance=CROSSWALK_SNAP, all_matches=True)
        if not len(near):
            continue
        k = int(min(near))  # equally near roads: the first
        road, line = roads[idx[k]], lines[k]
        if road.kind not in MARKED_KINDS:
            continue
        s = line.project(p)
        centre = np.array(line.interpolate(s).coords[0][:2])
        a = np.array(line.interpolate(max(s - 1.0, 0.0)).coords[0][:2])
        b = np.array(line.interpolate(min(s + 1.0, line.length)).coords[0][:2])
        t = (b - a) / np.hypot(*(b - a))
        across = np.array([t[1], -t[0]]) * road.width / 2
        z = _z_at(road, s)
        name = f"x_{n['id']}"
        names[name] += 1
        if names[name] > 1:
            name = f"{name}_{names[name] - 1}"
        out.append(Marking(name, "crosswalk", CROSSWALK, CROSSWALK_W, MARK_PRIORITY, TEX_CROSSWALK,
                           bool(road.is_bridge), np.array([[*(centre - across), z], [*(centre + across), z]])))
    return out


# ------------------------------------------------------------------ paint on bridge decks

def _slices(deck_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The deck's cross-sections as (points (S, 2), unit right normals (S, 2), kind (S,)):
    kind 0 at each node (the normal averaged over the segments either side, as
    offset_polyline), 1 at the middle of each segment and 2 at its DECK_STATIONS
    (perpendicular to the segment)."""
    seg = np.diff(deck_xy, axis=0)
    u = seg / np.hypot(seg[:, 0], seg[:, 1])[:, None]
    across = np.column_stack([u[:, 1], -u[:, 0]])
    m = offset_polyline(deck_xy, 1.0) - deck_xy
    m /= np.hypot(m[:, 0], m[:, 1])[:, None]
    fractions = (0.5, *DECK_STATIONS)
    points = np.vstack([deck_xy] + [deck_xy[:-1] + f * seg for f in fractions])
    normals = np.vstack([m] + [across] * len(fractions))
    kind = np.r_[np.zeros(len(deck_xy), int), np.ones(len(seg), int), np.full(len(seg) * len(DECK_STATIONS), 2)]
    return points, normals, kind


def _crossings(xy: np.ndarray, cum: np.ndarray, points, normals, reach: float) -> np.ndarray:
    """Arc length along the polyline xy (cum) where it crosses each cross-section line (through
    points[j] along normals[j], at most reach from points[j]); NaN where it doesn't."""
    along = np.column_stack([-normals[:, 1], normals[:, 0]])
    rel = xy[None, :, :] - points[:, None, :]
    f = (rel * along[:, None, :]).sum(axis=2)  # (S, N) ahead of each cross-section
    lat = (rel * normals[:, None, :]).sum(axis=2)
    a, b = f[:, :-1], f[:, 1:]
    hit = (np.minimum(a, b) <= 0) & (np.maximum(a, b) >= 0) & (a != b)
    t = np.where(hit, a / np.where(hit, a - b, 1.0), 0.0)
    side = np.abs(lat[:, :-1] + t * (lat[:, 1:] - lat[:, :-1]))
    side = np.where(hit & (side <= reach), side, np.inf)
    k = np.argmin(side, axis=1)  # several crossings: the one nearest the deck's centreline
    rows = np.arange(len(points))
    s = cum[k] + t[rows, k] * (cum[k + 1] - cum[k])
    return np.where(np.isfinite(side[rows, k]), s, np.nan)


def _clear(s: float, seams: np.ndarray, step: int) -> float:
    """s moved on (step +1) or back (-1) until it is at least DECK_SEAM_GAP from every seam."""
    for _ in range(len(seams) + 1):
        close = seams[np.abs(seams - s) < DECK_SEAM_GAP]
        if not len(close):
            break
        s = close.max() + DECK_SEAM_GAP + 1e-9 if step > 0 else close.min() - DECK_SEAM_GAP - 1e-9
    return s


def deck_line(nodes, deck) -> np.ndarray | None:
    """A paint line on a bridge deck (N x 3, along deck, a Road) resampled off the deck
    MeshRoad's seams: one vertex where each of the DECK_STATIONS cross-sections of each deck
    segment cuts the line, and the ends DECK_END_INSET in from the line's own ends; every
    vertex at least DECK_SEAM_GAP along the line from the cross-sections at the deck's nodes
    and at the middles of its segments (an end that is not moves further in). z is the
    deck's, interpolated along it. None when under MIN_LINE once inset."""
    line = _dedupe(np.asarray(nodes, dtype=float))
    deck_xy = _dedupe(np.asarray(deck.nodes, dtype=float))[:, :2]
    if len(line) < 2 or len(deck_xy) < 2:
        return None
    cum = _cum(line)
    lo, hi = DECK_END_INSET, cum[-1] - DECK_END_INSET
    if hi - lo < MIN_LINE:
        return None
    points, normals, kind = _slices(deck_xy)
    s = _crossings(line[:, :2], cum, points, normals, float(deck.width))
    seams = np.sort(s[(kind < 2) & np.isfinite(s)])
    lo, hi = _clear(lo, seams, +1), _clear(hi, seams, -1)
    if hi - lo < MIN_LINE:
        return None
    inner = s[(kind == 2) & np.isfinite(s)]
    if len(seams):
        inner = inner[np.abs(inner[:, None] - seams[None, :]).min(axis=1) >= DECK_SEAM_GAP]
    inner = np.sort(inner[(inner > lo + DECK_SEAM_GAP) & (inner < hi - DECK_SEAM_GAP)])
    out = np.array([_at(line, cum, a) for a in np.r_[lo, inner, hi]])
    centre, stations = LineString(deck.nodes[:, :2]), _cum(deck.nodes)
    out[:, 2] = np.interp([centre.project(Point(p)) for p in out[:, :2]], stations, deck.nodes[:, 2])
    return out


# ------------------------------------------------------------------ stage

def road_lines(road, i: int, juncs) -> list[Marking]:
    """The lane lines of roads[i], trimmed at its junctions (z still the road's)."""
    lines = lane_lines(road)
    out = []
    for p, (piece, _, _) in enumerate(_cut(road.nodes, juncs) if lines else []):
        for k, (kind, offset, material, width) in enumerate(lines):
            nodes = np.column_stack([offset_polyline(piece[:, :2], offset), piece[:, 2]])
            if kind == "centre_turn" and offset < 0:  # mirror the solid/dashed pair on the far edge
                nodes = nodes[::-1]
            out.append(Marking(f"r{i}_{p}_{k}_{kind}", kind, material, width, LINE_PRIORITY,
                               TEX_DASHED if material in (DASHED, TURN) else TEX_SOLID, bool(road.is_bridge),
                               nodes))
    return out


def _finish(marks: list[Marking], heights: np.ndarray, extent: Extent, decks=None) -> list[Marking]:
    """Clip to the box, put ground paint on the terrain, move deck lines off the deck's
    seams (deck_line; decks[k] is the deck road under marks[k], or None), drop what is too short."""
    decks = decks if decks is not None else [None] * len(marks)
    out, dropped = [], 0
    for m, deck in zip(marks, decks):
        parts = _clip_box(m.nodes, extent.half_size)
        for k, part in enumerate(parts):
            if not m.over_objects:
                part = part.copy()
                part[:, 2] = sample_height(heights, extent, part[:, 0], part[:, 1])
            elif deck is not None and m.kind in LINE_KINDS:
                part = deck_line(part, deck)
                if part is None:
                    dropped += 1
                    continue
            least = MIN_LINE if m.kind in LINE_KINDS else MIN_ACROSS
            if len(part) < 2 or _cum(part)[-1] < least or not np.isfinite(part).all():
                dropped += 1
                continue
            out.append(replace(m, name=m.name if len(parts) == 1 else f"{m.name}_{k}", nodes=part))
    if dropped:
        log.info("%d paint pieces dropped (too short after the box edge or the deck end insets, "
                 "or no deck height)", dropped)
    return out


def build_markings(roads, nodes, heights: np.ndarray, extent: Extent) -> list[Marking]:
    """All the paint: lane lines, stop bars and crosswalks (nodes are level_nodes)."""
    juncs = junctions(roads)
    log.info("%d junctions on %d marked roads", sum(len(v) for v in juncs.values()), len(juncs))
    lanes = carriageways(roads)
    log.info("%d marked roads overlap a carriageway their paint keeps off", sum(map(bool, lanes.values())))
    marks, decks = [], []
    for i, road in enumerate(roads):
        if road.kind in MARKED_KINDS and len(road.nodes) >= 2:
            lines = keep_off(road_lines(road, i, juncs.get(i, [])), lanes.get(i, []))
            marks += lines
            decks += [road if road.is_bridge else None] * len(lines)
    marks += stop_bars(roads, juncs, nodes)
    marks += crosswalks(roads, nodes)
    return _finish(marks, heights, extent, decks + [None] * (len(marks) - len(decks)))


def run(cfg, raw_dir=None, out_dir=None) -> None:
    raw_dir, out_dir = Path(raw_dir or cfg.raw_dir), Path(out_dir or cfg.out_dir)
    path = out_dir / "markings.json"
    if not cfg.markings_enabled:
        save_markings([], path)
        log.info("markings are switched off (markings.enabled): wrote an empty %s", path)
        return
    extent = Extent.from_config(cfg)
    roads = load_roads(out_dir / "roads.json")
    feats = raw_dir / "osm_features.json"
    if feats.exists():
        osm_nodes = json.loads(feats.read_text(encoding="utf-8")).get("nodes", [])
    else:
        log.warning("no %s (run the features stage): lane lines only, no stop bars or crosswalks", feats)
        osm_nodes = []
    marks = build_markings(roads, level_nodes(osm_nodes, extent), load_heights(out_dir=out_dir), extent)
    save_markings(marks, path)
    kinds = Counter(m.kind for m in marks)
    painted = len({m.name.split("_")[0] for m in marks if m.kind in LINE_KINDS})
    log.info("%d markings on %d roads: %s", len(marks), painted,
             ", ".join(f"{k} {v}" for k, v in kinds.most_common()) or "none")


# ------------------------------------------------------------------ level

def _r(v, nd=3) -> float:
    return round(float(v), nd)


def decal_road(m: Marking) -> dict:
    """The DecalRoad for one marking: paint (drivability -1) stays out of the AI road network."""
    lift = DECK_LIFT if m.over_objects else 0.0
    nodes = [[_r(x), _r(y), _r(z + lift), _r(m.width, 2)] for x, y, z in m.nodes]
    return {"name": f"br_mark_{m.name}", "class": "DecalRoad", "persistentId": bf.persistent_id("mark", m.name),
            "position": nodes[0][:3],
            "breakAngle": 1, "distanceFade": [200, 50], "drivability": -1, "improvedSpline": True,
            "material": m.material, "nodes": nodes, "overObjects": bool(m.over_objects),
            "renderPriority": int(m.priority), "startEndFade": [0.1, 0.1],
            "textureLength": float(m.texture_length)}


def level_content(cfg, data_dir, root, heights, extent) -> dict[str, list[dict]]:
    """The level.FEATURE_MODULES hook: writes art/road/br_markings.materials.json (the
    materials used) under root and returns {"markings": [DecalRoad, ...]}, or {} when there
    is no paint. The paint's z was set by run, so heights and extent are not needed here."""
    path = Path(data_dir) / "markings.json"
    if not cfg.markings_enabled or not path.exists():
        return {}
    marks = load_markings(path)
    if not marks:
        return {}
    used = sorted({m.material for m in marks})
    unknown = [k for k in used if k not in MATERIALS]
    if unknown:
        raise ValueError(f"{path} uses unknown materials {unknown}; run the markings stage again")
    dest = Path(root) / MATERIALS_FILE
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({k: MATERIALS[k] for k in used}, indent=1, ensure_ascii=False), encoding="utf-8")
    return {GROUP: [decal_road(m) for m in marks]}
