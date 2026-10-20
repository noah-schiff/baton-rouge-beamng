"""Stage 8: sanity checks on the stage outputs, plus overlay images to eyeball them.

The overlay is a hillshade blended 50/50 with the aerial base color, with roads
drawn at their width (bridges in red), water outlines in blue, building footprints
in grey, trees as small green dots and spawn points as yellow dots. run() writes a
2048 px view of the whole map and a 1024 px crop around each spawn point, and lists
any problems in data/out/problems.txt.

Besides the terrain, the roads and the bridge decks, check() looks at each feature
output that exists:

* buildings.json: no footprint, shrunk 0.5 m, on a road corridor (the centreline
  buffered by width / 2; every class, bridge decks included), and the base within 1 m
  of the lowest terrain along the footprint's walls, found as the buildings stage does
  (buildings.lowest_ground: sampled every 2 m, the points in a water body left out).
  The styled buildings of styled.json are checked the same way, along the outlines their walls
  follow (styled.drawn_buildings), and drawn in the overlay with the others.
* trees.json: no tree within 1.5 m of a road corridor or 3 m of a bridge deck's, or
  inside a footprint, and z within 1 m of the terrain.
* landmarks.json: no drivable road (landmarks.ground_roads: drivability > 0, so not a covered
  passage, and not a bridge deck) with its centreline across the footprint shrunk 0.5 m, nor
  across what the model stands on (landmarks.ground_outline from out/models, shrunk 0.5 m: a
  model can reach past its footprint), nor with its corridor grown to width / 2 - 0.5 m (the
  buildings' rule) reaching into either (the lanes beside it; not for the roads the landmark's
  params.keep_off_roads exempts, landmarks.exempt_ways), and the base no more than 1 m above the
  lowest terrain along the footprint's outline (sampled every 2 m). A landmark over bridge decks
  (params.decks: the Wilkinson truss) is in the air over its footprint, so only what its model
  stands on counts. The lanes cut the footprint (landmarks.resolve keeps it off them and records
  the cut as lane_cut in landmarks.json), which hides a road across it from the checks above: so
  the cut itself may take no more than LANE_CUT_SHARE (2 %) of the footprint, nor split it into
  pieces or leave a hole in it, and a landmark that does is reported with the roads that cut it;
  and a road whose centreline crossed the footprint before the cut (lane_cut's through: a road
  into it, whose notch can be well under 2 %) is reported as passing through it.
* The terrain for these is the one the level writes, cut under the bridge decks.
* markings.json: every paint node inside the box.
* bayous.json: every run has at least 2 nodes, each inside the box and with a width over 0.
* dressing.json: no barrier vertex inside another road's corridor on its level (that road's surface
  within dressing.SAME_LEVEL of the vertex), but for those within dressing.JOINT_REACH of a
  straight joint where that barrier runs on onto the next deck's, in the corridor of a deck that
  leaves it be there (dressing.joint_decks); and no pier footprint
  (dressing.pier_footprint) on the corridor of another road running under its top (a deck over it,
  or one at its own deck's level, stands on it too). A barrier's or a pier's own deck is the one
  its "deck" id names.
* The built level folder (<dist>/level/levels/<name>): every .dae shape's levels of
  detail at most 60,000 vertices.
"""
from __future__ import annotations

import logging
import math
import re
from pathlib import Path

import numpy as np
import shapely
from PIL import Image, ImageDraw
from shapely.geometry import LineString, Point, Polygon
from shapely.strtree import STRtree

from .bayous import load_bayous
from .bridges import END_FRACTION, JOIN_TOL, MARGIN, TERRAIN_GAP, clear_under_decks, deck_crossings
from .buildings import STYLED_FILE, drop_conflicts, load_buildings, lowest_ground, wet_areas
from .dressing import JOINT_REACH, SAME_LEVEL, deck_id, joint_decks, load_dressing, pier_footprint, road_z
from .geo import Extent
from .landmarks import exempt, exempt_ways, ground_outline, ground_roads, load_landmarks, road_label
from .markings import load_markings
from .roads import load_roads
from .styled import drawn_buildings
from .terrain import sample_height
from .trees import load_trees
from .water import load_water

log = logging.getLogger(__name__)

OUT_DIR = Path("data/out")
LANDMARKS = {"State Capitol": (-91.1868, 30.4560), "Tiger Stadium": (-91.1855, 30.4115)}
LANDMARK_TOL = 2.0  # m
CUT_WARN = 1.0      # m a deck may sit under the terrain (plus margin) before it is a problem;
                    # shallower dips are cut out of the terrain by the level writer
JOINT_STEP = 0.5    # m: a deck end that meets another deck has to sit at its height
JOINT_DIP = 2.0     # m a joint may sit below both decks a tenth of their length away
CROSS_WARN = 1.0    # m decks crossing may come closer than clearance_m before it is a problem
OVERLAP_EDGE = 0.5  # m of two decks' edges that may touch without counting as shared pavement
SHRINK = 0.5        # m taken off a footprint before it is tested against the road corridors
LANE_CUT_SHARE = 0.02  # of a landmark's footprint the roads beside it may cut away (landmarks.resolve) unremarked
TREE_ROAD = 1.5     # m a tree keeps off a road corridor
TREE_DECK = 3.0     # m a tree keeps off a bridge deck's corridor
ON_TERRAIN = 1.0    # m a tree's z or a building's base may sit above or below the terrain
DAE_MAX_VERTS = 60000  # per level of detail of a shape (16-bit indices, with a margin)

FOOTPRINT_COLOUR = (185, 185, 185)
TREE_COLOUR = (30, 190, 60)
OVERLAY_MAX_TREES = 200_000  # more trees in view are thinned to this many dots


def landmark_errors(cfg) -> list[str]:
    """Landmarks must survive lon/lat -> grid pixel -> level coordinates within 2 m."""
    extent = Extent.from_config(cfg)
    n = cfg.terrain_size
    t = extent.transform(n)
    out = []
    for name, (lon, lat) in LANDMARKS.items():
        x, y = extent.to_level(lon, lat)
        if not extent.contains(x, y):
            continue
        e, nn = extent.level_to_utm(x, y)
        col, row = ~t @ (e, nn)
        bx, by = extent.utm_to_level(*(t @ (col, row)))
        err = math.hypot(bx - x, by - y)
        if err > LANDMARK_TOL:
            out.append(f"{name}: grid round trip is off by {err:.2f} m")
    return out


def check(cfg, out_dir=None) -> list[str]:
    """Problems found in the stage outputs; an empty list means clean."""
    out_dir = Path(out_dir or cfg.out_dir)
    extent = Extent.from_config(cfg)
    problems: list[str] = []
    heights = np.load(out_dir / "heights.npy")
    if heights.shape != (cfg.terrain_size, cfg.terrain_size):
        problems.append(f"heightmap is {heights.shape[0]} x {heights.shape[1]}, "
                        f"config says {cfg.terrain_size}")
    roads = load_roads(out_dir / "roads.json")
    for road in roads:
        xyz = road.nodes
        if np.isnan(xyz).any():
            problems.append(f"{road_label(road)}: NaN coordinates")
            continue
        inside = extent.contains(xyz[:, 0], xyz[:, 1])
        if not inside.all():
            i = int(np.argmin(inside))
            problems.append(f"{road_label(road)}: node outside the box at ({xyz[i, 0]:.1f}, {xyz[i, 1]:.1f})")
        if road.is_bridge and len(xyz) > 2:
            d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xyz[:, :2], axis=0).T))]
            t = d / d[-1]
            mid = (t >= END_FRACTION) & (t <= 1 - END_FRACTION)
            ground = sample_height(heights, extent, xyz[:, 0], xyz[:, 1])
            gap = xyz[:, 2] - (ground + MARGIN)
            if mid.any() and gap[mid].min() < -CUT_WARN:
                k = int(np.argmin(np.where(mid, gap, np.inf)))
                problems.append(f"{road_label(road)}: deck {xyz[k, 2] - ground[k]:.1f} m above the ground at "
                                f"{100 * t[k]:.0f} % of its length (needs {MARGIN} m), so the level "
                                f"cuts the terrain {-gap[k] + TERRAIN_GAP - MARGIN:.1f} m deep there")
    problems += deck_problems([r for r in roads if r.is_bridge and not np.isnan(r.nodes).any()],
                              cfg.clearance_m)
    problems += landmark_errors(cfg)
    problems += feature_problems(cfg, out_dir, extent, heights, roads)
    return problems


def level_root(cfg) -> Path:
    """The level folder level.run() builds (before it is zipped)."""
    return Path(cfg.dist_dir) / "level" / "levels" / cfg.level_name


def feature_problems(cfg, out_dir: Path, extent: Extent, heights: np.ndarray, roads) -> list[str]:
    """The checks on buildings, trees, landmarks, paint, bayous, deck dressing and shapes, each run
    only when its input exists."""
    out_dir = Path(out_dir)
    paths = {name: out_dir / f"{name}.json" for name in ("buildings", "trees", "landmarks", "markings", "bayous",
                                                       "dressing")}
    buildings = load_buildings(paths["buildings"]) if paths["buildings"].exists() else []
    buildings += drawn_buildings(out_dir / STYLED_FILE)
    trees = load_trees(paths["trees"]) if paths["trees"].exists() else []
    landmarks = load_landmarks(paths["landmarks"]) if paths["landmarks"].exists() else []
    usable = [r for r in roads if len(r.nodes) >= 2 and np.isfinite(r.nodes).all()]
    problems: list[str] = []
    if buildings or trees or landmarks:
        terrain, _ = clear_under_decks(heights, extent, roads)  # what the level writes to the .ter
        if buildings:
            water = load_water(out_dir / "water.json") if (out_dir / "water.json").exists() else []
            problems += building_problems(buildings, usable, terrain, extent, water)
        if trees:
            problems += tree_problems(trees, buildings, usable, terrain, extent)
        if landmarks:
            problems += landmark_problems(landmarks, usable, terrain, extent, out_dir / "models")
    if paths["markings"].exists():
        problems += marking_problems(load_markings(paths["markings"]), extent)
    if paths["bayous"].exists():
        problems += bayou_problems(load_bayous(paths["bayous"]), extent)
    if paths["dressing"].exists():
        problems += dressing_problems(load_dressing(paths["dressing"]), usable)
    problems += dae_problems(level_root(cfg))
    return problems


def corridors(roads, extra: float = 0.0) -> np.ndarray:
    """Each road's corridor, its centreline buffered by width / 2 + extra, as the stages draw it."""
    lines = np.array([LineString(r.nodes[:, :2]) for r in roads], dtype=object)
    return shapely.buffer(lines, np.array([r.width / 2.0 + extra for r in roads], float))


def footprints(buildings) -> np.ndarray:
    """The buildings' footprints (with their holes) as shapely polygons, repaired where invalid;
    a ring of fewer than 3 points gives an empty polygon."""
    polys = np.array([Polygon(b.ring, b.holes) if len(b.ring) >= 3 else Polygon() for b in buildings],
                     dtype=object)
    bad = ~shapely.is_valid(polys)
    if bad.any():
        polys[bad] = shapely.make_valid(polys[bad])
    return polys


def _first_hits(tree: STRtree, geoms) -> dict[int, int]:
    """{tree item: the first of geoms that intersects it}. The geoms are the ones GEOS
    prepares, so the large ones (corridors, footprints) go there and the many small ones
    in the tree."""
    hits: dict[int, int] = {}
    if len(geoms):
        for g, t in zip(*tree.query(geoms, predicate="intersects").tolist()):
            hits.setdefault(t, g)
    return hits


def building_problems(buildings, roads, terrain: np.ndarray, extent: Extent, water=()) -> list[str]:
    """Footprints on a road corridor, and bases off the terrain (the water bodies' points
    left out, as the stage does), in building order."""
    polys = footprints(buildings)
    cores = shapely.buffer(polys, -SHRINK)
    thin_ = shapely.is_empty(cores)
    cores[thin_] = polys[thin_]  # a sliver thinner than the shrink is tested whole, as the stage does
    on_road = _first_hits(STRtree(cores), corridors(roads)) if roads else {}  # the corridors are prepared
    if on_road:  # the stage's own rules keep some: covered passages, service drives ending in big buildings
        hit = [i for i in on_road if "/t" not in buildings[i].id]  # a tower part stands on its checked podium
        kept, _ = drop_conflicts([(i, "", buildings[i].kind, polys[i], {}, None) for i in hit], roads, [])
        on_road = {i: j for i, j in on_road.items() if i in hit and i not in {k[0] for k in kept}}
    sizes = np.array([len(b.ring) for b in buildings])
    lowest = lowest_ground([(b.ring, b.holes) for b in buildings], terrain, extent,
                           wet_areas(water, extent, terrain.shape[0]))
    out = []
    for i, b in enumerate(buildings):
        name = f"building {b.id} ({b.source})"
        if sizes[i] < 3:
            out.append(f"{name}: its ring has only {sizes[i]} points")
            continue
        if i in on_road:
            c = polys[i].representative_point()
            out.append(f"{name}: footprint on the corridor of {road_label(roads[on_road[i]])} "
                       f"near ({c.x:.0f}, {c.y:.0f})")
        dz = b.base_z - lowest[i]
        if "/t" in b.id and dz < 0:  # a tower part on its podium's base: buried in the podium
            continue
        if not abs(dz) <= ON_TERRAIN:
            out.append(f"{name}: base {b.base_z:.2f} m is {abs(dz):.1f} m {'above' if dz > 0 else 'below'} "
                       f"the terrain at its lowest point ({lowest[i]:.2f} m)")
    return out


def tree_problems(trees, buildings, roads, terrain: np.ndarray, extent: Extent) -> list[str]:
    """Trees on a road (+1.5 m) or a deck (+3 m) or inside a footprint, and trees off the
    terrain, in tree order; a tree is named by its index in trees.json."""
    xy = np.array([[t["x"], t["y"]] for t in trees], float).reshape(-1, 2)
    z = np.array([t["z"] for t in trees], float)
    points = STRtree(shapely.points(xy))
    why: dict[int, str] = {}
    ground = [r for r in roads if not r.is_bridge]
    decks = [r for r in roads if r.is_bridge]
    for group, extra, what in ((ground, TREE_ROAD, f"within {TREE_ROAD:g} m of the corridor of"),
                               (decks, TREE_DECK, f"within {TREE_DECK:g} m of the deck of")):
        if group:
            for t, g in _first_hits(points, corridors(group, extra)).items():
                why.setdefault(t, f"{what} {road_label(group[g])}")
    if buildings:
        for t, b in _first_hits(points, footprints(buildings)).items():
            why.setdefault(t, f"inside building {buildings[b].id}")
    dz = z - np.atleast_1d(sample_height(terrain, extent, xy[:, 0], xy[:, 1]))
    out = []
    for i in sorted(set(why) | set(np.flatnonzero(~(np.abs(dz) <= ON_TERRAIN)).tolist())):
        t = trees[i]
        name = f"tree {i} ({t['item']}) at ({xy[i, 0]:.1f}, {xy[i, 1]:.1f})"
        if i in why:
            out.append(f"{name} is {why[i]}")
        if not abs(dz[i]) <= ON_TERRAIN:
            out.append(f"{name}: z {z[i]:.2f} m is {abs(dz[i]):.1f} m {'above' if dz[i] > 0 else 'below'} "
                       "the terrain")
    return out


def _core(poly):
    """poly shrunk SHRINK; poly itself when it is a sliver thinner than that."""
    core = poly.buffer(-SHRINK)
    return poly if core.is_empty else core


def _hits(tree: STRtree, geom) -> set[int]:
    """The items of tree that intersect geom (none for an empty one)."""
    return set(tree.query(geom, predicate="intersects").tolist()) if not geom.is_empty else set()


def _lane_cut_problem(name: str, cut: dict) -> list[str]:
    """The one line for a landmark whose lane_cut (landmarks.json, from landmarks.resolve) took more than
    LANE_CUT_SHARE of its footprint, split it or left a hole in it, naming the roads that cut it; [] when
    it did none of these."""
    why = []
    if cut["share"] > LANE_CUT_SHARE:
        why.append(f"removed {100 * cut['share']:.1f} % of its footprint ({cut['m2']:.0f} m2)")
    if cut["pieces"] > 1:
        why.append(f"split it into {cut['pieces']} pieces")
    if cut["holes"]:
        why.append(f"left {cut['holes']} hole{'s' if cut['holes'] > 1 else ''} in it")
    if not why:
        return []
    said = ", ".join(why[:-1]) + (" and " if len(why) > 1 else "") + why[-1]
    roads = ", ".join(f"{r['road']} {r['m2']:.1f} m2" for r in cut["roads"])
    return [f"{name}: the lane cut {said}" + (f", by {roads}" if roads else "")]


def landmark_problems(landmarks, roads, heights: np.ndarray, extent: Extent, models_dir=None) -> list[str]:
    """Drivable ground roads (landmarks.ground_roads: drivability > 0, not a bridge deck) whose
    centreline crosses a landmark's footprint shrunk 0.5 m (not for a landmark over bridge decks,
    params.decks: its footprint is in the air) or crossed it before the lane cut (the roads its
    lane_cut's through names; an older landmarks.json has none), or (given models_dir, the models
    stage's out/models) what its model stands on (landmarks.ground_outline) shrunk 0.5 m; those
    whose corridor grown to width / 2 - SHRINK (building_problems' rule) reaches into the footprint
    or what the model stands on (not for the roads the landmark's params.keep_off_roads exempts,
    landmarks.exempt_ways); a landmark whose lane_cut (landmarks.resolve: the cut of the footprint
    by the lanes beside it, so the roads it names are not the exempt ones) took more than
    LANE_CUT_SHARE of its footprint, split it into pieces or left a hole in it (an older
    landmarks.json has no lane_cut); and landmark bases over 1 m above the lowest terrain along the
    footprint's outline. In landmark order, one line per landmark and road (a road that passes
    through the footprint is named for that alone), and one for the lane cut."""
    drivable = ground_roads(roads)
    lines = STRtree([LineString(r.nodes[:, :2]) for r in drivable]) if drivable else None
    lanes = STRtree(corridors(drivable, -SHRINK)) if drivable else None
    rings = [np.asarray(lm["footprint"], float).reshape(-1, 2) for lm in landmarks]
    lowest = lowest_ground([(ring, []) for ring in rings], heights, extent)
    out = []
    for lm, ring, low in zip(landmarks, rings, lowest):
        params = lm.get("params") or {}
        aloft, ways = bool(params.get("decks")), exempt_ways(lm["name"], params)
        foot = Polygon(ring) if len(ring) >= 3 and not aloft else Polygon()
        outline = ground_outline(lm, models_dir) if models_dir is not None else Polygon()
        where = "what its model stands on" if aloft else "its model's ground outline, past the footprint"
        # the roads across the footprint as it is, then those resolve found across it before the lane cut
        # (lane_cut's through; an older landmarks.json has none), each once
        here = sorted(_hits(lines, _core(foot))) if lines is not None else []
        crossing = list(dict.fromkeys([road_label(drivable[j]) for j in here] +
                                      list((lm.get("lane_cut") or {}).get("through") or [])))
        out += [f"{lm['name']}: road {label} passes through it" for label in crossing]
        if lines is not None:
            said = set(crossing)                 # a road that passes through it is not named again below
            under = {j for j in _hits(lines, _core(outline)) if road_label(drivable[j]) not in said}
            into = {j for j in _hits(lanes, foot)
                    if not exempt(drivable[j], ways) and road_label(drivable[j]) not in said} - under
            past = {j for j in _hits(lanes, outline)
                    if not exempt(drivable[j], ways) and road_label(drivable[j]) not in said} - under - into
            for j in sorted(under):
                out.append(f"{lm['name']}: road {road_label(drivable[j])} passes through {where}")
            for js, geom, what in ((into, foot, "it"), (past, outline, where)):
                for j in sorted(js):
                    c = geom.intersection(lanes.geometries[j]).representative_point()
                    out.append(f"{lm['name']}: the corridor of road {road_label(drivable[j])} reaches into {what} "
                               f"near ({c.x:.0f}, {c.y:.0f})")
        if lm.get("lane_cut"):    # an older landmarks.json has none
            out += _lane_cut_problem(lm["name"], lm["lane_cut"])
        dz = lm["base_z"] - low
        if dz > ON_TERRAIN:
            out.append(f"{lm['name']}: base floats {dz:.1f} m")
    return out


def marking_problems(marks, extent: Extent) -> list[str]:
    """Paint with a node outside the box (or a NaN one)."""
    out = []
    for m in marks:
        xyz = np.asarray(m.nodes, float).reshape(-1, 3)
        inside = extent.contains(xyz[:, 0], xyz[:, 1])
        if not inside.all():
            i = int(np.argmin(inside))
            out.append(f"marking {m.name}: node outside the box at ({xyz[i, 0]:.1f}, {xyz[i, 1]:.1f})")
    return out


def bayou_problems(bayous: dict, extent: Extent) -> list[str]:
    """Bayou runs (bayous.json) with a node outside the box (or a NaN one), a width of 0 or less, or
    fewer than 2 nodes, in run order."""
    out = []
    for r in bayous["runs"]:
        name = f"bayou {r['id']}" + (f" ({r['name']})" if r.get("name") else "")
        nodes = np.asarray(r["nodes"], float).reshape(-1, 4)
        if len(nodes) < 2:
            out.append(f"{name}: {len(nodes)} node{'' if len(nodes) == 1 else 's'}, a run needs 2")
        inside = extent.contains(nodes[:, 0], nodes[:, 1])
        if not inside.all():
            i = int(np.argmin(inside))
            out.append(f"{name}: node outside the box at ({nodes[i, 0]:.1f}, {nodes[i, 1]:.1f})")
        flat = ~(nodes[:, 3] > 0)
        if flat.any():
            i = int(np.argmax(flat))
            out.append(f"{name}: width {nodes[i, 3]:g} m at ({nodes[i, 0]:.1f}, {nodes[i, 1]:.1f})")
    return out


def dressing_problems(dressing: dict, roads) -> list[str]:
    """Barriers (dressing.json) with vertices inside another road's corridor on their level (but
    near a straight joint, as the module docstring says), and piers on the corridor of another road
    running under their top, in file order: one line per barrier or pier and road."""
    usable = [r for r in roads if len(r.nodes) >= 2 and np.isfinite(r.nodes).all()]
    if not usable:
        return []
    ids = [deck_id(r) for r in usable]
    lines = [LineString(r.nodes[:, :2]) for r in usable]
    tree = STRtree(corridors(usable))
    decks = [r for r in usable if r.is_bridge]
    run_on: dict[tuple[str, str], list] = {}         # (deck id, side) -> [(deck id, straight joint (x, y))]
    for i, at in joint_decks(decks).items():
        for k, p, sides in at:
            for side in sides:
                run_on.setdefault((deck_id(decks[i]), side), []).append((deck_id(decks[k]), p))
    out = []
    for bar in dressing["barriers"]:
        pts = np.asarray(bar["points"], float).reshape(-1, 3)
        point, road = tree.query(shapely.points(pts[:, :2]), predicate="within")
        for j in sorted(set(road.tolist())):
            if ids[j] == bar["deck"]:
                continue
            here = point[road == j]
            for other, p in run_on.get((bar["deck"], bar["side"]), []):
                if other == ids[j]:                  # the barrier runs on across a straight joint
                    here = here[np.hypot(*(pts[here, :2] - p).T) > JOINT_REACH]
            if not len(here):
                continue
            on = here[np.abs(pts[here, 2] - road_z(usable[j], lines[j], pts[here, :2])) < SAME_LEVEL]
            if len(on):
                x, y, _ = pts[on[0]]
                out.append(f"barrier {bar['side']} of deck {bar['deck']}: {len(on)} vertices inside the corridor "
                           f"of {ids[j]} ({road_label(usable[j])}) on its level, the first at ({x:.0f}, {y:.0f})")
    for pier in dressing["piers"]:
        for j in sorted(int(j) for j in tree.query(pier_footprint(pier), predicate="intersects")):
            if ids[j] != pier["deck"] and road_z(usable[j], lines[j], [(pier["x"], pier["y"])])[0] < pier["z_top"]:
                out.append(f"pier of deck {pier['deck']} at ({pier['x']:.0f}, {pier['y']:.0f}) stands on the "
                           f"corridor of {ids[j]} ({road_label(usable[j])})")
    return out


_TAG = re.compile(rb"<(geometry|source|accessor|input|triangles|polylist|p|node|instance_geometry)\b([^>]*)>")
_ATTR = re.compile(rb'([\w:.-]+)\s*=\s*"([^"]*)"')


def dae_vertex_counts(path) -> dict[str, int]:
    """Vertices per mesh node of a COLLADA file, as the engine builds them: one per distinct
    tuple of indices in a primitive's <p>. With every input at offset 0 (dae.py's layout)
    that is the count of the mesh's POSITION source, so only meshes with inputs at other
    offsets have their indices read. A scan of the tags rather than a full XML parse,
    since the full map's shapes come to about a gigabyte."""
    data = Path(path).read_bytes()
    counts: dict[bytes, int] = {}         # source id -> accessor count
    position: dict[bytes, bytes] = {}     # geometry id -> its POSITION source id
    split: dict[bytes, list] = {}         # geometry id -> index tuples of its split primitives
    instanced: dict[bytes, bytes] = {}    # node name -> geometry id
    geom = source = node = None
    stride = 1
    for m in _TAG.finditer(data):
        tag, attrs = m.group(1), dict(_ATTR.findall(m.group(2)))
        if tag == b"geometry":
            geom, stride = attrs.get(b"id"), 1
        elif tag == b"source":
            source = attrs.get(b"id")
        elif tag == b"accessor" and source is not None:
            counts[source] = int(attrs.get(b"count", b"0"))
        elif tag == b"input":
            if attrs.get(b"semantic") == b"POSITION" and geom is not None:
                position[geom] = attrs.get(b"source", b"").lstrip(b"#")
            if b"offset" in attrs:
                stride = max(stride, int(attrs[b"offset"]) + 1)
        elif tag in (b"triangles", b"polylist"):
            stride = 1
        elif tag == b"p" and stride > 1 and geom is not None:
            idx = np.array(data[m.end():data.find(b"</p>", m.end())].split(), dtype=np.int64)
            split.setdefault(geom, []).append(idx[:len(idx) - len(idx) % stride].reshape(-1, stride))
        elif tag == b"node":
            node = attrs.get(b"name") or attrs.get(b"id")
        elif tag == b"instance_geometry" and node is not None:
            instanced[node] = attrs.get(b"url", b"").lstrip(b"#")

    def vertices(g: bytes) -> int:
        if g in split:
            return sum(len(np.unique(a, axis=0)) for a in split[g])
        return counts.get(position.get(g, b""), 0)

    named = instanced or {g: g for g in position}
    return {n.decode(): vertices(g) for n, g in named.items()}


def dae_problems(root: Path) -> list[str]:
    """Shapes in the built level folder with a level of detail over DAE_MAX_VERTS vertices."""
    root = Path(root)
    if not root.is_dir():
        return []
    out = []
    for f in sorted(root.rglob("*.dae")):
        for node, n in dae_vertex_counts(f).items():
            if n > DAE_MAX_VERTS:
                out.append(f"{f.relative_to(root).as_posix()}: {node} has {n} vertices "
                           f"(the cap is {DAE_MAX_VERTS})")
    return out


def deck_problems(decks, clearance: float) -> list[str]:
    """Steps and dips where decks join, and decks crossing too close to each other."""
    lines = [LineString(r.nodes[:, :2]) for r in decks]
    arcs = [np.r_[0.0, np.cumsum(np.hypot(*np.diff(r.nodes[:, :2], axis=0).T))] for r in decks]

    def z_at(i: int, s: float) -> float:
        return float(np.interp(s, arcs[i], decks[i].nodes[:, 2]))

    def inward(i: int, s: float) -> float:  # z a tenth of the deck in from the end at s
        return z_at(i, 0.1 * arcs[i][-1] if s < arcs[i][-1] / 2 else 0.9 * arcs[i][-1])

    tree = STRtree(lines) if lines else None
    out, seen = [], set()
    for i, r in enumerate(decks):
        for end in (0, -1):
            p = Point(r.nodes[end, :2])
            for j in (int(j) for j in tree.query(p.buffer(JOIN_TOL))):
                if j == i or lines[j].distance(p) >= JOIN_TOL:
                    continue
                key = (min(i, j), max(i, j), round(p.x), round(p.y))
                if key in seen:
                    continue
                seen.add(key)
                sj = float(lines[j].project(p))
                zi, zj = float(r.nodes[end, 2]), z_at(j, sj)
                where = f"({p.x:.0f}, {p.y:.0f})"
                if abs(zi - zj) > JOINT_STEP:
                    out.append(f"{road_label(r)} ends {abs(zi - zj):.1f} m {'above' if zi > zj else 'below'} "
                               f"{road_label(decks[j])} where it joins it at {where}")
                elif min(sj, arcs[j][-1] - sj) < JOIN_TOL:
                    dip = min(inward(i, arcs[i][-1] if end else 0.0), inward(j, sj)) - max(zi, zj)
                    if dip > JOINT_DIP:
                        out.append(f"{road_label(r)} and {road_label(decks[j])} dip {dip:.1f} m to their "
                                   f"joint at {where}")
    flagged = set()
    for i in range(len(decks)):
        for j in (int(j) for j in tree.query(lines[i])) if tree is not None else ():
            if j <= i:
                continue
            for si, sj in deck_crossings(lines[i], lines[j]):
                dz = abs(z_at(i, si) - z_at(j, sj))
                if dz < clearance - CROSS_WARN and (i, j) not in flagged:
                    flagged.add((i, j))
                    p = lines[i].interpolate(si)
                    out.append(f"{road_label(decks[i])} and {road_label(decks[j])} cross only {dz:.1f} m "
                               f"apart at ({p.x:.0f}, {p.y:.0f}) (needs {clearance} m)")
    # where one deck's centerline runs on another's pavement (a ramp leaving it) the two
    # have to be level, or a car following it meets the other deck's edge
    widest = max((r.width for r in decks), default=0.0)
    for i, r in enumerate(decks):
        for j in (int(j) for j in tree.query(lines[i].buffer(widest / 2))) if tree else ():
            pair = (min(i, j), max(i, j))
            if j == i or pair in flagged:
                continue
            reach = decks[j].width / 2 - OVERLAP_EDGE
            worst = None
            for x, y, zi in r.nodes:
                p = Point(x, y)
                sj = float(lines[j].project(p))
                if lines[j].distance(p) < reach and 1.0 < sj < arcs[j][-1] - 1.0:
                    dz = zi - z_at(j, sj)
                    if JOINT_STEP < abs(dz) < clearance - CROSS_WARN and (worst is None or abs(dz) > abs(worst[0])):
                        worst = (dz, x, y)
            if worst:
                flagged.add(pair)
                dz, x, y = worst
                out.append(f"{road_label(r)} and {road_label(decks[j])} overlap with a {abs(dz):.1f} m step "
                           f"at ({x:.0f}, {y:.0f})")
    return out


def hillshade(h: np.ndarray, px: float, azimuth=315.0, altitude=45.0) -> np.ndarray:
    """0..1 shaded relief (sun from the north-west by default)."""
    dy, dx = np.gradient(h.astype(np.float64), px)
    slope = np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(dy, -dx)
    az, alt = math.radians(360.0 - azimuth + 90.0), math.radians(altitude)
    shade = math.sin(alt) * np.cos(slope) + math.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(shade, 0.0, 1.0)


def thin(count: int, cap: int) -> np.ndarray:
    """Indices of at most `cap` of `count` items, evenly spread (all of them up to cap)."""
    if count <= cap:
        return np.arange(count)
    return np.arange(cap, dtype=np.int64) * count // cap


def overlay_features(out_dir) -> dict:
    """What the overlay draws from the feature stages: footprint rings with their bounds, and
    tree x, y (empty where the stage has not run)."""
    out_dir = Path(out_dir)
    rings, trees = [], np.empty((0, 2))
    if (out_dir / "buildings.json").exists():
        rings = [b.ring for b in load_buildings(out_dir / "buildings.json") if len(b.ring) >= 3]
    rings += [b.ring for b in drawn_buildings(out_dir / STYLED_FILE) if len(b.ring) >= 3]
    bounds = np.array([(r[:, 0].min(), r[:, 1].min(), r[:, 0].max(), r[:, 1].max()) for r in rings],
                      float).reshape(-1, 4)
    if (out_dir / "trees.json").exists():
        trees = np.array([(t["x"], t["y"]) for t in load_trees(out_dir / "trees.json")], float).reshape(-1, 2)
    return {"rings": rings, "bounds": bounds, "trees": trees}


def render_overlay(cfg, path=None, crop=None, size: int = 2048, out_dir=None, features=None) -> Path:
    """Draw the overlay for the whole map, or for crop = (x, y, half_size_m) in level meters.
    `features` is overlay_features(out_dir), loaded when not given."""
    out_dir = Path(out_dir or cfg.out_dir)
    features = features if features is not None else overlay_features(out_dir)
    path = Path(path) if path else out_dir / "overlay.png"
    extent = Extent.from_config(cfg)
    h = np.load(out_dir / "heights.npy")
    n = h.shape[0]
    px = extent.pixel_size(n)
    half = extent.half_size
    cx, cy, ch = crop if crop else (0.0, 0.0, half)
    # grid window for the view
    c0, r0 = extent.to_pixel(cx - ch, cy + ch, n)
    c1, r1 = extent.to_pixel(cx + ch, cy - ch, n)
    c0, r0 = max(0, int(math.floor(c0 + 0.5))), max(0, int(math.floor(r0 + 0.5)))
    c1, r1 = min(n, int(math.ceil(c1 + 0.5))), min(n, int(math.ceil(r1 + 0.5)))
    shade = Image.fromarray((hillshade(h, px)[r0:r1, c0:c1] * 255).astype(np.uint8)).convert("RGB")
    with Image.open(out_dir / "basecolor.png") as im:
        color = im.convert("RGB").crop((c0, r0, c1, r1))
    img = Image.blend(shade.resize((size, size)), color.resize((size, size)), 0.5)
    draw = ImageDraw.Draw(img)
    scale = size / (2 * ch)  # pixels per meter

    def to_px(x, y):
        return ((np.asarray(x) - (cx - ch)) * scale, ((cy + ch) - np.asarray(y)) * scale)

    if (out_dir / "water.json").exists():
        for body in load_water(out_dir / "water.json"):
            polys = getattr(body.polygon, "geoms", [body.polygon])
            for p in polys:
                xs, ys = to_px(*np.asarray(p.exterior.coords).T)
                draw.line(list(zip(xs, ys)), fill=(40, 120, 255), width=2)
    x0, y0, x1, y1 = cx - ch, cy - ch, cx + ch, cy + ch
    b = features["bounds"]
    for i in np.flatnonzero((b[:, 2] >= x0) & (b[:, 0] <= x1) & (b[:, 3] >= y0) & (b[:, 1] <= y1)):
        xs, ys = to_px(features["rings"][i][:, 0], features["rings"][i][:, 1])
        draw.polygon(list(zip(xs.tolist(), ys.tolist())), fill=FOOTPRINT_COLOUR)
    roads = load_roads(out_dir / "roads.json")
    for road in sorted(roads, key=lambda r: r.is_bridge):  # bridges on top
        xs, ys = to_px(road.nodes[:, 0], road.nodes[:, 1])
        w = max(1, int(round(road.width * scale)))
        color = (230, 30, 30) if road.is_bridge else (255, 255, 255)
        draw.line(list(zip(xs, ys)), fill=color, width=w)
    t = features["trees"]
    t = t[(t[:, 0] >= x0) & (t[:, 0] <= x1) & (t[:, 1] >= y0) & (t[:, 1] <= y1)]
    if len(t):  # on top of the roads, so a tree on one shows
        t = t[thin(len(t), OVERLAY_MAX_TREES)]
        cols, rows = (np.floor(v).astype(np.int64) for v in to_px(t[:, 0], t[:, 1]))
        k = int(min(2.0, scale + 0.5))  # dots 1 px across on the whole map, 3 px on the crops
        pixels = np.array(img)
        for dr in range(-k, k + 1):
            for dc in range(-k, k + 1):
                r, c = rows + dr, cols + dc
                ok = (r >= 0) & (r < size) & (c >= 0) & (c < size)
                pixels[r[ok], c[ok]] = TREE_COLOUR
        img = Image.fromarray(pixels)
        draw = ImageDraw.Draw(img)
    from .level import place_spawns

    for s in place_spawns(cfg, extent, h, roads):
        x, y = to_px(s["x"], s["y"])
        r = max(4, size // 200)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 220, 0), outline=(0, 0, 0))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def run(cfg, out_dir=None) -> list[str]:
    out_dir = Path(out_dir or cfg.out_dir)
    problems = check(cfg, out_dir)
    (out_dir / "problems.txt").write_text("\n".join(problems) if problems else "no problems found",
                                          encoding="utf-8")
    for p in problems[:40]:
        log.warning(p)
    log.info("%d problems (all of them in %s)", len(problems), out_dir / "problems.txt")
    features = overlay_features(out_dir)
    render_overlay(cfg, out_dir / "overlay.png", out_dir=out_dir, features=features)
    from .level import place_spawns

    extent = Extent.from_config(cfg)
    h = np.load(out_dir / "heights.npy")
    for s in place_spawns(cfg, extent, h, load_roads(out_dir / "roads.json")):
        render_overlay(cfg, out_dir / f"overlay_{s['name']}.png", crop=(s["x"], s["y"], 1024.0),
                       size=1024, out_dir=out_dir, features=features)
    return problems
