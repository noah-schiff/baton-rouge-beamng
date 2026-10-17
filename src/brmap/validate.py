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
* trees.json: no tree within 1.5 m of a road corridor or 3 m of a bridge deck's, or
  inside a footprint, and z within 1 m of the terrain.
* The terrain for both is the one the level writes, cut under the bridge decks.
* markings.json: every paint node inside the box.
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

from .bridges import END_FRACTION, JOIN_TOL, MARGIN, TERRAIN_GAP, clear_under_decks, deck_crossings
from .buildings import drop_conflicts, load_buildings, lowest_ground, wet_areas
from .geo import Extent
from .markings import load_markings
from .roads import load_roads
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
TREE_ROAD = 1.5     # m a tree keeps off a road corridor
TREE_DECK = 3.0     # m a tree keeps off a bridge deck's corridor
ON_TERRAIN = 1.0    # m a tree's z or a building's base may sit above or below the terrain
DAE_MAX_VERTS = 60000  # per level of detail of a shape (16-bit indices, with a margin)

FOOTPRINT_COLOUR = (185, 185, 185)
TREE_COLOUR = (30, 190, 60)
OVERLAY_MAX_TREES = 200_000  # more trees in view are thinned to this many dots


def _label(road) -> str:
    ids = ",".join(str(i) for i in road.osm_ids)
    return f"way {ids}" + (f" ({road.name})" if road.name else "")


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
            problems.append(f"{_label(road)}: NaN coordinates")
            continue
        inside = extent.contains(xyz[:, 0], xyz[:, 1])
        if not inside.all():
            i = int(np.argmin(inside))
            problems.append(f"{_label(road)}: node outside the box at ({xyz[i, 0]:.1f}, {xyz[i, 1]:.1f})")
        if road.is_bridge and len(xyz) > 2:
            d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xyz[:, :2], axis=0).T))]
            t = d / d[-1]
            mid = (t >= END_FRACTION) & (t <= 1 - END_FRACTION)
            ground = sample_height(heights, extent, xyz[:, 0], xyz[:, 1])
            gap = xyz[:, 2] - (ground + MARGIN)
            if mid.any() and gap[mid].min() < -CUT_WARN:
                k = int(np.argmin(np.where(mid, gap, np.inf)))
                problems.append(f"{_label(road)}: deck {xyz[k, 2] - ground[k]:.1f} m above the ground at "
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
    """The checks on buildings, trees, paint and shapes, each run only when its input exists."""
    out_dir = Path(out_dir)
    paths = {name: out_dir / f"{name}.json" for name in ("buildings", "trees", "markings")}
    buildings = load_buildings(paths["buildings"]) if paths["buildings"].exists() else []
    trees = load_trees(paths["trees"]) if paths["trees"].exists() else []
    usable = [r for r in roads if len(r.nodes) >= 2 and np.isfinite(r.nodes).all()]
    problems: list[str] = []
    if buildings or trees:
        terrain, _ = clear_under_decks(heights, extent, roads)  # what the level writes to the .ter
        if buildings:
            water = load_water(out_dir / "water.json") if (out_dir / "water.json").exists() else []
            problems += building_problems(buildings, usable, terrain, extent, water)
        if trees:
            problems += tree_problems(trees, buildings, usable, terrain, extent)
    if paths["markings"].exists():
        problems += marking_problems(load_markings(paths["markings"]), extent)
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
            out.append(f"{name}: footprint on the corridor of {_label(roads[on_road[i]])} "
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
                why.setdefault(t, f"{what} {_label(group[g])}")
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
                    out.append(f"{_label(r)} ends {abs(zi - zj):.1f} m {'above' if zi > zj else 'below'} "
                               f"{_label(decks[j])} where it joins it at {where}")
                elif min(sj, arcs[j][-1] - sj) < JOIN_TOL:
                    dip = min(inward(i, arcs[i][-1] if end else 0.0), inward(j, sj)) - max(zi, zj)
                    if dip > JOINT_DIP:
                        out.append(f"{_label(r)} and {_label(decks[j])} dip {dip:.1f} m to their "
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
                    out.append(f"{_label(decks[i])} and {_label(decks[j])} cross only {dz:.1f} m "
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
                out.append(f"{_label(r)} and {_label(decks[j])} overlap with a {abs(dz):.1f} m step "
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
