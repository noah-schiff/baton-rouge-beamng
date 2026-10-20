"""Buildings: OpenStreetMap and Microsoft footprints with lidar heights, as tiled LOD meshes.

Reads raw/osm_features.json and, when cfg.buildings_microsoft and the file exists,
raw/ms_buildings.json (both from features.py); the 2 m canopy grid (canopy.load_canopy);
out/roads.json, out/water.json and out/heights.npy. Writes out/buildings.json, a list of
Building.to_dict(). level_content, the level.FEATURE_MODULES hook, turns that into tiles
of COLLADA meshes (dae.py), each placed by one TSStatic. When out/landmarks.json exists, the
buildings the landmarks take the place of (landmarks.removed, with the models of out/models)
are left out, and the count per landmark logged. The restyled ones (the LSU campus core,
src/brmap/data/styled.yaml) go to out/styled.json instead (split_styled), which styled.py draws.

* split_styled(blds, groups) -> (rest, moved): each building whose OSM element is in a styled
  group (its parts '/<k>' and tower parts '/t<k>' with it) moves to
  [{"style", "group", "building": Building.to_dict()}]; run writes them to out/styled.json
  (always, an empty list when none) and logs the group's ids not in the box.

* footprints(osm_features, ms_list, extent) -> [(id, source, kind, Polygon, tags, ms_height)]:
  OSM ways and multipolygons with building=*, except building in {roof, carport,
  construction, ruins, no} and building:part; plus the Microsoft footprints that overlap
  the OSM ones (the left-out types included, so a canopy stays out) by less than 30 % of
  their own area. In level coordinates, simplified at 0.5 m, counter-clockwise. Broken
  rings are repaired with shapely.make_valid and each polygon part kept (a part of a
  several-part footprint gets "/<k>" on its id); what can't be repaired is skipped and
  logged. Under 15 m2, or not entirely inside +-(half - 1 m), is dropped. Two OSM
  footprints overlapping by 90 % or more of the smaller one's area are one building drawn
  twice (coplanar walls and roofs would z-fight): only the larger is kept, on a tie the
  first in id order (by number: w9 before w20), and the count is logged.
* drop_conflicts(polys, roads, water) -> (kept, {reason: [id, ...]}): a footprint shrunk
  0.5 m must not touch any road corridor (centreline buffered by width / 2, every class;
  reason "road", or "bridge" for a deck). A covered road (a passage under a building) does
  not count, and a footprint of 500 m2 or more keeps the service drives on the ground that
  only end in it: none passes through, none has over 30 m of centreline inside, and together
  they cover under 15 % of it. Every bridge deck counts, service ones included. One with half
  its area or more in a water body goes too ("water").
* building_height(poly, tags, ms_height, kind, ndsm, ndvi, extent, px) -> (h, source), the
  first that applies: lidar (p75 of ndsm over the footprint's cells with ndvi < 0.3, at
  least 3 cells, 2.5-150 m), OSM height, building:levels x 3.2 + 1, Microsoft height > 0,
  then a type default. Clamped to 2.5-150 m. NaN canopy (the lidar gap) falls through to
  the tags and defaults.
* tower_parts(poly, ndsm, ndvi, extent, px, base_height) -> [(Polygon, height)]: the towers
  the lidar shows rising well above a large footprint's roof (make_buildings adds each as
  '<id>/t<k>'). The traced cells get clean outlines from orthogonalize(poly, axis, tol,
  max_corners, rect_cover): right angles only, with at most 12 corners, along the podium's long
  axis, snapped to their bounding rectangle when they cover RECT_COVER of it (the styled
  buildings pass rect_cover=None to keep their OSM outlines). The
  part's own axis takes over only for a tower turned on its podium: the two axes are not
  both within 10 degrees of the lidar grid (whose staircase pulls a traced axis onto it), and
  the outline along the own axis fits the cells better, both as it is and with the two cut to
  the same number of corners.
* Base z (make_building, in run): the lowest terrain along the outline minus 0.3 m, from
  heights.npy (lowest_ground): the ring and its holes sampled every 2 m or less, not just
  at the vertices, leaving out the points in a water body (grown by 1.5 terrain cells, so
  no sample takes in the carved bed) unless all of them are. A tower part stands on its
  podium's base. level_content (settle) lowers a base where the level cut the terrain
  under a bridge deck, by the same rule and with out/water.json, keeping the roof where
  it was; a tower part goes down with its podium.
* roof_type(kind, source, poly, height): hipped on small residential footprints (house,
  detached, or Microsoft; under 300 m2 and 9 m, no holes, at least 85 % of their rotated
  rectangle, and tall enough to keep 2.5 m of wall under a 1 m rise); flat otherwise.
  wall_class(kind, height): glass over 20 m, else by type; building_colour picks a
  vertex colour per building from a small palette for its wall class.
* tiles(buildings, tile_m, max_verts=60000) -> [(key, origin_xy, [Building, ...])]: squares
  of cfg.building_tile_m on a grid through the level origin, keyed e<col>n<row> (w and s
  for negative columns and rows, so the tile west of and south of the origin is w01s01).
  A building goes to the tile holding its bounding-box centre. A tile whose meshes would
  pass max_verts in any level of detail is split into quarters, recursively, suffixed
  _sw, _se, _nw and _ne.
* MATERIALS: the br_bld_* Material entries, written to art/shapes/buildings/main.materials.json.
* level_content(cfg, data_dir, root, heights, extent) -> {"buildings": [TSStatic, ...]}: writes
  art/shapes/buildings/br_bld_<key>.dae per tile and the materials under root. Each
  TSStatic has shapeName /levels/<cfg.level_name>/art/shapes/buildings/br_bld_<key>.dae,
  position = the tile origin with z 0, no rotationMatrix, useInstanceRenderData true,
  collisionType "Collision Mesh", decalType "None" and persistentId
  bf.persistent_id("bld", key).
"""
from __future__ import annotations

import dataclasses
import json
import logging
import math
import re
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio.features
import shapely
from affine import Affine
from shapely.geometry import LineString, Polygon, shape
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from . import beamng_format as bf
from . import dae
from .canopy import RES, load_canopy
from .geo import Extent
from .roads import load_roads
from .terrain import load_heights, sample_height
from .water import load_water

log = logging.getLogger(__name__)

OUT_DIR = Path("data/out")
SOURCES = ("osm", "ms")
ROOFS = ("flat", "hip")
WALLS = ("res", "brick", "concrete", "glass")
HEIGHT_SOURCES = ("lidar", "osm_height", "levels", "ms", "default")

# footprints
EXCLUDED_KINDS = {"roof", "carport", "construction", "ruins", "no"}
MIN_AREA = 15.0          # m2
EDGE_MARGIN = 1.0        # m; a building must lie inside +-(half - this)
SIMPLIFY_TOL = 0.5       # m
MS_MAX_OVERLAP = 0.3     # a Microsoft footprint this much under OSM ones is the same building
OSM_DUP_OVERLAP = 0.9    # two OSM footprints overlapping by this much of the smaller are one building
SHRINK = 0.5             # m taken off a footprint before testing it against the road corridors
WATER_SHARE = 0.5        # a footprint with this share of its area in water is dropped
SERVICE_MIN_AREA = 500.0 # m2; a building this big may have a service drive ending in it...
SERVICE_SHARE = 0.15     # ...covering less than this share of it (the Capitol's is 6 %)
SERVICE_MAX_INSIDE = 30.0  # m; ...with at most this much of each drive's centreline inside it
THROUGH_TOL = 1.0        # m; a centreline piece with both ends this near the outline passes through

# heights
NDVI_MAX = 0.3           # cells greener than this are trees over the roof, not the roof
LIDAR_MIN_CELLS = 3
LIDAR_PERCENTILE = 75
MIN_HEIGHT, MAX_HEIGHT = 2.5, 150.0  # m
LEVEL_HEIGHT, LEVELS_EXTRA = 3.2, 1.0  # building:levels x 3.2 + 1 m
FOOT = 0.3048            # m
DEFAULT_HEIGHT = {"house": 6.0, "detached": 6.0, "semidetached_house": 6.0, "apartments": 12.0,
                  "commercial": 8.0, "retail": 8.0, "office": 8.0, "industrial": 9.0, "warehouse": 9.0,
                  "church": 12.0, "school": 9.0, "university": 9.0, "college": 9.0, "shed": 3.0,
                  "garage": 3.0, "garages": 3.0}
OTHER_HEIGHT = 5.0
BASE_DROP = 0.3          # m below the lowest ground at the footprint, so no edge floats
BASE_STEP = 2.0          # m; the outline's ground is sampled at least this often, not just at the corners
WET_PAD = 1.5            # terrain cells round a water body within which a sample takes in its carved bed

# roofs and walls
HIP_KINDS = {"house", "detached"}
HIP_MAX_AREA = 300.0     # m2
HIP_MAX_HEIGHT = 9.0     # m
HIP_MIN_COVER = 0.85     # footprint area / its minimum rotated rectangle's area
HIP_MIN_HEIGHT = dae.MIN_WALL + 1.0  # room for the walls under at least a 1 m rise
GLASS_HEIGHT = 20.0      # m; anything taller gets glass walls
SMALL_HEIGHT = 8.0       # m; an untyped building (building=yes) lower than this is a house...
LARGE_AREA = 400.0       # m2; ...unless its footprint is this big (a warehouse or a store)
STONE_KINDS = {"public", "civic", "government", "townhall", "church", "cathedral", "courthouse"}  # tall ones aren't glass
TOWER_MIN_AREA = 600.0   # m2; smaller footprints have no separate tower
TOWER_MIN_RISE = 12.0    # m from the roof (p75) to the top (p98): less is rooftop plant
TOWER_TOP_PERCENTILE = 98
TOWER_HEIGHT_PERCENTILE = 90
TOWER_MIN_PART = 60.0    # m2
TOWER_CORNERS = 12       # most corners a tower part's outline keeps
TOWER_GRID_PULL = math.radians(10)  # a traced part's own axis this near the lidar grid may be the grid's doing
STEP_TOL = 1.0           # m; orthogonalize takes out steps shorter than this
RECT_COVER = 0.85        # an orthogonalized outline covering this share of its bounding rectangle along the axis becomes it
SNAP_TIE = 1e-6          # rad; an edge at exactly 45 degrees to the axis snaps as if the axis were turned this much
RES_KINDS = {"house", "detached", "semidetached_house", "residential", "terrace", "bungalow",
             "cabin", "static_caravan", "farm", "hut", "shed", "garage"}
BRICK_KINDS = {"apartments", "dormitory", "church", "chapel", "cathedral", "school", "university",
               "college", "kindergarten", "public", "civic", "government", "hospital", "hotel",
               "fire_station", "townhall"}
PALETTES = {  # multiplied into the wall texture (vertColor)
    "res": ((0.97, 0.96, 0.93), (0.95, 0.90, 0.80), (0.86, 0.89, 0.91), (0.91, 0.87, 0.76),
            (0.84, 0.88, 0.82), (0.96, 0.93, 0.87), (0.80, 0.76, 0.70)),
    "brick": ((1.0, 1.0, 1.0), (0.93, 0.87, 0.83), (0.85, 0.80, 0.78), (0.96, 0.91, 0.86)),
    "concrete": ((0.96, 0.95, 0.93), (0.89, 0.87, 0.83), (0.80, 0.80, 0.80), (0.93, 0.89, 0.81),
                 (0.86, 0.83, 0.77)),
    "glass": ((0.86, 0.91, 0.96), (0.80, 0.84, 0.88), (0.91, 0.93, 0.95), (0.76, 0.81, 0.83)),
}

# tiles and level files
MAX_VERTS = 60000        # per level of detail and tile (the spec's budget)
MIN_TILE = 16.0          # m; tiles are not split below this
QUARTERS = (("sw", -1, -1), ("se", 1, -1), ("nw", -1, 1), ("ne", 1, 1))
SHAPES_DIR = "art/shapes/buildings"
MATERIALS_FILE = f"{SHAPES_DIR}/main.materials.json"
STYLED_FILE = "styled.json"  # the restyled buildings (split_styled), which styled.py draws
LOG_IDS = 10             # building ids listed per landmark in the log

# Shared texture sets, checked in content/assets/materials/tileable.zip (0.39.4): every map
# listed is shipped there (as .dds; the game resolves the .png names). windows1 has a
# metallic map and no ambient occlusion. The highrise atlas (atlas.zip) is not used: an
# atlas can't tile with the 3 m x 3.2 m wall UVs.
# name -> (folder under /assets/materials/tileable, file stem, maps, groundType, extra fields)
TEXTURE_SETS = {
    "br_bld_wall_res": ("wood/weatherboards", "t_weatherboards", "b nm r ao", "WOOD", {}),
    "br_bld_wall_brick": ("brick/brick_plain", "t_bricks_red_01", "b nm r ao", "CONCRETE", {}),
    "br_bld_wall_concrete": ("concrete/concrete_plain", "t_concrete_plain", "b nm r ao", "CONCRETE", {}),
    # metallic like the stock windows1 material (automation_test_track), reflecting the sky
    "br_bld_wall_glass": ("glass/windows1", "t_bld_ind_windows", "b nm r m", "CONCRETE",
                          {"metallicFactor": 1.0}),
    "br_bld_roof_shingle": ("roofing/asphalt_roof", "t_asphalt_roof", "b nm r ao", "ASPHALT", {}),
    "br_bld_roof_flat": ("concrete/concrete_plain", "t_concrete_plain", "b nm r ao", "CONCRETE", {}),
}
_MAP_FIELDS = {"b": ("baseColorMap", "b.color"), "nm": ("normalMap", "nm.normal"),
               "r": ("roughnessMap", "r.data"), "ao": ("ambientOcclusionMap", "ao.data"),
               "m": ("metallicMap", "m.data")}


def _material(name: str) -> dict:
    folder, stem, maps, ground, extra = TEXTURE_SETS[name]
    base = f"/assets/materials/tileable/{folder}/{stem}_"
    stage = {field: f"{base}{suffix}.png" for field, suffix in (_MAP_FIELDS[m] for m in maps.split())}
    m = {"name": name, "mapTo": name, "class": "Material", "persistentId": bf.persistent_id("material", name),
         "Stages": [{**stage, **extra, "vertColor": True}, {}, {}, {}], "annotation": "BUILDINGS",
         "groundType": ground, "materialTag0": "beamng", "materialTag1": "building", "version": 1.5}
    if name == "br_bld_wall_glass":
        m["dynamicCubemap"] = True
    return m


MATERIALS = {name: _material(name) for name in TEXTURE_SETS}


def _xy(points) -> np.ndarray:
    return np.asarray(points, dtype=float).reshape(-1, 2)


@dataclass
class Building:
    id: str                              # OSM "w123" / "r45"; Microsoft footprints get their own
    source: str                          # one of SOURCES
    kind: str                            # the OSM building=* value
    ring: np.ndarray                     # (N, 2) level x, y; counter-clockwise, not closed
    holes: list[np.ndarray]              # inner rings, (M, 2) each, not closed
    height: float                        # m from base_z to the top of the roof
    base_z: float                        # m: lowest dry terrain along the outline, minus 0.3
    roof: str                            # one of ROOFS
    wall: str                            # one of WALLS
    colour: tuple[float, float, float]   # vertex colour, 0..1
    height_source: str                   # one of HEIGHT_SOURCES

    def to_dict(self) -> dict:
        return {"id": self.id, "source": self.source, "kind": self.kind,
                "ring": [[float(x), float(y)] for x, y in self.ring],
                "holes": [[[float(x), float(y)] for x, y in h] for h in self.holes],
                "height": float(self.height), "base_z": float(self.base_z), "roof": self.roof,
                "wall": self.wall, "colour": [float(c) for c in self.colour],
                "height_source": self.height_source}

    @classmethod
    def from_dict(cls, d: dict) -> "Building":
        return cls(id=str(d["id"]), source=d["source"], kind=d["kind"], ring=_xy(d["ring"]),
                   holes=[_xy(h) for h in d.get("holes") or []], height=float(d["height"]),
                   base_z=float(d["base_z"]), roof=d["roof"], wall=d["wall"],
                   colour=tuple(float(c) for c in d["colour"]), height_source=d["height_source"])


def save_buildings(buildings: list[Building], path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([b.to_dict() for b in buildings]), encoding="utf-8")
    return path


def load_buildings(path=OUT_DIR / "buildings.json") -> list[Building]:
    return [Building.from_dict(d) for d in json.loads(Path(path).read_text(encoding="utf-8"))]


# ------------------------------------------------------------------ footprints

def _to_level_rings(extent: Extent, rings: list) -> list[np.ndarray]:
    """[[lon, lat], ...] rings -> (N, 2) level arrays, projected in one call."""
    arrays = [np.asarray(r, dtype=float).reshape(-1, 2) for r in rings]
    pts = np.concatenate(arrays) if arrays else np.zeros((0, 2))
    if not len(pts):
        return arrays
    x, y = extent.to_level(pts[:, 0], pts[:, 1])
    return np.split(np.column_stack([x, y]), np.cumsum([len(a) for a in arrays])[:-1])


def _polygon_parts(geom) -> list[Polygon]:
    if geom.geom_type == "Polygon":
        return [geom] if not geom.is_empty else []
    return [p for g in getattr(geom, "geoms", []) for p in _polygon_parts(g)]


def _repair(outer: np.ndarray, inners: list[np.ndarray]) -> list[Polygon]:
    """The valid polygon parts of a ring with holes, simplified at SIMPLIFY_TOL and
    counter-clockwise; [] when nothing with an area is left. A ring with a NaN or infinite
    coordinate (GEOS can't repair one) is left out: the outer means the whole footprint."""
    if len(outer) < 3 or not np.isfinite(outer).all():
        return []
    try:
        poly = Polygon(outer, [h for h in inners if len(h) >= 3 and np.isfinite(h).all()])
        if not poly.is_valid:
            poly = shapely.make_valid(poly)
        out = []
        for part in _polygon_parts(poly):
            simple = part.simplify(SIMPLIFY_TOL, preserve_topology=True)
            if not simple.is_valid:
                simple = shapely.make_valid(simple)
            out += [orient(p, 1.0) for p in _polygon_parts(simple) if p.area > 0]
    except (ValueError, shapely.errors.GEOSException):
        return []
    return out


def _inside(poly: Polygon, limit: float) -> bool:
    minx, miny, maxx, maxy = poly.bounds
    return -limit <= minx and maxx <= limit and -limit <= miny and maxy <= limit


def _id_key(pid: str) -> tuple:
    """Sorts ids by their numbers: w9 before w20, w1/2 before w1/10."""
    return tuple(int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(pid)))


def _duplicates(fps: list[tuple]) -> set[int]:
    """The indices of the footprints that overlap a larger one (on a tie, one earlier in id
    order) by OSM_DUP_OVERLAP or more of their own area: the same building drawn twice."""
    if len(fps) < 2:
        return set()
    polys = np.array([f[3] for f in fps], dtype=object)
    areas = shapely.area(polys)
    rank = np.empty(len(fps), int)
    rank[sorted(range(len(fps)), key=lambda k: (-areas[k], _id_key(fps[k][0])))] = np.arange(len(fps))
    a, b = shapely.STRtree(polys).query(polys, predicate="intersects")
    a, b = a[rank[b] < rank[a]], b[rank[b] < rank[a]]  # a is the smaller (or later) of the pair
    if not len(a):
        return set()
    shared = shapely.area(shapely.intersection(polys[a], polys[b]))
    return set(a[shared >= OSM_DUP_OVERLAP * areas[a]].tolist())


def footprints(osm_features: dict, ms_list: list, extent: Extent) -> list[tuple]:
    """(id, source, kind, Polygon, tags, ms_height) per building footprint; see the module docstring."""
    limit = extent.half_size - EDGE_MARGIN
    stats: Counter = Counter()
    out: list[tuple] = []

    def keep(pid, source, kind, poly, tags, ms_height) -> bool:
        if poly.area < MIN_AREA:
            stats["small"] += 1
        elif not _inside(poly, limit):
            stats["edge"] += 1
        else:
            out.append((pid, source, kind, poly, tags, ms_height))
            return True
        return False

    def parts_of(eid, polygons):
        parts = [p for outer, inners in polygons for p in _repair(outer, inners)]
        if not parts:
            stats["skipped"] += 1
            log.debug("skipped building %s: no polygon left after repair", eid)
        return [(eid if len(parts) == 1 else f"{eid}/{k}", p) for k, p in enumerate(parts)]

    # OpenStreetMap, with every ring projected at once
    wanted, rings = [], []
    for el in osm_features.get("buildings") or []:
        tags = el.get("tags") or {}
        kind = str(tags.get("building", "yes")).strip()
        excluded = kind in EXCLUDED_KINDS or "building:part" in tags
        stats["excluded"] += excluded
        polys = []
        for p in el.get("polygons") or []:
            polys.append((len(rings), len(p.get("inner") or [])))
            rings += [p.get("outer") or []] + list(p.get("inner") or [])
        wanted.append((str(el.get("id")), tags, kind, polys, excluded))
    level = _to_level_rings(extent, rings)
    # every OpenStreetMap footprint, the left-out types too: a Microsoft footprint under a
    # fuel-station canopy (building=roof) is that canopy, and must not come back solid
    osm_polys = []
    for eid, tags, kind, polys, excluded in wanted:
        outlines = [(level[i], level[i + 1:i + 1 + k]) for i, k in polys]
        if excluded:
            osm_polys += [p for outer, inners in outlines for p in _repair(outer, inners)]
            continue
        for pid, poly in parts_of(eid, outlines):
            osm_polys.append(poly)
            stats["osm"] += keep(pid, "osm", kind, poly, tags, None)
    dups = _duplicates(out)  # the same building drawn twice: coplanar walls and roofs would z-fight
    out[:] = [f for k, f in enumerate(out) if k not in dups]
    stats["osm"] -= len(dups)
    stats["osm_dup"] = len(dups)
    tree = shapely.STRtree(osm_polys)

    # Microsoft, where OpenStreetMap doesn't have the building already
    level = _to_level_rings(extent, [m.get("ring") or [] for m in ms_list])
    for i, (m, ring) in enumerate(zip(ms_list, level)):
        height = m.get("height")
        height = None if height is None else float(height)
        for pid, poly in parts_of(f"ms{i}", [(ring, [])]):
            near = tree.query(poly, predicate="intersects")
            if len(near) and (poly.intersection(unary_union([osm_polys[j] for j in near])).area
                              >= MS_MAX_OVERLAP * poly.area):
                stats["ms_in_osm"] += 1
                continue
            stats["ms"] += keep(pid, "ms", "yes", poly, {}, height)
    log.info("footprints: %d OpenStreetMap, %d Microsoft; left out %d excluded types, %d OpenStreetMap "
             "duplicates, %d Microsoft under OpenStreetMap ones, %d under %.0f m2, %d at the box edge; "
             "skipped %d that could not be repaired", stats["osm"], stats["ms"], stats["excluded"],
             stats["osm_dup"], stats["ms_in_osm"], stats["small"], MIN_AREA, stats["edge"], stats["skipped"])
    return out


# ------------------------------------------------------------------ conflicts

def _corridors(roads) -> dict[str, list]:
    """Road corridors (centreline buffered by width / 2): {"road": [...], "bridge": [...]}."""
    out: dict[str, list] = {"road": [], "bridge": []}
    for r in roads:
        if len(r.nodes) < 2:
            continue
        line = LineString(r.nodes[:, :2])
        if line.length > 0:
            out["bridge" if r.is_bridge else "road"].append(line.buffer(r.width / 2.0))
    return out


def _passes_through(line: LineString, poly: Polygon) -> bool:
    """True when a piece of the centreline inside poly enters and leaves it (both ends of the
    piece on the outline), so a car on it would have to drive through the building."""
    for piece in getattr(line.intersection(poly), "geoms", None) or [line.intersection(poly)]:
        if piece.geom_type == "LineString" and piece.length > THROUGH_TOL:  # not just touching the wall
            a, b = shapely.points(np.asarray(piece.coords)[[0, -1], :2])
            if poly.exterior.distance(a) <= THROUGH_TOL and poly.exterior.distance(b) <= THROUGH_TOL:
                return True
    return False


def _service_ends_in(i: int, js, shapes, service, lines) -> bool:
    """True when footprint i is big, the service roads js only end inside it (none passes
    through, none has more than SERVICE_MAX_INSIDE m of centreline inside it) and together
    cover under SERVICE_SHARE of it: a drive up to or under a building, not a lane in its lot."""
    poly = shapes[i]
    if poly.area < SERVICE_MIN_AREA or any(_passes_through(lines[j], poly) for j in js):
        return False
    if any(lines[j].intersection(poly).length > SERVICE_MAX_INSIDE for j in js):
        return False
    return sum(poly.intersection(service[j]).area for j in js) < SERVICE_SHARE * poly.area


def drop_conflicts(polys: list[tuple], roads, water) -> tuple[list[tuple], dict[str, list[str]]]:
    """The footprints (as footprints() returns them) clear of roads, bridge decks and water,
    and the ids dropped by reason."""
    if not polys:
        return [], {}
    shapes = np.array([p[3] for p in polys], dtype=object)
    cores = shapely.buffer(shapes, -SHRINK)
    empty = shapely.is_empty(cores)
    cores[empty] = shapes[empty]  # a sliver thinner than the shrink is tested whole
    reason: dict[int, str] = {}
    roads = [r for r in roads if not r.covered]  # a passage under a building is part of it
    # every deck counts, a service one too; only service drives on the ground may end in a building
    for why, geoms in _corridors([r for r in roads if r.kind != "service" or r.is_bridge]).items():
        if geoms:
            for i in shapely.STRtree(geoms).query(cores, predicate="intersects")[0]:
                reason.setdefault(int(i), why)
    svc = [r for r in roads if r.kind == "service" and not r.is_bridge and len(r.nodes) >= 2
           and LineString(r.nodes[:, :2]).length > 0]
    if svc:
        lines = [LineString(r.nodes[:, :2]) for r in svc]
        service = [ln.buffer(r.width / 2.0) for ln, r in zip(lines, svc)]
        hits: dict[int, list[int]] = {}
        for i, j in zip(*shapely.STRtree(service).query(cores, predicate="intersects")):
            hits.setdefault(int(i), []).append(int(j))
        for i, js in hits.items():
            if i not in reason and not _service_ends_in(i, js, shapes, service, lines):
                reason[i] = "road"
    bodies = [w.polygon for w in water]
    if bodies:
        wet: Counter = Counter()
        for i, j in zip(*shapely.STRtree(bodies).query(shapes, predicate="intersects")):
            if int(i) not in reason:
                wet[int(i)] += shapes[i].intersection(bodies[j]).area
        for i, area in wet.items():
            if area >= WATER_SHARE * shapes[i].area:
                reason[i] = "water"
    dropped: dict[str, list[str]] = {}
    for i in sorted(reason):
        dropped.setdefault(reason[i], []).append(polys[i][0])
    return [p for i, p in enumerate(polys) if i not in reason], dropped


# ------------------------------------------------------------------ heights, roofs, walls

def _number(value) -> float | None:
    """The largest number in an OSM value such as '12', '12 m', "40'", '40 ft' or '3;4' (feet
    turned into metres); None when nothing parses."""
    if value is None:
        return None
    best = None
    for part in str(value).split(";"):
        m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(m|ft|')?\s*", part.lower())
        if m:
            v = float(m[1]) * (FOOT if m[2] in ("ft", "'") else 1.0)
            best = v if best is None else max(best, v)
    return best


def _roof_cells(poly: Polygon, ndsm, ndvi, extent: Extent, px: float):
    """(window heights with NaN off the roof, the window's (row, col) origin) for the analysis
    cells inside poly that aren't vegetation, or None when there are none."""
    if ndsm is None or ndvi is None:
        return None
    n, half = ndsm.shape[0], extent.half_size
    minx, miny, maxx, maxy = poly.bounds
    c0, c1 = max(0, math.floor((minx + half) / px)), min(n, math.ceil((maxx + half) / px))
    r0, r1 = max(0, math.floor((half - maxy) / px)), min(n, math.ceil((half - miny) / px))
    if c1 <= c0 or r1 <= r0:
        return None
    X, Y = np.meshgrid(-half + (np.arange(c0, c1) + 0.5) * px, half - (np.arange(r0, r1) + 0.5) * px)
    h, v = ndsm[r0:r1, c0:c1], ndvi[r0:r1, c0:c1]
    roof = shapely.contains_xy(poly, X, Y) & np.isfinite(h) & (v < NDVI_MAX)  # NaN NDVI: unknown, not roof
    return np.where(roof, h, np.nan), (r0, c0)


def _lidar_height(poly: Polygon, ndsm, ndvi, extent: Extent, px: float) -> float | None:
    """p75 of ndsm over the analysis cells inside poly that aren't vegetation, or None."""
    cells = _roof_cells(poly, ndsm, ndvi, extent, px)
    if cells is None:
        return None
    h = cells[0][np.isfinite(cells[0])]
    if h.size < LIDAR_MIN_CELLS:
        return None
    value = float(np.percentile(h, LIDAR_PERCENTILE))
    return value if MIN_HEIGHT <= value <= MAX_HEIGHT else None


def _long_axis(geom) -> float:
    """The direction of the long side of geom's minimum rotated rectangle, in radians modulo pi
    (landmarks.py turns a landmark along it too)."""
    corners = np.asarray(geom.minimum_rotated_rectangle.exterior.coords)[:3]
    a, b = corners[1] - corners[0], corners[2] - corners[1]
    edge = a if math.hypot(*a) >= math.hypot(*b) else b
    return math.atan2(edge[1], edge[0]) % math.pi


def _orthogonal_ring(xy: np.ndarray, tol: float, max_corners: int | None) -> np.ndarray | None:
    """The corners (N, 2) of the ring xy ((M, 2), open, turned so that the axis is +x) with
    every edge horizontal or vertical, as orthogonalize describes; None when fewer than four
    edges would be left."""
    d = np.roll(xy, -1, axis=0) - xy
    length = np.hypot(d[:, 0], d[:, 1])
    xy, d, length = xy[length > 0], d[length > 0], length[length > 0]
    angle = (np.arctan2(d[:, 1], d[:, 0]) + SNAP_TIE) % np.pi
    vertical = (angle >= np.pi / 4) & (angle < 3 * np.pi / 4)  # the larger component is y
    if vertical.all() or not vertical.any():
        return None
    mid = np.where(vertical, xy[:, 0] + d[:, 0] / 2, xy[:, 1] + d[:, 1] / 2)  # x of a vertical edge, y of a horizontal
    lines: list[list] = []  # [vertical, sum of length * mid, sum of length] per run of edges snapped alike
    for i in np.roll(np.arange(len(xy)), -int(np.flatnonzero(vertical != np.roll(vertical, 1))[0])):
        if lines and lines[-1][0] == vertical[i]:
            lines[-1][1] += length[i] * mid[i]
            lines[-1][2] += length[i]
        else:
            lines.append([bool(vertical[i]), length[i] * mid[i], length[i]])
    while len(lines) > 4:  # a step: line k, from line k - 1 to line k + 1 (both the other way)
        at = np.array([s / w for _, s, w in lines])
        step = np.abs(np.roll(at, -1) - np.roll(at, 1))
        k = int(np.argmin(step))
        if step[k] >= tol and (max_corners is None or len(lines) <= max_corners):
            break
        lines = lines[k - 1:] + lines[:k - 1]  # line k - 1 first (the last one when k is 0)
        before, after = lines[0], lines[2]
        lines = [[before[0], before[1] + after[1], before[2] + after[2]]] + lines[3:]
    if len(lines) < 4:
        return None
    at = [s / w for _, s, w in lines]
    return np.array([(a, b) if v else (b, a) for (v, _, _), a, b in zip(lines, at, at[1:] + at[:1])])


def orthogonalize(poly: Polygon, axis: float | None = None, tol: float = STEP_TOL,
                  max_corners: int | None = None, rect_cover: float | None = RECT_COVER) -> Polygon:
    """poly with right-angled corners only, its edges along and across axis (radians from +x;
    by default the long axis of poly's minimum rotated rectangle). Turned by -axis, every edge
    (the holes' too) snaps to horizontal or vertical by its larger component, each run of edges
    snapped the same way becomes one line at their length-weighted mean, and the steps (an edge
    between two such lines) shorter than tol are taken out, shortest first, the two lines either
    side becoming one; while a ring has more than max_corners corners, so is its shortest step
    of any length. Turned back, the result's bounding rectangle along axis comes back instead
    when the result covers rect_cover (RECT_COVER by default; None: never) of it, and poly's
    minimum rotated rectangle when the result is not a valid polygon. A hole with fewer than
    four edges left is dropped. (RECT_COVER was tuned on the lidar-traced tower parts; the
    styled buildings keep their OSM outlines with None.)"""
    rect = poly.minimum_rotated_rectangle
    if rect.geom_type != "Polygon":  # nothing with an area
        return poly
    if axis is None:
        axis = _long_axis(poly)
    centre = np.asarray(rect.centroid.coords[0])
    cos, sin = math.cos(axis), math.sin(axis)
    turn = np.array([[cos, -sin], [sin, cos]])  # (row vector) @ turn turns it by -axis
    outer, *holes = [_orthogonal_ring((np.asarray(r.coords)[:-1, :2] - centre) @ turn, tol, max_corners)
                     for r in (poly.exterior, *poly.interiors)]
    if outer is None:
        return rect
    out = Polygon(outer @ turn.T + centre, [h @ turn.T + centre for h in holes if h is not None])
    if not out.is_valid or out.area <= 0:
        return rect
    if rect_cover is None:
        return out
    (x0, y0), (x1, y1) = outer.min(axis=0), outer.max(axis=0)
    around = Polygon(np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]) @ turn.T + centre)
    return around if out.area >= rect_cover * around.area else out


def _off_grid(angle: float) -> float:
    """How far (radians) the direction angle is from the lidar grid's (level x or y)."""
    a = angle % (math.pi / 2)
    return min(a, math.pi / 2 - a)


def _fit(outline: Polygon, part: Polygon) -> float:
    """outline's intersection over union with part."""
    return outline.intersection(part).area / (outline.union(part).area or 1.0)


def _tower_outline(part: Polygon, axis: float) -> Polygon:
    """A tower part traced from the lidar cells, orthogonalized with at most TOWER_CORNERS
    corners along the podium's long axis (axis). The part's own axis (its minimum rotated
    rectangle's) takes over only for a tower turned on its podium, whose axis then says little
    (a round or rambling podium). The cells' 2 m staircase misleads it two ways, so it must
    clear both: the two axes are not both within TOWER_GRID_PULL of the grid (the staircase of
    a tower a few degrees off the grid traces to an axis on it, and an outline along the grid
    fits those cells better); and the outline along it fits the cells better (_fit) both as it
    is and with the two cut to the same number of corners (along the grid, extra corners can
    follow the staircase of a tower well off it)."""
    podium = orthogonalize(part, axis, max_corners=TOWER_CORNERS)
    own_axis = _long_axis(part)
    if max(_off_grid(axis), _off_grid(own_axis)) <= TOWER_GRID_PULL:
        return podium
    own = orthogonalize(part, own_axis, max_corners=TOWER_CORNERS)
    same = min(len(podium.exterior.coords), len(own.exterior.coords)) - 1
    pairs = [(podium, own), (orthogonalize(part, axis, max_corners=same),
                             orthogonalize(part, own_axis, max_corners=same))]
    return own if all(_fit(o, part) > _fit(p, part) for p, o in pairs) else podium


def tower_parts(poly: Polygon, ndsm, ndvi, extent: Extent, px: float, base_height: float) -> list[tuple]:
    """[(Polygon, height)] of the towers rising well above a large footprint's roof (the
    Capitol's tower on its wings, a downtown tower on its podium), from the lidar: the roof
    cells above half way from the base height to the top, traced inside the footprint and
    given clean right-angled outlines (_tower_outline)."""
    if poly.area < TOWER_MIN_AREA:
        return []
    cells = _roof_cells(poly, ndsm, ndvi, extent, px)
    if cells is None:
        return []
    h, (r0, c0) = cells
    valid = np.isfinite(h)
    if valid.sum() < LIDAR_MIN_CELLS:
        return []
    top = float(np.percentile(h[valid], TOWER_TOP_PERCENTILE))
    if top - base_height < TOWER_MIN_RISE:
        return []
    high = valid & (np.nan_to_num(h, nan=-1.0) >= base_height + (top - base_height) / 2)
    half = extent.half_size
    transform = Affine(px, 0.0, -half + c0 * px, 0.0, -px, half - r0 * px)
    shapes = [shape(g) for g, _ in rasterio.features.shapes(high.astype(np.uint8), mask=high, transform=transform)]
    found = unary_union(shapes).intersection(poly) if shapes else Polygon()
    axis = _long_axis(poly)
    out = []
    for part in _polygon_parts(found):
        part = _tower_outline(part, axis)
        if part.is_empty or part.area < TOWER_MIN_PART:
            continue
        inside = _roof_cells(part, ndsm, ndvi, extent, px)
        vals = inside[0][np.isfinite(inside[0])] if inside is not None else np.empty(0)
        if vals.size >= LIDAR_MIN_CELLS:
            out.append((part, float(np.clip(np.percentile(vals, TOWER_HEIGHT_PERCENTILE), MIN_HEIGHT,
                                             MAX_HEIGHT))))
    return out


def building_height(poly: Polygon, tags: dict, ms_height, kind: str, ndsm, ndvi, extent: Extent,
                    px: float) -> tuple[float, str]:
    """(height in m, one of HEIGHT_SOURCES): the first of lidar, OSM height, building:levels,
    Microsoft height and the type default that applies, clamped to 2.5-150 m."""
    tag_height = _number(tags.get("height"))
    levels = _number(tags.get("building:levels"))
    lidar = _lidar_height(poly, ndsm, ndvi, extent, px)
    if lidar is not None:
        h, source = lidar, "lidar"
    elif tag_height:
        h, source = tag_height, "osm_height"
    elif levels:
        h, source = levels * LEVEL_HEIGHT + LEVELS_EXTRA, "levels"
    elif ms_height is not None and ms_height > 0:
        h, source = float(ms_height), "ms"
    else:
        h, source = DEFAULT_HEIGHT.get(kind, OTHER_HEIGHT), "default"
    return float(np.clip(h, MIN_HEIGHT, MAX_HEIGHT)), source


def roof_type(kind: str, source: str, poly: Polygon, height: float) -> str:
    """"hip" for a small residential footprint close to a rectangle, else "flat"."""
    residential = kind in HIP_KINDS or source == "ms"
    if (not residential or len(poly.interiors) or poly.area >= HIP_MAX_AREA
            or not HIP_MIN_HEIGHT <= height < HIP_MAX_HEIGHT):
        return "flat"
    rect = poly.minimum_rotated_rectangle.area
    return "hip" if rect > 0 and poly.area / rect >= HIP_MIN_COVER else "flat"


def wall_class(kind: str, height: float, area: float = 0.0) -> str:
    """One of WALLS: glass over 20 m, else residential, brick or concrete by type. An untyped
    building (building=yes, every Microsoft footprint) is a house only when low and small."""
    if height > GLASS_HEIGHT:
        return "concrete" if kind in STONE_KINDS else "glass"
    if kind in RES_KINDS or (kind == "yes" and height < SMALL_HEIGHT and area < LARGE_AREA):
        return "res"
    return "brick" if kind in BRICK_KINDS else "concrete"


def building_colour(id: str, wall: str) -> tuple[float, float, float]:
    """A palette colour for the wall class, the same for a building on every run."""
    palette = PALETTES.get(wall, PALETTES["concrete"])
    return palette[zlib.crc32(str(id).encode()) % len(palette)]


def outline_points(ring: np.ndarray, holes=(), step: float = BASE_STEP) -> np.ndarray:
    """(N, 2): the vertices of the ring and of each hole, with points added along every edge
    (the closing one too) so that none are more than step apart."""
    rings = [np.asarray(r, dtype=float).reshape(-1, 2) for r in (ring, *holes)]
    rings = [r for r in rings if len(r)]
    if not rings:
        return np.zeros((0, 2))
    starts = np.concatenate(rings)
    seg = np.concatenate([np.roll(r, -1, axis=0) - r for r in rings])
    n = np.maximum(1, np.ceil(np.hypot(seg[:, 0], seg[:, 1]) / step)).astype(int)
    t = (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)) / np.repeat(n, n)
    return np.repeat(starts, n, axis=0) + t[:, None] * np.repeat(seg, n, axis=0)


def wet_areas(water, extent: Extent, n: int) -> shapely.STRtree | None:
    """The water bodies grown by WET_PAD cells of an n x n terrain grid, prepared, in a tree
    (None without water): a point in one lies in the water, or so near it that its terrain
    sample takes in the carved bed."""
    pad = WET_PAD * extent.pixel_size(n)
    geoms = [shapely.buffer(w.polygon, pad) for w in water]
    if not geoms:
        return None
    shapely.prepare(geoms)
    return shapely.STRtree(geoms)


def _in_wet(xy: np.ndarray, wet) -> np.ndarray:
    """Which of the points lie in one of the wet_areas."""
    inside = np.zeros(len(xy), bool)
    if wet is None or not len(xy):
        return inside
    x, y = xy[:, 0], xy[:, 1]
    for k in wet.query(shapely.box(x.min(), y.min(), x.max(), y.max())):
        g = wet.geometries[k]
        x0, y0, x1, y1 = g.bounds
        near = ~inside & (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        if near.any():
            inside[near] = shapely.intersects_xy(g, x[near], y[near])
    return inside


def lowest_ground(outlines, heights: np.ndarray, extent: Extent, wet=None) -> np.ndarray:
    """Per (ring, holes) outline: the lowest terrain along it (outline_points), leaving out
    the points in the wet_areas unless all of the outline's are; NaN for an empty ring."""
    pts = [outline_points(ring, holes) for ring, holes in outlines]
    sizes = np.array([len(p) for p in pts], dtype=int)
    out = np.full(len(pts), np.nan)
    if not sizes.sum():
        return out
    xy = np.concatenate(pts)
    z = np.atleast_1d(sample_height(heights, extent, xy[:, 0], xy[:, 1])).astype(float)
    wet_pt = _in_wet(xy, wet)
    if wet_pt.any():
        owner = np.repeat(np.arange(len(pts)), sizes)
        all_wet = np.bincount(owner, weights=(~wet_pt).astype(float), minlength=len(pts)) == 0
        z = np.where(wet_pt & ~all_wet[owner], np.inf, z)
    some = sizes > 0
    out[some] = np.minimum.reduceat(z, np.r_[0, np.cumsum(sizes[some])[:-1]])
    return out


def _base_z(ring: np.ndarray, holes, heights: np.ndarray, extent: Extent, wet=None) -> float:
    return float(lowest_ground([(ring, holes)], heights, extent, wet)[0]) - BASE_DROP


def make_building(footprint: tuple, ndsm, ndvi, extent: Extent, px: float, heights: np.ndarray,
                  wet=None) -> Building:
    """A Building from a footprints() tuple, the canopy grids (None when missing), the terrain
    and the wet_areas (None: no water)."""
    id, source, kind, poly, tags, ms_height = footprint
    height, height_source = building_height(poly, tags, ms_height, kind, ndsm, ndvi, extent, px)
    poly = orient(poly, 1.0)
    ring = np.asarray(poly.exterior.coords, dtype=float)[:-1, :2]
    holes = [np.asarray(r.coords, dtype=float)[:-1, :2] for r in poly.interiors]
    wall = wall_class(kind, height, poly.area)
    return Building(id=id, source=source, kind=kind, ring=ring, holes=holes, height=height,
                    base_z=_base_z(ring, holes, heights, extent, wet), roof=roof_type(kind, source, poly, height),
                    wall=wall, colour=building_colour(id, wall), height_source=height_source)


def make_buildings(footprint: tuple, ndsm, ndvi, extent: Extent, px: float, heights: np.ndarray,
                   wet=None) -> list[Building]:
    """make_building's Building, then one more (id '<id>/t<k>') for each tower the lidar shows
    rising from it, standing on the same base."""
    base = make_building(footprint, ndsm, ndvi, extent, px, heights, wet)
    if base.height_source != "lidar":
        return [base]
    out = [base]
    for k, (part, height) in enumerate(tower_parts(footprint[3], ndsm, ndvi, extent, px, base.height)):
        part = orient(part, 1.0)
        tid = f"{base.id}/t{k}"
        wall = wall_class(base.kind, height, part.area)
        out.append(Building(id=tid, source=base.source, kind=base.kind,
                            ring=np.asarray(part.exterior.coords, dtype=float)[:-1, :2],
                            holes=[np.asarray(r.coords, dtype=float)[:-1, :2] for r in part.interiors],
                            height=height, base_z=base.base_z, roof="flat", wall=wall,
                            colour=building_colour(tid, wall), height_source="lidar"))
    return out


# ------------------------------------------------------------------ tiles

def _tile_key(ix: int, iy: int) -> str:
    return f"{'e' if ix >= 0 else 'w'}{abs(ix):02d}{'n' if iy >= 0 else 's'}{abs(iy):02d}"


def _split(key, origin, size, members, buildings, centres, counts, max_verts) -> list[tuple]:
    most = int(counts[members].sum(axis=0).max())
    if most <= max_verts or len(members) == 1 or size / 2 < MIN_TILE:
        if most > max_verts:
            log.warning("tile %s keeps %d vertices in one level of detail (cap %d)", key, most, max_verts)
        return [(key, origin, [buildings[k] for k in members])]
    out = []
    east, north = centres[members, 0] >= origin[0], centres[members, 1] >= origin[1]
    for name, sx, sy in QUARTERS:
        sub = members[(east == (sx > 0)) & (north == (sy > 0))]
        if len(sub):
            quarter = (origin[0] + sx * size / 4, origin[1] + sy * size / 4)
            out += _split(f"{key}_{name}", quarter, size / 2, sub, buildings, centres, counts, max_verts)
    return out


def tiles(buildings: list[Building], tile_m: float, max_verts: int = MAX_VERTS) -> list[tuple]:
    """(key, origin_xy, buildings) per non-empty tile; see the module docstring."""
    centres = np.array([(b.ring.min(axis=0) + b.ring.max(axis=0)) / 2 for b in buildings]).reshape(-1, 2)
    counts = dae.vertex_counts_each(buildings)
    cells: dict[tuple[int, int], list[int]] = {}
    for k, (ix, iy) in enumerate(np.floor(centres / tile_m).astype(int).tolist()):
        cells.setdefault((ix, iy), []).append(k)
    out = []
    for ix, iy in sorted(cells):
        origin = ((ix + 0.5) * tile_m, (iy + 0.5) * tile_m)
        out += _split(_tile_key(ix, iy), origin, tile_m, np.array(cells[(ix, iy)]), buildings, centres,
                      counts, max_verts)
    return out


# ------------------------------------------------------------------ the stage and the level

def _canopy(out_dir: Path, extent: Extent):
    """(ndsm, ndvi, px) from the canopy stage, or (None, None, RES) when it has not run or
    was made for another box."""
    try:
        ndsm, ndvi, px = load_canopy(out_dir)
    except FileNotFoundError:
        log.warning("no canopy grid in %s (run the canopy stage): heights come from tags and defaults",
                    out_dir)
        return None, None, RES
    if abs(ndsm.shape[0] * px - 2 * extent.half_size) > px:
        log.warning("the canopy grid (%d cells of %.2f m) does not span this box: not used",
                    ndsm.shape[0], px)
        return None, None, RES
    return ndsm, ndvi, px


def _counts(values) -> str:
    return ", ".join(f"{k} {v}" for k, v in Counter(values).most_common()) or "none"


def _drop_landmarked(blds: list[Building], path: Path) -> list[Building]:
    """blds without the ones the landmarks in landmarks.json take the place of (their clear
    zones grown to what their models in models/ beside it stand on), logging how many each
    landmark removes."""
    from . import landmarks  # landmarks builds on this module

    by_landmark = landmarks.removed_by(blds, landmarks.load_landmarks(path), path.parent / "models")
    gone = set().union(*by_landmark.values())

    def listed(ids) -> str:
        ids = sorted(ids, key=_id_key)
        return ", ".join(ids[:LOG_IDS]) + (", ..." if len(ids) > LOG_IDS else "")

    log.info("landmarks remove %d buildings: %s", len(gone),
             "; ".join(f"{name} {len(ids)} ({listed(ids)})" if ids else f"{name} 0"
                       for name, ids in by_landmark.items()) or "no landmarks")
    return [b for b in blds if b.id not in gone]


def _osm_id(id: str) -> str:
    """The OSM element a building comes from: its id without a part ('/<k>') or tower part ('/t<k>')."""
    return str(id).split("/", 1)[0]


def split_styled(blds: list[Building], groups: dict) -> tuple[list[Building], list[dict]]:
    """(the buildings left as they are, the styled ones): each building whose OSM element is in
    a group of styled.yaml ({group: {"style", "ids"}}), its parts and tower parts too, moves out
    as {"style", "group", "building": Building.to_dict()}, in building order. ValueError for an
    id listed in two groups."""
    owner: dict[str, str] = {}
    for group, g in groups.items():
        for i in (g or {}).get("ids") or []:
            if str(i) in owner:
                raise ValueError(f"building {i} is in two styled groups, {owner[str(i)]} and {group}")
            owner[str(i)] = group
    rest, moved = [], []
    for b in blds:
        group = owner.get(_osm_id(b.id))
        if group is None:
            rest.append(b)
        else:
            moved.append({"style": groups[group]["style"], "group": group, "building": b.to_dict()})
    return rest, moved


def _move_styled(blds: list[Building], path: Path) -> list[Building]:
    """blds without the styled ones (styled.yaml), which go to path (styled.json), logged."""
    from . import styled  # styled builds on this module

    groups = styled.load_registry()["groups"]
    blds, moved = split_styled(blds, groups)
    path.write_text(json.dumps(moved), encoding="utf-8")
    found = {_osm_id(m["building"]["id"]) for m in moved}
    missing = sorted({str(i) for g in groups.values() for i in g.get("ids") or []} - found, key=_id_key)
    log.info("%d styled buildings (%s) in %s; not in this box: %s", len(moved),
             _counts(m["group"] for m in moved), path.name, ", ".join(missing) or "none")
    return blds


def run(cfg, raw_dir=None, out_dir=None) -> None:
    extent = Extent.from_config(cfg)
    raw_dir, out_dir = Path(raw_dir or cfg.raw_dir), Path(out_dir or cfg.out_dir)
    features = raw_dir / "osm_features.json"
    if not features.exists():
        raise FileNotFoundError(f"no {features}; run the features stage first")
    osm = json.loads(features.read_text(encoding="utf-8"))
    ms_file = raw_dir / "ms_buildings.json"
    ms = []
    if cfg.buildings_microsoft:
        if ms_file.exists():
            ms = json.loads(ms_file.read_text(encoding="utf-8"))
        else:
            log.warning("no %s: OpenStreetMap footprints only", ms_file)
    found = footprints(osm, ms, extent)
    water = load_water(out_dir / "water.json") if (out_dir / "water.json").exists() else []
    kept, dropped = drop_conflicts(found, load_roads(out_dir / "roads.json"), water)
    ndsm, ndvi, px = _canopy(out_dir, extent)
    heights = load_heights(out_dir=out_dir)
    wet = wet_areas(water, extent, heights.shape[0])
    blds = [b for f in kept for b in make_buildings(f, ndsm, ndvi, extent, px, heights, wet)]
    if (out_dir / "landmarks.json").exists():
        blds = _drop_landmarked(blds, out_dir / "landmarks.json")
    blds = _move_styled(blds, out_dir / STYLED_FILE)
    save_buildings(blds, out_dir / "buildings.json")
    log.info("%d buildings (%s); dropped: %s", len(blds), _counts(b.source for b in blds),
             _counts(why for why, ids in dropped.items() for _ in ids))
    log.info("heights from: %s; median %.1f m, tallest %.1f m", _counts(b.height_source for b in blds),
             float(np.median([b.height for b in blds])) if blds else 0.0,
             max((b.height for b in blds), default=0.0))
    log.info("roofs: %s; walls: %s", _counts(b.roof for b in blds), _counts(b.wall for b in blds))


def _podium(id: str) -> str | None:
    """The id of the building a tower part ('<id>/t<k>') stands on, or None for a building."""
    head, sep, k = id.rpartition("/t")
    return head if sep and k.isdigit() else None


def settle(blds: list[Building], heights: np.ndarray, extent: Extent, wet=None) -> list[Building]:
    """The buildings standing on the level's terrain, which is cut under the bridge decks: a
    base goes down where that terrain is lower (the stage's rule, with the wet_areas left
    out), and the roof stays where it was. A tower part goes down with its podium."""
    lows = lowest_ground([(b.ring, b.holes) for b in blds], heights, extent, wet) - BASE_DROP
    bases = [float(low) if low < b.base_z else b.base_z for b, low in zip(blds, lows)]
    podiums = {b.id: z for b, z in zip(blds, bases) if _podium(b.id) is None}
    out = []
    for b, z in zip(blds, bases):
        podium = _podium(b.id)
        if podium in podiums:  # a tower part: its podium's settled base
            z = podiums[podium]
        out.append(b if not z < b.base_z else dataclasses.replace(b, base_z=z, height=b.height + (b.base_z - z)))
    return out


def ts_static(level_name: str, key: str, origin) -> dict:
    """The TSStatic placing one tile's shape at the tile origin, unrotated."""
    return {"class": "TSStatic", "persistentId": bf.persistent_id("bld", key),
            "shapeName": f"/levels/{level_name}/{SHAPES_DIR}/br_bld_{key}.dae",
            "position": [round(float(origin[0]), 2), round(float(origin[1]), 2), 0.0],
            "useInstanceRenderData": True, "collisionType": "Collision Mesh", "decalType": "None"}


def level_content(cfg, data_dir, root, heights, extent) -> dict[str, list[dict]]:
    data_dir, root = Path(data_dir), Path(root)
    water = load_water(data_dir / "water.json") if (data_dir / "water.json").exists() else []
    blds = settle(load_buildings(data_dir / "buildings.json"), heights, extent,
                  wet_areas(water, extent, heights.shape[0]))
    folder = root / SHAPES_DIR
    folder.mkdir(parents=True, exist_ok=True)
    objs, most = [], dict.fromkeys(dae.LODS, 0)
    size = 0
    for key, origin, members in tiles(blds, cfg.building_tile_m):
        xml, counts = dae.tile_dae(members, origin)
        path = folder / f"br_bld_{key}.dae"
        path.write_text(xml, encoding="utf-8")
        size += path.stat().st_size
        most = {lod: max(most[lod], counts[lod]) for lod in most}
        objs.append(ts_static(cfg.level_name, key, origin))
    (root / MATERIALS_FILE).write_text(json.dumps(MATERIALS, indent=1), encoding="utf-8")
    log.info("%d buildings in %d tiles (%.1f MB of .dae); most vertices in a tile: %s", len(blds),
             len(objs), size / 1e6, ", ".join(f"{k} {v}" for k, v in most.items()))
    return {"buildings": objs}
