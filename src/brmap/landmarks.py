"""Landmarks: models built in Blender (the models stage) placed on their real footprints, in place
of the generic buildings and trees there.

The registry, src/brmap/data/landmarks.yaml (load_registry), holds one entry per landmark:
title, model, anchor ({osm: id} or {lat, lon}), yaw ({from: id} or {azimuth_deg}), footprint
({osm: [ids]} or {ring: [[lat, lon], ...]}), replaces, clear_pad_m, params and sources (a cited
source for each param in CITED). OSM ids name raw/osm_features.json "buildings" or "areas".

out/landmarks.json (run) is a list of {"name", "model", "x", "y", "yaw", "base_z", "ground_z",
"footprint", "clear", "replaces", "params", "lane_cut"} in level coordinates, yaw in radians
counter-clockwise from +x. params are the registry's plus, in model coordinates (origin at the
anchor, +u along the yaw, +v left of it, z up from base_z), footprint_local and ground_dz.
lane_cut is not a param, so it is no part of a model's key (models.model_key): it records, for
validate, what the cut of the footprint by the lanes beside it (below) took:
{"m2": the area it removed, "share": that as a fraction of the footprint before the cut, "pieces":
how many pieces it left (the largest is kept), "holes": how many holes (a corridor wholly inside
the footprint; the kept outline fills them again, so they remove nothing), "roads": [{"road":
road_label, "m2": what that road took of the m2}, ...], "through": [road_label, ...]}. Each road is
charged what is left of its bite after the roads before it, so an overlap of corridors counts once
and the roads' m2 add up to the m2 less the pieces not kept. through names, in road order, the
roads of the cut whose centreline crosses the footprint before the cut shrunk THROUGH_SHRINK
(validate's "passes through" test): a road into the landmark, not beside it, whose own notch would
hide it from validate. A footprint nothing was cut from (no road beside it, a landmark over decks
or exempt from every road) has {"m2": 0.0, "share": 0.0, "pieces": 1, "holes": 0, "roads": [],
"through": []}; the roads named are never the exempt ones.

* resolve(registry, osm, extent, heights, ndsm=None, px=None, roads=()) -> the entries:
  - an entry whose anchor lies outside the box, or that names an OSM id osm_features.json
    lacks, is skipped with a log line;
  - x, y: the anchor, the centroid of an OSM polygon or a lat/lon;
  - yaw: the long axis of the "from" polygon's minimum rotated rectangle, or
    radians(90 - azimuth_deg); both taken modulo pi;
  - footprint: the union of the OSM polygons (their outer outline), or the ring, counter-clockwise,
    kept off the lanes beside it: less the corridors of the drivable ground roads (ground_roads,
    the roads validate checks), each road's centreline buffered by width / 2 + LANE_CLEAR, keeping
    the largest piece left (counter-clockwise); what each road took (m2, less the holes the kept
    outline fills again) is logged and recorded as lane_cut, which validate checks: a cut over 2 %
    of the footprint, into pieces or leaving a hole is a problem, and so is every road of its
    through (a centreline across the footprint before the cut, however small its notch). Everything
    below follows the cut footprint (a road wholly inside the outline stays there). Not for a
    landmark over bridge decks (params.decks: it is in the air), nor for the roads
    params.keep_off_roads exempts (exempt_ways): false, every road (for a landmark the cut would
    cost over 2 % of its footprint); {except: [w<id>, ...]}, only the roads of those OSM ways (the
    Capitol: two service drives end at its walls; the cut would cost it 0.6 %, but the round ends
    of their corridors would bite notches into the walls that break its model's triangle budget).
    What an exempt road would have cut is logged, and a named way whose corridor does not reach
    the footprint is a warning (a typo, or a way renumbered since the last fetch);
  - base_z: the lowest terrain along the footprint's outline, sampled every 2 m
    (buildings.lowest_ground), minus 0.3 m; ground_z: the median terrain inside the footprint;
  - clear: the footprint buffered by clear_pad_m.
  Per-landmark extras, added to params in model coordinates:
  - params.rim: lidar (a bowl round a field hole: the footprint's OSM polygon has a hole, else
    ValueError) adds rim_profile, rim_profile()'s sectors turned into model angles: [[angle_deg,
    h], ...], angle counter-clockwise from +u round the footprint's centroid, sorted, h in m above
    the ground; and hole_local, the field hole's ring (counter-clockwise). It needs ndsm and px
    (canopy.load_canopy): without them, or with no lidar between the rings, the entry is skipped
    with a warning.
  - params.pitch: <osm id> (the leisure=pitch area) adds, from that polygon's minimum rotated
    rectangle, pitch_local (its centre, [u, v]) and pitch_yaw (its long axis in model
    coordinates, radians in [-pi/2, pi/2)); and field_dz, the highest terrain on the field (the
    field hole when there is one, else the pitch, widened by one terrain cell) above base_z, so
    a field laid at it is never under the ground.
  - params.field: <osm id> (a ball field drawn as polygons with holes: the playing field round
    the infield, the infield grass round the mound) adds field_local, each of its polygons as
    [outer ring (counter-clockwise), hole (clockwise), ...], the largest first; and field_dz as
    for a pitch, over the polygons' outer outlines.
  - params.outlines: [<osm id>, ...] (other polygons the model builds on, e.g. the bleachers
    it replaces) adds outlines_local, each one's outline (its largest polygon's, counter-
    clockwise), in the same order.
  - params.decks: [w<osm id>, ...] (the bridge decks a model spans, the Wilkinson truss) with
    from_lon and to_lon (its ends) and peaks_lon adds, from roads (the Road records of
    out/roads.json, whose bridge deck holding that way is the deck), deck_local: per deck in the
    same order {"id", "width", "nodes": [[u, v, z], ...]}, the deck's nodes whose longitude lies
    between from_lon and to_lon, in u order, z up from base_z; and from_u, to_u and peaks_u, where
    the first deck crosses those longitudes, as u. A deck missing from roads, with fewer than two
    nodes between the ends or not reaching one of the longitudes skips the entry with a warning.
* road_label(road) -> 'way <osm ids> (<name>)', how the log lines and validate's problems name a road.
* ground_roads(roads) -> the drivable roads on the ground: not a bridge deck, drivability > 0 (so not
  a covered passage), at least two nodes, all finite.
* exempt_ways(name, params) -> the OSM ways ('w<id>') whose lanes landmark name need not keep off, by
  params.keep_off_roads: none for true (the default), those named for {except: [...]}, None (every
  road) for false; ValueError for any other value. exempt(road, ways) says whether a Road is one of
  them. resolve and validate both read them.
* rim_profile(outer, inner, ndsm, extent, px, sector_deg=15) -> [[azimuth_deg, h], ...]: for each
  sector_deg sector centred on azimuth 0, sector_deg, ... (compass bearings from outer's
  centroid), the RIM_PERCENTILE-th percentile of the finite ndsm cells whose centres lie inside
  outer and outside inner, averaged with the sectors either side. A sector with no such cell
  (a lidar gap) first takes the value interpolated from its neighbours; ValueError when there
  is none at all.
* ground_outline(lm, models_dir) -> what the landmark's model stands on, in level coordinates:
  the xy projections of the Colmesh-1 triangles of each part (models_dir/<name>/<part>.npz)
  that have a vertex lower than params.ground_dz + GROUND_RISE (the triangles touching the
  ground; a triangle counts whole, its upper corners too), turned by the yaw and moved to x, y
  (a MultiPolygon when they stand apart). It follows the collision meshes, concave bays
  included, not their convex hulls. The models' collision meshes are mostly floorless shells,
  walls with no triangle that has an area at the ground (a vertical triangle, under TINY,
  projects to a line): the areas those walls enclose count too, so a shell is filled while a
  fence or an open chain of walls adds nothing. The footprint when there is no npz, or nothing
  touches the ground. A model can reach past its footprint (the PMAC's podium overhangs the OSM
  footprint by about 11 m to the south-west).
* clear_zone(lm, models_dir=None) -> the clear polygon, grown to cover ground_outline too when
  models_dir is given (the models stage has run by the time buildings and trees do).
* removed(buildings, landmarks, models_dir=None) -> ids: the buildings of the OSM elements in a
  landmark's replaces (with their parts '<id>/<k>', as buildings._osm_id reads them), the
  buildings with REMOVE_SHARE or more of their area inside one landmark's clear zone, and the
  tower parts ('<id>/t<k>') of every building removed. removed_by gives them per landmark, with a
  warning for a replaced id that matches no building at all and one for a replaced building that
  does not touch that landmark's clear zone (a typo in the registry, or an id that names no
  building, or another one, after a new fetch). buildings.run drops them (models_dir out/models).
* clear_polygons(path, models_dir=None) -> the clear zones of landmarks.json ([] without the
  file); trees.run keeps the trees out of them (models_dir out/models).
* run(cfg) writes landmarks.json from the registry, raw/osm_features.json and out/heights.npy;
  an empty list when cfg.landmarks_enabled is off (so the models stage builds none).
* level_content(cfg, data_dir, root, heights, extent) (the level.FEATURE_MODULES hook) writes,
  for each landmark and each out/models/<name>/<part>.npz, SHAPES_DIR/<name>.dae (part "main")
  or SHAPES_DIR/<name>_<part>.dae from collada.read_npz and shape_dae, placed by one TSStatic at
  [x, y, base_z] turned by trees.rotation_matrix(yaw) with collisionType and decalType
  "Collision Mesh", useInstanceRenderData and persistentId bf.persistent_id("lm", name, part),
  in group GROUP. SHAPES_DIR/main.materials.json holds only the materials the shapes bind,
  from src/brmap/data/landmark_materials.yaml (materials.resolve), less those an earlier feature
  module's materials file already defines (materials.write_materials: the styled buildings' tile
  roof), and their tex: textures are copied from out/textures/ to TEX_DIR. A landmark not built
  yet is left out with a warning.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import shapely
import yaml
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
from shapely.strtree import STRtree

from . import beamng_format as bf
from . import collada, materials
from .buildings import BASE_DROP, _long_axis, _osm_id, _podium, _to_level_rings, lowest_ground
from .canopy import load_canopy
from .geo import Extent
from .roads import load_roads
from .terrain import load_heights, sample_height
from .trees import rotation_matrix

log = logging.getLogger(__name__)

DATA = Path(__file__).resolve().parent / "data"
REGISTRY = DATA / "landmarks.yaml"
MATERIAL_REGISTRY = DATA / "landmark_materials.yaml"
LANDMARKS_FILE = "landmarks.json"
CITED = {"height_m", "diameter_m", "crown_m", "rim_m"}   # params that need a cited source
REQUIRED = ("model", "anchor", "yaw", "footprint")        # registry keys every entry has
REMOVE_SHARE = 0.5     # a building with this share of its area in a clear zone goes
LANE_CLEAR = 0.5       # m a footprint keeps off the corridor (width / 2) of a drivable road beside it
THROUGH_SHRINK = 0.5   # m: a road passes through a footprint when its centreline crosses it shrunk this much
SHAPES_DIR = "art/shapes/landmarks"
MATERIALS_FILE = f"{SHAPES_DIR}/main.materials.json"
TEX_DIR = materials.TEX_DIR
GROUP = "landmarks"
MAIN_PART = "main"     # models/<name>/main.npz is the shape <name>.dae; other parts add _<part>
CLEAR_SEGMENTS = 4     # segments per quarter circle round the clear zone's corners
COLLISION = "Colmesh-1"  # a model part's collision node (the npz contract)
GROUND_RISE = 3.0      # m above the ground (ground_dz) a collision vertex may be and still stand on it
TINY = 1e-6            # an xy area (m2) or edge length (m) under this is degenerate: a vertical collision triangle
RIM_SECTOR_DEG = 15.0  # rim_profile's sectors
RIM_PERCENTILE = 90.0  # of the seats' lidar heights in a sector: the top rows, not a light mast


# ------------------------------------------------------------------ the registry and the file

def load_registry(path=None) -> dict[str, dict]:
    """The landmark registry (REGISTRY by default) as {name: entry}."""
    path = Path(path or REGISTRY)
    registry = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for name, entry in registry.items():
        missing = [k for k in REQUIRED if k not in (entry or {})]
        if missing:
            raise ValueError(f"landmark {name} in {path.name} lacks {', '.join(missing)}")
    return registry


def save_landmarks(entries: list[dict], path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=1), encoding="utf-8")
    return path


def load_landmarks(path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ resolve

class _Skip(Exception):
    """An entry that can't be placed in this box; the message says why, logged at `level`."""

    def __init__(self, why: str, level: int = logging.WARNING):
        super().__init__(why)
        self.level = level


def _osm_index(osm: dict) -> dict[str, dict]:
    return {str(e.get("id")): e for key in ("buildings", "areas") for e in osm.get(key) or []}


def _osm_polygon(eid: str, index: dict, extent: Extent):
    """The union of OSM entry eid's polygons (with their holes) in level coordinates."""
    if eid not in index:
        raise _Skip(f"{eid} is not in osm_features.json (outside the features box?)")
    rings, sizes = [], []
    for p in index[eid].get("polygons") or []:
        inner = list(p.get("inner") or [])
        rings += [p.get("outer") or []] + inner
        sizes.append(len(inner))
    level, parts, i = _to_level_rings(extent, rings), [], 0
    for k in sizes:
        outer, holes = level[i], [h for h in level[i + 1:i + 1 + k] if len(h) >= 3]
        i += 1 + k
        if len(outer) >= 3:
            poly = Polygon(outer, holes)
            parts.append(poly if poly.is_valid else shapely.make_valid(poly))
    geom = unary_union(parts) if parts else Polygon()
    if geom.is_empty or geom.area <= 0:
        raise _Skip(f"{eid} has no polygon with an area")
    return geom


def _anchor(spec: dict, index: dict, extent: Extent) -> tuple[float, float]:
    if "osm" in spec:
        c = _osm_polygon(str(spec["osm"]), index, extent).centroid
        return float(c.x), float(c.y)
    x, y = extent.to_level(float(spec["lon"]), float(spec["lat"]))
    return float(x), float(y)


def _yaw(spec: dict, index: dict, extent: Extent) -> float:
    if "from" in spec:
        return _long_axis(_osm_polygon(str(spec["from"]), index, extent))
    return math.radians(90.0 - float(spec["azimuth_deg"])) % math.pi


def _footprint(name: str, spec: dict, index: dict, extent: Extent) -> Polygon:
    """The footprint's outline (no holes), counter-clockwise."""
    if "osm" in spec:
        geom = unary_union([_osm_polygon(str(eid), index, extent) for eid in spec["osm"]])
    else:
        lat, lon = np.asarray(spec["ring"], float).reshape(-1, 2).T
        x, y = extent.to_level(lon, lat)
        geom = Polygon(np.column_stack([x, y]))
        geom = geom if geom.is_valid else shapely.make_valid(geom)
    parts = [g for g in getattr(geom, "geoms", [geom]) if g.geom_type == "Polygon" and g.area > 0]
    if len(parts) != 1:
        raise ValueError(f"landmark {name}: the footprint is {len(parts)} separate pieces; "
                         "give one outline (footprint: {ring: ...})")
    return orient(Polygon(parts[0].exterior), 1.0)


def ground_roads(roads) -> list:
    """The drivable roads on the ground (see the module docstring): those validate checks the
    landmarks against."""
    return [r for r in roads if not r.is_bridge and len(r.nodes) >= 2 and np.isfinite(r.nodes).all()
            and bf._ai_fields(r)["drivability"] > 0]


def _lanes(roads) -> tuple[list, STRtree] | None:
    """(ground_roads, an STRtree of their centrelines), or None when there are none."""
    ground = ground_roads(roads)
    return (ground, STRtree([LineString(r.nodes[:, :2]) for r in ground])) if ground else None


def road_label(road) -> str:
    """'way <osm ids>' with the road's name in brackets, as the log lines and validate's problems name it."""
    return f"way {','.join(map(str, road.osm_ids))}" + (f" ({road.name})" if road.name else "")


def _no_cut() -> dict:
    """The lane_cut of a footprint nothing was cut from (see the module docstring)."""
    return {"m2": 0.0, "share": 0.0, "pieces": 1, "holes": 0, "roads": [], "through": []}


def exempt_ways(name: str, params: dict) -> frozenset[str] | None:
    """The OSM ways whose lanes landmark name need not keep off (see the module docstring); None for
    every road."""
    rule = params.get("keep_off_roads", True)
    if isinstance(rule, bool):
        return frozenset() if rule else None
    if isinstance(rule, dict) and set(rule) == {"except"} and isinstance(rule["except"], list):
        return frozenset(str(w) for w in rule["except"])
    raise ValueError(f"landmark {name}: keep_off_roads is true, false or {{except: [w<id>, ...]}}, not {rule!r}")


def exempt(road, ways: frozenset[str] | None) -> bool:
    """Whether road is one of ways (exempt_ways; None: every road)."""
    return ways is None or not ways.isdisjoint(f"w{i}" for i in road.osm_ids)


def _keep_off(name: str, foot: Polygon, lanes, ways: frozenset[str] | None) -> tuple[Polygon, dict]:
    """(foot less the corridors of the roads beside it, each grown LANE_CLEAR, but those of the exempt
    ways (exempt_ways): the largest piece left, counter-clockwise; and the lane_cut record of what that
    cut took and of the roads whose centreline crosses foot shrunk THROUGH_SHRINK, see the module
    docstring). lanes is _lanes(roads); what each road cuts off, or would have, is logged."""
    roads, tree = lanes
    reach = max(r.width for r in roads) / 2 + LANE_CLEAR
    core = foot.buffer(-THROUGH_SHRINK)
    core = foot if core.is_empty else core      # a sliver thinner than that is tested whole, as validate does
    cuts, spared, through = [], [], []
    for j in sorted(int(j) for j in tree.query(foot, predicate="dwithin", distance=reach)):
        line = tree.geometries[j]
        corridor = line.buffer(roads[j].width / 2 + LANE_CLEAR)
        area = foot.intersection(corridor).area
        if area > 0 and exempt(roads[j], ways):
            spared.append((roads[j], corridor, area))
        elif area > 0:
            cuts.append((roads[j], corridor, area))
            if line.intersects(core):           # a road into the footprint, not beside it: validate reports it
                through.append(road_label(roads[j]))
    if spared:
        log.info("landmark %s leaves its footprint whole by %s (keep_off_roads)", name,
                 ", ".join(f"{road_label(r)} {a:.1f} m2" for r, _, a in spared))
    reached = {f"w{i}" for r, _, _ in spared for i in r.osm_ids}
    for way in sorted((ways or frozenset()) - reached):
        log.warning("landmark %s: keep_off_roads names %s, whose lanes do not reach its footprint: a typo in %s, "
                    "or an id that names another road since the last fetch?", name, way, REGISTRY.name)
    if not cuts:
        return foot, _no_cut()
    left = foot.difference(unary_union([c for _, c, _ in cuts]))
    pieces = [g for g in getattr(left, "geoms", [left]) if g.geom_type == "Polygon" and g.area > TINY]
    if not pieces:
        raise ValueError(f"landmark {name}: the roads' corridors cover its whole footprint")
    kept = orient(Polygon(max(pieces, key=lambda g: g.area).exterior), 1.0)
    holes = sum(1 for g in pieces for ring in g.interiors if Polygon(ring).area > TINY)
    # what the cut really took: the corridors' bites, less the holes the kept outline fills again, plus the pieces
    # not kept. Each road is charged what is left of it after the roads before it, so an overlap counts once
    taken, bites = foot.difference(kept), []
    for road, corridor, _ in cuts:
        bites.append((road, taken.intersection(corridor).area))
        taken = taken.difference(corridor)
    lost = foot.area - kept.area
    notes = ([f"split into {len(pieces)} pieces, the largest kept"] if len(pieces) > 1 else []) + (
        [f"{holes} hole{'s' if holes > 1 else ''} filled again"] if holes else [])
    log.info("landmark %s keeps %g m off the lanes beside it: %s; %.1f of %.1f m2 (%.2f %%) cut%s", name, LANE_CLEAR,
             ", ".join(f"{road_label(r)} {a:.1f} m2" for r, a in bites), lost, foot.area, 100 * lost / foot.area,
             "".join(f", {note}" for note in notes))
    return kept, {"m2": round(lost, 3) + 0.0, "share": round(lost / foot.area, 5) + 0.0, "pieces": len(pieces),
                  "holes": holes, "roads": [{"road": road_label(r), "m2": round(a, 3) + 0.0} for r, a in bites],
                  "through": through}


def _terrain_inside(poly: Polygon, heights: np.ndarray, extent: Extent) -> np.ndarray:
    """The terrain cells whose centres lie inside poly (the terrain at a point inside it when
    none do)."""
    xs, ys = extent.pixel_centers(heights.shape[0])
    x0, y0, x1, y1 = poly.bounds
    cols, rows = np.flatnonzero((xs >= x0) & (xs <= x1)), np.flatnonzero((ys >= y0) & (ys <= y1))
    if len(cols) and len(rows):
        X, Y = np.meshgrid(xs[cols], ys[rows])
        inside = shapely.contains_xy(poly, X, Y)
        if inside.any():
            return heights[np.ix_(rows, cols)][inside]
    p = poly.representative_point()
    return np.array([float(sample_height(heights, extent, p.x, p.y))])


def _median_inside(poly: Polygon, heights: np.ndarray, extent: Extent) -> float:
    return float(np.median(_terrain_inside(poly, heights, extent)))


def to_local(xy, x0: float, y0: float, yaw: float) -> np.ndarray:
    """Level x, y -> model u, v: origin at (x0, y0), +u along yaw, +v left of it."""
    d = np.asarray(xy, float).reshape(-1, 2) - (x0, y0)
    c, s = math.cos(yaw), math.sin(yaw)
    return np.column_stack([d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c])


def _points(a) -> list[list[float]]:
    return [[round(float(x), 3) + 0.0, round(float(y), 3) + 0.0] for x, y in np.asarray(a).reshape(-1, 2)]


# ------------------------------------------------------------------ per-landmark extras

def rim_profile(outer: Polygon, inner: Polygon, ndsm, extent: Extent, px: float,
                sector_deg: float = RIM_SECTOR_DEG) -> list[list[float]]:
    """[[azimuth_deg, h], ...] round a bowl, one per sector; see the module docstring."""
    count = max(3, round(360.0 / sector_deg))
    step = 360.0 / count
    n, half = ndsm.shape[0], extent.half_size
    minx, miny, maxx, maxy = outer.bounds
    c0, c1 = max(0, math.floor((minx + half) / px)), min(n, math.ceil((maxx + half) / px))
    r0, r1 = max(0, math.floor((half - maxy) / px)), min(n, math.ceil((half - miny) / px))
    raw = np.full(count, np.nan)
    if c1 > c0 and r1 > r0:
        X, Y = np.meshgrid(-half + (np.arange(c0, c1) + 0.5) * px, half - (np.arange(r0, r1) + 0.5) * px)
        h = ndsm[r0:r1, c0:c1]
        seats = shapely.contains_xy(outer, X, Y) & ~shapely.contains_xy(inner, X, Y) & np.isfinite(h)
        centre = outer.centroid
        azimuth = np.degrees(np.arctan2(X[seats] - centre.x, Y[seats] - centre.y)) % 360.0
        sector = np.floor(((azimuth + step / 2) % 360.0) / step).astype(int) % count
        values = h[seats]
        for k in range(count):
            mine = values[sector == k]
            if mine.size:
                raw[k] = np.percentile(mine, RIM_PERCENTILE)
    found = np.isfinite(raw)
    if not found.any():
        raise ValueError("no lidar cells between the rings")
    centres = np.arange(count) * step
    raw = np.interp(centres, centres[found], raw[found], period=360.0)
    smooth = (np.roll(raw, 1) + raw + np.roll(raw, -1)) / 3.0
    return [[round(float(a), 3), round(float(v), 2)] for a, v in zip(centres, smooth)]


def _hole(spec: dict, index: dict, extent: Extent) -> Polygon | None:
    """The largest hole of the footprint's OSM polygons (counter-clockwise), or None."""
    if "osm" not in spec:
        return None
    geom = unary_union([_osm_polygon(str(eid), index, extent) for eid in spec["osm"]])
    holes = [Polygon(r) for g in getattr(geom, "geoms", [geom]) if g.geom_type == "Polygon" for r in g.interiors]
    holes = [h for h in holes if h.area > 0]
    return orient(max(holes, key=lambda h: h.area), 1.0) if holes else None


def _deck_u(road, lon: float, extent: Extent, at) -> float:
    """u where the deck's centre line first crosses the longitude lon."""
    x, y, yaw, _ = at
    lons = np.asarray(extent.to_lonlat(road.nodes[:, 0], road.nodes[:, 1])[0], float) - lon
    for i in np.flatnonzero(lons[:-1] * lons[1:] <= 0):
        if lons[i] != lons[i + 1] or lons[i] == 0:
            f = 0.0 if lons[i] == lons[i + 1] else lons[i] / (lons[i] - lons[i + 1])
            p = road.nodes[i, :2] + f * (road.nodes[i + 1, :2] - road.nodes[i, :2])
            return round(float(to_local([p], x, y, yaw)[0, 0]), 3) + 0.0
    raise _Skip(f"its deck way {'+'.join(map(str, road.osm_ids))} does not reach longitude {lon}")


def _decks(params: dict, roads, extent: Extent, at) -> dict:
    """deck_local, from_u, to_u and peaks_u for params.decks (see the module docstring)."""
    x, y, yaw, base_z = at
    missing = [k for k in ("from_lon", "to_lon") if k not in params]
    if missing:
        raise ValueError(f"params.decks needs {', '.join(missing)}")
    by_way = {f"w{int(i)}": r for r in roads if r.is_bridge and len(r.nodes) >= 2 for i in r.osm_ids}
    lo, hi = sorted((float(params["from_lon"]), float(params["to_lon"])))
    decks = []
    for way in map(str, params["decks"]):
        road = by_way.get(way)
        if road is None:
            raise _Skip(f"its deck {way} (params.decks) is not a bridge deck of roads.json")
        lon = np.asarray(extent.to_lonlat(road.nodes[:, 0], road.nodes[:, 1])[0], float)
        keep = (lon >= lo) & (lon <= hi)
        if keep.sum() < 2:
            raise _Skip(f"its deck {way} has {int(keep.sum())} nodes between from_lon and to_lon")
        local = to_local(road.nodes[keep, :2], x, y, yaw)
        z = road.nodes[keep, 2] - base_z
        order = np.argsort(local[:, 0], kind="stable")
        decks.append({"id": way, "width": float(road.width),
                      "nodes": [[round(float(u), 3) + 0.0, round(float(v), 3) + 0.0, round(float(h), 3) + 0.0]
                                for (u, v), h in zip(local[order], z[order])]})
    first = by_way[str(params["decks"][0])]
    return {"deck_local": decks, "from_u": _deck_u(first, float(params["from_lon"]), extent, at),
            "to_u": _deck_u(first, float(params["to_lon"]), extent, at),
            "peaks_u": [_deck_u(first, float(lon), extent, at) for lon in params.get("peaks_lon") or []]}


def _extras(name: str, params: dict, footprint: dict, foot: Polygon, index: dict, extent: Extent,
            heights: np.ndarray, at: tuple[float, float, float, float], ndsm, px, roads=()) -> dict:
    """The per-landmark extras for params (see the module docstring); at is (x, y, yaw, base_z)."""
    x, y, yaw, base_z = at
    rim, out = params.get("rim") == "lidar", {}
    hole = _hole(footprint, index, extent) if rim or "pitch" in params else None
    if rim:
        if hole is None:
            raise ValueError(f"landmark {name}: params.rim lidar needs a footprint with a field hole")
        if ndsm is None or px is None:
            raise _Skip("its rim (params.rim: lidar) needs the canopy stage's ndsm (run canopy first)")
        try:
            profile = rim_profile(foot, hole, ndsm, extent, px)
        except ValueError as why:
            raise _Skip(f"its rim (params.rim: lidar): {why}") from None
        turn = math.degrees(yaw)
        out["rim_profile"] = sorted([round((90.0 - a - turn) % 360.0, 3) % 360.0, h] for a, h in profile)
        out["hole_local"] = _points(to_local(np.asarray(hole.exterior.coords)[:-1, :2], x, y, yaw))
    if "pitch" in params:
        pitch = _osm_polygon(str(params["pitch"]), index, extent)
        c = pitch.minimum_rotated_rectangle.centroid
        out["pitch_local"] = _points(to_local([(c.x, c.y)], x, y, yaw))[0]
        out["pitch_yaw"] = round((_long_axis(pitch) - yaw + math.pi / 2) % math.pi - math.pi / 2, 6) + 0.0
        out["field_dz"] = _highest(hole if hole is not None else pitch, heights, extent, base_z)
    if "field" in params:
        field = _osm_polygon(str(params["field"]), index, extent)
        polys = sorted((orient(g, 1.0) for g in getattr(field, "geoms", [field]) if g.geom_type == "Polygon"),
                       key=lambda g: -g.area)
        out["field_local"] = [[_points(to_local(np.asarray(r.coords)[:-1, :2], x, y, yaw))
                               for r in (g.exterior, *g.interiors)] for g in polys]
        out["field_dz"] = _highest(unary_union([Polygon(g.exterior) for g in polys]), heights, extent, base_z)
    if "outlines" in params:
        rings = []
        for eid in params["outlines"]:
            geom = _osm_polygon(str(eid), index, extent)
            largest = max(getattr(geom, "geoms", [geom]), key=lambda g: g.area)
            ring = orient(Polygon(largest.exterior), 1.0).exterior.coords
            rings.append(_points(to_local(np.asarray(ring)[:-1, :2], x, y, yaw)))
        out["outlines_local"] = rings
    if params.get("decks"):
        out |= _decks(params, roads, extent, at)
    return out


def _highest(poly: Polygon, heights: np.ndarray, extent: Extent, base_z: float) -> float:
    """The highest terrain on poly widened by one terrain cell, above base_z."""
    cells = _terrain_inside(poly.buffer(extent.pixel_size(heights.shape[0])), heights, extent)
    return round(float(np.max(cells)) - base_z, 3)


def _resolve_one(name: str, e: dict, index: dict, extent: Extent, heights: np.ndarray, ndsm=None,
                 px=None, roads=(), lanes=None) -> dict:
    """One registry entry's landmarks.json entry; lanes is _lanes(roads)."""
    x, y = _anchor(e["anchor"], index, extent)
    if not extent.contains(x, y):
        raise _Skip(f"its anchor ({x:.0f}, {y:.0f}) is outside the box", logging.INFO)
    yaw = _yaw(e["yaw"], index, extent)
    foot = _footprint(name, e["footprint"], index, extent)
    params = dict(e.get("params") or {})
    ways, lane_cut = exempt_ways(name, params), _no_cut()
    if lanes is not None and not params.get("decks"):   # decks: in the air
        foot, lane_cut = _keep_off(name, foot, lanes, ways)
    ring = np.asarray(foot.exterior.coords)[:-1, :2]
    base_z = round(float(lowest_ground([(ring, [])], heights, extent)[0]) - BASE_DROP, 3)
    ground_z = round(_median_inside(foot, heights, extent), 3)
    clear = foot.buffer(float(e.get("clear_pad_m") or 0.0), quad_segs=CLEAR_SEGMENTS)
    extras = _extras(name, params, e["footprint"], foot, index, extent, heights, (x, y, yaw, base_z), ndsm, px, roads)
    params |= {"footprint_local": _points(to_local(ring, x, y, yaw)), "ground_dz": round(ground_z - base_z, 3),
               **extras}
    return {"name": name, "model": e["model"], "x": round(x, 3), "y": round(y, 3), "yaw": yaw,
            "base_z": base_z, "ground_z": ground_z, "footprint": _points(ring),
            "clear": _points(np.asarray(orient(clear, 1.0).exterior.coords)[:-1, :2]),
            "replaces": [str(i) for i in e.get("replaces") or []], "params": params, "lane_cut": lane_cut}


def resolve(registry: dict, osm: dict, extent: Extent, heights: np.ndarray, ndsm=None, px=None,
            roads=()) -> list[dict]:
    """The landmarks.json entries of the registry's landmarks inside the box; see the module
    docstring. ndsm and px are for the rim profiles, roads (Road records) for the decks and the
    lanes the footprints keep off."""
    index, lanes = _osm_index(osm), _lanes(roads)
    out = []
    for name, e in registry.items():
        try:
            out.append(_resolve_one(name, e, index, extent, heights, ndsm, px, roads, lanes))
        except _Skip as why:
            log.log(why.level, "landmark %s skipped: %s", name, why)
    return out


# ------------------------------------------------------------------ what a landmark clears

def _polygon(b) -> Polygon:
    poly = Polygon(b.ring, b.holes) if len(b.ring) >= 3 else Polygon()
    return poly if poly.is_valid else shapely.make_valid(poly)


def _ground_triangles(lm: dict, models_dir) -> np.ndarray:
    """(T, 3, 2): the xy corners, in level coordinates, of the Colmesh-1 triangles of every part
    of the landmark that have a vertex lower than ground_dz + GROUND_RISE."""
    params = lm.get("params") or {}
    top = float(params.get("ground_dz", lm["ground_z"] - lm["base_z"])) + GROUND_RISE
    c, s = math.cos(lm["yaw"]), math.sin(lm["yaw"])
    out = []
    for part in sorted((Path(models_dir) / lm["name"]).glob("*.npz")):
        with np.load(part, allow_pickle=False) as data:
            if f"{COLLISION}/pos" not in data.files:
                continue
            pos = data[f"{COLLISION}/pos"]
        tris = pos.reshape(-1, 3, 3)
        tris = tris[(tris[:, :, 2] < top).any(axis=1)]
        u, v = tris[:, :, 0], tris[:, :, 1]
        out.append(np.stack([lm["x"] + u * c - v * s, lm["y"] + u * s + v * c], axis=-1))
    return np.concatenate(out) if out else np.empty((0, 3, 2))


def ground_outline(lm: dict, models_dir) -> Polygon | MultiPolygon:
    """What the landmark's model stands on, in level coordinates; see the module docstring."""
    foot = Polygon(lm["footprint"]) if len(lm.get("footprint") or []) >= 3 else Polygon()
    xy = _ground_triangles(lm, models_dir)
    a, b, d = xy[:, 0], xy[:, 1], xy[:, 2]
    area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (d[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (d[:, 0] - a[:, 0]))
    solid = area >= TINY
    walls = xy[~solid]                    # vertical triangles: in plan a line
    edges = np.concatenate([walls[:, [0, 1]], walls[:, [1, 2]], walls[:, [2, 0]]])
    edges = edges[np.linalg.norm(edges[:, 1] - edges[:, 0], axis=1) >= TINY]
    enclosed = shapely.polygonize([unary_union(shapely.linestrings(edges))]).geoms if len(edges) else []
    ground = unary_union([*shapely.polygons(xy[solid]), *enclosed]).buffer(0)
    return foot if ground.is_empty else ground


def clear_zone(lm: dict, models_dir=None) -> Polygon | MultiPolygon:
    """The landmark's clear polygon, with its ground_outline when models_dir is given."""
    clear = Polygon(lm["clear"]) if len(lm.get("clear") or []) >= 3 else Polygon()
    return clear if models_dir is None else clear.union(ground_outline(lm, models_dir))


def removed_by(buildings, landmarks, models_dir=None) -> dict[str, set[str]]:
    """{landmark name: the ids of the buildings it removes}; see removed()."""
    ids = [b.id for b in buildings]
    elements = [_osm_id(i) for i in ids]
    present = set(elements)
    boxes = np.array([[*b.ring.min(axis=0), *b.ring.max(axis=0)] if len(b.ring) else [np.nan] * 4
                      for b in buildings], float).reshape(-1, 4)
    out = {}
    for lm in landmarks:
        replaces = dict.fromkeys(str(i) for i in lm.get("replaces") or [])   # the registry's order, once each
        for element in replaces:
            if element not in present:
                log.warning("landmark %s replaces %s, which matches no building: a typo in %s, or an id that names no "
                            "building since the last fetch?", lm["name"], element, REGISTRY.name)
        replaced = [k for k, element in enumerate(elements) if element in replaces]
        clear = clear_zone(lm, models_dir)
        for k in replaced:
            if not _polygon(buildings[k]).intersects(clear):
                log.warning("landmark %s replaces building %s, which lies outside its clear zone: a typo in %s, or an "
                            "id that names another building since the last fetch?", lm["name"], ids[k], REGISTRY.name)
        gone = {ids[k] for k in replaced}
        if not clear.is_empty:
            x0, y0, x1, y1 = clear.bounds
            near = np.flatnonzero((boxes[:, 2] >= x0) & (boxes[:, 0] <= x1) & (boxes[:, 3] >= y0) & (boxes[:, 1] <= y1))
            for i in near:
                poly = _polygon(buildings[i])
                if poly.area > 0 and poly.intersection(clear).area >= REMOVE_SHARE * poly.area:
                    gone.add(ids[i])
        gone |= {i for i in ids if _podium(i) in gone}   # a tower part goes with its podium
        out[lm["name"]] = gone
    return out


def removed(buildings, landmarks, models_dir=None) -> set[str]:
    """The ids of the buildings the landmarks remove: those of the OSM elements in a landmark's
    replaces (their parts '<id>/<k>' too), those with REMOVE_SHARE or more of their area inside a
    landmark's clear zone (clear_zone), and the tower parts ('<id>/t<k>') of every one of them."""
    return set().union(*removed_by(buildings, landmarks, models_dir).values())


def clear_polygons(path, models_dir=None) -> list[Polygon | MultiPolygon]:
    """The clear zones (clear_zone) of a landmarks.json; [] when there is no such file."""
    path = Path(path)
    if not path.exists():
        return []
    zones = [clear_zone(e, models_dir) for e in load_landmarks(path)]
    return [z for z in zones if not z.is_empty]


# ------------------------------------------------------------------ the stage

def _canopy(out_dir: Path):
    try:
        ndsm, _, px = load_canopy(out_dir)
    except FileNotFoundError:
        return None, None
    return ndsm, px


def run(cfg, raw_dir=None, out_dir=None) -> None:
    raw_dir, out_dir = Path(raw_dir or cfg.raw_dir), Path(out_dir or cfg.out_dir)
    path = out_dir / LANDMARKS_FILE
    if not cfg.landmarks_enabled:
        save_landmarks([], path)
        log.info("landmarks are off (landmarks.enabled in the config): %s lists none", path)
        return
    features = raw_dir / "osm_features.json"
    if not features.exists():
        raise FileNotFoundError(f"no {features}; run the features stage first")
    extent = Extent.from_config(cfg)
    osm = json.loads(features.read_text(encoding="utf-8"))
    ndsm, px = _canopy(out_dir)
    roads = load_roads(out_dir / "roads.json") if (out_dir / "roads.json").exists() else []
    entries = resolve(load_registry(), osm, extent, load_heights(out_dir=out_dir), ndsm, px, roads)
    save_landmarks(entries, path)
    for e in entries:
        log.info("%s at (%.1f, %.1f), yaw %.1f deg, base_z %.2f m, ground_z %.2f m, replaces %s", e["name"],
                 e["x"], e["y"], math.degrees(e["yaw"]), e["base_z"], e["ground_z"],
                 ", ".join(e["replaces"]) or "nothing")
    log.info("%d landmarks in %s", len(entries), path)


# ------------------------------------------------------------------ level files

def ts_static(level_name: str, lm: dict, part: str, shape: str) -> dict:
    """The TSStatic placing one shape of a landmark at its anchor and base, turned by its yaw."""
    return {"class": "TSStatic", "persistentId": bf.persistent_id("lm", lm["name"], part),
            "shapeName": f"/levels/{level_name}/{SHAPES_DIR}/{shape}.dae",
            "position": [lm["x"], lm["y"], lm["base_z"]], "rotationMatrix": rotation_matrix(lm["yaw"]),
            "useInstanceRenderData": True, "collisionType": "Collision Mesh", "decalType": "Collision Mesh"}


def level_content(cfg, data_dir, root, heights, extent) -> dict[str, list[dict]]:
    data_dir, root = Path(data_dir), Path(root)
    objs, bound = [], set()
    for lm in load_landmarks(data_dir / LANDMARKS_FILE):
        parts = sorted((data_dir / "models" / lm["name"]).glob("*.npz"))
        if not parts:
            log.warning("landmark %s has no model in %s (run the models stage): not placed", lm["name"],
                        data_dir / "models" / lm["name"])
            continue
        for part in parts:
            children, billboard = collada.read_npz(part)
            shape = lm["name"] if part.stem == MAIN_PART else f"{lm['name']}_{part.stem}"
            path = root / SHAPES_DIR / f"{shape}.dae"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(collada.shape_dae(children, billboard), encoding="utf-8")
            bound |= {m for _, mesh in children if mesh is not None for m in mesh.tris}
            objs.append(ts_static(cfg.level_name, lm, part.stem, shape))
    if not objs:
        return {}
    mats, files = materials.resolve(materials.load_registry(MATERIAL_REGISTRY), bound, cfg.level_name,
                                    materials.texture_index(data_dir))
    written = materials.write_materials(root, MATERIALS_FILE, mats)
    materials.copy_textures(files, data_dir, root)
    log.info("landmarks: %d shapes, %d materials (%d defined by an earlier feature), %d textures", len(objs),
             len(mats), len(mats) - len(written), len(files))
    return {GROUP: objs}
