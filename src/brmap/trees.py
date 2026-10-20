"""Trees: tree tops from the lidar canopy and NAIP, placed as stock BeamNG tree models and the
repo's own Blender tree models in a Forest.

Stock tree data, src/brmap/data/stock_trees.json, is written once by
tools/extract_stock_trees.py from the installed game (read only, never launched) and
committed; the build reads only that file (stock_trees()):

    {"game_version": "0.40.0.0",
     "items": {"br_<name>": {"source": "<stock item name>", "level": "<stock level>",
                             "shapeFile": "/levels/...dae", "native_height": float,
                             "materials": [<the Material names the shape binds>],
                             "collision": {"z0": m, "dz": m, "r": [m, ...]},
                             "item": {<stock TSForestItemData fields minus name/internalName/
                                      persistentId/class/__parent/shapeFile>}}},
     "materials": {<stock Material name>: <definition with absolute texture paths>},
     "material_conflicts": {<name>: [<level whose definition is kept>, <levels that differ>]}}

* Every candidate is extracted, so Task 7 can pick from an in-game lineup without a
  re-run: italy holm_oak (its item is holm_oak_test), holm_oak_city_tall,
  holm_oak_city_small, cork_oak_large_1, maritime_pine_tree, fanpalm_tall (as
  br_fanpalm_tall_italy); ECA tree_oak_large_a/b/c, tree_oak_sml_b; WCUSA oak_dry_a,
  pine_radiata_large, redwood_broad, fanpalm_tall.
* native_height = max z - min z of the highest-detail mesh's vertices.
* collision: how far the shape's collision mesh reaches from its trunk axis, r[k] between
  z0 + k dz and z0 + (k + 1) dz above its origin (unscaled). The stock meshes hold root
  flares and low branches 2-3 m out, so clear_collisions() checks them, not just the trunk.
* Items are br_-named; stock materials keep their stock names (the DAEs bind them by
  name). A level holds one definition per name, and a few names (oak_leaves_01,
  m_oak_leaves_distant, Oak_branch_01, leaves_strong) differ between the stock levels:
  the first candidate level's is kept and the others are listed in material_conflicts.

Custom trees, src/brmap/data/custom_trees.yaml (custom_trees()), are models the models stage
builds in Blender into out/models/<key>/main.npz: {key: {script, params, native_height,
collision: {trunk_radius_m, height_m}, item, materials, billboard}}. Their materials are in
src/brmap/data/tree_materials.yaml (TREE_MATERIALS, resolved by brmap.materials).

* tree_items(level_name) -> the stock items plus each custom one in the same form, with
  shapeFile /levels/<level_name>/art/shapes/trees/br_swamp/<key>.dae and its collision in the
  stock form: z0 0, dz 0.5, r the trunk radius in the bands below height_m and 0 above, up to
  native_height. assign, collision_reach, clear_collisions and level_content read it.
* The custom trees are the swamp trees (br_cypress_a, the mossy br_cypress_b and the water
  tupelo br_tupelo_a) and the Spanish moss overlays moss_item(oak) = "br_moss_" + oak[3:], one
  per stock live oak in LIVE_OAKS: strands that hang from that oak's branches, stood with it
  (level_content). They have no collision mesh and keep the oak's native height and item fields.

out/trees.json: [{"x", "y", "z", "item", "scale", "yaw", "height"}] in level coordinates
(inside the box), item a key of tree_items(), yaw in radians counter-clockwise, z the
terrain (heights.npy) - 0.2 and height the detected tree height in m (cap() and the checks
use it); a live oak hung with Spanish moss also has "moss": true.

* SPECIES: data only, context -> [(item, weight, max_height or None)], contexts
  evergreen, mixed, wetland, developed, other and small (under 8 m), holding the spec's
  mix as defaults (Task 7 may change the shapes). An entry is skipped for trees taller
  than its max_height. species(swamp) is SPECIES, or with swamp the swamp mixes: wetland
  SWAMP_WETLAND (the bald cypresses and the water tupelo in place of the redwood stand-in) and
  bank SWAMP_BANK (the same with more water oak); cfg.swamp_trees (config trees.swamp) picks it.
* detect_tops(ndsm, ndvi, px, extent) -> (N, 3) array of x, y, height: Gaussian
  sigma = 1 cell; ndsm >= 4 and ndvi >= 0.25; local maxima of a 5 x 5 maximum_filter,
  7 x 7 where the smoothed height is over 15 m. Nothing where the canopy is NaN.
* exclude(tops, buildings, roads, water, clear=(), bayous=()) -> (kept, dropped_by_reason):
  building footprints + 1 m, road corridors (centreline buffered by width / 2) + 1.5 m, bridge
  decks + 3 m, water, the landmarks' clear zones (shapely Polygons; reason "landmark") and the
  bayous' channels (bayous.json runs: each segment's centreline buffered by the wider of its two
  node widths / 2 + 0.5 m; reason "bayou").
* assign(tops, nlcd_at, rng_seed=0, swamp=True, moss_share=0.0, bayou_lines=(), wetland_polys=())
  -> list[dict]: species by context from species(swamp); scale = clamp(h / native_height, 0.6,
  1.6), or the item's SCALE_LIMITS; yaw from a hash of the position. The context is the NLCD
  one (context()); with swamp, a top that isn't "small" within BANK_DIST of a bayou run's
  centreline (bayou_lines) is "bank", and a top inside an OSM natural=wetland polygon
  (wetland_polys) is "wetland" whatever NLCD says. A "developed" tree whose item is in LIVE_OAKS
  gets "moss": true when a third number from its position's hash is below moss_share.
* clear_collisions(trees, roads) -> (kept, dropped_by_reason): drops the trees whose scaled
  collision mesh reaches into a road corridor from the ground to HEADROOM above it, or into
  a bridge deck's corridor at any height (a deck passes at crown height).
* cap(trees, max_count): the tallest per 50 m cell first, then the rest by height; logs
  the number dropped. max_count is cfg.trees_max.
* run(cfg, raw_dir=None, out_dir=None) writes out/trees.json from canopy.load_canopy,
  raw/nlcd.tif sampled at each top, out/buildings.json when present
  (buildings.load_buildings) and the styled buildings of out/styled.json as drawn
  (styled.drawn_buildings), roads, water, the clear zones of out/landmarks.json when
  present (landmarks.clear_polygons, grown to what the models of out/models stand on) and
  the runs of out/bayous.json when present
  (bayous.load_bayous): detect, exclude, assign (swamp = cfg.swamp_trees, moss_share =
  cfg.moss_share, the bayou runs' centrelines and the raw/osm_features.json "wetlands" polygons
  in level coordinates), clear_collisions, cap. OSM natural=tree nodes (raw/osm_features.json
  "trees") not within 5 m of a detected top are added.
* level_content(cfg, data_dir, root, heights, extent) -> {"vegetation": [Forest,
  ForestWindEmitter]} (the level.FEATURE_MODULES hook), writing under root:
  - art/forest/managedItemData.json with only the used items, each with name and
    internalName = the item key, class TSForestItemData and persistentId
    bf.persistent_id("fid", key);
  - forest/<item>.forest4.json, one line per tree: {"ctxid": 0, "pos": [x, y, z],
    "rotationMatrix": [cos t, sin t, 0, -sin t, cos t, 0, 0, 0, 1], "scale": s,
    "type": item} for a counter-clockwise yaw t, z = sample_height(heights, ...) - 0.2; a tree
    with moss also gets a line of type moss_item(item) with the same pos, rotationMatrix and
    scale, in forest/<moss item>.forest4.json;
  - art/shapes/trees/br_trees.materials.json with only the materials the used shapes need:
    the stock ones copied from stock_trees.json, the custom ones resolved from TREE_MATERIALS
    (their tex: textures copied to art/textures/br/);
  - for each custom item in use, CUSTOM_DIR/<key>.dae from out/models/<key>/main.npz
    (collada.read_npz, shape_dae) and, when its entry has a billboard,
    CUSTOM_DIR/<key>.dae.asset.json = IMPOSTERS, which asks the game for its imposters. A
    custom item whose model is missing (the models stage has not run) is an error, and so is
    a model that binds a material its entry does not list or whose billboard its entry
    contradicts.
"""
from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from functools import lru_cache
from importlib import resources
from pathlib import Path

import numpy as np
import rasterio
import shapely
import yaml
from pyproj import Transformer
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Polygon

from . import beamng_format as bf
from . import collada, materials
from .bayous import _wetland_polygon, load_bayous
from .buildings import STYLED_FILE, load_buildings
from .canopy import load_canopy
from .geo import LEVEL_CRS, Extent
from .roads import load_roads
from .styled import drawn_buildings
from .terrain import load_heights, sample_height
from .water import load_water

log = logging.getLogger(__name__)

OUT_DIR = Path("data/out")
RAW_DIR = Path("data/raw")
SIGMA = 1.0                # cells; the smoothing before looking for crowns
MIN_HEIGHT = 4.0           # m above ground
MIN_NDVI = 0.25
WINDOW, TALL_WINDOW = 5, 7  # cells a side of the local-maximum window
TALL = 15.0                # m; smoothed heights above this use TALL_WINDOW
BUILDING_CLEARANCE = 1.0   # m around a footprint
ROAD_CLEARANCE = 1.5       # m beyond a road corridor (centreline buffered by width / 2)
DECK_CLEARANCE = 3.0       # m beyond a bridge deck's corridor
OSM_MATCH = 5.0            # m; an OSM tree this close to a detected top is that tree
OSM_HEIGHT = 10.0          # m, for an OSM tree with no height tag and no canopy under it
SMALL = 8.0                # m; shorter trees use the "small" mix
SCALE_RANGE = (0.6, 1.6)
CAP_CELL = 50.0            # m
Z_OFFSET = -0.2            # m; trees sit a little into the ground so no root floats
HEADROOM = 3.0             # m above the ground over a road that no tree's collision mesh may enter
BAYOU_CLEARANCE = 0.5      # m beyond a bayou's half width
PALM_MAX = 20.0            # m; taller trees are not palms
BANK_DIST = 30.0           # m from a bayou run's centreline: a tree this close stands on its bank

# Species stand-ins, picked from an in-game lineup of the stock candidates (2026-10-16):
# context -> [(item, weight, max_height or None)]. Live oaks (wide, low, spreading crowns)
# are WCUSA oak_dry_a, italy cork_oak_large_1 and holm_oak and ECA tree_oak_large_b; water
# oaks are ECA tree_oak_large_a and _c; loblolly pine is WCUSA pine_radiata_large; bald
# cypress is the WCUSA redwood scaled down to the measured height; palms are rare.
# holm_oak_city_tall is left out: too spindly for a Baton Rouge street tree.
_BROADLEAF = [("br_oak_dry_a", 0.20, None), ("br_cork_oak_large_1", 0.15, None),
              ("br_tree_oak_large_b", 0.12, None), ("br_holm_oak", 0.10, None),
              ("br_tree_oak_large_a", 0.15, None), ("br_tree_oak_large_c", 0.20, None),
              ("br_pine_radiata_large", 0.06, None), ("br_fanpalm_tall", 0.02, PALM_MAX)]
SPECIES: dict[str, list[tuple[str, float, float | None]]] = {
    "evergreen": [("br_pine_radiata_large", 1.0, None)],
    "mixed": [("br_pine_radiata_large", 0.5, None), *[(i, w / 2, h) for i, w, h in _BROADLEAF]],
    "wetland": [("br_redwood_broad", 0.6, None), ("br_tree_oak_large_c", 0.4, None)],
    "developed": list(_BROADLEAF),
    "other": list(_BROADLEAF),
    "small": [("br_tree_oak_sml_b", 0.5, None), ("br_holm_oak_city_small", 0.5, None)],
}
# with the swamp trees (config trees.swamp), the bald cypresses (br_cypress_b hung with Spanish moss)
# and the water tupelo take the redwood's place in the wetlands, and line the bayous' banks
SWAMP_WETLAND = [("br_cypress_a", .35, None), ("br_cypress_b", .15, None), ("br_tupelo_a", .35, None),
                 ("br_tree_oak_large_c", .15, None)]
SWAMP_BANK = [("br_cypress_a", .30, None), ("br_cypress_b", .15, None), ("br_tupelo_a", .30, None),
              ("br_tree_oak_large_c", .25, None)]
SMALL_CONTEXTS = ("developed", "other")  # where a tree under SMALL is a small street tree
# the live oaks of the species mixes; in town (the "developed" context) moss_share of them carry
# Spanish moss, the custom overlay moss_item(oak)
LIVE_OAKS = ("br_oak_dry_a", "br_cork_oak_large_1", "br_tree_oak_large_b", "br_holm_oak")
# per-item scale limits where SCALE_RANGE does not fit the shape: the redwood is 38 m tall; the swamp
# trees keep their lidar height from 6.6 m (0.3 x the 22 m cypress) to 35.2 m, so a short wetland tree is
# not drawn at 0.6 x its model's height
SCALE_LIMITS = {"br_redwood_broad": (0.3, 0.75), "br_cypress_a": (0.3, 1.6),
                "br_cypress_b": (0.33, 1.76), "br_tupelo_a": (0.367, 1.956)}
NLCD_CONTEXT = {42: "evergreen", 43: "mixed", 90: "wetland", 95: "wetland",
                **dict.fromkeys((21, 22, 23, 24), "developed")}

GROUP = "vegetation"
FOREST_DIR = "forest"
ITEM_DATA_FILE = "art/forest/managedItemData.json"  # the loader takes art/forest/<letters>Data.json
MATERIALS_FILE = "art/shapes/trees/br_trees.materials.json"
CUSTOM_DIR = "art/shapes/trees/br_swamp"      # the custom trees' shapes, under the level folder
TREE_MATERIALS = Path(__file__).resolve().parent / "data" / "tree_materials.yaml"
MODEL_PART = "main"                           # out/models/<key>/main.npz is the tree's shape
COLLISION_DZ = 0.5                            # m; the stock collision form's band height
# <shape>.dae.asset.json: asks the game to render the shape's imposters (256 px, 16 views round
# the equator, size 25 as in its bb_autobillboard25 node); the stock cypress_tree's asks
# detailLevel 2 and 8 equator steps
IMPOSTERS = {"imposters": [{"detailLevel": 1, "dimension": 256, "equatorSteps": 16, "includePoles": False,
                            "polarAngle": 0, "polarSteps": 0, "size": 25}]}


@lru_cache(maxsize=1)
def stock_trees() -> dict:
    """src/brmap/data/stock_trees.json (tools/extract_stock_trees.py). Don't modify it."""
    text = resources.files("brmap").joinpath("data/stock_trees.json").read_text(encoding="utf-8")
    return json.loads(text)


@lru_cache(maxsize=1)
def custom_trees() -> dict[str, dict]:
    """src/brmap/data/custom_trees.yaml: the Blender tree models. Don't modify it."""
    text = resources.files("brmap").joinpath("data/custom_trees.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text) or {}


def _stock_collision(collision: dict, native_height: float) -> dict:
    """A custom tree's trunk prism in the stock collision form: 0.5 m bands from its origin up to
    its native height, the trunk radius in those below height_m and 0 above."""
    bands = math.ceil(native_height / COLLISION_DZ)
    r = [float(collision["trunk_radius_m"]) if k * COLLISION_DZ < collision["height_m"] else 0.0
         for k in range(bands)]
    return {"z0": 0.0, "dz": COLLISION_DZ, "r": r}


@lru_cache(maxsize=None)
def tree_items(level_name: str = "baton_rouge") -> dict[str, dict]:
    """The stock items and the custom ones, each with shapeFile, native_height, materials,
    collision (the stock form) and item. Don't modify it."""
    items = dict(stock_trees()["items"])
    for key, e in custom_trees().items():
        if key in items:
            raise ValueError(f"custom_trees.yaml: {key} is a stock item already")
        items[key] = {"shapeFile": f"/levels/{level_name}/{CUSTOM_DIR}/{key}.dae",
                      "native_height": float(e["native_height"]), "materials": list(e["materials"]),
                      "collision": _stock_collision(e["collision"], float(e["native_height"])),
                      "item": dict(e["item"])}
    return items


def species(swamp: bool) -> dict[str, list[tuple[str, float, float | None]]]:
    """The species mix per context: SPECIES, with the wetland mix SWAMP_WETLAND and the bayou banks'
    SWAMP_BANK when swamp."""
    return {**SPECIES, "wetland": list(SWAMP_WETLAND), "bank": list(SWAMP_BANK)} if swamp else SPECIES


def moss_item(oak: str) -> str:
    """The Spanish moss overlay of a live oak: br_moss_<oak without br_>."""
    return "br_moss_" + oak[3:]


# ------------------------------------------------------------------ tree tops

# The two filters below give what scipy.ndimage's gaussian_filter and maximum_filter
# give, from shifted whole-row slices: on the 8192 x 8192 grid scipy's passes along
# axis 0 take about 12 s each on the build laptop, these about 1 s.

def _shifted(p: np.ndarray, k: int, axis: int, shape) -> np.ndarray:
    return p[k:k + shape[0]] if axis == 0 else p[:, k:k + shape[1]]


def _gaussian(a: np.ndarray, sigma: float) -> np.ndarray:
    """ndimage.gaussian_filter(a, sigma) (mode reflect, truncate 4), in float32."""
    r = int(4.0 * sigma + 0.5)
    w = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    w = (w / w.sum()).astype(np.float32)
    a = np.asarray(a, np.float32)
    for axis in (0, 1):
        p = np.pad(a, [(r, r) if ax == axis else (0, 0) for ax in (0, 1)], mode="symmetric")
        out, tmp = np.zeros_like(a), np.empty_like(a)
        for k, wk in enumerate(w):
            out += np.multiply(_shifted(p, k, axis, a.shape), wk, out=tmp)
        a = out
    return a


def _max_filter(a: np.ndarray, size: int) -> np.ndarray:
    """ndimage.maximum_filter(a, size, mode="constant", cval=-inf)."""
    r = size // 2
    for axis in (0, 1):
        p = np.pad(a, [(r, r) if ax == axis else (0, 0) for ax in (0, 1)], constant_values=-np.inf)
        out = _shifted(p, 0, axis, a.shape).copy()
        for k in range(1, size):
            np.maximum(out, _shifted(p, k, axis, a.shape), out=out)
        a = out
    return a


def _smooth(ndsm: np.ndarray) -> np.ndarray:
    """Gaussian-smoothed ndsm that leaves NaN cells out (normalized convolution); NaN stays NaN."""
    valid = np.isfinite(ndsm)
    num = _gaussian(np.where(valid, ndsm, 0.0), SIGMA)
    den = _gaussian(valid, SIGMA)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = num / den
    s[~valid] = np.nan
    return s


def detect_tops(ndsm, ndvi, px: float, extent: Extent) -> np.ndarray:
    """(N, 3) x, y, height of the tree crowns on the analysis grid (row 0 = north).

    Crowns are local maxima of the smoothed ndsm; the height is the unsmoothed ndsm
    there. A flat crown (equal maxima side by side) gives one top, its middle cell.
    """
    ndsm, ndvi = np.asarray(ndsm, np.float32), np.asarray(ndvi, np.float32)
    s = _smooth(ndsm)
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(s) & (s >= MIN_HEIGHT) & np.isfinite(ndvi) & (ndvi >= MIN_NDVI)
    if not ok.any():
        return np.empty((0, 3))
    f = np.where(np.isfinite(s), s, -np.inf).astype(np.float32)
    peak = ok & (f == _max_filter(f, WINDOW))
    tall = peak & (f > TALL)
    if tall.any():
        peak &= ~tall | (f == _max_filter(f, TALL_WINDOW))
    # neighbouring maxima of the same window are equal: one top per such group
    labels, _ = ndimage.label(peak, structure=np.ones((3, 3), bool))
    rows, cols = np.nonzero(peak)
    lab = labels[rows, cols]
    count = np.bincount(lab)
    r_mid, c_mid = np.bincount(lab, rows)[lab] / count[lab], np.bincount(lab, cols)[lab] / count[lab]
    order = np.lexsort(((rows - r_mid) ** 2 + (cols - c_mid) ** 2, lab))
    _, first = np.unique(lab[order], return_index=True)
    rows, cols = rows[order[first]], cols[order[first]]
    half = extent.half_size
    return np.column_stack([-half + (cols + 0.5) * px, half - (rows + 0.5) * px,
                            ndsm[rows, cols].astype(float)])


def _corridors(roads, bridges: bool, clearance: float) -> list:
    lines = [(LineString(r.nodes[:, :2]), r.width / 2 + clearance) for r in roads
             if bool(r.is_bridge) == bridges and len(r.nodes) >= 2]
    if not lines:
        return []
    return list(shapely.buffer([g for g, _ in lines], [d for _, d in lines]))


def _channels(runs) -> list:
    """Each bayou run segment's centreline buffered by the wider of its node widths / 2 + 0.5 m."""
    segs, dists = [], []
    for r in runs:
        n = np.asarray(r["nodes"], float).reshape(-1, 4)
        segs += [((a[0], a[1]), (b[0], b[1])) for a, b in zip(n[:-1], n[1:])]
        dists += list(np.maximum(n[:-1, 3], n[1:, 3]) / 2 + BAYOU_CLEARANCE)
    return list(shapely.buffer(shapely.linestrings(segs), dists)) if segs else []


def exclude(tops, buildings, roads, water, clear=(), bayous=()) -> tuple[np.ndarray, dict[str, int]]:
    """Tops not on a road (+1.5 m), a bridge deck (+3 m), a footprint (+1 m), water, a
    landmark's clear zone (the Polygons in `clear`) or a bayou's channel (the bayous.json runs in
    `bayous`, + 0.5 m beyond their half width).

    Each dropped top is counted under the first of those reasons that applies.
    """
    tops = np.asarray(tops, float).reshape(-1, 3)
    zones = {"road": lambda: _corridors(roads, False, ROAD_CLEARANCE),
             "deck": lambda: _corridors(roads, True, DECK_CLEARANCE),
             "building": lambda: [Polygon(b.ring, b.holes).buffer(BUILDING_CLEARANCE) for b in buildings],
             "water": lambda: [w.polygon for w in water],
             "landmark": lambda: list(clear),
             "bayou": lambda: _channels(bayous)}
    dropped = dict.fromkeys(zones, 0)
    if not len(tops):
        return tops, dropped
    points = shapely.STRtree(shapely.points(tops[:, 0], tops[:, 1]))
    alive = np.ones(len(tops), bool)
    for reason, geoms in zones.items():
        g = geoms()
        if not g:
            continue
        hit = np.zeros(len(tops), bool)
        hit[points.query(g, predicate="intersects")[1]] = True
        dropped[reason] = int((hit & alive).sum())
        alive &= ~hit
    return tops[alive], dropped


# ------------------------------------------------------------------ species, scale, cap

def context(nlcd_code: int, height: float) -> str:
    """The SPECIES context of a tree: by its NLCD land-cover class, and in town (developed or
    other land) "small" under 8 m."""
    ctx = NLCD_CONTEXT.get(int(nlcd_code), "other")
    return "small" if height < SMALL and ctx in SMALL_CONTEXTS else ctx


def scale_limits(item: str) -> tuple[float, float]:
    """(min, max) scale of a tree item."""
    return SCALE_LIMITS.get(item, SCALE_RANGE)


def _mix(h: np.ndarray) -> np.ndarray:
    """splitmix64's finalizer, on uint64 arrays."""
    h = (h ^ (h >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    h = (h ^ (h >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return h ^ (h >> np.uint64(31))


def _uniforms(xy: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three independent numbers in [0, 1) per position (to 0.1 m), the same on every run."""
    q = np.round(np.asarray(xy, float) * 10).astype(np.int64).astype(np.uint64)
    with np.errstate(over="ignore"):
        h = _mix(_mix(q[:, 0] + np.uint64(seed)) ^ (q[:, 1] * np.uint64(0x9E3779B97F4A7C15)))
        out = tuple(_mix(h ^ np.uint64(k)) for k in (1, 2, 3))
    return tuple((v >> np.uint64(11)) / 2.0 ** 53 for v in out)


def _near(tops: np.ndarray, geoms, predicate: str, **kw) -> np.ndarray:
    """Which tops (x, y) the shapely geometries reach by predicate (intersects, dwithin, ...)."""
    hit = np.zeros(len(tops), bool)
    if len(geoms):
        hit[shapely.STRtree(list(geoms)).query(shapely.points(tops[:, 0], tops[:, 1]), predicate=predicate,
                                                **kw)[0]] = True
    return hit


def _pick(options, height: float, u: float) -> str:
    fits = [(item, w) for item, w, top in options if top is None or height <= top]
    fits = fits or [(item, w) for item, w, _ in options]
    total, acc = sum(w for _, w in fits), 0.0
    for item, w in fits:
        acc += w / total
        if u < acc:
            return item
    return fits[-1][0]


def assign(tops, nlcd_at, rng_seed: int = 0, swamp: bool = True, moss_share: float = 0.0, bayou_lines=(),
           wetland_polys=()) -> list[dict]:
    """One tree dict per top. nlcd_at holds the NLCD class at each top, or is a function
    of the x and y arrays that returns them. z is left 0; run() sets it. swamp picks the
    species mix (species()) and turns on the bayou banks (bayou_lines, shapely LineStrings) and
    the OSM wetlands (wetland_polys, shapely Polygons) as contexts; moss_share is the share of the
    live oaks in town that get "moss": True. See the module docstring."""
    tops = np.asarray(tops, float).reshape(-1, 3)
    if not len(tops):
        return []
    codes = nlcd_at(tops[:, 0], tops[:, 1]) if callable(nlcd_at) else nlcd_at
    codes = np.broadcast_to(np.asarray(codes).astype(int), (len(tops),))
    mix = species(swamp)
    contexts = [context(code, h) for code, h in zip(codes, tops[:, 2])]
    if swamp:
        bank = _near(tops, bayou_lines, "dwithin", distance=BANK_DIST)
        wet = _near(tops, wetland_polys, "intersects")
        contexts = ["wetland" if w else "bank" if b and c != "small" else c
                    for c, b, w in zip(contexts, bank, wet)]
    native = {k: v["native_height"] for k, v in tree_items().items()}
    pick_u, yaw_u, moss_u = _uniforms(tops[:, :2], rng_seed)
    out = []
    for (x, y, h), ctx, u, v, m in zip(tops, contexts, pick_u, yaw_u, moss_u):
        item = _pick(mix[ctx], h, u)
        lo, hi = scale_limits(item)
        scale = min(max(h / native[item], lo), hi)
        tree = {"x": round(float(x), 2), "y": round(float(y), 2), "z": 0.0, "item": item,
                "scale": round(scale, 3), "yaw": math.floor(v * 2 * math.pi * 1e4) / 1e4,
                "height": round(float(h), 2)}
        if ctx == "developed" and item in LIVE_OAKS and m < moss_share:
            tree["moss"] = True
        out.append(tree)
    return out


def collision_reach(item: str, scale: float, z0: float = -math.inf, z1: float = math.inf) -> float:
    """How far (m) from its trunk the item's collision mesh reaches between z0 and z1 above
    its origin, at the given scale (tree_items() "collision"; 0 when it has none)."""
    col = tree_items()[item].get("collision")
    if not col or not col["r"]:
        return 0.0
    r = np.asarray(col["r"], float)
    lo = (col["z0"] + col["dz"] * np.arange(len(r))) * scale
    inside = (lo + col["dz"] * scale >= z0) & (lo <= z1)
    return float(r[inside].max()) * scale if inside.any() else 0.0


def clear_collisions(trees: list[dict], roads) -> tuple[list[dict], dict[str, int]]:
    """The trees whose collision mesh stays out of every road corridor (centreline buffered
    by width / 2) from the ground to HEADROOM above it, and out of every bridge deck's
    corridor at any height, since a deck passes at crown height. exclude() keeps the trunks
    off; this keeps off the stock meshes' root flares and low branches (tree_items() "collision"
    via collision_reach).

    Each dropped tree is counted under "road" or "deck"; the kept trees stay in their order.
    """
    dropped = {"road": 0, "deck": 0}
    if not trees:
        return [], dropped
    points = shapely.points([t["x"] for t in trees], [t["y"] for t in trees])
    alive = np.ones(len(trees), bool)
    for reason, band in (("road", (-Z_OFFSET, HEADROOM - Z_OFFSET)), ("deck", (-math.inf, math.inf))):
        corridors = _corridors(roads, reason == "deck", 0.0)
        reach_of = {}
        for t in trees:
            key = (t["item"], float(t["scale"]))
            if key not in reach_of:
                reach_of[key] = collision_reach(key[0], key[1], *band)
        reach = np.array([reach_of[(t["item"], float(t["scale"]))] for t in trees])
        if not corridors or not (reach > 0).any():
            continue
        (near, _), dist = shapely.STRtree(corridors).query_nearest(
            points, max_distance=float(reach.max()), return_distance=True, all_matches=False)
        hit = np.zeros(len(trees), bool)
        hit[near[dist < reach[near]]] = True
        dropped[reason] = int((hit & alive).sum())
        alive &= ~hit
    return [t for t, keep in zip(trees, alive) if keep], dropped


def cap(trees: list[dict], max_count: int) -> list[dict]:
    """At most max_count trees: the tallest of each 50 m cell first, then the rest by height.
    The kept trees stay in their order."""
    if len(trees) <= max_count:
        return list(trees)
    seen, first, rest = set(), [], []
    for i in sorted(range(len(trees)), key=lambda i: -trees[i]["height"]):
        key = (math.floor(trees[i]["x"] / CAP_CELL), math.floor(trees[i]["y"] / CAP_CELL))
        (rest if key in seen else first).append(i)
        seen.add(key)
    keep = set((first + rest)[:max(0, max_count)])
    log.info("tree cap %d: dropped %d of %d trees (%d cells of %.0f m)", max_count, len(trees) - len(keep),
             len(trees), len(first), CAP_CELL)
    return [t for i, t in enumerate(trees) if i in keep]


# ------------------------------------------------------------------ inputs and stage

def save_trees(trees: list[dict], path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trees), encoding="utf-8")
    return path


def load_trees(path=OUT_DIR / "trees.json") -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def nlcd_sampler(path, extent: Extent):
    """A function (x, y arrays) -> NLCD class at each point (0 off the raster or without it)."""
    path = Path(path)
    if not path.exists():
        log.warning("no %s: every tree gets the 'other' species mix", path)
        return lambda x, y: np.zeros(np.size(x), np.uint8)
    with rasterio.open(path) as src:
        codes, inverse, crs = src.read(1), ~src.transform, src.crs
    to_src = Transformer.from_crs(LEVEL_CRS, crs, always_xy=True)

    def at(x, y):
        sx, sy = to_src.transform(*extent.level_to_utm(np.asarray(x, float), np.asarray(y, float)))
        a, b, c, d, e, f = inverse[:6]
        col, row = np.floor(a * sx + b * sy + c).astype(int), np.floor(d * sx + e * sy + f).astype(int)
        ok = (row >= 0) & (row < codes.shape[0]) & (col >= 0) & (col < codes.shape[1])
        out = np.zeros(len(col), codes.dtype)
        out[ok] = codes[row[ok], col[ok]]
        return out

    return at


def _tag_height(tags: dict) -> float | None:
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(?:m)?\s*", str(tags.get("height", "")))
    h = float(m.group(1)) if m else None
    return h if h is not None and 1.0 <= h <= 60.0 else None


def _canopy_at(ndsm: np.ndarray, extent: Extent, px: float, x: float, y: float) -> float | None:
    """The highest canopy cell within a cell of (x, y), if there is one."""
    n = ndsm.shape[0]
    col, row = int((x + extent.half_size) // px), int((extent.half_size - y) // px)
    win = ndsm[max(0, row - 1):min(n, row + 2), max(0, col - 1):min(n, col + 2)]
    win = win[np.isfinite(win)]
    return float(win.max()) if win.size and win.max() >= 2.0 else None


def osm_trees(path, extent: Extent, tops: np.ndarray, ndsm: np.ndarray, px: float) -> np.ndarray:
    """(M, 3) OSM natural=tree nodes in the box that no detected top within 5 m stands for.
    Height: the height tag, else the canopy there, else OSM_HEIGHT."""
    path = Path(path)
    nodes = (json.loads(path.read_text(encoding="utf-8")).get("trees") or []) if path.exists() else []
    if not nodes:
        return np.empty((0, 3))
    lon, lat = np.array([[t["lon"], t["lat"]] for t in nodes], float).T
    x, y = (np.atleast_1d(v) for v in extent.to_level(lon, lat))
    keep = np.asarray(extent.contains(x, y))
    if len(tops):
        dist, _ = cKDTree(tops[:, :2]).query(np.column_stack([x, y]))
        keep &= dist > OSM_MATCH
    out = []
    for i in np.flatnonzero(keep):
        h = _tag_height(nodes[i].get("tags") or {}) or _canopy_at(ndsm, extent, px, x[i], y[i]) or OSM_HEIGHT
        out.append((float(x[i]), float(y[i]), h))
    log.info("%d OSM trees in the box, %d added (the rest stand for detected tops)",
             int(np.asarray(extent.contains(x, y)).sum()), len(out))
    return np.asarray(out, float).reshape(-1, 3)


def _optional(path: Path, loader, what: str) -> list:
    if path.exists():
        return loader(path)
    log.warning("no %s: trees are not kept off %s", path, what)
    return []


def wetland_polygons(path, extent: Extent) -> list:
    """The raw/osm_features.json "wetlands" polygons (natural=wetland) in level coordinates."""
    path = Path(path)
    wetlands = (json.loads(path.read_text(encoding="utf-8")).get("wetlands") or []) if path.exists() else []
    polys = [_wetland_polygon(p, extent) for w in wetlands for p in w.get("polygons") or []]
    return [g for g in polys if not g.is_empty]


def bayou_centrelines(runs) -> list[LineString]:
    """The centreline (x, y) of each bayous.json run of two nodes or more."""
    xy = [np.asarray(r["nodes"], float).reshape(-1, 4)[:, :2] for r in runs]
    return [LineString(c) for c in xy if len(c) >= 2]


def run(cfg, raw_dir=None, out_dir=None) -> None:
    extent = Extent.from_config(cfg)
    raw_dir, out_dir = Path(raw_dir or cfg.raw_dir), Path(out_dir or cfg.out_dir)
    ndsm, ndvi, px = load_canopy(out_dir)
    tops = detect_tops(ndsm, ndvi, px, extent)
    log.info("%d tree tops on the %d x %d canopy grid (%.1f %% has lidar)", len(tops), ndsm.shape[0],
             ndsm.shape[1], 100.0 * float(np.isfinite(ndsm).mean()))
    tops = np.vstack([tops, osm_trees(raw_dir / "osm_features.json", extent, tops, ndsm, px)])
    roads = _optional(out_dir / "roads.json", load_roads, "roads")
    footprints = _optional(out_dir / "buildings.json", load_buildings, "buildings")
    footprints += drawn_buildings(out_dir / STYLED_FILE)  # the restyled ones, as their walls stand
    water = _optional(out_dir / "water.json", load_water, "water")
    from .landmarks import clear_polygons  # landmarks builds on this module

    runs = load_bayous(out_dir / "bayous.json")["runs"] if (out_dir / "bayous.json").exists() else []
    clear = clear_polygons(out_dir / "landmarks.json", out_dir / "models")
    tops, dropped = exclude(tops, footprints, roads, water, clear, runs)
    log.info("dropped %s; %d left", ", ".join(f"{v} on {k}" for k, v in dropped.items()), len(tops))
    lines, wet = bayou_centrelines(runs), wetland_polygons(raw_dir / "osm_features.json", extent)
    placed = assign(tops, nlcd_sampler(raw_dir / "nlcd.tif", extent), swamp=cfg.swamp_trees,
                    moss_share=cfg.moss_share, bayou_lines=lines, wetland_polys=wet)
    log.info("%d bayou runs and %d OSM wetland polygons set the swamp trees%s; %d live oaks with moss",
             len(lines), len(wet), "" if cfg.swamp_trees else " (trees.swamp is off: not used)",
             sum(t.get("moss", False) for t in placed))
    placed, dropped = clear_collisions(placed, roads)
    log.info("dropped %s (their collision mesh); %d left",
             ", ".join(f"{v} reaching a {k}" for k, v in dropped.items()), len(placed))
    placed = cap(placed, cfg.trees_max)
    if placed:
        z = sample_height(load_heights(out_dir=out_dir), extent, np.array([t["x"] for t in placed]),
                          np.array([t["y"] for t in placed])) + Z_OFFSET
        for t, v in zip(placed, np.atleast_1d(z)):
            t["z"] = round(float(v), 3)
    save_trees(placed, out_dir / "trees.json")
    log.info("%d trees (%d with moss): %s", len(placed), sum(t.get("moss", False) for t in placed),
             ", ".join(f"{k} {v}" for k, v in Counter(t["item"] for t in placed).most_common()))


# ------------------------------------------------------------------ level files

def rotation_matrix(yaw: float) -> list[float]:
    """Forest/TSStatic rotationMatrix for a counter-clockwise yaw (verified in game)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return [round(v, 6) + 0.0 for v in (c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0)]


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")


def _custom_shape(key: str, entry: dict, data_dir: Path, root: Path) -> None:
    """CUSTOM_DIR/<key>.dae from the custom tree's built model and, when it has a billboard,
    its .dae.asset.json asking for imposters."""
    part = data_dir / "models" / key / f"{MODEL_PART}.npz"
    if not part.is_file():
        raise FileNotFoundError(f"no {part}: the custom tree {key} is not built (run the models stage)")
    children, billboard = collada.read_npz(part)
    unlisted = sorted({m for _, mesh in children if mesh is not None for m in mesh.tris} - set(entry["materials"]))
    if unlisted:
        raise ValueError(f"{part} binds {', '.join(unlisted)}, which custom_trees.yaml does not list for {key}")
    if bool(billboard) != bool(entry["billboard"]):
        raise ValueError(f"{part} has {'a' if billboard else 'no'} billboard node, but custom_trees.yaml says "
                         f"billboard: {entry['billboard']} for {key} (run the models stage again)")
    path = root / CUSTOM_DIR / f"{key}.dae"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(collada.shape_dae(children, billboard), encoding="utf-8")
    if billboard:
        _write_json(path.with_name(path.name + ".asset.json"), IMPOSTERS)


def level_content(cfg, data_dir, root, heights, extent) -> dict[str, list[dict]]:
    data_dir, root = Path(data_dir), Path(root)
    placed = load_trees(data_dir / "trees.json")
    if not placed:
        return {}
    items, custom = tree_items(cfg.level_name), custom_trees()
    unknown = sorted(({t["item"] for t in placed} | {moss_item(t["item"]) for t in placed if t.get("moss")})
                     - set(items))
    if unknown:
        raise ValueError(f"trees.json names items that are neither in stock_trees.json nor in custom_trees.yaml: "
                         f"{', '.join(unknown)} (run the trees stage again)")
    by_item: dict[str, list[dict]] = {}
    for t in placed:
        by_item.setdefault(t["item"], []).append(t)
        if t.get("moss"):                           # its moss stands with it: same pos, rotation and scale
            by_item.setdefault(moss_item(t["item"]), []).append(t)
    used_custom = [key for key in sorted(by_item) if key in custom]
    for key in used_custom:
        _custom_shape(key, custom[key], data_dir, root)

    item_data = {}
    for key in sorted(by_item):
        item_data[key] = {"name": key, "internalName": key, "class": "TSForestItemData",
                          "persistentId": bf.persistent_id("fid", key), "shapeFile": items[key]["shapeFile"],
                          **items[key]["item"]}
        group = by_item[key]
        z = np.atleast_1d(sample_height(heights, extent, np.array([t["x"] for t in group], float),
                                        np.array([t["y"] for t in group], float))) + Z_OFFSET
        lines = [{"ctxid": 0, "pos": [round(float(t["x"]), 3), round(float(t["y"]), 3), round(float(zi), 3)],
                  "rotationMatrix": rotation_matrix(float(t["yaw"])), "scale": round(float(t["scale"]), 3),
                  "type": key} for t, zi in zip(group, z)]
        path = root / FOREST_DIR / f"{key}.forest4.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(o) + "\n" for o in lines), encoding="utf-8")
    _write_json(root / ITEM_DATA_FILE, item_data)

    stock = stock_trees()["materials"]
    mats = {name: {**stock[name], "persistentId": bf.persistent_id("material", name)}
            for name in {m for key in by_item if key not in custom for m in items[key]["materials"]}}
    if used_custom:
        names = {m for key in used_custom for m in items[key]["materials"]}
        resolved, files = materials.resolve(materials.load_registry(TREE_MATERIALS), names, cfg.level_name,
                                            materials.texture_index(data_dir))
        clash = sorted(set(resolved) & set(mats))
        if clash:
            raise ValueError(f"tree_materials.yaml redefines the stock tree materials {', '.join(clash)}")
        mats |= resolved
        materials.copy_textures(files, data_dir, root)
    _write_json(root / MATERIALS_FILE, dict(sorted(mats.items())))

    z0 = float(sample_height(heights, extent, 0.0, 0.0))
    forest = {"name": "theForest", "class": "Forest",
              "persistentId": bf.persistent_id("forest", cfg.level_name), "lodReflectScalar": 0.15}
    wind = {"class": "ForestWindEmitter", "persistentId": bf.persistent_id("forest_wind", cfg.level_name),
            "position": [0.0, 0.0, round(z0 + 30.0, 2)], "strength": 0.15, "gustYawFrequency": 0.1,
            "hasMount": "0"}
    log.info("forest: %d trees of %d types (%d custom), %d materials", len(placed), len(by_item), len(used_custom),
             len(mats))
    return {GROUP: [forest, wind]}
