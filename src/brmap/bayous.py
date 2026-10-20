"""Bayous: OSM waterways drawn as murky River objects in the channels the lidar shows, and swamp
pools as WaterBlocks. The terrain is not carved: a River's surface sits FILL above the channel
bottom found in the terrain, so heights.npy stays as it is, and bayous never go into water.json.

out/bayous.json (run) holds {"runs": [{"id", "name", "kind", "nodes": [[x, y, z, width], ...]}],
"pools": [{"id", "level_z", "rects": [[cx, cy, sx, sy], ...]}]} in level coordinates.

* chains(waterways, extent, kinds, min_length_m, water_polygons, roads) -> [{"id", "name", "kind",
  "xy"}]: raw/osm_features.json "waterways" (lon/lat) of the given kinds in level coordinates.
  Culverts are dropped (which splits a chain there); ways of the same kind and name (or both
  unnamed) are joined end to end where exactly two way ends meet within JOIN_TOL (a confluence
  stays apart); the chains are clipped to the box and cut out of every drivable ground road's
  corridor (centreline buffered by width / 2) + ROAD_GAP and every water.json polygon (the
  Mississippi and the lakes); pieces shorter than max(min_length_m, MIN_CHAIN) are dropped. id
  joins the way ids with "+", and every piece of a chain keeps it.
* profile(xy, kind, heights, extent, spacing_m) -> [(K, 4) arrays of x, y, z, width]: the chain
  resampled every spacing_m; at each node the terrain on the normal every 1 m out to
  +-SEARCH_HALF. Bottom is the lowest of it within +-BOTTOM_HALF, the surface bottom + FILL. A
  node whose bank (the BANK_PCT percentile of its section) is less than MIN_DEPTH above the bottom
  has no channel: the run is split there and no water drawn. Within each run, z is the surface
  smoothed by a running median over SMOOTH nodes; the span is where the terrain is under z round
  the bottom (crossings interpolated, the section's end on a side with none). Each node moves
  along its normal onto the middle of its span, the offsets smoothed by the same running median,
  so the ribbon follows the channel the lidar shows rather than the OSM line; the width is the
  narrowest about that node reaching both ends of its span (the span itself where the node is its
  middle), clamped to WIDTH[kind]. A span with no crossing on a side (the water runs past the
  section) has no middle: that node's offset is 0 before the smoothing and its width is the span,
  clamped. Nodes stay in the box. The widths are then smoothed along the run so the water's edge
  does not step from node to node: the centred moving average over WIDTH_SMOOTH nodes (fewer at the
  run's ends) of the widest width within WIDTH_SMOOTH nodes of each node, within WIDTH[kind]. No
  node is narrower than the span that reaches both its banks (a plain average of the widths
  narrows the wide nodes, and their edges hang over the bank's slope).
* pools(wetlands, heights, extent, roads, water) -> [{"id", "level_z", "rects"}]: each polygon of a
  raw/osm_features.json "wetlands" entry of a kind in POOL_KINDS, clipped to the box, in the order
  of their ids. A WaterBlock is water everywhere in its box under its level (on the full map the
  swamps' boxes stood as raised slabs over the Mississippi), so the rects start as the
  level.water_rects boxes (which cover a polygon and WATER_MARGIN round it) of the polygon less the
  water bodies (water.json's) grown by WATER_MARGIN. A box that shows more than POOL_FLOOD_CELLS
  terrain cells (centres) of other water, or that holds more than POOL_FLOOD_CELLS cells more than
  POOL_FLOOD_DEPTH under the level outside the polygon grown by WATER_MARGIN (square at its
  corners, as the boxes grow it), is split in halves across its long side, again and again down
  to halves of POOL_MIN_SIDE; a piece that still fails, or no longer meets the polygon less the
  water, is dropped, and after the search so is one that draws none of the swamp's water: one
  that covers no terrain cell (centre) inside the polygon under the level where no other water
  shows (the cells over other water the rules tolerate, the WATER_MARGIN round the polygon and the
  cells flooded past it are not the swamp's). Other water is a block of a water body (its
  water_rects boxes) or a box of a pool before it whose level differs by more than
  POOL_LEVEL_TOL; it shows in the cells inside both boxes whose terrain is under the lower of
  the two levels + POOL_SHOW_DEPTH, where one surface would stand over the other. Where the
  terrain is above it (the batture inside the Mississippi's boxes), the lower water is under the
  ground and the pool covers it. Every box is checked as
  bf.water_block writes it (centre and size to the centimetre). level_z is the POOL_PCT
  percentile of the terrain cells inside the polygon less the water (the cells the pool can
  draw; inside the whole polygon when fewer than POOL_LEVEL_CELLS are left) plus POOL_FILL,
  lowered to ROAD_FREEBOARD under the lowest terrain of any drivable ground road corridor under
  the boxes the rects start as, and rounded down to the millimetre. A pool whose level is under
  the SHOW_PCT percentile of the whole polygon's cells (the cells under other water too: the boxes
  that draw none of the swamp's water are dropped anyway, so this only stops a pool that would
  show no water at all) is dropped; so is one left with no box, or with none that draws water.
  Low spots fill with standing water, higher ground shows through.
* run(cfg, raw_dir=None, out_dir=None) writes bayous.json from raw/osm_features.json,
  out/heights.npy, out/roads.json and out/water.json; with cfg.bayous_enabled off it writes empty
  lists, so the level draws none. The runs profile gives are split where a node would show water on
  a road (a point across its ribbon, every width / ACROSS, inside a drivable ground road's corridor
  with the terrain under the water) or lies inside a water.json polygon: moving the nodes onto
  their channel must not take the water where chains cut it away. Then a ribbon segment (the
  segment between two nodes buffered by the larger of their half-widths, flat caps) covering more
  than RIBBON_OVERLAP m2 of such a corridor splits the run there, dropping the segment's end node:
  a ribbon between two nodes that each clear the road can still cross it. Runs left with fewer
  than 2 nodes are dropped.
* level_content(cfg, data_dir, root, heights, extent) (the level.FEATURE_MODULES hook) ->
  {GROUP: [river_object(run, i) for each run] + [pool_block(pool, j, k) for each rect k of each
  pool j]}.
* river_object: a River with every field of the stock italy muddy river (STOCK_RIVER) and the
  murky overrides; pool_block: bf.water_block with the same look (MURKY) and the rivers' mud ramp
  and no foam (STILL), not WATER_STYLE's.
"""
from __future__ import annotations

import json
import logging
import math
from collections import Counter
from pathlib import Path

import numpy as np
import shapely
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiLineString, Polygon, box

from . import beamng_format as bf
from .buildings import _to_level_rings
from .geo import Extent
from .level import WATER_MARGIN, water_rects
from .roads import load_roads
from .terrain import load_heights, sample_height
from .water import _polygonal, load_water

log = logging.getLogger(__name__)

BAYOUS_FILE = "bayous.json"
GROUP = "bayous"

# channels
FILL = 0.3            # m of water over the channel bottom
MIN_DEPTH = 0.4       # m of bank above the bottom needed to draw water
SEARCH_HALF = 15.0    # m either side of the centreline sampled for the channel
BOTTOM_HALF = 4.0     # m either side of the centreline the bottom is looked for in
BANK_PCT = 80         # percentile of a section's terrain that is its bank
WIDTH = {"river": (6, 30), "canal": (3, 20), "stream": (2, 12), "ditch": (1.5, 5), "drain": (1.5, 6)}  # m
SMOOTH = 5            # nodes in the running median over z
WIDTH_SMOOTH = 5      # nodes in the centred moving average over the widths
ROAD_GAP = 2.0        # m beyond a road corridor the water stops
JOIN_TOL = 0.5        # m; way ends this close are one shared node
MIN_CHAIN = 20.0      # m; shorter pieces are dropped whatever min_length_m says
RIVER_DEPTH = 1.5     # m, each River node's depth, and the pools' water blocks'
OFFSETS = np.arange(-SEARCH_HALF, SEARCH_HALF + 1.0)  # m along the normal, every 1 m
ACROSS = 40           # a ribbon is checked for water on a road every width / ACROSS across each node
RIBBON_OVERLAP = 0.05  # m2 of a road corridor a ribbon segment may cover

# swamp pools
POOL_KINDS = ("swamp", "marsh")   # OSM wetland=* drawn as standing water
POOL_PCT = 25                     # percentile of the terrain inside that sets the level
POOL_FILL = 0.15                  # m over that percentile
ROAD_FREEBOARD = 0.3              # m a pool stays under the terrain of a road it covers
SHOW_PCT = 5                      # a pool under this percentile of its whole swamp's terrain shows no water
OUTLINE_STEP = 1.0                # m between the samples along a road zone's outline
POOL_MIN_SIDE = 16.0              # m; a pool's box is split in halves no smaller than this
POOL_FLOOD_CELLS = 2              # terrain cells outside its swamp a pool's box may flood
POOL_FLOOD_DEPTH = 0.05           # m under the level a cell lies to count as flooded
POOL_LEVEL_TOL = 0.01             # m; pools whose levels differ by more are other water to each other
POOL_SHOW_DEPTH = 0.05            # m over the lower of two waters' levels under which the lower one shows
POOL_LEVEL_CELLS = 10             # cells clear of other water a pool's level is set from, else the whole swamp's
OVER_WATER, OVER_GROUND = "over other water", "over ground outside the swamp"   # the two rules a pool box keeps

# The stock muddy river of levels/italy (BeamNG.drive 0.40.0.0: content/levels/italy.zip,
# main/MissionGroup/Level_objects/water/items.level.json, "Water" in 0.39.4; the River with
# persistentId 6b03db61-baa6-49b2-884a-a468ef124469), every field but class, persistentId,
# __parent, position and nodes, copied as the game wrote them (its wave directions are as good
# as zero). 0.40 changed two fields, baseColor (was [197, 158, 97, 255]) and overallFoamOpacity
# (was 1.70000005); MURKY and RIVER_ONLY set both, so the bayous' Rivers are the same.
# Its textures are shared assets (assets/materials/tileable.zip, gameengine.zip), not italy's own.
STOCK_RIVER = {
    "Foam": [{}, {}],
    "Ripples (texture animation)": [
        {"rippleDir": [0.100000001, 1.10000002], "rippleMagnitude": 1.5, "rippleSpeed": 0.100000001,
         "rippleTexScale": [2, 2]},
        {"rippleDir": [0, 1], "rippleMagnitude": 2, "rippleSpeed": 0.200000003, "rippleTexScale": [5, 5]},
        {"rippleDir": [0.100000001, 0.899999976], "rippleMagnitude": 1, "rippleSpeed": 0.00999999978,
         "rippleTexScale": [20, 20]}],
    "Waves (vertex undulation)": [
        {"waveDir": [1.10564999e-38, 1.10203002e-38], "waveMagnitude": 0.200000003, "waveSpeed": 1},
        {"waveDir": [1.61149323e-43, 1.10203002e-38], "waveMagnitude": 0.200000003, "waveSpeed": 1},
        {"waveDir": [1.10564999e-38, 1.10203002e-38], "waveMagnitude": 0.200000003, "waveSpeed": 1}],
    "baseColor": [102, 87, 63, 255],
    "cubemap": "cubemap_ocean_reflection",
    "depthGradientMax": 15,
    "depthGradientTex": "/assets/materials/tileable/water/depthcolor_ramp/depthcolor_ramp_italy_muddy_b.png",
    "flowMagnitudePhysics": 3,
    "foamMaxDepth": 1,
    "foamTex": "/assets/materials/tileable/water/water_effects/foam2_b.color.dds",
    "fullReflect": False,
    "overallFoamOpacity": 0.5,
    "overallRippleMagnitude": 2,
    "overallWaveMagnitude": 0.150000006,
    "reflectDetailAdjust": -2,
    "reflectMaxRateMs": 10,
    "reflectivity": 0.200000003,
    "rippleTex": "/assets/materials/tileable/water/water_effects/ripple3_nm.normal.dds",
    "underwaterColor": [254, 253, 252, 250],
    "waterFogDensity": 0.5,
    "waterFogDensityOffset": 0.100000001,
    "wetDarkening": 0.300000012,
    "wetDepth": 0.349999994,
}
# the bayous' look, on the rivers and the pools alike: dark, murky, still
MURKY = {"baseColor": [40, 46, 24, 255], "waterFogDensity": 8, "waterFogDensityOffset": 0.2, "reflectivity": 0.2,
         "fullReflect": False, "overallWaveMagnitude": 0.02, "overallRippleMagnitude": 0.5, "clarity": 0.05}
# the rivers' own: a mud depth ramp, no current, no foam. The ribbon is only about 0.3 m deep over
# the terrain, so in game (spike S3) any foam covered all of it and read as grey gravel; the ramp
# and the fog reach their dark end within that depth.
RIVER_ONLY = {"depthGradientTex": "/core/art/water/mud_depthcolor_ramp_b.png", "depthGradientMax": 1.0,
              "flowMagnitudePhysics": 0, "overallFoamOpacity": 0, "foamMaxDepth": 0}
# the pools' share of it: the same ramp and no foam (most pool water is under 0.5 m deep), not the
# River's physics
STILL = {k: RIVER_ONLY[k] for k in ("depthGradientTex", "depthGradientMax", "overallFoamOpacity", "foamMaxDepth")}


# ------------------------------------------------------------------ chains

def _drivable_corridors(roads, extra: float) -> list:
    """The corridors (centreline buffered by width / 2 + extra) of the drivable ground roads: not a
    bridge deck, drivability > 0 (so not a covered passage)."""
    keep = [r for r in roads if not r.is_bridge and len(r.nodes) >= 2 and bf._ai_fields(r)["drivability"] > 0]
    if not keep:
        return []
    return list(shapely.buffer([LineString(r.nodes[:, :2]) for r in keep], [r.width / 2 + extra for r in keep]))


def _merge(group: list[tuple[str, np.ndarray]]) -> list[tuple[str, np.ndarray]]:
    """Join (id, xy) ways end to end where exactly two ends (of two different ways) meet within
    JOIN_TOL. A chain starts at a free end when it has one; ids are joined with "+" in its order."""
    n = len(group)
    ends = np.array([xy[0] for _, xy in group] + [xy[-1] for _, xy in group], float)
    parent = list(range(2 * n))

    def root(e: int) -> int:
        while parent[e] != e:
            parent[e] = parent[parent[e]]
            e = parent[e]
        return e

    for a, b in cKDTree(ends).query_pairs(JOIN_TOL):
        parent[root(a)] = root(b)
    clusters: dict[int, list[int]] = {}
    for e in range(2 * n):
        clusters.setdefault(root(e), []).append(e)
    link = {}   # end index (way i: start i, end i + n) -> the end it joins
    for members in clusters.values():
        if len(members) == 2 and members[0] % n != members[1] % n:
            a, b = members
            link[a], link[b] = b, a
    seen = [False] * n
    out = []
    for i in [i for i in range(n) if i not in link or i + n not in link] + list(range(n)):
        if seen[i]:
            continue
        j, rev = i, i in link and i + n not in link     # walk from the free end
        ids, parts = [], []
        while True:
            seen[j] = True
            xy = group[j][1][::-1] if rev else group[j][1]
            ids.append(group[j][0])
            parts.append(xy if not parts else xy[1:])
            nxt = link.get(j if rev else j + n)
            if nxt is None or seen[nxt % n]:
                break
            j, rev = nxt % n, nxt >= n
        out.append(("+".join(ids), np.vstack(parts)))
    return out


def _lines(geom) -> list[LineString]:
    """The line parts of a geometry, contiguous pieces merged."""
    parts = [g for g in getattr(geom, "geoms", [geom]) if g.geom_type in ("LineString", "MultiLineString")
             and not g.is_empty]
    lines = [p for g in parts for p in getattr(g, "geoms", [g])]
    if not lines:
        return []
    merged = shapely.line_merge(MultiLineString(lines))
    return [g for g in getattr(merged, "geoms", [merged]) if g.geom_type == "LineString" and not g.is_empty]


def chains(waterways, extent: Extent, kinds, min_length_m: float, water_polygons, roads) -> list[dict]:
    """The waterways of the given kinds as chains of level x, y; see the module docstring."""
    groups: dict[tuple, list] = {}
    for w in waterways:
        coords = np.asarray(w.get("coords") or [], float).reshape(-1, 2)
        if w.get("kind") not in kinds or w.get("culvert") or len(coords) < 2:
            continue
        x, y = extent.to_level(coords[:, 0], coords[:, 1])
        groups.setdefault((w["kind"], w.get("name")), []).append((str(w["id"]), np.column_stack([x, y])))
    h = extent.half_size
    square = box(-h, -h, h, h)
    cuts = _drivable_corridors(roads, ROAD_GAP) + list(water_polygons)
    tree = shapely.STRtree(cuts) if cuts else None
    shortest = max(float(min_length_m), MIN_CHAIN)
    out = []
    for (kind, name), group in groups.items():
        for cid, xy in _merge(group):
            line = LineString(xy).intersection(square)
            hits = tree.query(line, predicate="intersects") if tree is not None and not line.is_empty else []
            if len(hits):
                line = line.difference(shapely.union_all([cuts[i] for i in hits]))
            for piece in _lines(line):
                if piece.length >= shortest:
                    out.append({"id": cid, "name": name, "kind": kind, "xy": np.asarray(piece.coords)[:, :2]})
    return out


# ------------------------------------------------------------------ profile

def _resample(xy, spacing: float) -> np.ndarray:
    """Points every spacing m (as near as fits the length) along a polyline, both ends included."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    xy = xy[np.r_[True, np.hypot(*np.diff(xy, axis=0).T) > 1e-9]]
    if len(xy) < 2:
        return xy
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    t = np.linspace(0.0, s[-1], max(1, int(round(s[-1] / spacing))) + 1)
    return np.column_stack([np.interp(t, s, xy[:, 0]), np.interp(t, s, xy[:, 1])])


def _span(h: np.ndarray, b: np.ndarray, surface: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per section (row of h, sampled at OFFSETS): the offsets lx <= rx of the interpolated crossings
    either side of index b between which h is under the surface (the end of the section on a side
    with none), and whether both sides have one. A section dry at b itself has no water there:
    lx = rx = OFFSETS[b], both sides closed."""
    m = h.shape[1]
    idx, k = np.arange(m), np.arange(len(h))
    dry = h >= surface[:, None]
    wet = ~dry[k, b]
    left = np.where(dry & (idx < b[:, None]), idx, -1).max(axis=1)
    right = np.where(dry & (idx > b[:, None]), idx, m).min(axis=1)
    lx, rx = np.full(len(h), OFFSETS[0]), np.full(len(h), OFFSETS[-1])
    has_l = wet & (left >= 0)
    j, kk = left[has_l], k[has_l]
    lx[has_l] = OFFSETS[j] + (h[kk, j] - surface[has_l]) / (h[kk, j] - h[kk, j + 1])
    has_r = wet & (right < m)
    j, kk = right[has_r], k[has_r]
    rx[has_r] = OFFSETS[j] - (h[kk, j] - surface[has_r]) / (h[kk, j] - h[kk, j - 1])
    lx[~wet] = rx[~wet] = OFFSETS[b[~wet]]
    return lx, rx, (has_l & has_r) | ~wet


def _true_runs(mask: np.ndarray) -> list[slice]:
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return [slice(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def _smooth_widths(width: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """The centred moving average over WIDTH_SMOOTH nodes (the nodes there are, at a run's ends) of
    the widest width within WIDTH_SMOOTH nodes of each node, within lo..hi. Averaging the widths as
    they are would narrow the ribbon at a wide node under the span that reaches both its banks, and
    its edge would hang above the bank's slope (on the full map the share of ribbon edges more than
    0.2 m over the ground nearly doubled); the average of the widest never does, and the water's
    edge no longer steps from node to node. The price is a wider ribbon beside every wide node (mean
    node width 6.35 to 7.87 m on the full map, the water about 24 % more in plan, 165 more trees cleared
    beside bayous), which is what keeping each edge on its bank costs."""
    n, half = len(width), WIDTH_SMOOTH // 2
    widest = ndimage.maximum_filter1d(np.asarray(width, float), size=WIDTH_SMOOTH, mode="nearest")
    total = np.r_[0.0, np.cumsum(widest)]
    i = np.arange(n)
    first, last = np.maximum(i - half, 0), np.minimum(i + half + 1, n)
    return np.clip((total[last] - total[first]) / (last - first), lo, hi)


def profile(xy, kind: str, heights: np.ndarray, extent: Extent, spacing_m: float) -> list[np.ndarray]:
    """The runs of water along a chain, (K, 4) arrays of x, y, z, width; see the module docstring."""
    lo, hi = WIDTH[kind]
    pts = _resample(xy, spacing_m)
    if len(pts) < 2:
        return []
    d = np.gradient(pts, axis=0)
    d /= np.hypot(d[:, 0], d[:, 1])[:, None]
    normal = np.column_stack([-d[:, 1], d[:, 0]])
    sx = pts[:, :1] + normal[:, :1] * OFFSETS
    sy = pts[:, 1:2] + normal[:, 1:2] * OFFSETS
    h = np.asarray(sample_height(heights, extent, sx, sy), float).reshape(sx.shape)
    k = np.arange(len(pts))
    b = np.argmin(np.where(np.abs(OFFSETS) <= BOTTOM_HALF, h, np.inf), axis=1)
    bottom = h[k, b]
    channel = np.percentile(h, BANK_PCT, axis=1) - bottom >= MIN_DEPTH
    half = extent.half_size
    runs = []
    for run in _true_runs(channel):
        if run.stop - run.start < 2:
            continue
        z = ndimage.median_filter(bottom[run] + FILL, size=SMOOTH, mode="nearest")
        lx, rx, closed = _span(h[run], b[run], z)
        # a span open on a side (the water runs past the section) has no middle: that node stays put
        mid = ndimage.median_filter(np.where(closed, (lx + rx) / 2, 0.0), size=SMOOTH, mode="nearest")
        width = np.clip(np.where(closed, 2 * np.maximum(rx - mid, mid - lx), rx - lx), lo, hi)
        xy_run = np.clip(pts[run] + normal[run] * mid[:, None], -half, half)
        runs.append(np.column_stack([xy_run, z, _smooth_widths(width, lo, hi)]))
    return runs


def _ribbon_overlaps(run: np.ndarray, corridors: list, road_tree) -> np.ndarray:
    """Per segment of run (between nodes i and i + 1): the m2 its ribbon polygon, the segment buffered
    by the larger of its two nodes' half-widths with flat caps, shares with the road corridors."""
    xy = run[:, :2]
    ribbons = shapely.buffer(shapely.linestrings(np.stack([xy[:-1], xy[1:]], axis=1)),
                             np.maximum(run[:-1, 3], run[1:, 3]) / 2, cap_style="flat")
    area = np.zeros(len(ribbons))
    segment, road = road_tree.query(ribbons, predicate="intersects")
    for i in np.unique(segment):
        area[i] = shapely.intersection(ribbons[i], shapely.union_all([corridors[j] for j in road[segment == i]])).area
    return area


def _keep_clear(runs: list[np.ndarray], corridors: list, lakes: list, heights: np.ndarray,
                extent: Extent) -> list[np.ndarray]:
    """runs (from profile) split where water would be where it must not be: a node with a point across
    its ribbon (every width / ACROSS) inside a drivable road corridor with the terrain under the
    water, or itself inside a water.json polygon (where moving it onto its channel took it); and
    then, between two nodes that pass, a ribbon segment (see _ribbon_overlaps) covering more than
    RIBBON_OVERLAP m2 of a corridor, which drops the segment's end node. Pieces of fewer than 2
    nodes are dropped."""
    road_tree = shapely.STRtree(corridors) if len(corridors) else None
    lake_tree = shapely.STRtree(lakes) if len(lakes) else None
    if road_tree is None and lake_tree is None:
        return runs
    u = np.linspace(-0.5, 0.5, ACROSS + 1)
    out = []
    for r in runs:
        xy, z, w = r[:, :2], r[:, 2], r[:, 3]
        bad = np.zeros(len(r), bool)
        if road_tree is not None:
            d = np.gradient(xy, axis=0)
            d /= np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)[:, None]
            X = xy[:, :1] - d[:, 1:2] * (w[:, None] * u)
            Y = xy[:, 1:2] + d[:, :1] * (w[:, None] * u)
            on_road = np.zeros(X.size, bool)
            on_road[road_tree.query(shapely.points(X.ravel(), Y.ravel()), predicate="intersects")[0]] = True
            wet = np.asarray(sample_height(heights, extent, X, Y), float).reshape(X.shape) < z[:, None]
            bad |= (on_road.reshape(X.shape) & wet).any(axis=1)
        if lake_tree is not None:
            bad[lake_tree.query(shapely.points(xy), predicate="within")[0]] = True
        for piece in _true_runs(~bad):
            if piece.stop - piece.start < 2:
                continue
            kept = r[piece]
            if road_tree is not None:
                over = np.r_[False, _ribbon_overlaps(kept, corridors, road_tree) > RIBBON_OVERLAP]
                out += [kept[part] for part in _true_runs(~over) if part.stop - part.start >= 2]
            else:
                out.append(kept)
    return out


# ------------------------------------------------------------------ swamp pools

def _cells_inside(geom, heights: np.ndarray, extent: Extent) -> np.ndarray:
    """The terrain of the cells whose centres lie inside geom."""
    xs, ys = extent.pixel_centers(heights.shape[0])
    x0, y0, x1, y1 = geom.bounds
    cols, rows = np.flatnonzero((xs >= x0) & (xs <= x1)), np.flatnonzero((ys >= y0) & (ys <= y1))
    if not len(cols) or not len(rows):
        return np.empty(0)
    X, Y = np.meshgrid(xs[cols], ys[rows])
    return np.asarray(heights[np.ix_(rows, cols)], float)[shapely.contains_xy(geom, X, Y)]


def _lowest_under(zone, heights: np.ndarray, extent: Extent) -> float | None:
    """The lowest terrain under zone: at the cells whose centres lie inside it and along its outline
    every OUTLINE_STEP m, which is where the interpolated terrain has its lowest points."""
    zone = _polygonal(zone)
    if zone.is_empty:
        return None
    edge = shapely.get_coordinates(shapely.segmentize(zone.boundary, OUTLINE_STEP))
    vals = np.r_[_cells_inside(zone, heights, extent),
                 np.atleast_1d(sample_height(heights, extent, edge[:, 0], edge[:, 1])) if len(edge) else []]
    return float(vals.min()) if vals.size else None


def _wetland_polygon(p: dict, extent: Extent):
    rings = _to_level_rings(extent, [p.get("outer") or []] + list(p.get("inner") or []))
    if len(rings[0]) < 3:
        return Polygon()
    poly = Polygon(rings[0], [r for r in rings[1:] if len(r) >= 3])
    return poly if poly.is_valid else _polygonal(shapely.make_valid(poly))


def _edges(rects) -> np.ndarray:
    """(cx, cy, sx, sy) rects as a (K, 4) array of x0, y0, x1, y1."""
    r = np.asarray(rects, float).reshape(-1, 4)
    return np.column_stack([r[:, 0] - r[:, 2] / 2, r[:, 1] - r[:, 3] / 2, r[:, 0] + r[:, 2] / 2, r[:, 1] + r[:, 3] / 2])


def _written(rects, level_z: float) -> tuple[np.ndarray, float]:
    """The boxes ((K, 4) x0, y0, x1, y1) and the level of the WaterBlocks bf.water_block writes for the
    (cx, cy, sx, sy) rects at level_z, their centre and size to the centimetre: the pools are checked on
    what reaches the game, so no sub-centimetre sliver gets past."""
    blocks = [bf.water_block(*r, level_z, RIVER_DEPTH, "") for r in rects]
    pos = np.asarray([b["position"] for b in blocks], float).reshape(-1, 3)
    size = np.asarray([b["scale"] for b in blocks], float).reshape(-1, 3)
    edges = np.column_stack([pos[:, :2] - size[:, :2] / 2, pos[:, :2] + size[:, :2] / 2])
    return edges, (float(pos[0, 2]) if len(pos) else level_z)


def _window(edges: np.ndarray, heights: np.ndarray, extent: Extent):
    """The cell centres xs (east) and ys (south) and the terrain h of the cells whose centres lie within a
    metre of the bounds of edges ((K, 4) x0, y0, x1, y1), so every half of those boxes, as written, is inside."""
    xs, ys = extent.pixel_centers(heights.shape[0])
    (x0, y0), (x1, y1) = edges[:, :2].min(axis=0) - 1.0, edges[:, 2:].max(axis=0) + 1.0
    cols, rows = np.flatnonzero((xs >= x0) & (xs <= x1)), np.flatnonzero((ys >= y0) & (ys <= y1))
    return xs[cols], ys[rows], np.asarray(heights[np.ix_(rows, cols)], float)


def _cells_of(e, xs: np.ndarray, ys: np.ndarray) -> tuple[slice, slice]:
    """The rows and columns of the cells (centres xs, ys) inside the box e (x0, y0, x1, y1), its edges included."""
    c0, c1 = np.searchsorted(xs, e[0], "left"), np.searchsorted(xs, e[2], "right")
    r0, r1 = np.searchsorted(-ys, -e[3], "left"), np.searchsorted(-ys, -e[1], "right")   # rows run south
    return slice(int(r0), int(r1)), slice(int(c0), int(c1))


def _box_counter(mask: np.ndarray, xs: np.ndarray, ys: np.ndarray):
    """count(e): how many cells of mask (on the cell centres xs, ys) that are True lie inside the box e
    (x0, y0, x1, y1), from a summed-area table."""
    total = np.zeros((len(ys) + 1, len(xs) + 1), np.int64)
    total[1:, 1:] = mask.cumsum(axis=0).cumsum(axis=1)

    def count(e) -> int:
        rows, cols = _cells_of(e, xs, ys)
        return int(total[rows.stop, cols.stop] - total[rows.start, cols.stop] - total[rows.stop, cols.start]
                   + total[rows.start, cols.start])
    return count


def _shown(level_z: float, others, xs: np.ndarray, ys: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Which cells (centres xs, ys, terrain h) lie inside a block of another water (others: [(its written
    boxes, its level)]) with their terrain under the lower of the two levels (level_z and that water's) +
    POOL_SHOW_DEPTH: there the lower water shows, and a pool box would stand one surface over the other.
    Where the terrain is above it, the lower water is under the ground and the other may cover it."""
    top = np.full(h.shape, -np.inf)
    for edges, z in others:
        cut = min(level_z, z) + POOL_SHOW_DEPTH
        for e in edges:
            rows, cols = _cells_of(e, xs, ys)
            top[rows, cols] = np.maximum(top[rows, cols], cut)
    return h < top


def _less_water(poly, lakes: list, lake_tree):
    """poly less the water polygons near it grown by WATER_MARGIN (each clipped to a box round poly
    first, so the Mississippi is not grown whole for every swamp beside it)."""
    if lake_tree is None:
        return poly
    x0, y0, x1, y1 = poly.bounds
    near = lake_tree.query(box(x0 - WATER_MARGIN, y0 - WATER_MARGIN, x1 + WATER_MARGIN, y1 + WATER_MARGIN))
    if not len(near):
        return poly
    clip = box(x0 - 2 * WATER_MARGIN, y0 - 2 * WATER_MARGIN, x1 + 2 * WATER_MARGIN, y1 + 2 * WATER_MARGIN)
    grown = shapely.buffer(shapely.intersection([lakes[i] for i in near], clip), WATER_MARGIN)
    return _polygonal(poly.difference(shapely.union_all(grown)))


def _flood_counter(poly, level_z: float, xs: np.ndarray, ys: np.ndarray, h: np.ndarray):
    """count(e): the terrain cells (centres xs, ys, terrain h) in the box e (x0, y0, x1, y1) more than
    POOL_FLOOD_DEPTH under level_z and outside poly grown by WATER_MARGIN (square at its corners, as the
    boxes grow it, and by the millimetre the rects are written to)."""
    grown = poly.buffer(WATER_MARGIN + 0.001, join_style="mitre", mitre_limit=math.sqrt(2))
    shapely.prepare(grown)
    X, Y = np.meshgrid(xs, ys)
    return _box_counter((h < level_z - POOL_FLOOD_DEPTH) & ~shapely.intersects_xy(grown, X, Y), xs, ys)


def _pool_rects(base, poly, level_z: float, start, others, heights: np.ndarray, extent: Extent,
                tally: Counter) -> tuple[list[tuple], int]:
    """start (the water_rects of base, the swamp poly less the water) with every rect that shows more
    than POOL_FLOOD_CELLS cells of other water (_shown; others: [(written boxes, level)]) or floods more
    than that many cells outside poly split in halves across its long side, again and again down to
    halves of POOL_MIN_SIDE; a piece that still fails, or that no longer meets base, is dropped. After the
    search a piece that draws none of the swamp's water is dropped too: one that covers no terrain cell
    (centre) inside poly under level_z where no other water shows. Each piece is checked as written: to the
    millimetre in bayous.json, then bf.water_block's centimetre. Returns the pieces kept (to the
    millimetre) and how many were dropped drawing no water; tally counts the splits and the other drops."""
    xs, ys, h = _window(_written(start, level_z)[0], heights, extent)
    over_other = _shown(level_z, others, xs, ys, h)
    shown = _box_counter(over_other, xs, ys)
    flooded = _flood_counter(poly, level_z, xs, ys, h)
    # the cells that are the swamp's water: not the other water the rules let a piece show, nor the
    # WATER_MARGIN round the swamp or the cells it floods past that (outside poly too)
    shapely.prepare(poly)
    X, Y = np.meshgrid(xs, ys)
    wet = _box_counter((h < level_z) & ~over_other & shapely.contains_xy(poly, X, Y), xs, ys)
    shapely.prepare(base)
    out, todo = [], [tuple(r) for r in reversed(start)]
    while todo:
        cx, cy, sx, sy = todo.pop()
        stored = tuple(round(float(v), 3) + 0.0 for v in (cx, cy, sx, sy))     # as bayous.json holds it
        e = _written([stored], level_z)[0][0]
        piece = box(*e)
        if not base.intersects(piece) or base.touches(piece):
            tally["dropped", "off the swamp"] += 1
            continue
        rule = OVER_WATER if shown(e) > POOL_FLOOD_CELLS else OVER_GROUND if flooded(e) > POOL_FLOOD_CELLS else None
        if rule is None:
            out.append((stored, e))
        elif max(sx, sy) / 2 < POOL_MIN_SIDE:
            tally["dropped", rule] += 1
        else:
            tally["split", rule] += 1
            todo += ([(cx + sx / 4, cy, sx / 2, sy), (cx - sx / 4, cy, sx / 2, sy)] if sx >= sy else
                     [(cx, cy + sy / 4, sx, sy / 2), (cx, cy - sy / 4, sx, sy / 2)])
    drawing = [r for r, e in out if wet(e)]
    return drawing, len(out) - len(drawing)


def pools(wetlands, heights: np.ndarray, extent: Extent, roads, water=()) -> list[dict]:
    """The swamp pools of the wetlands, kept off the water bodies (water.json's); see the module docstring."""
    h = extent.half_size
    square = box(-h, -h, h, h)
    corridors = _drivable_corridors(roads, 0.0)
    tree = shapely.STRtree(corridors) if corridors else None
    lakes = [b.polygon for b in water]
    lake_tree = shapely.STRtree(lakes) if lakes else None
    waters = [_written(water_rects(b.polygon), b.level_z) for b in water]
    found = []
    for w in wetlands:
        if w.get("wetland") not in POOL_KINDS:
            continue
        polygons = w.get("polygons") or []
        found += [(str(w["id"]) if len(polygons) == 1 else f"{w['id']}/{k}",
                   _polygonal(_wetland_polygon(p, extent).intersection(square))) for k, p in enumerate(polygons)]
    out, placed, tally = [], [], Counter()
    lowered = whole = dry = gone = drawless = started = drawless_boxes = 0
    for pid, poly in sorted(found, key=lambda f: f[0]):
        terrain = _cells_inside(poly, heights, extent) if not poly.is_empty else np.empty(0)
        if not terrain.size:
            continue
        base = _less_water(poly, lakes, lake_tree)
        start = water_rects(base) if not base.is_empty else []
        if not start:
            gone += 1
            continue
        # the cells the pool can draw (clear of the other water) set its level; too few, and the whole swamp's do
        clear = _cells_inside(base, heights, extent)
        on_whole = clear.size < POOL_LEVEL_CELLS
        cells = terrain if on_whole else clear
        level_z, under_road = float(np.percentile(cells, POOL_PCT)) + POOL_FILL, False
        if tree is not None:
            area = shapely.union_all([box(*e) for e in _edges(start)])
            hits = tree.query(area, predicate="intersects")
            low = (_lowest_under(shapely.union_all([corridors[i] for i in hits]).intersection(area), heights,
                                 extent) if len(hits) else None)
            if low is not None and low - ROAD_FREEBOARD < level_z:
                level_z, under_road = low - ROAD_FREEBOARD, True
        level_z = math.floor(level_z * 1000) / 1000
        # SHOW_PCT over the whole swamp, a lake bed in it too: the boxes that draw none of the swamp's water are
        # dropped after the search, so this only has to stop a pool that would show no water at all
        if level_z < np.percentile(terrain, SHOW_PCT):
            dry += 1
            continue
        others = waters + [(e, z) for e, z in placed if abs(z - level_z) > POOL_LEVEL_TOL]
        started += len(start)
        rects, undrawn = _pool_rects(base, poly, level_z, start, others, heights, extent, tally)
        drawless_boxes += undrawn
        if not rects:
            drawless += undrawn > 0       # left with boxes, none of them drawing water
            gone += undrawn == 0
            continue
        lowered += under_road
        whole += on_whole
        placed.append(_written(rects, level_z))
        out.append({"id": pid, "level_z": level_z, "rects": [list(r) for r in rects]})
    if out or dry or gone or drawless:
        log.info("%d swamp pools (%d lowered under a road, %d levelled on their whole swamp: fewer than %d cells "
                 "clear of other water), %d dropped with no water showing, %d with no box left, %d with no box "
                 "that draws water", len(out), lowered, whole, POOL_LEVEL_CELLS, dry, gone, drawless)
        log.info("pool boxes: %d to start; %d split and %d dropped %s, %d split and %d dropped %s, %d dropped off "
                 "the swamp, %d dropped drawing no water; %d in all", started, tally["split", OVER_WATER],
                 tally["dropped", OVER_WATER], OVER_WATER, tally["split", OVER_GROUND], tally["dropped", OVER_GROUND],
                 OVER_GROUND, tally["dropped", "off the swamp"], drawless_boxes, sum(len(p["rects"]) for p in out))
    return out


# ------------------------------------------------------------------ the file and the stage

def save_bayous(data: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"runs": list(data["runs"]), "pools": list(data["pools"])}), encoding="utf-8")
    return path


def load_bayous(path) -> dict:
    """bayous.json as {"runs": [...], "pools": [...]}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {"runs": list(data.get("runs") or []), "pools": list(data.get("pools") or [])}


def _optional(path: Path, loader, what: str) -> list:
    if path.exists():
        return loader(path)
    log.warning("no %s: bayous are not %s", path, what)
    return []


def _length(nodes) -> float:
    xy = np.asarray(nodes, float).reshape(-1, 4)[:, :2]
    return float(np.hypot(*np.diff(xy, axis=0).T).sum())


def run(cfg, raw_dir=None, out_dir=None) -> None:
    raw_dir, out_dir = Path(raw_dir or cfg.raw_dir), Path(out_dir or cfg.out_dir)
    path = out_dir / BAYOUS_FILE
    if not cfg.bayous_enabled:
        save_bayous({"runs": [], "pools": []}, path)
        log.info("bayous are off (bayous.enabled in the config): %s lists none", path)
        return
    features = raw_dir / "osm_features.json"
    if not features.exists():
        raise FileNotFoundError(f"no {features}; run the features stage first")
    osm = json.loads(features.read_text(encoding="utf-8"))
    extent = Extent.from_config(cfg)
    heights = load_heights(out_dir=out_dir)
    roads = _optional(out_dir / "roads.json", load_roads, "cut at the roads or kept off them")
    water = _optional(out_dir / "water.json", load_water, "cut where they enter a lake or the river")
    found = chains(osm.get("waterways") or [], extent, tuple(cfg.bayou_kinds), cfg.bayou_min_length_m,
                   [b.polygon for b in water], roads)
    corridors, lakes = _drivable_corridors(roads, 0.0), [b.polygon for b in water]
    runs = [{"id": c["id"], "name": c["name"], "kind": c["kind"],
             "nodes": [[round(float(v), 3) + 0.0 for v in node] for node in r]}
            for c in found
            for r in _keep_clear(profile(c["xy"], c["kind"], heights, extent, cfg.bayou_node_spacing_m),
                                 corridors, lakes, heights, extent)]
    swamp = pools(osm.get("wetlands") or [], heights, extent, roads, water)
    save_bayous({"runs": runs, "pools": swamp}, path)
    log.info("%d chains from %d waterways; %d runs, %d nodes, %.1f km of water (%s); %d pools in %d water blocks",
             len(found), len(osm.get("waterways") or []), len(runs), sum(len(r["nodes"]) for r in runs),
             sum(_length(r["nodes"]) for r in runs) / 1000,
             ", ".join(f"{k} {v}" for k, v in Counter(r["kind"] for r in runs).most_common()) or "none",
             len(swamp), sum(len(p["rects"]) for p in swamp))
    named = Counter()
    for r in runs:
        named[r["name"] or "(unnamed)"] += _length(r["nodes"]) / 1000
    for name, km in named.most_common(12):
        log.info("  %s: %.2f km", name, km)


# ------------------------------------------------------------------ level objects

def _r(v, nd=3) -> float:
    return round(float(v), nd) + 0.0


def river_object(b: dict, i: int) -> dict:
    """The River for run b (the i-th of bayous.json): STOCK_RIVER made murky and still."""
    nodes = [[_r(x), _r(y), _r(z), _r(w, 2), RIVER_DEPTH, 0, 0, 1] for x, y, z, w in b["nodes"]]
    return {"class": "River", "persistentId": bf.persistent_id("bayou", b["id"], i), "position": nodes[0][:3],
            **STOCK_RIVER, **MURKY, **RIVER_ONLY, "nodes": nodes}


def pool_block(pool: dict, j: int, k: int) -> dict:
    """The WaterBlock of rect k of pool j, with the bayous' still-water look."""
    cx, cy, sx, sy = pool["rects"][k]
    return {**bf.water_block(cx, cy, sx, sy, pool["level_z"], RIVER_DEPTH, f"pool/{j}/{k}"), **MURKY, **STILL}


def level_content(cfg, data_dir, root, heights, extent) -> dict[str, list[dict]]:
    data = load_bayous(Path(data_dir) / BAYOUS_FILE)
    rivers = [river_object(b, i) for i, b in enumerate(data["runs"])]
    blocks = [pool_block(p, j, k) for j, p in enumerate(data["pools"]) for k in range(len(p["rects"]))]
    if not rivers and not blocks:
        return {}
    log.info("bayous: %d Rivers (%d nodes), %d pool water blocks", len(rivers),
             sum(len(r["nodes"]) for r in rivers), len(blocks))
    return {GROUP: rivers + blocks}
