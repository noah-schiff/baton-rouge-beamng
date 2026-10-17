"""Stage 4: OSM water polygons -> water bodies at their real surface height; carve their beds.

The lidar DEMs are hydro-flattened, so the median raw (uncarved) height inside
a water polygon is its surface level. Re-runs read the raw heights, so the
level stays the same after the terrain has been carved.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from affine import Affine
from rasterio import features
from scipy import ndimage
from shapely.geometry import LineString, MultiPolygon, Polygon, box, mapping, shape
from shapely.ops import polygonize, unary_union

from .geo import Extent
from .terrain import load_heights, sample_height

log = logging.getLogger(__name__)

OUT_DIR = Path("data/out")
RAW_DIR = Path("data/raw")
MIN_AREA = 2000.0      # m^2; smaller ponds are dropped
RIVER_DEPTH = 15.0     # m, for the Mississippi
RIVER_MIN_AREA = 1e6   # m^2; an unnamed water=river area this big is the Mississippi
DEFAULT_DEPTH = 2.0    # m, lakes and ponds
BANK_WIDTH = 60.0      # m over which the bed slopes down from the shore


@dataclass
class WaterBody:
    name: str | None
    polygon: Polygon | MultiPolygon  # level coordinates
    level_z: float
    depth: float


def _is_water(tags: dict) -> bool:
    return tags.get("natural") == "water" or tags.get("waterway") == "riverbank"


def _to_level(extent: Extent, geometry) -> np.ndarray:
    pts = [(g["lon"], g["lat"]) for g in geometry if g]
    if not pts:
        return np.empty((0, 2))
    lon, lat = np.asarray(pts).T
    return np.column_stack(extent.to_level(lon, lat))


def _polygonal(geom):
    """Only the polygon parts of a geometry (clipping can leave lines or points)."""
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    parts = [g for g in getattr(geom, "geoms", []) if g.geom_type in ("Polygon", "MultiPolygon")]
    return unary_union(parts) if parts else Polygon()


def _relation_polygon(extent: Extent, rel: dict):
    outer, inner = [], []
    for m in rel.get("members", []):
        if m.get("type") != "way":
            continue
        xy = _to_level(extent, m.get("geometry") or [])
        if len(xy) >= 2:
            (inner if m.get("role") == "inner" else outer).append(LineString(xy))
    if not outer:
        return Polygon()
    poly = unary_union(list(polygonize(unary_union(outer))))
    if inner:
        poly = poly.difference(unary_union(list(polygonize(unary_union(inner)))))
    return poly


def _mask(geom, extent: Extent, n: int) -> np.ndarray:
    return features.rasterize([(mapping(geom), 1)], out_shape=(n, n),
                              transform=extent.level_transform(n), fill=0,
                              dtype="uint8").astype(bool)


def _window_mask(geom, extent: Extent, n: int, pad: int = 2):
    """Rasterize geom only over its own bounding window (plus pad cells).

    Returns (mask, (r0, r1, c0, c1)); mask covers grid[r0:r1, c0:c1].
    """
    px = extent.pixel_size(n)
    h = extent.half_size
    minx, miny, maxx, maxy = geom.bounds
    c0 = max(0, int(math.floor((minx + h) / px)) - pad)
    c1 = min(n, int(math.ceil((maxx + h) / px)) + pad)
    r0 = max(0, int(math.floor((h - maxy) / px)) - pad)
    r1 = min(n, int(math.ceil((h - miny) / px)) + pad)
    if c1 <= c0 or r1 <= r0:
        return np.zeros((0, 0), bool), (0, 0, 0, 0)
    transform = extent.level_transform(n) @ Affine.translation(c0, r0)
    mask = features.rasterize([(mapping(geom), 1)], out_shape=(r1 - r0, c1 - c0),
                              transform=transform, fill=0, dtype="uint8").astype(bool)
    return mask, (r0, r1, c0, c1)


def _median_inside(heights: np.ndarray, extent: Extent, geom) -> float:
    mask, (r0, r1, c0, c1) = _window_mask(geom, extent, heights.shape[0])
    vals = heights[r0:r1, c0:c1][mask]
    if vals.size == 0:
        p = geom.representative_point()
        return float(sample_height(heights, extent, p.x, p.y))
    return float(np.median(vals))


def _is_river(tags: dict, name: str | None, area: float) -> bool:
    if name and "mississippi" in name.lower():
        return True
    # OSM tags the Mississippi's water area water=river without a name
    riverish = tags.get("water") == "river" or tags.get("waterway") == "riverbank"
    return riverish and area >= RIVER_MIN_AREA


def build_water(osm_json: dict, extent: Extent, heights: np.ndarray) -> list[WaterBody]:
    """Water bodies from OSM ways and multipolygon relations, clipped to the map square.

    heights should be the raw (uncarved) grid.
    """
    elements = osm_json.get("elements", [])
    in_relations = {m.get("ref") for el in elements
                    if el.get("type") == "relation" and _is_water(el.get("tags") or {})
                    for m in el.get("members", []) if m.get("type") == "way"}
    h = extent.half_size
    square = box(-h, -h, h, h)
    bodies: list[WaterBody] = []
    for el in elements:
        tags = el.get("tags") or {}
        if not _is_water(tags):
            continue
        if el.get("type") == "way":
            if el.get("id") in in_relations:
                continue
            xy = _to_level(extent, el.get("geometry") or [])
            if len(xy) < 4 or math.dist(xy[0], xy[-1]) > 0.01:
                continue  # not a closed ring
            geom = Polygon(xy)
        elif el.get("type") == "relation":
            geom = _relation_polygon(extent, el)
        else:
            continue
        if not geom.is_valid:
            geom = geom.buffer(0)
        geom = _polygonal(geom.intersection(square))
        if geom.is_empty or geom.area < MIN_AREA:
            continue
        name = tags.get("name")
        depth = RIVER_DEPTH if _is_river(tags, name, geom.area) else DEFAULT_DEPTH
        bodies.append(WaterBody(name, geom, _median_inside(heights, extent, geom), depth))
    return bodies


def carve(heights: np.ndarray, extent: Extent, bodies: list[WaterBody]) -> np.ndarray:
    """Copy of heights with each body's bed cut in: level - depth * min(1, dist to shore / 60 m).

    Works in each body's own window; the pad keeps the shore distances identical
    to a whole-grid computation (see _carve_reference).
    """
    out = np.array(heights, dtype=np.float32, copy=True)
    n = out.shape[0]
    px = extent.pixel_size(n)
    for body in bodies:
        mask, (r0, r1, c0, c1) = _window_mask(body.polygon, extent, n)
        if not mask.any():
            continue
        dist = np.maximum(ndimage.distance_transform_edt(mask) - 0.5, 0.0) * px
        # a window cut by the map edge must not treat that edge as shore: EDT only
        # measures to zero cells inside the window, and beyond the map there are none
        bed = (body.level_z - body.depth * np.minimum(1.0, dist / BANK_WIDTH)).astype(np.float32)
        win = out[r0:r1, c0:c1]
        win[mask] = np.minimum(win[mask], bed[mask])
    return out


def _carve_reference(heights: np.ndarray, extent: Extent, bodies: list[WaterBody]) -> np.ndarray:
    """Whole-grid version of carve(), kept to test the windowed one against."""
    out = np.array(heights, dtype=np.float32, copy=True)
    n = out.shape[0]
    px = extent.pixel_size(n)
    for body in bodies:
        mask = _mask(body.polygon, extent, n)
        if not mask.any():
            continue
        dist = np.maximum(ndimage.distance_transform_edt(mask) - 0.5, 0.0) * px
        bed = body.level_z - body.depth * np.minimum(1.0, dist / BANK_WIDTH)
        out[mask] = np.minimum(out[mask], bed[mask].astype(np.float32))
    return out


def save_water(bodies: list[WaterBody], path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([{"name": b.name, "level_z": b.level_z, "depth": b.depth,
                                 "geometry": mapping(b.polygon)} for b in bodies]),
                    encoding="utf-8")
    return path


def load_water(path=OUT_DIR / "water.json") -> list[WaterBody]:
    return [WaterBody(d["name"], shape(d["geometry"]), float(d["level_z"]), float(d["depth"]))
            for d in json.loads(Path(path).read_text(encoding="utf-8"))]


def run(cfg, raw_dir=None, out_dir=None) -> None:
    extent = Extent.from_config(cfg)
    raw_dir, out_dir = Path(raw_dir or cfg.raw_dir), Path(out_dir or cfg.out_dir)
    osm = json.loads((raw_dir / "osm.json").read_text(encoding="utf-8"))
    bodies = build_water(osm, extent, load_heights(raw=True, out_dir=out_dir))
    save_water(bodies, out_dir / "water.json")
    log.info("%d water bodies", len(bodies))
    for b in sorted(bodies, key=lambda b: -b.polygon.area)[:15]:
        log.info("  %s: %.3f km2, level %.2f m, depth %.0f m", b.name or "(unnamed)",
                 b.polygon.area / 1e6, b.level_z, b.depth)
