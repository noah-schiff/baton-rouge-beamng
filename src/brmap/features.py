"""Inputs for buildings and trees: OpenStreetMap footprints and nodes, Microsoft
footprints, the lidar surface model and NAIP near-infrared.

Everything goes into raw_dir (cfg.raw_dir by default) and is cached: a step whose output
exists is skipped. fetch.py keeps its behaviour and raw/osm.json is left alone; its
download() and remote-read helpers are reused here.

* raw/osm_features.json, from the cached Geofabrik pbf (raw/<the file named by
  cfg.sources["osm_extract"]>), by extract_osm_features(pbf_path, extent, buffer_m) -> dict:

      {"buildings": [{"id": "w123", "tags": {...},
                      "polygons": [{"outer": [[lon, lat], ...],
                                    "inner": [[[lon, lat], ...], ...]}]}],
       "trees": [{"id": "n1", "lon": .., "lat": .., "tags": {...}}],
       "nodes": [{"id": "n2", "lon": .., "lat": .., "tags": {...}}],
       "waterways": [{"id": "w3", "kind": "stream", "name": str | None, "culvert": bool,
                      "coords": [[lon, lat], ...]}],
       "wetlands": [{"id": "w4", "wetland": str | None, "polygons": [...]}],
       "areas": [{"id": "w5", "tags": {...}, "polygons": [...]}]}

  - buildings: OSM ways ("w<id>") and multipolygon relations ("r<id>", several outers
    and holes) with building=*, whose centroid lies in the box plus cfg.buffer_m. Rings
    are closed (first point = last). pyosmium SimpleHandler, apply_file(...,
    locations=True), and its area handler for the multipolygons. building:part is not a
    building, and a way that is also the outer of a building relation is only kept as
    the relation. Filtering by building value is left to the buildings stage.
  - trees: natural=tree nodes ("n<id>") in the box plus the buffer.
  - nodes: highway in {traffic_signals, stop, crossing} with all their tags (crossing,
    crossing:markings), for stop bars and crosswalks.
  - waterways: open ways with waterway in WATERWAY_KINDS (river, stream, canal, ditch,
    drain), whole, with at least one node in the box plus the buffer. A waterway area
    (waterway=riverbank) is not one. culvert is True for tunnel=culvert|yes or culvert=yes.
  - wetlands: natural=wetland areas (polygons as for buildings), wetland=* is the kind
    (swamp, marsh, ...). Kept when a vertex lies in the box plus the buffer.
  - areas: the landmark areas, with all their tags: building:part=*, leisure in
    {stadium, pitch} and man_made=tower, kept when a vertex lies in the box plus the buffer.
    One that is also a building=* is in buildings as well.
* raw/ms_buildings.json, by fetch_ms_buildings(cfg, raw_dir) -> Path, only when
  cfg.buildings_microsoft: [{"ring": [[lon, lat], ...], "height": float}] for the
  Microsoft GlobalMLBuildingFootprints intersecting the box plus the buffer. Rings are
  closed; height -1 means unknown.
  - cfg.sources["ms_buildings_links"] (dataset-links.csv: Location, QuadKey, Url, ...)
    lists a .csv.gz per location and zoom-9 quadkey; the United States rows whose
    quadkey covers the box are taken (the quadkeys come from the box corners).
  - Each .csv.gz line is a GeoJSON Feature with properties.height. Downloads are cached
    under raw/ms/.
* raw/dsm/<item id>.tif, by fetch_dsm(cfg, raw_dir) -> list[Path]: the items of STAC
  collection cfg.sources["dsm_collection"] (server cfg.sources["stac"]) over the box,
  each a windowed read clipped to the box plus the buffer.
* raw/naip_rn_2m.tif, by fetch_naip_rn(cfg, raw_dir) -> Path: the latest year of STAC
  collection cfg.sources["naip_collection"] covering the box. Band 1 (red) and band 4
  (NIR) of each item through a WarpedVRT onto the 2 m analysis grid
  (canopy.analysis_grid) in EPSG:26915 with average resampling, mosaicked (the first
  valid value wins): 2 bands, uint8, 0 = no data. The images are 0.3 m, so each one is
  read from its overview nearest 2 m instead of at full resolution.

raw/features_box.json records the box these files were cut to; when the box grows
beyond it they are made again (see check_box).

run(cfg, raw_dir=None, out_dir=None) does all four steps and logs the counts. A cached
osm_features.json without all of OSM_KEYS (one from before the waterways, wetlands and
areas) is made again from the pbf.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import logging
import math
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds

from .fetch import GDAL_REMOTE, WORKERS, FetchError, _cached, _clip_cached, _retrying, clip_raster, download
from .geo import LEVEL_CRS, Extent

log = logging.getLogger(__name__)

OSM_FEATURES = "osm_features.json"
MS_BUILDINGS = "ms_buildings.json"
MS_DIR = "ms"                        # the dataset-links.csv and the quadkey .csv.gz downloads
DSM_DIR = "dsm"
NAIP_RN_FILE = "naip_rn_2m.tif"
BOX_FILE = "features_box.json"
QUADKEY_ZOOM = 9                     # the Microsoft footprints are split by zoom-9 quadkey
NODE_HIGHWAYS = ("traffic_signals", "stop", "crossing")
WATERWAY_KINDS = ("river", "stream", "canal", "ditch", "drain")  # waterway=* values that are waterways
CULVERT_TUNNELS = ("culvert", "yes")                             # tunnel=* values that make a culvert
AREA_LEISURE = ("stadium", "pitch")                              # leisure=* values that are landmark areas
OSM_KEYS = ("buildings", "trees", "nodes", "waterways", "wetlands", "areas")  # the keys of osm_features.json
US_LOCATIONS = {"unitedstates", "unitedstatesofamerica", "usa", "us"}  # lower case, letters only
NAIP_COVER = 0.999                   # share of the box a NAIP year must cover to count as whole
MERCATOR_LAT = 85.05112878           # the tile system's latitude limit


def _write_json(dest: Path, obj) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    part.write_text(json.dumps(obj), encoding="utf-8")
    part.replace(dest)
    return dest


def check_box(raw_dir, bbox_utm) -> None:
    """Delete this stage's box-cut files when bbox_utm isn't inside the box they were cut to.

    Like fetch.check_cache_box, but it only touches the files written here; the pbf and
    the Microsoft quadkey files cover far more than any box and are kept.
    """
    raw_dir = Path(raw_dir)
    marker = raw_dir / BOX_FILE
    if marker.exists():
        old = json.loads(marker.read_text(encoding="utf-8"))
        if (bbox_utm[0] >= old[0] - 1e-6 and bbox_utm[1] >= old[1] - 1e-6
                and bbox_utm[2] <= old[2] + 1e-6 and bbox_utm[3] <= old[3] + 1e-6):
            return
        log.info("box grew beyond the cached feature data; making it again")
        for name in (OSM_FEATURES, MS_BUILDINGS, NAIP_RN_FILE):
            (raw_dir / name).unlink(missing_ok=True)
        shutil.rmtree(raw_dir / DSM_DIR, ignore_errors=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps([float(v) for v in bbox_utm]), encoding="utf-8")


# ---------------------------------------------------------------- OpenStreetMap


def _ring(coords) -> list[list[float]]:
    return [[round(x, 7), round(y, 7)] for x, y in coords]


def _polygons(geom) -> list[dict]:
    """The osm_features.json polygon list of a shapely (Multi)Polygon in lon/lat."""
    parts = getattr(geom, "geoms", [geom])
    return [{"outer": _ring(p.exterior.coords), "inner": [_ring(r.coords) for r in p.interiors]}
            for p in parts if not p.is_empty]


def _in_box(extent: Extent, lim: float, lonlat) -> np.ndarray:
    """Which of these lon/lat points (a list of pairs or an n x 2 array) lie within lim m of
    the box centre on both axes."""
    if len(lonlat) == 0:
        return np.zeros(0, bool)
    lon, lat = np.asarray(lonlat, dtype=float).T
    x, y = extent.to_level(lon, lat)
    return (np.abs(x) <= lim) & (np.abs(y) <= lim)


def extract_osm_features(pbf_path, extent: Extent, buffer_m: float) -> dict:
    """Buildings, trees, signal/stop/crossing nodes, waterways, wetlands and landmark areas
    from an OSM extract (see the module doc)."""
    import osmium
    import shapely

    w, s, e, n = extent.bbox_lonlat(buffer_m)  # a quick lon/lat test; the exact one follows

    def near(lon: float, lat: float) -> bool:
        return w <= lon <= e and s <= lat <= n

    def touches(geom) -> bool:
        x, y = shapely.get_coordinates(geom).T
        return bool(((w <= x) & (x <= e) & (s <= y) & (y <= n)).any())

    class Features(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.wkb = osmium.geom.WKBFactory()
            self.buildings: list[tuple[str, dict, object]] = []
            self.wetlands: list[tuple[str, dict, object]] = []
            self.areas: list[tuple[str, dict, object]] = []
            self.waterways: list[dict] = []
            self.points: dict[str, list[dict]] = {"trees": [], "nodes": []}
            self.outers: dict[int, set[int]] = {}  # building relation id -> its outer ways
            self.broken = 0

        def node(self, nd):
            tags = nd.tags
            if tags.get("natural") == "tree":
                kind = "trees"
            elif tags.get("highway") in NODE_HIGHWAYS:
                kind = "nodes"
            else:
                return
            loc = nd.location
            if loc.valid() and near(loc.lon, loc.lat):
                self.points[kind].append({"id": f"n{nd.id}", "lon": round(loc.lon, 7),
                                          "lat": round(loc.lat, 7), "tags": {t.k: t.v for t in tags}})

        def relation(self, r):
            if "building" in r.tags:
                self.outers[r.id] = {m.ref for m in r.members if m.type == "w" and m.role in ("outer", "")}

        def way(self, wy):
            tags = wy.tags
            kind = tags.get("waterway")  # waterway=riverbank (an area) is no kind of waterway
            if kind not in WATERWAY_KINDS:
                return
            pts = [(nd.lon, nd.lat) for nd in wy.nodes if nd.location.valid()]
            if len(pts) >= 2 and any(near(lon, lat) for lon, lat in pts):
                self.waterways.append({
                    "id": f"w{wy.id}", "kind": kind, "name": tags.get("name"),
                    "culvert": tags.get("tunnel") in CULVERT_TUNNELS or tags.get("culvert") == "yes",
                    "coords": [[round(lon, 7), round(lat, 7)] for lon, lat in pts]})

        def area(self, a):
            tags = a.tags
            # building:part is another key: a part alone is no building (but a landmark area)
            building = "building" in tags
            wetland = tags.get("natural") == "wetland"
            landmark = ("building:part" in tags or tags.get("leisure") in AREA_LEISURE
                        or tags.get("man_made") == "tower")
            if not (building or wetland or landmark):
                return
            try:
                geom = shapely.from_wkb(self.wkb.create_multipolygon(a))
            except RuntimeError:  # the assembler couldn't close its rings
                self.broken += 1
                return
            if geom.is_empty:
                return
            oid = f"{'w' if a.from_way() else 'r'}{a.orig_id()}"
            all_tags = {t.k: t.v for t in tags}
            if building:
                c = geom.centroid
                if near(c.x, c.y):
                    self.buildings.append((oid, all_tags, geom))
            if (wetland or landmark) and touches(geom):
                if wetland:
                    self.wetlands.append((oid, all_tags, geom))
                if landmark:
                    self.areas.append((oid, all_tags, geom))

    h = Features()
    # Only objects with these keys reach the callbacks (and only such relations are
    # assembled); every node location is still kept for the ways.
    h.apply_file(str(pbf_path), locations=True,
                 filters=[osmium.filter.KeyFilter("building", "natural", "highway", "waterway", "leisure",
                                                  "building:part", "man_made")])

    relations = {int(oid[1:]) for oid, _, _ in h.buildings if oid[0] == "r"}
    doubled = {ref for rid in relations for ref in h.outers.get(rid, ())}
    found = [b for b in h.buildings if not (b[0][0] == "w" and int(b[0][1:]) in doubled)]
    lim = extent.half_size + buffer_m
    keep = _in_box(extent, lim, [(g.centroid.x, g.centroid.y) for _, _, g in found])
    out = {"buildings": [{"id": oid, "tags": tags, "polygons": _polygons(g)}
                         for (oid, tags, g), k in zip(found, keep) if k]}
    for kind, pts in h.points.items():
        inside = _in_box(extent, lim, [(p["lon"], p["lat"]) for p in pts])
        out[kind] = [p for p, k in zip(pts, inside) if k]
    # Only what has a node (a vertex) in the box plus the buffer, whole.
    out["waterways"] = [ww for ww in h.waterways if _in_box(extent, lim, ww["coords"]).any()]
    out["wetlands"] = [{"id": oid, "wetland": tags.get("wetland"), "polygons": _polygons(g)}
                       for oid, tags, g in h.wetlands if _in_box(extent, lim, shapely.get_coordinates(g)).any()]
    out["areas"] = [{"id": oid, "tags": tags, "polygons": _polygons(g)}
                    for oid, tags, g in h.areas if _in_box(extent, lim, shapely.get_coordinates(g)).any()]
    log.info("OSM features: %d buildings (%d multipolygons), %d trees, %d signal/stop/crossing nodes, "
             "%d waterways (%d culverts), %d wetlands, %d landmark areas; "
             "%d broken areas skipped, %d ways counted once as their relation",
             len(out["buildings"]), sum(b["id"][0] == "r" for b in out["buildings"]), len(out["trees"]),
             len(out["nodes"]), len(out["waterways"]), sum(ww["culvert"] for ww in out["waterways"]),
             len(out["wetlands"]), len(out["areas"]), h.broken, len(h.buildings) - len(found))
    return out


# ---------------------------------------------------------------- Microsoft footprints


def tile_xy(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    """Column and row of the Web Mercator (Bing Maps) tile holding lon/lat at this zoom."""
    size = 1 << zoom
    sin = math.sin(math.radians(max(-MERCATOR_LAT, min(MERCATOR_LAT, lat))))
    x = (lon + 180.0) / 360.0 * size
    y = (0.5 - math.log((1 + sin) / (1 - sin)) / (4 * math.pi)) * size
    return min(size - 1, max(0, math.floor(x))), min(size - 1, max(0, math.floor(y)))


def quadkey(x: int, y: int, zoom: int) -> str:
    """The Bing Maps quadkey of tile (x, y): one digit per zoom level, x bit + 2 * y bit."""
    return "".join(str(((x >> i) & 1) + 2 * ((y >> i) & 1)) for i in range(zoom - 1, -1, -1))


def quadkeys(extent: Extent, buffer_m: float = 0.0, zoom: int = QUADKEY_ZOOM) -> list[str]:
    """Every quadkey at this zoom touching the box plus the buffer, sorted."""
    w, s, e, n = extent.bbox_lonlat(buffer_m)
    x0, y0 = tile_xy(w, n, zoom)  # north-west corner: the smallest row
    x1, y1 = tile_xy(e, s, zoom)
    return sorted(quadkey(x, y, zoom) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1))


def ms_tile_urls(links_csv: str, keys) -> list[tuple[str, str]]:
    """(quadkey, url) of the dataset-links.csv rows for these quadkeys.

    Column names are matched without regard to case (Location, QuadKey, Url). A quadkey
    that lost its leading zeros (a spreadsheet round trip) is padded back, and one in the
    url (".../quadkey=023131130/...") wins over the column. United States rows are taken;
    if a quadkey has none, the rows of every location are taken, with a warning.
    """
    keys = sorted(set(keys))
    zoom = len(keys[0]) if keys else QUADKEY_ZOOM
    rows: dict[str, list[tuple[bool, str]]] = {k: [] for k in keys}
    for raw in csv.DictReader(io.StringIO(links_csv.lstrip("\ufeff"))):
        row = {k.strip().lower(): v.strip() for k, v in raw.items()
               if isinstance(k, str) and isinstance(v, str)}  # short rows give None, long ones a list
        url = row.get("url", "")
        in_url = re.search(r"quadkey=(\d+)", url)
        qk = in_url.group(1) if in_url else row.get("quadkey", "")
        qk = qk.zfill(zoom) if qk.isdigit() else qk
        if url and qk in rows:
            us = re.sub(r"[^a-z]", "", row.get("location", "").lower()) in US_LOCATIONS
            rows[qk].append((us, url))
    out = []
    for qk in keys:
        urls = [u for us, u in rows[qk] if us]
        if not urls and rows[qk]:
            urls = [u for _, u in rows[qk]]
            log.warning("no United States row for quadkey %s in dataset-links.csv; taking %d row(s) of other "
                        "locations", qk, len(urls))
        out += [(qk, u) for u in dict.fromkeys(urls)]
    return out


def _basename(url: str) -> str:
    return url.split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1]


def _ms_downloads(cfg, raw_dir: Path, keys: list[str]) -> list[Path]:
    """The footprint files for these quadkeys (cached). A cached dataset-links.csv that
    names no file, or a file that no longer downloads, is fetched again once."""
    links = raw_dir / MS_DIR / "dataset-links.csv"
    for attempt in range(2):
        was_cached = _cached(links)
        download(cfg.sources["ms_buildings_links"], links)
        try:
            rows = ms_tile_urls(links.read_text(encoding="utf-8-sig"), keys)
            if not rows:
                raise FetchError(f"dataset-links.csv lists no footprint file for quadkeys {keys}")
            return [download(url, raw_dir / MS_DIR / f"{qk}_{_basename(url)}", timeout=600)
                    for qk, url in rows]
        except FetchError as exc:
            if not was_cached or attempt:
                raise
            log.warning("%s; downloading dataset-links.csv again", exc)
            links.unlink(missing_ok=True)
    raise AssertionError("unreachable")


def _ms_candidates(path: Path, bbox_lonlat) -> tuple[list[tuple[list, float]], int]:
    """(outer ring, height) of the features in a quadkey file whose ring touches bbox_lonlat,
    and the number of unreadable lines."""
    w, s, e, n = bbox_lonlat
    found, bad = [], 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                feat = json.loads(line)
                geom = feat.get("geometry") or {}
                polys = {"Polygon": [geom.get("coordinates")], "MultiPolygon": geom.get("coordinates")}.get(
                    geom.get("type"), [])
                height = (feat.get("properties") or {}).get("height")
                height = float(height) if isinstance(height, (int, float)) else -1.0
                for poly in polys or []:
                    ring = [p[:2] for p in poly[0]] if poly else []
                    if len(ring) < 3:
                        continue
                    lons, lats = [p[0] for p in ring], [p[1] for p in ring]
                    if min(lons) <= e and max(lons) >= w and min(lats) <= n and max(lats) >= s:
                        found.append((ring, height))
            except (ValueError, TypeError, IndexError, AttributeError):
                bad += 1
    return found, bad


def _ms_in_box(cands: list[tuple[list, float]], extent: Extent, buffer_m: float) -> list[dict]:
    """The candidates whose ring intersects the box plus the buffer, in ms_buildings.json form."""
    from shapely.geometry import LineString, box

    if not cands:
        return []
    lim = extent.half_size + buffer_m
    square = box(-lim, -lim, lim, lim)
    pts = np.array([p for ring, _ in cands for p in ring], dtype=float)
    x, y = extent.to_level(pts[:, 0], pts[:, 1])
    out, start = [], 0
    for ring, height in cands:
        xs, ys = x[start:start + len(ring)], y[start:start + len(ring)]
        start += len(ring)
        lo_x, hi_x, lo_y, hi_y = xs.min(), xs.max(), ys.min(), ys.max()
        if hi_x < -lim or lo_x > lim or hi_y < -lim or lo_y > lim:
            continue
        inside = lo_x >= -lim and hi_x <= lim and lo_y >= -lim and hi_y <= lim
        # An outline test, so a self-intersecting ring can't make shapely fail.
        if inside or LineString(np.column_stack([xs, ys])).intersects(square):
            out.append({"ring": _ring(ring), "height": height})
    return out


def fetch_ms_buildings(cfg, raw_dir) -> Path:
    """raw/ms_buildings.json: the Microsoft footprints intersecting the box plus the buffer."""
    raw_dir = Path(raw_dir)
    dest = raw_dir / MS_BUILDINGS
    if _cached(dest):
        log.info("cached: %s", dest)
        return dest
    extent = Extent.from_config(cfg)
    keys = quadkeys(extent, cfg.buffer_m)
    files = _ms_downloads(cfg, raw_dir, keys)
    found, total_bad = [], 0
    for path in files:
        cands, bad = _ms_candidates(path, extent.bbox_lonlat(cfg.buffer_m))
        found += _ms_in_box(cands, extent, cfg.buffer_m)
        total_bad += bad
    if total_bad:
        log.warning("Microsoft footprints: %d unreadable lines skipped", total_bad)
    _write_json(dest, found)
    known = sum(f["height"] > 0 for f in found)
    log.info("Microsoft footprints: %d in the box plus %g m (%d with a height) from quadkey(s) %s",
             len(found), cfg.buffer_m, known, ", ".join(keys))
    return dest


# ---------------------------------------------------------------- Planetary Computer rasters


def _stac_items(cfg, collection: str, bbox_lonlat) -> list:
    """Every item of a STAC collection (unsigned) over a lon/lat box."""
    import pystac_client

    def search():
        cat = pystac_client.Client.open(cfg.sources["stac"])
        return list(cat.search(collections=[collection], bbox=list(bbox_lonlat)).items())

    return _retrying(f"{collection} STAC search", search)


def _href(item, key: str) -> str:
    """The asset key's href, or the first GeoTIFF asset's if the item has no such key."""
    asset = item.assets.get(key) or next(
        (a for a in item.assets.values() if "tiff" in str(getattr(a, "media_type", "") or "")), None)
    if asset is None:
        raise FetchError(f"STAC item {item.id} has no {key!r} GeoTIFF asset")
    return asset.href


def fetch_dsm(cfg, raw_dir) -> list[Path]:
    """raw/dsm/<item id>.tif: each lidar DSM item clipped to the box plus the buffer."""
    import planetary_computer

    extent = Extent.from_config(cfg)
    raw_dir = Path(raw_dir)
    collection = cfg.sources["dsm_collection"]
    items = _stac_items(cfg, collection, extent.bbox_lonlat(cfg.buffer_m))
    log.info("DSM: %d %s items over the box", len(items), collection)
    bbox = extent.bbox_utm(cfg.buffer_m)
    (raw_dir / DSM_DIR).mkdir(parents=True, exist_ok=True)

    def one(item):
        dest = raw_dir / DSM_DIR / f"{item.id}.tif"
        return _clip_cached(item.id, dest, lambda: clip_raster(
            planetary_computer.sign(_href(item, "data")), bbox, dest, bbox_crs=LEVEL_CRS))

    with ThreadPoolExecutor(WORKERS) as ex:
        paths = [p for p in ex.map(one, items) if p is not None]
    if not paths:
        raise FetchError(f"no lidar surface model (DSM) found for the box in {collection}")
    return sorted(paths)


def naip_items(items, bbox_lonlat) -> list:
    """The items of the newest NAIP year that covers the whole box, sorted by id.

    If no year covers it, the year covering most of it (the newest of equals), with a warning.
    """
    from shapely.geometry import box
    from shapely.ops import unary_union

    area = box(*bbox_lonlat)
    years: dict[str, list] = {}
    for it in items:
        years.setdefault(str(it.properties.get("naip:year", "")), []).append(it)
    if not years:
        return []
    share = {y: unary_union([box(*it.bbox) for it in its]).intersection(area).area / area.area
             for y, its in years.items()}
    whole = [y for y, v in share.items() if v >= NAIP_COVER]
    year = max(whole) if whole else max(share, key=lambda y: (share[y], y))
    if not whole:
        log.warning("no NAIP year covers the whole box; %s covers %.1f %%", year, 100 * share[year])
    return sorted(years[year], key=lambda it: it.id)


def _overview_level(ds, res: float) -> int | None:
    """Index of the coarsest overview still as fine as res (None: the full image)."""
    level = None
    for i, factor in enumerate(ds.overviews(1)):
        if abs(ds.res[0]) * factor <= res * (1 + 1e-6):
            level = i
    return level


def _grid_window(src, dst_transform, n: int) -> Window | None:
    """The part of the n x n grid an image covers, or None if it misses it."""
    left, bottom, right, top = transform_bounds(src.crs, LEVEL_CRS, *src.bounds, densify_pts=21)
    w = from_bounds(left, bottom, right, top, transform=dst_transform)
    c0, r0 = max(0, math.floor(w.col_off)), max(0, math.floor(w.row_off))
    c1, r1 = min(n, math.ceil(w.col_off + w.width)), min(n, math.ceil(w.row_off + w.height))
    return Window(c0, r0, c1 - c0, r1 - r0) if c1 > c0 and r1 > r0 else None


def _warp_red_nir(href: str, dst_transform, n: int, res: float):
    """(window, 2 x h x w uint8 red and NIR) of one NAIP image on the grid; None if it misses it."""
    with rasterio.open(href) as base:
        if base.count < 4:
            raise FetchError(f"{_basename(href)} has {base.count} bands; NIR is band 4")
        level = _overview_level(base, res)
    with rasterio.open(href, **({} if level is None else {"overview_level": level})) as src:
        win = _grid_window(src, dst_transform, n)
        if win is None:
            return None
        # Fill outside the image (and the black collar, all four bands 0) is no data.
        with WarpedVRT(src, crs=LEVEL_CRS, transform=dst_transform, width=n, height=n,
                       resampling=Resampling.average, src_nodata=0, nodata=0) as vrt:
            return win, vrt.read([1, 4], window=win)


def fetch_naip_rn(cfg, raw_dir) -> Path:
    """raw/naip_rn_2m.tif: NAIP red and near-infrared on the 2 m analysis grid."""
    import planetary_computer

    from .canopy import analysis_grid

    raw_dir = Path(raw_dir)
    dest = raw_dir / NAIP_RN_FILE
    if _cached(dest):
        log.info("cached: %s", dest)
        return dest
    extent = Extent.from_config(cfg)
    n, px = analysis_grid(extent)
    transform = extent.transform(n)
    box = extent.bbox_lonlat()
    items = naip_items(_stac_items(cfg, cfg.sources["naip_collection"], box), box)
    if not items:
        raise FetchError("no NAIP imagery found for the box")
    log.info("NAIP red/NIR: %d items from %s onto %d x %d cells of %.1f m", len(items),
             items[0].properties.get("naip:year"), n, n, px)

    def one(item):
        def read():
            with rasterio.Env(**GDAL_REMOTE):
                return _warp_red_nir(planetary_computer.sign(_href(item, "image")), transform, n, px)

        return _retrying(item.id, read)

    rn = np.zeros((2, n, n), np.uint8)
    with ThreadPoolExecutor(WORKERS) as ex:
        for item, got in zip(items, ex.map(one, items)):  # map keeps the order: the first item wins
            if got is None:
                log.info("NAIP %s misses the box", item.id)
                continue
            win, data = got
            target = rn[:, win.row_off:win.row_off + win.height, win.col_off:win.col_off + win.width]
            take = (target[0] == 0) & (target[1] == 0) & ((data[0] > 0) | (data[1] > 0))
            target[:, take] = data[:, take]
    empty = float(((rn[0] == 0) & (rn[1] == 0)).mean())
    if empty > 0.001:
        log.warning("NAIP red/NIR: %.2f %% of the grid has no image", 100 * empty)
    profile = {"driver": "GTiff", "width": n, "height": n, "count": 2, "dtype": "uint8", "crs": LEVEL_CRS,
               "transform": transform, "nodata": 0, "tiled": True, "blockxsize": 256, "blockysize": 256,
               "compress": "deflate", "predictor": 2}
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    try:
        with rasterio.open(part, "w", **profile) as out:
            out.write(rn)
            out.descriptions = ("red", "nir")
        part.replace(dest)
    finally:
        part.unlink(missing_ok=True)
    log.info("saved %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
    return dest


# ---------------------------------------------------------------- stage


def _count(path: Path, key: str | None = None) -> int | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return len(data[key] if key else data)


def _osm_current(path: Path) -> bool:
    """Whether raw/osm_features.json exists and has every key of OSM_KEYS. One from before the
    waterways, wetlands and areas (or one that doesn't parse) is made again."""
    if not _cached(path):
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return False
    return isinstance(data, dict) and all(k in data for k in OSM_KEYS)


def run(cfg, raw_dir=None, out_dir=None) -> None:
    """All four inputs into raw_dir. out_dir is unused: nothing here is a stage output."""
    extent = Extent.from_config(cfg)
    raw_dir = Path(raw_dir or cfg.raw_dir)
    check_box(raw_dir, extent.bbox_utm(cfg.buffer_m))
    osm = raw_dir / OSM_FEATURES
    if _osm_current(osm):
        log.info("cached: %s", osm)
    else:
        if _cached(osm):
            log.info("%s is from before waterways, wetlands and areas (or unreadable); extracting it again",
                     osm.name)
        url = cfg.sources["osm_extract"]
        pbf = download(url, raw_dir / url.rsplit("/", 1)[-1], timeout=600)
        _write_json(osm, extract_osm_features(pbf, extent, cfg.buffer_m))
    if cfg.buildings_microsoft:
        fetch_ms_buildings(cfg, raw_dir)
    else:
        log.info("Microsoft footprints are off (buildings: microsoft: false)")
    dsm = fetch_dsm(cfg, raw_dir)
    naip = fetch_naip_rn(cfg, raw_dir)
    ms = _count(raw_dir / MS_BUILDINGS) if cfg.buildings_microsoft else None
    log.info("features: %s OSM buildings, %s trees, %s nodes, %s waterways, %s wetlands, %s landmark areas; "
             "%s Microsoft footprints; %d DSM tiles; %s",
             _count(osm, "buildings"), _count(osm, "trees"), _count(osm, "nodes"), _count(osm, "waterways"),
             _count(osm, "wetlands"), _count(osm, "areas"), "no" if ms is None else ms, len(dsm), naip.name)
