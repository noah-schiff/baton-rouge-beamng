"""Stage 1: download the raw inputs into data/raw, with a cache and retries.

The USGS 1 m DEM and the NAIP aerials are cloud-optimized GeoTIFFs, so only the
window covering the buffered box is read from each file (NAIP at a reduced
resolution through its overviews) instead of downloading whole 10 km tiles.

Several lidar projects cover each DEM tile here, and the newest one often covers
only part of it, so every project's piece is kept. File names start with the
publication date and the terrain stage merges them newest first, letting older
projects fill the gaps.
"""
from __future__ import annotations

import json
import logging
import math
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import requests
from affine import Affine
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds

from .geo import LEVEL_CRS, Extent

log = logging.getLogger(__name__)

BACKOFF = [2, 4, 8, 16]               # seconds between attempts (patched to 0 in tests)
# overpass-api.de rejects generic clients (python-requests) with HTTP 406.
HEADERS = {"User-Agent": "brmap/0.1 (BeamNG.drive Baton Rouge map pipeline)"}
RETRY_STATUS = {429, 500, 502, 503, 504}
NAIP_RES_M = 1.2                      # NAIP 2023 is 0.3 m; 1.2 m is plenty for a 2 to 4 m texture
DEM_DATASET = "Digital Elevation Model (DEM) 1 meter"
NLCD_COVERAGE = "mrlc_download__NLCD_2021_Land_Cover_L48"
WORKERS = 4
GDAL_REMOTE = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.tiff",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MAX_RETRY": "4",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "VSI_CACHE": "TRUE",
}


class FetchError(RuntimeError):
    """A download or remote read failed for good (the message names the source)."""


class _Retryable(Exception):
    pass


def _sleep(attempt: int) -> None:
    time.sleep(BACKOFF[min(attempt, len(BACKOFF) - 1)])


def _cached(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def download(url: str, dest, retries: int = 4, params=None, data=None, timeout: float = 300) -> Path:
    """GET (or POST when data is given) url into dest. Skips files already present.

    Writes to dest.part and renames on success, so a dropped connection never
    leaves a truncated file behind. 429/5xx and network errors are retried.
    """
    dest = Path(dest)
    if _cached(dest):
        log.info("cached: %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    method = "POST" if data is not None else "GET"
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with requests.request(method, url, params=params, data=data, stream=True,
                                  timeout=timeout, headers=HEADERS) as r:
                if r.status_code in RETRY_STATUS:
                    raise _Retryable(f"HTTP {r.status_code}")
                if r.status_code >= 400:
                    raise FetchError(f"{url}: HTTP {r.status_code} {r.reason}")
                with open(part, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            part.replace(dest)
            return dest
        except (_Retryable, requests.ConnectionError, requests.Timeout,
                requests.exceptions.ChunkedEncodingError) as exc:
            last = exc
            part.unlink(missing_ok=True)
            log.warning("attempt %d/%d failed for %s: %s", attempt + 1, retries + 1, url, exc)
            if attempt < retries:
                _sleep(attempt)
        except BaseException:
            part.unlink(missing_ok=True)
            raise
    raise FetchError(f"giving up on {url} after {retries + 1} attempts: {last}")


def _retrying(what: str, fn, retries: int = 4):
    """Call fn(), retrying any non-FetchError exception with backoff."""
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except FetchError:
            raise
        except Exception as exc:  # network and GDAL read errors
            last = exc
            log.warning("attempt %d/%d failed for %s: %s", attempt + 1, retries + 1, what, exc)
            if attempt < retries:
                _sleep(attempt)
    raise FetchError(f"giving up on {what} after {retries + 1} attempts: {last}")


def clip_raster(src, bbox, dest, bbox_crs=None, res: float | None = None,
                bands: list[int] | None = None) -> Path | None:
    """Copy the part of raster src (path or URL) inside bbox to a local GeoTIFF.

    bbox is (left, bottom, right, top) in bbox_crs (default: the raster's own CRS).
    res resamples to that pixel size (averaging, which uses the overviews of a COG).
    Returns None, writing nothing, when the box misses the raster or only has nodata.
    """
    dest = Path(dest)
    with rasterio.open(src) as ds:
        if bbox_crs is not None and ds.crs is not None and ds.crs != rasterio.crs.CRS.from_user_input(bbox_crs):
            bbox = transform_bounds(bbox_crs, ds.crs, *bbox, densify_pts=21)
        w = from_bounds(*bbox, transform=ds.transform)
        c0 = max(0, math.floor(w.col_off + 1e-6))
        r0 = max(0, math.floor(w.row_off + 1e-6))
        c1 = min(ds.width, math.ceil(w.col_off + w.width - 1e-6))
        r1 = min(ds.height, math.ceil(w.row_off + w.height - 1e-6))
        if c1 <= c0 or r1 <= r0:
            return None
        win = Window(c0, r0, c1 - c0, r1 - r0)
        idx = bands or list(range(1, ds.count + 1))
        scale = 1.0 if res is None else res / abs(ds.res[0])
        out_w = max(1, int(round(win.width / scale)))
        out_h = max(1, int(round(win.height / scale)))
        data = ds.read(idx, window=win, out_shape=(len(idx), out_h, out_w),
                       resampling=Resampling.average if scale > 1 else Resampling.nearest)
        if ds.nodata is not None and np.all(data == ds.nodata):
            return None
        transform = ds.window_transform(win) @ Affine.scale(win.width / out_w, win.height / out_h)
        profile = {
            "driver": "GTiff", "width": out_w, "height": out_h, "count": len(idx),
            "dtype": data.dtype, "crs": ds.crs, "transform": transform, "nodata": ds.nodata,
            "tiled": True, "blockxsize": 256, "blockysize": 256, "compress": "deflate",
            "predictor": 3 if np.issubdtype(data.dtype, np.floating) else 2, "BIGTIFF": "IF_SAFER",
        }
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    try:
        with rasterio.open(part, "w", **profile) as out:
            out.write(data)
        part.replace(dest)
    finally:
        part.unlink(missing_ok=True)
    return dest


# ---------------------------------------------------------------- sources

def overpass_query(extent: Extent, buffer_m: float) -> str:
    """Roads and water features in the buffered box, with full geometry."""
    w, s, e, n = extent.bbox_lonlat(buffer_m)
    b = f"({s:.6f},{w:.6f},{n:.6f},{e:.6f})"
    return (
        "[out:json][timeout:900][maxsize:1073741824];\n(\n"
        f'  way["highway"]{b};\n'
        f'  way["natural"="water"]{b};\n'
        f'  way["waterway"="riverbank"]{b};\n'
        f'  relation["natural"="water"]{b};\n'
        f'  relation["waterway"="riverbank"]{b};\n'
        ");\nout body geom;\n"
    )


@dataclass(frozen=True)
class DemSource:
    url: str
    date: str
    title: str

    @property
    def filename(self) -> str:
        return f"{self.date}_{self.url.rsplit('/', 1)[-1]}"


def dem_candidates(items: list[dict]) -> list[DemSource]:
    """Every distinct DEM GeoTIFF in a TNM product listing, newest first."""
    seen: set[str] = set()
    out: list[DemSource] = []
    for it in items:
        url = it.get("downloadURL") or ""
        if not url.lower().endswith((".tif", ".tiff")) or url in seen:
            continue
        seen.add(url)
        date = str(it.get("publicationDate") or "0000-00-00")[:10]
        out.append(DemSource(url=url, date=date, title=it.get("title", "")))
    out.sort(key=lambda d: (d.date, d.url), reverse=True)
    return out


def nlcd_params(extent: Extent, buffer_m: float) -> list[tuple[str, str]]:
    """WCS 2.0.1 GetCoverage parameters for the buffered box (NLCD is in EPSG:5070)."""
    w, s, e, n = Transformer.from_crs(LEVEL_CRS, "EPSG:5070", always_xy=True).transform_bounds(
        *extent.bbox_utm(buffer_m), densify_pts=21)
    return [("service", "WCS"), ("version", "2.0.1"), ("request", "GetCoverage"),
            ("coverageId", NLCD_COVERAGE), ("subset", f"X({w:.1f},{e:.1f})"),
            ("subset", f"Y({s:.1f},{n:.1f})"), ("format", "image/geotiff")]


def latest_naip(items) -> list:
    """The NAIP items from the most recent year only (keeps their order)."""
    items = list(items)
    if not items:
        return []
    year = max(str(i.properties.get("naip:year", "")) for i in items)
    return [i for i in items if str(i.properties.get("naip:year", "")) == year]


def check_cache_box(raw_dir: Path, bbox_utm) -> None:
    """Clear cached clips when the requested box isn't inside the one they were cut to."""
    raw_dir = Path(raw_dir)
    marker = raw_dir / "box.json"
    if marker.exists():
        old = json.loads(marker.read_text(encoding="utf-8"))
        inside = (bbox_utm[0] >= old[0] - 1e-6 and bbox_utm[1] >= old[1] - 1e-6
                  and bbox_utm[2] <= old[2] + 1e-6 and bbox_utm[3] <= old[3] + 1e-6)
        if inside:
            return
        log.info("box grew beyond the cached data; clearing %s", raw_dir)
        for p in raw_dir.iterdir():
            if p.name != "box.json":
                shutil.rmtree(p) if p.is_dir() else p.unlink()
    raw_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps([float(v) for v in bbox_utm]), encoding="utf-8")


# ---------------------------------------------------------------- fetchers

def _clip_cached(what: str, dest: Path, read) -> Path | None:
    """Run read() unless dest (or its 'nothing here' marker) already exists."""
    empty = dest.with_name(dest.name + ".empty")
    if _cached(dest):
        return dest
    if empty.exists():
        return None

    def attempt():
        with rasterio.Env(**GDAL_REMOTE):
            return read()

    out = _retrying(what, attempt)
    if out is None:
        empty.touch()
        log.info("no data in the box: %s", what)
    else:
        log.info("saved %s (%.1f MB)", out.name, out.stat().st_size / 1e6)
    return out


def _tnm_items(cfg, extent: Extent) -> list[dict]:
    w, s, e, n = extent.bbox_lonlat(cfg.buffer_m)
    items: list[dict] = []
    while True:
        r = requests.get(cfg.sources["tnm_api"], timeout=120, headers=HEADERS, params={
            "datasets": DEM_DATASET, "bbox": f"{w},{s},{e},{n}", "prodFormats": "GeoTIFF",
            "max": 100, "offset": len(items)})
        r.raise_for_status()
        j = r.json()
        batch = j.get("items") or []
        items += batch
        if not batch or len(items) >= int(j.get("total") or 0):
            return items


def fetch_dem(cfg, extent: Extent, raw_dir: Path) -> list[Path]:
    """Download each DEM tile whole (fast, resumable per file), clip it, delete the tile.

    Remote windowed reads of the 1 m tiles turned out far slower than plain
    downloads, so only NAIP (read through its overviews) is clipped remotely.
    """
    cands = dem_candidates(_retrying("TNM DEM search", lambda: _tnm_items(cfg, extent)))
    log.info("DEM: %d candidate files", len(cands))
    bbox = extent.bbox_utm(cfg.buffer_m)

    def one(c: DemSource):
        dest = raw_dir / "dem" / c.filename
        empty = dest.with_name(dest.name + ".empty")
        if _cached(dest):
            return dest
        if empty.exists():
            return None
        full = download(c.url, raw_dir / "dem_full" / c.filename, timeout=600)
        try:
            out = clip_raster(full, bbox, dest, bbox_crs=LEVEL_CRS)
        finally:
            full.unlink(missing_ok=True)
        if out is None:
            empty.parent.mkdir(parents=True, exist_ok=True)
            empty.touch()
            log.info("no data in the box: %s", c.url)
        else:
            log.info("saved %s (%.1f MB)", out.name, out.stat().st_size / 1e6)
        return out

    with ThreadPoolExecutor(3) as ex:
        paths = [p for p in ex.map(one, cands) if p is not None]
    if not paths:
        raise FetchError("no DEM data found for the box")
    return sorted(paths, key=lambda p: p.name, reverse=True)


def fetch_naip(cfg, extent: Extent, raw_dir: Path) -> list[Path]:
    import planetary_computer
    import pystac_client

    w, s, e, n = extent.bbox_lonlat(cfg.buffer_m)

    def search():
        cat = pystac_client.Client.open(cfg.sources["stac"])
        return list(cat.search(collections=["naip"], bbox=[w, s, e, n]).items())

    items = latest_naip(_retrying("NAIP STAC search", search))
    log.info("NAIP: %d items from %s", len(items),
             items[0].properties.get("naip:year") if items else "none")
    bbox = extent.bbox_utm(cfg.buffer_m)

    def one(item):
        dest = raw_dir / "naip" / f"{item.id}.tif"
        return _clip_cached(item.id, dest, lambda: clip_raster(
            planetary_computer.sign(item.assets["image"].href), bbox, dest,
            bbox_crs=LEVEL_CRS, res=NAIP_RES_M, bands=[1, 2, 3]))

    with ThreadPoolExecutor(WORKERS) as ex:
        paths = [p for p in ex.map(one, items) if p is not None]
    if not paths:
        raise FetchError("no NAIP imagery found for the box")
    return sorted(paths)


def fetch_nlcd(cfg, extent: Extent, raw_dir: Path) -> Path:
    dest = download(cfg.sources["nlcd_wcs"], raw_dir / "nlcd.tif",
                    params=nlcd_params(extent, cfg.buffer_m))
    try:
        with rasterio.open(dest) as ds:
            log.info("NLCD: %s px, %s", ds.shape, ds.crs)
    except rasterio.errors.RasterioIOError as exc:
        head = dest.read_bytes()[:300]
        dest.unlink()
        raise FetchError(f"NLCD service returned something that isn't a GeoTIFF: {head!r}") from exc
    return dest


def _wanted(tags) -> bool:
    return "highway" in tags or tags.get("natural") == "water" or tags.get("waterway") == "riverbank"


def _geom(coords) -> list[dict]:
    return [{"lat": round(lat, 7), "lon": round(lon, 7)} for lon, lat in coords]


def osm_json_from_extract(path, bbox_lonlat) -> dict:
    """Read an OSM extract (.osm.pbf or .osm) into the same JSON an Overpass
    'out body geom' query returns: roads and water ways touching the box, plus
    water relations with their member ways' geometry."""
    import osmium

    w, s, e, n = bbox_lonlat

    def touches(coords) -> bool:
        if not coords:
            return False
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        return min(lons) <= e and max(lons) >= w and min(lats) <= n and max(lats) >= s

    class Relations(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.found: dict[int, tuple[dict, list]] = {}

        def relation(self, r):
            if _wanted(r.tags):
                self.found[r.id] = ({t.k: t.v for t in r.tags},
                                    [(m.ref, m.role) for m in r.members if m.type == "w"])

    rels = Relations()
    rels.apply_file(str(path))
    member_ids = {ref for _, members in rels.found.values() for ref, _ in members}

    class Ways(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.out: list[dict] = []
            self.members: dict[int, list] = {}

        def way(self, wy):
            direct = _wanted(wy.tags)
            if not direct and wy.id not in member_ids:
                return
            refs, coords = [], []
            for nd in wy.nodes:
                refs.append(nd.ref)
                if nd.location.valid():
                    coords.append((nd.location.lon, nd.location.lat))
            if wy.id in member_ids:
                self.members[wy.id] = coords
            if direct and touches(coords):
                self.out.append({"type": "way", "id": wy.id, "nodes": refs,
                                 "tags": {t.k: t.v for t in wy.tags}, "geometry": _geom(coords)})

    ways = Ways()
    ways.apply_file(str(path), locations=True)
    relations = []
    for rid, (tags, members) in rels.found.items():
        present = [(ref, role) for ref, role in members if ref in ways.members]
        if not any(touches(ways.members[ref]) for ref, _ in present):
            continue
        relations.append({"type": "relation", "id": rid, "tags": tags, "members": [
            {"type": "way", "ref": ref, "role": role, "geometry": _geom(ways.members[ref])}
            for ref, role in present]})
    return {"version": 0.6, "generator": "brmap (OSM extract)", "elements": ways.out + relations}


def _log_osm(j: dict, dest: Path) -> None:
    kinds: dict[str, int] = {}
    for el in j.get("elements", []):
        kinds[el["type"]] = kinds.get(el["type"], 0) + 1
    log.info("OSM: %s (%.1f MB)", kinds, dest.stat().st_size / 1e6)


def fetch_osm_extract(cfg, extent: Extent, raw_dir: Path) -> Path:
    """OSM data from a Geofabrik extract (no rate limits), converted to osm.json."""
    try:
        import osmium  # noqa: F401
    except ImportError as exc:
        raise FetchError("the osmium package is needed to read OSM extracts") from exc
    url = cfg.sources["osm_extract"]
    pbf = download(url, raw_dir / url.rsplit("/", 1)[-1], timeout=600)
    j = osm_json_from_extract(pbf, extent.bbox_lonlat(cfg.buffer_m))
    dest = raw_dir / "osm.json"
    part = dest.with_name(dest.name + ".part")
    part.write_text(json.dumps(j), encoding="utf-8")
    part.replace(dest)
    _log_osm(j, dest)
    return dest


def fetch_osm(cfg, extent: Extent, raw_dir: Path) -> Path:
    """osm.json from the configured source, falling back to the other one."""
    dest = raw_dir / "osm.json"
    if _cached(dest):
        log.info("cached: %s", dest)
        return dest
    order = [fetch_osm_extract, fetch_osm_overpass]
    if cfg.osm_source == "overpass":
        order.reverse()
    for i, source in enumerate(order):
        try:
            return source(cfg, extent, raw_dir)
        except FetchError as exc:
            if i == len(order) - 1:
                raise
            log.warning("OSM source failed (%s); trying the other source", exc)
    raise AssertionError("unreachable")


def fetch_osm_overpass(cfg, extent: Extent, raw_dir: Path) -> Path:
    query = {"data": overpass_query(extent, cfg.buffer_m)}
    servers = [cfg.sources["overpass"]] + [u for u in [cfg.sources.get("overpass_fallback")] if u]
    dest = None
    for i, url in enumerate(servers):
        try:
            dest = download(url, raw_dir / "osm.json", data=query, timeout=1200)
            break
        except FetchError as exc:
            if i == len(servers) - 1:
                raise
            log.warning("Overpass server failed (%s); trying %s", exc, servers[i + 1])
    try:
        j = json.loads(dest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        dest.unlink()
        raise FetchError(f"Overpass returned invalid JSON: {exc}") from exc
    remark = str(j.get("remark", ""))
    if "error" in remark.lower():
        dest.unlink()
        raise FetchError(f"Overpass reported: {remark[:300]}")
    _log_osm(j, dest)
    return dest


def fetch_all(cfg, raw_dir=Path("data/raw")) -> dict[str, list[Path]]:
    extent = Extent.from_config(cfg)
    raw_dir = Path(raw_dir)
    check_cache_box(raw_dir, extent.bbox_utm(cfg.buffer_m))
    # The four sources are independent, so they download side by side.
    with ThreadPoolExecutor(3) as ex:
        osm = ex.submit(fetch_osm, cfg, extent, raw_dir)
        dem = ex.submit(fetch_dem, cfg, extent, raw_dir)
        naip = ex.submit(fetch_naip, cfg, extent, raw_dir)
        nlcd = fetch_nlcd(cfg, extent, raw_dir)
        out = {"osm": [osm.result()], "nlcd": [nlcd], "dem": dem.result(), "naip": naip.result()}
    for k, paths in out.items():
        log.info("%s: %d file(s), %.1f MB", k, len(paths), sum(p.stat().st_size for p in paths) / 1e6)
    return out


def run(cfg) -> None:
    fetch_all(cfg, raw_dir=Path(cfg.raw_dir))
