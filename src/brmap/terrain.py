"""Stage 2: DEM files -> an n x n height grid -> 16-bit heightmap PNG.

Grid convention (see geo.py): row 0 is the north edge and column 0 the west
edge. Each cell holds the height in meters (NAVD88) at the cell center.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy import ndimage

from .geo import LEVEL_CRS, Extent

log = logging.getLogger(__name__)

OUT_DIR = Path("data/out")
RAW_DIR = Path("data/raw")
# Anything outside this range is a nodata sentinel (-9999, -999999, ...), declared or not.
VALID_RANGE = (-1000.0, 10000.0)


def _mask_invalid(z: np.ndarray, nodata) -> np.ndarray:
    bad = ~np.isfinite(z) | (z < VALID_RANGE[0]) | (z > VALID_RANGE[1])
    if nodata is not None and np.isfinite(nodata):
        bad |= z == nodata
    z[bad] = np.nan
    return z


def build_heights(dem_paths, extent: Extent, n: int, newest=lambda p: p.name) -> np.ndarray:
    """Average every DEM file into the grid; returns float32 n x n with NaN where no data.

    The newest file is applied first and older files only fill cells that are still empty.
    By default newness is the file name: DEM names start with their publication date
    (see fetch.py). Pass another key for files named otherwise.
    """
    paths = sorted((Path(p) for p in dem_paths), key=newest, reverse=True)
    dst_transform = extent.transform(n)
    px = extent.pixel_size(n)
    heights = np.full((n, n), np.nan, np.float32)
    for p in paths:
        with rasterio.open(p) as src:
            z = _mask_invalid(src.read(1).astype(np.float32), src.nodata)
            down = px >= min(abs(src.res[0]), abs(src.res[1]))
            layer = np.full((n, n), np.nan, np.float32)
            reproject(z, layer, src_transform=src.transform, src_crs=src.crs, src_nodata=np.nan,
                      dst_transform=dst_transform, dst_crs=LEVEL_CRS, dst_nodata=np.nan,
                      resampling=Resampling.average if down else Resampling.bilinear)
        empty = np.isnan(heights)
        new = empty & ~np.isnan(layer)
        heights[new] = layer[new]
        log.info("DEM %s: +%d cells, %d still empty", p.name, int(new.sum()),
                 int(np.isnan(heights).sum()))
    return heights


def fill_voids(h: np.ndarray) -> tuple[np.ndarray, int]:
    """Copy of h with each NaN replaced by the nearest valid cell; also the count filled."""
    h = np.array(h, dtype=np.float32, copy=True)
    nan = np.isnan(h)
    count = int(nan.sum())
    if count == 0:
        return h, 0
    if count == h.size:
        raise ValueError("the height grid has no valid cells to fill from")
    idx = ndimage.distance_transform_edt(nan, return_distances=False, return_indices=True)
    h[nan] = h[tuple(idx)][nan]
    return h, count


def encode_u16(h, z_min: float, z_max: float) -> np.ndarray:
    """Heights in meters -> uint16 with z_min at 0 and z_max at 65535 (clamped)."""
    z = np.asarray(h, dtype=np.float64)
    if np.isnan(z).any():
        raise ValueError("fill the voids before encoding")
    z = np.clip(z, z_min, z_max)
    return np.round((z - z_min) / (z_max - z_min) * 65535.0).astype(np.uint16)


def sample_height(h: np.ndarray, extent: Extent, x, y):
    """Bilinear height at level coordinates (scalar or arrays); clamps at the box edge."""
    rows, cols = h.shape
    col, row = extent.to_pixel(x, y, cols)
    col = np.clip(col, 0, cols - 1)
    row = np.clip(row, 0, rows - 1)
    c0 = np.floor(col).astype(int)
    r0 = np.floor(row).astype(int)
    c1 = np.minimum(c0 + 1, cols - 1)
    r1 = np.minimum(r0 + 1, rows - 1)
    fc, fr = col - c0, row - r0
    v = (h[r0, c0] * (1 - fc) * (1 - fr) + h[r0, c1] * fc * (1 - fr)
         + h[r1, c0] * (1 - fc) * fr + h[r1, c1] * fc * fr)
    return float(v) if np.ndim(v) == 0 else v


def write_png16(u16: np.ndarray, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(u16, dtype=np.uint16)).save(path)
    return path


def load_heights(raw: bool = False, out_dir=OUT_DIR) -> np.ndarray:
    """The height grid written by run(): raw (before water carving) or final."""
    return np.load(Path(out_dir) / ("heights_raw.npy" if raw else "heights.npy"))


def _carve_if_water(h: np.ndarray, extent: Extent, out_dir: Path) -> np.ndarray:
    water_json = out_dir / "water.json"
    if not water_json.exists():
        log.info("no water.json yet: terrain left uncarved (run water, then terrain again)")
        return h
    from . import water

    bodies = water.load_water(water_json)
    log.info("carving %d water bodies into the terrain", len(bodies))
    return water.carve(h, extent, bodies)


def run(cfg, raw_dir=None, out_dir=None) -> None:
    extent = Extent.from_config(cfg)
    n = cfg.terrain_size
    raw_dir, out_dir = Path(raw_dir or cfg.raw_dir), Path(out_dir or cfg.out_dir)
    dem = sorted((raw_dir / "dem").glob("*.tif"))
    if not dem:
        raise FileNotFoundError(f"no DEM files in {raw_dir / 'dem'}; run the fetch stage first")
    log.info("building %d x %d heights at %.2f m/px from %d DEM files",
             n, n, extent.pixel_size(n), len(dem))
    h, filled = fill_voids(build_heights(dem, extent, n))
    log.info("filled %d empty cells (%.3f %%)", filled, 100.0 * filled / h.size)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "heights_raw.npy", h)
    final = _carve_if_water(h, extent, out_dir)
    np.save(out_dir / "heights.npy", final)
    low, high = int((final < cfg.z_min).sum()), int((final > cfg.z_max).sum())
    if low or high:
        log.warning("%d cells below z_min and %d above z_max are clamped in the PNG", low, high)
    write_png16(encode_u16(final, cfg.z_min, cfg.z_max), out_dir / "heightmap.png")
    log.info("heights %.2f to %.2f m, median %.2f m", float(final.min()), float(final.max()),
             float(np.median(final)))
