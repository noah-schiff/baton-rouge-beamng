"""The 2 m analysis grid shared by buildings and trees: height above ground (ndsm) from
the lidar surface model minus the bare ground, and NDVI from NAIP red and near-infrared.

* The grid is north-up (row 0 = north) and aligned with Extent like the terrain:
  n = round(2 * half / res) cells a side, px = 2 * half / n, and a cell centre at
  -half + (col + 0.5) * px. For the full box that is 8192 x 8192 at 2 m.
* out/ndsm.npy and out/ndvi.npy: float32, shape (n, n), NaN = no data. out/canopy.json
  holds {"px": px, "n": n}, so load_canopy can hand back the cell size.
* build(cfg, raw_dir, dem_dir) -> (ndsm, ndvi):
  - the raw/dsm tiles mosaicked onto the grid (average resampling) with terrain.py's
    merge: where lidar projects overlap, the file later in name order is applied first
    and the others only fill its gaps;
  - bare ground from the raw/dem 1 m tiles onto the same grid the same way (both are
    NAVD88);
  - ndsm = dsm - dem, values < 0 set to 0, values > 150 set to NaN (outliers: the DSM
    reaches 1,000+ m in places), NaN where either is missing;
  - ndvi = (nir - red) / (nir + red) from raw/naip_rn_2m.tif, in float, NaN where both
    are 0 or missing. That file may be on a bigger box's grid (the raw folder is
    shared by the debug and full configs), so it is resampled onto this one.
* run(cfg, raw_dir=None, out_dir=None) writes the files with save_canopy and logs the
  valid share of each grid.

Readers (buildings, trees) call load_canopy(out_dir) -> (ndsm, ndvi, px). They must cope
with NaN regions (the lidar gap in the north-east quadrant): no trees are detected there
and building heights fall back to tags or defaults.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from .features import DSM_DIR, NAIP_RN_FILE
from .geo import LEVEL_CRS, Extent
from .terrain import build_heights

log = logging.getLogger(__name__)

OUT_DIR = Path("data/out")
RES = 2.0  # m, analysis cell size
MAX_HEIGHT = 150.0  # m above ground; anything higher is a lidar outlier (the Capitol is 124 m)
NDSM_FILE = "ndsm.npy"
NDVI_FILE = "ndvi.npy"
META_FILE = "canopy.json"


def analysis_grid(extent: Extent, res: float = RES) -> tuple[int, float]:
    """(cells a side, cell size in m) of the analysis grid over the box."""
    n = max(1, round(2 * extent.half_size / res))
    return n, 2 * extent.half_size / n


def save_canopy(out_dir, ndsm: np.ndarray, ndvi: np.ndarray, px: float) -> Path:
    """Write out/ndsm.npy, out/ndvi.npy (float32) and out/canopy.json; returns out_dir."""
    ndsm, ndvi = np.asarray(ndsm, np.float32), np.asarray(ndvi, np.float32)
    if ndsm.ndim != 2 or ndsm.shape[0] != ndsm.shape[1] or ndvi.shape != ndsm.shape:
        raise ValueError(f"ndsm {ndsm.shape} and ndvi {ndvi.shape} must be the same square grid")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / NDSM_FILE, ndsm)
    np.save(out_dir / NDVI_FILE, ndvi)
    (out_dir / META_FILE).write_text(json.dumps({"px": float(px), "n": int(ndsm.shape[0])}),
                                     encoding="utf-8")
    return out_dir


def load_canopy(out_dir=OUT_DIR) -> tuple[np.ndarray, np.ndarray, float]:
    """(ndsm, ndvi, px) as written by save_canopy."""
    out_dir = Path(out_dir)
    meta = json.loads((out_dir / META_FILE).read_text(encoding="utf-8"))
    return np.load(out_dir / NDSM_FILE), np.load(out_dir / NDVI_FILE), float(meta["px"])


def height_above_ground(dsm: np.ndarray, dem: np.ndarray) -> np.ndarray:
    """dsm - dem as float32: negatives set to 0, above MAX_HEIGHT set to NaN, NaN where either is."""
    h = np.asarray(dsm, np.float32) - np.asarray(dem, np.float32)
    h[h < 0] = 0.0
    h[h > MAX_HEIGHT] = np.nan
    return h


def _onto_grid(a: np.ndarray, src, extent: Extent, n: int) -> np.ndarray:
    """A float band of the open raster src (NaN = no data) resampled onto the n x n grid."""
    out = np.full((n, n), np.nan, np.float32)
    down = extent.pixel_size(n) >= min(abs(src.res[0]), abs(src.res[1]))
    reproject(a, out, src_transform=src.transform, src_crs=src.crs, src_nodata=np.nan,
              dst_transform=extent.transform(n), dst_crs=LEVEL_CRS, dst_nodata=np.nan,
              resampling=Resampling.average if down else Resampling.bilinear)
    return out


def ndvi_grid(path, extent: Extent, n: int) -> np.ndarray:
    """NDVI on the n x n grid from a red/NIR raster (bands 1 and 2); NaN where both are 0."""
    with rasterio.open(path) as src:
        red, nir = src.read(1).astype(np.float32), src.read(2).astype(np.float32)
        gap = (red == 0) & (nir == 0)
        red[gap] = nir[gap] = np.nan
        red, nir = _onto_grid(red, src, extent, n), _onto_grid(nir, src, extent, n)
    total = nir + red
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total > 0, (nir - red) / total, np.nan).astype(np.float32)


def build(cfg, raw_dir, dem_dir) -> tuple[np.ndarray, np.ndarray]:
    """(ndsm, ndvi) on the analysis grid from raw/dsm, the DEM tiles and raw/naip_rn_2m.tif."""
    extent = Extent.from_config(cfg)
    n, px = analysis_grid(extent)
    raw_dir, dem_dir = Path(raw_dir), Path(dem_dir)
    dsm_paths = sorted((raw_dir / DSM_DIR).glob("*.tif"))
    naip = raw_dir / NAIP_RN_FILE
    if not dsm_paths or not naip.exists():
        raise FileNotFoundError(f"no lidar DSM tiles in {raw_dir / DSM_DIR} or no {naip}; "
                                "run the features stage first")
    dem_paths = sorted(dem_dir.glob("*.tif"))
    if not dem_paths:
        raise FileNotFoundError(f"no DEM files in {dem_dir}; run the fetch stage first")
    log.info("canopy grid %d x %d at %.1f m from %d DSM and %d DEM files", n, n, px, len(dsm_paths),
             len(dem_paths))
    dsm = build_heights(dsm_paths, extent, n, newest=survey_year)
    ndsm = height_above_ground(dsm, build_heights(dem_paths, extent, n))
    return ndsm, ndvi_grid(naip, extent, n)


def survey_year(path) -> tuple[int, str]:
    """Sort key for 3DEP item files: the first year in the name is the survey's
    (USGS_LPC_LA_Amite_2018_LAS_2019-... was flown in 2018), so newer lidar wins overlaps."""
    m = re.search(r"(?<!\d)(19|20)\d\d(?!\d)", Path(path).name)
    return (int(m.group(0)) if m else 0, Path(path).name)


def _share(valid: np.ndarray) -> float:
    return 100.0 * float(valid.mean()) if valid.size else 0.0


def run(cfg, raw_dir=None, out_dir=None) -> None:
    raw_dir, out_dir = Path(raw_dir or cfg.raw_dir), Path(out_dir or cfg.out_dir)
    ndsm, ndvi = build(cfg, raw_dir, raw_dir / "dem")
    _, px = analysis_grid(Extent.from_config(cfg))
    save_canopy(out_dir, ndsm, ndvi, px)
    has_h, has_v = np.isfinite(ndsm), np.isfinite(ndvi)
    with np.errstate(invalid="ignore"):
        crowns = (ndsm >= 4.0) & (ndvi >= 0.25)
    log.info("height above ground valid in %.1f %% of the box (max %.1f m), NDVI in %.1f %%; "
             "%.1f %% looks like canopy (>= 4 m, NDVI >= 0.25)", _share(has_h),
             float(ndsm[has_h].max()) if has_h.any() else float("nan"), _share(has_v), _share(crowns))
