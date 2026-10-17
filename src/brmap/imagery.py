"""Stage 3: NAIP aerials -> terrain base color; NLCD land cover -> material layer map.

Both outputs use the terrain grid (n x n, row 0 = north), so pixel (row, col)
of basecolor.png and layers.npy sits on the same spot as the heightmap's.
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

LAYER_NAMES: list[str] = ["grass", "dirt", "asphalt", "mud", "forest"]
NLCD_TO_LAYER = {
    21: 0, 22: 0, 71: 0, 81: 0, 82: 0,  # developed open space, low intensity, grass, pasture, crops
    31: 1,                              # barren
    23: 2, 24: 2,                       # developed medium and high intensity
    90: 3, 95: 3, 11: 3,                # wetlands and open water (banks)
    41: 4, 42: 4, 43: 4, 52: 4,         # forest and shrub
}


def nlcd_to_layer(nlcd) -> np.ndarray:
    """NLCD class codes -> layer index into LAYER_NAMES (unknown codes -> grass)."""
    lut = np.zeros(256, np.uint8)
    for code, layer in NLCD_TO_LAYER.items():
        lut[code] = layer
    return lut[np.clip(np.asarray(nlcd), 0, 255).astype(np.uint8)]


def _fill_nearest(stack: np.ndarray) -> np.ndarray:
    """Fill NaN cells of a (bands, n, n) stack with the nearest covered cell's values."""
    nan = np.isnan(stack[0])
    if nan.all():
        raise ValueError("no imagery covers the box")
    if nan.any():
        idx = ndimage.distance_transform_edt(nan, return_distances=False, return_indices=True)
        for b in range(stack.shape[0]):
            stack[b][nan] = stack[b][tuple(idx)][nan]
        log.info("filled %d cells (%.2f %%) with no imagery from their neighbors",
                 int(nan.sum()), 100.0 * nan.sum() / nan.size)
    return stack


def build_basecolor(naip_paths, extent: Extent, n: int) -> np.ndarray:
    """Mosaic the aerials' first three bands (RGB) onto the grid: uint8 (n, n, 3), row 0 north."""
    transform = extent.transform(n)
    px = extent.pixel_size(n)
    acc = np.full((3, n, n), np.nan, np.float32)
    for p in sorted(Path(p) for p in naip_paths):
        with rasterio.open(p) as src:
            if src.count < 3:
                raise ValueError(f"{p} has {src.count} bands; expected RGB")
            down = px >= abs(src.res[0])
            layer = np.full((3, n, n), np.nan, np.float32)
            for b in range(3):
                reproject(rasterio.band(src, b + 1), layer[b], src_transform=src.transform,
                          src_crs=src.crs, src_nodata=src.nodata, dst_transform=transform,
                          dst_crs=LEVEL_CRS, dst_nodata=np.nan,
                          resampling=Resampling.average if down else Resampling.bilinear)
        new = np.isnan(acc[0]) & ~np.isnan(layer[0])
        acc[:, new] = layer[:, new]
    acc = _fill_nearest(acc)
    return np.clip(np.round(acc), 0, 255).astype(np.uint8).transpose(1, 2, 0)


def build_layers(nlcd_path, extent: Extent, n: int) -> np.ndarray:
    """NLCD classes sampled onto the grid (nearest), mapped to layer indices."""
    classes = np.zeros((n, n), np.uint8)
    with rasterio.open(nlcd_path) as src:
        reproject(rasterio.band(src, 1), classes, src_transform=src.transform, src_crs=src.crs,
                  src_nodata=src.nodata, dst_transform=extent.transform(n), dst_crs=LEVEL_CRS,
                  dst_nodata=0, resampling=Resampling.nearest)
    return nlcd_to_layer(classes)


def run(cfg, raw_dir=None, out_dir=None) -> None:
    extent = Extent.from_config(cfg)
    n = cfg.terrain_size
    raw_dir, out_dir = Path(raw_dir or cfg.raw_dir), Path(out_dir or cfg.out_dir)
    naip = sorted((raw_dir / "naip").glob("*.tif"))
    nlcd = raw_dir / "nlcd.tif"
    if not naip or not nlcd.exists():
        raise FileNotFoundError(f"missing NAIP tiles or nlcd.tif in {raw_dir}; run fetch first")
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("base color %d x %d from %d aerial files", n, n, len(naip))
    Image.fromarray(build_basecolor(naip, extent, n)).save(out_dir / "basecolor.png")
    layers = build_layers(nlcd, extent, n)
    np.save(out_dir / "layers.npy", layers)
    counts = np.bincount(layers.ravel(), minlength=len(LAYER_NAMES))
    log.info("layers: %s", ", ".join(f"{name} {100.0 * c / layers.size:.1f}%"
                                     for name, c in zip(LAYER_NAMES, counts)))
