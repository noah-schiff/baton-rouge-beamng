"""Level coordinates: meters east/north of the map center in NAD83 / UTM 15N.

+x is east, +y is north, z is elevation (NAVD88 m). Raster grids are north-up:
row 0 is the north edge and pixel (row, col) has its center at
x = -half + (col + 0.5) * px, y = half - (row + 0.5) * px.
"""
from __future__ import annotations

import numpy as np
from affine import Affine
from pyproj import Transformer

LEVEL_CRS = "EPSG:26915"
LONLAT_CRS = "EPSG:4326"


def _shift(v, d: float):
    """v + d for a scalar (returns float) or an array-like (returns ndarray)."""
    return np.asarray(v, dtype=float) + d if np.ndim(v) else float(v) + d


class Extent:
    def __init__(self, center_lon: float, center_lat: float, half_size: float):
        if half_size <= 0:
            raise ValueError("half_size must be positive")
        self.center_lon = float(center_lon)
        self.center_lat = float(center_lat)
        self.half_size = float(half_size)
        self._to_utm = Transformer.from_crs(LONLAT_CRS, LEVEL_CRS, always_xy=True)
        self._to_ll = Transformer.from_crs(LEVEL_CRS, LONLAT_CRS, always_xy=True)
        self.center_e, self.center_n = self._to_utm.transform(self.center_lon, self.center_lat)

    @classmethod
    def from_config(cls, cfg) -> "Extent":
        half = cfg.debug_half_size_m if cfg.debug_half_size_m else cfg.half_size_m
        return cls(cfg.center_lon, cfg.center_lat, half)

    def __repr__(self) -> str:
        return f"Extent({self.center_lon}, {self.center_lat}, half_size={self.half_size})"

    # lon/lat <-> level
    def to_level(self, lon, lat):
        e, n = self._to_utm.transform(lon, lat)
        return self.utm_to_level(e, n)

    def to_lonlat(self, x, y):
        return self._to_ll.transform(*self.level_to_utm(x, y))

    # UTM <-> level
    def utm_to_level(self, e, n):
        return _shift(e, -self.center_e), _shift(n, -self.center_n)

    def level_to_utm(self, x, y):
        return _shift(x, self.center_e), _shift(y, self.center_n)

    # boxes
    def bbox_utm(self, buffer_m: float = 0.0):
        r = self.half_size + buffer_m
        return (self.center_e - r, self.center_n - r, self.center_e + r, self.center_n + r)

    def bbox_lonlat(self, buffer_m: float = 0.0):
        """(west, south, east, north) in degrees, covering the whole buffered square."""
        return self._to_ll.transform_bounds(*self.bbox_utm(buffer_m), densify_pts=21)

    def contains(self, x, y):
        h = self.half_size + 1e-6
        return (np.abs(x) <= h) & (np.abs(y) <= h)

    # grids
    def pixel_size(self, n: int) -> float:
        return 2.0 * self.half_size / n

    def transform(self, n: int) -> Affine:
        """North-up UTM transform of an n x n grid covering the box (origin = NW corner)."""
        px = self.pixel_size(n)
        return Affine(px, 0.0, self.center_e - self.half_size,
                      0.0, -px, self.center_n + self.half_size)

    def level_transform(self, n: int) -> Affine:
        """Like transform() but in level coordinates (origin at the map center)."""
        px = self.pixel_size(n)
        return Affine(px, 0.0, -self.half_size, 0.0, -px, self.half_size)

    def pixel_centers(self, n: int):
        """Level x of each column center and level y of each row center."""
        px = self.pixel_size(n)
        idx = np.arange(n) + 0.5
        return -self.half_size + idx * px, self.half_size - idx * px

    def to_pixel(self, x, y, n: int):
        """Fractional (col, row) with pixel centers at whole numbers."""
        px = self.pixel_size(n)
        return (np.asarray(x) + self.half_size) / px - 0.5, (self.half_size - np.asarray(y)) / px - 0.5
