"""Helpers that write small synthetic GeoTIFFs for the raster tests."""
import numpy as np
import rasterio
from rasterio.transform import from_origin


def write_dem(path, extent, fn, res=2.0, margin=200.0, nodata=-999999.0,
              declare_nodata=True, mutate=None):
    """A float32 DEM in EPSG:26915 covering the extent plus a margin.

    fn(E, N) gives elevation from UTM easting/northing arrays. mutate(z, E, N)
    may edit z in place (to punch nodata holes, for example).
    """
    w, s, e, n = extent.bbox_utm(margin)
    width = int(round((e - w) / res))
    height = int(round((n - s) / res))
    E, N = np.meshgrid(w + (np.arange(width) + 0.5) * res, n - (np.arange(height) + 0.5) * res)
    z = np.asarray(fn(E, N), dtype="float32") * np.ones_like(E, dtype="float32")
    if mutate is not None:
        mutate(z, E, N)
    with rasterio.open(path, "w", driver="GTiff", width=width, height=height, count=1,
                       dtype="float32", crs="EPSG:26915", transform=from_origin(w, n, res, res),
                       nodata=nodata if declare_nodata else None) as dst:
        dst.write(z, 1)
    return path
