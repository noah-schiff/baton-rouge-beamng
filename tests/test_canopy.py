"""canopy: the 2 m grid of height above ground (lidar DSM minus the DEM) and NDVI (NAIP)."""
import numpy as np
import pytest
import rasterio

from brmap import canopy
from brmap.config import Config
from rasters import write_dem


def test_analysis_grid_sizes(extent, small_extent):
    assert canopy.analysis_grid(extent) == (8192, 2.0)
    assert canopy.analysis_grid(small_extent) == (512, 2.0)
    n, px = canopy.analysis_grid(small_extent, res=4.0)
    assert (n, px) == (256, 4.0) and px == small_extent.pixel_size(n)  # the terrain's cell layout


def _ground(E, N, ext):
    return 5.0 + 0.01 * (N - ext.center_n) + 0.005 * (E - ext.center_e)


def _square(E, N, ext, x0, x1, y0, y1):
    x, y = E - ext.center_e, N - ext.center_n
    return (x >= x0) & (x < x1) & (y >= y0) & (y < y1)


def _write_rn(path, ext, red, nir):
    n = red.shape[0]
    with rasterio.open(path, "w", driver="GTiff", width=n, height=n, count=2, dtype="uint8", crs="EPSG:26915",
                       transform=ext.transform(n), nodata=0) as dst:
        dst.write(np.stack([red, nir]).astype(np.uint8))
    return path


def _inputs(tmp_path, ext, dsm_fn, dsm_mutate=None):
    raw = tmp_path / "raw"
    (raw / "dem").mkdir(parents=True)
    (raw / "dsm").mkdir()
    write_dem(raw / "dem" / "2023-01-01_dem.tif", ext, lambda E, N: _ground(E, N, ext), res=1.0)
    write_dem(raw / "dsm" / "dsm-a.tif", ext, dsm_fn, res=2.0, nodata=-9999.0, mutate=dsm_mutate)
    # NAIP: vegetation (NDVI 0.5) west of x = 0, bare (NDVI 0) east of it, a gap (both 0) in the south-east
    red = np.full((512, 512), 100, np.uint8)
    nir = np.full((512, 512), 100, np.uint8)
    red[:, :256], nir[:, :256] = 50, 150
    red[400:, 400:] = nir[400:, 400:] = 0
    _write_rn(raw / "naip_rn_2m.tif", ext, red, nir)
    return raw


def test_newest_lidar_wins_where_surveys_overlap(tmp_path, small_extent):
    # 3DEP item ids carry the survey year; where two surveys overlap the newer one must win,
    # whatever the alphabetical order of the names (here the 2017 one sorts last)
    ext = small_extent
    raw = _inputs(tmp_path, ext, lambda E, N: _ground(E, N, ext) + 3.0)
    (raw / "dsm" / "dsm-a.tif").unlink()
    write_dem(raw / "dsm" / "USGS_LPC_LA_UpperDeltaPlain_2017_LAS_2018-dsm-2m-3-12.tif", ext,
              lambda E, N: _ground(E, N, ext) + 3.0, res=2.0, nodata=-9999.0)
    write_dem(raw / "dsm" / "LA_CPRA_B5_2019-dsm-2m-26-5.tif", ext,
              lambda E, N: _ground(E, N, ext) + 8.0, res=2.0, nodata=-9999.0)
    cfg = Config(debug_half_size_m=512.0, raw_dir=str(raw), out_dir=str(tmp_path / "out"))
    canopy.run(cfg)
    ndsm, _, _ = canopy.load_canopy(tmp_path / "out")
    np.testing.assert_allclose(ndsm[10:500, 10:500], 8.0, atol=0.01)


def test_canopy_ndsm_and_ndvi(tmp_path, small_extent):
    ext = small_extent

    def dsm(E, N):
        z = _ground(E, N, ext) - 0.5                                     # just under the DEM: clamped to 0
        z = np.where(_square(E, N, ext, -100, 100, 200, 400), _ground(E, N, ext) + 10.0, z)   # a 10 m crown
        z = np.where(_square(E, N, ext, 200, 260, -300, -240), _ground(E, N, ext) + 400.0, z)  # an outlier
        return z

    raw = _inputs(tmp_path, ext, dsm)
    cfg = Config(debug_half_size_m=512.0, raw_dir=str(raw), out_dir=str(tmp_path / "out"))
    canopy.run(cfg)
    ndsm, ndvi, px = canopy.load_canopy(tmp_path / "out")
    assert ndsm.shape == ndvi.shape == (512, 512) and ndsm.dtype == ndvi.dtype == np.float32 and px == 2.0

    # x -100..100 is columns 206..305 and y 400..200 rows 56..155: row 0 is the north edge
    crown, outlier = np.zeros((512, 512), bool), np.zeros((512, 512), bool)
    crown[56:156, 206:306] = True
    outlier[376:406, 356:386] = True
    np.testing.assert_allclose(ndsm[crown], 10.0, atol=0.01)
    assert np.isnan(ndsm[outlier]).all()                                  # over 150 m: not a real height
    assert (ndsm[~crown & ~outlier] == 0.0).all()

    np.testing.assert_allclose(ndvi[:400, :256], 0.5, atol=1e-6)
    np.testing.assert_allclose(ndvi[:, 256:400], 0.0, atol=1e-6)
    gap = np.zeros((512, 512), bool)
    gap[400:, 400:] = True
    assert np.isnan(ndvi[gap]).all() and np.isfinite(ndvi[~gap]).all()   # red and NIR both 0: no data


def test_canopy_nan_where_dsm_missing(tmp_path, small_extent):
    ext = small_extent

    def gap(z, E, N):  # like the lidar gap: the whole north-east quadrant is nodata
        z[_square(E, N, ext, 0, 2000, 0, 2000)] = -9999.0

    raw = _inputs(tmp_path, ext, lambda E, N: _ground(E, N, ext) + 3.0, dsm_mutate=gap)
    ndsm, ndvi = canopy.build(Config(debug_half_size_m=512.0), raw, raw / "dem")
    ne = np.zeros((512, 512), bool)
    ne[:256, 256:] = True
    assert np.isnan(ndsm[ne]).all()
    np.testing.assert_allclose(ndsm[~ne], 3.0, atol=0.01)
    assert np.isfinite(ndvi[ne]).all()  # NDVI doesn't depend on the lidar


def test_canopy_without_inputs_says_what_to_run(tmp_path, small_extent):
    cfg = Config(debug_half_size_m=512.0)
    with pytest.raises(FileNotFoundError, match="features"):
        canopy.build(cfg, tmp_path / "empty", tmp_path / "empty" / "dem")
    raw = _inputs(tmp_path, small_extent, lambda E, N: _ground(E, N, small_extent))
    with pytest.raises(FileNotFoundError, match="fetch"):
        canopy.build(cfg, raw, tmp_path / "no_dem")
    (raw / "naip_rn_2m.tif").unlink()
    with pytest.raises(FileNotFoundError, match="features"):
        canopy.build(cfg, raw, raw / "dem")
