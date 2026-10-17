import numpy as np
import pytest
from PIL import Image

from brmap import terrain
from brmap.config import Config
from brmap.terrain import build_heights, encode_u16, fill_voids, sample_height
from rasters import write_dem


def test_north_row_is_row_zero(tmp_path, small_extent):
    base = small_extent.center_n - 600
    dem = write_dem(tmp_path / "dem.tif", small_extent, lambda E, N: N - base)
    h = build_heights([dem], small_extent, 64)
    assert h.shape == (64, 64) and h.dtype == np.float32
    assert h[0].mean() > h[-1].mean()
    # row 0 center sits 8 m below the north edge (16 m pixels)
    assert h[0, 32] == pytest.approx(small_extent.center_n + 512 - 8 - base, abs=0.5)


def test_west_column_is_column_zero(tmp_path, small_extent):
    dem = write_dem(tmp_path / "dem.tif", small_extent, lambda E, N: E - small_extent.center_e)
    h = build_heights([dem], small_extent, 64)
    assert h[:, 0].mean() < h[:, -1].mean()


def test_sample_height_matches_grid_orientation(small_extent):
    h = np.zeros((64, 64), np.float32)
    h[:32, 32:] = 10.0  # north-east quadrant
    half = small_extent.half_size
    assert sample_height(h, small_extent, 0.5 * half, 0.5 * half) == pytest.approx(10.0)
    assert sample_height(h, small_extent, -0.5 * half, 0.5 * half) == pytest.approx(0.0)
    assert sample_height(h, small_extent, 0.5 * half, -0.5 * half) == pytest.approx(0.0)


def test_sample_height_is_bilinear_and_vectorized(small_extent):
    xs, ys = small_extent.pixel_centers(64)
    h = np.tile(np.arange(64, dtype=np.float32), (64, 1))  # value = column index
    mid_x = (xs[10] + xs[11]) / 2
    assert sample_height(h, small_extent, mid_x, ys[5]) == pytest.approx(10.5)
    out = sample_height(h, small_extent, np.array([xs[3], xs[60]]), np.array([ys[0], ys[0]]))
    np.testing.assert_allclose(out, [3.0, 60.0])
    # outside the box clamps to the edge instead of failing
    assert sample_height(h, small_extent, 10 * small_extent.half_size, 0.0) == pytest.approx(63.0)


def _hole(z, E, N, ext, size):
    nw_e, nw_n = ext.center_e - ext.half_size, ext.center_n + ext.half_size
    z[(E > nw_e) & (E < nw_e + size) & (N < nw_n) & (N > nw_n - size)] = -999999.0


@pytest.mark.parametrize("declared", [True, False])
def test_nodata_not_in_range(tmp_path, small_extent, declared):
    dem = write_dem(tmp_path / "dem.tif", small_extent, lambda E, N: 5.0 + 0 * E,
                    declare_nodata=declared, mutate=lambda z, E, N: _hole(z, E, N, small_extent, 40))
    h = build_heights([dem], small_extent, 64)
    assert np.nanmin(h) > -10            # nodata is never averaged into a height
    assert np.isnan(h).sum() >= 4        # the 40 m hole shows up as NaN cells
    filled, count = fill_voids(h)
    assert np.isfinite(filled).all() and filled.min() > -10
    assert count == np.isnan(h).sum()


def test_newest_dem_wins_and_older_fills_holes(tmp_path, small_extent):
    newer = write_dem(tmp_path / "2023-10-30_new.tif", small_extent, lambda E, N: 5.0 + 0 * E,
                      mutate=lambda z, E, N: _hole(z, E, N, small_extent, 200))
    older = write_dem(tmp_path / "2019-01-01_old.tif", small_extent, lambda E, N: 9.0 + 0 * E)
    h = build_heights([older, newer], small_extent, 64)  # order given doesn't matter
    assert h[32, 32] == pytest.approx(5.0)
    assert h[2, 2] == pytest.approx(9.0)    # inside the newer file's hole
    assert np.isfinite(h).all()


def test_fill_voids_counts():
    h = np.ones((5, 5), np.float32)
    h[0, 0] = h[2, 3] = h[4, 4] = np.nan
    filled, count = fill_voids(h)
    assert count == 3 and np.isfinite(filled).all() and (filled == 1).all()
    assert np.isnan(h).sum() == 3  # input left untouched


def test_fill_voids_uses_nearest_value():
    h = np.full((1, 6), np.nan, np.float32)
    h[0, 0], h[0, 5] = 1.0, 7.0
    filled, _ = fill_voids(h)
    assert filled.tolist() == [[1, 1, 1, 7, 7, 7]]


def test_fill_voids_all_nan_raises():
    with pytest.raises(ValueError):
        fill_voids(np.full((3, 3), np.nan, np.float32))


def test_encode_u16_endpoints():
    assert encode_u16(np.array([[-5.0, 40.0]]), -5, 40).tolist() == [[0, 65535]]


def test_encode_u16_clamps():
    assert encode_u16(np.array([[-50.0, 99.0]]), -5, 40).tolist() == [[0, 65535]]


def test_encode_precision():
    z = np.linspace(-5, 40, 100001)
    u = encode_u16(z, -5, 40)
    assert u.dtype == np.uint16
    decoded = -5 + u.astype(np.float64) / 65535 * 45
    assert np.abs(decoded - z).max() < 0.0004


def test_png_roundtrip(tmp_path):
    u = (np.arange(64 * 64, dtype=np.uint32) * 16 % 65536).astype(np.uint16).reshape(64, 64)
    terrain.write_png16(u, tmp_path / "h.png")
    with Image.open(tmp_path / "h.png") as im:
        back = np.array(im).astype(np.uint16)
    np.testing.assert_array_equal(back, u)


def test_run_writes_outputs(tmp_path, monkeypatch, small_extent):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "raw" / "dem").mkdir(parents=True)
    write_dem(tmp_path / "data" / "raw" / "dem" / "2023-01-01_a.tif", small_extent,
              lambda E, N: 3.0 + 0 * E)
    cfg = Config(debug_half_size_m=512.0, terrain_size=256)
    terrain.run(cfg)
    out = tmp_path / "data" / "out"
    raw, final = np.load(out / "heights_raw.npy"), np.load(out / "heights.npy")
    assert raw.shape == (256, 256) and np.allclose(raw, 3.0) and np.array_equal(raw, final)
    with Image.open(out / "heightmap.png") as im:
        png = np.array(im)
    assert png.shape == (256, 256)
    assert int(png[0, 0]) == int(encode_u16(np.array(3.0), -5, 40))
    np.testing.assert_array_equal(terrain.load_heights(raw=True, out_dir=out), raw)
