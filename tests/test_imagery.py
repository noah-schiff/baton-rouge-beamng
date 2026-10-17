import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

from brmap import imagery
from brmap.config import Config
from brmap.imagery import LAYER_NAMES, build_basecolor, build_layers, nlcd_to_layer


def _naip(path, extent, red, res=4.0, margin=100.0, bands=4, clip_east_m=None):
    """Synthetic aerial: band 1 = red(E, N), other bands 0. Like real NAIP, no alpha band."""
    w, s, e, n = extent.bbox_utm(margin)
    if clip_east_m is not None:
        e = extent.center_e + clip_east_m
    width, height = int(round((e - w) / res)), int(round((n - s) / res))
    E, N = np.meshgrid(w + (np.arange(width) + 0.5) * res, n - (np.arange(height) + 0.5) * res)
    arr = np.zeros((bands, height, width), np.uint8)
    arr[0] = np.asarray(red(E, N)) * np.ones_like(E)
    extra = {"photometric": "RGB", "ALPHA": "UNSPECIFIED"} if bands == 4 else {}
    with rasterio.open(path, "w", driver="GTiff", width=width, height=height, count=bands,
                       dtype="uint8", crs="EPSG:26915", transform=from_origin(w, n, res, res),
                       **extra) as dst:
        dst.write(arr)
    return path


def _nlcd(path, extent, north_cls, south_cls):
    """Synthetic NLCD in EPSG:5070 (like the real WCS output), split north/south at the center."""
    w, s, e, n = transform_bounds("EPSG:26915", "EPSG:5070", *extent.bbox_utm(300))
    res = 30.0
    width, height = int((e - w) / res) + 1, int((n - s) / res) + 1
    _, cy = transform_bounds("EPSG:26915", "EPSG:5070", extent.center_e, extent.center_n,
                             extent.center_e, extent.center_n)[0:2]
    ys = n - (np.arange(height) + 0.5) * res
    cls = np.where(ys[:, None] > cy, north_cls, south_cls).astype(np.uint8) * np.ones((1, width), np.uint8)
    with rasterio.open(path, "w", driver="GTiff", width=width, height=height, count=1,
                       dtype="uint8", crs="EPSG:5070", transform=from_origin(w, n, res, res)) as dst:
        dst.write(cls, 1)
    return path


def test_layer_names():
    assert LAYER_NAMES == ["grass", "dirt", "asphalt", "mud", "forest"]


def test_nlcd_mapping():
    assert nlcd_to_layer(np.array([[21, 31, 24, 95, 42, 0]])).tolist() == [[0, 1, 2, 3, 4, 0]]


@pytest.mark.parametrize("code,layer", [
    (21, 0), (22, 0), (71, 0), (81, 0), (82, 0), (31, 1), (23, 2), (24, 2), (90, 3), (95, 3),
    (11, 3), (41, 4), (42, 4), (43, 4), (52, 4), (12, 0), (250, 0)])
def test_nlcd_mapping_table(code, layer):
    assert nlcd_to_layer(np.array([code]))[0] == layer


def test_basecolor_orientation(tmp_path, small_extent):
    cn = small_extent.center_n
    p = _naip(tmp_path / "n.tif", small_extent, lambda E, N: np.where(N > cn, 255, 0))
    img = build_basecolor([p], small_extent, 32)
    assert img[0, :, 0].mean() > 200 and img[-1, :, 0].mean() < 50


def test_basecolor_shape_and_dtype(tmp_path, small_extent):
    p = _naip(tmp_path / "n.tif", small_extent, lambda E, N: 120, bands=3)
    img = build_basecolor([p], small_extent, 32)
    assert img.shape == (32, 32, 3) and img.dtype == np.uint8
    assert (img[..., 0] == 120).all() and (img[..., 1] == 0).all()


def test_basecolor_fills_uncovered_cells_from_neighbors(tmp_path, small_extent):
    # this file stops 200 m east of the center, so the east edge has no imagery
    p = _naip(tmp_path / "n.tif", small_extent, lambda E, N: 90, clip_east_m=200)
    img = build_basecolor([p], small_extent, 32)
    assert (img[..., 0] == 90).all()  # gap filled with the nearest real color, not black


def test_basecolor_mosaics_several_files(tmp_path, small_extent):
    west = _naip(tmp_path / "w.tif", small_extent, lambda E, N: 50, clip_east_m=0)
    east = _naip(tmp_path / "e.tif", small_extent, lambda E, N: 200)
    img = build_basecolor([west, east], small_extent, 32)
    assert img[16, 0, 0] == 50 or img[16, 0, 0] == 200  # a value from a real file
    assert img[16, -1, 0] == 200


def test_layers_from_nlcd_orientation(tmp_path, small_extent):
    p = _nlcd(tmp_path / "nlcd.tif", small_extent, north_cls=41, south_cls=24)
    layers = build_layers(p, small_extent, 64)
    assert layers.shape == (64, 64) and layers.dtype == np.uint8
    assert (layers[0] == 4).all() and (layers[-1] == 2).all()  # forest north, asphalt south


def test_run_writes_outputs(tmp_path, monkeypatch, small_extent):
    monkeypatch.chdir(tmp_path)
    raw = tmp_path / "data" / "raw"
    (raw / "naip").mkdir(parents=True)
    _naip(raw / "naip" / "a.tif", small_extent, lambda E, N: 77, bands=3)
    _nlcd(raw / "nlcd.tif", small_extent, 41, 82)
    imagery.run(Config(debug_half_size_m=512.0, terrain_size=256))
    out = tmp_path / "data" / "out"
    with Image.open(out / "basecolor.png") as img:
        assert img.mode == "RGB" and img.size == (256, 256)
    layers = np.load(out / "layers.npy")
    assert layers.shape == (256, 256) and set(np.unique(layers)) <= {0, 4}
