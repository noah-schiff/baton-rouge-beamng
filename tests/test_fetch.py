import re
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
import requests
import responses
from pyproj import Transformer
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin

from brmap import fetch
from brmap.fetch import FetchError, clip_raster, dem_candidates, download

URL = "https://example.test/data.bin"


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr(fetch, "BACKOFF", [0, 0, 0, 0])


@responses.activate
def test_download_skips_cached(tmp_path):
    dest = tmp_path / "a.bin"
    dest.write_bytes(b"cached")
    assert download(URL, dest) == dest
    assert len(responses.calls) == 0 and dest.read_bytes() == b"cached"


@responses.activate
def test_download_retries_then_succeeds(tmp_path):
    responses.add(responses.GET, URL, status=503)
    responses.add(responses.GET, URL, status=503)
    responses.add(responses.GET, URL, body=b"payload", status=200)
    dest = download(URL, tmp_path / "a.bin")
    assert dest.read_bytes() == b"payload" and len(responses.calls) == 3


@responses.activate
def test_download_gives_up_with_url_in_error(tmp_path):
    for _ in range(5):
        responses.add(responses.GET, URL, status=503)
    with pytest.raises(FetchError, match=re.escape(URL)):
        download(URL, tmp_path / "a.bin", retries=4)
    assert len(responses.calls) == 5


@responses.activate
def test_download_client_error_not_retried(tmp_path):
    responses.add(responses.GET, URL, status=404)
    with pytest.raises(FetchError, match="404"):
        download(URL, tmp_path / "a.bin")
    assert len(responses.calls) == 1


@responses.activate
def test_download_posts_when_data_given(tmp_path):
    responses.add(responses.POST, URL, body=b"{}", status=200)
    download(URL, tmp_path / "osm.json", data={"data": "query"})
    assert responses.calls[0].request.method == "POST"


@responses.activate
def test_download_sends_descriptive_user_agent(tmp_path):
    # overpass-api.de answers 406 to generic clients such as python-requests
    responses.add(responses.GET, URL, body=b"ok", status=200)
    download(URL, tmp_path / "a.bin")
    ua = responses.calls[0].request.headers["User-Agent"]
    assert ua.startswith("brmap/") and "python-requests" not in ua


@responses.activate
def test_fetch_osm_falls_back_to_mirror(tmp_path, extent):
    from brmap.config import Config

    cfg = Config()
    cfg.sources["overpass"] = "https://primary.test/api/interpreter"
    cfg.sources["overpass_fallback"] = "https://mirror.test/api/interpreter"
    responses.add(responses.POST, cfg.sources["overpass"], status=406)
    responses.add(responses.POST, cfg.sources["overpass_fallback"],
                  json={"elements": [{"type": "way", "id": 1}]}, status=200)
    out = fetch.fetch_osm_overpass(cfg, extent, tmp_path)
    assert out.exists() and len(responses.calls) == 2


FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"
BOX = (-91.25, 30.35, -91.07, 30.51)


def test_extract_keeps_roads_and_water_in_box():
    j = fetch.osm_json_from_extract(FIXTURES / "extract_small.osm", BOX)
    ways = {e["id"]: e for e in j["elements"] if e["type"] == "way"}
    assert set(ways) == {100, 102, 103}   # 101 is outside, 104 is a building
    assert ways[100]["tags"]["name"] == "Test Street"
    assert ways[100]["geometry"][0] == {"lat": 30.43, "lon": -91.17}
    assert len(ways[102]["geometry"]) == 2  # crossing the edge: kept whole
    assert ways[103]["nodes"][0] == ways[103]["nodes"][-1] == 10


def test_extract_keeps_water_relations_with_member_geometry():
    j = fetch.osm_json_from_extract(FIXTURES / "extract_small.osm", BOX)
    rels = {e["id"]: e for e in j["elements"] if e["type"] == "relation"}
    assert set(rels) == {200}  # 201 lies far outside the box
    members = rels[200]["members"]
    assert [(m["ref"], m["role"]) for m in members] == [(110, "outer"), (111, "outer"), (112, "inner")]
    assert members[2]["geometry"][0] == {"lat": 30.444, "lon": -91.196}
    assert rels[200]["tags"]["name"] == "Mississippi River"


def test_fetch_osm_prefers_extract_and_falls_back(tmp_path, extent, monkeypatch):
    from brmap.config import Config

    calls = []

    def boom(*a):
        calls.append("geofabrik")
        raise FetchError("extract unavailable")

    monkeypatch.setattr(fetch, "fetch_osm_extract", boom)
    monkeypatch.setattr(fetch, "fetch_osm_overpass",
                        lambda cfg, ext, raw: calls.append("overpass") or (raw / "osm.json"))
    assert fetch.fetch_osm(Config(), extent, tmp_path) == tmp_path / "osm.json"
    assert calls == ["geofabrik", "overpass"]


@responses.activate
def test_partial_file_not_left_behind(tmp_path, monkeypatch):
    responses.add(responses.GET, URL, body=b"x" * 100, status=200)

    def dropping(self, chunk_size=1, decode_unicode=False):
        yield b"x" * 10
        raise requests.exceptions.ChunkedEncodingError("connection dropped")

    monkeypatch.setattr(requests.models.Response, "iter_content", dropping)
    dest = tmp_path / "a.bin"
    with pytest.raises(FetchError):
        download(URL, dest, retries=1)
    assert not dest.exists()
    assert list(tmp_path.iterdir()) == []  # the .part file was removed too


def test_overpass_query_uses_buffered_bbox(extent):
    q = fetch.overpass_query(extent, 500)
    s, w, n, e = map(float, re.search(r"\(([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+)\)", q).groups())
    assert s < 30.358 and n > 30.502 and w < -91.243 and e > -91.077
    assert '"highway"' in q and '"natural"="water"' in q and '"waterway"="riverbank"' in q
    assert q.rstrip().endswith("out body geom;")


TNM_ITEMS = [
    {"title": "USGS 1 Meter 15 x67y338 LA_2021GreaterNewOrleans_C22", "publicationDate": "2023-10-30",
     "downloadURL": "https://h/Projects/GNO/TIFF/USGS_1M_15_x67y338_GNO.tif", "format": "GeoTIFF"},
    {"title": "USGS 1 Meter 15 x67y338 LA_CPRA_2019_C20", "publicationDate": "2023-06-09",
     "downloadURL": "https://h/Projects/CPRA/TIFF/USGS_1M_15_x67y338_CPRA.tif", "format": "GeoTIFF"},
    {"title": "USGS one meter x68y337 LA Amite 2018", "publicationDate": "2020-03-30",
     "downloadURL": "https://h/Projects/Amite/TIFF/USGS_one_meter_x68y337_LA_Amite_2018.tif",
     "format": "GeoTIFF"},
    {"title": "duplicate listing", "publicationDate": "2023-10-30",
     "downloadURL": "https://h/Projects/GNO/TIFF/USGS_1M_15_x67y338_GNO.tif", "format": "GeoTIFF"},
]


def test_dem_candidates_keep_every_project_newest_first():
    c = dem_candidates(TNM_ITEMS)
    assert [x.url.split("/")[-1] for x in c] == [
        "USGS_1M_15_x67y338_GNO.tif", "USGS_1M_15_x67y338_CPRA.tif",
        "USGS_one_meter_x68y337_LA_Amite_2018.tif"]
    assert c[0].filename.startswith("2023-10-30_") and c[-1].filename.startswith("2020-03-30_")
    assert sorted(x.filename for x in c) == sorted({x.filename for x in c})  # unique


def _write_raster(path, data, x0, y0, res, crs="EPSG:26915", nodata=None):
    count = 1 if data.ndim == 2 else data.shape[0]
    arr = data[None] if data.ndim == 2 else data
    # Like real NAIP: R, G, B, NIR with the 4th band NOT flagged as alpha.
    extra = {"photometric": "RGB", "ALPHA": "UNSPECIFIED"} if count == 4 else {}
    with rasterio.open(path, "w", driver="GTiff", width=arr.shape[2], height=arr.shape[1],
                       count=count, dtype=arr.dtype, crs=crs, nodata=nodata,
                       transform=from_origin(x0, y0, res, res), **extra) as dst:
        dst.write(arr)
    if count == 4:
        with rasterio.open(path) as ds:
            assert ds.colorinterp[3] != ColorInterp.alpha


def test_clip_raster_reads_only_the_window(tmp_path):
    data = np.arange(100 * 100, dtype="float32").reshape(100, 100)
    src = tmp_path / "src.tif"
    _write_raster(src, data, 600000, 3400100, 1.0, nodata=-999999)
    out = clip_raster(src, (600010, 3400050, 600030, 3400090), tmp_path / "clip.tif")
    with rasterio.open(out) as ds:
        assert ds.width == 20 and ds.height == 40
        assert ds.bounds.left == 600010 and ds.bounds.top == 3400090
        assert ds.nodata == -999999
        np.testing.assert_array_equal(ds.read(1), data[10:50, 10:30])


def test_clip_raster_outside_returns_none(tmp_path):
    src = tmp_path / "src.tif"
    _write_raster(src, np.zeros((10, 10), "float32"), 600000, 3400010, 1.0)
    assert clip_raster(src, (700000, 3500000, 700100, 3500100), tmp_path / "c.tif") is None
    assert not (tmp_path / "c.tif").exists()


def test_clip_raster_decimates_to_resolution_and_bands(tmp_path):
    rgbn = np.zeros((4, 80, 80), "uint8")
    rgbn[0] = 200  # red
    src = tmp_path / "naip.tif"
    _write_raster(src, rgbn, 600000, 3400080, 0.5)
    out = clip_raster(src, (600000, 3400000, 600040, 3400080), tmp_path / "c.tif",
                      res=2.0, bands=[1, 2, 3])
    with rasterio.open(out) as ds:
        assert ds.count == 3 and ds.res == (2.0, 2.0) and ds.shape == (20, 20)
        assert ds.read(1).mean() == 200 and ds.read(3).max() == 0


def test_nlcd_params_cover_box_in_albers(extent):
    params = dict((k, v) for k, v in fetch.nlcd_params(extent, 500) if k != "subset")
    subsets = [v for k, v in fetch.nlcd_params(extent, 500) if k == "subset"]
    (x0, x1), (y0, y1) = [tuple(map(float, re.findall(r"[-\d.]+", s))) for s in subsets]
    ce, cn = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True).transform(-91.16, 30.43)
    assert x0 < ce < x1 and y0 < cn < y1 and (x1 - x0) > 17000
    assert params["request"] == "GetCoverage" and params["format"] == "image/geotiff"


def test_cache_cleared_when_box_grows(tmp_path):
    raw = tmp_path / "raw"
    (raw / "dem").mkdir(parents=True)
    small, big = (0, 0, 10, 10), (-5, -5, 20, 20)
    fetch.check_cache_box(raw, small)
    (raw / "dem" / "a.tif").write_bytes(b"x")
    (raw / "osm.json").write_bytes(b"{}")
    fetch.check_cache_box(raw, small)          # same box: cache kept
    assert (raw / "dem" / "a.tif").exists()
    fetch.check_cache_box(raw, (1, 1, 9, 9))   # smaller box: still covered
    assert (raw / "osm.json").exists()
    fetch.check_cache_box(raw, big)            # bigger box: cache cleared
    assert not (raw / "dem" / "a.tif").exists() and not (raw / "osm.json").exists()


@responses.activate
def test_fetch_dem_downloads_clips_and_cleans_up(tmp_path, small_extent, monkeypatch):
    from brmap.config import Config
    from rasters import write_dem

    inside = write_dem(tmp_path / "inside.tif", small_extent, lambda E, N: 4.0 + 0 * E)
    far = small_extent.__class__(-92.5, 31.5, 512.0)  # a DEM nowhere near the box
    outside = write_dem(tmp_path / "outside.tif", far, lambda E, N: 9.0 + 0 * E)
    urls = {"https://h/a/USGS_in.tif": inside, "https://h/b/USGS_out.tif": outside}
    for url, path in urls.items():
        responses.add(responses.GET, url, body=path.read_bytes(), status=200)
    monkeypatch.setattr(fetch, "_tnm_items", lambda cfg, ext: [
        {"downloadURL": u, "publicationDate": "2023-01-01", "title": u} for u in urls])
    raw = tmp_path / "raw"
    out = fetch.fetch_dem(Config(debug_half_size_m=512.0), small_extent, raw)
    assert [p.name for p in out] == ["2023-01-01_USGS_in.tif"]
    assert (raw / "dem" / "2023-01-01_USGS_out.tif.empty").exists()
    assert list((raw / "dem_full").glob("*.tif")) == []  # full tiles deleted after clipping
    with rasterio.open(out[0]) as ds:
        assert ds.read(1).max() == 4.0


def test_latest_naip_items_only():
    items = [SimpleNamespace(id=f"i{y}{k}", properties={"naip:year": str(y)})
             for y in (2021, 2023) for k in range(2)]
    assert [i.id for i in fetch.latest_naip(items)] == ["i20230", "i20231"]
