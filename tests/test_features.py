"""features: OSM footprints and nodes, Microsoft footprints, the lidar DSM and NAIP red/NIR."""
import gzip
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import planetary_computer
import pytest
import rasterio
import responses
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds
from shapely.geometry import Polygon

from brmap import features, fetch
from brmap.config import Config
from brmap.geo import Extent
from rasters import write_dem

FIX = Path(__file__).parent / "fixtures"
OSM = FIX / "extract_features_small.osm"
LINKS = "https://ms.test/global-buildings/dataset-links.csv"


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr(fetch, "BACKOFF", [0, 0, 0, 0])


@pytest.fixture
def responses_mock():
    with responses.RequestsMock() as rsps:
        yield rsps


def _level(extent, ring):
    x, y = extent.to_level(*np.asarray(ring, dtype=float).T)
    return np.column_stack([x, y])


# ---------------------------------------------------------------- OSM


def test_extract_buildings_ways_and_multipolygon(extent):
    j = features.extract_osm_features(OSM, extent, 500.0)
    b = {x["id"]: x for x in j["buildings"]}
    # w301 is a building:part, w302 lies far away, w304 isn't closed, w312 is r401's outer
    # (tagged as a building as well, so it would be a second copy), r403 is grass
    assert set(b) == {"w300", "w303", "r400", "r401", "r402"}
    assert b["w300"]["tags"] == {"building": "house"}
    (house,) = b["w300"]["polygons"]
    assert house["inner"] == [] and len(house["outer"]) == 5 and house["outer"][0] == house["outer"][-1]
    poly = Polygon(_level(extent, house["outer"]))
    assert poly.area == pytest.approx(120.0, rel=0.01)
    np.testing.assert_allclose(poly.centroid.coords[0], (106.0, 205.0), atol=0.1)

    mall = b["r400"]
    assert mall["tags"]["building"] == "commercial" and mall["tags"]["name"] == "Test Mall"
    (part,) = mall["polygons"]
    (hole,) = part["inner"]
    assert Polygon(_level(extent, part["outer"])).area == pytest.approx(1600.0, rel=0.01)
    assert Polygon(_level(extent, hole)).area == pytest.approx(100.0, rel=0.01)
    assert len(b["r402"]["polygons"]) == 2 and b["r401"]["tags"]["building:levels"] == "3"

    # w303's centroid is 250 m north of the box: only kept with the buffer
    assert "w303" not in {x["id"] for x in features.extract_osm_features(OSM, extent, 0.0)["buildings"]}


def test_extract_tree_and_signal_nodes(extent):
    j = features.extract_osm_features(OSM, extent, 500.0)
    assert [(t["id"], t["tags"]) for t in j["trees"]] == [
        ("n500", {"natural": "tree", "leaf_type": "broadleaved"})]  # n505 is far away
    x, y = extent.to_level(j["trees"][0]["lon"], j["trees"][0]["lat"])
    assert (x, y) == (pytest.approx(50.0, abs=0.05), pytest.approx(50.0, abs=0.05))
    # bus stops and peaks are ignored, and so is a crossing 1 km outside the box
    assert {n["id"]: n["tags"] for n in j["nodes"]} == {
        "n501": {"highway": "traffic_signals"},
        "n502": {"highway": "stop"},
        "n503": {"highway": "crossing", "crossing": "marked", "crossing:markings": "zebra"},
    }
    assert json.loads(json.dumps(j)) == j  # plain JSON all the way down


def _sha1(obj) -> str:
    return hashlib.sha1(json.dumps(obj).encode("utf-8")).hexdigest()


def test_existing_keys_unchanged(extent, small_extent):
    # buildings, trees and nodes of the old fixture are byte for byte what the code gave before
    # waterways, wetlands and areas were added (the SHA-1s of json.dumps of each list)
    golden = {
        (8192.0, 500.0): {"buildings": (5, "b6215acb323477cb62622fba945874597c341d4f"),
                          "trees": (1, "f76911fef4503adfa3de8f26c0bda510676cdef6"),
                          "nodes": (3, "fc8b7b0ae4a0b09ebf9fca1b4ba3aa05e5f2bd5e")},
        (512.0, 100.0): {"buildings": (4, "30bb841686dbfd36fcea977757c19016d1bc7ea3"),
                         "trees": (1, "f76911fef4503adfa3de8f26c0bda510676cdef6"),
                         "nodes": (3, "fc8b7b0ae4a0b09ebf9fca1b4ba3aa05e5f2bd5e")},
    }
    for ext, buf in ((extent, 500.0), (small_extent, 100.0)):
        out = features.extract_osm_features(OSM, ext, buf)
        for key, (n, sha) in golden[(ext.half_size, buf)].items():
            assert (len(out[key]), _sha1(out[key])) == (n, sha), key
    # the one building:part of that fixture (w301) is an area now, and still no building
    out = features.extract_osm_features(OSM, extent, 500.0)
    assert [a["id"] for a in out["areas"]] == ["w301"] and "w301" not in {b["id"] for b in out["buildings"]}
    assert out["waterways"] == [] and out["wetlands"] == []  # r403 is grass, not a wetland


def test_extract_waterways(small_extent):
    out = features.extract_osm_features(FIX / "extract_v02_small.osm", small_extent, 100.0)
    ww = {w["id"]: w for w in out["waterways"]}
    assert {w["kind"] for w in ww.values()} == {"stream"} and len(ww) == 3
    assert sum(w["culvert"] for w in ww.values()) == 1
    assert all(len(w["coords"]) >= 2 for w in ww.values())
    # w650 is a stream far outside the box and w610 a river bank (an area)
    assert set(ww) == {"w600", "w601", "w602"}
    assert ww["w601"]["culvert"] and ww["w601"]["name"] is None
    assert ww["w600"]["name"] == "Test Creek" and ww["w600"]["culvert"] is False
    # ordered along the way, in [lon, lat], and the middle way starts where the first one ends
    assert ww["w600"]["coords"][-1] == ww["w601"]["coords"][0]
    x, y = small_extent.to_level(*np.asarray(ww["w600"]["coords"], dtype=float).T)
    np.testing.assert_allclose(x, [-450, -300, -150], atol=0.05)
    np.testing.assert_allclose(y, [120, 140, 110], atol=0.05)
    assert json.loads(json.dumps(out)) == out


def test_extract_wetlands_and_areas(small_extent):
    out = features.extract_osm_features(FIX / "extract_v02_small.osm", small_extent, 100.0)
    assert [w["wetland"] for w in out["wetlands"]] == ["swamp"]
    tags = {a["id"]: a["tags"] for a in out["areas"]}
    assert any(t.get("building:part") and t.get("height") == "53" for t in tags.values())
    assert any(t.get("leisure") == "stadium" for t in tags.values())
    # nothing else: no river bank, no swamp (it has its own key), no stream
    assert set(tags) == {"w630", "w640"} and tags["w640"] == {"leisure": "stadium", "name": "Test Stadium"}
    (swamp,) = out["wetlands"]
    assert swamp["id"] == "w620" and len(swamp["polygons"]) == 1 and swamp["polygons"][0]["inner"] == []
    area = Polygon(_level(small_extent, swamp["polygons"][0]["outer"])).area
    assert area == pytest.approx(100 * 80, rel=0.01)
    # the new keys leave the buildings alone: a building:part is not a building
    assert out["buildings"] == [] and out["trees"] == [] and out["nodes"] == []


def _osm(path, ext, ways=(), relations=()):
    """Write an OSM file: ways are (id, [(x, y), ...] in level metres, tags), relations are
    (id, [(way id, role)], tags). Every way gets its own nodes."""
    def tag_xml(tags):
        return "".join('<tag k="%s" v="%s"/>' % kv for kv in tags.items())

    nodes, lines, nid = [], [], 0
    for wid, pts, tags in ways:
        refs = []
        for x, y in pts:
            nid += 1
            lon, lat = ext.to_lonlat(x, y)
            nodes.append('<node id="%d" version="1" lat="%.7f" lon="%.7f"/>' % (nid, lat, lon))
            refs.append('<nd ref="%d"/>' % nid)
        lines.append('<way id="%d" version="1">%s%s</way>' % (wid, "".join(refs), tag_xml(tags)))
    for rid, members, tags in relations:
        m = "".join('<member type="way" ref="%d" role="%s"/>' % wr for wr in members)
        lines.append('<relation id="%d" version="1">%s%s</relation>' % (rid, m, tag_xml(tags)))
    path.write_text('<?xml version="1.0" encoding="UTF-8"?><osm version="0.6">' + "".join(nodes + lines)
                    + "</osm>", encoding="utf-8")
    return path


def _sq(x, y, size=40.0):
    return [(x, y), (x + size, y), (x + size, y + size), (x, y + size), (x, y)]


def test_extract_waterway_kinds_and_culverts(tmp_path, small_extent):
    def line(i):
        return [(-300.0 + 10 * i, 50.0 * i), (-250.0 + 10 * i, 50.0 * i + 20)]

    kinds = ["river", "stream", "canal", "ditch", "drain", "weir", "dam", "riverbank", "lock_gate"]
    ways = [(10 + i, line(i), {"waterway": kind}) for i, kind in enumerate(kinds)]
    ways += [(30, line(-1), {"waterway": "canal", "tunnel": "culvert"}),
             (31, line(-2), {"waterway": "ditch", "tunnel": "yes", "name": "Under The Road"}),
             (32, line(-3), {"waterway": "drain", "culvert": "yes"}),
             (33, line(-4), {"waterway": "stream", "tunnel": "building_passage"}),
             (34, line(-5), {"waterway": "stream", "culvert": "no"})]
    out = features.extract_osm_features(_osm(tmp_path / "k.osm", small_extent, ways), small_extent, 100.0)
    ww = {w["id"]: w for w in out["waterways"]}
    assert {w["kind"] for w in ww.values()} == {"river", "stream", "canal", "ditch", "drain"}
    assert set(ww) == {"w%d" % i for i in (10, 11, 12, 13, 14, 30, 31, 32, 33, 34)}  # no weir, dam, bank, gate
    assert {i for i, w in ww.items() if w["culvert"]} == {"w30", "w31", "w32"}
    assert ww["w31"]["name"] == "Under The Road" and ww["w10"]["name"] is None


def test_extract_keeps_only_what_has_a_node_in_the_box(tmp_path, small_extent):
    # the box is +-512 m, +-612 m with the buffer
    ways = [(1, [(-700, 0), (-600, 0), (-500, 0), (700, 0)], {"waterway": "river"}),   # node at -600: kept whole
            (2, [(-700, 700), (700, 700)], {"waterway": "river"}),                       # misses the box
            (3, [(-700, -605), (-700, -700)], {"waterway": "river"}),                    # all outside
            (4, _sq(590, 590, 20), {"natural": "wetland", "wetland": "marsh"}),          # corner node at 590: kept
            (5, _sq(620, 0, 20), {"natural": "wetland"}),                                # outside
            (6, _sq(0, 640, 20), {"leisure": "pitch"}),                                  # outside
            (7, _sq(-10, -10, 20), {"leisure": "pitch"})]                                # inside
    osm = _osm(tmp_path / "b.osm", small_extent, ways)
    out = features.extract_osm_features(osm, small_extent, 100.0)
    assert [w["id"] for w in out["waterways"]] == ["w1"] and len(out["waterways"][0]["coords"]) == 4
    assert [(w["id"], w["wetland"]) for w in out["wetlands"]] == [("w4", "marsh")]
    assert [a["id"] for a in out["areas"]] == ["w7"]
    # with no buffer the marsh corner at 590 is out of the +-512 m box, the river node at -500 is in
    out0 = features.extract_osm_features(osm, small_extent, 0.0)
    assert [w["id"] for w in out0["waterways"]] == ["w1"] and out0["wetlands"] == []


def test_extract_area_kinds_and_building_overlap(tmp_path, small_extent):
    ways = [(1, _sq(-300, -300), {"leisure": "pitch"}),
            (2, _sq(-250, -300), {"leisure": "park"}),
            (3, _sq(-200, -300), {"man_made": "tower", "tower:type": "communication"}),
            (4, _sq(-150, -300), {"man_made": "pier"}),
            (5, _sq(-100, -300), {"building": "yes", "leisure": "stadium"}),     # a building and an area
            (6, _sq(-50, -300), {"building": "yes", "building:part": "yes"}),      # likewise
            (7, _sq(0, -300), {"building": "yes"}),                                # a building only
            (8, _sq(50, -300), {"building:part": "yes", "building:levels": "3"})]  # an area only
    out = features.extract_osm_features(_osm(tmp_path / "a.osm", small_extent, ways), small_extent, 100.0)
    assert {a["id"] for a in out["areas"]} == {"w1", "w3", "w5", "w6", "w8"}
    assert {b["id"] for b in out["buildings"]} == {"w5", "w6", "w7"}
    assert {a["id"]: a["tags"] for a in out["areas"]}["w5"] == {"building": "yes", "leisure": "stadium"}


def test_extract_wetland_multipolygon(tmp_path, small_extent):
    ways = [(1, _sq(0, 0, 100), {}), (2, _sq(40, 40, 20), {})]
    rel = [(9, [(1, "outer"), (2, "inner")], {"type": "multipolygon", "natural": "wetland", "wetland": "marsh"})]
    out = features.extract_osm_features(_osm(tmp_path / "m.osm", small_extent, ways, rel), small_extent, 100.0)
    (marsh,) = out["wetlands"]
    assert marsh["id"] == "r9" and marsh["wetland"] == "marsh"
    (part,) = marsh["polygons"]
    (hole,) = part["inner"]
    assert Polygon(_level(small_extent, part["outer"])).area == pytest.approx(10000, rel=0.01)
    assert Polygon(_level(small_extent, hole)).area == pytest.approx(400, rel=0.01)
    assert part["outer"][0] == part["outer"][-1]


# ---------------------------------------------------------------- Microsoft footprints


def test_quadkeys_cover_box(extent):
    assert features.quadkeys(extent) == ["023131130"]
    assert features.quadkeys(extent, 500.0) == ["023131130"]
    assert features.quadkey(3, 5, 3) == "213"  # the example in the Bing Maps tile system docs
    # a box on the corner of four zoom-9 tiles gets all four, 023131130 (x 126, y 210) among them
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * 210 / 512))))
    corner = features.quadkeys(Extent(126 / 512 * 360 - 180, lat, 2000.0))
    assert len(corner) == 4 and "023131130" in corner and all(len(q) == 9 for q in corner)


def _ring(extent, x, y, size=10.0):
    corners = [(x, y), (x + size, y), (x + size, y + size), (x, y + size), (x, y)]
    return [list(extent.to_lonlat(cx, cy)) for cx, cy in corners]


def _feature(ring, height):
    return json.dumps({"type": "Feature", "properties": {"height": height, "confidence": -1},
                       "geometry": {"type": "Polygon", "coordinates": [ring]}})


def test_fetch_ms_buildings_filters_to_box(tmp_path, small_extent, responses_mock):
    cfg = Config(debug_half_size_m=512.0)
    cfg.sources["ms_buildings_links"] = LINKS
    tile = ("https://ms.test/global-buildings/2026-02-03/global-buildings.geojsonl/RegionName=UnitedStates/"
            "quadkey=023131130/part-00004.csv.gz")
    responses_mock.add(responses.GET, LINKS, status=200, body=(
        "Location,QuadKey,Url,Size,UploadDate\n"
        f"United States,023131130,{tile},36.2MB,2026-02-03\n"
        "United States,023131131,https://ms.test/elsewhere.csv.gz,1MB,2026-02-03\n"
        "Mexico,023131130,https://ms.test/mexico.csv.gz,1KB,2026-02-03\n"))
    edge = 512 + 500 - 5  # straddles the buffer's east edge: it intersects the box plus buffer
    lines = [_feature(_ring(small_extent, 0, 0), 4.5), _feature(_ring(small_extent, edge, 0), -1),
             _feature(_ring(small_extent, 3000, 0), 7.0), ""]
    responses_mock.add(responses.GET, tile, status=200, body=gzip.compress("\n".join(lines).encode()))
    raw = tmp_path / "raw"

    out = features.fetch_ms_buildings(cfg, raw)
    assert out == raw / "ms_buildings.json"
    got = json.loads(out.read_text(encoding="utf-8"))
    assert [g["height"] for g in got] == [4.5, -1.0]
    np.testing.assert_allclose(_level(small_extent, got[0]["ring"])[:4], [[0, 0], [10, 0], [10, 10], [0, 10]],
                               atol=0.05)
    assert (raw / "ms" / "dataset-links.csv").exists()
    assert [p.name for p in (raw / "ms").glob("*.csv.gz")] == ["023131130_part-00004.csv.gz"]

    out.unlink()  # the downloads are cached: rebuilding needs no requests
    features.fetch_ms_buildings(cfg, raw)
    assert len(responses_mock.calls) == 2 and len(json.loads(out.read_text(encoding="utf-8"))) == 2


@pytest.mark.parametrize("header,location,qk", [
    ("Location,QuadKey,Url,Size,UploadDate", "United States", "023131130"),
    ("location,quadkey,url,size", "UnitedStates", "23131130"),  # leading zero lost by a spreadsheet
    ("\ufeffLOCATION,QUADKEY,URL", "united_states", "023131130"),
])
def test_ms_links_accept_naming_variants(header, location, qk):
    cols = header.lstrip("\ufeff").split(",")
    row = {"location": location, "quadkey": qk, "url": "https://ms.test/us.csv.gz", "size": "1MB",
           "uploaddate": "2026-02-03"}
    other = {**row, "location": "Mexico", "url": "https://ms.test/mx.csv.gz"}
    text = "\n".join([header] + [",".join(r[c.lower()] for c in cols) for r in (row, other)])
    assert features.ms_tile_urls(text, ["023131130"]) == [("023131130", "https://ms.test/us.csv.gz")]
    assert features.ms_tile_urls(text, ["023131131"]) == []


def test_ms_links_without_us_rows_take_every_location():
    text = "Location,QuadKey,Url\nUSA-ish,023131130,https://ms.test/a.csv.gz\n"
    assert features.ms_tile_urls(text, ["023131130"]) == [("023131130", "https://ms.test/a.csv.gz")]


# ---------------------------------------------------------------- lidar DSM


def _item(id, path, year=None, asset="data"):
    with rasterio.open(path) as ds:
        bbox = list(transform_bounds(ds.crs, "EPSG:4326", *ds.bounds))
    props = {"naip:year": str(year)} if year else {}
    return SimpleNamespace(id=id, bbox=bbox, properties=props,
                           assets={asset: SimpleNamespace(href=str(path))})


def test_fetch_dsm_clips_items_to_box(tmp_path, small_extent, monkeypatch):
    src = write_dem(tmp_path / "dsm.tif", small_extent, lambda E, N: 7.0 + 0 * E, margin=1000.0,
                    nodata=-9999.0)
    far = write_dem(tmp_path / "far.tif", Extent(-92.5, 31.5, 512.0), lambda E, N: 9.0 + 0 * E)
    searched = []

    def search(cfg, collection, bbox):
        searched.append((collection, bbox))
        return [_item("dsm-far", far), _item("dsm-a", src)]

    monkeypatch.setattr(features, "_stac_items", search)
    monkeypatch.setattr(planetary_computer, "sign", lambda href: href)
    cfg = Config(debug_half_size_m=512.0)
    raw = tmp_path / "raw"
    out = features.fetch_dsm(cfg, raw)
    assert [p.name for p in out] == ["dsm-a.tif"]
    assert searched == [("3dep-lidar-dsm", pytest.approx(small_extent.bbox_lonlat(500.0)))]
    assert (raw / "dsm" / "dsm-far.tif.empty").exists()
    w, s, e, n = small_extent.bbox_utm(500.0)
    with rasterio.open(out[0]) as ds:
        assert ds.bounds.left == pytest.approx(w, abs=2.0) and ds.bounds.top == pytest.approx(n, abs=2.0)
        assert ds.bounds.right == pytest.approx(e, abs=2.0) and ds.bounds.bottom == pytest.approx(s, abs=2.0)
        assert ds.nodata == -9999.0 and ds.read(1).max() == 7.0

    def no_reads(*a, **k):
        raise AssertionError("cached clips are not read again")

    monkeypatch.setattr(features, "clip_raster", no_reads)
    assert features.fetch_dsm(cfg, raw) == out


def test_fetch_dsm_with_nothing_found_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(features, "_stac_items", lambda cfg, collection, bbox: [])
    with pytest.raises(fetch.FetchError, match="DSM"):
        features.fetch_dsm(Config(debug_half_size_m=512.0), tmp_path)


# ---------------------------------------------------------------- NAIP red and near-infrared


def _naip(path, extent, x0, x1, red, nir, res=1.0):
    """A 4-band (R, G, B, NIR) uint8 image over level x0..x1 and the box height, with overviews."""
    w, e = extent.center_e + x0, extent.center_e + x1
    n = extent.center_n + extent.half_size + 50
    width, height = int(round((e - w) / res)), int(round((2 * extent.half_size + 100) / res))
    data = np.full((4, height, width), 60, np.uint8)
    data[0], data[3] = red, nir
    with rasterio.open(path, "w", driver="GTiff", width=width, height=height, count=4, dtype="uint8",
                       crs="EPSG:26915", transform=from_origin(w, n, res, res), tiled=True,
                       photometric="RGB", ALPHA="UNSPECIFIED") as dst:
        dst.write(data)
        dst.build_overviews([2, 4], Resampling.average)
    return path


def test_fetch_naip_rn_mosaics_latest_year(tmp_path, small_extent, monkeypatch):
    a = _naip(tmp_path / "a.tif", small_extent, -560, 100, red=40, nir=120)
    b = _naip(tmp_path / "b.tif", small_extent, -100, 560, red=90, nir=30)
    old = _naip(tmp_path / "old.tif", small_extent, -560, 560, red=200, nir=200)
    items = [_item("b_2023", b, 2023, "image"), _item("old_2021", old, 2021, "image"),
             _item("a_2023", a, 2023, "image")]
    monkeypatch.setattr(features, "_stac_items", lambda cfg, collection, bbox: collection == "naip" and items)
    monkeypatch.setattr(planetary_computer, "sign", lambda href: href)
    opened = []
    real_open = rasterio.open
    monkeypatch.setattr(features.rasterio, "open",
                        lambda fp, *a, **k: opened.append(k) or real_open(fp, *a, **k))

    out = features.fetch_naip_rn(Config(debug_half_size_m=512.0), tmp_path / "raw")
    assert out == tmp_path / "raw" / "naip_rn_2m.tif"
    with real_open(out) as ds:
        assert (ds.count, ds.dtypes[0], ds.shape, ds.nodata) == (2, "uint8", (512, 512), 0)
        assert ds.transform.almost_equals(small_extent.transform(512))
        red, nir = ds.read(1), ds.read(2)
    cols = {x: int((x + 512) / 2) for x in (-300, 0, 300)}
    assert (red[256, cols[-300]], nir[256, cols[-300]]) == (40, 120)
    assert (red[256, cols[300]], nir[256, cols[300]]) == (90, 30)
    assert (red[256, cols[0]], nir[256, cols[0]]) == (40, 120)  # the overlap: the first item (by id) wins
    assert not (red == 200).any()  # the 2021 images are not used
    # the 1 m images are read through their 2 m overview, not at full resolution
    assert {"overview_level": 0} in opened


def test_naip_year_must_cover_box():
    def item(year, bbox):
        return SimpleNamespace(id=f"i{year}_{bbox[0]}", bbox=bbox, properties={"naip:year": str(year)})

    box = (-91.2, 30.4, -91.1, 30.5)
    half = [item(2025, [-91.3, 30.3, -91.15, 30.6])]
    full = [item(2023, [-91.3, 30.3, -91.15, 30.6]), item(2023, [-91.16, 30.3, -91.0, 30.6])]
    older = [item(2021, [-91.3, 30.3, -91.0, 30.6])]
    assert {i.properties["naip:year"] for i in features.naip_items(half + full + older, box)} == {"2023"}
    # no year covers the whole box: the one covering most of it
    assert {i.properties["naip:year"] for i in features.naip_items(half + full[:1], box)} == {"2025"}
    assert features.naip_items([], box) == []


# ---------------------------------------------------------------- the stage


def test_run_does_each_step_and_caches_osm(tmp_path, monkeypatch):
    calls = []
    empty = {"buildings": [], "trees": [], "nodes": [], "waterways": [], "wetlands": [], "areas": []}
    monkeypatch.setattr(features, "download", lambda url, dest, **k: calls.append("pbf") or dest)
    monkeypatch.setattr(features, "extract_osm_features",
                        lambda pbf, ext, buf: calls.append(("osm", pbf.name, buf)) or empty)
    monkeypatch.setattr(features, "fetch_ms_buildings", lambda cfg, raw: calls.append(("ms", raw))
                        or raw / "ms_buildings.json")
    monkeypatch.setattr(features, "fetch_dsm", lambda cfg, raw: calls.append(("dsm", raw)) or [])
    monkeypatch.setattr(features, "fetch_naip_rn", lambda cfg, raw: calls.append(("naip", raw))
                        or raw / "naip_rn_2m.tif")
    raw = tmp_path / "raw"
    cfg = Config(debug_half_size_m=512.0, raw_dir=str(raw), buildings_microsoft=False)
    features.run(cfg)
    assert calls == ["pbf", ("osm", "louisiana-latest.osm.pbf", 500.0), ("dsm", raw), ("naip", raw)]
    assert json.loads((raw / "osm_features.json").read_text(encoding="utf-8")) == empty

    calls.clear()
    features.run(replace(cfg, buildings_microsoft=True))
    assert calls == [("ms", raw), ("dsm", raw), ("naip", raw)]  # the OSM extract is cached


def test_run_extracts_osm_again_when_the_cache_predates_waterways(tmp_path, monkeypatch):
    calls = []
    new = {"buildings": [], "trees": [], "nodes": [], "waterways": [], "wetlands": [], "areas": []}
    monkeypatch.setattr(features, "download", lambda url, dest, **k: dest)
    monkeypatch.setattr(features, "extract_osm_features", lambda pbf, ext, buf: calls.append("osm") or new)
    monkeypatch.setattr(features, "fetch_dsm", lambda cfg, raw: [])
    monkeypatch.setattr(features, "fetch_naip_rn", lambda cfg, raw: raw / "naip_rn_2m.tif")
    raw = tmp_path / "raw"
    raw.mkdir()
    cfg = Config(debug_half_size_m=512.0, raw_dir=str(raw), buildings_microsoft=False)
    osm = raw / "osm_features.json"
    osm.write_text(json.dumps({"buildings": [], "trees": [], "nodes": []}), encoding="utf-8")  # as in v0.1
    features.run(cfg)
    assert calls == ["osm"] and json.loads(osm.read_text(encoding="utf-8")) == new
    features.run(cfg)
    assert calls == ["osm"]  # now it is current
    osm.write_text("{broken", encoding="utf-8")  # unreadable: made again as well
    features.run(cfg)
    assert calls == ["osm", "osm"]


def test_cached_outputs_cleared_when_box_grows(tmp_path):
    raw = tmp_path / "raw"
    (raw / "dsm").mkdir(parents=True)
    (raw / "ms").mkdir()
    keep = [raw / "ms" / "023131130_part.csv.gz", raw / "louisiana-latest.osm.pbf", raw / "osm.json"]
    outputs = [raw / "osm_features.json", raw / "ms_buildings.json", raw / "dsm" / "a.tif",
               raw / "naip_rn_2m.tif"]
    for p in keep + outputs:
        p.write_bytes(b"x")
    features.check_box(raw, (0, 0, 10, 10))
    features.check_box(raw, (1, 1, 9, 9))  # inside the box they were cut to: kept
    assert all(p.exists() for p in outputs)
    features.check_box(raw, (-5, -5, 20, 20))
    assert not any(p.exists() for p in outputs) and all(p.exists() for p in keep)
