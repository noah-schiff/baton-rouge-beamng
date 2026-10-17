import json
import logging
import math
from pathlib import Path

import numpy as np
import pytest

from brmap import roads
from brmap.roads import Road, build_roads, clip_to_box, parse_ways, road_width
from brmap.terrain import sample_height

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def osm_small():
    return json.loads((FIXTURES / "osm_small.json").read_text(encoding="utf-8"))


@pytest.fixture
def sloped(extent):
    """A 256 x 256 height grid that rises 1 m per km to the east."""
    xs, _ = extent.pixel_centers(256)
    return np.tile((xs / 1000.0 + 10.0).astype(np.float32), (256, 1))


def test_width_motorway_oneway_lanes_tag():
    assert road_width({"highway": "motorway", "oneway": "yes", "lanes": "3"}) == (3 * 3.6 + 3.0, 3)


def test_width_residential_default():
    assert road_width({"highway": "residential"}) == (6.5, 2)


def test_width_primary_oneway_no_lanes():
    assert road_width({"highway": "primary", "oneway": "yes"}) == (2 * 3.3 + 0.5, 2)


def test_width_bad_lanes_falls_back():
    assert road_width({"highway": "secondary", "lanes": "two"})[1] == 2


def test_width_semicolon_lanes():
    assert road_width({"highway": "tertiary", "lanes": "2;3"})[1] == 3


def test_width_service_without_lanes():
    assert road_width({"highway": "service"}) == (4.0, 1)


def test_width_link_uses_its_base_class():
    assert road_width({"highway": "motorway_link"}) == (3.6 + 3.0, 1)
    assert road_width({"highway": "primary_link"}) == (3.3 + 0.5, 1)


def test_width_motorway_implies_oneway():
    assert road_width({"highway": "motorway"}) == (2 * 3.6 + 3.0, 2)


def test_parse_ways_keeps_highways_only(osm_small):
    ways = parse_ways(osm_small)
    assert {w["id"] for w in ways} == {1, 2, 3, 4, 5, 6, 7, 8}
    assert ways[0]["lonlat"][0] == (-91.16, 30.43)


def test_footway_dropped(osm_small, extent, sloped):
    kinds = {r.kind for r in build_roads(osm_small, extent, sloped)}
    assert "footway" not in kinds and "construction" not in kinds
    assert kinds == {"residential", "motorway", "primary", "secondary"}


def test_clip_way_crossing_edge(extent):
    h = extent.half_size
    pieces = clip_to_box(np.array([[0.0, 0.0], [h + 500, 0.0]]), extent)
    assert len(pieces) == 1
    assert pieces[0][:, 0].max() <= h + 1e-9 and abs(pieces[0][-1, 0] - h) <= 0.01


def test_clip_way_exits_and_reenters(extent):
    h = extent.half_size
    xy = np.array([[h - 100, 0.0], [h + 100, 50.0], [h - 100, 100.0]])
    assert len(clip_to_box(xy, extent)) == 2


def test_clip_way_fully_outside(extent):
    h = extent.half_size
    assert clip_to_box(np.array([[h + 10, 0.0], [h + 20, 0.0]]), extent) == []


def test_degenerate_way_skipped(osm_small, extent, sloped, caplog):
    with caplog.at_level(logging.WARNING, logger="brmap.roads"):
        out = build_roads(osm_small, extent, sloped)
    ids = {i for r in out for i in r.osm_ids}
    assert 5 not in ids and 6 not in ids
    assert "5" in caplog.text and "6" in caplog.text


def test_densify_spacing(osm_small, extent, sloped):
    for r in build_roads(osm_small, extent, sloped):
        seg = np.linalg.norm(np.diff(r.nodes[:, :2], axis=0), axis=1)
        assert seg.max() <= 20.0 + 1e-6 and seg.min() > 0.05


def test_ground_road_z_from_terrain(osm_small, extent, sloped):
    r = next(r for r in build_roads(osm_small, extent, sloped) if r.kind == "residential")
    expected = sample_height(sloped, extent, r.nodes[:, 0], r.nodes[:, 1])
    np.testing.assert_allclose(r.nodes[:, 2], expected, atol=1e-6)


def test_bridge_flag_and_layer(osm_small, extent, sloped):
    r = next(r for r in build_roads(osm_small, extent, sloped) if r.kind == "primary")
    assert r.is_bridge and r.layer == 1 and r.name == "Test Bridge"
    assert np.isnan(r.nodes[:, 2]).all()


def test_road_attributes(osm_small, extent, sloped):
    by_kind = {r.kind: r for r in build_roads(osm_small, extent, sloped)}
    mw = by_kind["motorway"]
    assert mw.oneway and mw.lanes == 3 and mw.width == pytest.approx(3 * 3.6 + 3.0)
    assert mw.name == "I 10" and mw.osm_ids == [3]
    res = by_kind["residential"]
    assert not res.oneway and not res.is_bridge and res.layer == 0


def test_oneway_minus_one_is_reversed(osm_small, extent, sloped):
    r = next(r for r in build_roads(osm_small, extent, sloped) if r.kind == "secondary")
    assert r.oneway and r.nodes[0, 0] > r.nodes[-1, 0]  # drawn east to west, traffic flows west


def test_clipped_way_keeps_osm_id_on_each_piece(extent, sloped):
    h = extent.half_size
    lon0, lat0 = extent.to_lonlat(h - 100, 0.0)
    lon1, lat1 = extent.to_lonlat(h + 100, 50.0)
    lon2, lat2 = extent.to_lonlat(h - 100, 100.0)
    osm = {"elements": [{"type": "way", "id": 77, "tags": {"highway": "tertiary"}, "geometry": [
        {"lat": lat0, "lon": lon0}, {"lat": lat1, "lon": lon1}, {"lat": lat2, "lon": lon2}]}]}
    out = build_roads(osm, extent, sloped)
    assert len(out) == 2 and all(r.osm_ids == [77] for r in out)
    for r in out:
        assert extent.contains(r.nodes[:, 0], r.nodes[:, 1]).all()


def test_road_json_roundtrip():
    r = Road(osm_ids=[1, 2], kind="primary", nodes=np.array([[0, 0, 1.0], [10, 0, np.nan]]),
             width=7.1, lanes=2, oneway=False, is_bridge=True, layer=1, name=None)
    back = Road.from_dict(json.loads(json.dumps(r.to_dict())))
    assert back.osm_ids == [1, 2] and back.name is None and back.is_bridge
    np.testing.assert_array_equal(np.isnan(back.nodes), np.isnan(r.nodes))


def test_run_writes_roads_json(tmp_path, monkeypatch, osm_small):
    from brmap.config import Config

    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "out").mkdir(parents=True)
    (tmp_path / "data" / "raw" / "osm.json").write_text(json.dumps(osm_small), encoding="utf-8")
    np.save(tmp_path / "data" / "out" / "heights.npy", np.full((256, 256), 7.0, np.float32))
    np.save(tmp_path / "data" / "out" / "heights_raw.npy", np.full((256, 256), 7.0, np.float32))
    roads.run(Config(terrain_size=256))
    data = json.loads((tmp_path / "data" / "out" / "roads.json").read_text(encoding="utf-8"))
    assert {d["kind"] for d in data} == {"residential", "motorway", "primary", "secondary"}
    assert all(math.isfinite(z) for d in data for _, _, z in d["nodes"])  # bridges got decks
    assert any(d["is_bridge"] for d in data)


def test_run_gives_bridges_the_uncarved_heights(tmp_path, monkeypatch, osm_small):
    """Decks over water clear the water surface, not the carved bed."""
    from brmap import bridges
    from brmap.config import Config

    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "out").mkdir(parents=True)
    (tmp_path / "data" / "raw" / "osm.json").write_text(json.dumps(osm_small), encoding="utf-8")
    np.save(tmp_path / "data" / "out" / "heights.npy", np.full((256, 256), -8.0, np.float32))
    np.save(tmp_path / "data" / "out" / "heights_raw.npy", np.full((256, 256), 4.0, np.float32))
    seen = []

    def fake_apply(rs, heights, extent, cfg):
        seen.append(float(heights.mean()))
        for r in rs:
            r.nodes[:, 2] = 0.0
        return rs

    monkeypatch.setattr(bridges, "apply_bridges", fake_apply)
    roads.run(Config(terrain_size=256))
    assert seen == [4.0]


@pytest.mark.parametrize("tag, expected", [
    ("35 mph", 15.6464), ("35mph", 15.6464), ("50", 13.889), ("50 km/h", 13.889),
    ("signals", None), ("none", None), ("35 mph;45 mph", None), ("0", None), ("", None), (None, None),
])
def test_maxspeed_parsing(tag, expected):
    got = roads.parse_maxspeed(tag)
    if expected is None:
        assert got is None
    else:
        assert got == pytest.approx(expected, abs=1e-3)


def test_road_from_old_dict_has_no_maxspeed():
    old = {"osm_ids": [1], "kind": "residential", "nodes": [[0, 0, 1.0], [10, 0, 1.0]], "width": 6.5,
           "lanes": 2, "oneway": False, "is_bridge": False, "layer": 0, "name": None}
    assert Road.from_dict(old).maxspeed is None
    r = Road([1], "primary", np.array([[0, 0, 1.0], [10, 0, 1.0]]), 7.1, 2, False, False, 0, None,
             maxspeed=15.6464)
    assert Road.from_dict(json.loads(json.dumps(r.to_dict()))).maxspeed == 15.6464


def test_build_roads_reads_maxspeed(extent, sloped):
    lon0, lat0 = extent.to_lonlat(0.0, 0.0)
    lon1, lat1 = extent.to_lonlat(100.0, 0.0)
    geom = [{"lat": lat0, "lon": lon0}, {"lat": lat1, "lon": lon1}]
    osm = {"elements": [
        {"type": "way", "id": 1, "tags": {"highway": "primary", "maxspeed": "45 mph"}, "geometry": geom},
        {"type": "way", "id": 2, "tags": {"highway": "residential"}, "geometry": geom}]}
    by_id = {r.osm_ids[0]: r for r in build_roads(osm, extent, sloped)}
    assert by_id[1].maxspeed == pytest.approx(20.1168) and by_id[2].maxspeed is None


def test_build_roads_reads_covered(extent, sloped):
    # a drive under a building (the State Capitol's is tunnel=building_passage) is covered
    lon0, lat0 = extent.to_lonlat(0.0, 0.0)
    lon1, lat1 = extent.to_lonlat(100.0, 0.0)
    geom = [{"lat": lat0, "lon": lon0}, {"lat": lat1, "lon": lon1}]
    tags = [{"tunnel": "building_passage"}, {"covered": "yes"}, {"tunnel": "yes"}, {"covered": "no"}, {}]
    osm = {"elements": [{"type": "way", "id": i, "tags": {"highway": "service", **t}, "geometry": geom}
                        for i, t in enumerate(tags, 1)]}
    by_id = {r.osm_ids[0]: r.covered for r in build_roads(osm, extent, sloped)}
    assert by_id == {1: True, 2: True, 3: True, 4: False, 5: False}
    old = Road.from_dict({"osm_ids": [1], "kind": "service", "nodes": [[0, 0, 0], [1, 0, 0]],
                          "width": 4.0, "lanes": 1, "oneway": False, "is_bridge": False, "layer": 0})
    assert old.covered is False and old.to_dict()["covered"] is False


def test_bridge_merge_keeps_maxspeed():
    from brmap.bridges import merge_bridges

    a = Road([1], "motorway", np.array([[0, 0, np.nan], [50, 0, np.nan]]), 13.8, 3, True, True, 1,
             "Deck", maxspeed=31.2928)
    b = Road([2], "motorway", np.array([[50, 0, np.nan], [100, 0, np.nan]]), 13.8, 3, True, True, 1, "Deck")
    (merged,) = merge_bridges([b, a])
    assert merged.osm_ids == [1, 2] and merged.maxspeed == 31.2928
