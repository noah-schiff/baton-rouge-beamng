from pathlib import Path

import pytest

from brmap.config import Config, load_config
from brmap.geo import Extent

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


def test_repo_config_matches_spec():
    cfg = load_config(REPO_CONFIG)
    assert (cfg.center_lat, cfg.center_lon) == (30.430, -91.160)
    assert cfg.half_size_m == 8192 and cfg.buffer_m == 500
    # z_min is -15 rather than the plan's -5 so the carved Mississippi bed (about -11 m) fits
    assert cfg.terrain_size == 4096 and (cfg.z_min, cfg.z_max) == (-15.0, 40.0)
    assert cfg.clearance_m == 5.5
    assert cfg.bridge_overrides == {"Horace Wilkinson Bridge": 52.0}
    assert cfg.debug_half_size_m is None
    assert cfg.sources["overpass"].startswith("https://")


def test_repo_config_level_and_spawns():
    cfg = load_config(REPO_CONFIG)
    assert cfg.level_name == "baton_rouge" and cfg.level_title == "Baton Rouge 1:1"
    assert [s.name for s in cfg.spawns] == ["spawn_lsu", "spawn_capitol", "spawn_i10_bridge"]
    lsu = cfg.spawns[0]
    assert (lsu.lat, lsu.lon) == (30.4115, -91.1855) and lsu.title


def test_spawns_default_when_missing(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("{}\n", encoding="utf-8")
    assert [s.name for s in load_config(p).spawns] == ["spawn_lsu", "spawn_capitol",
                                                       "spawn_i10_bridge"]


def test_paths_default_and_override(tmp_path):
    assert (Config().raw_dir, Config().out_dir, Config().dist_dir) == ("data/raw", "data/out", "dist")
    p = tmp_path / "config.yaml"
    p.write_text("paths:\n  out: data/out_debug\n  dist: dist/debug\n", encoding="utf-8")
    cfg = load_config(p)
    assert (cfg.raw_dir, cfg.out_dir, cfg.dist_dir) == ("data/raw", "data/out_debug", "dist/debug")


def test_stages_write_to_configured_dirs(tmp_path, monkeypatch, small_extent):
    import numpy as np

    from brmap import terrain
    from rasters import write_dem

    monkeypatch.chdir(tmp_path)
    (tmp_path / "raw" / "dem").mkdir(parents=True)
    write_dem(tmp_path / "raw" / "dem" / "2023-01-01_a.tif", small_extent, lambda E, N: 3.0 + 0 * E)
    cfg = Config(debug_half_size_m=512.0, terrain_size=256, raw_dir="raw", out_dir="elsewhere")
    terrain.run(cfg)
    assert np.load(tmp_path / "elsewhere" / "heights.npy").shape == (256, 256)
    assert not (tmp_path / "data").exists()


def test_debug_config_file():
    cfg = load_config(REPO_CONFIG.with_name("config.debug.yaml"))
    assert cfg.debug_half_size_m == 1500 and cfg.terrain_size == 1024
    assert cfg.level_name == "baton_rouge_debug" and cfg.out_dir != "data/out"
    assert cfg.raw_dir == "data/raw"  # shares the downloads


def test_debug_half_size_overrides(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("extent:\n  debug_half_size_m: 1000\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.debug_half_size_m == 1000
    assert Extent.from_config(cfg).half_size == 1000


def test_missing_keys_use_defaults(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("terrain:\n  size: 1024\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.terrain_size == 1024 and cfg.half_size_m == Config().half_size_m


@pytest.mark.parametrize("body", [
    "terrain:\n  size: 1000\n",            # not a power of two
    "terrain:\n  z_min: 40\n  z_max: 10\n",  # inverted range
    "extent:\n  half_size_m: -5\n",
    "buildings:\n  tile_m: 0\n",
    "trees:\n  max_count: -1\n",
])
def test_invalid_values_rejected(tmp_path, body):
    p = tmp_path / "config.yaml"
    p.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(p)


def test_feature_config_defaults_and_yaml(tmp_path):
    d = Config()
    assert (d.buildings_microsoft, d.building_tile_m, d.trees_max, d.markings_enabled,
            d.traffic_enabled) == (True, 256.0, 250000, True, True)
    assert d.sources["ms_buildings_links"] == ("https://minedbuildings.z5.web.core.windows.net/"
                                               "global-buildings/dataset-links.csv")
    assert d.sources["dsm_collection"] == "3dep-lidar-dsm"
    assert d.sources["naip_collection"] == "naip"
    p = tmp_path / "config.yaml"
    p.write_text("buildings:\n  microsoft: false\n  tile_m: 128\ntrees:\n  max_count: 500\n"
                 "markings:\n  enabled: false\ntraffic:\n  enabled: false\n", encoding="utf-8")
    cfg = load_config(p)
    assert (cfg.buildings_microsoft, cfg.building_tile_m, cfg.trees_max, cfg.markings_enabled,
            cfg.traffic_enabled) == (False, 128.0, 500, False, False)
    assert isinstance(cfg.building_tile_m, float) and isinstance(cfg.trees_max, int)
    assert cfg.sources["dsm_collection"] == "3dep-lidar-dsm"  # sources not in the file keep their default
    repo = load_config(REPO_CONFIG)
    assert repo.trees_max == 250000 and repo.markings_enabled and repo.traffic_enabled
    assert repo.sources["ms_buildings_links"] == d.sources["ms_buildings_links"]
    debug = load_config(REPO_CONFIG.with_name("config.debug.yaml"))
    assert debug.trees_max == 20000 and debug.buildings_microsoft and debug.markings_enabled


def test_feature_flags_accept_quoted_strings(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("markings:\n  enabled: 'false'\ntraffic:\n  enabled: 'yes'\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.markings_enabled is False and cfg.traffic_enabled is True
