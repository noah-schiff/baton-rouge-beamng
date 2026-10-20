"""The shared contracts the feature stages are built against (plan 2026-10-16, Shared contracts)."""
import importlib
import inspect
import json

import numpy as np
import pytest

from brmap import buildings, canopy, level
from brmap.buildings import Building
from brmap.geo import Extent


@pytest.mark.parametrize("name", ["buildings", "styled", "trees", "markings", "landmarks", "bayous", "dressing"])
def test_feature_module_has_level_content(name):
    mod = importlib.import_module(f"brmap.{name}")
    assert list(inspect.signature(mod.level_content).parameters) == ["cfg", "data_dir", "root", "heights",
                                                                     "extent"]
    assert name in level.FEATURE_MODULES


def _house():
    return Building(id="w123", source="osm", kind="house",
                    ring=np.array([[0.0, 0.0], [12.0, 0.0], [12.0, 9.0], [0.0, 9.0]]),
                    holes=[np.array([[2.0, 2.0], [4.0, 2.0], [4.0, 4.0]])], height=6.0, base_z=4.25,
                    roof="hip", wall="res", colour=(0.9, 0.85, 0.8), height_source="lidar")


def test_building_json_roundtrip(tmp_path):
    path = buildings.save_buildings([_house()], tmp_path / "out" / "buildings.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw == [{"id": "w123", "source": "osm", "kind": "house",
                    "ring": [[0.0, 0.0], [12.0, 0.0], [12.0, 9.0], [0.0, 9.0]],
                    "holes": [[[2.0, 2.0], [4.0, 2.0], [4.0, 4.0]]], "height": 6.0, "base_z": 4.25,
                    "roof": "hip", "wall": "res", "colour": [0.9, 0.85, 0.8], "height_source": "lidar"}]
    (back,) = buildings.load_buildings(path)
    assert back.ring.shape == (4, 2) and back.holes[0].shape == (3, 2)
    np.testing.assert_array_equal(back.ring, _house().ring)
    assert back.colour == (0.9, 0.85, 0.8) and back.height_source == "lidar"
    assert Building.from_dict({**raw[0], "holes": []}).holes == []


def test_analysis_grid(extent, small_extent):
    assert canopy.analysis_grid(extent) == (8192, 2.0)
    assert canopy.analysis_grid(small_extent) == (512, 2.0)
    n, px = canopy.analysis_grid(Extent(-91.160, 30.430, 1500.3))
    assert n == 1500 and n * px == pytest.approx(3000.6)  # the grid spans the box exactly


def test_canopy_files_roundtrip(tmp_path):
    ndsm = np.full((4, 4), np.nan)
    ndsm[0, 0] = 12.0
    ndvi = np.zeros((4, 4))
    canopy.save_canopy(tmp_path, ndsm, ndvi, 2.0)
    a, b, px = canopy.load_canopy(tmp_path)
    assert a.dtype == b.dtype == np.float32 and a.shape == b.shape == (4, 4) and px == 2.0
    assert a[0, 0] == 12.0 and np.isnan(a[1, 1])
    with pytest.raises(ValueError):
        canopy.save_canopy(tmp_path, ndsm, np.zeros((3, 3)), 2.0)
