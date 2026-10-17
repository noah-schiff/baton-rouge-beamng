"""Checks against the installed game's shared assets (skipped when BeamNG.drive is absent).

Every texture map the level's materials name (terrain, roads, buildings, paint and the
copied stock tree materials) has to ship in the game, and every stock tree shape the
forest borrows has to be in a stock level zip. Set BEAMNG_GAME_DIR to the game folder if
it is not in Steam's default library.
"""
import io
import os
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from brmap import beamng_format as bf
from brmap import buildings, markings, trees
from brmap.imagery import LAYER_NAMES

GAME = Path(os.environ.get("BEAMNG_GAME_DIR",
                           r"C:\Program Files (x86)\Steam\steamapps\common\BeamNG.drive"))
CONTENT = GAME / "content"
pytestmark = pytest.mark.skipif(not (CONTENT / "assets").is_dir(),
                                reason="BeamNG.drive is not installed here")


@lru_cache(maxsize=1)
def shipped() -> dict[str, tuple[Path, str]]:
    """Lower-case path without extension -> (zip, member) for every texture the game ships."""
    out = {}
    for zp in sorted((CONTENT / "assets").rglob("*.zip")):
        with zipfile.ZipFile(zp) as z:
            for n in z.namelist():
                if n.lower().endswith((".dds", ".png")):
                    out[n.lower().rsplit(".", 1)[0]] = (zp, n)
    return out


def find(path: str):
    """The game resolves a .png reference to the .dds it ships."""
    return shipped().get(path.lstrip("/").lower().rsplit(".", 1)[0])


@lru_cache(maxsize=1)
def level_files() -> frozenset[str]:
    """Lower-case paths of every file in the stock level zips (levels/<name>/...)."""
    out = set()
    for zp in sorted((CONTENT / "levels").glob("*.zip")):
        with zipfile.ZipFile(zp) as z:
            out.update(n.lower() for n in z.namelist())
    return frozenset(out)


def stage_textures(materials: dict) -> list[tuple[str, str]]:
    return [(name, v) for name, m in materials.items() for st in m.get("Stages", [])
            for k, v in st.items() if isinstance(v, str) and k.endswith("Map")]


def test_road_textures_exist_in_the_game():
    missing = [(name, path) for name, path in stage_textures(bf.road_materials()) if not find(path)]
    assert missing == []


def test_terrain_detail_textures_exist_in_the_game():
    mats = bf.terrain_materials("x", "/levels/x/b.png", 1024.0, LAYER_NAMES, 1024)
    refs = [(k, v) for m in mats.values() for k, v in m.items()
            if isinstance(v, str) and k.endswith("Tex") and v.startswith("/assets/")]
    assert refs and [r for r in refs if not find(r[1])] == []


def test_road_opacity_maps_cover_the_road():
    """A decal road drawn through an overlay's opacity map (most pixels under alphaRef) is invisible."""
    for name, m in bf.road_materials().items():
        st = m["Stages"][0]
        if "opacityMap" not in st:
            continue
        zp, member = find(st["opacityMap"])
        with zipfile.ZipFile(zp) as z, Image.open(io.BytesIO(z.read(member))) as im:
            a = np.asarray(im.convert("L"))
        assert (a >= m.get("alphaRef", 0)).mean() > 0.8, name


def missing_maps(materials: dict) -> list[tuple[str, str]]:
    refs = stage_textures(materials)
    assert refs, "no texture maps found"
    return [(name, path) for name, path in refs if not find(path)]


def test_building_textures_exist_in_the_game():
    materials = getattr(buildings, "MATERIALS", None)
    if materials is None:
        pytest.skip("buildings.MATERIALS comes with the buildings stage (plan Task 3)")
    assert missing_maps(materials) == []


def test_marking_textures_exist_in_the_game():
    assert missing_maps(markings.MATERIALS) == []


def test_tree_textures_exist_in_the_game():
    assert missing_maps(trees.stock_trees()["materials"]) == []


def test_tree_shapes_exist_in_the_stock_levels():
    items = trees.stock_trees()["items"]
    missing = [(key, item["shapeFile"]) for key, item in items.items()
               if item["shapeFile"].lstrip("/").lower() not in level_files()]
    assert items and missing == []
