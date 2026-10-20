"""Checks against the installed game's shared assets (skipped when BeamNG.drive is absent).

Every texture map the level's materials name (terrain, roads, buildings, paint, the
copied stock tree materials and the landmark and custom tree materials' stock: maps) has to
ship in the game, every ground type they name has to be one of the game's, every stock
tree shape the forest borrows has to be in a stock level zip, and the bayous' River has to be
italy's muddy river with textures the game ships. Set BEAMNG_GAME_DIR to the
game folder if it is not in Steam's default library.
"""
import io
import json
import os
import re
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from brmap import beamng_format as bf
from brmap import bayous, buildings, dressing, landmarks, markings, materials, trees
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


def test_landmark_stock_textures_exist_in_the_game():
    registry = materials.load_registry(landmarks.MATERIAL_REGISTRY)
    refs = [(name, ref.split(":", 1)[1]) for name, e in registry.items()
            for ref in (e.get("maps") or {}).values() if ref.startswith("stock:")]
    # the stucco, the Italian roof tiles (back from the CC0 roof_barrel), the trim's marble, the stadium's
    # and the PMAC's concrete (concrete_plain, as the buildings use it), the ballpark's clay (terrain
    # dirt_crumbly), the habitat's rock wall, the Capitol's limestone (the marble slabs, tinted) and the Old
    # State Capitol's white stucco (stucco1_white) and the Wilkinson truss's painted steel (metal_paint) are stock
    assert {name for name, _ in refs} >= {"br_lm_stucco_tan", "br_lm_roof_tile", "br_lm_trim_stone",
                                          "br_lm_concrete_tan", "br_lm_concrete", "br_lm_clay", "br_lm_rock",
                                          "br_lm_limestone", "br_lm_stucco_white", "br_lm_truss_steel"}
    assert [r for r in refs if not find(r[1])] == []


def test_truss_steel_ground_type_exists_in_the_game():
    with zipfile.ZipFile(GAME / "gameengine.zip") as z:   # relaxed json, with comments: the names only
        text = z.read("art/groundmodels.json").decode("utf-8", "replace")
    ground = set(re.findall(r'"([A-Z][A-Z0-9_]*)"\s*:\s*\{', text))
    steel = materials.load_registry(landmarks.MATERIAL_REGISTRY)["br_lm_truss_steel"]
    assert steel["flags"]["groundType"] == "METAL" and "METAL" in ground


def test_dressing_material_is_a_global_stock_material():
    # the barriers and piers bind concrete_plain, defined for every level by the game's art/shapes, so the
    # dressing ships no materials file
    with zipfile.ZipFile(CONTENT / "art_shapes.zip") as z:
        mats = json.loads(z.read("art/shapes/objects/main.materials.json").decode("utf-8"))
    m = mats[dressing.MATERIAL]
    assert (m["name"], m["mapTo"], m["class"]) == (dressing.MATERIAL, dressing.MATERIAL, "Material")
    assert [r for r in stage_textures({dressing.MATERIAL: m}) if not find(r[1])] == []


def stock_refs(registry_path) -> list[tuple[str, str]]:
    """(material, path) for every stock: map of a material registry."""
    registry = materials.load_registry(registry_path)
    return [(name, ref.split(":", 1)[1]) for name, e in registry.items()
            for ref in (e.get("maps") or {}).values() if ref.startswith("stock:")]


def test_custom_tree_stock_textures_exist_in_the_game():
    refs = stock_refs(trees.TREE_MATERIALS)
    # the stock fir_bark (tree/fir/t_fir_bark) and the stock cypress sprays (tree/cypress/t_cypress_leaves); the
    # tupelo's poplar bark (tree/poplar/t_poplar_bark) and beech sprays (tree/beech/t_beech_leaves)
    assert {path.rsplit("/", 2)[1] for _, path in refs} == {"t_fir_bark", "t_cypress_leaves", "t_poplar_bark",
                                                            "t_beech_leaves"}
    assert {name for name, _ in refs} == {"br_tree_cypress_bark", "br_tree_cypress_leaves", "br_tree_tupelo_bark",
                                          "br_tree_tupelo_leaves"}
    assert [r for r in refs if not find(r[1])] == []


def test_custom_tree_ground_types_exist_in_the_game():
    with zipfile.ZipFile(GAME / "gameengine.zip") as z:   # relaxed json, with comments: the names only
        text = z.read("art/groundmodels.json").decode("utf-8", "replace")
    ground = set(re.findall(r'"([A-Z][A-Z0-9_]*)"\s*:\s*\{', text))
    assert {"ASPHALT", "GRASS"} <= ground
    used = {e["flags"]["groundType"] for e in materials.load_registry(trees.TREE_MATERIALS).values()}
    assert used == {"WOOD", "LEAVES_THIN"} and used <= ground


def test_tree_shapes_exist_in_the_stock_levels():
    items = trees.stock_trees()["items"]
    missing = [(key, item["shapeFile"]) for key, item in items.items()
               if item["shapeFile"].lstrip("/").lower() not in level_files()]
    assert items and missing == []


PLATES = "vehicles/common/licenseplates/"
# the stock files the level's plate design names: the default plate's normal map (its embossed rim) and the
# preview mesh every stock design names
PLATE_FILES = {PLATES + "default/licenseplate-default_n.png", PLATES + "driver_training/licenseplate-52-11.dae"}


def test_licence_plate_design_reuses_stock_files():
    design = bf.plate_design("baton_rouge")
    fmt = design["format"]["30-15"]
    with zipfile.ZipFile(CONTENT / "vehicles" / "common.zip") as z:
        files = set(z.namelist())
        stock, eca = (json.loads(z.read(f"{PLATES}{level}/licensePlate-default.sktemplate.json").decode("utf-8"))
                      for level in ("default", "east_coast_usa"))
    assert {fmt["normal"], design["preview"]["file"]} == PLATE_FILES and PLATE_FILES <= files
    # the stock default design's own plain 30-15 background: the default folder's licenseplate-default_d.png
    # is a state's plate (Utah's, which the utah level's design uses too)
    assert fmt["background"] == stock["format"]["30-15"]["background"] == "#ffffff"
    assert design["version"] == stock["version"] == 3
    assert fmt["root"] == eca["format"]["30-15"]["root"]  # lettered as the stock east_coast_usa design
    assert design["preview"] == eca["preview"]


RIVER_ID = "6b03db61-baa6-49b2-884a-a468ef124469"


def test_stock_river_is_italys_muddy_river():
    # the member's case changes between versions (0.39.4 .../Level_objects/Water/, 0.40 .../water/):
    # every items.level.json of the italy level, whatever its case, holding that River
    rivers = []
    with zipfile.ZipFile(CONTENT / "levels" / "italy.zip") as z:
        for member in z.namelist():
            low = member.lower()
            if low.startswith("levels/italy/") and low.endswith("/items.level.json"):
                objs = map(json.loads, filter(str.strip, z.read(member).decode("utf-8").splitlines()))
                rivers += [o for o in objs if o.get("persistentId") == RIVER_ID]
    (river,) = rivers
    assert river["class"] == "River"
    assert {k: v for k, v in river.items()
            if k not in ("class", "persistentId", "__parent", "position", "nodes")} == bayous.STOCK_RIVER


def test_bayou_water_textures_exist_in_the_game():
    river = bayous.river_object({"id": "w1", "nodes": [[0, 0, 5, 6], [10, 0, 5, 6]]}, 0)
    assert [k for k in ("foamTex", "rippleTex") if not find(river[k])] == []
    with zipfile.ZipFile(GAME / "gameengine.zip") as z:
        engine = {n.lower() for n in z.namelist()}
    assert river["depthGradientTex"].lstrip("/").lower() in engine
    assert f"art/cubemaps/{river['cubemap']}/main.materials.json" in engine
