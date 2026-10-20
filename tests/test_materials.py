"""Material definitions for the modelled shapes (materials.py) and the landmark material registry."""
from pathlib import Path

import pytest

from brmap import beamng_format as bf
from brmap import landmarks, materials


def test_resolve_maps_stock_and_tex():
    registry = {
        "br_lm_clock": {"maps": {"b": "tex:clock_face", "nm": "stock:/assets/materials/x/t_x_nm.normal.png"},
                        "factors": {"roughnessFactor": 0.4},
                        "flags": {"alphaTest": False, "alphaRef": 80, "doubleSided": True, "groundType": "CONCRETE"}},
        "br_lm_unused": {"maps": {"r": "stock:/assets/materials/x/t_x_r.data.png"}},
    }
    index = {"clock_face": {"source": "generated", "files": {"b": "clock_face_b.color.dds"}}}
    mats, files = materials.resolve(registry, ["br_lm_clock"], "baton_rouge", index)
    assert set(mats) == {"br_lm_clock"}                    # only the names asked for
    m = mats["br_lm_clock"]
    stage = m["Stages"][0]
    # referenced by its .png name (the stock convention, NOTE_FOR_MODDERS.md in 0.40), the .dds shipped
    assert stage["baseColorMap"] == "/levels/baton_rouge/art/textures/br/clock_face_b.color.png"
    assert stage["normalMap"] == "/assets/materials/x/t_x_nm.normal.png"      # stock: stays as it is
    assert stage["roughnessFactor"] == 0.4
    assert files == [Path("clock_face_b.color.dds")]       # only the .dds is copied into art/textures/br/
    assert (m["alphaTest"], m["alphaRef"], m["doubleSided"], m["groundType"]) == (False, 80, True, "CONCRETE")
    assert not {"alphaTest", "alphaRef", "doubleSided", "groundType"} & set(stage)


def test_every_map_key_has_its_stage_field():
    registry = {"br_x": {"maps": {k: f"tex:t{k}" for k in ("b", "nm", "r", "o", "ao")}}}
    index = {f"t{k}": {"files": {k: f"t{k}_{suffix}.dds"}} for k, (_, suffix) in materials.MAP_FIELDS.items()}
    mats, files = materials.resolve(registry, ["br_x"], "lvl", index)
    assert mats["br_x"]["Stages"][0] == {
        "baseColorMap": "/levels/lvl/art/textures/br/tb_b.color.png",
        "normalMap": "/levels/lvl/art/textures/br/tnm_nm.normal.png",
        "roughnessMap": "/levels/lvl/art/textures/br/tr_r.data.png",
        "opacityMap": "/levels/lvl/art/textures/br/to_o.data.png",
        "ambientOcclusionMap": "/levels/lvl/art/textures/br/tao_ao.data.png"}
    assert files == sorted(Path(f"t{k}_{suffix}.dds") for k, (_, suffix) in materials.MAP_FIELDS.items())


def test_tex_ref_names_the_indexed_file():
    # the path is the .png name of the file the textures stage shipped (the index's .dds), whatever
    # it is called; the game loads the .dds beside it
    index = {"clock_face": {"source": "generated", "files": {"b": "dial.dds"}}}
    mats, files = materials.resolve({"br_x": {"maps": {"b": "tex:clock_face"}}}, ["br_x"], "lvl", index)
    assert mats["br_x"]["Stages"][0]["baseColorMap"] == "/levels/lvl/art/textures/br/dial.png"
    assert files == [Path("dial.dds")]
    index["clock_face"]["files"]["b"] = "dial.jpg"          # nothing the game would find under dial.png
    with pytest.raises(ValueError, match="dial.jpg"):
        materials.resolve({"br_x": {"maps": {"b": "tex:clock_face"}}}, ["br_x"], "lvl", index)


def test_tex_ref_has_no_file_format_suffix():
    # "tex:<id>" only: the shipped format is the textures stage's business (textures.SHIP)
    index = {"clock_face": {"source": "generated", "files": {"b": "clock_face_b.color.dds"}}}
    with pytest.raises(ValueError, match="clock_face.dds"):
        materials.resolve({"br_x": {"maps": {"b": "tex:clock_face.dds"}}}, ["br_x"], "lvl", index)


def test_material_is_a_v15_definition():
    m = materials.material("br_x", {"baseColorMap": "/assets/a_b.color.png"}, alphaTest=True, alphaRef=100)
    assert m == {"name": "br_x", "mapTo": "br_x", "class": "Material", "version": 1.5,
                 "persistentId": bf.persistent_id("material", "br_x"),
                 "Stages": [{"baseColorMap": "/assets/a_b.color.png"}, {}, {}, {}],
                 "alphaTest": True, "alphaRef": 100}
    with pytest.raises(ValueError, match="name"):
        materials.material("br_x", {}, name="other")


def test_vertex_colour_switch_goes_in_the_stage():
    # every stock v1.5 material sets vertColor in its stage; at the top level the game ignores it
    mats, _ = materials.resolve({"br_x": {"maps": {}, "flags": {"vertColor": True, "doubleSided": True}}},
                                ["br_x"], "lvl", {})
    assert mats["br_x"]["Stages"][0] == {"vertColor": True} and mats["br_x"]["doubleSided"] is True
    assert "vertColor" not in mats["br_x"]


def test_resolve_refuses_what_it_cannot_resolve():
    with pytest.raises(ValueError, match="br_missing"):
        materials.resolve({}, ["br_missing"], "lvl", {})
    with pytest.raises(ValueError, match="clock_face"):
        materials.resolve({"br_x": {"maps": {"b": "tex:clock_face"}}}, ["br_x"], "lvl", {})
    with pytest.raises(ValueError, match="glow"):
        materials.resolve({"br_x": {"maps": {"glow": "stock:/a.png"}}}, ["br_x"], "lvl", {})
    with pytest.raises(ValueError, match="http"):
        materials.resolve({"br_x": {"maps": {"b": "http://example.com/a.png"}}}, ["br_x"], "lvl", {})


def test_landmark_material_registry():
    registry = materials.load_registry(landmarks.MATERIAL_REGISTRY)
    assert "br_lm_stucco_tan" in registry
    for name, entry in registry.items():
        assert name.startswith("br_"), name
        assert set(entry) <= {"maps", "factors", "flags"}, name
        for key, ref in (entry.get("maps") or {}).items():
            assert key in materials.MAP_FIELDS and ref.split(":", 1)[0] in ("stock", "tex"), (name, key)
    stucco = registry["br_lm_stucco_tan"]
    assert stucco["factors"]["baseColorFactor"] == [0.86, 0.76, 0.60, 1]
    assert set(stucco["maps"]) == {"b", "nm", "r"}
    assert all("t_stucco_white_" in ref for ref in stucco["maps"].values())
    tex = {ref[4:] for e in registry.values() for ref in (e.get("maps") or {}).values() if ref.startswith("tex:")}
    assert not any("." in t for t in tex)                      # a bare tex id, no file format
    index = {t: {"files": dict.fromkeys(("b", "nm", "r", "o", "ao"), f"{t}.dds")} for t in tex}
    mats, _ = materials.resolve(registry, registry, "baton_rouge", index)
    assert set(mats) == set(registry)


def test_memorial_tower_materials():
    # plan Task 8: the tower's stucco, roof tiles, cast-stone trim and one clock material
    registry = materials.load_registry(landmarks.MATERIAL_REGISTRY)
    roof, trim = registry["br_lm_roof_tile"]["maps"], registry["br_lm_trim_stone"]["maps"]
    # back to the stock Italian roof tiles (Task 17b): at the models' roof UVs the CC0 roof_barrel drew
    # each tile about 7.7 cm wide
    tiles = "stock:/assets/materials/tileable/roofing/t_italy_bld_roof_tiles/t_italy_bld_roof_tiles_"
    assert roof == {"b": tiles + "b.color.png", "nm": tiles + "nm.normal.png", "r": tiles + "r.data.png"}
    assert trim["nm"] == "tex:bevel" and all("t_slabs_marble_" in trim[k] for k in ("b", "r"))
    assert set(trim) == {"b", "nm", "r"}
    assert registry["br_lm_clock"]["maps"] == {"b": "tex:clock_face"}      # one material for all four dials
    assert registry["br_lm_clock"]["flags"]["alphaTest"] is False
    assert not {"br_lm_clock_png", "br_lm_clock_dds"} & set(registry)      # the PNG / DDS spike pair is gone
