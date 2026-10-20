"""Precompiled meshes (.cdae) and tree imposters: collected from the game's cache, reused while the .dae is unchanged."""
import hashlib
import json
import logging
import os
import shutil
import time
from pathlib import Path

from brmap import meshcache


def _write(path, data: bytes, mtime=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def test_collect_takes_only_meshes_compiled_from_this_build(tmp_path):
    level, temp, cache = tmp_path / "level", tmp_path / "temp", tmp_path / "cache"
    now = time.time()
    _write(level / "art/shapes/buildings/a.dae", b"<a/>", now - 100)
    _write(level / "art/shapes/buildings/b.dae", b"<b/>", now - 100)
    _write(level / "art/shapes/buildings/c.dae", b"<c/>", now - 100)
    _write(temp / "art/shapes/buildings/a.cdae", b"A", now - 50)    # compiled after the build: fresh
    _write(temp / "art/shapes/buildings/b.cdae", b"B", now - 200)   # compiled before it: stale
    got, missing = meshcache.collect(level, temp, cache)
    assert got == 1 and missing == ["art/shapes/buildings/b.dae", "art/shapes/buildings/c.dae"]
    assert (cache / "art/shapes/buildings/a.cdae").read_bytes() == b"A"
    assert not (cache / "art/shapes/buildings/b.cdae").exists()


def test_apply_reuses_only_while_the_dae_is_unchanged(tmp_path):
    level, temp, cache = tmp_path / "level", tmp_path / "temp", tmp_path / "cache"
    now = time.time()
    _write(level / "art/shapes/buildings/a.dae", b"<a/>", now - 100)
    _write(level / "art/shapes/buildings/b.dae", b"<b/>", now - 100)
    _write(temp / "art/shapes/buildings/a.cdae", b"A", now - 50)
    _write(temp / "art/shapes/buildings/b.cdae", b"B", now - 50)
    assert meshcache.collect(level, temp, cache) == (2, [])
    # a rebuild writes the same a.dae again (newer, same bytes) and a changed b.dae
    _write(level / "art/shapes/buildings/a.dae", b"<a/>")
    _write(level / "art/shapes/buildings/b.dae", b"<b changed/>")
    used, stale = meshcache.apply(level, cache)
    assert (used, stale) == (1, ["art/shapes/buildings/b.dae"])
    a = level / "art/shapes/buildings/a.cdae"
    assert a.read_bytes() == b"A"
    assert a.stat().st_mtime >= (level / "art/shapes/buildings/a.dae").stat().st_mtime  # the game must see it as newer
    assert not (level / "art/shapes/buildings/b.cdae").exists()


def test_collect_keeps_cached_meshes_still_valid(tmp_path):
    # after a rebuild that changed one tile, the game loads the shipped .cdae for the rest and
    # compiles only the changed one: collecting again must keep the rest, not forget them
    level, temp, cache = tmp_path / "level", tmp_path / "temp", tmp_path / "cache"
    now = time.time()
    _write(level / "a.dae", b"<a/>", now - 100)
    _write(level / "b.dae", b"<b/>", now - 100)
    _write(temp / "a.cdae", b"A", now - 50)
    _write(temp / "b.cdae", b"B", now - 50)
    assert meshcache.collect(level, temp, cache) == (2, [])
    (temp / "a.cdae").unlink()
    (temp / "b.cdae").unlink()
    _write(level / "a.dae", b"<a/>", now - 10)          # rewritten, same bytes
    _write(level / "b.dae", b"<b changed/>", now - 10)  # changed
    _write(temp / "b.cdae", b"B2", now)                 # only b compiled again
    assert meshcache.collect(level, temp, cache) == (2, [])
    assert (cache / "a.cdae").read_bytes() == b"A" and (cache / "b.cdae").read_bytes() == b"B2"
    assert meshcache.apply(level, cache) == (2, [])


def test_apply_without_a_cache_does_nothing(tmp_path):
    _write(tmp_path / "level/x.dae", b"<x/>")
    assert meshcache.apply(tmp_path / "level", tmp_path / "nothing") == (0, ["x.dae"])


IMPOSTERS = (".imposter.dds", ".imposter_normals.dds")


def test_imposters_travel_with_their_mesh(tmp_path):
    # a custom tree's imposters are built by the game at runtime, like its .cdae: they ship beside the .dae
    level, temp, cache = tmp_path / "level", tmp_path / "temp", tmp_path / "cache"
    now = time.time()
    _write(level / "trees/a.dae", b"<a/>", now - 100)
    _write(level / "trees/b.dae", b"<b/>", now - 100)
    _write(temp / "trees/a.cdae", b"A", now - 50)
    _write(temp / "trees/a.dae.imposter.dds", b"a-color", now - 50)
    _write(temp / "trees/a.dae.imposter_normals.dds", b"a-normals", now - 50)
    _write(temp / "trees/b.cdae", b"B", now - 200)                  # compiled before the build: stale
    _write(temp / "trees/b.dae.imposter.dds", b"b-color", now - 50)  # its imposters are not to be trusted either
    got, missing = meshcache.collect(level, temp, cache)
    assert got == 1 and missing == ["trees/b.dae"]
    assert (cache / "trees/a.dae.imposter.dds").read_bytes() == b"a-color"
    assert (cache / "trees/a.dae.imposter_normals.dds").read_bytes() == b"a-normals"
    assert not (cache / "trees/b.dae.imposter.dds").exists()
    assert json.loads((cache / "manifest.json").read_text(encoding="utf-8")) == {  # the format is unchanged
        "trees/a.dae": hashlib.sha1(b"<a/>").hexdigest()}
    # a rebuild writes the same dae files again (newer, same bytes for a)
    _write(level / "trees/a.dae", b"<a/>")
    _write(level / "trees/b.dae", b"<b/>")
    assert meshcache.apply(level, cache) == (1, ["trees/b.dae"])
    dae = level / "trees/a.dae"
    for name, data in (("a.cdae", b"A"), ("a.dae.imposter.dds", b"a-color"),
                       ("a.dae.imposter_normals.dds", b"a-normals")):
        placed = level / "trees" / name
        assert placed.read_bytes() == data
        assert placed.stat().st_mtime >= dae.stat().st_mtime  # the game must see each as newer than the .dae
    assert not list((level / "trees").glob("b.*dds")) and not (level / "trees/b.cdae").exists()


def test_imposters_are_read_from_their_own_folder(tmp_path):
    # where the game writes imposters need not be where it writes the .cdae: imposter_root says
    level, temp, found, cache = (tmp_path / n for n in ("level", "temp", "found", "cache"))
    now = time.time()
    _write(level / "a.dae", b"<a/>", now - 100)
    _write(temp / "a.cdae", b"A", now - 50)
    _write(temp / "a.dae.imposter.dds", b"wrong", now - 50)          # not where they were looked for
    _write(found / "a.dae.imposter.dds", b"color", now - 50)
    assert meshcache.collect(level, temp, cache, imposter_root=found) == (1, [])
    assert (cache / "a.dae.imposter.dds").read_bytes() == b"color"
    assert not (cache / "a.dae.imposter_normals.dds").exists()
    assert meshcache.count_imposters(cache) == 1


def test_imposters_only_ship_while_the_dae_is_unchanged(tmp_path):
    level, temp, cache = tmp_path / "level", tmp_path / "temp", tmp_path / "cache"
    now = time.time()
    _write(level / "a.dae", b"<a/>", now - 100)
    _write(temp / "a.cdae", b"A", now - 50)
    for s in IMPOSTERS:
        _write(temp / f"a.dae{s}", b"old", now - 50)
    assert meshcache.collect(level, temp, cache) == (1, [])
    _write(level / "a.dae", b"<a changed/>")
    assert meshcache.apply(level, cache) == (0, ["a.dae"])
    assert not list(level.glob("*.dds"))                       # an imposter of the old shape would be wrong


def test_collect_keeps_imposters_while_the_dae_is_unchanged_and_drops_them_when_it_changes(tmp_path):
    level, temp, cache = tmp_path / "level", tmp_path / "temp", tmp_path / "cache"
    now = time.time()
    _write(level / "a.dae", b"<a/>", now - 100)
    _write(temp / "a.cdae", b"A", now - 50)
    for s in IMPOSTERS:
        _write(temp / f"a.dae{s}", b"first", now - 50)
    assert meshcache.collect(level, temp, cache) == (1, [])
    assert meshcache.count_imposters(cache) == 2
    # the game loads the shipped copies next time and writes nothing: collecting again keeps them
    for f in temp.iterdir():
        f.unlink()
    _write(level / "a.dae", b"<a/>", now - 10)                 # rewritten, same bytes
    assert meshcache.collect(level, temp, cache) == (1, [])
    assert meshcache.count_imposters(cache) == 2
    # the .dae changed and was compiled again, but no imposter was made for the new shape
    _write(level / "a.dae", b"<a changed/>", now - 5)
    _write(temp / "a.cdae", b"A2", now)
    assert meshcache.collect(level, temp, cache) == (1, [])
    assert meshcache.count_imposters(cache) == 0
    assert (cache / "a.cdae").read_bytes() == b"A2"


def test_count_imposters_counts_only_the_manifest_meshes(tmp_path):
    cache = tmp_path / "cache"
    assert meshcache.count_imposters(cache) == 0               # no cache yet
    _write(cache / "manifest.json", json.dumps({"a.dae": "x"}).encode())
    _write(cache / "a.dae.imposter.dds", b"1")
    _write(cache / "a.dae.imposter_normals.dds", b"2")
    _write(cache / "gone.dae.imposter.dds", b"3")              # its mesh is no longer in the manifest
    assert meshcache.count_imposters(cache) == 2


# --- imposters are made again when what they picture changes -----------------------------------
# An imposter is a rendered picture of the tree: besides the .dae it depends on the shape's asset.json,
# the materials the .dae binds and the textures those use. Their fingerprint (imposters.json) decides.

TREE = "art/shapes/trees/br_swamp/t.dae"
TREE_DAE = ('<COLLADA><library_materials><material id="br_moss-material" name="br_moss">'
            '<instance_effect url="#br_moss-effect"/></material><material id="bark-material" name="bark">'
            '<instance_effect url="#bark-effect"/></material></library_materials></COLLADA>').encode()
ASSET = b'{"imposters": [{"dimension": 256, "size": 25}]}'


def tree_level(level, mtime=None, texture=b"DDS moss", factor=(1, 1, 1, 1), asset=ASSET, other=(1, 1, 1, 1)):
    """A level folder (levels/<name>/) with one tree shape and its asset.json, as the level stage writes it:
    the shape binds br_moss, whose texture ships in the level as a .dds under its .png name, and bark, a
    stock texture; `other` is a material no shape here binds."""
    if level.exists():
        shutil.rmtree(level)                             # write_level starts the level folder afresh
    _write(level / TREE, TREE_DAE, mtime)
    _write(level / (TREE + ".asset.json"), asset, mtime)
    mats = {"br_moss": {"name": "br_moss", "mapTo": "br_moss", "class": "Material", "Stages": [
                {"baseColorMap": f"/levels/{level.name}/art/textures/br/moss_b.color.png",
                 "baseColorFactor": list(factor)}, {}, {}, {}], "alphaTest": True},
            "bark": {"name": "bark", "mapTo": "bark", "class": "Material", "Stages": [
                {"baseColorMap": "/assets/materials/tree/fir/t_fir_bark/t_fir_bark_b.color.png"}, {}, {}, {}]},
            "other": {"name": "other", "mapTo": "other", "class": "Material", "Stages": [
                {"baseColorFactor": list(other)}, {}, {}, {}]}}
    _write(level / "art/shapes/trees/br_trees.materials.json", json.dumps(mats).encode(), mtime)
    _write(level / "art/textures/br/moss_b.color.dds", texture, mtime)
    return level / TREE


def collected_tree(tmp_path):
    """(level, temp, cache) after the game compiled the tree and made its imposters, and collect took them."""
    level, temp, cache = tmp_path / "levels" / "lvl", tmp_path / "temp", tmp_path / "cache"
    now = time.time()
    tree_level(level, now - 100)
    _write(temp / TREE.replace(".dae", ".cdae"), b"T", now - 50)
    for s in IMPOSTERS:
        _write(temp / (TREE + s), b"picture", now - 50)
    assert meshcache.collect(level, temp, cache) == (1, [])
    return level, temp, cache


def recorded(cache):
    return json.loads((cache / "imposters.json").read_text(encoding="utf-8"))


def placed(level):
    return sorted(p.name for p in (level / TREE).parent.iterdir() if p.suffix in (".cdae", ".dds"))


SHIPPED = ["t.cdae", "t.dae.imposter.dds", "t.dae.imposter_normals.dds"]


def _withheld_after(tmp_path, caplog, **change):
    level, _, cache = collected_tree(tmp_path)
    assert recorded(cache) == {TREE: meshcache.fingerprint(level, TREE)}
    tree_level(level)                                    # rebuilt the same: the imposters still picture it
    assert meshcache.apply(level, cache) == (1, [])
    assert placed(level) == SHIPPED
    tree_level(level, **change)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="brmap.meshcache"):
        assert meshcache.apply(level, cache) == (1, [])  # the .dae is the same: its .cdae ships
    assert placed(level) == ["t.cdae"]                   # and the game renders the imposters at load
    assert "0 pairs shipped, 1 withheld" in caplog.text and TREE in caplog.text


def test_apply_withholds_imposters_when_a_bound_texture_changed(tmp_path, caplog):
    _withheld_after(tmp_path, caplog, texture=b"DDS moss, paler")


def test_apply_withholds_imposters_when_a_bound_material_changed(tmp_path, caplog):
    _withheld_after(tmp_path, caplog, factor=(0.65, 0.65, 0.5, 1))


def test_apply_withholds_imposters_when_the_asset_json_changed(tmp_path, caplog):
    _withheld_after(tmp_path, caplog, asset=b'{"imposters": [{"dimension": 512, "size": 25}]}')


def test_apply_ships_imposters_when_only_an_unbound_material_changed(tmp_path, caplog):
    level, _, cache = collected_tree(tmp_path)
    tree_level(level, other=(0.2, 0.2, 0.2, 1))
    with caplog.at_level(logging.INFO, logger="brmap.meshcache"):
        assert meshcache.apply(level, cache) == (1, [])
    assert placed(level) == SHIPPED
    assert "1 pairs shipped, 0 withheld" in caplog.text


def test_fingerprint_reads_the_dds_the_game_loads_for_a_png_name(tmp_path):
    level = tmp_path / "levels" / "lvl"
    tree_level(level)
    first = meshcache.fingerprint(level, TREE)
    (level / "art/textures/br/moss_b.color.png").write_bytes(b"a png the game does not load")
    assert meshcache.fingerprint(level, TREE) == first
    (level / "art/textures/br/moss_b.color.dds").write_bytes(b"DDS changed")
    assert meshcache.fingerprint(level, TREE) != first
    (level / (TREE + ".asset.json")).unlink()
    assert meshcache.fingerprint(level, TREE) is None   # no asset.json: the game makes no imposters


def test_apply_withholds_imposters_collected_before_fingerprints(tmp_path):
    # a cache from before imposters.json: what its imposters picture is unknown, so the game renders them
    level, _, cache = collected_tree(tmp_path)
    (cache / "imposters.json").unlink()
    tree_level(level)
    assert meshcache.apply(level, cache) == (1, [])
    assert placed(level) == ["t.cdae"]


def test_collect_takes_new_imposters_for_an_unchanged_dae(tmp_path):
    level, temp, cache = tmp_path / "levels" / "lvl", tmp_path / "temp", tmp_path / "cache"
    now = time.time()
    tree_level(level, now - 100)
    _write(temp / TREE.replace(".dae", ".cdae"), b"T", now - 50)
    assert meshcache.collect(level, temp, cache) == (1, [])          # compiled, but no imposters made yet
    assert meshcache.count_imposters(cache) == 0
    # rebuilt with the same .dae: its .cdae ships, so the game compiles nothing and only renders the imposters
    tree_level(level, now - 20)
    assert meshcache.apply(level, cache) == (1, [])
    for s in IMPOSTERS:
        _write(temp / (TREE + s), b"before this build", now - 30)    # older than the .dae: not this level's
    assert meshcache.collect(level, temp, cache) == (1, [])
    assert meshcache.count_imposters(cache) == 0 and recorded(cache) == {}
    for s in IMPOSTERS:
        _write(temp / (TREE + s), b"fresh", now - 10)
    assert meshcache.collect(level, temp, cache) == (1, [])
    assert [(cache / (TREE + s)).read_bytes() for s in IMPOSTERS] == [b"fresh", b"fresh"]
    assert (cache / TREE.replace(".dae", ".cdae")).read_bytes() == b"T"
    assert recorded(cache) == {TREE: meshcache.fingerprint(level, TREE)}
    tree_level(level)
    assert meshcache.apply(level, cache) == (1, [])
    assert placed(level) == SHIPPED


def test_collect_keeps_the_fingerprint_of_the_imposters_it_keeps(tmp_path):
    # the game loaded the shipped imposters and made none: the cached ones still picture what they did
    level, temp, cache = collected_tree(tmp_path)
    before = recorded(cache)
    for f in [p for p in temp.rglob("*") if p.is_file()]:
        f.unlink()
    tree_level(level, texture=b"DDS moss, paler")        # changed since: the next apply must still withhold them
    assert meshcache.collect(level, temp, cache) == (1, [])
    assert recorded(cache) == before and meshcache.count_imposters(cache) == 2
    tree_level(level, texture=b"DDS moss, paler")
    meshcache.apply(level, cache)
    assert placed(level) == ["t.cdae"]


def test_collect_drops_the_fingerprint_with_the_imposters(tmp_path):
    level, temp, cache = collected_tree(tmp_path)
    now = time.time()
    for f in [p for p in temp.rglob("*") if p.is_file()]:
        f.unlink()
    tree_level(level, now - 5)
    _write(level / TREE, TREE_DAE.replace(b"bark", b"bark2"), now - 5)  # a changed shape, compiled, no imposters yet
    _write(temp / TREE.replace(".dae", ".cdae"), b"T2", now)
    assert meshcache.collect(level, temp, cache) == (1, [])
    assert meshcache.count_imposters(cache) == 0 and recorded(cache) == {}


def test_imposter_status_names_the_trees_without_imposters_to_ship(tmp_path):
    level, _, cache = collected_tree(tmp_path)
    assert meshcache.imposter_status(level, cache) == ([TREE], [])
    tree_level(level, texture=b"DDS moss, paler")
    assert meshcache.imposter_status(level, cache) == ([], [TREE])
    other = "art/shapes/trees/br_swamp/u.dae"            # an imposter tree the game has not loaded yet
    _write(level / other, TREE_DAE)
    _write(level / (other + ".asset.json"), ASSET)
    assert meshcache.imposter_status(level, cache) == ([], [TREE, other])


def test_collect_meshes_reports_imposter_pairs_and_warns_for_the_missing(tmp_path, monkeypatch, capsys):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "collect_meshes", Path(__file__).resolve().parent.parent / "tools" / "ingame" / "collect_meshes.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    dist, out, local = tmp_path / "dist", tmp_path / "out", tmp_path / "local"
    config = tmp_path / "c.yaml"
    paths = {"raw": (tmp_path / "raw").as_posix(), "out": out.as_posix(), "dist": dist.as_posix()}
    config.write_text(json.dumps({"level": {"name": "lvl"}, "paths": paths}), encoding="utf-8")  # JSON is YAML
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    level = dist / "level" / "levels" / "lvl"
    temp = local / "BeamNG" / "BeamNG.drive" / "current" / "temp" / "levels" / "lvl"
    now = time.time()
    tree_level(level, now - 100)
    other = "art/shapes/trees/br_swamp/u.dae"            # compiled, but the game made no imposters for it
    _write(level / other, TREE_DAE.replace(b"bark", b"bark2"), now - 100)
    _write(level / (other + ".asset.json"), ASSET, now - 100)
    for rel in (TREE, other):
        _write(temp / rel.replace(".dae", ".cdae"), b"C", now - 50)
    for s in IMPOSTERS:
        _write(temp / (TREE + s), b"picture", now - 50)
    assert tool.main(str(config)) == 0
    printed = capsys.readouterr().out
    assert "collected 2 compiled meshes and 1 imposter pairs" in printed
    assert f"warning: {other} has no cached imposters" in printed and f"warning: {TREE}" not in printed
