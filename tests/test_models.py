"""The Blender runner and the models stage (models.py): the run logic against a fake Blender
(tests/fake_blender.py), and the smoke cube and brexport.py against the real one."""
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from brmap import __main__ as cli
from brmap import collada, models, validate
from brmap.config import Config

ROOT = Path(__file__).resolve().parent.parent
FAKE = Path(__file__).with_name("fake_blender.py")
FAKE_PREFIX = [sys.executable, "-B", str(FAKE)]
needs_blender = pytest.mark.skipif(not Path(Config().blender_exe).exists(), reason="Blender is not installed")


@pytest.fixture
def script(tmp_path):
    """A stand-in model script: the fake Blender never runs it, but its bytes go into the key."""
    path = tmp_path / "fake_script.py"
    path.write_text("# stands in for a Blender script\n", encoding="utf-8")
    return path


@pytest.fixture
def fake_blender(monkeypatch):
    """models.run launches the fake instead of blender.exe."""
    monkeypatch.setattr(models, "blender_prefix", lambda cfg: FAKE_PREFIX)
    monkeypatch.setattr(models, "blender_version", lambda prefix: "fake 1")


def calls(out_dir):
    """What the fake Blender was run with in this out dir: [{"script", "params"}, ...]."""
    path = Path(out_dir) / "count.txt"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def count(out_dir):
    return len(calls(out_dir))


def sha1(path):
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def save(tmp_path, xml):
    path = tmp_path / "shape.dae"
    path.write_text(xml, encoding="utf-8")
    return path


# ------------------------------------------------------------------ the command line

def test_blender_prefix():
    cfg = Config(blender_exe="D:/tools/blender.exe")
    assert models.blender_prefix(cfg) == ["D:/tools/blender.exe", "--background", "--factory-startup",
                                          "--python-exit-code", "1"]


def test_blender_version_is_the_first_line_and_cached(monkeypatch):
    monkeypatch.setattr(models, "_versions", {})
    runs = []
    real = subprocess.run
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: runs.append(a[0]) or real(*a, **k))
    assert models.blender_version(FAKE_PREFIX) == "Blender 9.9.9 (fake)"
    assert models.blender_version(FAKE_PREFIX) == "Blender 9.9.9 (fake)"
    assert runs == [[*FAKE_PREFIX, "--version"]]


def test_blender_version_names_a_missing_exe(monkeypatch):
    monkeypatch.setattr(models, "_versions", {})
    with pytest.raises(FileNotFoundError, match="nowhere.exe"):
        models.blender_version(["Z:/nowhere.exe", "--background"])


# ------------------------------------------------------------------ the key

def test_model_key_covers_script_lib_params_and_version(tmp_path, monkeypatch, script):
    lib = tmp_path / "lib"
    (lib / "sub").mkdir(parents=True)
    (lib / "a.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(models, "LIB_DIR", lib)

    key = models.model_key(script, {"a": 1, "b": 2}, "v1")
    assert len(key) == 40 and int(key, 16) >= 0
    assert models.model_key(script, {"b": 2, "a": 1}, "v1") == key        # param order is not a change
    assert models.model_key(script, {"a": 1, "b": 3}, "v1") != key
    assert models.model_key(script, {"a": 1, "b": 2}, "v2") != key

    (lib / "__pycache__").mkdir()
    (lib / "__pycache__" / "a.cpython-313.pyc").write_bytes(b"\0\1")        # compiled files are no change
    assert models.model_key(script, {"a": 1, "b": 2}, "v1") == key

    (lib / "sub" / "b.py").write_text("y = 2\n", encoding="utf-8")          # a file in a subfolder
    with_sub = models.model_key(script, {"a": 1, "b": 2}, "v1")
    assert with_sub != key
    (lib / "a.py").write_text("x = 2\n", encoding="utf-8")                  # a lib file's bytes
    with_edit = models.model_key(script, {"a": 1, "b": 2}, "v1")
    assert with_edit != with_sub
    script.write_text("# another script\n", encoding="utf-8")               # the script's bytes
    assert models.model_key(script, {"a": 1, "b": 2}, "v1") != with_edit


def test_model_key_reads_the_repos_blender_lib():
    assert models.LIB_DIR == ROOT / "blender" / "lib" and (models.LIB_DIR / "brexport.py").is_file()


# ------------------------------------------------------------------ build

def test_build_runs_once(tmp_path, script):
    paths = models.build(FAKE_PREFIX, script, {"h": 1}, tmp_path / "m", "v1")
    assert [p.name for p in paths] == ["main.npz"]
    models.build(FAKE_PREFIX, script, {"h": 1}, tmp_path / "m", "v1")
    assert count(tmp_path / "m") == 1
    models.build(FAKE_PREFIX, script, {"h": 2}, tmp_path / "m", "v1")
    assert count(tmp_path / "m") == 2


def test_build_passes_the_script_and_the_params(tmp_path, script):
    models.build(FAKE_PREFIX, script, {"h": 1, "name": "x"}, tmp_path / "m", "v1")
    assert calls(tmp_path / "m") == [{"script": str(script), "params": {"h": 1, "name": "x"}}]
    assert (tmp_path / "m" / "key.txt").read_text(encoding="utf-8").strip() == models.model_key(
        script, {"h": 1, "name": "x"}, "v1")


def test_build_returns_the_parts_sorted(tmp_path, script):
    paths = models.build(FAKE_PREFIX, script, {"parts": ["b", "a", "c"]}, tmp_path / "m", "v1")
    assert [p.name for p in paths] == ["a.npz", "b.npz", "c.npz"]


def test_build_rebuilds_for_a_new_version(tmp_path, script):
    models.build(FAKE_PREFIX, script, {}, tmp_path / "m", "v1")
    models.build(FAKE_PREFIX, script, {}, tmp_path / "m", "v2")
    assert count(tmp_path / "m") == 2


def test_build_rebuilds_when_the_parts_are_gone(tmp_path, script):
    (path,) = models.build(FAKE_PREFIX, script, {}, tmp_path / "m", "v1")
    path.unlink()
    assert [p.name for p in models.build(FAKE_PREFIX, script, {}, tmp_path / "m", "v1")] == ["main.npz"]
    assert count(tmp_path / "m") == 2


def test_build_drops_the_parts_an_earlier_build_made(tmp_path, script):
    models.build(FAKE_PREFIX, script, {"parts": ["a", "b"]}, tmp_path / "m", "v1")
    paths = models.build(FAKE_PREFIX, script, {"parts": ["a"]}, tmp_path / "m", "v1")
    assert [p.name for p in paths] == ["a.npz"] and not (tmp_path / "m" / "b.npz").exists()


def test_build_failure_names_the_script(tmp_path, script):
    with pytest.raises(RuntimeError, match="fake_script"):
        models.build(FAKE_PREFIX, script, {"fail": True}, tmp_path / "m", "v1")


def test_build_failure_shows_the_last_20_lines_of_stderr(tmp_path, script):
    with pytest.raises(RuntimeError) as err:
        models.build(FAKE_PREFIX, script, {"fail": True}, tmp_path / "m", "v1")
    lines = str(err.value).splitlines()
    assert "fake blender: boom" in lines and "noise 29" in lines and "noise 11" in lines
    assert "noise 10" not in lines          # 20 lines: noise 11..29 and the last one


def test_build_without_npz_raises(tmp_path, script):
    with pytest.raises(RuntimeError, match=r"fake_script.*\.npz"):
        models.build(FAKE_PREFIX, script, {"empty": True}, tmp_path / "m", "v1")
    assert not (tmp_path / "m" / "key.txt").exists()


def test_a_failed_rebuild_leaves_no_key(tmp_path, script):
    out = tmp_path / "m"
    models.build(FAKE_PREFIX, script, {"n": 1}, out, "v1")
    with pytest.raises(RuntimeError):
        models.build(FAKE_PREFIX, script, {"n": 2, "fail": True}, out, "v1")
    assert not (out / "key.txt").exists()
    models.build(FAKE_PREFIX, script, {"n": 1}, out, "v1")      # the first build's key is gone: runs again
    assert count(out) == 3


# ------------------------------------------------------------------ the stage

def write_landmarks(out_dir, entries):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "landmarks.json").write_text(json.dumps(entries), encoding="utf-8")


def write_trees(monkeypatch, tmp_path, entries):
    path = tmp_path / "custom_trees.yaml"
    path.write_text(json.dumps(entries), encoding="utf-8")      # json is yaml
    monkeypatch.setattr(models, "CUSTOM_TREES", path)


def test_run_builds_landmarks_and_trees(tmp_path, monkeypatch, script, fake_blender):
    out = tmp_path / "out"
    write_landmarks(out, [
        {"name": "tower", "model": script.as_posix(), "params": {"floors": 3}},
        {"name": "shell", "model": "styled", "params": {}},        # styled landmarks have no Blender script
        {"name": "cube", "model": "blender/smoke/cube.py", "params": {}},   # relative to the repo
    ])
    write_trees(monkeypatch, tmp_path, {"cypress": {"script": script.as_posix(), "params": {"n": 2},
                                                    "native_height": 20.0, "billboard": True}})
    models.run(Config(out_dir=str(out)))
    assert [p.name for p in (out / "models" / "tower").glob("*.npz")] == ["main.npz"]
    assert calls(out / "models" / "tower") == [{"script": str(script), "params": {"floors": 3}}]
    assert calls(out / "models" / "cypress") == [{"script": str(script), "params": {"n": 2}}]
    assert Path(calls(out / "models" / "cube")[0]["script"]) == ROOT / "blender" / "smoke" / "cube.py"
    assert not (out / "models" / "shell").exists()


def test_run_is_incremental(tmp_path, monkeypatch, script, fake_blender):
    out = tmp_path / "out"
    write_landmarks(out, [{"name": "tower", "model": script.as_posix(), "params": {}}])
    write_trees(monkeypatch, tmp_path, {})
    cfg = Config(out_dir=str(out))
    models.run(cfg)
    models.run(cfg)
    assert count(out / "models" / "tower") == 1


def test_run_skips_missing_inputs_with_a_log_line(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(models, "CUSTOM_TREES", tmp_path / "no_custom_trees.yaml")
    monkeypatch.setattr(models, "blender_prefix", lambda cfg: pytest.fail("Blender is not needed"))
    caplog.set_level(logging.INFO, logger="brmap.models")
    models.run(Config(out_dir=str(tmp_path / "out")))
    text = caplog.text
    assert "landmarks.json" in text and "no_custom_trees.yaml" in text


def test_run_builds_what_exists_when_one_input_is_missing(tmp_path, monkeypatch, script, fake_blender, caplog):
    out = tmp_path / "out"
    out.mkdir()
    write_trees(monkeypatch, tmp_path, {"oak": {"script": script.as_posix(), "params": {}}})
    caplog.set_level(logging.INFO, logger="brmap.models")
    models.run(Config(out_dir=str(out)))
    assert count(out / "models" / "oak") == 1 and "landmarks.json" in caplog.text


def test_run_builds_no_custom_trees_with_the_swamp_trees_off(tmp_path, monkeypatch, script, fake_blender, caplog):
    # trees.swamp off places no custom tree, so it needs neither the models nor Blender
    out = tmp_path / "out"
    write_landmarks(out, [{"name": "tower", "model": script.as_posix(), "params": {}}])
    write_trees(monkeypatch, tmp_path, {"cypress": {"script": script.as_posix(), "params": {}}})
    caplog.set_level(logging.INFO, logger="brmap.models")
    models.run(Config(out_dir=str(out), swamp_trees=False))
    assert count(out / "models" / "tower") == 1 and not (out / "models" / "cypress").exists()
    assert "trees.swamp" in caplog.text
    monkeypatch.setattr(models, "blender_prefix", lambda cfg: pytest.fail("Blender is not needed"))
    write_landmarks(out, [])
    models.run(Config(out_dir=str(out), swamp_trees=False))


def write_moss_points(monkeypatch, tmp_path, points):
    path = tmp_path / "moss_points.json"
    path.write_text(json.dumps(points), encoding="utf-8")
    monkeypatch.setattr(models, "MOSS_POINTS", path)


def test_run_builds_moss_overlays_on_their_own_gate(tmp_path, monkeypatch, script, fake_blender, caplog):
    # the moss overlays hang on the stock live oaks, so they follow trees.moss_share, not trees.swamp; their
    # hang_points_from param becomes the points moss_points.json holds for that oak
    out = tmp_path / "out"
    out.mkdir()
    write_trees(monkeypatch, tmp_path, {
        "cypress": {"script": script.as_posix(), "params": {"seed": 1}},
        "br_moss_oak": {"script": script.as_posix(), "params": {"hang_points_from": "br_oak", "seed": 3}}})
    write_moss_points(monkeypatch, tmp_path, {"br_oak": [[1.0, 2.0, 5.0], [0.5, -1.0, 6.0]], "br_other": [[0, 0, 1]]})
    caplog.set_level(logging.INFO, logger="brmap.models")
    models.run(Config(out_dir=str(out), swamp_trees=False, moss_share=0.3))
    assert calls(out / "models" / "br_moss_oak") == [
        {"script": str(script), "params": {"hang_points": [[1.0, 2.0, 5.0], [0.5, -1.0, 6.0]], "seed": 3}}]
    assert not (out / "models" / "cypress").exists()

    models.run(Config(out_dir=str(tmp_path / "out2"), swamp_trees=True, moss_share=0.0))
    assert count(tmp_path / "out2" / "models" / "cypress") == 1
    assert not (tmp_path / "out2" / "models" / "br_moss_oak").exists() and "moss_share" in caplog.text


def test_run_names_an_oak_without_moss_points(tmp_path, monkeypatch, script, fake_blender):
    write_trees(monkeypatch, tmp_path, {"br_moss_x": {"script": script.as_posix(),
                                                      "params": {"hang_points_from": "br_no_such_oak"}}})
    write_moss_points(monkeypatch, tmp_path, {"br_oak": [[1.0, 2.0, 5.0]]})
    with pytest.raises(ValueError, match="br_no_such_oak"):
        models.run(Config(out_dir=str(tmp_path / "out")))


def test_the_registrys_moss_overlays_have_their_points():
    points = json.loads(models.MOSS_POINTS.read_text(encoding="utf-8"))
    entries = yaml.safe_load(models.CUSTOM_TREES.read_text(encoding="utf-8"))
    moss = {k: e for k, e in entries.items() if models.MOSS_PARAM in (e.get("params") or {})}
    assert len(moss) == 4 and all(e["params"][models.MOSS_PARAM] in points for e in moss.values())


def test_run_refuses_two_shapes_with_one_name(tmp_path, monkeypatch, script, fake_blender):
    out = tmp_path / "out"
    write_landmarks(out, [{"name": "oak", "model": script.as_posix(), "params": {}}])
    write_trees(monkeypatch, tmp_path, {"oak": {"script": script.as_posix(), "params": {}}})
    with pytest.raises(ValueError, match="oak"):
        models.run(Config(out_dir=str(out)))


def test_models_stage_is_registered_after_canopy():
    stages = cli.FEATURE_STAGES
    assert stages.index("canopy") < stages.index("models") < stages.index("buildings")
    assert "models" in cli.STAGES and "models" in cli.ALL_ORDER
    assert models.__doc__


# ------------------------------------------------------------------ Blender itself

@needs_blender
def test_smoke_cube_is_deterministic(tmp_path):
    cfg = Config()
    a = models.build(models.blender_prefix(cfg), ROOT / "blender/smoke/cube.py", {}, tmp_path / "a", "x")
    b = models.build(models.blender_prefix(cfg), ROOT / "blender/smoke/cube.py", {}, tmp_path / "b", "x")
    assert sha1(a[0]) == sha1(b[0])
    children, _ = collada.read_npz(a[0])
    counts = validate.dae_vertex_counts(save(tmp_path, collada.shape_dae(children)))
    assert counts == {"cube_a100": 24, "Colmesh-1": 8}


@needs_blender
def test_smoke_cube_follows_the_npz_contract(tmp_path):
    cube = ROOT / "blender/smoke/cube.py"
    (path,) = models.build(models.blender_prefix(Config()), cube, {}, tmp_path / "m", "x")
    with np.load(path, allow_pickle=False) as z:
        assert z["meta/nodes"].tolist() == ["cube_a100", "Colmesh-1"]
        assert z["meta/billboard"].item() == "" and z["meta/blender"].item().startswith("5.")
        assert z["meta/script_sha1"].item() == sha1(cube)
        for node in ("cube_a100", "Colmesh-1"):
            assert z[f"{node}/pos"].shape == z[f"{node}/nrm"].shape == (36, 3)      # 12 triangles
            assert z[f"{node}/uv"].shape == (36, 2) and z[f"{node}/col"].shape == (36, 4)
            assert z[f"{node}/mat"].shape == (12,) and z[f"{node}/mat"].dtype == np.int32
            assert z[f"{node}/materials"].tolist() == ["br_smoke"]
            assert {z[f"{node}/{k}"].dtype for k in ("pos", "nrm", "uv", "col")} == {np.dtype(float)}
        cube_pos = z["cube_a100/pos"]
        # the object sits at z = 1 in Blender: the exported corners are in world space, 0..2 high
        np.testing.assert_allclose([cube_pos.min(0), cube_pos.max(0)], [[-1, -1, 0], [1, 1, 2]], atol=1e-6)
        # flat faces: six normals, each along an axis; the collision cube is smooth: (+-1, +-1, +-1) / sqrt 3
        assert len(np.unique(np.round(z["cube_a100/nrm"], 3), axis=0)) == 6
        np.testing.assert_allclose(np.abs(z["Colmesh-1/nrm"]), 1 / np.sqrt(3), atol=1e-6)
        # the colour bytes arrive unchanged (not converted to linear); no attribute means white, no uv zeros
        np.testing.assert_allclose(z["cube_a100/col"], np.tile([0.2, 0.4, 1.0, 1.0], (36, 1)), atol=1e-6)
        np.testing.assert_array_equal(z["Colmesh-1/col"], np.ones((36, 4)))
        np.testing.assert_array_equal(z["Colmesh-1/uv"], np.zeros((36, 2)))
        assert z["cube_a100/uv"].min() == 0.0 and z["cube_a100/uv"].max() == 1.0


BRIDGE = """
import json, sys
from pathlib import Path
sys.path.insert(0, {lib!r})
import bpy, brexport
params, out = brexport.args()


def triangle(name, faces=True, material=True):
    me = bpy.data.meshes.new(name)
    me.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)] if faces else [])
    if faces:
        for loop, uv in zip(me.uv_layers.new(name="UVMap").data, [(0, 0), (1, 0), (0, 1)]):
            loop.uv = uv
    if material:
        me.materials.append(brexport.material("br_t"))
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


bad = {{
    "underscore": {{"main": [("tri_100", triangle("a"))]}},
    "no_material": {{"main": [("tri_a100", triangle("b", material=False))]}},
    "no_faces": {{"main": [("tri_a100", triangle("c", faces=False))]}},
}}
errors = {{}}
for label, parts in bad.items():
    try:
        brexport.export(parts, out / label)
    except ValueError as e:
        errors[label] = str(e)
(out / "errors.json").write_text(json.dumps(errors), encoding="utf-8")

mirrored = triangle("m")
mirrored.scale = (-1, 1, 1)
mirrored.location = (0, 0, 5)
brexport.export({{"main": [("tri_a100", mirrored), ("nulldetail30", None)]}}, out, billboard="bb_autobillboard25")
"""


@needs_blender
def test_export_refuses_bad_input_and_applies_the_object_transform(tmp_path):
    script = tmp_path / "bridge.py"
    script.write_text(BRIDGE.format(lib=str(ROOT / "blender" / "lib")), encoding="utf-8")
    (path,) = models.build(models.blender_prefix(Config()), script, {}, tmp_path / "m", "x")
    errors = json.loads((tmp_path / "m" / "errors.json").read_text(encoding="utf-8"))
    assert set(errors) == {"underscore", "no_material", "no_faces"}
    assert "tri_100" in errors["underscore"] and "underscore" in errors["underscore"]
    assert "tri_a100" in errors["no_material"] and "material" in errors["no_material"]
    assert "tri_a100" in errors["no_faces"] and "faces" in errors["no_faces"]

    with np.load(path, allow_pickle=False) as z:
        assert z["meta/nodes"].tolist() == ["tri_a100", "nulldetail30"]
        assert z["meta/billboard"].item() == "bb_autobillboard25"
        assert "nulldetail30/pos" not in z.files
        pos, nrm, uv = z["tri_a100/pos"], z["tri_a100/nrm"], z["tri_a100/uv"]
    # x mirrored and lifted by 5; the winding flips with the mirror, so the face still looks up
    np.testing.assert_allclose(pos, [[0, 0, 5], [0, 1, 5], [-1, 0, 5]], atol=1e-6)
    assert np.cross(pos[1] - pos[0], pos[2] - pos[0])[2] > 0
    np.testing.assert_allclose(nrm, np.tile([0, 0, 1.0], (3, 1)), atol=1e-6)
    np.testing.assert_allclose(uv, [[0, 0], [0, 1], [1, 0]], atol=1e-6)   # the uv of each corner follows it
    children, bb = collada.read_npz(path)
    assert bb == "bb_autobillboard25" and [n for n, _ in children] == ["tri_a100", "nulldetail30"]
