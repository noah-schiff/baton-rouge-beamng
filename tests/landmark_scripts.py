"""What the landmark model script tests share: loading a script (blender/landmarks/<name>.py) as a
module without Blender (brexport is stubbed out; its args() ends the script's main() before
anything needs Blender), the real params the models stage hands it (tests/fixtures/<name>_params.json,
from out_lsu/landmarks.json) and the checks every script's parts pass."""
import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import pytest

from brmap import landmarks, materials, validate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "blender" / "lib"))


class Stop(Exception):
    """Raised by the stub brexport.args, to end the script's main() before it needs Blender."""


def load_script(name: str):
    stub = types.ModuleType("brexport")

    def args():
        raise Stop

    stub.args = args
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "brexport", stub)
        spec = importlib.util.spec_from_file_location(name, ROOT / "blender" / "landmarks" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        with pytest.raises(Stop):
            spec.loader.exec_module(module)
    return module


def fixture_params(name: str) -> dict:
    return json.loads((ROOT / "tests" / "fixtures" / f"{name}_params.json").read_text(encoding="utf-8"))


def levels(part):
    """(name, Builder) of every level of detail of a part and its collision mesh."""
    return [(f"a{px}", b) for px, _, b in part["lods"]] + [("Colmesh-1", part["collision"][1])]


def check_parts(built: dict, lods) -> None:
    """Every part has the levels `lods` (largest first) and a collision mesh, each under the
    vertex cap and its triangle budget, the levels getting cheaper; node names the engine reads
    right; every material bound is in the landmark material registry."""
    registry = materials.load_registry(landmarks.MATERIAL_REGISTRY)
    for name, part in built.items():
        assert [px for px, _, _ in part["lods"]] == list(lods), name
        tris = [b.tris for _, _, b in part["lods"]]
        assert tris == sorted(tris, reverse=True) and tris[0] > tris[-1], (name, tris)
        for px, budget, b in part["lods"]:
            assert b.tris <= budget, (name, px, b.tris, budget)
        budget, col = part["collision"]
        assert col.faces and col.tris <= budget, (name, col.tris, budget)
        for level, b in levels(part):
            assert b.corners <= validate.DAE_MAX_VERTS, (name, level, b.corners)
            assert b.materials <= set(registry), (name, level, b.materials - set(registry))
        for px in lods:
            node = f"{part['node']}_a{px}"
            assert re.fullmatch(r"[a-z]+(_[a-z]+)*_a\d+", node) and not re.search(r"_\d+$", node), node


def close(c, want, tol=1.5 / 255):
    return c is not None and all(abs(a - b) <= tol for a, b in zip(c, want))
