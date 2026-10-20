"""The Memorial Tower's model script (blender/landmarks/memorial_tower.py): its plain-Python
geometry, built without Blender. (The script is loaded as a module with brexport stubbed out; the
Blender run itself is covered by the models stage and the in-game spike.)"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

from brmap import landmarks, materials

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "blender" / "lib"))
import brlm  # noqa: E402

# the params the models stage hands the script (out_lsu/models/memorial_tower/params.json, 2026-10-19)
PARAMS = {"height_m": 53.3, "shaft_m": [8.0, 6.6], "ground_dz": 0.66,
          "footprint_local": [(6.735, 9.211), (-9.725, 9.11), (-9.706, 5.63), (-22.206, 5.548), (-22.143, -6.66),
                              (-9.631, -6.579), (-9.625, -6.752), (6.835, -6.651), (6.828, -6.487), (18.097, -6.412),
                              (18.033, 5.797), (6.764, 5.731)]}
UNIT_NORMALS = {(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)}


class Stop(Exception):
    """Raised by the stub brexport.args, to end the script's main() before it needs Blender."""


@pytest.fixture(scope="module")
def tower():
    stub = types.ModuleType("brexport")

    def args():
        raise Stop

    stub.args = args
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "brexport", stub)
        spec = importlib.util.spec_from_file_location("memorial_tower", ROOT / "blender/landmarks/memorial_tower.py")
        module = importlib.util.module_from_spec(spec)
        with pytest.raises(Stop):
            spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def dims(tower):
    return tower.resolve(PARAMS)


def plane(pts):
    """(unit normal, offset) of a face, rounded so equal planes compare equal."""
    n = brlm.normal(pts)
    return tuple(round(c, 6) + 0.0 for c in n), round(brlm.dot(n, pts[0]), 6) + 0.0


def test_levels_of_detail_stay_in_budget(tower, dims):
    assert tower.BUDGETS == {400: 6000, 100: 800, 25: 100} and tower.COLLISION_BUDGET == 60
    assert tower.high(dims).tris <= 6000
    assert tower.mid(dims).tris <= 800
    assert tower.low(dims).tris <= 100
    assert tower.collision(dims).tris <= 60


def test_the_four_dials_share_one_clock_material(tower, dims):
    for build in (tower.high, tower.mid):
        b = build(dims)
        dials = [pts for pts, _, mat in b.faces if mat == "br_lm_clock"]
        assert len(dials) == 4
        assert {tuple(round(c) for c in brlm.normal(pts)) for pts in dials} == UNIT_NORMALS   # one on each side
    for build in (tower.low, tower.collision):
        assert "br_lm_clock" not in build(dims).materials


def test_every_material_it_binds_is_in_the_registry(tower, dims):
    registry = materials.load_registry(landmarks.MATERIAL_REGISTRY)
    for build in (tower.high, tower.mid, tower.low, tower.collision):
        assert build(dims).materials <= set(registry), build.__name__


def pilaster_faces(tower, dims):
    """(the pilaster faces, every other face of the full level of detail)."""
    pilasters = brlm.Builder()
    tower.pilasters(pilasters, dims["hu"], dims["hv"], dims["g"] + tower.PLINTH_TOP,
                    dims["g"] + tower.SHAFT_TOP - tower.COURSE)
    mine = {tuple(pts) for pts, _, _ in pilasters.faces}
    faces = tower.high(dims).faces
    ours = [pts for pts, _, _ in faces if tuple(pts) in mine]
    others = [pts for pts, _, _ in faces if tuple(pts) not in mine]
    return ours, others


def test_pilasters_read_as_solid_strips(tower, dims):
    # the strip of a pilaster's side face beside the wall was 0.15 m: too thin to draw cleanly at a distance
    ours, _ = pilaster_faces(tower, dims)
    assert len(ours) == 16                                           # four corners, four visible faces
    stands_out = max(abs(c[0]) for pts in ours for c in pts) - dims["hu"]
    assert stands_out >= 0.25 - 1e-9


def touch(a, b, eps=1e-6):
    """Whether the bounding boxes of two faces meet or overlap."""
    return all(min(p[i] for p in a) <= max(p[i] for p in b) + eps and
               min(p[i] for p in b) <= max(p[i] for p in a) + eps for i in range(3))


def test_pilasters_share_no_plane_with_the_faces_beside_them(tower, dims):
    # a pilaster face in the plane of a neighbouring shaft, plinth or string-course face facing the
    # same way would z-fight with it where they overlap, and read as a seam where they meet
    # (the dials hang in that plane too, but 38 m up and well clear of the pilasters: not neighbours)
    ours, others = pilaster_faces(tower, dims)
    clash = [(pts[0], other[0]) for pts in ours for other in others if plane(pts) == plane(other) and touch(pts, other)]
    assert not clash
