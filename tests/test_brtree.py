"""The tree model helpers (blender/lib/brtree.py) and the bald cypress script
(blender/trees/bald_cypress.py): their plain-Python geometry without Blender, and the real
Blender build of the cypress (skipped without Blender)."""
import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from brmap import collada, models, trees, validate
from brmap.config import Config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "blender" / "lib"))
import brlm  # noqa: E402
import brtree  # noqa: E402

needs_blender = pytest.mark.skipif(not Path(Config().blender_exe).exists(), reason="Blender is not installed")
PARAMS = {"height_m": 22, "seed": 11, "knees": 8}          # custom_trees.yaml br_cypress_a
UP = (0.0, 0.0, 1.0)


def arrays(b):
    """(pos, nrm, uv, wind) of a TreeBuilder's vertices as arrays."""
    pos, nrm, uv, wind = (np.array([v[i] for v in b.verts], float) for i in range(4))
    return pos, nrm, uv, wind


def face_normals(b):
    pos = arrays(b)[0]
    tri = np.array([t[:3] for t in b.triangles])
    n = np.cross(pos[tri[:, 1]] - pos[tri[:, 0]], pos[tri[:, 2]] - pos[tri[:, 0]])
    return tri, n / np.linalg.norm(n, axis=1, keepdims=True)


# ------------------------------------------------------------------ brtree

def test_flare_widens_the_base():
    assert brtree.flare(0.0, 0.5) == pytest.approx(1.1)            # radius x 2.2 at the ground
    assert brtree.flare(-0.4, 0.5) == pytest.approx(1.1)           # and below it
    assert brtree.flare(1.5, 0.5) == pytest.approx(0.5)            # back to the trunk by 1.5 m
    assert brtree.flare(4.0, 0.5) == pytest.approx(0.5)
    zs = np.linspace(0, 1.5, 16)
    assert all(np.diff([brtree.flare(z, 0.5) for z in zs]) < 0)
    assert brtree.flare(0.0, 0.5, factor=3.0, height=2.0) == pytest.approx(1.5)


def test_tube_follows_its_path_and_faces_out():
    b = brtree.TreeBuilder()
    path = [(0.0, 0.0, 0.0), (0.0, 0.0, 2.0), (0.5, 0.0, 4.0), (1.5, 0.0, 5.5)]
    brtree.tube(b, path, [0.5, 0.4, 0.3, 0.1], 8, "bark", wind=(0.0, 0.25, 0.0), tile=(1.0, 4.0), cap=True)
    assert b.tris == 3 * 8 * 2 + 8 and b.materials == {"bark"}
    pos, nrm, uv, wind = arrays(b)
    np.testing.assert_allclose(np.linalg.norm(nrm, axis=1), 1.0, atol=1e-9)
    np.testing.assert_array_equal(wind, np.tile([0.0, 0.25, 0.0], (len(wind), 1)))
    ring0 = pos[:9]                                        # the first ring, its seam vertex twice
    np.testing.assert_allclose(np.hypot(ring0[:, 0], ring0[:, 1]), 0.5, atol=1e-9)
    np.testing.assert_allclose(ring0[0], ring0[8], atol=1e-9)
    assert uv[0, 1] == 0.0 and uv[9, 1] == pytest.approx(0.5)  # 2 m up a 4 m tile
    assert uv[8, 0] == pytest.approx(round(2 * math.pi * 0.325))  # whole repeats round the mean radius
    # every triangle is counter-clockwise seen from outside: its normal agrees with its vertices'
    tri, fn = face_normals(b)
    assert (np.einsum("ij,ij->i", fn, nrm[tri].mean(axis=1)) > 0.5).all()


def test_tube_flutes():
    b = brtree.TreeBuilder()
    brtree.tube(b, [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)], [1.0, 1.0], 12, "bark", lobes=3, depth=[0.2, 0.0])
    pos = arrays(b)[0]
    r0 = np.hypot(pos[:12, 0], pos[:12, 1])
    assert r0.max() == pytest.approx(1.2) and r0.min() == pytest.approx(0.8)
    np.testing.assert_allclose(np.hypot(pos[13:25, 0], pos[13:25, 1]), 1.0)


def test_cone_stands_on_its_sunk_base():
    b = brtree.TreeBuilder()
    brtree.cone(b, (2.0, 1.0, 0.0), 0.3, 0.8, 6, "bark", sink=0.3)
    assert b.tris == 6
    pos, nrm, _, wind = arrays(b)
    assert pos[:, 2].min() == pytest.approx(-0.3) and pos[:, 2].max() == pytest.approx(0.8)
    assert np.hypot(pos[:, 0] - 2.0, pos[:, 1] - 1.0).max() == pytest.approx(0.3)
    assert (wind == 0).all() and (nrm[:, 2] > 0).all()
    tri, fn = face_normals(b)
    assert (np.einsum("ij,ij->i", fn, nrm[tri].mean(axis=1)) > 0.3).all()


def test_leaf_fan_cards():
    b = brtree.TreeBuilder()
    rect = (0.5, 0.7, 1.0, 1.0)
    cards = [(-1.0, 0.1, 0.0, 1.5, 1.2, rect), (0.0, -0.2, 1.3, 1.5, 1.2, rect), (1.0, 0.3, -0.4, 1.0, 0.8, rect)]
    brtree.leaf_fan(b, (2.0, 0.0, 10.0), (1.0, 0.0, 0.0), cards, "leaves", atten=lambda p: 0.1 * p[0],
                    phase=0.4, centre=(0.0, 0.0, 12.0))
    assert b.tris == 6 and len(b.verts) == 12
    pos, nrm, uv, wind = arrays(b)
    assert uv[:, 0].min() == 0.5 and uv[:, 0].max() == 1.0 and uv[:, 1].min() == 0.7 and uv[:, 1].max() == 1.0
    for k, (yaw, pitch, roll, length, width, _) in enumerate(cards):
        c = pos[4 * k:4 * k + 4]
        assert np.linalg.norm(c[1] - c[0]) == pytest.approx(width)       # the base edge
        assert np.linalg.norm(c[3] - c[0]) == pytest.approx(length)      # base to tip
        mid_tip = (c[2] + c[3]) / 2 - (2.0, 0.0, 10.0)
        assert math.atan2(mid_tip[1], mid_tip[0]) == pytest.approx(yaw)  # the card's heading
        assert math.asin(mid_tip[2] / length) == pytest.approx(pitch)
    # the fan's middle card is rolled steep; the others lie nearly flat and face up
    _, fn = face_normals(b)
    assert abs(fn[2, 2]) < 0.5 and fn[0, 2] > 0.8 and fn[4, 2] > 0.8
    assert (np.einsum("ij,ij->i", fn, nrm[np.array([t[:3] for t in b.triangles])].mean(axis=1)) > 0.5).all()
    # wind: the branch phase everywhere, the flutter on the tip edge, attenuation growing outward
    np.testing.assert_allclose(wind[:, 1], 0.4)
    assert set(wind[:, 2].round(3)) == {brtree.LEAF_FLUTTER[0], brtree.LEAF_FLUTTER[1]}
    tips = np.isclose(wind[:, 2], brtree.LEAF_FLUTTER[1])
    np.testing.assert_allclose(wind[:, 0], 0.1 * pos[:, 0])
    assert tips.sum() == 6 and (wind[:, 2] <= 1).all()


def test_builder_extends_and_counts():
    a, b = brtree.TreeBuilder(), brtree.TreeBuilder()
    brtree.cone(a, (0.0, 0.0, 0.0), 0.2, 0.5, 5, "bark")
    brtree.cone(b, (1.0, 0.0, 0.0), 0.2, 0.5, 4, "other")
    a.extend(b)
    assert a.tris == 9 and a.materials == {"bark", "other"} and len(a.verts) == (6 + 5) + (5 + 4)
    assert max(i for t in a.triangles for i in t[:3]) == len(a.verts) - 1


def test_hanging_card():
    b = brtree.TreeBuilder()
    rect = brtree.moss_rect(0.5, 0.4)
    brtree.hanging_card(b, (3.0, 1.0, 8.0), 0.0, 1.5, 0.4, rect, "moss", atten=0.12, phase=0.3,
                        outward=(-1.0, 0.0, 0.0))
    assert b.tris == 2 and len(b.verts) == 4 and b.materials == {"moss"}
    pos, nrm, uv, wind = arrays(b)
    top, foot = np.isclose(pos[:, 2], 8.0), np.isclose(pos[:, 2], 6.5)
    assert top.sum() == 2 and foot.sum() == 2                              # it hangs 1.5 m straight down
    np.testing.assert_allclose(pos[top].mean(axis=0), (3.0, 1.0, 8.0), atol=1e-9)   # its top edge on the point
    np.testing.assert_allclose(np.ptp(pos[:, :2], axis=0), (0.0, 0.4), atol=1e-9)   # 0.4 m wide, facing +-x
    # the uv: the card's top row at the top edge, 0.4 m of the 1.1 m wide card across
    assert set(uv[top, 1]) == {1.0} and set(uv[foot, 1]) == {0.0}
    assert np.ptp(uv[:, 0]) == pytest.approx(0.4 / brtree.MOSS_CARD_M[0]) and 0 <= uv[:, 0].min() <= uv[:, 0].max() <= 1
    # wind: r and g at every corner, the flutter growing from the top to the foot
    np.testing.assert_allclose(wind[:, :2], np.tile([0.12, 0.3], (4, 1)))
    assert set(wind[top, 2]) == {0.0} and set(wind[foot, 2]) == {brtree.MOSS_FLUTTER}
    # it faces `outward` (-x), lit a little from above
    _, fn = face_normals(b)
    assert (fn[:, 0] < -0.99).all() and (nrm[:, 0] < -0.8).all() and (nrm[:, 2] > 0).all()
    lean = brtree.TreeBuilder()
    brtree.hanging_card(lean, (0.0, 0.0, 5.0), 0.0, 2.0, 0.3, rect, "moss", lean=(0.1, 0.0))
    q = arrays(lean)[0]
    assert q[q[:, 2] < 4].mean(axis=0)[0] == pytest.approx(0.2)            # 0.1 m out per m down


def test_moss_rect_stays_on_the_card():
    for u in (0.0, 0.5, 0.999):
        u0, v0, u1, v1 = brtree.moss_rect(u, 0.5)
        assert 0 <= u0 < u1 <= 1 and (v0, v1) == (0.0, 1.0) and u1 - u0 == pytest.approx(0.5 / brtree.MOSS_CARD_M[0])


def test_moss_strand_faces_out_of_the_tree():
    b = brtree.TreeBuilder()
    brtree.moss_strand(b, (-2.0, 0.5, 6.0), 0.1, 1.2, 0.3, 0.5, "moss", atten=0.1, phase=0.4, lean=(0.05, 0.0))
    same = brtree.TreeBuilder()
    brtree.hanging_card(same, (-2.0, 0.5, 6.0), 0.1, 1.2, 0.3, brtree.moss_rect(0.5, 0.3), "moss", atten=0.1,
                        phase=0.4, lean=(0.05, 0.0), outward=(-2.0, 0.5, 0.0))
    assert b.verts == same.verts and b.triangles == same.triangles
    _, fn = face_normals(b)
    assert (fn[:, 0] < -0.9).all()                                    # away from the trunk on the z axis


def test_seeded_choices_repeat_and_pick_by_weight():
    a, b = brtree.seeded(11, "branch", 3), brtree.seeded(11, "branch", 3)
    assert [a.random() for _ in range(5)] == [b.random() for _ in range(5)]
    assert brtree.seeded(11, "branch", 4).random() != brtree.seeded(11, "branch", 3).random()
    rng, again = brtree.seeded("u"), brtree.seeded("u")
    assert rng.uniform(2.0, 5.0) == 2.0 + (5.0 - 2.0) * again.random()      # the scripts' draws, as before
    weighted = [("a", 0.25), ("b", 0.5), ("c", 0.25)]
    picks = [brtree.pick(brtree.seeded("pick", i), weighted) for i in range(4000)]
    assert picks.count("b") / 4000 == pytest.approx(0.5, abs=0.03) and set(picks) == {"a", "b", "c"}
    assert brtree.pick(types.SimpleNamespace(random=lambda: 0.999), [("a", 0.5), ("b", 0.4)]) == "b"


def test_trunk_rings_flutes_and_cap():
    b = brtree.TreeBuilder()
    brtree.trunk(b, lambda z: 1.0 if z <= 1.0 else 0.5, 0.5, 10.0, 1.5, (-0.5, 0.0, 0.75), 3.0, 8, "bark",
                 tile=(1.0, 4.0), lobes=4, flute=0.2)
    pos, nrm, uv, wind = arrays(b)
    # the rings given, the swell's top, then three even steps (at most 3 m) up to the top, and the cap
    np.testing.assert_allclose(sorted(set(np.round(pos[:, 2], 9))), [-0.5, 0.0, 0.75, 1.5, 1.5 + 8.5 / 3,
                                                                       1.5 + 17 / 3, 10.0, 10.5], atol=1e-9)
    assert b.tris == 6 * 8 * 2 + 8 and b.materials == {"bark"} and (wind == 0).all()
    radius = np.hypot(pos[:, 0], pos[:, 1])
    for z, (lo, hi) in ((-0.5, (0.8, 1.2)), (0.0, (0.8, 1.2)), (1.5, (0.5, 0.5)), (10.0, (0.5, 0.5))):
        ring = radius[np.isclose(pos[:, 2], z)]
        assert (ring.min(), ring.max()) == pytest.approx((lo, hi)), z               # fluted to the ground only
    flute_75 = radius[np.isclose(pos[:, 2], 0.75)]
    assert flute_75.max() == pytest.approx(1.0 + 0.2 * 0.5 ** 2)                    # (1 - z / swell_h)^2
    assert uv[:, 0].max() == pytest.approx(3.0) and uv[:, 1].max() > 10.0 / 4.0    # 3 repeats round 0.5 m


def test_branch_tapers_along_its_curve():
    taper = brtree.branch_taper(2.0, 2.5, (0.06, 0.05, 0.015))
    assert (taper(0.0), taper(0.5), taper(1.0)) == pytest.approx((0.1, (0.1 + 0.015) / 2, 0.015))
    atten = brtree.radial_atten(0.2, 1.0)
    assert (atten((0.0, 0.0, 3.0)), atten((0.5, 0.0, 9.0)), atten((3.0, 4.0, 0.0))) == pytest.approx((0, 0.1, 0.2))
    curve = ((0.1, 0.0, 5.0), (1.0, 0.0, 5.3), (2.0, 0.0, 5.4))
    b = brtree.TreeBuilder()
    brtree.branch(b, curve, taper, 4, 3, "bark", atten, 0.7, tile=(1.0, 4.0))
    pos, nrm, uv, wind = arrays(b)
    assert b.tris == 3 * 4 * 2 and len(pos) == 4 * 5                 # 3 pieces, 4 sides (the seam twice)
    for k, ring in enumerate(pos.reshape(4, 5, 3)):
        np.testing.assert_allclose(np.linalg.norm(ring - brtree.bezier(*curve, k / 3), axis=1), taper(k / 3))
    np.testing.assert_allclose(wind[:, 0], [atten(p) for p in pos])
    assert (wind[:, 1] == 0.7).all() and (wind[:, 2] == 0).all() and uv[:, 0].max() == pytest.approx(1.0)


def test_fan_base_pulls_in_but_stays_clear_of_the_trunk():
    curve = ((0.2, 0.0, 6.0), (1.5, 0.0, 6.4), (3.0, 0.0, 6.5))
    base, flat = brtree.fan_base(curve, 1.0, 0.0, lambda z: 0.4)
    np.testing.assert_allclose(base, curve[2])
    np.testing.assert_allclose(flat, (1.0, 0.0, 0.0))                 # the branch's level heading
    np.testing.assert_allclose(brtree.fan_base(curve, 1.0, 1.0, lambda z: 0.4)[0], (2.0, 0.0, 6.5))   # 1 m in
    base, _ = brtree.fan_base(curve, 0.2, 5.0, lambda z: 0.4)
    assert math.hypot(base[0], base[1]) == pytest.approx(1.5 * 0.4)   # never inside 1.5 x the trunk's radius


def test_trunk_collision():
    col = brtree.trunk_collision(0.7, 8.0, "bark")
    assert col.tris <= brtree.COLLISION_BUDGET and col.materials == {"bark"}
    assert len(col.faces) == brtree.COLLISION_SIDES + 1                 # the sides and the top
    pts = np.array([p for pts, _, _ in col.faces for p in pts])
    np.testing.assert_allclose(np.hypot(pts[:, 0], pts[:, 1]), 0.7)
    assert (pts[:, 2].min(), pts[:, 2].max()) == pytest.approx((0.0, 8.0))


def test_export_tree_names_checks_and_writes_the_nodes(monkeypatch, tmp_path):
    written = {}
    stub = types.ModuleType("brexport")
    stub.args = lambda: ({"height_m": 4.0}, tmp_path / "br_test")
    stub.export = lambda parts, out, billboard=None: written.update(parts=parts, out=out, billboard=billboard)
    monkeypatch.setitem(sys.modules, "brexport", stub)
    monkeypatch.setattr(brtree.TreeBuilder, "object", lambda self, name: ("tree", self.tris))
    monkeypatch.setattr(brlm.Builder, "object", lambda self, name: ("collision", self.tris))

    def build(params):
        out = {}
        for px, n in ((80, 1), (600, 3), (250, 2)):
            b = brtree.TreeBuilder()
            for k in range(n):
                brtree.cone(b, (float(k), 0.0, 0.0), 0.2, params["height_m"], 4, "bark")
            out[px] = b
        return out

    budgets = {600: 12, 250: 8, 80: 4}
    brtree.export_tree(build, budgets, collision=brtree.trunk_collision(0.5, 4.0, "bark"),
                       billboard="bb_autobillboard25", tail=[("nulldetail30", None)])
    assert written["out"] == tmp_path / "br_test" and written["billboard"] == "bb_autobillboard25"
    assert written["parts"]["main"] == [("br_test_a600", ("tree", 12)), ("br_test_a250", ("tree", 8)),
                                        ("br_test_a80", ("tree", 4)), ("Colmesh-1", ("collision", 22)),
                                        ("nulldetail30", None)]
    brtree.export_tree(build, budgets)                                  # no collision, no billboard
    assert [n for n, _ in written["parts"]["main"]] == ["br_test_a600", "br_test_a250", "br_test_a80"]
    assert written["billboard"] is None
    with pytest.raises(ValueError, match="br_test_a600 has 12 triangles"):
        brtree.export_tree(build, {**budgets, 600: 11})


# ------------------------------------------------------------------ the bald cypress script

class Stop(Exception):
    """Raised by the stub brexport.args, to end the script's main() before it needs Blender."""


def load_tree_script(name: str):
    """blender/trees/<name>.py as a module, its main() stopped before it needs Blender."""
    stub = types.ModuleType("brexport")

    def args():
        raise Stop

    stub.args = args
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "brexport", stub)
        spec = importlib.util.spec_from_file_location(name, ROOT / f"blender/trees/{name}.py")
        module = importlib.util.module_from_spec(spec)
        with pytest.raises(Stop):
            spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cypress():
    return load_tree_script("bald_cypress")


@pytest.fixture(scope="module")
def lods(cypress):
    return cypress.build(PARAMS)


def test_cypress_agrees_with_its_registry_entry(cypress):
    entry = trees.custom_trees()["br_cypress_a"]
    assert ROOT / entry["script"] == ROOT / "blender/trees/bald_cypress.py"
    assert {cypress.BARK, cypress.LEAVES} == set(entry["materials"])
    assert cypress.COLLISION == (entry["collision"]["trunk_radius_m"], entry["collision"]["height_m"])
    assert bool(cypress.BILLBOARD) == entry["billboard"] and cypress.BILLBOARD == "bb_autobillboard25"
    assert entry["native_height"] == entry["params"]["height_m"]


def test_cypress_levels_of_detail_stay_in_budget(cypress, lods):
    assert cypress.BUDGETS == {600: 3000, 250: 1200, 80: 400} and brtree.COLLISION_BUDGET == 24
    assert sorted(lods) == [80, 250, 600]
    for px, b in lods.items():
        assert 0.6 * cypress.BUDGETS[px] < b.tris <= cypress.BUDGETS[px], px
        assert b.materials == {cypress.BARK, cypress.LEAVES}, px
    col = brtree.trunk_collision(*cypress.COLLISION, cypress.BARK)
    assert col.tris <= brtree.COLLISION_BUDGET and col.materials == {cypress.BARK}
    pts = np.array([p for pts, _, _ in col.faces for p in pts])
    assert np.hypot(pts[:, 0], pts[:, 1]).max() == pytest.approx(0.7) and pts[:, 2].max() == pytest.approx(8.0)
    assert pts[:, 2].min() == pytest.approx(0.0)


def test_cypress_is_as_tall_as_its_native_height(lods):
    for px, b in lods.items():
        z = arrays(b)[0][:, 2]
        assert 0.95 * 22 <= z.max() <= 1.03 * 22, px
        assert -0.6 <= z.min() <= -0.2, px                       # into the ground, never floating


def test_cypress_knees(cypress):
    plan = cypress.plan(PARAMS)
    knees = plan["knees"]
    assert len(knees) == 8
    for x, y, r, h in knees:
        assert 0.3 <= h <= 0.9 and math.hypot(x, y) + r <= 4.0
        assert math.hypot(x, y) - r > cypress.FLARE * plan["trunk_r"]     # clear of the flared base
    for n in (6, 12):
        assert len(cypress.plan({**PARAMS, "knees": n})["knees"]) == n
    for n in (5, 13):
        with pytest.raises(ValueError, match="knees"):
            cypress.plan({**PARAMS, "knees": n})


def leaf_and_bark(cypress, b):
    pos, nrm, uv, wind = arrays(b)
    mat = {}
    for t in b.triangles:
        for i in t[:3]:
            mat[i] = t[3]
    leaf = np.array([mat[i] == cypress.LEAVES for i in range(len(pos))])
    return pos, wind, leaf


def test_cypress_shape(cypress, lods):
    plan = cypress.plan(PARAMS)
    r = plan["trunk_r"]
    pos, wind, leaf = leaf_and_bark(cypress, lods[600])
    radius = np.hypot(pos[:, 0], pos[:, 1])
    trunk = ~leaf & (radius < 1.6 * r * cypress.FLARE)

    def ring(z):                                    # the trunk's ring at height z, its seam vertex once
        return np.unique(np.round(pos[trunk & np.isclose(pos[:, 2], z)], 9), axis=0)

    # the flared base: x 2.2 at the ground (the flutes average out), back to the trunk by 1.5 m
    assert np.hypot(*ring(0.0)[:, :2].T).mean() == pytest.approx(cypress.FLARE * r)
    assert np.hypot(*ring(1.5)[:, :2].T) == pytest.approx(r)
    # a straight trunk: its rings stay on the axis
    np.testing.assert_allclose([ring(0.0)[:, :2].mean(axis=0), ring(1.5)[:, :2].mean(axis=0)], 0.0, atol=1e-9)
    # the crown: branches over the top 60 %, the leaves no lower than their droop, narrow, flat-topped
    z, h = pos[leaf, 2], 22.0
    assert min(br["z"] for br in plan["branches"]) >= 0.4 * h
    assert z.min() >= 0.4 * h - 1.5
    width = radius[leaf].max()
    assert width <= 0.3 * h
    assert radius[leaf & (pos[:, 2] > 0.85 * h)].max() >= 0.5 * width


def test_cypress_wind_weights(cypress, lods):
    for px, b in lods.items():
        pos, wind, leaf = leaf_and_bark(cypress, b)
        assert ((0 <= wind) & (wind <= 1)).all(), px
        assert set(np.round(wind[leaf, 2], 3)) == set(brtree.LEAF_FLUTTER), px   # leaf-edge flutter
        assert (wind[~leaf, 2] == 0).all(), px                                     # wood never flutters
        radius = np.hypot(pos[:, 0], pos[:, 1])
        trunk = ~leaf & (radius < 1.6 * cypress.FLARE * cypress.plan(PARAMS)["trunk_r"]) & (pos[:, 2] < 8)
        assert (wind[trunk] == 0).all(), px                                        # nor sways at the trunk
        assert wind[leaf, 0].max() <= cypress.BRANCH_ATTEN + 1e-9, px
        assert np.corrcoef(radius[leaf], wind[leaf, 0])[0, 1] > 0.9, px             # more toward the tips
        assert len(np.unique(np.round(wind[leaf, 1], 3))) >= 20, px                # a phase per branch


def test_cypress_is_deterministic(cypress, lods):
    again = cypress.build(PARAMS)
    assert all(again[px].verts == lods[px].verts and again[px].triangles == lods[px].triangles for px in lods)
    other = cypress.build({**PARAMS, "seed": 12})
    assert other[600].verts != lods[600].verts


# ------------------------------------------------------------------ the mossy cypress (br_cypress_b)

def mossy_params():
    return trees.custom_trees()["br_cypress_b"]["params"]


def moss_strands(b, mat):
    """(top centre, foot centre, width, wind at the top, wind at the foot) of each moss quad of a builder."""
    pos, _, _, wind = arrays(b)
    corners = sorted({i for t in b.triangles if t[3] == mat for i in t[:3]})
    out = []
    for k in range(0, len(corners), 4):
        q = np.array(corners[k:k + 4])
        p, w = pos[q], wind[q]
        hi = p[:, 2] >= p[:, 2].mean()
        out.append((p[hi].mean(axis=0), p[~hi].mean(axis=0), np.linalg.norm(p[hi][0] - p[hi][1]), w[hi], w[~hi]))
    return out


def test_mossy_cypress(cypress):
    MOSSY = mossy_params()
    assert MOSSY["moss"] is True and (MOSSY["seed"], MOSSY["height_m"]) == (12, 20)
    lods = cypress.build(MOSSY)
    counts = {}
    for px, b in lods.items():
        assert 0.6 * cypress.BUDGETS[px] < b.tris <= cypress.BUDGETS[px], px
        assert b.materials == {cypress.BARK, cypress.LEAVES, cypress.MOSS}, px
        strands = moss_strands(b, cypress.MOSS)
        counts[px] = len(strands)
        for top, foot, width, w_top, w_foot in strands:
            length = top[2] - foot[2]
            assert 0.6 - 1e-9 <= length <= 2.2 + 1e-9 and 0.25 - 1e-9 <= width <= 0.5 + 1e-9, px
            assert top[2] >= cypress.CROWN[0] * 20 - 1.0, px                  # under the crown's branches
            assert (w_top[:, 2] == 0).all() and np.allclose(w_foot[:, 2], brtree.MOSS_FLUTTER), px
            assert np.allclose(w_foot[:, :2], w_top[:, :2]), px                # it moves with its branch
            assert 0 < w_top[0, 0] <= cypress.BRANCH_ATTEN + 1e-9, px
    assert counts[600] > counts[250] > counts[80] > 0
    # the strands hang from the branches: each top just under a branch's curve
    plan = cypress.plan(MOSSY)
    curves = np.array([brtree.bezier(*cypress._branch_curve(plan, br), t / 200) for br in plan["branches"]
                       for t in range(201)])
    for top, *_ in moss_strands(lods[600], cypress.MOSS):
        d = curves - top
        assert ((np.hypot(d[:, 0], d[:, 1]) < 0.05) & (d[:, 2] >= 0) & (d[:, 2] <= 0.3)).any()
    # no moss param, no moss
    assert cypress.MOSS not in cypress.build(PARAMS)[600].materials


# ------------------------------------------------------------------ the water tupelo (br_tupelo_a)

def tupelo_params():
    return trees.custom_trees()["br_tupelo_a"]["params"]


@pytest.fixture(scope="module")
def tupelo():
    return load_tree_script("water_tupelo")


@pytest.fixture(scope="module")
def tupelo_lods(tupelo):
    return tupelo.build(tupelo_params())


def test_tupelo_agrees_with_its_registry_entry(tupelo):
    entry = trees.custom_trees()["br_tupelo_a"]
    assert {tupelo.BARK, tupelo.LEAVES} == set(entry["materials"])
    assert tupelo.COLLISION == (entry["collision"]["trunk_radius_m"], entry["collision"]["height_m"])
    assert tupelo.BILLBOARD == "bb_autobillboard25" and entry["billboard"] is True
    assert entry["native_height"] == tupelo_params()["height_m"] == 18
    assert (tupelo.BOTTLE, tupelo.BOTTLE_H) == (2.8, 2.5)


def test_tupelo_levels_of_detail_stay_in_budget(tupelo, cypress, tupelo_lods):
    assert tupelo.BUDGETS == cypress.BUDGETS == brtree.TREE_BUDGETS
    assert sorted(tupelo_lods) == [80, 250, 600]
    for px, b in tupelo_lods.items():
        assert 0.6 * tupelo.BUDGETS[px] < b.tris <= tupelo.BUDGETS[px], px
        assert b.materials == {tupelo.BARK, tupelo.LEAVES}, px
        z = arrays(b)[0][:, 2]
        assert 0.95 * 18 <= z.max() <= 1.03 * 18 and -0.6 <= z.min() <= -0.2, px
    col = brtree.trunk_collision(*tupelo.COLLISION, tupelo.BARK)
    assert col.tris <= brtree.COLLISION_BUDGET and col.materials == {tupelo.BARK}
    pts = np.array([p for pts, _, _ in col.faces for p in pts])
    assert np.hypot(pts[:, 0], pts[:, 1]).max() == pytest.approx(tupelo.COLLISION[0])
    assert (pts[:, 2].min(), pts[:, 2].max()) == pytest.approx((0.0, tupelo.COLLISION[1]))


def test_tupelo_bottle_base(tupelo, tupelo_lods):
    plan = tupelo.plan(tupelo_params())
    r = plan["trunk_r"]
    pos, wind, leaf = leaf_and_bark(tupelo, tupelo_lods[600])
    radius = np.hypot(pos[:, 0], pos[:, 1])
    trunk = ~leaf & (radius < 1.6 * tupelo.BOTTLE * r)

    def ring(z):
        return np.unique(np.round(pos[trunk & np.isclose(pos[:, 2], z)], 9), axis=0)

    assert np.hypot(*ring(0.0)[:, :2].T).mean() == pytest.approx(tupelo.BOTTLE * r, rel=1e-6)  # x 2.8 at the ground
    assert np.hypot(*ring(tupelo.BOTTLE_H)[:, :2].T) == pytest.approx(r)                      # the trunk by 2.5 m
    # a bottle, not a flare: still swollen half way up, then rounding in to the trunk
    zs = np.linspace(0, tupelo.BOTTLE_H, 11)
    rs = [tupelo.trunk_radius(z, r, plan["top"]) for z in zs]
    assert all(np.diff(rs) < 0) and rs[5] >= 1.8 * r
    assert 0.35 <= r <= 0.45                                   # a 0.7-0.9 m trunk above the swell


def test_tupelo_narrow_oval_crown(tupelo, tupelo_lods):
    pos, wind, leaf = leaf_and_bark(tupelo, tupelo_lods[600])
    radius, z, h = np.hypot(pos[:, 0], pos[:, 1]), pos[:, 2], 18.0
    assert z[leaf].min() >= 0.4 * h - 1.5 and z[leaf].max() <= 1.03 * h
    width = radius[leaf].max()
    assert width <= 0.25 * h                                    # narrow: the crown at most half as wide as tall
    mid = radius[leaf & (z > 0.55 * h) & (z < 0.8 * h)].max()
    assert radius[leaf & (z > 0.9 * h)].max() < 0.8 * mid       # oval: narrower at the top ...
    assert radius[leaf & (z < 0.5 * h)].max() < mid             # ... and at the bottom


def test_tupelo_wind_weights(tupelo, tupelo_lods):
    for px, b in tupelo_lods.items():
        pos, wind, leaf = leaf_and_bark(tupelo, b)
        assert ((0 <= wind) & (wind <= 1)).all(), px
        assert set(np.round(wind[leaf, 2], 3)) == set(brtree.LEAF_FLUTTER), px
        assert (wind[~leaf, 2] == 0).all(), px
        radius = np.hypot(pos[:, 0], pos[:, 1])
        assert (wind[~leaf & (radius < 1.2) & (pos[:, 2] < 6)] == 0).all(), px
        assert np.corrcoef(radius[leaf], wind[leaf, 0])[0, 1] > 0.8, px
        assert len(np.unique(np.round(wind[leaf, 1], 3))) >= 15, px


def test_tupelo_is_deterministic(tupelo, tupelo_lods):
    again = tupelo.build(tupelo_params())
    assert all(again[px].verts == tupelo_lods[px].verts for px in tupelo_lods)
    assert tupelo.build({**tupelo_params(), "seed": tupelo_params()["seed"] + 1})[600].verts != tupelo_lods[600].verts


# ------------------------------------------------------------------ the moss drape (br_moss_<live oak>)

@pytest.fixture(scope="module")
def drape():
    return load_tree_script("moss_drape")


POINTS = [[2.0, 0.5, 6.0], [-1.5, 2.0, 5.2], [0.3, -3.0, 7.4], [3.5, 3.5, 4.8]]


def test_moss_drape_hangs_from_its_points(drape):
    lods = drape.build({"hang_points": POINTS})
    assert sorted(lods) == [100, 300] and drape.BUDGETS == {300: 600, 100: 300}
    for px, b in lods.items():
        assert b.materials == {drape.MOSS} and b.tris == 2 * drape.STRANDS[px] * len(POINTS), px
        for top, foot, width, w_top, w_foot in moss_strands(b, drape.MOSS):
            d = np.linalg.norm(np.array(POINTS) - top, axis=1)
            assert d.min() <= drape.EMBED + 1e-9, px                          # its top edge at a hang point
            assert 0.6 - 1e-9 <= top[2] - foot[2] <= 2.2 + 1e-9 and 0.25 - 1e-9 <= width <= 0.5 + 1e-9, px
            assert (w_top == w_top[0]).all() and w_top[0, 0] == 0 and w_top[0, 2] == 0, px   # no bending, flutter 0
            assert np.allclose(w_foot[:, 2], brtree.MOSS_FLUTTER), px
    assert drape.STRANDS == {300: 2, 100: 1}
    # the a100 strand of each point is one of its a300 strands
    hi = {tuple(np.round(t[0], 6)) for t in moss_strands(lods[300], drape.MOSS)}
    assert {tuple(np.round(t[0], 6)) for t in moss_strands(lods[100], drape.MOSS)} <= hi
    assert drape.build({"hang_points": POINTS})[300].verts == lods[300].verts      # deterministic


def test_moss_drape_budget_holds_for_the_most_points(drape):
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(-5, 5, (120, 2)), rng.uniform(4, 9, 120)]).tolist()
    lods = drape.build({"hang_points": pts})
    assert all(lods[px].tris <= drape.BUDGETS[px] for px in lods)
    with pytest.raises(ValueError, match="hang_points"):
        drape.build({"hang_points": []})


# ------------------------------------------------------------------ Blender

@needs_blender
def test_cypress_builds_in_blender(tmp_path):
    entry = trees.custom_trees()["br_cypress_a"]
    out = tmp_path / "models" / "br_cypress_a"           # the out folder names the shape
    (part,) = models.build(models.blender_prefix(Config()), ROOT / entry["script"], entry["params"], out, "x")
    children, billboard = collada.read_npz(part)
    assert billboard == "bb_autobillboard25"
    assert [n for n, _ in children] == ["br_cypress_a_a600", "br_cypress_a_a250", "br_cypress_a_a80", "Colmesh-1"]
    with np.load(part, allow_pickle=False) as z:
        for node, budget in (("br_cypress_a_a600", 3000), ("br_cypress_a_a250", 1200), ("br_cypress_a_a80", 400),
                             ("Colmesh-1", 24)):
            assert len(z[f"{node}/mat"]) <= budget, node
        col, nrm = z["br_cypress_a_a600/col"], z["br_cypress_a_a600/nrm"]
        names = z["br_cypress_a_a600/materials"].tolist()
        leaf = np.repeat(np.array(names)[z["br_cypress_a_a600/mat"]] == "br_tree_cypress_leaves", 3)
        # the wind weights arrive as bytes / 255
        assert set(np.round(col[leaf, 2] * 255).astype(int)) == {round(f * 255) for f in brtree.LEAF_FLUTTER}
        assert (col[:, 3] == 1).all() and (col[~leaf, 2] == 0).all()
        np.testing.assert_allclose(np.linalg.norm(nrm, axis=1), 1.0, atol=1e-3)
        pos = z["br_cypress_a_a600/pos"]
        assert 0.95 * 22 <= pos[:, 2].max() <= 1.03 * 22
        cpos = z["Colmesh-1/pos"]
        assert np.hypot(cpos[:, 0], cpos[:, 1]).max() == pytest.approx(entry["collision"]["trunk_radius_m"], abs=1e-3)
        assert cpos[:, 2].max() == pytest.approx(entry["collision"]["height_m"], abs=1e-3)
    dae = tmp_path / "br_cypress_a.dae"
    dae.write_text(collada.shape_dae(children, billboard), encoding="utf-8")
    counts = validate.dae_vertex_counts(dae)
    assert set(counts) == {n for n, _ in children} and max(counts.values()) <= validate.DAE_MAX_VERTS


@needs_blender
def test_tupelo_and_moss_build_in_blender(tmp_path):
    custom = trees.custom_trees()
    entry = custom["br_tupelo_a"]
    (part,) = models.build(models.blender_prefix(Config()), ROOT / entry["script"], entry["params"],
                           tmp_path / "models" / "br_tupelo_a", "x")
    children, billboard = collada.read_npz(part)
    assert billboard == "bb_autobillboard25"
    assert [n for n, _ in children] == ["br_tupelo_a_a600", "br_tupelo_a_a250", "br_tupelo_a_a80", "Colmesh-1"]

    moss = custom["br_moss_cork_oak_large_1"]
    params = models.moss_params(moss["params"])
    (part,) = models.build(models.blender_prefix(Config()), ROOT / moss["script"], params,
                           tmp_path / "models" / "br_moss_cork_oak_large_1", "x")
    children, billboard = collada.read_npz(part)
    assert billboard is None
    assert [n for n, _ in children] == ["br_moss_cork_oak_large_1_a300", "br_moss_cork_oak_large_1_a100",
                                        "nulldetail30"]
    assert children[-1][1] is None
    with np.load(part, allow_pickle=False) as z:
        assert z["br_moss_cork_oak_large_1_a300/materials"].tolist() == ["br_tree_moss"]
        col = z["br_moss_cork_oak_large_1_a300/col"]
        assert set(np.round(col[:, 2] * 255).astype(int)) == {0, round(brtree.MOSS_FLUTTER * 255)}
    dae = tmp_path / "moss.dae"
    dae.write_text(collada.shape_dae(children, billboard), encoding="utf-8")
    assert max(validate.dae_vertex_counts(dae).values()) <= validate.DAE_MAX_VERTS
