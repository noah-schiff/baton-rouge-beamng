"""The COLLADA writer shared by the building tiles (dae.py) and the models (collada.py)."""
import hashlib
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from test_dae import bld, octagon, rect

from brmap import collada, dae, validate

# sha1 of the tile text below, recorded with the code before collada.py was split out of dae.py.
# The 3,007 shipped tile meshes (.cdae) are only valid for byte-identical .dae files.
GOLDEN = "1a6b7d11f1a4d33be9ff34a29431333777b5077d"

C = "http://www.collada.org/2005/11/COLLADASchema"
NS = {"c": C}


def golden_buildings():
    """A flat box, a hipped house, a courtyard with a hole, a glass tower with a lower
    part beside it, a notched footprint (its simplification) and an octagon."""
    notch = [(0, 0), (20, 0), (20, 10), (14, 10), (14, 8.5), (6, 8.5), (6, 10), (0, 10)]
    return [
        bld(rect(100, 200, 112, 209), id="box", height=7.5, base_z=3.0, wall="concrete", kind="retail"),
        bld(rect(120, 200, 132, 209), id="house", height=6.0, base_z=3.2, roof="hip", wall="res"),
        bld(rect(140, 200, 170, 230), id="court", height=9.0, base_z=3.0, wall="brick", kind="school",
            holes=[rect(150, 210, 160, 220)[::-1]]),
        bld(rect(180, 200, 210, 230), id="tower", height=60.0, base_z=3.1, wall="glass", kind="office",
            colour=(0.55, 0.65, 0.75)),
        bld(rect(210, 200, 230, 215), id="tower_podium", height=12.0, base_z=3.1, wall="glass",
            kind="office", colour=(0.55, 0.65, 0.75)),
        bld(notch, id="notch", height=6.0, base_z=3.0, roof="hip", wall="brick"),
        bld(octagon(260, 215), id="oct", height=5.0, base_z=2.9, wall="concrete"),
    ]


def one_triangle(uv_shift=0.0):
    """One triangle as corner arrays (pos, nrm, uv, col, mat): material 0, uv shifted on corner 0."""
    pos = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    nrm = np.tile([0.0, 0.0, 1.0], (3, 1))
    uv = np.array([[0.25 + uv_shift, 0.25], [0.75, 0.25], [0.25, 0.75]])
    col = np.tile([1.0, 0.5, 0.25, 1.0], (3, 1))
    return pos, nrm, uv, col, np.zeros(1, np.int32)


def quad_as_six_corners():
    """A unit quad as two triangles of three separate corners each: four distinct corners."""
    pos = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 0], [1, 1, 0], [0, 1, 0]])
    uv = pos[:, :2].copy()
    return pos, np.tile([0.0, 0.0, 1.0], (6, 1)), uv, np.ones((6, 4)), np.zeros(2, np.int32)


def vertex_count_for_uv_offset(offset):
    """Vertices of one triangle drawn twice, the second copy's first corner shifted in uv."""
    a, b = one_triangle(), one_triangle(offset)
    corners = [np.concatenate([x, y]) for x, y in zip(a[:4], b[:4])]
    mat = np.zeros(2, np.int32)
    return len(collada.weld(*corners, mat, np.array(["br_x"])).pos)


def mesh_at(y):
    """One triangle whose corners all have this y, so the number is not the array's first."""
    pos, nrm, uv, col, mat = one_triangle()
    pos[:, 1] = y
    return collada.weld(pos, nrm, uv, col, mat, np.array(["br_x"]))


def write_contract_npz(tmp_path, nodes, billboard=""):
    """A shape part as Blender writes it: a triangle in every node that isn't a nulldetail."""
    arrays = {"meta/nodes": np.array(nodes), "meta/billboard": np.array(billboard),
              "meta/blender": np.array("test"), "meta/script_sha1": np.array("0" * 40)}
    for name in nodes:
        if name.startswith("nulldetail"):
            continue
        pos, nrm, uv, col, mat = one_triangle()
        arrays |= {f"{name}/pos": pos, f"{name}/nrm": nrm, f"{name}/uv": uv, f"{name}/col": col,
                   f"{name}/mat": mat, f"{name}/materials": np.array(["br_x"])}
    path = tmp_path / "part.npz"
    np.savez_compressed(path, **arrays)
    return path


def save(tmp_path, xml):
    path = tmp_path / "shape.dae"
    path.write_text(xml, encoding="utf-8")
    return path


def test_tile_dae_is_byte_identical():
    xml, _ = dae.tile_dae(golden_buildings(), (100.0, 200.0))
    assert hashlib.sha1(xml.encode()).hexdigest() == GOLDEN


def test_shape_dae_layout():
    tri = collada.weld(*one_triangle(), materials=np.array(["br_x"]))
    xml = collada.shape_dae([("x_a400", tri), ("x_a100", tri), ("Colmesh-1", tri)],
                            billboard="bb_autobillboard25")
    root = ET.fromstring(xml)
    base = root.find(".//{*}node[@name='base00']")
    assert [n.get("name") for n in base.findall("{*}node")] == ["bb_autobillboard25", "start01", "collision-1"]
    start = base.find("{*}node[@name='start01']")
    assert [n.get("name") for n in start.findall("{*}node")] == ["x_a400", "x_a100", "Colmesh-1"]
    assert "nulldetail" not in xml


def test_shape_dae_header_ids_and_empty_nodes():
    tri = collada.weld(*one_triangle(), materials=np.array(["br_x"]))
    root = ET.fromstring(collada.shape_dae([("x_a100", tri), ("nulldetail20", None)]))
    assert root.get("version") == "1.4.1"
    assert root.find("c:asset/c:up_axis", NS).text == "Z_UP"
    assert root.find("c:asset/c:unit", NS).get("meter") == "1"
    base = root.find(".//{*}node[@name='base00']")
    assert [n.get("name") for n in base.findall("{*}node")] == ["start01", "collision-1"]  # no billboard
    start = base.find("{*}node[@name='start01']")
    assert len(start.find("{*}node[@name='nulldetail20']")) == 0
    node = start.find("{*}node[@name='x_a100']")
    assert node.find("{*}instance_geometry").get("url") == "#x_a100-mesh"
    assert [g.get("id") for g in root.findall(".//{*}geometry")] == ["x_a100-mesh"]


def test_shape_dae_sorts_materials_over_all_nodes():
    a = collada.weld(*one_triangle(), materials=np.array(["br_z"]))
    b = collada.weld(*one_triangle(), materials=np.array(["br_a"]))
    root = ET.fromstring(collada.shape_dae([("x_a100", a), ("Colmesh-1", b)]))
    assert [m.get("name") for m in root.findall(".//{*}library_materials/{*}material")] == ["br_a", "br_z"]
    assert [e.get("id") for e in root.findall(".//{*}library_effects/{*}effect")] == [
        "br_a-effect", "br_z-effect"]


def test_shape_dae_rejects_duplicate_node_names():
    tri = collada.weld(*one_triangle(), materials=np.array(["br_x"]))
    with pytest.raises(ValueError, match="x_a100"):
        collada.shape_dae([("x_a100", tri), ("x_a100", None)])
    with pytest.raises(ValueError, match="start01"):
        collada.shape_dae([("start01", tri)])


def test_weld_merges_identical_corners():
    m = collada.weld(*quad_as_six_corners(), materials=np.array(["br_x"]))
    assert len(m.pos) == 4 and len(m.tris["br_x"]) == 2


def test_weld_keeps_first_seen_order_and_the_corners():
    pos, nrm, uv, col, mat = quad_as_six_corners()
    m = collada.weld(pos, nrm, uv, col, mat, np.array(["br_x"]))
    np.testing.assert_array_equal(m.pos, pos[[0, 1, 2, 5]])  # in order of first appearance
    tris = m.tris["br_x"]
    np.testing.assert_array_equal(m.pos[tris.ravel()], pos)  # every corner is where it was
    np.testing.assert_array_equal(tris, [[0, 1, 2], [0, 2, 3]])


def test_weld_respects_decimals():
    # two corners whose uv differ by 1e-6 merge at uv=5 decimals; by 1e-4 they don't
    assert vertex_count_for_uv_offset(1e-6) == 3 and vertex_count_for_uv_offset(1e-4) == 4


def test_weld_splits_triangles_by_material_in_name_order():
    pos, nrm, uv, col, _ = quad_as_six_corners()
    m = collada.weld(pos, nrm, uv, col, np.array([1, 0], np.int32), np.array(["br_b", "br_a", "br_unused"]))
    assert list(m.tris) == ["br_a", "br_b"]  # sorted, and a material no triangle uses is left out
    np.testing.assert_array_equal(m.pos[m.tris["br_a"].ravel()], pos[:3])
    np.testing.assert_array_equal(m.pos[m.tris["br_b"].ravel()], pos[3:])


def test_weld_merges_slots_that_share_a_material_name():
    # Blender scripts often append one material to two slots; both slots' triangles must survive
    pos, nrm, uv, col, _ = quad_as_six_corners()
    m = collada.weld(pos, nrm, uv, col, np.array([0, 1], np.int32), np.array(["br_a", "br_a"]))
    assert list(m.tris) == ["br_a"] and len(m.tris["br_a"]) == 2
    np.testing.assert_array_equal(m.pos[m.tris["br_a"].ravel()], pos)  # in triangle order
    assert collada.vertex_count(m) == 4
    assert set(np.unique(m.tris["br_a"])) == set(range(collada.vertex_count(m)))  # no orphan vertices
    # a repeated name next to another one: slots 0 and 2 are "br_b", slot 1 is "br_a"
    m = collada.weld(pos, nrm, uv, col, np.array([2, 1], np.int32), np.array(["br_b", "br_a", "br_b"]))
    assert list(m.tris) == ["br_a", "br_b"] and len(m.tris["br_a"]) == len(m.tris["br_b"]) == 1


def test_weld_of_nothing_is_an_empty_mesh():
    m = collada.weld(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 2)), np.zeros((0, 4)),
                     np.zeros(0, np.int32), np.array([], dtype=str))
    assert collada.vertex_count(m) == 0 and m.tris == {}


def test_vertex_count():
    assert collada.vertex_count(collada.weld(*quad_as_six_corners(), materials=np.array(["br_x"]))) == 4


def test_model_positions_keep_millimetres():
    xml = collada.shape_dae([("x_a100", mesh_at(0.1234))], decimals=collada.MODEL)
    assert " 0.123 " in xml
    assert " 0.12 " in collada.shape_dae([("x_a100", mesh_at(0.1234))], decimals=collada.TILE)


def test_decimals_of_the_tiles_and_the_models():
    assert collada.TILE == collada.Decimals(pos=2, nrm=3, uv=3, col=3)
    assert collada.MODEL == collada.Decimals(pos=3, nrm=3, uv=5, col=3)


def test_read_npz_round_trip(tmp_path):
    path = write_contract_npz(tmp_path, nodes=["x_a100", "Colmesh-1", "nulldetail20"], billboard="")
    children, bb = collada.read_npz(path)
    assert [n for n, _ in children] == ["x_a100", "Colmesh-1", "nulldetail20"] and children[2][1] is None
    assert bb is None
    counts = validate.dae_vertex_counts(save(tmp_path, collada.shape_dae(children)))
    assert counts == {"x_a100": 3, "Colmesh-1": 3}


def test_read_npz_billboard(tmp_path):
    path = write_contract_npz(tmp_path, nodes=["x_a100"], billboard="bb_autobillboard25")
    _, bb = collada.read_npz(path)
    assert bb == "bb_autobillboard25"


def test_collada_needs_no_shapely():
    code = "import sys; sys.modules['shapely'] = None; import brmap.collada"
    src = Path(__file__).resolve().parent.parent / "src"
    assert subprocess.run([sys.executable, "-B", "-c", code], cwd=src).returncode == 0
