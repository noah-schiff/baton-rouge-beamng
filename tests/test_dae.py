"""COLLADA building tiles (dae.py) and the tile split (buildings.tiles)."""
import re
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import shapely
from shapely.geometry import Polygon

from brmap import buildings, dae
from brmap.buildings import Building

C = "http://www.collada.org/2005/11/COLLADASchema"
NS = {"c": C}
LOD_NODES = ["bld_L300", "bld_L60", "nulldetail15", "Colmesh-1"]
TWO_DECIMALS = re.compile(r"-?\d+(\.\d{1,2})?")


def rect(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)


def octagon(cx, cy, r=2.5):
    a = np.arange(8) * np.pi / 4
    return np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])


def bld(ring, id="w1", height=6.0, base_z=5.0, roof="flat", wall="res", holes=(), kind="house",
        colour=(0.9, 0.85, 0.8)):
    return Building(id=id, source="osm", kind=kind, ring=np.asarray(ring, float),
                    holes=[np.asarray(h, float) for h in holes], height=height, base_z=base_z,
                    roof=roof, wall=wall, colour=colour, height_source="default")


def start_node(root):
    base = root.find("c:library_visual_scenes/c:visual_scene/c:node", NS)
    assert base.get("name") == "base00"
    return base, base.find("c:node", NS)


def geometry(root, lod):
    """The <geometry> instanced by the LOD node named lod."""
    _, start = start_node(root)
    node = start.find(f"c:node[@name='{lod}']", NS)
    url = node.find("c:instance_geometry", NS).get("url")
    return root.find(f"c:library_geometries/c:geometry[@id='{url[1:]}']", NS)


def source_values(geom, semantic):
    """(values as text tokens, stride) of the source behind an input semantic of the geometry."""
    mesh = geom.find("c:mesh", NS)
    if semantic == "POSITION":
        src_id = mesh.find("c:vertices/c:input[@semantic='POSITION']", NS).get("source")
    else:
        src_id = mesh.find(f"c:triangles/c:input[@semantic='{semantic}']", NS).get("source")
    src = mesh.find(f"c:source[@id='{src_id[1:]}']", NS)
    stride = int(src.find("c:technique_common/c:accessor", NS).get("stride"))
    return src.find("c:float_array", NS).text.split(), stride


def positions(root, lod) -> np.ndarray:
    tokens, stride = source_values(geometry(root, lod), "POSITION")
    assert stride == 3
    return np.array(tokens, float).reshape(-1, 3)


def triangles(geom) -> dict[str, np.ndarray]:
    """material symbol -> (m, 3) vertex indices."""
    out = {}
    for t in geom.find("c:mesh", NS).findall("c:triangles", NS):
        idx = np.array(t.find("c:p", NS).text.split(), int).reshape(-1, 3)
        assert len(idx) == int(t.get("count"))
        out[t.get("material")] = idx
    return out


def test_dae_structure():
    house = bld(rect(0, 0, 12, 9), roof="hip", wall="res")
    tower = bld(rect(20, 0, 50, 30), id="w2", height=40.0, wall="glass", kind="office")
    xml, counts = dae.tile_dae([house, tower], (0.0, 0.0))
    root = ET.fromstring(xml)
    assert root.tag == f"{{{C}}}COLLADA"
    assert root.find("c:asset/c:up_axis", NS).text == "Z_UP"
    assert root.find("c:asset/c:unit", NS).get("meter") == "1"
    base, start = start_node(root)
    assert [n.get("name") for n in base.findall("c:node", NS)] == ["start01", "collision-1"]
    assert [n.get("name") for n in start.findall("c:node", NS)] == LOD_NODES
    assert len(start.find("c:node[@name='nulldetail15']", NS)) == 0
    assert len(base.find("c:node[@name='collision-1']", NS)) == 0
    assert set(counts) == {"bld_L300", "bld_L60", "Colmesh-1"}

    library = {m.get("id"): m.get("name") for m in root.findall("c:library_materials/c:material", NS)}
    assert {"br_bld_wall_res", "br_bld_wall_glass", "br_bld_roof_shingle",
            "br_bld_roof_flat"} <= set(library.values())
    for name in library.values():  # bound by name: each is a Material whose mapTo is that name
        assert buildings.MATERIALS[name]["mapTo"] == name
    for lod in ("bld_L300", "bld_L60", "Colmesh-1"):
        geom = geometry(root, lod)
        for t in geom.find("c:mesh", NS).findall("c:triangles", NS):
            assert {i.get("semantic") for i in t.findall("c:input", NS)} == {
                "VERTEX", "NORMAL", "TEXCOORD", "COLOR"}
            assert t.get("material") in library
        node = start.find(f"c:node[@name='{lod}']", NS)
        bound = {m.get("symbol"): m.get("target")
                 for m in node.iter(f"{{{C}}}instance_material")}
        for symbol in triangles(geom):
            assert bound[symbol] == f"#{symbol}"
        tokens, stride = source_values(geom, "POSITION")
        assert all(TWO_DECIMALS.fullmatch(v) for v in tokens)
        assert len(tokens) // stride == counts[lod]
        assert source_values(geom, "COLOR")[1] == 4 and source_values(geom, "TEXCOORD")[1] == 2
    # the hip roof is only in the full LOD; the far LOD and the collision mesh are flat
    full = {library[s] for s in triangles(geometry(root, "bld_L300"))}
    low = {library[s] for s in triangles(geometry(root, "bld_L60"))}
    assert "br_bld_roof_shingle" in full and "br_bld_roof_shingle" not in low


def test_dae_wall_uvs_and_heights():
    house = bld(rect(0, 0, 12, 9), height=6.4, base_z=2.0, wall="brick", kind="yes")
    xml, _ = dae.tile_dae([house], (0.0, 0.0))
    root = ET.fromstring(xml)
    geom = geometry(root, "bld_L300")
    pos = positions(root, "bld_L300")
    uv = np.array(source_values(geom, "TEXCOORD")[0], float).reshape(-1, 2)
    assert pos[:, 2].min() == pytest.approx(2.0) and pos[:, 2].max() == pytest.approx(8.4)
    walls = triangles(geom)["br_bld_wall_brick-material"]
    w = np.unique(walls)
    # v = height above the base / 3.2 m, so one texture repeat per floor
    np.testing.assert_allclose(uv[w, 1], (pos[w, 2] - 2.0) / 3.2, atol=1e-3)
    # u = distance along the wall / 3 m: the 12 m wall spans 4 repeats
    assert np.ptp(uv[w, 0]) == pytest.approx((12 + 9 + 12 + 9) / 3.0, abs=1e-3)


def test_dae_normals_face_out():
    house = bld(rect(0, 0, 12, 9), roof="hip")
    xml, _ = dae.tile_dae([house], (6.0, 4.5))  # origin at the footprint's centre
    root = ET.fromstring(xml)
    geom = geometry(root, "bld_L300")
    pos = positions(root, "bld_L300")
    tris = np.vstack(list(triangles(geom).values()))
    a, b, c = pos[tris[:, 0]], pos[tris[:, 1]], pos[tris[:, 2]]
    n = np.cross(b - a, c - a)
    assert (np.linalg.norm(n, axis=1) > 1e-6).all()  # no degenerate triangles
    # counter-clockwise from outside: the winding normal of every wall and roof face points
    # away from the middle of the walls; only the soffit under the eaves faces down
    away = (a + b + c) / 3 - np.array([0.0, 0.0, 5.0 + 1.5])
    down = n[:, 2] < -1e-9
    assert down.sum() == 2 and np.allclose(a[down, 2], 5.0 + 3.0)
    assert (np.einsum("ij,ij->i", n[~down], away[~down]) > 0).all()
    # stored normals agree with the winding
    nrm = np.array(source_values(geometry(root, "bld_L300"), "NORMAL")[0], float).reshape(-1, 3)
    unit = n / np.linalg.norm(n, axis=1)[:, None]
    np.testing.assert_allclose(nrm[tris[:, 0]], unit, atol=2e-3)


def test_dae_vertex_counts_and_split():
    xs = 2.5 + 4.6 * np.arange(55)
    blds = [bld(octagon(x, y), id=f"m{i}") for i, (x, y) in
            enumerate((x, y) for x in xs for y in xs)][:3000]
    assert dae.vertex_counts(blds)["bld_L300"] > 60000  # too big for one tile
    out = buildings.tiles(blds, 256.0)
    assert len(out) >= 4
    assert sorted(b.id for _, _, m in out for b in m) == sorted(b.id for b in blds)
    for key, origin, members in out:
        assert key.startswith("e00n00_")
        xml, counts = dae.tile_dae(members, origin)
        assert max(counts.values()) <= 60000
        assert dae.vertex_counts(members) == counts  # the count the split went by
        root = ET.fromstring(xml)
        for lod, n in counts.items():
            assert len(positions(root, lod)) == n


def test_tiles_keep_small_tiles_whole():
    blds = [bld(rect(10, 10, 20, 20), id="a"), bld(rect(30, 10, 40, 20), id="b"),
            bld(rect(-20, -30, -10, -20), id="c")]
    out = {key: (origin, [b.id for b in m]) for key, origin, m in buildings.tiles(blds, 256.0)}
    assert out == {"e00n00": ((128.0, 128.0), ["a", "b"]), "w01s01": ((-128.0, -128.0), ["c"])}


def test_dae_local_coordinates():
    blds = [bld(rect(1030 + 10 * i, 2000, 1038 + 10 * i, 2008), id=f"w{i}", base_z=7.0)
            for i in range(5)]
    ((key, origin, members),) = buildings.tiles(blds, 256.0)
    assert key == "e04n07" and origin == (1152.0, 1920.0) and len(members) == 5
    xml, _ = dae.tile_dae(members, origin)
    root = ET.fromstring(xml)
    for lod in ("bld_L300", "bld_L60", "Colmesh-1"):
        pos = positions(root, lod)
        assert np.abs(pos[:, :2]).max() <= 256.0
        assert pos[:, 0].min() + origin[0] == pytest.approx(1030.0)
        assert pos[:, 1].max() + origin[1] == pytest.approx(2008.0)
        assert pos[:, 2].min() == pytest.approx(7.0) and pos[:, 2].max() == pytest.approx(13.0)


def test_dae_hole_gets_walls_and_no_roof():
    court = bld(rect(0, 0, 30, 30), holes=[rect(10, 10, 20, 20)[::-1]], height=9.0, kind="school",
                wall="brick", base_z=0.0)
    xml, _ = dae.tile_dae([court], (0.0, 0.0))
    root = ET.fromstring(xml)
    geom = geometry(root, "bld_L300")
    pos = positions(root, "bld_L300")
    tris = triangles(geom)
    walls = tris["br_bld_wall_brick-material"]
    assert len(walls) == 2 * (4 + 4)  # two triangles per wall, outer ring and courtyard
    on_hole = pos[np.unique(walls)]
    assert ((np.isclose(on_hole[:, 0], 10) | np.isclose(on_hole[:, 0], 20))
            & (on_hole[:, 1] >= 10) & (on_hole[:, 1] <= 20)).any()
    roof = tris["br_bld_roof_flat-material"]
    a, b, c = (pos[roof[:, k], :2] for k in range(3))
    cross = (b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0]
    assert (cross > 0).all()  # counter-clockwise from above
    area = 0.5 * cross.sum()
    assert area == pytest.approx(30 * 30 - 10 * 10)


def test_far_detail_and_collision_stay_inside_the_footprint():
    """The 2 m simplification must not fill a notch: the collision mesh would stand where
    no wall is drawn, out in a road or a loading bay."""
    notch = [(0, 0), (20, 0), (20, 10), (14, 10), (14, 8.5), (6, 8.5), (6, 10), (0, 10)]
    foot = Polygon(notch)
    root = ET.fromstring(dae.tile_dae([bld(notch, base_z=0.0)], (0.0, 0.0))[0])
    for lod in ("bld_L60", "Colmesh-1"):
        pos = positions(root, lod)
        tris = np.vstack(list(triangles(geometry(root, lod)).values()))
        top = tris[np.isclose(pos[tris][:, :, 2], 6.0).all(axis=1)]  # the flat roof
        a, b, c = (pos[top[:, k], :2] for k in range(3))
        cross = (b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0]
        assert 0.5 * np.abs(cross).sum() == pytest.approx(foot.area)
        assert shapely.contains_xy(foot.buffer(1e-6), *((a + b + c) / 3).T).all()


def test_dae_hip_roof_rise():
    house = bld(rect(0, 0, 12, 9), roof="hip", height=6.0, base_z=0.0)
    root = ET.fromstring(dae.tile_dae([house], (0.0, 0.0))[0])
    geom = geometry(root, "bld_L300")
    pos = positions(root, "bld_L300")
    walls = pos[np.unique(triangles(geom)["br_bld_wall_res-material"])]
    roof = pos[np.unique(triangles(geom)["br_bld_roof_shingle-material"])]
    rise = min(0.35 * 9, 3.0)
    assert walls[:, 2].max() == pytest.approx(6.0 - rise)  # the rise comes out of the walls
    assert roof[:, 2].max() == pytest.approx(6.0)
    ridge = roof[np.isclose(roof[:, 2], 6.0)]
    assert np.ptp(ridge[:, 0]) == pytest.approx(12 - 9) and np.ptp(ridge[:, 1]) == pytest.approx(0)
