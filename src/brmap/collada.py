"""COLLADA (.dae) text for BeamNG.drive shapes: the building tiles (dae.py) and the models.

BeamNG.drive imports a .dae as a TSStatic shape and compiles it to .cdae itself. The layout
follows the stock shapes and was loaded in game in the spike (0.39.4):

    base00
      <billboard>   an empty node naming the autobillboard, when the shape has one
      start01
        <child>     one node per level of detail, in the order given: a mesh, or empty
        ...
      collision-1   an empty node beside start01, as in the stock shapes

* The number ending a node name is its detail size in pixels. An underscore before it would
  read as a minus sign, hence "bld_L300" and "x_a400".
* Z_UP, metres, COLLADA 1.4.1. x, y and z are written as given.
* A <material name> binds to the level Material whose mapTo is that name; one <triangles>
  per material. Effects and materials are listed in name order over all nodes.
* Every vertex has POSITION, NORMAL, TEXCOORD set 0 and COLOR set 0 (RGBA), all at offset 0,
  so one index per corner. Faces are counter-clockwise seen from outside.
* A node's geometry is "<node>-mesh".
* Decimals: the building tiles round positions to 2 decimals (TILE); models keep millimetres
  and 5 decimals of uv (MODEL). The 3,007 shipped tile meshes (.cdae) are only valid for
  byte-identical tile .dae files, so TILE's output must never change.

Only numpy and the standard library are imported, so a model build needs no shapely.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

COLLADA_NS = "http://www.collada.org/2005/11/COLLADASchema"
RESERVED = ("base00", "start01", "collision-1")  # the node ids the document itself uses
CORNER_ARRAYS = ("pos", "nrm", "uv", "col", "mat", "materials")  # per node in the npz contract


@dataclass(frozen=True)
class Decimals:
    """Decimals each vertex attribute is rounded to, in the file and when welding."""
    pos: int = 2
    nrm: int = 3
    uv: int = 3
    col: int = 3


TILE = Decimals()
MODEL = Decimals(pos=3, uv=5)


@dataclass
class Mesh:
    """Vertices (position, normal, uv, RGBA) and triangles (indices into them) per material."""
    pos: np.ndarray                  # (V, 3)
    nrm: np.ndarray                  # (V, 3)
    uv: np.ndarray                   # (V, 2)
    col: np.ndarray                  # (V, 4)
    tris: dict[str, np.ndarray]      # material name -> (T, 3) int


def vertex_count(mesh: Mesh) -> int:
    return len(mesh.pos)


# ------------------------------------------------------------------ welding and reading

def _rounded(a: np.ndarray, decimals: int) -> np.ndarray:
    return np.round(np.asarray(a, float), decimals) + 0.0  # + 0.0: no -0


def weld(pos, nrm, uv, col, mat, materials, decimals: Decimals = MODEL) -> Mesh:
    """A Mesh from per-corner arrays: pos (3T, 3), nrm (3T, 3), uv (3T, 2), col (3T, 4) and the
    material index of each triangle, mat (T,), into the names in `materials`. Corners that are
    equal once rounded to `decimals` become one vertex; vertices keep the order in which their
    first corner appears, and triangles are grouped by material name, in name order."""
    mat = np.asarray(mat)
    if not all(len(a) == 3 * len(mat) for a in (pos, nrm, uv, col)):
        raise ValueError("weld wants three corners for every entry of mat")
    if not len(mat):
        return Mesh(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 2)), np.zeros((0, 4)), {})
    rows = np.hstack([_rounded(pos, decimals.pos), _rounded(nrm, decimals.nrm),
                      _rounded(uv, decimals.uv), _rounded(col, decimals.col)])
    _, first, inverse = np.unique(rows, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")      # the unique rows by first appearance
    rank = np.empty(len(order), np.int64)
    rank[order] = np.arange(len(order))
    tris = rank[inverse.reshape(-1)].reshape(-1, 3)
    verts = rows[first[order]]
    tri_names = np.array([str(m) for m in materials])[mat]   # slots may share a name: group by name
    return Mesh(verts[:, 0:3], verts[:, 3:6], verts[:, 6:8], verts[:, 8:12],
                {n: tris[tri_names == n] for n in sorted(set(tri_names.tolist()))})


def read_npz(path) -> tuple[list[tuple[str, Mesh | None]], str | None]:
    """(start01's children as (node name, Mesh or None for an empty node), the billboard node
    name or None) of a shape part written by Blender (the npz contract), welded at MODEL's
    decimals."""
    with np.load(path, allow_pickle=False) as z:
        children = []
        for name in (str(n) for n in z["meta/nodes"]):
            has_mesh = f"{name}/pos" in z.files
            children.append((name, weld(*(z[f"{name}/{k}"] for k in CORNER_ARRAYS)) if has_mesh else None))
        return children, z["meta/billboard"].item() or None


# ------------------------------------------------------------------ COLLADA text

def _numbers(a: np.ndarray, decimals: int) -> str:
    """Space-separated values rounded to `decimals`, with no '-0' and no trailing '.0'."""
    text = " ".join(map(repr, (np.round(np.asarray(a, float), decimals) + 0.0).ravel().tolist())) + " "
    return text.replace(".0 ", " ").rstrip()


def _source(gid: str, name: str, values: np.ndarray, params: str, decimals: int) -> str:
    count, stride = len(values), len(params)
    p = "".join(f'<param name="{c}" type="float"/>' for c in params)
    return (f'<source id="{gid}-{name}"><float_array id="{gid}-{name}-array" count="{count * stride}">'
            f'{_numbers(values, decimals)}</float_array><technique_common>'
            f'<accessor source="#{gid}-{name}-array" count="{count}" stride="{stride}">{p}</accessor>'
            f'</technique_common></source>')


def _geometry(gid: str, mesh: Mesh, decimals: Decimals) -> str:
    inputs = (f'<input semantic="VERTEX" source="#{gid}-vertices" offset="0"/>'
              f'<input semantic="NORMAL" source="#{gid}-normals" offset="0"/>'
              f'<input semantic="TEXCOORD" source="#{gid}-uvs" offset="0" set="0"/>'
              f'<input semantic="COLOR" source="#{gid}-colors" offset="0" set="0"/>')
    prims = "".join(f'<triangles material="{m}-material" count="{len(t)}">{inputs}'
                    f'<p>{" ".join(map(str, t.ravel().tolist()))}</p></triangles>'
                    for m, t in mesh.tris.items())
    return (f'<geometry id="{gid}" name="{gid}"><mesh>'
            + _source(gid, "positions", mesh.pos, "XYZ", decimals.pos)
            + _source(gid, "normals", mesh.nrm, "XYZ", decimals.nrm)
            + _source(gid, "uvs", mesh.uv, "ST", decimals.uv)
            + _source(gid, "colors", mesh.col, "RGBA", decimals.col)
            + f'<vertices id="{gid}-vertices"><input semantic="POSITION" source="#{gid}-positions"/></vertices>'
            + prims + "</mesh></geometry>")


def _instance(gid: str, materials) -> str:
    bind = "".join(f'<instance_material symbol="{m}-material" target="#{m}-material"/>' for m in materials)
    return (f'<instance_geometry url="#{gid}"><bind_material><technique_common>{bind}'
            f'</technique_common></bind_material></instance_geometry>')


def shape_dae(children, billboard: str | None = None, decimals: Decimals = MODEL) -> str:
    """The .dae text of a shape. `children` are start01's nodes in order, as (name, Mesh); a
    None mesh makes an empty node (a nulldetail level). `billboard` names an empty node put
    first in base00."""
    names = [n for n, _ in children] + ([billboard] if billboard else [])
    clash = sorted({n for n in names if names.count(n) > 1 or n in RESERVED})
    if clash:
        raise ValueError(f"node names used twice or reserved by the document: {', '.join(clash)}")
    meshes = [(name, mesh) for name, mesh in children if mesh is not None]
    materials = sorted({m for _, mesh in meshes for m in mesh.tris})
    effects = "".join(f'<effect id="{m}-effect"><profile_COMMON><technique sid="common"><lambert>'
                      f'<diffuse><color>0.8 0.8 0.8 1</color></diffuse></lambert></technique>'
                      f'</profile_COMMON></effect>' for m in materials)
    library = "".join(f'<material id="{m}-material" name="{m}"><instance_effect url="#{m}-effect"/>'
                      f'</material>' for m in materials)
    nodes = "".join(f'<node id="{name}" name="{name}">{_instance(f"{name}-mesh", mesh.tris)}</node>'
                    if mesh is not None else f'<node id="{name}" name="{name}"/>'
                    for name, mesh in children)
    empty_billboard = f'<node id="{billboard}" name="{billboard}"/>' if billboard else ""
    return ('<?xml version="1.0" encoding="utf-8"?>\n'
            f'<COLLADA xmlns="{COLLADA_NS}" version="1.4.1">\n'
            '<asset><unit name="meter" meter="1"/><up_axis>Z_UP</up_axis></asset>\n'
            f'<library_effects>{effects}</library_effects>\n'
            f'<library_materials>{library}</library_materials>\n'
            '<library_geometries>\n'
            + "\n".join(_geometry(f"{name}-mesh", mesh, decimals) for name, mesh in meshes)
            + '\n</library_geometries>\n'
            '<library_visual_scenes><visual_scene id="Scene" name="Scene">'
            f'<node id="base00" name="base00">{empty_billboard}<node id="start01" name="start01">{nodes}</node>'
            '<node id="collision-1" name="collision-1"/></node>'
            '</visual_scene></library_visual_scenes>\n'
            '<scene><instance_visual_scene url="#Scene"/></scene>\n</COLLADA>\n')
