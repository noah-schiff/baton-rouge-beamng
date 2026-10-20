"""What every Blender model script shares: reading its arguments and writing its shape parts as
npz files for brmap.collada (the npz contract). It runs inside Blender's own Python, which has
numpy but neither shapely nor brmap, so it uses nothing else. A script puts this folder on
sys.path from its own location, then imports it:

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
    import brexport

    params, out = brexport.args()
    ...build the objects, link them to the scene...
    brexport.export({"main": [("tower_a300", high), ("tower_a60", low), ("Colmesh-1", col)]}, out)

A script is run as `blender --background --factory-startup --python-exit-code 1 --python
<script> -- --params <json file> --out <dir>` (brmap.models builds that command). Any exception
ends the run with exit code 1.

Conventions of export:

* A node is an object linked to the scene. Its modifiers are applied and its mesh is written in
  world space (the object's own location, rotation and scale are applied), so a script may
  place its objects either way. A mirroring scale flips the winding back, so faces still look
  outwards. The uv is layer 0, written as it is in Blender (no V flip).
* Colours come from the byte-colour attribute "Col" on the corner domain (any other type or
  domain is an error); without it a node is white. The bytes are written as byte / 255, the
  value the game reads, never converted to linear. They hold tint or, on trees and moss, wind
  weights.
* A triangle's material is the name of its polygon's material slot.
* Node names follow `<shape>_a<px>`, `Colmesh-1` or `nulldetail<px>`: an underscore right
  before the trailing number reads as a minus sign in the engine, so it is refused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import bpy
import numpy as np

COLOR_ATTRIBUTE = "Col"
UNDERSCORE_NUMBER = re.compile(r"_\d+$")


def args() -> tuple[dict, Path]:
    """(the params from the json file, the out dir), as given after `--` on the command line."""
    parser = argparse.ArgumentParser(description="Blender model script")
    parser.add_argument("--params", required=True, type=Path, help="json file with the script's params")
    parser.add_argument("--out", required=True, type=Path, help="folder for the npz files")
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ns = parser.parse_args(argv)
    return json.loads(ns.params.read_text(encoding="utf-8")), ns.out


def material(name: str) -> bpy.types.Material:
    """The material of that name, made when the file has none yet."""
    return bpy.data.materials.get(name) or bpy.data.materials.new(name)


def _script_sha1() -> str:
    """SHA-1 of the script Blender was started with."""
    return hashlib.sha1(Path(sys.modules["__main__"].__file__).read_bytes()).hexdigest()


def _corners(name: str, ob: bpy.types.Object) -> dict[str, np.ndarray]:
    """The arrays of one node (pos, nrm, uv, col, mat, materials) in the npz contract."""
    slots = [s.material.name if s.material else "" for s in ob.material_slots]
    evaluated = ob.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        mesh.calc_loop_triangles()
        tris = len(mesh.loop_triangles)
        if not tris:
            raise ValueError(f"{name}: the object {ob.name!r} has no faces")
        corner = np.empty(tris * 3, np.int64)       # the mesh corner (loop) of each triangle corner
        mesh.loop_triangles.foreach_get("loops", corner)
        mat = np.empty(tris, np.int32)
        mesh.loop_triangles.foreach_get("material_index", mat)
        vertex = np.empty(len(mesh.loops), np.int64)
        mesh.loops.foreach_get("vertex_index", vertex)
        co = np.empty(len(mesh.vertices) * 3)
        mesh.vertices.foreach_get("co", co)
        nrm = np.empty(len(mesh.loops) * 3)
        mesh.corner_normals.foreach_get("vector", nrm)
        uv = np.zeros((len(mesh.loops), 2))
        if len(mesh.uv_layers):
            mesh.uv_layers[0].data.foreach_get("uv", uv.ravel())
        col = np.ones((len(mesh.loops), 4))
        attribute = mesh.color_attributes.get(COLOR_ATTRIBUTE)
        if attribute is not None:
            if attribute.data_type != "BYTE_COLOR" or attribute.domain != "CORNER":
                raise ValueError(f"{name}: the attribute {COLOR_ATTRIBUTE!r} of {ob.name!r} must be a byte "
                                 f"colour on the corner domain, not {attribute.data_type} on {attribute.domain}")
            attribute.data.foreach_get("color_srgb", col.ravel())   # color_srgb: byte / 255, as stored
        world = np.array(evaluated.matrix_world)
    finally:
        evaluated.to_mesh_clear()
    missing = sorted({int(i) for i in np.unique(mat) if i >= len(slots) or not slots[i]})
    if missing:
        raise ValueError(f"{name}: the object {ob.name!r} has faces on material slots with no material: "
                         f"{missing}")

    rot = world[:3, :3]
    pos = co.reshape(-1, 3)[vertex[corner]] @ rot.T + world[:3, 3]
    nrm = nrm.reshape(-1, 3)[corner] @ np.linalg.inv(rot)           # the inverse transpose, on rows
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    uv, col = uv[corner], col[corner]
    if np.linalg.det(rot) < 0:                                      # mirrored: swap corners 2 and 3
        flip = np.arange(tris * 3).reshape(-1, 3)[:, [0, 2, 1]].ravel()
        pos, nrm, uv, col = pos[flip], nrm[flip], uv[flip], col[flip]
    return {"pos": pos, "nrm": nrm, "uv": uv, "col": col, "mat": mat, "materials": np.array(slots)}


def export(parts: dict[str, list[tuple[str, bpy.types.Object | None]]], out: Path,
           billboard: str | None = None) -> None:
    """Write <out>/<part>.npz for every shape part. A part is its nodes in the order they go in
    start01, as (node name, object); an object of None makes an empty node (a nulldetail).
    `billboard` names the empty billboard node, for every part."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"meta/billboard": np.array(billboard or ""), "meta/blender": np.array(bpy.app.version_string),
            "meta/script_sha1": np.array(_script_sha1())}
    for part, nodes in parts.items():
        if not nodes:
            raise ValueError(f"part {part!r} has no nodes")
        arrays = {}
        for name, ob in nodes:
            if UNDERSCORE_NUMBER.search(name):
                raise ValueError(f"node name {name!r}: an underscore right before the trailing number reads "
                                 f"as a minus sign; write it like 'x_a100'")
            if ob is not None:
                arrays |= {f"{name}/{key}": value for key, value in _corners(name, ob).items()}
        np.savez_compressed(out / f"{part}.npz",
                            **{"meta/nodes": np.array([n for n, _ in nodes]), **meta, **arrays})
