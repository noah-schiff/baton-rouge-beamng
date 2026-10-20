"""Smoke test of the Blender runner (tests/test_models.py): a 2 m cube with flat faces, which
welds to 24 vertices, and a collision cube with smooth normals, which welds to 8.

The visual cube carries a uv and a "Col" attribute of (0.2, 0.4, 1, 1), so the test can see the
colour bytes arrive unchanged; the collision cube has neither, so it exports as uv 0 and white.
Both sit at z = 1 through the object's location, which the export applies."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

import bmesh  # noqa: E402
import bpy  # noqa: E402
import brexport  # noqa: E402

SIDE = 2.0
COLOUR = (0.2, 0.4, 1.0, 1.0)     # 51, 102, 255 as bytes
QUAD_UV = [(0, 0), (1, 0), (1, 1), (0, 1)]


def make_cube(name: str, smooth: bool, visual: bool) -> bpy.types.Object:
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=SIDE)
    if visual:
        layer = bm.loops.layers.uv.new("UVMap")
        for face in bm.faces:
            for loop, uv in zip(face.loops, QUAD_UV):
                loop[layer].uv = uv
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    mesh.materials.append(brexport.material("br_smoke"))
    if smooth:
        mesh.shade_smooth()
    else:
        mesh.shade_flat()
    if visual:
        attribute = mesh.color_attributes.new("Col", "BYTE_COLOR", "CORNER")
        for corner in attribute.data:
            corner.color_srgb = COLOUR
    ob = bpy.data.objects.new(name, mesh)
    ob.location = (0.0, 0.0, SIDE / 2)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def main() -> None:
    _, out = brexport.args()
    brexport.export({"main": [("cube_a100", make_cube("cube", smooth=False, visual=True)),
                              ("Colmesh-1", make_cube("collision", smooth=True, visual=False))]}, out)


main()
