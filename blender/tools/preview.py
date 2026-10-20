"""A look at a built model before the game: renders the visible levels of detail of one shape
part (an npz from the models stage) side by side with Blender's Workbench engine, headless.
Not part of the build (blender/lib holds what the models share; this sits outside it).

    C:/Desktop/blender.exe --background --factory-startup --python blender/tools/preview.py --
        --npz data/out_lsu/models/memorial_tower/main.npz --out docs/screenshots/v02/tower_blender.png
        [--textures data/out_lsu/textures] [--game "C:/Program Files (x86)/Steam/steamapps/common/BeamNG.drive"]
        [--view 1,0.65,-0.3] [--box x0,y0,x1,y1]

The levels stand left to right from the most detailed, seen along --view (the direction the
camera looks, in model coordinates; by default from the model's -u, -v side, south-east of the
Memorial Tower on campus), orthographic, smooth where the model's normals are. Given several
--npz parts, it shows the whole model instead: each part's most detailed level, in place.
With --box, only the triangles with a corner in that box (model x, y) are drawn.
Materials are flat colours, except that the ones named in IMAGES show that texture from
--textures (the shipped .dds) and the ones in STOCK the stock texture they use, read from the
game's zips (--game, default BEAMNG_GAME_DIR or Steam's folder), cut out by its opacity map.
The ones in ALPHA are cut out by their image's alpha, the ones in CUTOUT by an opacity map of
their own from --textures (the moss), and the ones in TINTED (vertColor in their registry) are
multiplied by each face's vertex colour, as the game does. A stock tree material
(src/brmap/data/stock_trees.json, e.g. a live oak's, for a stock oak converted to an npz) shows
its stock colour map, cut out by its opacity map at its alphaRef.
"""
import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import bpy
import numpy as np

COLOURS = {"br_lm_stucco_tan": (0.86, 0.76, 0.60), "br_lm_roof_tile": (0.62, 0.27, 0.18),
           "br_lm_trim_stone": (0.90, 0.88, 0.82),
           "br_tree_cypress_bark": (0.45, 0.38, 0.32), "br_tree_cypress_leaves": (0.33, 0.45, 0.24),
           "br_lm_concrete_tan": (0.80, 0.73, 0.60), "br_lm_seats": (1.0, 1.0, 1.0), "br_lm_turf": (0.30, 0.50, 0.22),
           "br_lm_field_paint": (0.95, 0.95, 0.93), "br_lm_steel": (0.55, 0.57, 0.60), "br_lm_lamp": (1.0, 0.97, 0.88),
           "br_lm_concrete": (0.80, 0.80, 0.78), "br_lm_clay": (0.78, 0.48, 0.30), "br_lm_paint": (1.0, 1.0, 1.0),
           "br_lm_rock": (0.56, 0.53, 0.48), "br_lm_water": (0.10, 0.17, 0.15), "br_lm_glass": (0.55, 0.66, 0.68),
           "br_sty_lsu_wall": (0.84, 0.75, 0.59), "br_lm_limestone": (0.88, 0.84, 0.74),
           "br_lm_stucco_white": (0.94, 0.94, 0.93), "br_lm_gothic_window": (0.20, 0.24, 0.30),
           "br_sty_ribbon_dark": (0.20, 0.20, 0.21), "br_sty_precast_punched": (0.70, 0.68, 0.64),
           "br_sty_brick_hotel": (0.55, 0.32, 0.25), "br_sty_precast_arena": (0.75, 0.71, 0.63),
           "br_sty_glass_blue": (0.25, 0.38, 0.46), "br_tree_tupelo_bark": (0.45, 0.42, 0.38),
           "br_tree_tupelo_leaves": (0.25, 0.40, 0.20), "br_tree_moss": (0.53, 0.56, 0.48)}
IMAGES = {"br_lm_clock": "clock_face_b.color.dds",
          "br_lm_turf": "turf_b.color.dds", "br_lm_field_marks": "field_atlas_b.color.dds",
          "br_lm_board": "board_b.color.dds", "br_sty_lsu_wall": "lsu_wall_b.color.dds",
          "br_lm_gothic_window": "gothic_window_b.color.dds",
          **{f"br_sty_{style}": f"facade_{style}_b.color.dds"
             for style in ("ribbon_dark", "precast_punched", "brick_hotel", "precast_arena", "glass_blue")}}
ALPHA = {"br_lm_field_marks": 127, "br_lm_gothic_window": 127}   # material -> alphaRef
CUTOUT = {"br_tree_moss": ("moss_b.color.dds", "moss_o.data.dds", 67)}   # material -> (colour, opacity, alphaRef)
TINTED = {"br_lm_limestone", "br_lm_concrete_tan", "br_lm_seats", "br_lm_turf", "br_lm_concrete", "br_lm_paint"}
# material -> (zip under content/, colour map, opacity map or None, baseColorFactor, alphaRef):
# the stock maps of src/brmap/data/tree_materials.yaml and the landmarks' stock roof tiles, limestone
# (the marble slabs, tinted), white stucco and the truss's painted steel (metal_paint, tinted)
STOCK = {
    "br_lm_truss_steel": ("assets/materials/tileable.zip",
                          "assets/materials/tileable/metal/metal_paint/t_metal_paint_b.color.dds",
                          None, (0.53, 0.60, 0.55), 0),
    "br_lm_limestone": ("assets/materials/tileable.zip",
                        "assets/materials/tileable/marble/slabs_marble/t_slabs_marble_b.color.dds",
                        None, (0.93, 0.89, 0.79), 0),
    "br_lm_stucco_white": ("assets/materials/tileable.zip",
                           "assets/materials/tileable/plaster/stucco1_white/t_stucco1_white_b.color.dds",
                           None, (1.0, 1.0, 1.0), 0),
    "br_lm_roof_tile": ("assets/materials/tileable.zip",
                        "assets/materials/tileable/roofing/t_italy_bld_roof_tiles/t_italy_bld_roof_tiles_b.color.dds",
                        None, (1.0, 1.0, 1.0), 0),
    "br_tree_cypress_bark": ("assets/materials/tree.zip", "assets/materials/tree/fir/t_fir_bark/t_fir_bark_b.color.dds",
                             None, (1.0, 1.0, 1.0), 0),
    "br_tree_cypress_leaves": ("assets/materials/tree.zip",
                               "assets/materials/tree/cypress/t_cypress_leaves/t_cypress_leaves_b.color.dds",
                               "assets/materials/tree/cypress/t_cypress_leaves/t_cypress_leaves_o.data.dds",
                               (0.78, 0.92, 0.62), 90),
    "br_tree_tupelo_bark": ("assets/materials/tree.zip",
                            "assets/materials/tree/poplar/t_poplar_bark/t_poplar_bark_b.color.dds",
                            None, (0.93, 0.88, 0.80), 0),
    "br_tree_tupelo_leaves": ("assets/materials/tree.zip",
                              "assets/materials/tree/beech/t_beech_leaves/t_beech_leaves_b.color.dds",
                              "assets/materials/tree/beech/t_beech_leaves/t_beech_leaves_o.data.dds",
                              (0.72, 0.86, 0.66), 105),
}
STOCK_TREES = Path(__file__).resolve().parents[2] / "src" / "brmap" / "data" / "stock_trees.json"
GAME = Path(os.environ.get("BEAMNG_GAME_DIR", r"C:\Program Files (x86)\Steam\steamapps\common\BeamNG.drive"))
VIEW = (1.0, 0.65, -0.3)          # the direction the camera looks, model coordinates
GAP = 1.25                        # space between the levels, in model widths
SIZE = (1800, 900)


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True, type=Path, nargs="+")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--textures", type=Path)
    p.add_argument("--game", type=Path, default=GAME)
    p.add_argument("--view", type=lambda v: tuple(float(c) for c in v.split(",")), default=VIEW)
    p.add_argument("--box", type=lambda v: tuple(float(c) for c in v.split(",")), default=None)
    return p.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])


def stock_tree_material(name: str):
    """A stock tree material (stock_trees.json) in the STOCK form, or None: its colour and opacity
    maps (.png names -> the .dds the game ships, in content/assets/materials/<folder>.zip)."""
    m = json.loads(STOCK_TREES.read_text(encoding="utf-8"))["materials"].get(name)
    if m is None:
        return None
    stage = m["Stages"][0]
    colour, opacity = stage.get("baseColorMap"), stage.get("opacityMap")
    if not colour or not colour.startswith("/assets/materials/"):
        return None
    folder = colour.split("/")[3]

    def member(path):
        return path.lstrip("/").rsplit(".", 1)[0] + ".dds" if path else None

    return (f"assets/materials/{folder}.zip", member(colour), member(opacity), (1.0, 1.0, 1.0),
            int(m.get("alphaRef", 127)) if opacity else 0)


def cutout_image(name: str, colour: Path, opacity: Path | None, tint) -> bpy.types.Image:
    """An image of the colour map, tinted, with the opacity map's grey (scaled to it) in its alpha."""
    maps = [bpy.data.images.load(str(Path(path).resolve())) for path in (colour, opacity) if path]
    image = maps[0]
    w, h = image.size
    px = np.array(image.pixels[:]).reshape(-1, 4)
    px[:, :3] *= tint
    if len(maps) > 1:
        if tuple(maps[1].size) != (w, h):          # the stock beech's opacity map is twice its colour map
            maps[1].scale(w, h)
        px[:, 3] = np.array(maps[1].pixels[:]).reshape(-1, 4)[:, 0]
    out = bpy.data.images.new(name, w, h, alpha=True)
    out.pixels.foreach_set(px.astype(np.float32).ravel())
    return out


def stock_image(name: str, spec, game: Path, scratch: Path) -> bpy.types.Image | None:
    """The stock texture of a material (spec in the STOCK form), tinted, its opacity map in the
    alpha; None without the game."""
    zip_name, colour, opacity, tint, _ = spec
    if not (game / "content" / zip_name).is_file():
        return None
    paths = []
    with zipfile.ZipFile(game / "content" / zip_name) as z:
        for member in (colour, opacity):
            if member:
                path = scratch / Path(member).name
                path.write_bytes(z.read(member))
                paths.append(path)
    return cutout_image(name, paths[0], paths[1] if len(paths) > 1 else None, tint)


def material(name: str, textures: Path | None, game: Path, scratch: Path) -> bpy.types.Material:
    """A material whose image texture Workbench shows: the texture, or one flat colour."""
    mat = bpy.data.materials.new(name)          # Blender 5 gives every new material a node tree
    node = mat.node_tree.nodes.new("ShaderNodeTexImage")
    spec = STOCK.get(name) or stock_tree_material(name)
    image = stock_image(name, spec, game, scratch) if spec else None
    if image is not None:
        node.image = image
        if spec[2]:                             # cut out where the opacity is under alphaRef
            mat.blend_method = "CLIP"
            mat.alpha_threshold = spec[4] / 255
    elif textures and name in CUTOUT and (textures / CUTOUT[name][0]).is_file():
        colour, opacity, ref = CUTOUT[name]
        node.image = cutout_image(name, textures / colour, textures / opacity, (1.0, 1.0, 1.0))
        mat.blend_method = "CLIP"
        mat.alpha_threshold = ref / 255
    elif textures and name in IMAGES and (textures / IMAGES[name]).is_file():
        node.image = bpy.data.images.load(str((textures / IMAGES[name]).resolve()))
        if name in ALPHA:
            mat.blend_method = "CLIP"
            mat.alpha_threshold = ALPHA[name] / 255
    else:
        node.image = bpy.data.images.new(name, 4, 4)
        node.image.generated_color = (*COLOURS.get(name, (1.0, 0.0, 1.0)), 1.0)
    mat.node_tree.nodes.active = node
    return mat


class Materials:
    """The preview's materials by (name, vertex colour as bytes): a TINTED material in a colour
    other than white is a copy whose image is multiplied by it."""

    def __init__(self, textures, game, scratch):
        self.textures, self.game, self.scratch = textures, game, scratch
        self.made = {}

    def get(self, name: str, rgb=(255, 255, 255)) -> bpy.types.Material:
        rgb = tuple(int(c) for c in rgb) if name in TINTED else (255, 255, 255)
        key = (name, rgb)
        if key not in self.made:
            if rgb == (255, 255, 255):
                self.made[key] = material(name, self.textures, self.game, self.scratch)
            else:
                mat = self.get(name).copy()
                mat.name = f"{name} {bytes(rgb).hex()}"
                node = mat.node_tree.nodes.active
                px = np.empty(len(node.image.pixels), np.float32)
                node.image.pixels.foreach_get(px)
                px = px.reshape(-1, 4)
                px[:, :3] *= np.array(rgb, np.float32) / 255
                tinted = bpy.data.images.new(mat.name, *node.image.size, alpha=True)
                tinted.pixels.foreach_set(px.ravel())
                node.image = tinted
                self.made[key] = mat
        return self.made[key]


def node_arrays(data, node: str, box=None) -> dict:
    """A node's arrays (pos, nrm, uv, col per corner; mat per triangle; materials), with box
    (x0, y0, x1, y1) only its triangles with a corner inside it."""
    out = {key: data[f"{node}/{key}"] for key in ("pos", "nrm", "uv", "col", "mat", "materials")}
    if box is not None:
        x0, y0, x1, y1 = box
        xy = out["pos"][:, :2].reshape(-1, 3, 2)
        keep = ((xy[..., 0] >= x0) & (xy[..., 0] <= x1) & (xy[..., 1] >= y0) & (xy[..., 1] <= y1)).any(axis=1)
        for key in ("pos", "nrm", "uv", "col"):
            out[key] = out[key].reshape(len(keep), 3, -1)[keep].reshape(-1, out[key].shape[1])
        out["mat"] = out["mat"][keep]
    return out


def mesh_object(node: str, arrays: dict, materials: Materials, offset) -> bpy.types.Object:
    pos, uv = arrays["pos"], arrays["uv"]
    tris = len(pos) // 3
    mesh = bpy.data.meshes.new(node)
    mesh.from_pydata((pos + offset).tolist(), [], np.arange(tris * 3).reshape(-1, 3).tolist())
    names = [str(n) for n in arrays["materials"]]
    rgb = np.round(arrays["col"][0::3, :3] * 255).astype(np.int64)    # each face's first corner
    keys, slot = np.unique(np.column_stack([arrays["mat"].astype(np.int64), rgb]), axis=0, return_inverse=True)
    for m, r, g, b in keys:
        mesh.materials.append(materials.get(names[m], (r, g, b)))
    mesh.polygons.foreach_set("material_index", slot.ravel().astype(np.int32))
    mesh.uv_layers.new(name="UVMap").data.foreach_set("uv", uv.ravel())
    mesh.shade_smooth()
    mesh.normals_split_custom_set(arrays["nrm"].tolist())   # the model's own: flat faces stay flat
    ob = bpy.data.objects.new(node, mesh)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def main() -> None:
    a = args()
    for ob in list(bpy.data.objects):           # the factory scene's cube, camera and light
        bpy.data.objects.remove(ob)
    parts = [np.load(path, allow_pickle=False) for path in a.npz]
    shown = [[str(n) for n in data["meta/nodes"] if not str(n).lower().startswith(("colmesh", "nulldetail"))]
             for data in parts]
    scratch = Path(tempfile.mkdtemp(prefix="preview_"))
    materials = Materials(a.textures, a.game, scratch)
    view = np.array(a.view, float) / np.linalg.norm(a.view)
    across = np.cross(view, (0.0, 0.0, 1.0))
    across /= np.linalg.norm(across)
    if len(parts) == 1:                         # one part: its levels side by side
        nodes = shown[0]
        arrays = [node_arrays(parts[0], n, a.box) for n in nodes]
        width = max(float(np.ptp(x["pos"][:, :2] @ across[:2])) for x in arrays)
        obs = [mesh_object(n, x, materials, across * k * width * GAP) for k, (n, x) in enumerate(zip(nodes, arrays))]
    else:                                       # the whole model: every part's most detailed level
        nodes = [names[0] for names in shown]
        obs = [mesh_object(n, node_arrays(data, n, a.box), materials, np.zeros(3)) for data, n in zip(parts, nodes)]

    corners = np.vstack([np.array([v.co for v in ob.data.vertices]) for ob in obs])
    centre = (corners.min(axis=0) + corners.max(axis=0)) / 2
    cam = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
    bpy.context.scene.collection.objects.link(cam)
    cam.data.type = "ORTHO"
    rel = corners - centre
    h_extent = np.ptp(rel @ across)
    up = np.cross(across, view)
    v_extent = np.ptp(rel @ up)
    cam.data.ortho_scale = 1.08 * max(h_extent, v_extent * SIZE[0] / SIZE[1])
    cam.location = centre - view * 400.0
    cam.data.clip_end = 2000.0
    cam.rotation_euler = (math.pi / 2 - math.asin(-view[2]) + 0.0, 0.0, math.atan2(view[1], view[0]) - math.pi / 2)
    scene = bpy.context.scene
    scene.camera = cam
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "TEXTURE"
    scene.display.shading.show_shadows = False     # cast shadows read as dents in a still
    scene.display.shading.show_cavity = True
    scene.render.resolution_x, scene.render.resolution_y = SIZE
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.compression = 100
    scene.world = bpy.data.worlds.new("sky")
    scene.world.color = (0.62, 0.72, 0.85)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(a.out.resolve())
    bpy.ops.render.render(write_still=True)
    shutil.rmtree(scratch, ignore_errors=True)
    print(f"rendered {', '.join(nodes)} to {a.out}")


main()
