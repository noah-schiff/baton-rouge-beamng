"""Write src/brmap/data/stock_trees.json: the stock BeamNG tree models brmap plants.

Reads the installed game once (read only; the game is not launched), so the build
never needs it. For each candidate (plan 2026-10-16, Task 4) it takes from the stock
level's zip in content/levels:

* the ForestItemData entry (wind and placement settings) from art/forest/*Data.json,
  minus the fields the level writes itself (name, internalName, persistentId, class,
  __parent and shapeFile, which is kept on its own, made absolute);
* the native height: max z minus min z of the vertices of the shape's highest-detail
  mesh (the node whose name ends in the largest _a<pixels>), after its node transforms;
* the collision reach: {"z0", "dz", "r"}, where r[k] is how far (m) from the shape's
  vertical axis its collision mesh (the Colmesh nodes) reaches between z0 + k dz and
  z0 + (k + 1) dz. The stock meshes hold root flares and low branches 2-3 m out, so the
  build keeps them, not only the trunks, off the roads;
* the Material definitions the shape binds (every LOD's and the collision mesh's), from
  the level's *materials.json, under their stock names (the DAE binds them by name),
  texture paths made absolute and the stock persistentId dropped.

Tree materials exist only inside the level zips and some names repeat across levels
with different textures (italy and east_coast_usa both define oak_leaves_01). A level
holds one definition per name, so the first candidate's wins; the clashes are listed
in "material_conflicts" (name -> [level used, other levels]).

    python tools/extract_stock_trees.py [game_dir] [out.json]

game_dir defaults to $BEAMNG_GAME_DIR, else Steam's default library.
"""
import json
import math
import os
import posixpath
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src" / "brmap" / "data" / "stock_trees.json"
STEAM_GAME = Path(r"C:\Program Files (x86)\Steam\steamapps\common\BeamNG.drive")

# (item key, stock level, stock item name). The keys are br_<stock name>; italy's palm
# gets a suffix because WCUSA's fanpalm_tall (the spec's palm) has the same name.
CANDIDATES = [
    ("br_holm_oak", "italy", "holm_oak"),
    ("br_holm_oak_city_tall", "italy", "holm_oak_city_tall"),
    ("br_holm_oak_city_small", "italy", "holm_oak_city_small"),
    ("br_cork_oak_large_1", "italy", "cork_oak_large_1"),
    ("br_maritime_pine_tree", "italy", "maritime_pine_tree"),
    ("br_fanpalm_tall_italy", "italy", "fanpalm_tall"),
    ("br_tree_oak_large_a", "east_coast_usa", "tree_oak_large_a"),
    ("br_tree_oak_large_b", "east_coast_usa", "tree_oak_large_b"),
    ("br_tree_oak_large_c", "east_coast_usa", "tree_oak_large_c"),
    ("br_tree_oak_sml_b", "east_coast_usa", "tree_oak_sml_b"),
    ("br_oak_dry_a", "west_coast_usa", "oak_dry_a"),
    ("br_pine_radiata_large", "west_coast_usa", "pine_radiata_large"),
    ("br_redwood_broad", "west_coast_usa", "redwood_broad"),
    ("br_fanpalm_tall", "west_coast_usa", "fanpalm_tall"),
]
LEVEL_FIELDS = ("name", "internalName", "persistentId", "class", "__parent", "shapeFile")
NOT_A_LOD = ("colmesh", "collision", "bb_", "nulldetail")  # node names that aren't render meshes
COLLISION_MESH = ("colmesh",)                               # node names of the collision meshes
COLLISION_DZ = 0.5                                          # m, height band of the collision reach
ROOT_FOLDERS = ("levels/", "assets/", "art/", "core/")      # a path starting so is from the root


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(el, name: str):
    return [c for c in el if _local(c.tag) == name]


def _iter(el, name: str):
    return [c for c in el.iter() if _local(c.tag) == name]


class LevelZip:
    """A stock level zip with case-insensitive member lookup."""

    def __init__(self, path: Path):
        self.zip = zipfile.ZipFile(path)
        self.members = {n.lower(): n for n in self.zip.namelist()}

    def member(self, path: str) -> str | None:
        return self.members.get(path.lstrip("/").lower())

    def read(self, member: str) -> bytes:
        return self.zip.read(member)

    def json(self, member: str):
        return json.loads(self.read(member).decode("utf-8-sig"))


def game_version(game_dir: Path) -> str:
    """'0.39.4' from integrity.json's '0.39.4.0'."""
    try:
        version = json.loads((game_dir / "integrity.json").read_text(encoding="utf-8"))["version"]
    except (OSError, KeyError, ValueError):
        return "unknown"
    return ".".join(str(version).split(".")[:3])


def forest_items(lz: LevelZip) -> dict[str, dict]:
    """Stock item name -> ForestItemData entry, from art/forest/*Data.json (the loader's
    /%a+Data%.json$ pattern)."""
    out = {}
    for low, member in sorted(lz.members.items()):
        if "/art/forest/" in low and re.search(r"/[a-z]+data\.json$", low):
            for key, entry in lz.json(member).items():
                if isinstance(entry, dict):
                    out[entry.get("name") or key] = entry
    return out


def find_item(items: dict[str, dict], name: str) -> tuple[str, dict]:
    """The stock item called name, else the one whose shape file is called name.dae."""
    if name in items:
        return name, items[name]
    by_shape = sorted(k for k, v in items.items()
                      if posixpath.splitext(posixpath.basename(str(v.get("shapeFile", ""))))[0].lower()
                      == name.lower())
    if not by_shape:
        raise LookupError(f"no forest item {name!r} (nor one for {name}.dae)")
    print(f"  {name}: no item of that name; using {by_shape[0]!r}, whose shape is {name}.dae")
    return by_shape[0], items[by_shape[0]]


def _node_matrix(node) -> np.ndarray:
    """The node's local transform from its <matrix>, <translate>, <rotate> and <scale>, in order."""
    m = np.eye(4)
    for c in node:
        tag, vals = _local(c.tag), [float(v) for v in (c.text or "").split()]
        if tag == "matrix":
            t = np.asarray(vals).reshape(4, 4)  # row-major, column vectors
        elif tag == "translate":
            t = np.eye(4)
            t[:3, 3] = vals
        elif tag == "scale":
            t = np.diag([*vals, 1.0])
        elif tag == "rotate":
            x, y, z, deg = vals
            axis = np.asarray([x, y, z]) / np.linalg.norm([x, y, z])
            a = np.radians(deg)
            k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            t = np.eye(4)
            t[:3, :3] = np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * k @ k
        else:
            continue
        m = m @ t
    return m


def _positions(geometry) -> np.ndarray:
    mesh = _children(geometry, "mesh")[0]
    sources = {s.get("id"): s for s in _children(mesh, "source")}
    (verts,) = _children(mesh, "vertices")
    src = next(i.get("source") for i in _children(verts, "input") if i.get("semantic") == "POSITION")
    (arr,) = _iter(sources[src.lstrip("#")], "float_array")
    return np.asarray([float(v) for v in arr.text.split()]).reshape(-1, 3)


def _triangles(geometry, positions: np.ndarray) -> np.ndarray:
    """(T, 3, 3) corner positions of a mesh's <triangles>."""
    out = []
    for tri in _children(_children(geometry, "mesh")[0], "triangles"):
        inputs = _children(tri, "input")
        stride = max(int(i.get("offset", 0)) for i in inputs) + 1
        vertex = next(int(i.get("offset", 0)) for i in inputs if i.get("semantic") == "VERTEX")
        (p,) = _children(tri, "p")
        idx = np.asarray([int(v) for v in p.text.split()]).reshape(-1, stride)[:, vertex]
        out.append(positions[idx.reshape(-1, 3)])
    return np.concatenate(out) if out else np.empty((0, 3, 3))


def collision_profile(tris: np.ndarray, dz: float = COLLISION_DZ) -> dict | None:
    """{"z0", "dz", "r"} of (T, 3, 3) triangles (x, y, z up): r[k] is the largest distance from
    the z axis of the mesh's parts between z0 + k dz and z0 + (k + 1) dz, rounded up to 1 cm.

    Exact: the part of a triangle in a band is a convex polygon whose corners are the
    triangle's corners in the band and its edges' crossings with the band's two planes.
    """
    if not len(tris):
        return None
    corners = tris.reshape(-1, 3)
    a, b = tris.reshape(-1, 3), tris[:, [1, 2, 0]].reshape(-1, 3)  # every edge, a -> b
    z0 = math.floor(corners[:, 2].min() / dz) * dz
    bands = max(1, math.ceil((corners[:, 2].max() - z0) / dz - 1e-9))
    rise = b[:, 2] - a[:, 2]
    reach = []
    for k in range(bands):
        lo, hi = z0 + k * dz, z0 + (k + 1) * dz
        pts = [corners[(corners[:, 2] >= lo) & (corners[:, 2] <= hi), :2]]
        for plane in (lo, hi):
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (plane - a[:, 2]) / rise
            cut = (rise != 0) & (t >= 0) & (t <= 1)
            pts.append(a[cut, :2] + t[cut, None] * (b[cut, :2] - a[cut, :2]))
        p = np.vstack(pts)
        r = float(np.hypot(p[:, 0], p[:, 1]).max()) if len(p) else 0.0
        reach.append(math.ceil(r * 100 - 1e-6) / 100)
    return {"z0": round(z0, 2), "dz": dz, "r": reach}


def dae_info(xml: bytes) -> tuple[list[str], float, dict | None]:
    """(material names the shape binds, native height, collision reach) of a tree DAE."""
    root = ET.fromstring(xml)
    names = {}
    for mat in _iter(root, "material"):
        mid = mat.get("id") or ""
        names[mid] = mat.get("name") or (mid[:-9] if mid.endswith("-material") else mid)
    used = sorted({names.get(im.get("target", "").lstrip("#"), im.get("target", "").lstrip("#"))
                   for im in _iter(root, "instance_material")})
    geometries = {g.get("id"): g for g in _iter(root, "geometry")}
    up = next((u.text.strip() for u in _iter(root, "up_axis") if u.text), "Z_UP")
    axis = {"X_UP": 0, "Y_UP": 1}.get(up, 2)
    unit = next((float(u.get("meter", 1)) for u in _iter(root, "unit")), 1.0)

    lods, collision = [], []  # (detail pixels, world matrix, geometry ids); (world matrix, geometry ids)

    def walk(node, parent):
        world = parent @ _node_matrix(node)
        name = node.get("name") or node.get("id") or ""
        size = re.search(r"(\d+)$", name)
        geos = [g.get("url", "").lstrip("#") for g in _children(node, "instance_geometry")]
        if geos and size and not name.lower().startswith(NOT_A_LOD):
            lods.append((int(size.group(1)), world, geos))
        if geos and name.lower().startswith(COLLISION_MESH):
            collision.append((world, geos))
        for child in _children(node, "node"):
            walk(child, world)

    for scene in _iter(root, "visual_scene"):
        for node in _children(scene, "node"):
            walk(node, np.eye(4))
    if not lods:
        raise ValueError("no LOD mesh (a node named ..._a<size>) in the shape")
    _, world, geos = max(lods, key=lambda lod: lod[0])
    pts = np.vstack([_positions(geometries[g]) for g in geos])
    z = (np.column_stack([pts, np.ones(len(pts))]) @ world.T)[:, axis] * unit
    up_last = [i for i in range(3) if i != axis] + [axis]  # (horizontal, horizontal, up)
    tris = []
    for world, geos in collision:
        for g in geos:
            p = _positions(geometries[g])
            p = (np.column_stack([p, np.ones(len(p))]) @ world.T)[:, up_last] * unit
            tris.append(_triangles(geometries[g], p))
    return used, float(z.max() - z.min()), collision_profile(np.concatenate(tris) if tris else np.empty((0, 3, 3)))


def texture_paths(material: dict) -> list[str]:
    """Every texture file a Material names: the *Map / *Tex strings in its Stages and on itself."""
    out = []
    for d in [material, *[s for s in material.get("Stages", []) if isinstance(s, dict)]]:
        out += [v for k, v in d.items() if isinstance(v, str) and k.endswith(("Map", "Tex"))]
    return out


def _absolute(path: str, folder: str) -> str:
    if path.startswith("/"):
        return path
    if path.lower().startswith(ROOT_FOLDERS):
        return "/" + path
    return "/" + posixpath.normpath(posixpath.join(folder, path))


def absolute_material(material: dict, member: str) -> dict:
    """A copy with texture paths made absolute (relative ones are from the file's folder)
    and the stock persistentId dropped."""
    folder = posixpath.dirname(member)
    out = {k: v for k, v in material.items() if k != "persistentId"}
    out["Stages"] = [dict(s) if isinstance(s, dict) else s for s in material.get("Stages", [])]
    for d in [out, *[s for s in out["Stages"] if isinstance(s, dict)]]:
        for k, v in d.items():
            if isinstance(v, str) and k.endswith(("Map", "Tex")):
                d[k] = _absolute(v, folder)
    return out


def material_index(lz: LevelZip) -> dict[str, list[tuple[str, dict]]]:
    """Material name (its mapTo) -> [(materials file, definition)] for every *materials.json."""
    out: dict[str, list[tuple[str, dict]]] = {}
    for low, member in sorted(lz.members.items()):
        if not low.endswith("materials.json"):
            continue
        try:
            defs = lz.json(member)
        except ValueError as e:  # a few stock files are not valid JSON; none holds tree materials
            print(f"  skipped {member}: {e}")
            continue
        for key, m in defs.items():
            if isinstance(m, dict):
                out.setdefault(m.get("mapTo") or m.get("name") or key, []).append((member, m))
    return out


def _nearest(defs: list[tuple[str, dict]], shape_member: str) -> tuple[str, dict]:
    """Of several definitions of one name, the one in the folder closest to the shape's."""
    folder = posixpath.dirname(shape_member).lower()
    return max(defs, key=lambda d: (len(posixpath.commonpath([folder, posixpath.dirname(d[0]).lower()])),
                                    -len(d[0])))


def extract(game_dir, candidates=CANDIDATES) -> dict:
    game_dir = Path(game_dir)
    items, materials, conflicts, mat_level = {}, {}, {}, {}
    zips: dict[str, tuple[LevelZip, dict, dict]] = {}
    for key, level, name in candidates:
        if level not in zips:
            lz = LevelZip(game_dir / "content" / "levels" / f"{level}.zip")
            zips[level] = (lz, forest_items(lz), material_index(lz))
        lz, forest, mat_defs = zips[level]
        stock_name, entry = find_item(forest, name)
        shape = lz.member(str(entry["shapeFile"]))
        if shape is None:
            raise LookupError(f"{level} {stock_name}: shape {entry['shapeFile']} is not in {level}.zip")
        used, height, collision = dae_info(lz.read(shape))
        for mat in used:
            if mat not in mat_defs:
                raise LookupError(f"{level} {stock_name}: material {mat!r} is not defined in {level}.zip")
            member, stock = _nearest(mat_defs[mat], shape)
            definition = absolute_material(stock, member)
            if mat not in materials:
                materials[mat], mat_level[mat] = definition, level
            elif definition != materials[mat] and level not in conflicts.get(mat, [mat_level[mat]]):
                conflicts.setdefault(mat, [mat_level[mat]]).append(level)
        items[key] = {"source": stock_name, "level": level, "shapeFile": "/" + shape,
                      "native_height": round(height, 2), "materials": used, "collision": collision,
                      "item": {k: v for k, v in entry.items() if k not in LEVEL_FIELDS}}
        reach = max(collision["r"]) if collision else 0.0
        print(f"{key}: {level} {stock_name}, {height:.1f} m, {len(used)} materials, collision {reach:.1f} m out")
    for mat, levels in conflicts.items():
        print(f"warning: material {mat} differs in {', '.join(levels[1:])}; {levels[0]}'s is used")
    return {"game_version": game_version(game_dir), "items": items,
            "materials": dict(sorted(materials.items())), "material_conflicts": conflicts}


def missing_textures(game_dir, materials: dict) -> list[tuple[str, str]]:
    """(material, path) for each texture the game does not ship (as .dds or .png)."""
    content = Path(game_dir) / "content"
    shipped = set()
    for zp in [*sorted((content / "assets").rglob("*.zip")), *sorted((content / "levels").glob("*.zip"))]:
        with zipfile.ZipFile(zp) as z:
            shipped |= {n.lower().rsplit(".", 1)[0] for n in z.namelist() if n.lower().endswith((".dds", ".png"))}
    return [(name, p) for name, m in materials.items() for p in texture_paths(m)
            if p.lstrip("/").lower().rsplit(".", 1)[0] not in shipped]


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    game_dir = Path(argv[0] if argv else os.environ.get("BEAMNG_GAME_DIR", STEAM_GAME))
    out = Path(argv[1]) if len(argv) > 1 else OUT
    data = extract(game_dir)
    for name, path in missing_textures(game_dir, data["materials"]):
        print(f"warning: {name} names {path}, which the game does not ship")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    print(f"{len(data['items'])} items and {len(data['materials'])} materials from BeamNG.drive "
          f"{data['game_version']} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
