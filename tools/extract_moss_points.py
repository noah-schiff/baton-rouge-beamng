"""Write src/brmap/data/moss_points.json: where Spanish moss hangs from each stock live oak.

Reads the installed game (read only: the game is not launched, and its level zips are only opened
with zipfile, never extracted), so the build never needs it. For each live oak of
brmap.trees.LIVE_OAKS it reads the shape stock_trees.json names (shapeFile) from that level's zip in
content/levels and

* takes the triangles of its highest-detail mesh (the node whose name ends in the largest
  _a<pixels>, after its node transforms) bound to a bark material (bark_materials: groundType WOOD
  and no alpha test, so not the cut-out branch cards);
* keeps the undersides of its branches: the triangles facing down (their normal's z at most
  -UNDERSIDE), between LOW and HIGH of native_height above the shape's origin and at least
  MIN_REACH m out from the trunk's axis;
* spreads points over them: CANDIDATES random points, area-weighted, thinned by Poisson-disc
  sampling (a greedy pass in random order keeps a point no nearer than r to those kept) to about
  POINTS_PER_M x native_height of them, 40 to 120, r found by bisection.

    {"br_oak_dry_a": [[x, y, z], ...], ...}   # the oak shape's coordinates, m, z up, 3 decimals

models.run hands an overlay's points to blender/trees/moss_drape.py (custom_trees.yaml's
hang_points_from), which hangs the strands from them.

    python tools/extract_moss_points.py [game_dir] [out.json]

game_dir defaults to $BEAMNG_GAME_DIR, else Steam's default library (extract_stock_trees.STEAM_GAME).
"""
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import extract_stock_trees as est  # noqa: E402
from brmap.trees import LIVE_OAKS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "src" / "brmap" / "data"
OUT = DATA / "moss_points.json"
UNDERSIDE = 0.3            # a triangle faces down when its normal's z is at most -UNDERSIDE
LOW, HIGH = 0.35, 0.9      # shares of native_height above the origin where moss hangs
EDGE = 0.001               # m kept inside LOW and HIGH, so the rounded points stay in them
MIN_REACH = 0.6            # m from the trunk's axis: no moss in the trunk's crotches
CANDIDATES = 4000
POINTS_PER_M = 7.0         # points per m of native_height ...
POINTS = (40, 120)         # ... within these
SEED = 0


def bark_materials(materials: dict) -> set[str]:
    """The stock materials that are bark: groundType WOOD and no alpha test (the cut-out branch
    cards are WOOD too)."""
    return {name for name, m in materials.items()
            if str(m.get("groundType", "")).upper() == "WOOD" and not m.get("alphaTest")}


def lod_triangles(xml: bytes) -> dict[str, np.ndarray]:
    """{material name: (T, 3, 3) corners} of a tree DAE's highest-detail mesh, in metres, z up,
    after its node transforms."""
    root = ET.fromstring(xml)
    up = next((u.text.strip() for u in est._iter(root, "up_axis") if u.text), "Z_UP")
    if up != "Z_UP":
        raise ValueError(f"the shape is {up}; only Z_UP shapes are read")
    unit = next((float(u.get("meter", 1)) for u in est._iter(root, "unit")), 1.0)
    names = {m.get("id"): m.get("name") or m.get("id") for m in est._iter(root, "material")}
    geometries = {g.get("id"): g for g in est._iter(root, "geometry")}
    lods = []

    def walk(node, parent):
        world = parent @ est._node_matrix(node)
        name = node.get("name") or node.get("id") or ""
        size = re.search(r"(\d+)$", name)
        instances = est._children(node, "instance_geometry")
        if instances and size and not name.lower().startswith(est.NOT_A_LOD):
            lods.append((int(size.group(1)), world, instances))
        for child in est._children(node, "node"):
            walk(child, world)

    for scene in est._iter(root, "visual_scene"):
        for node in est._children(scene, "node"):
            walk(node, np.eye(4))
    if not lods:
        raise ValueError("no LOD mesh (a node named ..._a<size>) in the shape")
    _, world, instances = max(lods, key=lambda lod: lod[0])
    out: dict[str, list[np.ndarray]] = {}
    for inst in instances:
        bound = {im.get("symbol"): names.get(im.get("target", "").lstrip("#"), im.get("target", "").lstrip("#"))
                 for im in est._iter(inst, "instance_material")}
        geometry = geometries[inst.get("url", "").lstrip("#")]
        pos = est._positions(geometry)
        pos = (np.column_stack([pos, np.ones(len(pos))]) @ world.T)[:, :3] * unit
        for tri in est._children(est._children(geometry, "mesh")[0], "triangles"):
            inputs = est._children(tri, "input")
            stride = max(int(i.get("offset", 0)) for i in inputs) + 1
            vertex = next(int(i.get("offset", 0)) for i in inputs if i.get("semantic") == "VERTEX")
            (p,) = est._children(tri, "p")
            idx = np.asarray([int(v) for v in p.text.split()]).reshape(-1, stride)[:, vertex]
            symbol = tri.get("material")
            out.setdefault(bound.get(symbol, names.get(symbol, symbol)), []).append(pos[idx.reshape(-1, 3)])
    return {name: np.concatenate(parts) for name, parts in out.items()}


def _thin(points: np.ndarray, order: np.ndarray, radius: float) -> np.ndarray:
    """The points kept by a greedy Poisson-disc pass in `order`: none nearer than radius to another."""
    kept = np.empty((0, 3))
    for i in order:
        if not len(kept) or np.min(np.sum((kept - points[i]) ** 2, axis=1)) >= radius * radius:
            kept = np.vstack([kept, points[i]])
    return kept


def hang_points(tris: np.ndarray, native_height: float, seed: int = SEED) -> np.ndarray:
    """(N, 3) points spread over the downward-facing triangles between LOW and HIGH of native_height
    and MIN_REACH or more from the axis; see the module docstring."""
    tris = np.asarray(tris, float).reshape(-1, 3, 3)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    area = np.linalg.norm(n, axis=1) / 2
    down = (area > 1e-9) & (n[:, 2] <= -UNDERSIDE * 2 * np.maximum(area, 1e-12))
    if not down.any():
        raise ValueError("no downward-facing bark")
    t, w = tris[down], area[down]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(t), CANDIDATES, p=w / w.sum())
    r1, r2 = np.sqrt(rng.random(CANDIDATES)), rng.random(CANDIDATES)
    a, b, c = t[pick, 0], t[pick, 1], t[pick, 2]
    cand = (1 - r1)[:, None] * a + (r1 * (1 - r2))[:, None] * b + (r1 * r2)[:, None] * c
    lo, hi = LOW * native_height + EDGE, HIGH * native_height - EDGE
    cand = cand[(cand[:, 2] >= lo) & (cand[:, 2] <= hi) & (np.hypot(cand[:, 0], cand[:, 1]) >= MIN_REACH)]
    target = int(np.clip(round(POINTS_PER_M * native_height), *POINTS))
    if len(cand) < POINTS[0]:
        raise ValueError(f"only {len(cand)} candidate points under the branches; {POINTS[0]} are needed")
    order = rng.permutation(len(cand))
    low, high = 0.0, float(np.ptp(cand, axis=0).max())
    best = cand[order[:target]]
    for _ in range(40):                                    # the largest radius that still keeps the target
        mid = (low + high) / 2
        kept = _thin(cand, order, mid)
        if len(kept) >= target:
            low, best = mid, kept
        else:
            high = mid
    return best[:POINTS[1]]


def extract(game_dir, stock: dict | None = None, items=LIVE_OAKS) -> dict[str, list[list[float]]]:
    """{item: [[x, y, z], ...]} for each live oak, from the game's level zips (read only)."""
    stock = stock or json.loads((DATA / "stock_trees.json").read_text(encoding="utf-8"))
    bark = bark_materials(stock["materials"])
    out = {}
    for item in items:
        entry = stock["items"][item]
        lz = est.LevelZip(Path(game_dir) / "content" / "levels" / f"{entry['level']}.zip")
        member = lz.member(entry["shapeFile"])
        if member is None:
            raise LookupError(f"{item}: {entry['shapeFile']} is not in {entry['level']}.zip")
        by_mat = lod_triangles(lz.read(member))
        used = sorted(set(by_mat) & bark)
        if not used:
            raise LookupError(f"{item}: its highest-detail mesh binds no bark ({', '.join(sorted(by_mat))})")
        h = float(entry["native_height"])
        pts = hang_points(np.concatenate([by_mat[m] for m in used]), h)
        out[item] = [[round(float(v), 3) + 0.0 for v in p] for p in pts]
        print(f"{item}: {len(pts)} points on the undersides of {', '.join(used)}, z {pts[:, 2].min():.2f} to "
              f"{pts[:, 2].max():.2f} m of {h:.2f} m, {np.hypot(pts[:, 0], pts[:, 1]).max():.2f} m out at most")
    return out


def dumps(points: dict) -> str:
    """The JSON text, one point a line."""
    blocks = [f' {json.dumps(item)}: [\n' + ",\n".join(f"  {json.dumps(p)}" for p in pts) + "\n ]"
              for item, pts in points.items()]
    return "{\n" + ",\n".join(blocks) + "\n}\n"


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    game_dir = Path(argv[0] if argv else os.environ.get("BEAMNG_GAME_DIR", est.STEAM_GAME))
    out = Path(argv[1]) if len(argv) > 1 else OUT
    points = extract(game_dir)
    out.write_text(dumps(points), encoding="utf-8")
    print(f"{sum(map(len, points.values()))} moss points on {len(points)} live oaks from BeamNG.drive "
          f"{est.game_version(game_dir)} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
