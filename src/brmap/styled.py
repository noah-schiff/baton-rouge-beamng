"""Styled buildings: footprints drawn in a style of their own instead of the generic tiles (the LSU
campus core in tan stucco and red clay tile), as tiles of meshes built in numpy.

The registry, src/brmap/data/styled.yaml (load_registry), holds {"styles": {name: {wall, floor_m,
bay_m, roof, roof_material, cornice, floors_per_tile?, bays_per_tile?}}, "groups": {group: {style,
ids, floor_m?}}}: a wall texture repeat holds floors_per_tile floors and bays_per_tile bays (1
each by default), and a group's own floor_m (its buildings' building:levels) takes the place of
its style's (group_style). buildings.run moves each group's buildings (their parts and tower
parts with them) from out/buildings.json to out/styled.json (buildings.split_styled):
[{"style", "group", "building": Building.to_dict()}], read back by load_styled.

* The outlines (_outline): the footprint orthogonalized (buildings.orthogonalize with
  rect_cover=None, so never snapped to its bounding rectangle) along whichever axis keeps it
  nearer the footprint (Hausdorff, the edges sampled every OFF_STEP): the long axis of the
  footprint's minimum rotated rectangle, or the walls' own direction (the length-weighted mean of
  the edge directions, modulo 90 degrees; a staircase of wings turns the rectangle off them).
  The walls follow that outline (the frame) when it stays within OUTLINE_TOL of the footprint,
  and the footprint itself otherwise (diagonal walls a right-angled outline can't follow: a
  chamfered corner, a polygonal apse), so the walls never stray more than OUTLINE_TOL from the
  OSM outline.
* rectangles(poly) -> [Polygon]: what the walls enclose broken into rectangles along the axis, on
  the grid through the corners of the walls and of the frame, greedily: each time the rectangle
  of the grid's cells that covers the most area not yet covered (on a tie the larger one, so they
  may overlap where wings cross), until they cover COVER of the cells or there are MAX_RECTS.
  Only cells wholly inside the walls count, so a rectangle never reaches past a wall and a
  hipped roof never hangs more than its eaves past one (EAVES x sqrt 2 at a corner); along a
  diagonal wall the flat roof at the eaves covers the wedge the cells leave. On right-angled walls
  these are the cells whose centres are inside them.
* drawn(building) -> the Building with its ring and holes the outline its walls follow;
  drawn_buildings(path) gives them for a styled.json ([] without the file). trees.run keeps the
  trees out of them, and validate checks them, as it does buildings.json.
* mesh(buildings, style, origin) -> {FULL, LOW, COLLISION: collada.Mesh}, x and y relative to
  origin, z world, every vertex colour white:
  - FULL (sty_a400): walls along the outline from the base (the wall material; uv per texture
    repeat: u = metres along the wall / (bay_m x bays_per_tile), shifted so that each wall starts
    and ends in the pier between two bays; v = metres above the base / (floor_m x
    floors_per_tile)); a cornice band CORNICE_DEPTH out from the walls and CORNICE_HEIGHT high at
    their top (the cornice material, uv in metres / TRIM_TILE); and the roof in roof_material (uv in metres / ROOF_TILE):
    * hip_tile: a hipped roof on each rectangle grown by EAVES, its ridge along the long side,
      rising HIP_SLOPE x the rectangle's short side, at most HIP_MAX_RISE and leaving MIN_WALL of
      wall (a rise under MIN_RISE is no roof). The highest reaches the building's top (base_z +
      height); the walls stop at the eaves, that rise below it. A flat roof at the eaves covers
      what the rectangles leave, and a soffit closes the eaves from below.
    * flat_parapet: the walls go up to the top; the roof is flat PARAPET below it, inside a
      parapet PARAPET_WALL thick capped in the cornice material.
  - LOW (sty_a60): the outline as a prism with a flat roof at the eaves plus half the rise.
  - COLLISION (Colmesh-1): the same prism, closed, in the wall material (its groundType).
* level_content(cfg, data_dir, root, heights, extent) (the level.FEATURE_MODULES hook): per group,
  in its style (group_style), its buildings stood on the level's terrain along their drawn
  outlines (buildings.settle), in tiles of cfg.building_tile_m (buildings.tiles), each written to
  SHAPES_DIR/br_sty_<group>_<key>.dae (collada.TILE decimals; no nulldetail level, so the far
  level stays drawn) and placed by one TSStatic at the tile origin, z 0, unrotated, with
  persistentId bf.persistent_id("sty", group, key), in group GROUP. The materials the shapes
  bind come from landmark_materials.yaml (materials.resolve) into SHAPES_DIR/main.materials.json
  (materials.write_materials), and their tex: textures are copied into the level.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import math
from pathlib import Path

import numpy as np
import shapely
import yaml
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from . import beamng_format as bf
from . import collada, dae, materials
from .buildings import STYLED_FILE, Building, _long_axis, orthogonalize, settle, tiles, wet_areas
from .water import load_water

log = logging.getLogger(__name__)

DATA = Path(__file__).resolve().parent / "data"
REGISTRY = DATA / "styled.yaml"
MATERIAL_REGISTRY = DATA / "landmark_materials.yaml"
STYLE_KEYS = ("wall", "floor_m", "bay_m", "roof", "roof_material", "cornice")
STYLE_OPTIONAL = ("floors_per_tile", "bays_per_tile")   # per texture repeat; 1 each by default
GROUP_KEYS = ("style", "ids", "floor_m")                 # floor_m: the group's own floor height
ROOFS = ("hip_tile", "flat_parapet")
FULL, LOW, COLLISION = "sty_a400", "sty_a60", "Colmesh-1"
NODES = (FULL, LOW, COLLISION)
SHAPES_DIR = "art/shapes/styled"
MATERIALS_FILE = f"{SHAPES_DIR}/main.materials.json"
GROUP = "styled"

MAX_RECTS = 8          # rectangles per footprint at most
COVER = 0.97           # ... or as soon as they cover this share of the outline
SNAP = 1e-3            # m; outline coordinates closer than this are one line of the rectangles' grid
OUTLINE_TOL = 1.0      # m an orthogonalized outline may stray from the footprint before the walls keep the footprint
OFF_STEP = 0.25        # m between the points the outlines are compared at
HIP_SLOPE = 0.35       # a hipped roof rises 0.35 x its rectangle's short side ...
HIP_MAX_RISE = 4.0     # ... up to 4 m
MIN_WALL = dae.MIN_WALL    # m of wall a roof always leaves standing
MIN_RISE = dae.MIN_RISE    # m; a hipped roof any flatter is left out (the flat roof at the eaves shows)
EAVES = 0.6            # m the roofs reach past the rectangles
CORNICE_DEPTH = 0.4    # m the cornice stands out from the walls
CORNICE_HEIGHT = 0.5   # m
PARAPET = 1.0          # m from the flat roof to the parapet's top
PARAPET_WALL = 0.3     # m thick
ROOF_TILE = 1.5        # m per repeat of the roof texture (as on the Memorial Tower)
TRIM_TILE = 0.6        # m per repeat of the cornice texture (one block, as the tower's trim)
EPS = 1e-6
WHITE = (1.0, 1.0, 1.0, 1.0)


# ------------------------------------------------------------------ the registry and the file

def load_registry(path=None) -> dict:
    """styled.yaml (REGISTRY by default) as {"styles": {...}, "groups": {...}}, checked."""
    path = Path(path or REGISTRY)
    reg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    styles, groups = reg.get("styles") or {}, reg.get("groups") or {}
    for name, style in styles.items():
        missing = [k for k in STYLE_KEYS if k not in (style or {})]
        if missing:
            raise ValueError(f"style {name} in {path.name} lacks {', '.join(missing)}")
        unknown = sorted(set(style) - set(STYLE_KEYS) - set(STYLE_OPTIONAL))
        if unknown:
            raise ValueError(f"style {name} in {path.name}: unknown key {', '.join(unknown)}")
        if style["roof"] not in ROOFS:
            raise ValueError(f"style {name}: roof {style['roof']!r} is not one of {', '.join(ROOFS)}")
    for group, g in groups.items():
        if (g or {}).get("style") not in styles:
            raise ValueError(f"styled group {group} in {path.name} names no style of the file")
        unknown = sorted(set(g) - set(GROUP_KEYS))
        if unknown:
            raise ValueError(f"styled group {group} in {path.name}: unknown key {', '.join(unknown)}")
    return {"styles": styles, "groups": groups}


def group_style(registry: dict, group: str, style: str | None = None) -> dict:
    """The style a group's buildings are drawn in: its style (or `style`) with the group's own
    floor_m, when it has one, in place of the style's."""
    g = registry["groups"].get(group) or {}
    out = dict(registry["styles"][style or g["style"]])
    if "floor_m" in g:
        out["floor_m"] = g["floor_m"]
    return out


def load_styled(path) -> list[dict]:
    """A styled.json: its entries with "building" as a Building."""
    return [{**e, "building": Building.from_dict(e["building"])}
            for e in json.loads(Path(path).read_text(encoding="utf-8"))]


# ------------------------------------------------------------------ outlines and rectangles

def _turn(axis: float) -> np.ndarray:
    """(row vector) @ _turn(axis) turns a point by -axis; @ .T turns it back (as orthogonalize)."""
    c, s = math.cos(axis), math.sin(axis)
    return np.array([[c, -s], [s, c]])


def _parts(geom) -> list[Polygon]:
    return [p for p in dae._polygon_parts(geom) if p.area > EPS] if not geom.is_empty else []


def _polygon(b: Building) -> Polygon:
    """The building's footprint; the largest part of it repaired, when it isn't valid."""
    poly = Polygon(b.ring, b.holes) if len(b.ring) >= 3 else Polygon()
    if not poly.is_valid:
        poly = max(_parts(shapely.make_valid(poly)), key=lambda p: p.area, default=Polygon())
    return poly


def _wall_axis(poly: Polygon) -> float:
    """The direction the walls run: the length-weighted mean of the outline's edge directions
    modulo 90 degrees, as whichever of it and it + 90 degrees is nearer _long_axis(poly)."""
    xy = np.asarray(poly.exterior.coords)[:, :2]
    d = np.diff(xy, axis=0)
    mean = float(np.angle(np.sum(np.hypot(d[:, 0], d[:, 1]) * np.exp(4j * np.arctan2(d[:, 1], d[:, 0]))))) / 4
    long = _long_axis(poly)
    return min((mean % math.pi, (mean + math.pi / 2) % math.pi),
               key=lambda a: abs((a - long + math.pi / 2) % math.pi - math.pi / 2))


def _off(a, b) -> float:
    """The Hausdorff distance between two outlines, their edges sampled every OFF_STEP."""
    return float(shapely.hausdorff_distance(shapely.segmentize(a, OFF_STEP), shapely.segmentize(b, OFF_STEP)))


def _outline(poly: Polygon) -> tuple[Polygon, Polygon, float]:
    """(walls, frame, axis): frame is poly orthogonalized (never its bounding rectangle:
    rect_cover=None) along whichever of _long_axis(poly) and _wall_axis(poly) keeps it nearer
    poly, and axis that one; walls is frame when it stays within OUTLINE_TOL of poly (_off), else
    poly itself (diagonal walls a right-angled outline can't follow: a chamfered corner, a
    polygonal end). Both counter-clockwise with clockwise holes. The roof's rectangles lie inside
    walls, on the grid through the corners of both (_boxes(walls, axis, frame))."""
    tries = []
    for axis in (_long_axis(poly), _wall_axis(poly)):
        frame = orient(orthogonalize(poly, axis, rect_cover=None), 1.0)
        tries.append((_off(frame, poly), frame, axis))
    off, frame, axis = min(tries, key=lambda t: t[0])
    return (frame if off <= OUTLINE_TOL else orient(poly, 1.0)), frame, axis


def _lines(values: np.ndarray) -> np.ndarray:
    """The distinct values, those closer than SNAP to the one before taken as it."""
    v = np.sort(values)
    keep = [v[0]]
    for x in v[1:]:
        if x - keep[-1] > SNAP:
            keep.append(x)
    return np.array(keep)


def _runs(ok: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) of each run of True, end exclusive."""
    edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(np.int8), 0]))
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def _boxes(walls: Polygon, axis: float, frame: Polygon | None = None) -> list[tuple[float, float, float, float]]:
    """The rectangles (x0, y0, x1, y1) of rectangles(), turned by -axis: the grid runs through the
    corners of walls and of frame (when given), and only its cells wholly inside walls (to within
    2 SNAP, the grid's snapping) count."""
    turn = _turn(axis)
    inner = Polygon(np.asarray(walls.exterior.coords)[:, :2] @ turn,
                    [np.asarray(r.coords)[:, :2] @ turn for r in walls.interiors])
    pts = np.vstack([np.asarray(r.coords)[:, :2] for p in (walls, frame) if p is not None
                     for r in (p.exterior, *p.interiors)]) @ turn
    xs, ys = _lines(pts[:, 0]), _lines(pts[:, 1])
    if len(xs) < 2 or len(ys) < 2:
        return []
    cx, cy = np.meshgrid((xs[:-1] + xs[1:]) / 2, (ys[:-1] + ys[1:]) / 2, indexing="ij")
    room = inner.buffer(2 * SNAP, join_style="mitre")
    shapely.prepare(room)
    inside = (shapely.contains_xy(inner, cx, cy)               # (columns, rows) of the grid's cells
              & shapely.covers(room, shapely.box(xs[:-1, None], ys[None, :-1], xs[1:, None], ys[None, 1:])))
    cell = np.diff(xs)[:, None] * np.diff(ys)[None, :]
    total = float((cell * inside).sum())
    covered = np.zeros_like(inside)
    boxes = []
    while len(boxes) < MAX_RECTS and float((cell * covered).sum()) < COVER * total:
        best, key = None, (0.0, 0.0)
        for i0 in range(inside.shape[0]):
            ok, new = np.ones(inside.shape[1], bool), np.zeros(inside.shape[1])
            for i1 in range(i0, inside.shape[0]):
                ok &= inside[i1]
                if not ok.any():
                    break
                new += cell[i1] * ~covered[i1]
                for j0, j1 in _runs(ok):        # all of a run: a longer rectangle never covers less
                    k = (round(float(new[j0:j1].sum()), 6),
                         round(float((xs[i1 + 1] - xs[i0]) * (ys[j1] - ys[j0])), 6))
                    if k > key:
                        best, key = (i0, i1 + 1, j0, j1), k
        if best is None or key[0] <= 0.0:
            break
        i0, i1, j0, j1 = best
        covered[i0:i1, j0:j1] = True
        boxes.append((float(xs[i0]), float(ys[j0]), float(xs[i1]), float(ys[j1])))
    return boxes


def _box_polygon(box, axis: float, grow: float = 0.0) -> Polygon:
    x0, y0, x1, y1 = box
    corners = np.array([(x0 - grow, y0 - grow), (x1 + grow, y0 - grow), (x1 + grow, y1 + grow),
                        (x0 - grow, y1 + grow)])
    return Polygon(corners @ _turn(axis).T)


def rectangles(poly: Polygon) -> list[Polygon]:
    """What the footprint's walls enclose broken into rectangles; see the module docstring."""
    walls, frame, axis = _outline(poly)
    return [_box_polygon(b, axis) for b in _boxes(walls, axis, frame)]


def drawn(b: Building) -> Building:
    """b with the outline its walls follow (_outline's walls) as its ring and holes."""
    poly = _polygon(b)
    if poly.is_empty or poly.area <= EPS:
        return b
    outline, _, _ = _outline(poly)
    return dataclasses.replace(b, ring=np.asarray(outline.exterior.coords)[:-1, :2],
                               holes=[np.asarray(r.coords)[:-1, :2] for r in outline.interiors])


def drawn_buildings(path) -> list[Building]:
    """The drawn() buildings of a styled.json; [] when there is no such file."""
    path = Path(path)
    return [drawn(e["building"]) for e in load_styled(path)] if path.exists() else []


# ------------------------------------------------------------------ mesh pieces

def _rings(polys) -> dae._Rings:
    return dae._rings(dae._objects(list(polys)))


def _sides(mesh: dae._Mesh, r: dae._Rings, origin, z0: float, z1: float, material: str, along: float,
           up: float, v0: float = 0.0, centred: bool = False, inward: bool = False, bays: int = 1) -> None:
    """Upright faces along the rings r from z0 to z1, four vertices each, facing out of the
    polygons (into them with inward). uv: v = (z - v0) / up; u = metres along the face / along,
    from 0 at its start, or with centred shifted by half the difference between the face's length
    in bays (a repeat holds `bays`) and the nearest whole number, so both ends fall within a
    quarter bay of a bay's edge (on the walls drawn, the pier between two windows: no window is
    cut at a corner)."""
    if not len(r.points) or z1 <= z0:
        return
    a = r.points - origin
    b = a[dae._next_in_ring(r.sizes)]
    if inward:
        a, b = b, a
    d = b - a
    length = np.hypot(d[:, 0], d[:, 1])
    a, b, d, length = a[length > EPS], b[length > EPS], d[length > EPS], length[length > EPS]
    m = len(a)
    n = length / along
    k = n * bays                                                   # in bays
    u0 = -(k - np.floor(k + 0.5)) / 2 / bays if centred else np.zeros(m)
    pos = np.empty((m, 4, 3))
    pos[:, 0, :2], pos[:, 1, :2], pos[:, 2, :2], pos[:, 3, :2] = a, b, b, a
    pos[:, :2, 2], pos[:, 2:, 2] = z0, z1
    nrm = np.column_stack([d[:, 1] / length, -d[:, 0] / length, np.zeros(m)])
    uv = np.empty((m, 4, 2))
    uv[:, [0, 3], 0] = u0[:, None]
    uv[:, [1, 2], 0] = (u0 + n)[:, None]
    uv[:, :2, 1], uv[:, 2:, 1] = (z0 - v0) / up, (z1 - v0) / up
    tris = (np.arange(m)[:, None, None] * 4 + np.array([[0, 1, 2], [0, 2, 3]])).reshape(-1, 3)
    mesh.add(pos.reshape(-1, 3), np.repeat(nrm, 4, axis=0), uv.reshape(-1, 2), np.tile(WHITE, (4 * m, 1)),
             tris, np.full(len(tris), mesh.code(material)))


def _flat(mesh: dae._Mesh, polys, origin, z: float, material: str, tile: float, down: bool = False) -> None:
    """Level faces over the polygons at z, facing up (down with down); uv in metres / tile."""
    polys = [p for g in polys for p in _parts(g)]
    if not polys:
        return
    r = _rings(polys)
    tris = dae._caps(r)
    if down:
        tris = tris[:, ::-1]
    local = r.points - origin
    mesh.add(np.column_stack([local, np.full(len(local), z)]),
             np.tile([0.0, 0.0, -1.0 if down else 1.0], (len(local), 1)), local / tile,
             np.tile(WHITE, (len(local), 1)), tris, np.full(len(tris), mesh.code(material)))


def _hip(mesh: dae._Mesh, box, axis: float, eave: float, rise: float, origin, material: str) -> None:
    """A hipped roof on the rectangle box (of _boxes, turned by -axis) grown by EAVES: two slopes along
    its long side and a hip at each end (a pyramid on a square), the eaves at z eave."""
    x0, y0, x1, y1 = box[0] - EAVES, box[1] - EAVES, box[2] + EAVES, box[3] + EAVES
    c0, c1, c2, c3 = (x0, y0), (x1, y0), (x1, y1), (x0, y1)
    w, h = x1 - x0, y1 - y0
    if w >= h:
        r0, r1 = (x0 + h / 2, (y0 + y1) / 2), (x1 - h / 2, (y0 + y1) / 2)
        faces = ((c0, c1, r1, r0), (c1, c2, r1), (c2, c3, r0, r1), (c3, c0, r0))
    else:
        r0, r1 = ((x0 + x1) / 2, y0 + w / 2), ((x0 + x1) / 2, y1 - w / 2)
        faces = ((c0, c1, r0), (c1, c2, r1, r0), (c2, c3, r1), (c3, c0, r0, r1))
    ridge = {r0, r1}
    turn = _turn(axis)
    for face in faces:
        xy = np.array(face) @ turn.T - origin
        p = np.column_stack([xy, [eave + rise if q in ridge else eave for q in face]])
        n = np.cross(p[1] - p[0], p[2] - p[0])
        n /= np.linalg.norm(n)
        right = np.cross([0.0, 0.0, 1.0], n)
        right /= np.linalg.norm(right)
        uv = np.column_stack([p @ right, p @ np.cross(n, right)]) / ROOF_TILE   # u along, v up the slope
        tris = np.array([[0, 1, 2], [0, 2, 3]] if len(face) == 4 else [[0, 1, 2]])
        area = np.linalg.norm(np.cross(p[tris[:, 1]] - p[tris[:, 0]], p[tris[:, 2]] - p[tris[:, 0]]), axis=1)
        tris = tris[area > EPS]                                     # on a square the ridge is a point
        mesh.add(p, np.tile(n, (len(p), 1)), uv, np.tile(WHITE, (len(p), 1)), tris,
                 np.full(len(tris), mesh.code(material)))


def _cornice(mesh: dae._Mesh, outline: Polygon, origin, z1: float, material: str) -> None:
    """The cornice band round the top of the walls (z1): CORNICE_DEPTH out, CORNICE_HEIGHT high."""
    grown = outline.buffer(CORNICE_DEPTH, join_style="mitre")
    band = grown.difference(outline)
    z0 = z1 - CORNICE_HEIGHT
    _sides(mesh, _rings(_parts(grown)), origin, z0, z1, material, TRIM_TILE, TRIM_TILE)
    _flat(mesh, [band], origin, z0, material, TRIM_TILE, down=True)
    _flat(mesh, [band], origin, z1, material, TRIM_TILE)


def _building(meshes: dict[str, dae._Mesh], b: Building, style: dict, origin) -> None:
    poly = _polygon(b)
    if poly.is_empty or poly.area <= EPS:
        return
    outline, frame, axis = _outline(poly)
    rings = _rings([outline])
    wall, roof, trim = style["wall"], style["roof_material"], style["cornice"]
    bays = int(style.get("bays_per_tile", 1))
    floor = float(style["floor_m"]) * int(style.get("floors_per_tile", 1))     # one texture repeat
    bay = float(style["bay_m"]) * bays
    full = meshes[FULL]
    base, top = float(b.base_z), float(b.base_z + b.height)
    if style["roof"] == "hip_tile":
        boxes = _boxes(outline, axis, frame)
        rises = [min(HIP_SLOPE * min(x1 - x0, y1 - y0), HIP_MAX_RISE) for x0, y0, x1, y1 in boxes]
        rise = min(max(rises, default=0.0), float(b.height) - MIN_WALL)
        rise = rise if rise >= MIN_RISE else 0.0
        eave = top - rise
        hips = [(box, min(r, rise)) for box, r in zip(boxes, rises) if min(r, rise) >= MIN_RISE]
        _sides(full, rings, origin, base, eave, wall, bay, floor, v0=base, centred=True, bays=bays)
        _cornice(full, outline, origin, eave, trim)
        for box, r in hips:
            _hip(full, box, axis, eave, r, origin, roof)
        _flat(full, [outline], origin, eave, roof, ROOF_TILE)                   # under the hips
        eaves = unary_union([_box_polygon(box, axis, EAVES) for box, _ in hips]) if hips else Polygon()
        _flat(full, [eaves.difference(outline)], origin, eave, roof, ROOF_TILE, down=True)  # the soffit
        low_top = eave + rise / 2
    else:  # flat_parapet
        _sides(full, rings, origin, base, top, wall, bay, floor, v0=base, centred=True, bays=bays)
        _cornice(full, outline, origin, top, trim)
        inner = _parts(outline.buffer(-PARAPET_WALL, join_style="mitre"))
        if inner:
            _flat(full, inner, origin, top - PARAPET, roof, ROOF_TILE)
            _sides(full, _rings(inner), origin, top - PARAPET, top, wall, bay, floor, v0=base, centred=True,
                   inward=True, bays=bays)
            _flat(full, [outline.difference(unary_union(inner))], origin, top, trim, TRIM_TILE)
        else:                                                    # too thin for a parapet
            _flat(full, [outline], origin, top, roof, ROOF_TILE)
        low_top = top
    _sides(meshes[LOW], rings, origin, base, low_top, wall, bay, floor, v0=base, centred=True, bays=bays)
    caps = dae._caps(rings)
    dae._flat_roofs(meshes[LOW], rings, caps, origin, [low_top], meshes[LOW].code(roof), [WHITE])
    dae._prisms(meshes[COLLISION], rings, caps, origin, [base], [low_top], meshes[COLLISION].code(wall), [WHITE])


def mesh(buildings: list[Building], style: dict, origin) -> dict[str, collada.Mesh]:
    """The styled buildings' levels of detail; see the module docstring."""
    if style.get("roof") not in ROOFS:
        raise ValueError(f"roof {style.get('roof')!r} is not one of {', '.join(ROOFS)}")
    meshes = {node: dae._Mesh() for node in NODES}
    o = np.asarray(origin, float)
    for b in buildings:
        _building(meshes, b, style, o)
    return {node: m.to_mesh() for node, m in meshes.items()}


# ------------------------------------------------------------------ level files

def ts_static(level_name: str, group: str, key: str, origin) -> dict:
    """The TSStatic placing one tile's shape at the tile origin, unrotated."""
    return {"class": "TSStatic", "persistentId": bf.persistent_id("sty", group, key),
            "shapeName": f"/levels/{level_name}/{SHAPES_DIR}/br_sty_{group}_{key}.dae",
            "position": [round(float(origin[0]), 2), round(float(origin[1]), 2), 0.0],
            "useInstanceRenderData": True, "collisionType": "Collision Mesh", "decalType": "None"}


def _settled(blds: list[Building], heights: np.ndarray, extent, wet) -> list[Building]:
    """blds with the base and height buildings.settle gives their drawn outlines."""
    stood = settle([drawn(b) for b in blds], heights, extent, wet)
    return [dataclasses.replace(b, base_z=s.base_z, height=s.height) for b, s in zip(blds, stood)]


def level_content(cfg, data_dir, root, heights, extent) -> dict[str, list[dict]]:
    data_dir, root = Path(data_dir), Path(root)
    entries = load_styled(data_dir / STYLED_FILE)
    if not entries:
        return {}
    registry = load_registry()
    styles = registry["styles"]
    water = load_water(data_dir / "water.json") if (data_dir / "water.json").exists() else []
    wet = wet_areas(water, extent, heights.shape[0])
    groups: dict[str, list[dict]] = {}
    for e in entries:
        groups.setdefault(e["group"], []).append(e)
    objs, bound, most = [], set(), dict.fromkeys(NODES, 0)
    folder = root / SHAPES_DIR
    folder.mkdir(parents=True, exist_ok=True)
    for group, members in groups.items():
        names = sorted({e["style"] for e in members})
        if len(names) != 1 or names[0] not in styles:
            raise ValueError(f"styled group {group}: its buildings are in styles {', '.join(names)}; "
                             "it needs one style of styled.yaml")
        style = group_style(registry, group, names[0])
        blds = _settled([e["building"] for e in members], heights, extent, wet)
        for key, origin, tile in tiles(blds, cfg.building_tile_m):
            meshes = mesh(tile, style, origin)
            (folder / f"br_sty_{group}_{key}.dae").write_text(
                collada.shape_dae([(node, meshes[node]) for node in NODES], decimals=collada.TILE),
                encoding="utf-8")
            bound |= {m for x in meshes.values() for m in x.tris}
            most = {node: max(most[node], collada.vertex_count(meshes[node])) for node in NODES}
            objs.append(ts_static(cfg.level_name, group, key, origin))
    mats, files = materials.resolve(materials.load_registry(MATERIAL_REGISTRY), bound, cfg.level_name,
                                    materials.texture_index(data_dir))
    written = materials.write_materials(root, MATERIALS_FILE, mats)
    materials.copy_textures(files, data_dir, root)
    log.info("styled: %d buildings (%s) in %d tiles; most vertices in a tile: %s; %d materials (%d written "
             "here), %d textures", len(entries), ", ".join(f"{g} {len(m)}" for g, m in groups.items()),
             len(objs), ", ".join(f"{k} {v}" for k, v in most.items()), len(mats), len(written), len(files))
    return {GROUP: objs}
