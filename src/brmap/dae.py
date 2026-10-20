"""COLLADA (.dae) shapes for building tiles: many buildings in one mesh per level of detail.

BeamNG.drive imports a .dae as a TSStatic shape and compiles it to .cdae itself. The
layout follows the stock building shapes and was loaded in game in the spike (0.39.4):

    base00
      start01
        bld_L300      full detail: walls and hipped or flat roofs
        bld_L60       flat roofs on footprints simplified at 2 m
        nulldetail15  an empty detail level: a tile that small draws nothing
        Colmesh-1     the collision mesh ("Collision Mesh" collides with it): the simplified prisms
      collision-1     an empty node beside start01, as in the stock shapes

* The number ending a node name is its detail size in pixels. An underscore before it
  would read as a minus sign, hence "_L300".
* Z_UP, metres. x and y are relative to the tile origin, where the TSStatic stands;
  z is world z, since the TSStatic stands at z 0.
* A <material name> binds to the level Material whose mapTo is that name
  (buildings.MATERIALS); one <triangles> per material.
* Every vertex has POSITION (2 decimals), NORMAL, TEXCOORD set 0 and COLOR set 0 (RGBA, the
  per-building tint the materials multiply in with vertColor), all at offset 0, so one
  index per corner. Faces are counter-clockwise seen from outside.
* Wall UVs: u = distance along the wall / 3 m, v = height above the base / 3.2 m, so a
  texture repeat is one floor high.
* A tile's buildings are handled together, column-wise (numpy and vectorized shapely),
  since the map has a hundred thousand of them.
* The COLLADA text itself is written by collada.py (shared with the models); the tiles use
  its TILE decimals, which must never change: the shipped .cdae files only fit these bytes.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely.geometry import Polygon

from . import collada

FULL, LOW, NULL, COLLISION = "bld_L300", "bld_L60", "nulldetail15", "Colmesh-1"
LODS = (FULL, LOW, COLLISION)            # the levels that hold geometry

WALL_MATERIALS = {"res": "br_bld_wall_res", "brick": "br_bld_wall_brick",
                  "concrete": "br_bld_wall_concrete", "glass": "br_bld_wall_glass"}
ROOF_MATERIALS = {"hip": "br_bld_roof_shingle", "flat": "br_bld_roof_flat"}
COLLISION_MATERIAL = "br_bld_roof_flat"  # its groundType is what tyres feel on a roof

WALL_U = 3.0           # m of wall per texture repeat along it
WALL_V = 3.2           # m of wall per texture repeat up it (one floor)
ROOF_UV = 3.0          # m of roof per texture repeat
LOW_SIMPLIFY = 2.0     # m, footprint simplification for bld_L60 and the collision mesh
HIP_RISE_SLOPE = 0.35  # a hipped roof rises 0.35 x the short side of its rectangle ...
HIP_MAX_RISE = 3.0     # ... up to 3 m
MIN_WALL = 2.5         # m of wall a hipped roof always leaves standing
MIN_RISE = 0.5         # m; a hipped roof any flatter is drawn flat
HIP_VERTICES = 18      # two trapezoids, two triangles and the soffit
EPS = 1e-6
_POLYGON = 3           # shapely type id

# roof tints, picked per building from its id (the walls use Building.colour)
ROOF_TINTS = {"hip": ((0.95, 0.95, 0.95), (0.80, 0.74, 0.70), (0.70, 0.72, 0.76), (0.62, 0.56, 0.52)),
              "flat": ((0.92, 0.92, 0.90), (0.84, 0.84, 0.84), (0.76, 0.76, 0.78))}


def hip_rise(short_side, height):
    """Rise of a hipped roof over a rectangle with this short side, taken out of the walls
    (works on arrays too)."""
    return np.maximum(0.0, np.minimum(np.minimum(HIP_RISE_SLOPE * np.asarray(short_side), HIP_MAX_RISE),
                                      np.asarray(height) - MIN_WALL))


# ------------------------------------------------------------------ footprints as stacked rings

def _objects(items) -> np.ndarray:
    out = np.empty(len(items), dtype=object)
    out[:] = items
    return out


def _runs(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(start, length) of each run of equal values."""
    if not len(ids):
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    return starts, np.diff(np.r_[starts, len(ids)])


def _next_in_ring(sizes: np.ndarray) -> np.ndarray:
    """Index of the next point round its ring, for rings stacked in one array."""
    starts = np.r_[0, np.cumsum(sizes)[:-1]].astype(np.int64)
    nxt = np.arange(1, int(sizes.sum()) + 1)
    nxt[starts + sizes - 1] = starts
    return nxt


@dataclass
class _Rings:
    """Polygons with their rings' points stacked: exterior first, then the holes; not closed,
    no repeated points, the exterior counter-clockwise and the holes clockwise, so the solid
    is on the left of every ring and the right-hand normal points out."""
    polys: np.ndarray    # (P,) shapely Polygons
    points: np.ndarray   # (N, 2)
    sizes: np.ndarray    # (R,) points per ring
    part: np.ndarray     # (R,) the polygon each ring belongs to

    @property
    def point_part(self) -> np.ndarray:
        return np.repeat(self.part, self.sizes)

    def part_points(self) -> np.ndarray:
        return np.bincount(self.part, weights=self.sizes, minlength=len(self.polys)).astype(np.int64)


def _rings(polys: np.ndarray) -> _Rings:
    if not len(polys):
        return _Rings(polys, np.zeros((0, 2)), np.zeros(0, np.int64), np.zeros(0, np.int64))
    rings, part = shapely.get_rings(polys, return_index=True)
    exterior = np.r_[True, part[1:] != part[:-1]]
    xy, ring = shapely.get_coordinates(rings, return_index=True)
    is_open = np.r_[ring[1:] == ring[:-1], False]  # drops each ring's closing point
    xy, ring = xy[is_open], ring[is_open]
    step = xy[_next_in_ring(_runs(ring)[1])] - xy
    keep = step[:, 0] ** 2 + step[:, 1] ** 2 > EPS * EPS
    xy, ring = xy[keep], ring[keep]
    starts, sizes = _runs(ring)
    ok = np.repeat(sizes >= 3, sizes)
    xy, ring = xy[ok], ring[ok]
    starts, sizes = _runs(ring)
    ids = ring[starts]
    nxt = _next_in_ring(sizes)
    area2 = np.add.reduceat(xy[:, 0] * xy[nxt, 1] - xy[nxt, 0] * xy[:, 1], starts) if len(xy) else np.zeros(0)
    flip = np.repeat((area2 > 0) != exterior[ids], sizes)
    first, size = np.repeat(starts, sizes), np.repeat(sizes, sizes)
    j = np.arange(len(xy)) - first
    xy = xy[first + np.where(flip, size - 1 - j, j)]
    return _Rings(polys, xy, sizes, part[ids])


def _polygon_parts(geom) -> list:
    if geom.geom_type == "Polygon":
        return [geom]
    return [p for g in getattr(geom, "geoms", []) for p in _polygon_parts(g)]


@dataclass
class _Plan:
    """The levels of detail of a set of buildings, before any vertices are made."""
    full: _Rings          # every polygon of every building (a broken footprint repaired to its parts)
    low: _Rings           # the same simplified at LOW_SIMPLIFY, part for part
    owner: np.ndarray     # (P,) building of each polygon
    base: np.ndarray      # (B,) z of the wall bottoms
    eave: np.ndarray      # (B,) z of the wall tops in the full detail
    top: np.ndarray       # (B,) z of the roof tops
    low_top: np.ndarray   # (B,) z of the flat roofs of bld_L60 and the collision mesh
    hip: np.ndarray       # (B,) has a hipped roof
    corners: np.ndarray   # (H, 4, 2) rotated rectangles of the hipped roofs, counter-clockwise,
    long: np.ndarray      # (H,)     corner 0 -> 1 along a long side
    short: np.ndarray     # (H,)
    rise: np.ndarray      # (H,)
    wall: list[str]       # (B,) wall material
    colour: np.ndarray    # (B, 4) RGBA of the walls
    tint: np.ndarray      # (B, 4) RGBA of the roof

    def counts(self) -> np.ndarray:
        """(B, 3) vertices per building in bld_L300, bld_L60 and Colmesh-1."""
        b = len(self.base)
        full = np.bincount(self.owner, weights=self.full.part_points(), minlength=b).astype(np.int64)
        low = np.bincount(self.owner, weights=self.low.part_points(), minlength=b).astype(np.int64)
        # four corners per wall, then the roof: one vertex per corner when flat
        return np.column_stack([4 * full + np.where(self.hip, HIP_VERTICES, full), 5 * low, 2 * low])


def _plan(buildings) -> _Plan:
    polys, owner = [], []
    for i, b in enumerate(buildings):
        try:
            p = Polygon(np.asarray(b.ring, float), [np.asarray(h, float) for h in b.holes if len(h) >= 3])
        except (ValueError, shapely.errors.GEOSException):
            continue  # not a polygon at all
        polys.append(p)
        owner.append(i)
    polys, owner = _objects(polys), np.asarray(owner, np.int64)
    bad = ~shapely.is_valid(polys) if len(polys) else np.zeros(0, bool)
    if bad.any():  # repaired to their polygon parts
        fixed = [(q, o) for p, o, b_ in zip(polys, owner, bad)
                 for q in (_polygon_parts(shapely.make_valid(p)) if b_ else [p])]
        polys, owner = _objects([q for q, _ in fixed]), np.asarray([o for _, o in fixed], np.int64)
    area = shapely.area(polys) if len(polys) else np.zeros(0)
    polys, owner, area = polys[area > EPS], owner[area > EPS], area[area > EPS]

    # bld_L60 and the collision mesh: simplified, or the footprint as it is when that breaks
    # it or loses more than half its area (small buildings). Clipped to the footprint, since
    # a simplified edge cuts across notches: the collision mesh would stand where no wall
    # is drawn, out in a road. Never a bounding shape, for the same reason.
    low = shapely.simplify(polys, LOW_SIMPLIFY, preserve_topology=True) if len(polys) else polys
    if len(low):
        low = shapely.intersection(low, polys)
        broken = ((shapely.get_type_id(low) != _POLYGON) | ~shapely.is_valid(low)
                  | (shapely.area(low) < 0.5 * area))
        low[broken] = polys[broken]

    n = len(buildings)
    base = np.array([float(b.base_z) for b in buildings])
    height = np.array([float(b.height) for b in buildings])
    parts = np.bincount(owner, minlength=n)
    hip = np.array([b.roof == "hip" for b in buildings], bool) & (parts == 1)
    single = np.full(n, -1, np.int64)
    single[owner] = np.arange(len(owner))
    if hip.any():
        cand = polys[single[hip]]
        hip[hip] = shapely.get_num_interior_rings(cand) == 0
    corners, long, short, rise = np.zeros((0, 4, 2)), np.zeros(0), np.zeros(0), np.zeros(0)
    if hip.any():
        rect = shapely.minimum_rotated_rectangle(polys[single[hip]])
        ok = shapely.get_num_coordinates(rect) == 5
        hip[np.flatnonzero(hip)[~ok]] = False
        c = shapely.get_coordinates(rect[ok]).reshape(-1, 5, 2)[:, :4]
        cw = np.sum(c[:, :, 0] * np.roll(c[:, :, 1], -1, 1) - np.roll(c[:, :, 0], -1, 1) * c[:, :, 1], 1) < 0
        c[cw] = c[cw, ::-1]
        a, b_ = np.hypot(*(c[:, 1] - c[:, 0]).T), np.hypot(*(c[:, 2] - c[:, 1]).T)
        turn = a < b_  # make 0 -> 1 a long side
        c[turn] = np.roll(c[turn], -1, axis=1)
        long, short = np.maximum(a, b_), np.minimum(a, b_)
        rise = hip_rise(short, height[hip])
        flat = rise < MIN_RISE
        hip[np.flatnonzero(hip)[flat]] = False
        corners, long, short, rise = c[~flat], long[~flat], short[~flat], rise[~flat]
    hip_rise_b = np.zeros(n)
    hip_rise_b[hip] = rise
    top = base + height
    tints = [ROOF_TINTS["hip" if h else "flat"] for h in hip]
    return _Plan(
        full=_rings(polys), low=_rings(low), owner=owner, base=base, eave=top - hip_rise_b, top=top,
        low_top=top - hip_rise_b / 2, hip=hip, corners=corners, long=long, short=short, rise=rise,
        wall=[WALL_MATERIALS.get(b.wall, WALL_MATERIALS["concrete"]) for b in buildings],
        colour=np.array([[*(float(c) for c in b.colour[:3]), 1.0] for b in buildings]).reshape(-1, 4),
        tint=np.array([[*t[zlib.crc32(str(b.id).encode()) % len(t)], 1.0]
                       for b, t in zip(buildings, tints)]).reshape(-1, 4))


def vertex_counts_each(buildings) -> np.ndarray:
    """(B, 3) vertices each building gets in bld_L300, bld_L60 and Colmesh-1, without making them."""
    return _plan(buildings).counts()


def vertex_counts(buildings) -> dict[str, int]:
    return dict(zip(LODS, (int(v) for v in vertex_counts_each(buildings).sum(axis=0))))


# ------------------------------------------------------------------ meshes

@dataclass
class _Mesh:
    """Vertices (position, normal, uv, RGBA) and triangles with a material code each."""
    chunks: list = field(default_factory=list)
    codes: dict = field(default_factory=dict)  # material name -> code
    n: int = 0

    def code(self, material: str) -> int:
        return self.codes.setdefault(material, len(self.codes))

    def add(self, pos, nrm, uv, col, tris, mats) -> None:
        self.chunks.append((pos, nrm, uv, col, np.asarray(tris, np.int64).reshape(-1, 3) + self.n,
                            np.asarray(mats, np.int64)))
        self.n += len(pos)

    def to_mesh(self) -> collada.Mesh:
        """The vertices and the triangles of each material, in name order."""
        if not self.chunks:
            return collada.Mesh(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 2)), np.zeros((0, 4)), {})
        pos, nrm, uv, col, tris, mats = (np.concatenate(c) for c in zip(*self.chunks))
        return collada.Mesh(pos, nrm, uv, col, {m: tris[mats == c] for m, c in sorted(self.codes.items())
                                                if (mats == c).any()})


def _walls(mesh: _Mesh, r: _Rings, origin, z0, z1, codes, colours) -> None:
    """Four vertices per wall, normals out, u running on around each ring; z0, z1, codes and
    colours are per polygon."""
    if not len(r.points):
        return
    ring = np.repeat(np.arange(len(r.sizes)), r.sizes)
    owner = r.part[ring]
    a = r.points - origin
    b = a[_next_in_ring(r.sizes)]
    d = b - a
    length = np.hypot(d[:, 0], d[:, 1])
    walked = np.cumsum(length) - length
    starts = np.r_[0, np.cumsum(r.sizes)[:-1]].astype(np.int64)
    u0 = (walked - walked[starts][ring]) / WALL_U
    u1 = u0 + length / WALL_U
    lo, hi = np.asarray(z0, float)[owner], np.asarray(z1, float)[owner]
    m = len(a)
    pos = np.empty((m, 4, 3))
    pos[:, 0, :2], pos[:, 1, :2], pos[:, 2, :2], pos[:, 3, :2] = a, b, b, a
    pos[:, 0, 2] = pos[:, 1, 2] = lo
    pos[:, 2, 2] = pos[:, 3, 2] = hi
    nrm = np.column_stack([d[:, 1] / length, -d[:, 0] / length, np.zeros(m)])
    uv = np.empty((m, 4, 2))
    uv[:, 0, 0] = uv[:, 3, 0] = u0
    uv[:, 1, 0] = uv[:, 2, 0] = u1
    uv[:, 0, 1] = uv[:, 1, 1] = 0.0
    uv[:, 2, 1] = uv[:, 3, 1] = (hi - lo) / WALL_V
    tris = (np.arange(m)[:, None, None] * 4 + np.array([[0, 1, 2], [0, 2, 3]])).reshape(-1, 3)
    mesh.add(pos.reshape(-1, 3), np.repeat(nrm, 4, axis=0), uv.reshape(-1, 2),
             np.repeat(np.asarray(colours, float)[owner], 4, axis=0), tris,
             np.repeat(np.asarray(codes, np.int64)[owner], 2))


def _caps(r: _Rings) -> np.ndarray:
    """(T, 3) indices into r.points triangulating the polygons, counter-clockwise from above.
    The constrained Delaunay triangulation uses only the ring points; a corner not found
    among them (it shouldn't happen) snaps to its polygon's nearest point."""
    if not len(r.points):
        return np.zeros((0, 3), np.int64)
    owner = r.point_part
    coords, which = shapely.get_coordinates(shapely.constrained_delaunay_triangles(r.polys), return_index=True)
    corners = coords.reshape(-1, 4, 2)[:, :3].reshape(-1, 2)
    corner_owner = np.repeat(which.reshape(-1, 4)[:, 0], 3)
    index = {k: i for i, k in enumerate(zip(owner.tolist(), r.points[:, 0].tolist(), r.points[:, 1].tolist()))}
    idx = np.array([index.get(k, -1) for k in zip(corner_owner.tolist(), corners[:, 0].tolist(),
                                                   corners[:, 1].tolist())], dtype=np.int64)
    for k in np.flatnonzero(idx < 0):
        mine = np.flatnonzero(owner == corner_owner[k])
        idx[k] = mine[np.argmin(np.hypot(*(r.points[mine] - corners[k]).T))]
    idx = idx.reshape(-1, 3)
    a, b, c = r.points[idx[:, 0]], r.points[idx[:, 1]], r.points[idx[:, 2]]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    idx[cross < 0] = idx[cross < 0][:, ::-1]
    return idx[np.abs(cross) > EPS]


def _flat_roofs(mesh: _Mesh, r: _Rings, tris: np.ndarray, origin, z, code: int, colours) -> None:
    """Flat roofs on the polygons; z and colours are per polygon."""
    if not len(r.points):
        return
    owner = r.point_part
    local = r.points - origin
    mesh.add(np.column_stack([local, np.asarray(z, float)[owner]]), np.tile([0.0, 0.0, 1.0], (len(local), 1)),
             local / ROOF_UV, np.asarray(colours, float)[owner], tris, np.full(len(tris), code))


def _prisms(mesh: _Mesh, r: _Rings, tris: np.ndarray, origin, z0, z1, code: int, colours) -> None:
    """Closed-top prisms sharing their corner vertices (the collision mesh); z0, z1 and
    colours are per polygon."""
    if not len(r.points):
        return
    owner = r.point_part
    local = r.points - origin
    k = len(local)
    i, j = np.arange(k), _next_in_ring(r.sizes)
    pos = np.vstack([np.column_stack([local, np.asarray(z0, float)[owner]]),
                     np.column_stack([local, np.asarray(z1, float)[owner]])])
    faces = np.vstack([np.column_stack([i, j, j + k]), np.column_stack([i, j + k, i + k]), tris + k])
    col = np.asarray(colours, float)[owner]
    mesh.add(pos, np.tile([0.0, 0.0, 1.0], (2 * k, 1)), np.zeros((2 * k, 2)), np.vstack([col, col]),
             faces, np.full(len(faces), code))


# hipped roof: 18 vertices per building, faces counter-clockwise from outside
_HIP_TRIS = np.array([[0, 1, 2], [0, 2, 3],        # long side c0 c1 r1 r0
                      [4, 5, 6],                   # end c1 c2 r1
                      [7, 8, 9], [7, 9, 10],       # long side c2 c3 r0 r1
                      [11, 12, 13],                # end c3 c0 r0
                      [14, 15, 16], [14, 16, 17]])  # soffit c0 c3 c2 c1, facing down


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _hip_roofs(mesh: _Mesh, plan: _Plan, origin, code: int) -> None:
    """Hipped roofs on the footprints' rotated rectangles: equal slopes all round, the ridge
    along the long axis (a point on a square). Their eaves overhang the walls wherever a
    footprint isn't its rectangle, so a soffit closes them from below."""
    h = len(plan.corners)
    if not h:
        return
    eave = plan.eave[plan.hip]
    c = np.concatenate([plan.corners - origin, np.repeat(eave[:, None, None], 4, axis=1)], axis=2)
    half_ridge = (plan.long - plan.short)[:, None] / 2
    axis = (c[:, 1] - c[:, 0]) / plan.long[:, None]
    mid = c.mean(axis=1)
    mid[:, 2] += plan.rise
    r0, r1 = mid - axis * half_ridge, mid + axis * half_ridge
    c0, c1, c2, c3 = c[:, 0], c[:, 1], c[:, 2], c[:, 3]
    pos, nrm, uv = [], [], []
    for face in ((c0, c1, r1, r0), (c1, c2, r1), (c2, c3, r0, r1), (c3, c0, r0)):
        p = np.stack(face, axis=1)
        n = _unit(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]))
        e = _unit(p[:, 1] - p[:, 0])
        w = np.cross(n, e)
        w[w[:, 2] < 0] *= -1  # v runs up the slope
        rel = p - p[:, :1]
        pos.append(p)
        nrm.append(np.repeat(n[:, None], len(face), axis=1))
        uv.append(np.stack([np.einsum("hki,hi->hk", rel, e), np.einsum("hki,hi->hk", rel, w)], axis=2) / ROOF_UV)
    soffit = c[:, [0, 3, 2, 1]]
    pos.append(soffit)
    nrm.append(np.tile([0.0, 0.0, -1.0], (h, 4, 1)))
    uv.append(soffit[:, :, :2] / ROOF_UV)
    pos = np.concatenate(pos, axis=1).reshape(-1, 3)
    tris = (np.arange(h)[:, None, None] * HIP_VERTICES + _HIP_TRIS).reshape(-1, 3)
    a, b, cc = pos[tris[:, 0]], pos[tris[:, 1]], pos[tris[:, 2]]
    tris = tris[np.linalg.norm(np.cross(b - a, cc - a), axis=1) > EPS]  # the ridge of a square
    mesh.add(pos, np.concatenate(nrm, axis=1).reshape(-1, 3), np.concatenate(uv, axis=1).reshape(-1, 2),
             np.repeat(plan.tint[plan.hip], HIP_VERTICES, axis=0), tris, np.full(len(tris), code))


def _meshes(plan: _Plan, origin) -> dict[str, _Mesh]:
    o = np.asarray(origin, dtype=float)
    meshes = {lod: _Mesh() for lod in LODS}
    full, low, col = meshes[FULL], meshes[LOW], meshes[COLLISION]
    p = plan.owner  # building of each polygon

    # full detail: walls up to the eaves, then flat or hipped roofs
    walls = np.array([full.code(w) for w in plan.wall], np.int64)
    _walls(full, plan.full, o, plan.base[p], plan.eave[p], walls[p], plan.colour[p])
    flat = ~plan.hip[p]
    roofs = _rings(plan.full.polys[flat])
    _flat_roofs(full, roofs, _caps(roofs), o, plan.top[p][flat], full.code(ROOF_MATERIALS["flat"]),
                plan.tint[p][flat])
    if plan.hip.any():
        _hip_roofs(full, plan, o, full.code(ROOF_MATERIALS["hip"]))

    # far detail and collision: the simplified prisms with flat roofs, sharing one triangulation
    walls = np.array([low.code(w) for w in plan.wall], np.int64)
    _walls(low, plan.low, o, plan.base[p], plan.low_top[p], walls[p], plan.colour[p])
    tris = _caps(plan.low)
    _flat_roofs(low, plan.low, tris, o, plan.low_top[p], low.code(ROOF_MATERIALS["flat"]), plan.tint[p])
    _prisms(col, plan.low, tris, o, plan.base[p], plan.low_top[p], col.code(COLLISION_MATERIAL),
            plan.colour[p])
    return meshes


# ------------------------------------------------------------------ the tile's shape

def tile_dae(buildings, origin_xy) -> tuple[str, dict[str, int]]:
    """The .dae text of one tile and its vertex count per level of detail ({bld_L300, bld_L60,
    Colmesh-1}). x and y are written relative to origin_xy."""
    meshes = {lod: m.to_mesh() for lod, m in _meshes(_plan(buildings), origin_xy).items()}
    xml = collada.shape_dae([(FULL, meshes[FULL]), (LOW, meshes[LOW]), (NULL, None),
                             (COLLISION, meshes[COLLISION])], decimals=collada.TILE)
    return xml, {lod: collada.vertex_count(meshes[lod]) for lod in LODS}
