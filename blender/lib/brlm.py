"""Helpers for the landmark model scripts: flat-shaded faces with their uv in metres, collected
in a Builder and turned into one Blender object per level of detail.

A script builds each level of detail into its own Builder and hands them to lod_nodes, which
names the nodes `<shape>_a<px>` from a table, checks the triangle budgets and makes the objects
for brexport.export:

    nodes = brlm.lod_nodes("tower", [(400, 6000, high), (100, 800, mid)], collision=(60, col))
    brexport.export({"main": nodes}, out)

Coordinates are the model's: x along +u, y along +v, z up, in metres. Every face is planar and
convex, given counter-clockwise as seen from the side it faces; its own corners (no shared
vertices) keep the shading flat. The uv of a tiled material is the face's position in metres
divided by `tile`, laid out by planar_uv: u along the face's horizontal (left to right as seen
from the front), v up the face (up a wall, up a roof's slope). A face may carry a vertex colour
(Builder.cols), which Builder.object writes as brexport's "Col" byte colour.

Bowls are built from rings: ring_at samples an outline along rays from a centre, loft joins
rings of the same length with quads, step_rings and seat_steps make rows of seats between a
front and a back ring. lattice_tower and beam make steel work; blind_arches a wall with
recessed round arches. triangulate and fill cover any simple polygon (a field, a roof, a pool),
offset_ring and offset_line move outlines sideways, wall_along stands a wall on a line.

Only Builder.object and lod_nodes need Blender; the geometry is plain Python, so the tests can
check it without Blender.
"""
from __future__ import annotations

import math

Vec = tuple[float, float, float]
SNAP = 0.5   # m: footprint coordinates closer than this are one grid line in split_rectangles
WHITE = (1.0, 1.0, 1.0, 1.0)
PLANAR_TOL = 0.02   # m: a lofted quad whose fourth corner lies further off the plane of the other
#                     three is split into two triangles
TINY = 1e-6         # m2: a lofted face with less area is left out


# ------------------------------------------------------------------ vectors

def add(a, b) -> Vec:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a, b) -> Vec:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def mul(a, k: float) -> Vec:
    return (a[0] * k, a[1] * k, a[2] * k)


def dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b) -> Vec:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def unit(a) -> Vec:
    n = math.sqrt(dot(a, a))
    if n < 1e-12:
        raise ValueError("a zero-length vector has no direction")
    return mul(a, 1.0 / n)


def normal(pts) -> Vec:
    """The unit normal of a planar polygon given counter-clockwise (Newell's method)."""
    n = [0.0, 0.0, 0.0]
    for i, p in enumerate(pts):
        q = pts[(i + 1) % len(pts)]
        n[0] += (p[1] - q[1]) * (p[2] + q[2])
        n[1] += (p[2] - q[2]) * (p[0] + q[0])
        n[2] += (p[0] - q[0]) * (p[1] + q[1])
    return unit(n)


def planar_uv(pts, tile: float = 1.0) -> list[tuple[float, float]]:
    """The uv of each corner in metres / tile: u along the face's horizontal, v up the face."""
    n = normal(pts)
    right = cross((0.0, 0.0, 1.0), n)
    if dot(right, right) < 1e-9:              # a level face: u east, v north (south from below)
        right = (1.0, 0.0, 0.0)
    right = unit(right)
    up = cross(n, right)
    return [(dot(p, right) / tile, dot(p, up) / tile) for p in pts]


# ------------------------------------------------------------------ the builder

class Builder:
    """The faces of one mesh: (corners, uv per corner, material name) each, and beside them in
    `cols` each face's vertex colour, (r, g, b, a) in 0..1, or None for white."""

    def __init__(self):
        self.faces: list[tuple[list[Vec], list[tuple[float, float]], str]] = []
        self.cols: list[tuple[float, float, float, float] | None] = []

    @property
    def tris(self) -> int:
        return sum(len(pts) - 2 for pts, _, _ in self.faces)

    @property
    def corners(self) -> int:
        """The face corners: the most vertices the exported mesh can have (the export welds
        corners with the same position, normal, uv and colour into one)."""
        return sum(len(pts) for pts, _, _ in self.faces)

    @property
    def materials(self) -> set[str]:
        return {m for _, _, m in self.faces}

    def face(self, pts, mat: str, tile: float = 1.0, uv=None, double: bool = False, col=None) -> None:
        """One planar convex face, counter-clockwise from the side it faces; uv in metres / tile
        unless given; col its vertex colour, (r, g, b) or (r, g, b, a) in 0..1 (white when None).
        double adds the same face turned round, seen from behind."""
        pts = [tuple(float(c) for c in p) for p in pts]
        if len(pts) < 3:
            raise ValueError("a face needs three corners or more")
        if col is not None:
            col = tuple(float(c) for c in col)
            col += (1.0,) * (4 - len(col))
            if len(col) != 4 or not all(0.0 <= c <= 1.0 for c in col):
                raise ValueError(f"a face colour is (r, g, b[, a]) in 0..1, not {col}")
        uv = list(uv) if uv is not None else planar_uv(pts, tile)
        self.faces.append((pts, uv, mat))
        self.cols.append(col)
        if double:
            self.faces.append((pts[::-1], uv[::-1], mat))
            self.cols.append(col)

    def extend(self, other: "Builder") -> "Builder":
        self.faces += other.faces
        self.cols += other.cols
        return self

    def object(self, name: str):
        """A Blender mesh object of these faces, flat shaded, one material slot per material
        (brexport.material), uv in layer 0, linked to the scene."""
        import bpy        # only inside Blender
        import brexport

        if not self.faces:
            raise ValueError(f"{name}: no faces")
        names = sorted(self.materials)
        slot = {m: i for i, m in enumerate(names)}
        verts, polys, uvs, index = [], [], [], []
        for pts, uv, mat in self.faces:
            polys.append(list(range(len(verts), len(verts) + len(pts))))
            verts += pts
            uvs += uv
            index.append(slot[mat])
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(verts, [], polys)
        for m in names:
            mesh.materials.append(brexport.material(m))
        mesh.polygons.foreach_set("material_index", index)
        layer = mesh.uv_layers.new(name="UVMap")
        layer.data.foreach_set("uv", [c for p in uvs for c in p])
        if any(c is not None for c in self.cols):
            colour = mesh.color_attributes.new(brexport.COLOR_ATTRIBUTE, "BYTE_COLOR", "CORNER")
            colour.data.foreach_set("color_srgb", [v for (pts, _, _), c in zip(self.faces, self.cols)
                                                   for _ in pts for v in (c or WHITE)])
        mesh.shade_flat()
        mesh.update()
        ob = bpy.data.objects.new(name, mesh)
        bpy.context.scene.collection.objects.link(ob)
        return ob


# ------------------------------------------------------------------ shapes

BOX_FACES = ("-x", "+x", "-y", "+y", "-z", "+z")


def box(b: Builder, lo, hi, mat: str, tile: float = 1.0, skip=(), mats=None) -> Builder:
    """An axis-aligned box from corner lo to corner hi, faces outwards. skip leaves out faces by
    name ("-x", "+x", "-y", "+y", "-z", "+z"); mats gives some faces another material."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    faces = {"-x": [(x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1)],
             "+x": [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
             "-y": [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
             "+y": [(x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1)],
             "-z": [(x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0)],
             "+z": [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]}
    unknown = set(skip) - set(faces)
    if unknown:
        raise ValueError(f"box: no face called {', '.join(sorted(unknown))}")
    for key in BOX_FACES:
        if key not in skip:
            b.face(faces[key], (mats or {}).get(key, mat), tile)
    return b


def extrude_ring(b: Builder, ring, z0: float, z1: float, mat: str, tile: float = 1.0,
                 top: str | None = None) -> Builder:
    """Walls round a counter-clockwise ring [(x, y), ...] from z0 to z1, facing out; with top,
    a lid in that material (the ring must then be convex)."""
    n = len(ring)
    for i in range(n):
        (xa, ya), (xb, yb) = ring[i], ring[(i + 1) % n]
        b.face([(xa, ya, z0), (xb, yb, z0), (xb, yb, z1), (xa, ya, z1)], mat, tile)
    if top:
        b.face([(x, y, z1) for x, y in ring], top, tile)
    return b


def pyramid_roof(b: Builder, x0, y0, x1, y1, z: float, apex_z: float, mat: str, tile: float = 1.0) -> Builder:
    """A pyramid on the rectangle [x0, x1] x [y0, y1] at z, its apex over the centre."""
    apex = ((x0 + x1) / 2, (y0 + y1) / 2, apex_z)
    corners = [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]
    for i in range(4):
        b.face([corners[i], corners[(i + 1) % 4], apex], mat, tile)
    return b


def hipped_roof(b: Builder, x0, y0, x1, y1, z: float, pitch_deg: float, mat: str, tile: float = 1.0,
                overhang: float = 0.0, soffit: str | None = None) -> Builder:
    """A hipped roof on the rectangle [x0, x1] x [y0, y1] at eave height z, every slope at
    pitch_deg, the ridge along the longer side, the eaves overhang past the rectangle; with
    soffit, the underside of the eaves in that material (one face over the whole rectangle)."""
    x0, y0, x1, y1 = x0 - overhang, y0 - overhang, x1 + overhang, y1 + overhang
    half = min(x1 - x0, y1 - y0) / 2
    top = z + half * math.tan(math.radians(pitch_deg))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if abs((x1 - x0) - (y1 - y0)) < 1e-6:                               # square: no ridge
        pyramid_roof(b, x0, y0, x1, y1, z, top, mat, tile)
    elif x1 - x0 > y1 - y0:
        r0, r1 = (x0 + half, cy, top), (x1 - half, cy, top)
        b.face([(x0, y0, z), (x1, y0, z), r1, r0], mat, tile)            # south slope
        b.face([(x1, y1, z), (x0, y1, z), r0, r1], mat, tile)            # north slope
        b.face([(x1, y0, z), (x1, y1, z), r1], mat, tile)                # east hip
        b.face([(x0, y1, z), (x0, y0, z), r0], mat, tile)                # west hip
    else:
        r0, r1 = (cx, y0 + half, top), (cx, y1 - half, top)
        b.face([(x1, y0, z), (x1, y1, z), r1, r0], mat, tile)
        b.face([(x0, y1, z), (x0, y0, z), r0, r1], mat, tile)
        b.face([(x0, y0, z), (x1, y0, z), r0], mat, tile)
        b.face([(x1, y1, z), (x0, y1, z), r1], mat, tile)
    if soffit:
        b.face([(x0, y0, z), (x0, y1, z), (x1, y1, z), (x1, y0, z)], soffit, tile)
    return b


def prism(b: Builder, centre, radius: float, z0: float, z1: float, sides: int, mat: str,
          tip: bool = False, tile: float = 1.0) -> Builder:
    """A regular prism round a vertical axis (no bottom); with tip, a pyramid to a point at z1."""
    cx, cy = centre
    ring = [(cx + radius * math.cos(2 * math.pi * k / sides), cy + radius * math.sin(2 * math.pi * k / sides))
            for k in range(sides)]
    for k in range(sides):
        (xa, ya), (xb, yb) = ring[k], ring[(k + 1) % sides]
        if tip:
            b.face([(xa, ya, z0), (xb, yb, z0), (cx, cy, z1)], mat, tile)
        else:
            b.face([(xa, ya, z0), (xb, yb, z0), (xb, yb, z1), (xa, ya, z1)], mat, tile)
    if not tip:
        b.face([(x, y, z1) for x, y in ring], mat, tile)
    return b


def decal(b: Builder, centre, right, w: float, h: float, mat: str) -> Builder:
    """A w x h vertical quad centred at `centre`, facing the side from which `right` points
    right, with the whole texture on it (v up)."""
    r = unit(right)
    up = (0.0, 0.0, 1.0)
    pts = [add(add(centre, mul(r, sw * w / 2)), mul(up, sh * h / 2)) for sw, sh in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    b.face(pts, mat, uv=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    return b


# ------------------------------------------------------------------ the arch cut

def arch_outline(centre: float, width: float, sill: float, spring: float, segments: int):
    """The opening's outline in wall coordinates (s, z), counter-clockwise from the bottom left:
    the sill's two ends, then the round arch from the top of the right jamb over to the top of
    the left jamb in `segments` pieces (the jambs are the edges into and out of it)."""
    a, c, r = centre - width / 2, centre + width / 2, width / 2
    arch = [(centre + r * math.cos(math.pi * k / segments), spring + r * math.sin(math.pi * k / segments))
            for k in range(segments + 1)]
    return [(a, sill), (c, sill)] + arch


def arched_wall(b: Builder, origin, right, length: float, z0: float, z1: float, openings, thickness: float,
                mat: str, tile: float = 1.0, segments: int = 8, double_inner: bool = False,
                inner_mat: str | None = None) -> Builder:
    """A wall `thickness` thick with round-arched openings cut through it.

    The outer face lies in the vertical plane through `origin` (z = 0 there), from s = 0 to
    `length` along `right`, from z0 to z1; it faces the side from which `right` points right,
    and the inner face lies `thickness` behind it. openings are (centre s, width, sill z,
    spring z), left to right; each arch rises width / 2 above its spring and must stay under
    z1. The reveals line each opening (sill, jambs, the arch's underside). double_inner makes
    the inner face and the reveals double-sided."""
    at, back = _wall_frame(origin, right, thickness)
    inner_mat = inner_mat or mat
    for panel in _arch_panels("arched_wall", length, z0, z1, openings, segments):
        b.face([at(s, z) for s, z in panel], mat, tile)
        b.face([at(s, z, back) for s, z in panel[::-1]], inner_mat, tile, double=double_inner)
    for centre, width, sill, spring in openings:
        outline = arch_outline(centre, width, sill, spring, segments)
        for (sa, za), (sb, zb) in zip(outline, outline[1:] + outline[:1]):
            b.face([at(sa, za), at(sb, zb), at(sb, zb, back), at(sa, za, back)], inner_mat, tile,
                   double=double_inner)
    return b


def _wall_frame(origin, right, thickness: float):
    """(at(s, z, depth=0) -> the point s along `right` and z up from origin, `depth` behind the
    face; the offset `thickness` behind the face) for a vertical wall facing right x up."""
    r = unit((right[0], right[1], 0.0))
    out = cross(r, (0.0, 0.0, 1.0))              # the outward normal: right x up
    back = mul(out, -thickness)

    def at(s: float, z: float, depth=(0.0, 0.0, 0.0)) -> Vec:
        return add(add(add(origin, mul(r, s)), (0.0, 0.0, z)), depth)

    return at, back


def _arch_panels(who: str, length: float, z0: float, z1: float, openings, segments: int):
    """The wall round the openings as (s, z) quads, counter-clockwise from the front."""
    panels = []
    left = 0.0
    for centre, width, sill, spring in openings:
        a, c = centre - width / 2, centre + width / 2
        if a < left - 1e-6 or c > length + 1e-6 or spring + width / 2 > z1 + 1e-6 or sill < z0 - 1e-6:
            raise ValueError(f"{who}: the opening at s = {centre} doesn't fit the wall")
        if a > left:
            panels.append([(left, z0), (a, z0), (a, z1), (left, z1)])               # pier
        if sill > z0:
            panels.append([(a, z0), (c, z0), (c, sill), (a, sill)])                 # under the sill
        arch = arch_outline(centre, width, sill, spring, segments)[2:][::-1]        # left to right
        for (sa, za), (sb, zb) in zip(arch, arch[1:]):
            panels.append([(sa, za), (sb, zb), (sb, z1), (sa, z1)])                 # spandrel
        left = c
    if left < length:
        panels.append([(left, z0), (length, z0), (length, z1), (left, z1)])
    return panels


def blind_arches(b: Builder, origin, right, length: float, z0: float, z1: float, openings, depth: float,
                 mat: str, tile: float = 1.0, segments: int = 8, recess=None) -> Builder:
    """A wall face with round arches recessed `depth` into it: the face, as arched_wall's outer
    face (the same origin, right, length, z0, z1 and openings), the reveals round each arch and
    a back panel closing it in the vertex colour `recess`. Nothing faces inwards, so it is a
    facade over a closed building, not a wall to see through."""
    at, back = _wall_frame(origin, right, depth)
    for panel in _arch_panels("blind_arches", length, z0, z1, openings, segments):
        b.face([at(s, z) for s, z in panel], mat, tile)
    for centre, width, sill, spring in openings:
        outline = arch_outline(centre, width, sill, spring, segments)
        for (sa, za), (sb, zb) in zip(outline, outline[1:] + outline[:1]):
            b.face([at(sa, za), at(sb, zb), at(sb, zb, back), at(sa, za, back)], mat, tile)
        b.face([at(s, z, back) for s, z in outline], mat, tile, col=recess)
    return b


# ------------------------------------------------------------------ rings, lofts and seats

def centroid(ring) -> tuple[float, float]:
    """The area centroid of a ring [(x, y), ...]."""
    a = cx = cy = 0.0
    for (x0, y0), (x1, y1) in zip(ring, list(ring[1:]) + list(ring[:1])):
        w = x0 * y1 - x1 * y0
        a += w
        cx += (x0 + x1) * w
        cy += (y0 + y1) * w
    if abs(a) < 1e-12:
        raise ValueError("a ring with no area has no centroid")
    return cx / (3 * a), cy / (3 * a)


def angle_of(centre, p) -> float:
    """The direction from centre to p, degrees counter-clockwise from +x, in [0, 360)."""
    return math.degrees(math.atan2(p[1] - centre[1], p[0] - centre[0])) % 360.0


def ring_at(ring, centre, angle: float) -> tuple[float, float]:
    """Where the ray from centre at `angle` (degrees counter-clockwise from +x) leaves the ring
    [(x, y), ...]: its farthest crossing (the only one when the ring is star-shaped round centre)."""
    cx, cy = centre
    dx, dy = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    best = None
    for (xa, ya), (xb, yb) in zip(ring, list(ring[1:]) + list(ring[:1])):
        ex, ey = xb - xa, yb - ya
        den = dx * ey - dy * ex
        if abs(den) < 1e-12:
            continue
        t = ((xa - cx) * ey - (ya - cy) * ex) / den           # along the ray
        s = ((xa - cx) * dy - (ya - cy) * dx) / den           # along the edge
        if t > 1e-9 and -1e-9 <= s <= 1 + 1e-9 and (best is None or t > best):
            best = t
    if best is None:
        raise ValueError(f"the ray from {centre} at {angle} deg meets no edge: is the centre outside the ring?")
    return cx + best * dx, cy + best * dy


def _area(pts) -> float:
    n = [0.0, 0.0, 0.0]
    for i, p in enumerate(pts):
        q = pts[(i + 1) % len(pts)]
        n[0] += (p[1] - q[1]) * (p[2] + q[2])
        n[1] += (p[2] - q[2]) * (p[0] + q[0])
        n[2] += (p[0] - q[0]) * (p[1] + q[1])
    return math.sqrt(dot(n, n)) / 2


def quad(b: Builder, pts, mat: str, tile: float = 1.0, col=None, tol: float = PLANAR_TOL) -> Builder:
    """Four corners as one face when they are planar (the fourth within tol of the plane of the
    other three), else as two triangles; faces with no area (repeated corners) are left out."""
    pts = [tuple(float(c) for c in p) for p in pts]
    distinct = [p for i, p in enumerate(pts) if math.dist(p, pts[i - 1]) > 1e-9]
    if len(distinct) < 3 or _area(distinct) < TINY:
        return b
    if len(distinct) == 3:
        b.face(distinct, mat, tile, col=col)
        return b
    a, c, d, e = distinct
    if _area([a, c, d]) >= TINY and abs(dot(normal([a, c, d]), sub(e, a))) <= tol:
        b.face(distinct, mat, tile, col=col)
        return b
    for tri in ([a, c, d], [a, d, e]):
        if _area(tri) >= TINY:
            b.face(tri, mat, tile, col=col)
    return b


def loft(b: Builder, rings, mat: str, tile: float = 1.0, closed: bool = False, flip: bool = False,
         col=None) -> Builder:
    """Quads joining each ring [(x, y, z), ...] to the next (all the same length): between
    corners k and k + 1 (and the last and the first when closed). A quad faces the way of
    (next ring - this ring) x (corner k + 1 - corner k): up for rings running counter-clockwise
    from an inner ring to an outer one, inwards for a ring under one above it; flip turns them
    all round. Made by quad(), so a twisted quad is two triangles."""
    for lower, upper in zip(rings, rings[1:]):
        if len(lower) != len(upper):
            raise ValueError(f"loft: rings of {len(lower)} and {len(upper)} corners")
        n = len(lower)
        for k in range(n if closed else n - 1):
            k1 = (k + 1) % n
            pts = [lower[k], upper[k], upper[k1], lower[k1]]
            quad(b, pts[::-1] if flip else pts, mat, tile, col)
    return b


def lerp(p, q, t: float) -> Vec:
    return tuple(a + (c - a) * t for a, c in zip(p, q))


def step_rings(front, back, rows: int) -> list[list[Vec]]:
    """The rings of `rows` seat steps from the ring front [(x, y, z), ...] up to back (as long):
    front, then for each row its tread (level, to the next row's line) and its riser (upright,
    to the next row's height), ending at back. Each corner's rows are even in plan and height."""
    if rows < 1:
        raise ValueError("step_rings: a row or more")
    out = [list(front)]
    for j in range(rows):
        t0, t1 = j / rows, (j + 1) / rows
        out.append([(*lerp(f, k, t1)[:2], lerp(f, k, t0)[2]) for f, k in zip(front, back)])   # the tread
        out.append([lerp(f, k, t1) for f, k in zip(front, back)])                           # the riser
    out[-1] = list(back)
    return out


def seat_steps(b: Builder, front, back, rows: int, mat: str, tile: float = 1.0, tread=None, riser=None,
               closed: bool = False) -> Builder:
    """`rows` seat steps lofted from the ring front up to back, both running counter-clockwise
    round the field (the field on their left), so the treads face up and the risers the field;
    tread and riser are their vertex colours."""
    rings = step_rings(front, back, rows)
    for i, pair in enumerate(zip(rings, rings[1:])):
        loft(b, list(pair), mat, tile, closed=closed, col=riser if i % 2 else tread)
    return b


# ------------------------------------------------------------------ steel

def beam(b: Builder, p, q, width: float, mat: str, tile: float = 1.0) -> Builder:
    """A square tube `width` across from p to q: its four sides, facing out (no ends)."""
    d = unit(sub(q, p))
    side = cross(d, (0.0, 0.0, 1.0))
    s = unit(side) if dot(side, side) > 1e-12 else (1.0, 0.0, 0.0)
    t = cross(d, s)
    h = width / 2
    ring = [add(mul(s, h), mul(t, h)), add(mul(s, -h), mul(t, h)), add(mul(s, -h), mul(t, -h)),
            add(mul(s, h), mul(t, -h))]
    for oa, ob in zip(ring, ring[1:] + ring[:1]):
        b.face([add(p, oa), add(p, ob), add(q, ob), add(q, oa)], mat, tile)
    return b


def lattice_tower(b: Builder, centre, width: float, z0: float, z1: float, mat: str, panels: int = 6,
                  top_width: float | None = None, leg: float = 0.3, brace: float = 0.12, braces: bool = True,
                  tile: float = 1.0) -> Builder:
    """A square lattice tower on centre (x, y) from z0 to z1, `width` across its legs at the
    bottom and top_width (default width) at the top: four corner legs `leg` thick, a ring of
    struts at the top of each of its `panels` panels and, with braces, an X of braces `brace`
    thick in each panel's four faces."""
    if panels < 1:
        raise ValueError("lattice_tower: a panel or more")
    top_width = width if top_width is None else top_width
    cx, cy = centre

    def corners(z: float) -> list[Vec]:
        h = (width + (top_width - width) * (z - z0) / (z1 - z0)) / 2
        return [(cx + h, cy + h, z), (cx - h, cy + h, z), (cx - h, cy - h, z), (cx + h, cy - h, z)]

    bottom, top = corners(z0), corners(z1)
    for p, q in zip(bottom, top):
        beam(b, p, q, leg, mat, tile)
    for i in range(panels):
        lo, hi = corners(z0 + (z1 - z0) * i / panels), corners(z0 + (z1 - z0) * (i + 1) / panels)
        for k in range(4):
            k1 = (k + 1) % 4
            beam(b, hi[k], hi[k1], brace, mat, tile)
            if braces:
                beam(b, lo[k], hi[k1], brace, mat, tile)
                beam(b, lo[k1], hi[k], brace, mat, tile)
    return b


def oriented_box(b: Builder, centre, axes, half, mat: str, tile: float = 1.0) -> Builder:
    """A box round centre along the unit axes (a1, a2, a3), right-handed (a1 x a2 = a3), half[i]
    from the centre along axes[i]; its six faces face out."""
    if dot(cross(axes[0], axes[1]), axes[2]) < 0.99:
        raise ValueError("oriented_box: the axes must be unit, square to each other and right-handed")

    def corner(signs) -> Vec:
        p = centre
        for s, a, h in zip(signs, axes, half):
            p = add(p, mul(a, s * h))
        return p

    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        for side in (1, -1):
            pts = []
            for sj, sk in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                signs = [0, 0, 0]
                signs[i], signs[j], signs[k] = side, sj, sk
                pts.append(corner(signs))
            b.face(pts if side > 0 else pts[::-1], mat, tile)
    return b


def light_tower(b: Builder, centre, width: float, z0: float, z1: float, facing, mat: str, lamp_mat: str,
                top_width: float | None = None, bank=(8.0, 4.0, 0.6), lamps=(4, 6), tilt_deg: float = 20.0,
                panels: int = 6, braces: bool = True, lattice: bool = True, tile: float = 1.0) -> Builder:
    """A floodlight tower: a lattice_tower (or, without lattice, a plain square mast) on centre
    from z0 to z1, and on top of it a lamp bank (w, h, d) facing the horizontal direction
    `facing`, tilted tilt_deg down; lamps (rows, columns) puts lamp faces on its front in
    lamp_mat, a little proud of it."""
    if lattice:
        lattice_tower(b, centre, width, z0, z1, mat, panels, top_width, braces=braces, tile=tile)
    else:
        half = (top_width or width) / 2
        box(b, (centre[0] - half, centre[1] - half, z0), (centre[0] + half, centre[1] + half, z1), mat, tile,
            skip=("-z",))
    f = unit((facing[0], facing[1], 0.0))
    t = math.radians(tilt_deg)
    n = (f[0] * math.cos(t), f[1] * math.cos(t), -math.sin(t))         # the bank's front faces this way
    across = unit(cross((0.0, 0.0, 1.0), n))
    up = cross(n, across)
    w, h, d = bank
    middle = (centre[0], centre[1], z1 + h / 2 * math.cos(t) + d / 2 * math.sin(t))
    oriented_box(b, middle, (n, across, up), (d / 2, w / 2, h / 2), mat, tile)
    if lamps:
        rows, cols = lamps
        front = add(middle, mul(n, d / 2 + 0.03))
        cw, ch = w / cols, h / rows
        for r in range(rows):
            for c in range(cols):
                cx, cy = -w / 2 + (c + 0.5) * cw, -h / 2 + (r + 0.5) * ch
                pts = [add(front, add(mul(across, cx + sx * 0.4 * cw), mul(up, cy + sy * 0.4 * ch)))
                       for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
                b.face(pts, lamp_mat, tile)
    return b


# ------------------------------------------------------------------ fills, offsets, walls along a line

MITRE_LIMIT = 4.0      # an offset corner moves at most this many offsets from its point


def _signed_area(ring) -> float:
    ring = list(ring)
    return sum(x0 * y1 - x1 * y0 for (x0, y0, *_), (x1, y1, *_) in zip(ring, ring[1:] + ring[:1])) / 2


def triangulate(ring) -> list[tuple[int, int, int]]:
    """Triangles (index triples into ring, each counter-clockwise) covering a simple polygon
    [(x, y), ...] given either way round, by ear clipping. A corner on a straight edge is kept
    in a triangle where it can be and dropped where it can't."""
    pts = [(float(p[0]), float(p[1])) for p in ring]
    area = _signed_area(pts)
    if abs(area) < 1e-9:
        raise ValueError("triangulate: the ring has no area")
    idx = list(range(len(pts))) if area > 0 else list(range(len(pts)))[::-1]

    def turn(i, j, k) -> float:
        (ax, ay), (bx, by), (cx, cy) = pts[i], pts[j], pts[k]
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

    def ear(k: int) -> bool:
        i, j, l = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
        if turn(i, j, l) <= 1e-12:
            return False
        corners = {pts[i], pts[j], pts[l]}
        return not any(pts[m] not in corners and turn(i, j, m) >= -1e-12 and turn(j, l, m) >= -1e-12
                       and turn(l, i, m) >= -1e-12 for m in idx)

    out = []
    while len(idx) > 3:
        k = next((k for k in range(len(idx)) if ear(k)), None)
        if k is None:                       # only flat corners left to cut: drop one
            k = next((k for k in range(len(idx)) if abs(turn(idx[k - 1], idx[k], idx[(k + 1) % len(idx)])) <= 1e-12),
                     None)
            if k is None:
                raise ValueError("triangulate: no ear to cut (does the ring cross itself?)")
            idx.pop(k)
            continue
        out.append((idx[k - 1], idx[k], idx[(k + 1) % len(idx)]))
        idx.pop(k)
    if turn(*idx) > 1e-12:
        out.append(tuple(idx))
    return out


def fill(b: Builder, ring, z: float, mat: str, tile: float = 1.0, col=None, down: bool = False) -> Builder:
    """The polygon ring [(x, y), ...] (a simple one, either way round) as a level surface at z,
    facing up (down: facing down), in triangles."""
    pts = [(float(p[0]), float(p[1])) for p in ring]
    for tri in triangulate(pts):
        corners = [(*pts[i], z) for i in tri]
        b.face(corners[::-1] if down else corners, mat, tile, col=col)
    return b


def _offset(pts, d: float, closed: bool) -> list[tuple[float, float]]:
    """Each point of a line moved d to the left of the way along it (mitred at the corners)."""
    pts = [(float(p[0]), float(p[1])) for p in pts]
    n = len(pts)
    edges = list(zip(pts, pts[1:] + pts[:1])) if closed else list(zip(pts, pts[1:]))
    lefts = []
    for (ax, ay), (bx, by) in edges:
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy)
        if length < 1e-12:
            raise ValueError("offset: two points of the line are the same")
        lefts.append((-dy / length, dx / length))
    out = []
    for i, (x, y) in enumerate(pts):
        if closed:
            n0, n1 = lefts[i - 1], lefts[i]
        else:
            n0, n1 = lefts[max(i - 1, 0)], lefts[min(i, n - 2)]
        mx, my = n0[0] + n1[0], n0[1] + n1[1]
        m = math.hypot(mx, my)
        if m < 1e-9:                         # the line turns right back
            mx, my, k = n1[0], n1[1], 1.0
        else:
            mx, my = mx / m, my / m
            k = min(1.0 / max(mx * n1[0] + my * n1[1], 1e-9), MITRE_LIMIT)
        out.append((x + mx * d * k, y + my * d * k))
    return out


def offset_line(pts, d: float) -> list[tuple[float, float]]:
    """The line [(x, y), ...] moved d to its left (right for a negative d), mitred at its bends;
    its ends move square to their segments."""
    return _offset(pts, d, closed=False)


def offset_ring(ring, d: float) -> list[tuple[float, float]]:
    """The ring [(x, y), ...] (either way round) with every edge moved d outwards (inwards for a
    negative d), mitred at the corners."""
    return _offset(ring, -d if _signed_area(ring) > 0 else d, closed=True)


def wall_along(b: Builder, pts, z0: float, z1: float, thickness: float, mat: str, back: str | None = None,
               top: str | None = None, tile: float = 1.0, col=None, ends: bool = True) -> Builder:
    """A wall from z0 to z1 along the line pts [(x, y), ...]: its front face on the line, facing
    to the left of the way along it, in mat (and colour col); its back face `thickness` to the
    right, in back (else mat); with top, its top in that material; with ends, its two ends
    closed (in mat)."""
    front = [(float(p[0]), float(p[1])) for p in pts]
    rear = offset_line(front, -thickness)
    back = back or mat
    for (a, c), (ra, rc) in zip(zip(front, front[1:]), zip(rear, rear[1:])):
        b.face([(*c, z0), (*a, z0), (*a, z1), (*c, z1)], mat, tile, col=col)
        b.face([(*ra, z0), (*rc, z0), (*rc, z1), (*ra, z1)], back, tile)
        if top:
            b.face([(*a, z1), (*ra, z1), (*rc, z1), (*c, z1)], top, tile, col=col)
    if ends:
        a, ra, c, rc = front[0], rear[0], front[-1], rear[-1]
        b.face([(*a, z0), (*ra, z0), (*ra, z1), (*a, z1)], mat, tile, col=col)
        b.face([(*rc, z0), (*c, z0), (*c, z1), (*rc, z1)], mat, tile, col=col)
    return b


# ------------------------------------------------------------------ footprints

def _inside(ring, x: float, y: float) -> bool:
    hit = False
    for i in range(len(ring)):
        (xa, ya), (xb, yb) = ring[i], ring[i - 1]
        if (ya > y) != (yb > y) and x < xa + (y - ya) * (xb - xa) / (yb - ya):
            hit = not hit
    return hit


def _lines(values, fixed, snap: float) -> list[float]:
    """Sorted values merged into grid lines: values within `snap` of each other become one,
    at a fixed value when the group holds one, else at the group's mean."""
    pts = sorted([(v, False) for v in values] + [(v, True) for v in fixed])
    groups, out = [[pts[0]]], []
    for p in pts[1:]:
        if p[0] - groups[-1][-1][0] < snap:
            groups[-1].append(p)
        else:
            groups.append([p])
    for g in groups:
        exact = [v for v, f in g if f]
        out.append(exact[0] if exact else sum(v for v, _ in g) / len(g))
    return out


def split_rectangles(ring, holes=(), snap: float = SNAP) -> list[tuple[float, float, float, float]]:
    """Axis-aligned rectangles (x0, y0, x1, y1) covering a nearly rectilinear ring [(x, y), ...]
    less the hole rectangles (x0, y0, x1, y1): the grid of the ring's and the holes' coordinates
    (merged within `snap`) is filled where a cell's centre lies inside the ring and outside the
    holes, then the largest rectangle of filled cells is taken, again and again (the larger,
    then the lower and more western, first)."""
    xs = _lines([p[0] for p in ring], [h[i] for h in holes for i in (0, 2)], snap)
    ys = _lines([p[1] for p in ring], [h[i] for h in holes for i in (1, 3)], snap)
    nx, ny = len(xs) - 1, len(ys) - 1
    fill = [[False] * nx for _ in range(ny)]
    for j in range(ny):
        for i in range(nx):
            cx, cy = (xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2
            in_hole = any(h[0] <= cx <= h[2] and h[1] <= cy <= h[3] for h in holes)
            fill[j][i] = _inside(ring, cx, cy) and not in_hole
    rects = []
    while True:
        best = None
        for j0 in range(ny):
            for i0 in range(nx):
                if not fill[j0][i0]:
                    continue
                width = nx - i0
                for j1 in range(j0, ny):
                    w = 0
                    while w < width and fill[j1][i0 + w]:
                        w += 1
                    width = w
                    if not width:
                        break
                    area = (xs[i0 + width] - xs[i0]) * (ys[j1 + 1] - ys[j0])
                    if best is None or area > best[0] + 1e-9:
                        best = (area, i0, j0, i0 + width, j1 + 1)
        if best is None:
            return rects
        _, i0, j0, i1, j1 = best
        rects.append((xs[i0], ys[j0], xs[i1], ys[j1]))
        for j in range(j0, j1):
            for i in range(i0, i1):
                fill[j][i] = False


# ------------------------------------------------------------------ levels of detail

def lod_nodes(shape: str, table, collision=None) -> list[tuple[str, object]]:
    """The nodes for brexport.export: `<shape>_a<px>` for each (px, max triangles, Builder) of
    the table, largest px first, then "Colmesh-1" from collision = (max triangles, Builder).
    A level over its triangle budget is an error."""
    levels = sorted(table, key=lambda row: -row[0])
    rows = [(f"{shape}_a{px}", limit, b) for px, limit, b in levels]
    if collision is not None:
        rows.append(("Colmesh-1", *collision))
    for name, limit, b in rows:
        if b.tris > limit:
            raise ValueError(f"{name} has {b.tris} triangles; its budget is {limit}")
    return [(name, b.object(name)) for name, _, b in rows]
