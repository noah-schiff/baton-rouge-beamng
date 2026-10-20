"""Helpers for the tree model scripts (blender/trees/): tapered tubes along a curve (the trunk,
branches along a quadratic curve tapering by branch_taper), the flared base, cone knees, fans of
leaf cards (fan_base, leaf_fan) and strands of hanging Spanish moss (moss_strand: hanging_card on
the generated moss card's moss_rect), collected in a TreeBuilder and turned into one Blender object
per level of detail. Every random choice draws from a generator of its own (seeded), so each level
of detail draws the same numbers. A script ends with export_tree, which names and checks the levels
as for the landmarks (brlm.lod_nodes) and writes them:

    export_tree(build, TREE_BUDGETS, collision=trunk_collision(0.7, 8.0, bark), billboard=BILLBOARD)

Unlike brlm's flat faces, a TreeBuilder's vertices carry their own normal (smooth tubes, leaf
cards lit as part of a round crown) and the wind weights the game's tree shader reads from
COLOR0 (shaders/common/material/shadergen/defaultMat.hlsl:446, windBranchBending):

* r: branch attenuation, how far the vertex bends up and down with its branch: 0 on the
  trunk, growing toward the branch tips;
* g: branch phase, 0..1, one value per branch so neighbouring branches swing apart;
* b: leaf-edge flutter, along the normal: 0 on wood, LEAF_FLUTTER on a card's base and tip, 0 to
  MOSS_FLUTTER down a moss strand.

They are never a tint; brexport writes them as the "Col" byte colour (alpha 1).

Coordinates are the model's: x, y across, z up from the ground at the trunk, in metres. Only
TreeBuilder.object needs Blender; the geometry is plain Python, so the tests can check it
without Blender.
"""
from __future__ import annotations

import math
import random

import brlm

Vec = tuple[float, float, float]
UP = (0.0, 0.0, 1.0)
GOLDEN = math.pi * (3 - math.sqrt(5))   # the golden angle, between one branch's heading and the next
WOOD = (0.0, 0.0, 0.0)          # the trunk's wind weights: it only sways as a whole
LEAF_FLUTTER = (0.15, 0.85)     # b on a card's base edge and on its tip edge
BEND = 0.5                      # how far a card's normals lean away from the crown's centre
MOSS_FLUTTER = 0.6              # b at a moss strand's foot (0 where it hangs from)
MOSS_CARD_M = (1.1, 2.2)        # m: the moss card's (texgen.moss_card, 512 x 1024) width and full height
MOSS_LIFT = 0.35                # how far a moss strand's normals lean up (lit from above, not edge-on)
MOSS_LENGTH, MOSS_WIDTH = (0.6, 2.2), (0.25, 0.5)   # m: a moss strand's length and width
MOSS_LEAN = 0.08                # m out per m down a moss strand leans, at most, in x and in y

# the custom trees' nodes (custom_trees.yaml): their levels of detail {px: most triangles}, the
# trunk prism Colmesh-1 (COLLISION_SIDES sides, at most COLLISION_BUDGET triangles) and the empty
# billboard node the level asks the game to fill with imposters (trees.IMPOSTERS)
TREE_BUDGETS = {600: 3000, 250: 1200, 80: 400}
COLLISION_SIDES, COLLISION_BUDGET = 8, 24
BILLBOARD = "bb_autobillboard25"


# ------------------------------------------------------------------ random choices

def seeded(*keys) -> random.Random:
    """A generator of its own for one choice, seeded by keys (joined with "/"), so each level of
    detail draws the same numbers. Its uniform(lo, hi) is lo + (hi - lo) x random()."""
    return random.Random("/".join(map(str, keys)))


def pick(rng: random.Random, weighted):
    """One item of weighted, [(item, weight), ...] with the weights summing to 1, by rng's next
    number (the last item when rounding leaves the sum short of it)."""
    roll, acc = rng.random(), 0.0
    for item, weight in weighted:
        acc += weight
        if roll < acc:
            break
    return item


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


def bezier(p0, p1, p2, t: float) -> Vec:
    """The quadratic Bezier curve through p0 and p2 pulled toward p1, at t in [0, 1]."""
    a, b, c = (1 - t) ** 2, 2 * (1 - t) * t, t * t
    return tuple(a * p0[i] + b * p1[i] + c * p2[i] for i in range(3))


def bezier_tangent(p0, p1, p2, t: float) -> Vec:
    """The (unit) direction of the curve at t."""
    return unit(add(mul(sub(p1, p0), 2 * (1 - t)), mul(sub(p2, p1), 2 * t)))


# ------------------------------------------------------------------ the builder

class TreeBuilder:
    """Vertices (position, normal, uv, wind weights) and triangles (three vertex indices and a
    material name), counter-clockwise seen from the side their normals face."""

    def __init__(self):
        self.verts: list[tuple[Vec, Vec, tuple[float, float], Vec]] = []
        self.triangles: list[tuple[int, int, int, str]] = []

    @property
    def tris(self) -> int:
        return len(self.triangles)

    @property
    def materials(self) -> set[str]:
        return {t[3] for t in self.triangles}

    def vertex(self, pos, nrm, uv, wind=WOOD) -> int:
        self.verts.append((tuple(float(c) for c in pos), unit(nrm), (float(uv[0]), float(uv[1])),
                           tuple(float(c) for c in wind)))
        return len(self.verts) - 1

    def triangle(self, i: int, j: int, k: int, mat: str) -> None:
        self.triangles.append((i, j, k, mat))

    def extend(self, other: "TreeBuilder") -> "TreeBuilder":
        base = len(self.verts)
        self.verts += other.verts
        self.triangles += [(i + base, j + base, k + base, m) for i, j, k, m in other.triangles]
        return self

    def object(self, name: str):
        """A Blender mesh object of these triangles: one material slot per material
        (brexport.material), the uv in layer 0, the normals as custom normals and the wind
        weights in the "Col" byte colour, linked to the scene."""
        import bpy        # only inside Blender
        import brexport

        if not self.triangles:
            raise ValueError(f"{name}: no triangles")
        names = sorted(self.materials)
        slot = {m: i for i, m in enumerate(names)}
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata([v[0] for v in self.verts], [], [t[:3] for t in self.triangles])
        for m in names:
            mesh.materials.append(brexport.material(m))
        mesh.polygons.foreach_set("material_index", [slot[t[3]] for t in self.triangles])
        corners = [i for t in self.triangles for i in t[:3]]      # from_pydata keeps this order
        mesh.uv_layers.new(name="UVMap").data.foreach_set("uv", [c for i in corners for c in self.verts[i][2]])
        colour = mesh.color_attributes.new(brexport.COLOR_ATTRIBUTE, "BYTE_COLOR", "CORNER")
        colour.data.foreach_set("color_srgb", [c for i in corners for c in (*self.verts[i][3], 1.0)])
        mesh.shade_smooth()
        mesh.normals_split_custom_set_from_vertices([v[1] for v in self.verts])
        mesh.update()
        ob = bpy.data.objects.new(name, mesh)
        bpy.context.scene.collection.objects.link(ob)
        return ob


# ------------------------------------------------------------------ wood

def flare(z: float, radius: float, factor: float = 2.2, height: float = 1.5) -> float:
    """The trunk's radius at z: radius x factor at the ground (and below it), back to radius by
    `height` above it, along (1 - z / height)^2."""
    t = min(max(z / height, 0.0), 1.0)
    return radius * (1.0 + (factor - 1.0) * (1.0 - t) ** 2)


def _frames(path) -> list[tuple[Vec, Vec, Vec]]:
    """(tangent, normal, binormal) at each point of a path, the normal carried along it without
    twisting (parallel transport); the first normal is as near +x as the tangent allows."""
    n = len(path)
    tangents = [unit(sub(path[min(i + 1, n - 1)], path[max(i - 1, 0)])) for i in range(n)]
    ref = (1.0, 0.0, 0.0) if abs(tangents[0][0]) < 0.9 else (0.0, 1.0, 0.0)
    frames, normal = [], ref
    for t in tangents:
        normal = unit(sub(normal, mul(t, dot(normal, t))))
        frames.append((t, normal, cross(t, normal)))
    return frames


def tube(b: TreeBuilder, path, radii, sides: int, mat: str, wind=WOOD, tile=(1.0, 1.0), lobes: int = 0,
         depth=None, cap: bool = False, u_repeat: int | None = None) -> TreeBuilder:
    """A tapered tube along `path` (points, at least two) with radii[i] at path[i]: one ring of
    `sides` vertices per point (the seam vertex twice, for the uv), smooth, facing out.

    * wind: the weights of every vertex, or a function of (position, s) giving them, s the
      share of the path's length up to the ring.
    * The uv wraps u_repeat times round (default: whole repeats of tile[0] metres round the mean
      radius) and runs tile[1] metres per repeat along the path.
    * lobes and depth (one value per point) flute the rings: radius x (1 + depth cos(lobes a)).
    * cap closes the far end with a short cone, one apex vertex per side.
    """
    if len(path) < 2 or len(radii) != len(path) or sides < 3:
        raise ValueError("tube wants two points or more, a radius for each and three sides or more")
    depth = depth or [0.0] * len(path)
    if u_repeat is None:
        u_repeat = max(1, round(2 * math.pi * (sum(radii) / len(radii)) / tile[0]))
    lengths = [0.0]
    for p, q in zip(path, path[1:]):
        lengths.append(lengths[-1] + math.dist(p, q))
    weights = wind if callable(wind) else (lambda pos, s: wind)
    frames = _frames(path)

    rings = []
    for (p, r, d, (_, nrm, bin_)) in zip(path, radii, depth, frames):
        ring = []
        for k in range(sides + 1):
            a = 2 * math.pi * k / sides
            rr = r * (1.0 + d * math.cos(lobes * a)) if lobes else r
            ring.append(add(p, add(mul(nrm, rr * math.cos(a)), mul(bin_, rr * math.sin(a)))))
        rings.append(ring)
    index = []
    for i, ring in enumerate(rings):
        row = []
        for k, pos in enumerate(ring):
            around = sub(ring[(k + 1) % sides], ring[(k - 1) % sides])
            along = sub(rings[min(i + 1, len(rings) - 1)][k], rings[max(i - 1, 0)][k])
            normal = cross(around, along)                      # outward: (t x dir) x t = dir
            if dot(normal, normal) < 1e-18:                    # a ring of radius 0
                normal = frames[i][0]
            s = lengths[i] / lengths[-1]
            row.append(b.vertex(pos, normal, (u_repeat * k / sides, lengths[i] / tile[1]), weights(pos, s)))
        index.append(row)
    for i in range(len(rings) - 1):
        for k in range(sides):
            a, bb, c, d = index[i][k], index[i][k + 1], index[i + 1][k + 1], index[i + 1][k]
            b.triangle(a, bb, c, mat)
            b.triangle(a, c, d, mat)
    if cap:
        t = frames[-1][0]
        apex = add(path[-1], mul(t, radii[-1]))
        v = (lengths[-1] + radii[-1]) / tile[1]
        for k in range(sides):
            top = b.vertex(apex, t, (u_repeat * (k + 0.5) / sides, v), weights(apex, 1.0))
            b.triangle(index[-1][k], index[-1][k + 1], top, mat)
    return b


def cone(b: TreeBuilder, base, radius: float, height: float, sides: int, mat: str, sink: float = 0.0,
         wind=WOOD, tile=(1.0, 1.0)) -> TreeBuilder:
    """An upright cone (a cypress knee) on `base`: its ring `sink` below it, its tip `height`
    above it, no bottom; smooth sides and a rounded-looking tip (one apex vertex per side,
    normal up)."""
    x, y, z = base
    h = height + sink
    slant = math.hypot(h, radius)
    ring = []
    for k in range(sides + 1):
        a = 2 * math.pi * k / sides
        d = (math.cos(a), math.sin(a), 0.0)
        ring.append(b.vertex((x + radius * d[0], y + radius * d[1], z - sink), add(mul(d, h), mul(UP, radius)),
                             (k / sides, 0.0), wind))
    for k in range(sides):
        top = b.vertex((x, y, z + height), UP, ((k + 0.5) / sides, slant / tile[1]), wind)
        b.triangle(ring[k], ring[k + 1], top, mat)
    return b


def trunk(b: TreeBuilder, radius_at, radius: float, top: float, swell_h: float, rings, step: float, sides: int,
          mat: str, tile=(1.0, 1.0), lobes: int = 0, flute: float = 0.0) -> TreeBuilder:
    """A capped trunk up the z axis to `top`: a ring at each z of `rings` (the swollen base, from below
    the ground up to swell_h), one at swell_h and then evenly up to top, at most `step` apart; radius_at(z)
    round. `lobes` flutes (tube's lobes), `flute` deep at z 0 and below and none from swell_h up, along
    (1 - z / swell_h)^2. The uv wraps whole repeats of tile[0] m round `radius` (the trunk's above the
    swell) and runs tile[1] m per repeat up."""
    steps = max(1, math.ceil((top - swell_h) / step))
    zs = [*rings, swell_h, *(swell_h + (top - swell_h) * k / steps for k in range(1, steps + 1))]
    depth = [flute * (1.0 - min(max(z / swell_h, 0.0), 1.0)) ** 2 for z in zs]
    return tube(b, [(0.0, 0.0, z) for z in zs], [radius_at(z) for z in zs], sides, mat, tile=tile, lobes=lobes,
                depth=depth, cap=True, u_repeat=max(1, round(2 * math.pi * radius / tile[0])))


def branch_taper(length: float, reach: float, sizes):
    """radius(s) of a branch `length` m long, s from 0 where it leaves the trunk to 1 at its tip, for
    sizes = (base, extra, tip) in m: base + extra x length / reach at the trunk (reach the longest
    branch's length, so the longer branches are the thicker), straight down to tip."""
    root = sizes[0] + sizes[1] * length / reach
    return lambda s: root + (sizes[2] - root) * s


def branch(b: TreeBuilder, curve, radius_at, sides: int, segments: int, mat: str, atten, phase: float,
           tile=(1.0, 1.0)) -> TreeBuilder:
    """A branch along the quadratic curve = (p0, p1, p2) (bezier): `segments` straight pieces, `sides`
    round, radius_at(s) round at s from 0 at p0 to 1 at p2 (branch_taper); the uv once round and
    tile[1] m per repeat along. Wind: r = atten(position), g = phase, b 0."""
    path = [bezier(*curve, k / segments) for k in range(segments + 1)]
    radii = [radius_at(k / segments) for k in range(segments + 1)]
    return tube(b, path, radii, sides, mat, wind=lambda pos, s: (atten(pos), phase, 0.0), tile=tile, u_repeat=1)


def radial_atten(peak: float, reach: float):
    """The wind's branch attenuation r as a function of position: 0 on the trunk's axis (z), growing
    with the distance from it to `peak` at `reach` m out and staying there beyond."""
    return lambda pos: peak * min(math.hypot(pos[0], pos[1]) / reach, 1.0)


# ------------------------------------------------------------------ leaves

def card_axes(heading, yaw: float, pitch: float, roll: float) -> tuple[Vec, Vec]:
    """(direction, side) of a leaf card: the horizontal heading turned by yaw about z, raised
    by pitch, and the card's width axis, level at roll 0 and turned by roll about the
    direction (roll pi / 2 stands the card on its edge)."""
    h = unit((heading[0], heading[1], 0.0))
    c, s = math.cos(yaw), math.sin(yaw)
    h = (c * h[0] - s * h[1], s * h[0] + c * h[1], 0.0)
    d = add(mul(h, math.cos(pitch)), mul(UP, math.sin(pitch)))
    side0 = unit(cross(UP, d))                                 # level, to the card's left
    side = add(mul(side0, math.cos(roll)), mul(cross(d, side0), math.sin(roll)))
    return d, unit(side)


def fan_base(curve, s: float, pull: float, radius_at) -> tuple[Vec, Vec]:
    """(base, heading) for a fan of leaf cards at s along a branch's curve = (p0, p1, p2): the
    curve's point there pulled `pull` m back toward the trunk along the branch's level heading (so
    larger cards start further in), but never closer to the z axis than 1.5 x the trunk's
    radius_at(z) there."""
    at = bezier(*curve, s)
    tangent = bezier_tangent(*curve, s)
    flat = unit((tangent[0], tangent[1], 0.0))
    inward = min(pull, max(0.0, math.hypot(at[0], at[1]) - 1.5 * radius_at(at[2])))
    return sub(at, mul(flat, inward)), flat


def leaf_fan(b: TreeBuilder, base, heading, cards, mat: str, atten, phase: float, centre) -> TreeBuilder:
    """Leaf cards from one point of a branch, each a quad whose base edge is centred on `base`.

    cards: (yaw, pitch, roll, length, width, uv_rect) each, the angles in radians (card_axes),
    uv_rect = (u0, v0, u1, v1) the sprig's place in the texture, its base at v0. A card faces up
    and out from the crown (its winding; the material is double-sided) and its normals lean
    BEND toward the direction away from `centre`, so the crown lights as one round mass.
    Wind: r = atten(position) at each corner, g = phase, b = LEAF_FLUTTER (base, tip).
    """
    out = (base[0] - centre[0], base[1] - centre[1], 0.0)
    prefer = add(UP, mul(unit(out), 0.5)) if dot(out, out) > 1e-12 else UP
    for yaw, pitch, roll, length, width, (u0, v0, u1, v1) in cards:
        d, side = card_axes(heading, yaw, pitch, roll)
        if dot(cross(side, d), prefer) < 0:
            side = mul(side, -1.0)
        normal = unit(cross(side, d))
        half = mul(side, width / 2)
        tip = add(base, mul(d, length))
        corners = [(sub(base, half), (u0, v0), 0), (add(base, half), (u1, v0), 0),
                   (add(tip, half), (u1, v1), 1), (sub(tip, half), (u0, v1), 1)]
        idx = []
        for pos, uv, edge in corners:
            away = sub(pos, centre)
            nrm = add(normal, mul(unit(away), BEND)) if dot(away, away) > 1e-12 else normal
            idx.append(b.vertex(pos, nrm, uv, (atten(pos), phase, LEAF_FLUTTER[edge])))
        b.triangle(idx[0], idx[1], idx[2], mat)
        b.triangle(idx[0], idx[2], idx[3], mat)
    return b


# ------------------------------------------------------------------ Spanish moss

def moss_rect(u: float, width: float) -> tuple[float, float, float, float]:
    """The uv rectangle (u0, v0, u1, v1) of a moss strand `width` m wide on the moss card: a slice
    width / MOSS_CARD_M[0] wide, starting at u (0..1) of the room left, the card's full height (v1 at
    its top row, where the filaments hang from)."""
    span = min(width / MOSS_CARD_M[0], 1.0)
    u0 = u * (1.0 - span)
    return (u0, 0.0, u0 + span, 1.0)


def hanging_card(b: TreeBuilder, top, yaw: float, length: float, width: float, uv_rect, mat: str,
                 atten: float = 0.0, phase: float = 0.0, flutter: float = MOSS_FLUTTER, lean=(0.0, 0.0),
                 outward=None) -> TreeBuilder:
    """A strand of hanging moss: one quad (the material is double-sided) whose top edge, `width` wide
    and centred on `top`, lies across the heading yaw (radians from +x), and which hangs `length` down
    from it, leaning lean = (x, y) m per m down. It faces along yaw, or against it when `outward` (a
    direction) says the other side faces out of the tree; its normals lean MOSS_LIFT up.

    uv_rect = (u0, v0, u1, v1) on the moss card, v1 along the top edge. Wind: r = atten and g = phase
    at every corner (the strand moves with what it hangs from), b 0 along the top edge and `flutter`
    along the foot, so the strand sways and its top stays put."""
    face = (math.cos(yaw), math.sin(yaw), 0.0)
    if outward is not None and dot(face, outward) < 0:
        face = mul(face, -1.0)
    side = (-face[1], face[0], 0.0)                            # across the strand: face x side = up
    half = mul(side, width / 2)
    foot = add(top, (lean[0] * length, lean[1] * length, -length))
    normal = unit(add(face, mul(UP, MOSS_LIFT)))
    u0, v0, u1, v1 = uv_rect
    corners = [(sub(top, half), (u0, v1), 0.0), (add(top, half), (u1, v1), 0.0),
               (add(foot, half), (u1, v0), flutter), (sub(foot, half), (u0, v0), flutter)]
    idx = [b.vertex(pos, normal, uv, (atten, phase, b_)) for pos, uv, b_ in corners]
    b.triangle(idx[0], idx[2], idx[1], mat)
    b.triangle(idx[0], idx[3], idx[2], mat)
    return b


def moss_strand(b: TreeBuilder, top, yaw: float, length: float, width: float, u: float, mat: str,
                atten: float = 0.0, phase: float = 0.0, lean=(0.0, 0.0)) -> TreeBuilder:
    """A strand of Spanish moss hanging from `top` in a tree whose trunk stands on the z axis:
    hanging_card on the slice moss_rect(u, width) of the moss card, facing out of the tree."""
    return hanging_card(b, top, yaw, length, width, moss_rect(u, width), mat, atten=atten, phase=phase, lean=lean,
                        outward=(top[0], top[1], 0.0))


# ------------------------------------------------------------------ the nodes

def trunk_collision(radius: float, top: float, mat: str) -> brlm.Builder:
    """Colmesh-1 of a tree: a COLLISION_SIDES-sided prism `radius` round the z axis from the ground to
    `top` (custom_trees.yaml's collision: trunk_radius_m and height_m)."""
    return brlm.prism(brlm.Builder(), (0.0, 0.0), radius, 0.0, top, COLLISION_SIDES, mat)


def export_tree(build, budgets: dict, collision: brlm.Builder | None = None, billboard: str | None = None,
                tail=()) -> None:
    """A tree script's main: build(params) gives {px: TreeBuilder}, for the params and out folder
    brexport.args() reads (the folder's name is the shape's); brlm.lod_nodes names the levels
    <shape>_a<px> and checks them against budgets {px: most triangles}, then collision (a
    trunk_collision, at most COLLISION_BUDGET triangles) as Colmesh-1; the nodes of `tail`, as
    (name, None) for an empty one (a nulldetail), follow; brexport.export writes the part "main" with
    the billboard node."""
    import brexport       # only inside Blender (the tests stub it, to stop here)

    params, out = brexport.args()
    levels = build(params)
    nodes = brlm.lod_nodes(out.name, [(px, budgets[px], levels[px]) for px in budgets],
                           collision=None if collision is None else (COLLISION_BUDGET, collision))
    brexport.export({"main": [*nodes, *tail]}, out, billboard=billboard)
