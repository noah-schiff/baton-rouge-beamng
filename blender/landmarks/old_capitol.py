"""The Old State Capitol (plan Task 18): the white Gothic Revival "castle" on North Boulevard, its
crenellated octagonal towers, pointed-arch windows and the entrances north and south.

Params (out/landmarks.json): height_m (the towers' battlements; the entrance block's stand
BLOCK_UNDER lower), towers_m (the top: the raised roof over the rotunda, where the lidar's highest
return is),
footprint_local and ground_dz. The model's +u is the footprint's long axis (2.5 degrees north of
east), +v north. Heights are above the ground (added to ground_dz); walls reach down to z 0.

On the lidar (data/out, 2026-10-19) the building is a cross: the wings east and west read 19-21 m
(WING_M), the entrance block across them (the footprint between its north and south entrance
fronts) 24-27 m, the corner towers 23-24 m, and the centre 28-36 m: a raised block over the
rotunda with a glazed roof (the dark square on NAIP 2023).

* The sections: the footprint cut at the entrance block's sides (the span of its corners beyond
  ENTRANCE_V) into the west wing (to WING_M), the entrance block (BLOCK_UNDER under height_m) and
  the east wing; each one's walls from the ground (or from the lower section's roof beside them)
  to its battlements, under a flat roof at the foot of its merlons (MERLON_W wide every
  MERLON_PITCH, MERLON_H high, MERLON_T thick).
* Eight octagonal towers (TOWERS: on the footprint's corner turrets and either side of each
  entrance, a little proud of its walls) to height_m, a merlon on every other side.
* The rotunda's block (ROTUNDA) from the entrance block's roof to ROTUNDA_M, under a glazed
  pyramid (br_lm_paint, dark) to towers_m.
* Windows: decal quads of texgen.gothic_window (br_lm_gothic_window, cut out by its alpha) in
  rows (ROWS) along every wall, each row where it fits (the last only over the wings' roofs, on
  the entrance block), clear of the towers; at each entrance (where the footprint's axis meets it,
  north and south) a tall pointed doorway and the great window over it.

Levels of detail: old_capitol_a400 (all of it, at most 4,500 triangles), _a100 (no merlons; 1,500),
_a25 (the sections and the rotunda as prisms, the towers over them, no windows; 360),
Colmesh-1 (the sections' prisms; 220).
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

import brexport  # noqa: E402
import brlm  # noqa: E402

SHAPE = "old_capitol"
STUCCO, WINDOW, ROOF, GLAZING = "br_lm_stucco_white", "br_lm_gothic_window", "br_lm_steel", "br_lm_paint"
TILE, ROOF_TILE = 3.0, 4.0      # m per texture repeat
SKYLIGHT = (0.20, 0.24, 0.28)   # the rotunda's glazed roof

WING_M = 21.0                   # the wings' battlements (the lidar's median over the footprint, 21.0 m)
BLOCK_UNDER = 2.0               # the entrance block's battlements stand this far under the towers' (height_m):
#                                 24.3 m, in the lidar's 24-27 m there, so the towers rise clear of it
ENTRANCE_V = 15.0               # footprint corners further north or south than this (the wings reach 14.3 m) are
#                                 the entrance fronts'
MERLON_W, MERLON_PITCH, MERLON_H, MERLON_T = 0.9, 1.6, 0.9, 0.45
# the towers: centre (u, v) and apothem, round the footprint's turrets (its outline, 2026-10-19 resolve)
TOWERS = ((-25.8, 6.8, 3.5), (-25.9, -7.1, 3.5), (26.0, 8.0, 3.7), (26.05, -7.85, 3.7),       # the corners
          (-6.4, 17.3, 3.3), (7.47, 17.3, 3.3), (-6.46, -17.4, 3.3), (7.67, -17.45, 3.3))      # by the entrances
TOWER_PROUD = 0.15              # how far a tower's sides stand out of the walls its outline traces
MERLONS_PER_TOWER = 4
ROTUNDA = (-5.0, -9.0, 10.5, 9.0)   # u0, v0, u1, v1: the lidar's 28-33 m block round the 34-36 m top
ROTUNDA_M = 31.0
# window rows on the walls: (bottom above the ground, decal width); a decal is GOTHIC_ASPECT times as
# tall as it is wide (texgen), and a row goes only where it fits on the wall
GOTHIC_ASPECT = 2.2
ROWS = ((2.2, 1.3), (8.0, 2.0), (14.8, 1.5), (20.5, 1.0))   # the last on the entrance block over the wings
WINDOW_PITCH = 4.2              # m between windows along a wall
DECAL_OUT = 0.03                # m in front of the wall
DOOR, GREAT_WINDOW = (3.4, 0.0), (4.4, 8.6)   # the entrances: (width, bottom above the ground)

BUDGETS = {400: 4500, 100: 1500, 25: 360}
COLLISION_BUDGET = 220


# ------------------------------------------------------------------ the params

def clip(ring, c: float, keep_low: bool):
    """The part of the counter-clockwise ring [(u, v), ...] with u <= c (keep_low) or u >= c
    (Sutherland-Hodgman), counter-clockwise, without repeated corners."""
    def inside(p):
        return p[0] <= c if keep_low else p[0] >= c

    out = []
    for i, p in enumerate(ring):
        q = ring[(i + 1) % len(ring)]
        if inside(p):
            out.append(p)
        if inside(p) != inside(q):
            t = (c - p[0]) / (q[0] - p[0])
            out.append((c, p[1] + t * (q[1] - p[1])))
    return [p for i, p in enumerate(out) if math.dist(p, out[i - 1]) > 1e-6]


def resolve(params: dict) -> dict:
    """The dimensions the builders share, from the script's params."""
    g = float(params.get("ground_dz", 0.0))
    foot = [tuple(map(float, q)) for q in params["footprint_local"]]
    fronts = [u for u, v in foot if abs(v) > ENTRANCE_V]
    u0, u1 = min(fronts), max(fronts)
    wing, top = g + WING_M, g + float(params["height_m"])
    block = top - BLOCK_UNDER
    sections = [(clip(foot, u0, True), wing), (clip(clip(foot, u0, False), u1, True), block),
                (clip(foot, u1, False), wing)]
    uc = (u0 + u1) / 2
    entrances = [(uc, brlm.ring_at(foot, (uc, 0.0), a)[1]) for a in (90.0, 270.0)]
    return {"g": g, "footprint": foot, "bar": (u0, u1), "sections": sections, "top": top, "block": block,
            "towers": [(u, v, r / math.cos(math.pi / 8) + TOWER_PROUD, top) for u, v, r in TOWERS],
            "rotunda": ROTUNDA, "rotunda_top": g + ROTUNDA_M, "lantern_top": g + float(params["towers_m"]),
            "entrances": entrances}


# ------------------------------------------------------------------ pieces

def face(b, pts, normal, mat=STUCCO, tile=TILE, col=None) -> None:
    """A planar convex face, its corners counter-clockwise as seen from `normal`'s side."""
    if brlm.dot(brlm.normal(pts), normal) < 0:
        pts = pts[::-1]
    b.face(pts, mat, tile, col=col)


def roof_z(top: float, px) -> float:
    """Where a section's roof lies: at the foot of its merlons (a400), else at its top."""
    return top - MERLON_H if px == 400 else top


def in_tower(p: dict, u: float, v: float, pad: float = 0.0) -> bool:
    return any(math.hypot(u - tu, v - tv) <= r + pad for tu, tv, r, _ in p["towers"])


def merlon(b, P, t, n, s0: float, s1: float, z0: float, z1: float) -> None:
    """A merlon from s0 to s1 along a wall frame P(s, z, out), its front in the wall's face."""
    T = -MERLON_T
    face(b, [P(s0, z0), P(s1, z0), P(s1, z1), P(s0, z1)], n)
    face(b, [P(s0, z0, T), P(s1, z0, T), P(s1, z1, T), P(s0, z1, T)], (-n[0], -n[1], 0.0))
    face(b, [P(s0, z0), P(s0, z0, T), P(s0, z1, T), P(s0, z1)], (-t[0], -t[1], 0.0))
    face(b, [P(s1, z0), P(s1, z0, T), P(s1, z1, T), P(s1, z1)], (t[0], t[1], 0.0))
    face(b, [P(s0, z1), P(s1, z1), P(s1, z1, T), P(s0, z1, T)], (0.0, 0.0, 1.0))


def wall(b, p: dict, a, c, z0: float, z1: float, px) -> None:
    """One wall from (u, v) a to c, facing to the right of a -> c, from z0 to z1: at a400 to the foot
    of its merlons (z1 - MERLON_H) and the merlons on it; at a400 and a100 the rows of windows that
    fit on it."""
    du, dv = c[0] - a[0], c[1] - a[1]
    length = math.hypot(du, dv)
    crenel = roof_z(z1, px)
    if length < 1e-6 or crenel <= z0:
        return
    t = (du / length, dv / length)
    n = (t[1], -t[0], 0.0)

    def P(s, z, out=0.0):
        return (a[0] + t[0] * s + n[0] * out, a[1] + t[1] * s + n[1] * out, z)

    face(b, [P(0, z0), P(length, z0), P(length, crenel), P(0, crenel)], n)
    if px == 25:
        return
    if px == 400:
        count = int((length + MERLON_PITCH - MERLON_W) // MERLON_PITCH)
        start = (length - (count - 1) * MERLON_PITCH - MERLON_W) / 2
        for k in range(count):
            s0 = start + k * MERLON_PITCH
            mid = P(s0 + MERLON_W / 2, crenel, -MERLON_T / 2)
            if not in_tower(p, mid[0], mid[1], 0.3):
                merlon(b, P, t, n, s0, s0 + MERLON_W, crenel, z1)
    count = int(length // WINDOW_PITCH)
    for k in range(count):
        s = (k + 0.5) * length / count
        mid = P(s, 0.0)
        if any(math.hypot(mid[0] - u, mid[1] - v) < 4.0 for u, v in p["entrances"]):
            continue
        for bottom, width in ROWS:
            zb, zt = p["g"] + bottom, p["g"] + bottom + width * GOTHIC_ASPECT
            edges = [P(s + side * (width / 2 + 0.2), 0.0) for side in (-1, 1)]   # clear of the towers
            fits = zb >= z0 + 0.3 and zt <= crenel - 0.3 and width <= length / count - 0.6
            if fits and not any(in_tower(p, e[0], e[1]) for e in edges):
                brlm.decal(b, P(s, (zb + zt) / 2, DECAL_OUT), (t[0], t[1], 0.0), width, zt - zb, WINDOW)


def sections(b, p: dict, px) -> None:
    """Each section's walls (from the ground, or from the lower section's roof beside them) and roof."""
    secs = p["sections"]
    for i, (ring, top) in enumerate(secs):
        for a, c in zip(ring, ring[1:] + ring[:1]):
            du, dv = c[0] - a[0], c[1] - a[1]
            length = math.hypot(du, dv)
            if length < 1e-6:
                continue
            mid = ((a[0] + c[0]) / 2 + dv / length * 0.3, (a[1] + c[1]) / 2 - du / length * 0.3)  # just outside
            beside = [t for j, (r, t) in enumerate(secs) if j != i and brlm._inside(r, *mid)]
            if beside and beside[0] >= top:
                continue                                       # inside the taller section next to it
            z0 = roof_z(beside[0], px) if beside else 0.0
            wall(b, p, a, c, z0, top, px)
        brlm.fill(b, ring, roof_z(top, px), ROOF, ROOF_TILE)


def towers(b, p: dict, px) -> None:
    """The octagonal towers to height_m, a merlon on every other side (a400); at a25 only the parts
    over the wings and the entrance block."""
    for cu, cv, r, top in p["towers"]:
        z0 = 0.0
        if px == 25:
            z0 = p["block"] if brlm._inside(p["sections"][1][0], cu, cv) else p["g"] + WING_M
        ring = [(cu + r * math.cos(math.radians(22.5 + 45 * k)), cv + r * math.sin(math.radians(22.5 + 45 * k)))
                for k in range(8)]
        crenel = roof_z(top, px)
        brlm.extrude_ring(b, ring, z0, crenel, STUCCO, TILE)
        brlm.fill(b, ring, crenel, ROOF, ROOF_TILE)
        if px != 400:
            continue
        for k in range(0, 8, 8 // MERLONS_PER_TOWER):
            a, c = ring[k], ring[(k + 1) % 8]
            du, dv = c[0] - a[0], c[1] - a[1]
            length = math.hypot(du, dv)
            t = (du / length, dv / length)
            n = (t[1], -t[0], 0.0)

            def P(s, z, out=0.0, a=a, t=t, n=n):
                return (a[0] + t[0] * s + n[0] * out, a[1] + t[1] * s + n[1] * out, z)

            merlon(b, P, t, n, 0.15 * length, 0.85 * length, crenel, top)


def rotunda(b, p: dict, px) -> None:
    """The raised block over the rotunda and its glazed pyramid to towers_m."""
    u0, v0, u1, v1 = p["rotunda"]
    z0, z1 = roof_z(p["block"], px), p["rotunda_top"]
    ring = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
    for a, c in zip(ring, ring[1:] + ring[:1]):
        du, dv = c[0] - a[0], c[1] - a[1]
        length = math.hypot(du, dv)
        t = (du / length, dv / length)
        face(b, [(*a, z0), (*c, z0), (*c, z1), (*a, z1)], (t[1], -t[0], 0.0))
        if px != 25:                                           # a row of small windows round it
            count = int(length // 3.0)
            for k in range(count):
                s = (k + 0.5) * length / count
                width = 1.0
                h = width * GOTHIC_ASPECT
                centre = (a[0] + t[0] * s + t[1] * DECAL_OUT, a[1] + t[1] * s - t[0] * DECAL_OUT, (z0 + z1) / 2)
                if z1 - z0 > h + 0.6:
                    brlm.decal(b, centre, (t[0], t[1], 0.0), width, h, WINDOW)
    apex = ((u0 + u1) / 2, (v0 + v1) / 2, p["lantern_top"])
    for a, c in zip(ring, ring[1:] + ring[:1]):
        b.face([(*a, z1), (*c, z1), apex], GLAZING, col=SKYLIGHT)


def entrances(b, p: dict) -> None:
    """At each entrance, the doorway and the great window over it, on the entrance face."""
    for (u, v), right in zip(p["entrances"], ((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0))):
        out = DECAL_OUT if v > 0 else -DECAL_OUT
        for width, bottom in (DOOR, GREAT_WINDOW):
            h = width * GOTHIC_ASPECT
            brlm.decal(b, (u, v + out, p["g"] + bottom + h / 2), right, width, h, WINDOW)


# ------------------------------------------------------------------ levels of detail

def level(p: dict, px) -> brlm.Builder:
    b = brlm.Builder()
    sections(b, p, px)
    towers(b, p, px)
    rotunda(b, p, px)
    if px != 25:
        entrances(b, p)
    return b


def collision(p: dict) -> brlm.Builder:
    b = brlm.Builder()
    for ring, top in p["sections"]:
        brlm.extrude_ring(b, ring, 0.0, top - MERLON_H, STUCCO)
        brlm.fill(b, ring, top - MERLON_H, STUCCO)
    return b


def parts(p: dict) -> dict[str, dict]:
    """{part: {"node": its nodes' name, "lods": [(px, triangle budget, Builder), ...],
    "collision": (budget, Builder)}}."""
    return {"main": {"node": SHAPE, "lods": [(px, BUDGETS[px], level(p, px)) for px in (400, 100, 25)],
                     "collision": (COLLISION_BUDGET, collision(p))}}


def main() -> None:
    params, out = brexport.args()
    nodes = {}
    for name, part in parts(resolve(params)).items():
        nodes[name] = brlm.lod_nodes(part["node"], part["lods"], collision=part["collision"])
    brexport.export(nodes, out)
    for name, part_nodes in nodes.items():
        for node, ob in part_nodes:
            print(f"{name} {node}: {sum(len(f.vertices) - 2 for f in ob.data.polygons)} triangles, "
                  f"{sum(len(f.vertices) for f in ob.data.polygons)} corners")


main()
