"""Mike the Tiger's habitat at LSU (plan Task 16): the rock work, the pool and the stream, the
night house with its tower, and the glass and fence round the enclosure. No animal.

Params (out/landmarks.json, from landmarks.resolve), traced on the NAIP 2023 aerial and read from
the lidar (src/brmap/data/landmarks.yaml cites each): footprint_local (the enclosure and the night
house), tower_m (the tower's top, above the ground) and night_house_m (its centre), pool_m (the
pool's outline), stream_m (the stream from the falls down to the pool), rocks_m ([x, y, radius,
height] of each pile of rock), viewing_m (the enclosure's public edge, along the paved walk), and
ground_dz. The model's +u is east and +v north (the registry's yaw, azimuth 90) and the origin is
the OSM node, so the params are metres east and north of it. Heights below are above the ground
(added to ground_dz); walls and rocks reach down to z 0, so nothing floats. The terrain is not
carved: the pool and the stream stand POOL_LIFT and more above the ground (the LSU and full-map
terrain rise at most 0.14 m over the median under the pool and 0.56 m along the stream), inside
rock rims and banks.

One shape part, under validate.DAE_MAX_VERTS vertices per level of detail:
* main: the night house (a stucco block under a hipped tile roof, BODY round the tower's centre)
  and its tower (TOWER_W square, blind arches round its belvedere, a stone cornice, a pyramid tile
  roof to tower_m); a faceted pile of rock for each rocks_m, the falls on the one by the stream's
  head; the stream (STREAM_W wide, falling from STREAM_HEAD to the pool) between rock banks; the
  pool (water at POOL_LIFT) inside a rock rim; along viewing_m a concrete curb with glass panes
  between steel posts under a rail; round the rest of the outline (not the night house's side) a
  steel fence of posts, rails and bars.
Levels of detail: habitat_a300, _a80 and _a20, and Colmesh-1 (the curb and glass, the fence, the
night house and its tower, the rocks).
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

import brexport  # noqa: E402
import brlm  # noqa: E402

STUCCO, TILE_ROOF, TRIM = "br_lm_stucco_tan", "br_lm_roof_tile", "br_lm_trim_stone"
ROCK, WATER, GLASS, STEEL, CONCRETE = "br_lm_rock", "br_lm_water", "br_lm_glass", "br_lm_steel", "br_lm_concrete"
TILE, ROCK_TILE = 2.0, 3.0      # m per texture repeat

# ------------------------------------------------------------------ the night house (m)
TOWER_W = 4.5                   # the lidar's 14 m cells cover 4 x 4 m
TOWER_ROOF = 1.6                # the pyramid's rise
CORNICE, CORNICE_OUT = 0.4, 0.25
BELVEDERE = 3.0                 # the band of blind arches under the cornice
BODY = (-4.9, -4.1, 4.6, 3.4)   # the block round the tower's centre (x0, y0, x1, y1): the lidar's 4-6 m returns
BODY_H, BODY_PITCH, BODY_OVERHANG = 5.0, 25.0, 0.4
HOUSE_MARGIN = 1.2              # an outline edge this close to the block is its wall, not a fence

# ------------------------------------------------------------------ water and rock (m)
POOL_LIFT = 0.45                # the pool's water over the ground
RIM_W, RIM_UP = 0.8, 0.25       # the pool's rock rim: width, over the water
STREAM_W, STREAM_HEAD = 1.4, 0.95   # the stream: width, its water over the ground at the falls
BANK_W, BANK_UP = 0.5, 0.2
FALLS_W = 1.6
ROCK_RINGS = ((1.0, 0.0), (0.78, 0.55), (0.42, 0.88))   # (radius share, height share) up a pile
ROCK_JITTER = 0.18
ROCK_SIDES = {300: 7, 80: 5, 20: 4, "collision": 5}

# ------------------------------------------------------------------ the edge (m)
CURB_H, CURB_T = 0.9, 0.4
GLASS_TOP = 3.4                 # the panes' top over the ground
POST_GAP, POST_W = 2.5, 0.12
RAIL_H = 0.1
FENCE_H, BAR_GAP, BAR_W = 4.0, 0.4, 0.04

LODS = (300, 80, 20)
BUDGETS = {300: 2500, 80: 1000, 20: 180}      # ~20 % over the LSU build
COLLISION_BUDGET = 160


# ------------------------------------------------------------------ the params

def _ccw(ring) -> list:
    ring = [tuple(map(float, q)) for q in ring]
    return ring if brlm._signed_area(ring) > 0 else ring[::-1]


def resolve(params: dict) -> dict:
    """The dimensions the builders share, from the script's params."""
    g = float(params.get("ground_dz", 0.0))
    tx, ty = map(float, params["night_house_m"])
    return {"g": g, "outline": _ccw(params["footprint_local"]), "tower": (tx, ty),
            "z_tower": g + float(params["tower_m"]),
            "body": (tx + BODY[0], ty + BODY[1], tx + BODY[2], ty + BODY[3]),
            "pool": _ccw(params["pool_m"]), "z_pool": g + POOL_LIFT,
            "stream": [tuple(map(float, q)) for q in params["stream_m"]],
            "rocks": [tuple(map(float, r)) for r in params["rocks_m"]],
            "viewing": [tuple(map(float, q)) for q in params["viewing_m"]]}


# ------------------------------------------------------------------ the night house

def night_house(b, p: dict, px) -> None:
    g = p["g"]
    x0, y0, x1, y1 = p["body"]
    brlm.box(b, (x0, y0, 0.0), (x1, y1, g + BODY_H), STUCCO, TILE, skip=("-z", "+z"))
    brlm.hipped_roof(b, x0, y0, x1, y1, g + BODY_H, BODY_PITCH, TILE_ROOF, TILE, overhang=BODY_OVERHANG,
                     soffit=STUCCO if px != 20 else None)
    tx, ty = p["tower"]
    h = TOWER_W / 2
    eaves = p["z_tower"] - TOWER_ROOF
    band = eaves - CORNICE - BELVEDERE
    if px == 300:                     # plain walls, then blind arches round the belvedere
        brlm.box(b, (tx - h, ty - h, 0.0), (tx + h, ty + h, band), STUCCO, TILE, skip=("-z", "+z"))
        corners = [(tx - h, ty - h), (tx + h, ty - h), (tx + h, ty + h), (tx - h, ty + h)]
        for (ax, ay), (cx, cy) in zip(corners, corners[1:] + corners[:1]):
            right = ((cx - ax) / TOWER_W, (cy - ay) / TOWER_W, 0.0)
            openings = [(TOWER_W * k / 4, 1.1, band + 0.4, eaves - CORNICE - 1.2) for k in (1, 3)]
            brlm.blind_arches(b, (ax, ay, 0.0), right, TOWER_W, band, eaves - CORNICE, openings, 0.3, STUCCO, TILE,
                              segments=6)
    else:
        brlm.box(b, (tx - h, ty - h, 0.0), (tx + h, ty + h, eaves - CORNICE), STUCCO, TILE, skip=("-z", "+z"))
    o = CORNICE_OUT
    brlm.box(b, (tx - h - o, ty - h - o, eaves - CORNICE), (tx + h + o, ty + h + o, eaves), TRIM, TILE, skip=("+z",))
    brlm.pyramid_roof(b, tx - h - o, ty - h - o, tx + h + o, ty + h + o, eaves, p["z_tower"], TILE_ROOF, TILE)


def night_house_collision(b, p: dict) -> None:
    x0, y0, x1, y1 = p["body"]
    brlm.box(b, (x0, y0, 0.0), (x1, y1, p["g"] + BODY_H), STUCCO, skip=("-z",))
    tx, ty = p["tower"]
    h = TOWER_W / 2
    brlm.box(b, (tx - h, ty - h, 0.0), (tx + h, ty + h, p["z_tower"] - TOWER_ROOF), STUCCO, skip=("-z",))


# ------------------------------------------------------------------ rock and water

def jitter(*keys) -> float:
    """A repeatable number in [-1, 1) from integer keys."""
    v = math.sin(sum((k + 1) * f for k, f in zip(keys, (12.9898, 78.233, 37.719)))) * 43758.5453
    return 2.0 * (v - math.floor(v)) - 1.0


def rock(b, p: dict, x: float, y: float, r: float, h: float, sides: int, seed: int, collision: bool = False) -> None:
    """A faceted pile of rock round (x, y): rings of ROCK_RINGS, each corner jittered, to an apex h
    over the ground; for collision a prism to 0.6 h."""
    g = p["g"]
    if collision:
        ring = [(x + r * math.cos(2 * math.pi * k / sides), y + r * math.sin(2 * math.pi * k / sides))
                for k in range(sides)]
        brlm.extrude_ring(b, ring, 0.0, g + 0.6 * h, ROCK, top=ROCK)
        return
    rings = []
    for j, (share, up) in enumerate(ROCK_RINGS):
        z = 0.0 if j == 0 else g + up * h
        turn = 0.45 * j
        radii = [r * share * (1 + ROCK_JITTER * jitter(seed, j, k)) for k in range(sides)]
        angles = [2 * math.pi * k / sides + turn for k in range(sides)]
        rings.append([(x + rk * math.cos(a), y + rk * math.sin(a), z) for a, rk in zip(angles, radii)])
    brlm.loft(b, rings, ROCK, ROCK_TILE, closed=True, flip=True)       # outwards and up
    top = rings[-1]
    apex = (x + 0.1 * r * jitter(seed, 9, 1), y + 0.1 * r * jitter(seed, 9, 2), g + h)
    for k in range(sides):
        brlm.quad(b, [top[k], top[(k + 1) % sides], apex], ROCK, ROCK_TILE)


def stream_heights(p: dict) -> list[float]:
    """The water's z at each point of the stream: from STREAM_HEAD over the ground at the falls
    down to the pool's water, even along its length."""
    pts = p["stream"]
    along = [0.0]
    for a, c in zip(pts, pts[1:]):
        along.append(along[-1] + math.dist(a, c))
    z0, z1 = p["g"] + STREAM_HEAD, p["z_pool"]
    return [z0 + (z1 - z0) * s / along[-1] for s in along]


def falls(b, p: dict) -> None:
    """A sheet of water down the face of the rock nearest the stream's head, towards the stream."""
    head = p["stream"][0]
    x, y, r, h = min(p["rocks"], key=lambda q: math.dist(q[:2], head))
    ux, uy = p["stream"][1][0] - x, p["stream"][1][1] - y
    n = math.hypot(ux, uy)
    ux, uy = ux / n, uy / n
    tx, ty = -uy * FALLS_W / 2, ux * FALLS_W / 2
    z0, z1 = stream_heights(p)[0], p["g"] + 0.75 * h
    b0, t0 = (x + ux * r, y + uy * r), (x + ux * 0.6 * r, y + uy * 0.6 * r)
    sheet = [(b0[0] - tx, b0[1] - ty, z0), (b0[0] + tx, b0[1] + ty, z0), (t0[0] + tx, t0[1] + ty, z1),
             (t0[0] - tx, t0[1] - ty, z1)]
    nrm = brlm.normal(sheet)
    b.face(sheet if nrm[0] * ux + nrm[1] * uy > 0 else sheet[::-1], WATER)


def stream(b, p: dict, px) -> None:
    """The water between rock banks, the banks' tops and sides (inner sides at the full level)."""
    pts, zs = p["stream"], stream_heights(p)
    L, R = brlm.offset_line(pts, STREAM_W / 2), brlm.offset_line(pts, -STREAM_W / 2)
    LL, RR = brlm.offset_line(pts, STREAM_W / 2 + BANK_W), brlm.offset_line(pts, -STREAM_W / 2 - BANK_W)
    for k in range(len(pts) - 1):
        w0, w1 = zs[k], zs[k + 1]
        k0, k1 = w0 + BANK_UP, w1 + BANK_UP
        brlm.quad(b, [(*R[k], w0), (*R[k + 1], w1), (*L[k + 1], w1), (*L[k], w0)], WATER)
        if px == 20:
            continue
        brlm.quad(b, [(*L[k], k0), (*L[k + 1], k1), (*LL[k + 1], k1), (*LL[k], k0)], ROCK, ROCK_TILE)
        brlm.quad(b, [(*RR[k], k0), (*RR[k + 1], k1), (*R[k + 1], k1), (*R[k], k0)], ROCK, ROCK_TILE)
        brlm.quad(b, [(*LL[k + 1], 0.0), (*LL[k], 0.0), (*LL[k], k0), (*LL[k + 1], k1)], ROCK, ROCK_TILE)
        brlm.quad(b, [(*RR[k], 0.0), (*RR[k + 1], 0.0), (*RR[k + 1], k1), (*RR[k], k0)], ROCK, ROCK_TILE)
        if px == 300:
            brlm.quad(b, [(*L[k], w0), (*L[k + 1], w1), (*L[k + 1], k1), (*L[k], k0)], ROCK, ROCK_TILE)
            brlm.quad(b, [(*R[k + 1], w1), (*R[k], w0), (*R[k], k0), (*R[k + 1], k1)], ROCK, ROCK_TILE)


def pool(b, p: dict, px) -> None:
    """The water and its rock rim (top, inner and outer sides)."""
    ring, z = p["pool"], p["z_pool"]
    brlm.fill(b, ring, z, WATER)
    if px == 20:
        return
    out = brlm.offset_ring(ring, RIM_W)
    zr = z + RIM_UP
    inner = [(*q, zr) for q in ring]
    outer = [(*q, zr) for q in out]
    brlm.loft(b, [inner, outer], ROCK, ROCK_TILE, closed=True)                          # the top
    brlm.loft(b, [[(*q, z) for q in ring], inner], ROCK, ROCK_TILE, closed=True)        # facing the water
    brlm.loft(b, [[(*q, 0.0) for q in out], outer], ROCK, ROCK_TILE, closed=True, flip=True)


# ------------------------------------------------------------------ the edge

def stations(a, c, gap: float) -> list[tuple[float, float]]:
    """Points from a to c (both ends) at most gap apart, evenly."""
    n = max(1, math.ceil(math.dist(a, c) / gap - 1e-9))
    return [(a[0] + (c[0] - a[0]) * k / n, a[1] + (c[1] - a[1]) * k / n) for k in range(n + 1)]


def post(b, q, z0: float, z1: float, w: float = POST_W, mat=STEEL) -> None:
    h = w / 2
    brlm.box(b, (q[0] - h, q[1] - h, z0), (q[0] + h, q[1] + h, z1), mat, skip=("-z",))


def viewing(b, p: dict, px) -> None:
    """The curb along the public edge, glass panes on it between steel posts, a rail on top."""
    g, line = p["g"], p["viewing"]
    if px == 20:                       # the curb and the glass, as one wall each
        brlm.wall_along(b, line, 0.0, g + CURB_H, CURB_T, CONCRETE, top=CONCRETE, ends=False)
        mid = brlm.offset_line(line, -CURB_T / 2)
        for a, c in zip(mid, mid[1:]):
            b.face([(*a, g + CURB_H), (*c, g + CURB_H), (*c, g + GLASS_TOP), (*a, g + GLASS_TOP)], GLASS)
        return
    brlm.wall_along(b, line, 0.0, g + CURB_H, CURB_T, CONCRETE, top=CONCRETE)
    mid = brlm.offset_line(line, -CURB_T / 2)
    posts = []
    for a, c in zip(mid, mid[1:]):
        pts = stations(a, c, POST_GAP)
        for q0, q1 in zip(pts, pts[1:]):
            b.face([(*q0, g + CURB_H), (*q1, g + CURB_H), (*q1, g + GLASS_TOP), (*q0, g + GLASS_TOP)], GLASS)
        posts += [q for q in pts if not posts or math.dist(q, posts[-1]) > 1e-6]
    for q in posts:
        post(b, q, g + CURB_H, g + GLASS_TOP)
    brlm.wall_along(b, brlm.offset_line(mid, POST_W / 2), g + GLASS_TOP, g + GLASS_TOP + RAIL_H, POST_W, STEEL,
                    top=STEEL)


def fence_runs(p: dict) -> list[list[tuple[float, float]]]:
    """The outline's stretches that are neither the viewing edge nor the night house's side, as lines."""
    x0, y0, x1, y1 = p["body"]
    m = HOUSE_MARGIN

    def near_house(q) -> bool:
        return x0 - m <= q[0] <= x1 + m and y0 - m <= q[1] <= y1 + m

    def near_view(q) -> bool:
        return min(_dist_to_segment(q, a, c) for a, c in zip(p["viewing"], p["viewing"][1:])) < 0.3

    ring = p["outline"]
    n = len(ring)
    fenced = [not ((near_view(ring[i]) and near_view(ring[(i + 1) % n])) or
                   (near_house(ring[i]) and near_house(ring[(i + 1) % n]))) for i in range(n)]
    if all(fenced):
        return [ring + ring[:1]]
    start = next(i for i in range(n) if not fenced[i - 1] and fenced[i]) if any(fenced) else 0
    runs, run = [], []
    for k in range(n):
        i = (start + k) % n
        if fenced[i]:
            run = run or [ring[i]]
            run.append(ring[(i + 1) % n])
        elif run:
            runs.append(run)
            run = []
    if run:
        runs.append(run)
    return runs


def _dist_to_segment(q, a, c) -> float:
    dx, dy = c[0] - a[0], c[1] - a[1]
    t = max(0.0, min(1.0, ((q[0] - a[0]) * dx + (q[1] - a[1]) * dy) / (dx * dx + dy * dy)))
    return math.dist(q, (a[0] + dx * t, a[1] + dy * t))


def fence(b, p: dict, px) -> None:
    """Steel posts, a top and a middle rail, and (at the full level) bars between them."""
    g = p["g"]
    for run in fence_runs(p):
        posts = []
        for a, c in zip(run, run[1:]):
            pts = stations(a, c, POST_GAP if px == 300 else 2 * POST_GAP)
            posts += [q for q in pts if not posts or math.dist(q, posts[-1]) > 1e-6]
            if px != 300:
                continue
            for q in stations(a, c, BAR_GAP)[1:-1]:                 # the bars, seen from both sides
                dx, dy = (c[0] - a[0]) / math.dist(a, c) * BAR_W / 2, (c[1] - a[1]) / math.dist(a, c) * BAR_W / 2
                bar = [(q[0] - dx, q[1] - dy, g + 0.1), (q[0] + dx, q[1] + dy, g + 0.1),
                       (q[0] + dx, q[1] + dy, g + FENCE_H - 0.05), (q[0] - dx, q[1] - dy, g + FENCE_H - 0.05)]
                b.face(bar, STEEL, double=True)
        for q in posts:
            post(b, q, 0.0, g + FENCE_H)
        for z in (g + FENCE_H / 2, g + FENCE_H - RAIL_H):
            brlm.wall_along(b, run, z, z + RAIL_H, 0.06, STEEL, top=STEEL)


# ------------------------------------------------------------------ the part

def habitat(p: dict, px) -> brlm.Builder:
    b = brlm.Builder()
    night_house(b, p, px)
    for i, (x, y, r, h) in enumerate(p["rocks"]):
        rock(b, p, x, y, r, h, ROCK_SIDES[px], i)
    if px != 20:
        falls(b, p)
    stream(b, p, px)
    pool(b, p, px)
    viewing(b, p, px)
    if px != 20:
        fence(b, p, px)
    return b


def habitat_collision(p: dict) -> brlm.Builder:
    b = brlm.Builder()
    g = p["g"]
    brlm.wall_along(b, p["viewing"], 0.0, g + GLASS_TOP, CURB_T, CONCRETE, top=CONCRETE)
    for run in fence_runs(p):
        brlm.wall_along(b, run, 0.0, g + FENCE_H, 0.1, STEEL, top=STEEL)
    night_house_collision(b, p)
    for i, (x, y, r, h) in enumerate(p["rocks"]):
        rock(b, p, x, y, r, h, ROCK_SIDES["collision"], i, collision=True)
    return b


def parts(p: dict) -> dict[str, dict]:
    """{part: {"node": its nodes' name, "lods": [(px, triangle budget, Builder), ...],
    "collision": (budget, Builder)}}."""
    return {"main": {"node": "habitat", "lods": [(px, BUDGETS[px], habitat(p, px)) for px in LODS],
                     "collision": (COLLISION_BUDGET, habitat_collision(p))}}


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
