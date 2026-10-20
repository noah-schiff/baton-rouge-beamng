"""The Louisiana State Capitol (plan Task 18): the 34-storey Art Deco tower over the centre block and
the two wings, and the grand front steps, in pale limestone.

Params (out/landmarks.json): height_m (the top: the published 450 ft), wings_m and centre_m (the
wings' and the centre block's roofs), setbacks ([[from_m, width_m, depth_m], ...]: from each height
up, the tower's section, measured on the lidar), footprint_local and ground_dz. The model's +u is
the footprint's long axis (2.9 degrees north of east) and +v north. Heights are above the ground
(added to ground_dz); walls reach down to z 0, so the model never floats.

* The wings: the whole footprint, its walls to wings_m under a flat roof.
* The centre block: CENTRE_BLOCK (the span of the footprint's north recess, to the lidar's
  26-30 m roofs south of the tower), from the wings' roof to centre_m.
* The tower, centred on TOWER_AT (the lidar's): its base (TOWER_BASE, the lidar's 40-55 m
  shoulders round the shaft) from the centre block's roof to the first setback, each setback's
  section up to the next, then the octagonal lantern (LANTERN) and its cap to height_m. A corner
  pylon stands at each corner of the shaft's top (the first setback's section).
* Every wall in limestone (br_lm_limestone) is a row of bays: limestone ribs (pilasters) between
  vertical window bands, the bands dark glass with a spandrel at each floor (FLOOR), as vertex
  colours; a plain band at the top of each wall and a plinth at the foot of the wings.
* The grand steps: STEPS steps (one per state when it was built) in a FLIGHT_W-wide flight down
  from a landing at the facade, south (-v), centred on the footprint, between two cheek walls with
  a pedestal at each foot. The flight's top and foot (STEPS_TOP, STEPS_FOOT, above the ground) sit
  just over the terrain, which the DEM already raises along the real steps.

Levels of detail: capitol_a600 (ribs proud of the walls, the spandrels; at most 14,000
triangles), _a150 (the bays drawn flat, fewer steps; 3,000), _a30 (the massing, the lantern and
the cap: it keeps the tower in sight across downtown; 300), and Colmesh-1 (the footprint's
prism, the centre block, the tower's boxes, the steps as a ramp and the pedestals; 200).
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

import brexport  # noqa: E402
import brlm  # noqa: E402

SHAPE = "capitol"
LIMESTONE = "br_lm_limestone"
TILE = 3.6                      # m per texture repeat
GLASS = (0.18, 0.20, 0.24)      # the window bands' glass ...
SPANDREL = (0.50, 0.46, 0.41)   # ... and the carved panel at each floor between them
BAND = tuple(round(0.56 * g + 0.44 * s, 3) for g, s in zip(GLASS, SPANDREL))   # both, seen from afar (a150)

# ------------------------------------------------------------------ the massing (model coordinates, m)
# Measured on the full map's lidar (data/out/ndsm.npy, 2 m cells) round the 2026-10-19 resolve:
TOWER_AT = (0.25, -1.5)         # the shaft's centre: its 60-100 m returns span u -11.75..12.25, v -12.5..9.5
TOWER_BASE = (28.0, 24.0)       # the 40-55 m shoulders round the shaft: u -13.75..14.25, v -13.5..10.5
CENTRE_BLOCK = (-18.3, -14.08, 18.85, 14.5)   # u0, v0, u1, v1: the north recess's span (the footprint), the
#                                               lidar's 26-30 m roofs to v 14.5 north of the tower
LANTERN = (123.0, 131.0)        # the lantern, above the ground: over the last 6 m section (whose top the
#                                 lidar clips at 123 m) to the foot of the cap
LANTERN_R = 2.6                 # the octagon's apothem
CAP = ((2.6, 0.0), (2.25, 0.35), (1.4, 0.7))   # the cap's rings: (apothem, share of the way to height_m)
PYLON, PYLON_H = 3.5, 4.0       # the corner pylons on the shaft's top: side, height

# ------------------------------------------------------------------ the walls
BAY, RIB_W, RIB_OUT = 3.0, 1.3, 0.35   # bay pitch, a rib's width and how far it stands out
FLOOR = 3.9                     # m per storey: 34 storeys in the 137 m tower
GLASS_BAND = (0.9, 3.1)         # within a floor, the glass; the spandrel between
TOP_BAND, PLINTH = 1.5, 1.2     # plain limestone at the top of every wall, at the foot of the wings
MIN_BAYED = 2.5                 # a wall shorter than this is plain

# ------------------------------------------------------------------ the grand steps
STEPS = 48                      # one per state when the Capitol was built (1932)
FLIGHT_W = 30.0                 # the flight's width, from the aerial
LANDING = 3.0                   # the landing at the top, before the entrance
TREAD = 0.48
STEPS_TOP, STEPS_FOOT = 6.0, 1.45   # above the ground (ground_dz): the terrain (heights.npy) is 7.9-8.0 m over
#                                     the base at the facade and 3.4 m at the foot, 2.25 m over at ground_dz
CHEEK_W, CHEEK_UP = 1.5, 1.0    # the cheek walls: width, height over the steps' line
PEDESTAL = (4.0, 5.0, 4.5)      # the sculptures' pedestals at the foot: across, along, over the foot
PEDESTAL_GAP = 0.3              # from the flight's edge (the cheek's inner face stays clear of it)
MID_STEP = 3                    # the mid level draws one step for this many

BUDGETS = {600: 14000, 150: 3000, 30: 300}
COLLISION_BUDGET = 200


# ------------------------------------------------------------------ the params

def resolve(params: dict) -> dict:
    """The dimensions the builders share, from the script's params."""
    g = float(params.get("ground_dz", 0.0))
    foot = [tuple(map(float, q)) for q in params["footprint_local"]]
    setbacks = [tuple(map(float, s)) for s in params["setbacks"]]
    top = g + float(params["height_m"])
    centre = g + float(params["centre_m"])
    sections = [(centre, g + setbacks[0][0], *TOWER_BASE)]
    for (start, w, d), nxt in zip(setbacks, setbacks[1:] + [(LANTERN[0], 0.0, 0.0)]):
        sections.append((g + start, g + nxt[0], w, d))
    us = [u for u, _ in foot]
    uc = (min(us) + max(us)) / 2
    front = [v for u, v in foot if abs(u - uc) <= FLIGHT_W / 2 and v < 0.0]   # the south facade's corners
    v_top = min(front)
    steps = {"u": uc, "width": FLIGHT_W, "v_top": v_top, "v_back": max(front), "z_top": g + STEPS_TOP,
             "z_foot": g + STEPS_FOOT, "v_foot": v_top - LANDING - STEPS * TREAD}
    return {"g": g, "top": top, "wings": g + float(params["wings_m"]), "centre": centre, "footprint": foot,
            "sections": sections, "tower_at": TOWER_AT, "centre_block": CENTRE_BLOCK,
            "lantern": (g + LANTERN[0], g + LANTERN[1], LANTERN_R), "steps": steps}


# ------------------------------------------------------------------ faces

def facing(b, pts, normal, col=None) -> None:
    """A planar convex face, its corners put counter-clockwise as seen from `normal`'s side."""
    if brlm.dot(brlm.normal(pts), normal) < 0:
        pts = pts[::-1]
    b.face(pts, LIMESTONE, TILE, col=col)


def wall(b, a, c, z0: float, z1: float, px, plinth: float = 0.0) -> None:
    """One wall from (u, v) a to c, facing to the right of a -> c (out of a counter-clockwise
    ring), from z0 to z1: at a600 and a150 a row of bays (ribs between window bands, the plain
    TOP_BAND at the top and the plinth at the foot), the a600's ribs RIB_OUT proud and its bands
    split into glass and spandrels per FLOOR; at a30, and when too short or low, one face."""
    du, dv = c[0] - a[0], c[1] - a[1]
    length = math.hypot(du, dv)
    if length < 1e-6 or z1 <= z0:
        return
    t = (du / length, dv / length)
    n = (t[1], -t[0], 0.0)

    def P(s: float, z: float, out: float = 0.0):
        return (a[0] + t[0] * s + n[0] * out, a[1] + t[1] * s + n[1] * out, z)

    def quad(s0, s1, za, zb, out=0.0, col=None, normal=n):
        if s1 - s0 > 1e-6 and zb - za > 1e-6:
            facing(b, [P(s0, za, out), P(s1, za, out), P(s1, zb, out), P(s0, zb, out)], normal, col)

    bays = round((length - RIB_W) / BAY)
    w0, w1 = z0 + plinth, z1 - TOP_BAND
    if px == 30 or length < MIN_BAYED or bays < 1 or w1 - w0 < FLOOR / 2:
        quad(0.0, length, z0, z1)
        return
    window = (length - (bays + 1) * RIB_W) / bays
    quad(0.0, length, z0, w0)
    quad(0.0, length, w1, z1)
    for k in range(bays):
        s0 = RIB_W + k * (RIB_W + window)
        if px == 600:
            band(quad, s0, s0 + window, w0, w1)
        else:
            quad(s0, s0 + window, w0, w1, col=BAND)
    for k in range(bays + 1):
        s0 = k * (RIB_W + window)
        s1 = s0 + RIB_W
        if px == 600:                                        # the rib stands out: front, sides, top
            quad(s0, s1, z0, z1, RIB_OUT)
            facing(b, [P(s0, z0), P(s0, z0, RIB_OUT), P(s0, z1, RIB_OUT), P(s0, z1)], (-t[0], -t[1], 0.0))
            facing(b, [P(s1, z0), P(s1, z0, RIB_OUT), P(s1, z1, RIB_OUT), P(s1, z1)], (t[0], t[1], 0.0))
            facing(b, [P(s0, z1), P(s1, z1), P(s1, z1, RIB_OUT), P(s0, z1, RIB_OUT)], (0.0, 0.0, 1.0))
        else:
            quad(s0, s1, w0, w1)


def band(quad, s0: float, s1: float, z0: float, z1: float) -> None:
    """A window band from z0 to z1: glass in GLASS_BAND of every FLOOR, spandrels between."""
    z, last = z0, z0
    while z < z1 - 1e-6:
        g0, g1 = min(z + GLASS_BAND[0], z1), min(z + GLASS_BAND[1], z1)
        quad(s0, s1, last, g0, col=SPANDREL)
        quad(s0, s1, g0, g1, col=GLASS)
        last, z = g1, z + FLOOR
    quad(s0, s1, last, z1, col=SPANDREL)


def ring_walls(b, ring, z0: float, z1: float, px, plinth: float = 0.0) -> None:
    for a, c in zip(ring, ring[1:] + ring[:1]):
        wall(b, a, c, z0, z1, px, plinth)


def rect(cu: float, cv: float, w: float, d: float):
    """The corners of a w x d rectangle round (cu, cv), counter-clockwise."""
    return [(cu - w / 2, cv - d / 2), (cu + w / 2, cv - d / 2), (cu + w / 2, cv + d / 2), (cu - w / 2, cv + d / 2)]


def roof(b, ring, z: float) -> None:
    brlm.fill(b, ring, z, LIMESTONE, TILE)


def octagon(r: float, z: float):
    """The octagon of apothem r at height z, centred on the tower, counter-clockwise, its faces facing
    every 45 degrees from +u."""
    cu, cv = TOWER_AT
    rc = r / math.cos(math.pi / 8)
    return [(cu + rc * math.cos(math.radians(22.5 + 45 * k)), cv + rc * math.sin(math.radians(22.5 + 45 * k)), z)
            for k in range(8)]


# ------------------------------------------------------------------ the parts of the building

def massing(b, p: dict, px) -> None:
    """The wings, the centre block and the tower's sections, their walls and roofs."""
    ring = p["footprint"]
    ring_walls(b, ring, 0.0, p["wings"], px, plinth=PLINTH)
    roof(b, ring, p["wings"])
    u0, v0, u1, v1 = p["centre_block"]
    block = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
    ring_walls(b, block, p["wings"], p["centre"], px)
    roof(b, block, p["centre"])
    for z0, z1, w, d in p["sections"]:
        r = rect(*TOWER_AT, w, d)
        ring_walls(b, r, z0, z1, px)
        roof(b, r, z1)
    if px != 30:                                            # the pylons on the shaft's corners
        z0, w, d = p["sections"][1][1], p["sections"][1][2], p["sections"][1][3]
        cu, cv = TOWER_AT
        for su in (-1, 1):
            for sv in (-1, 1):
                c = (cu + su * (w - PYLON) / 2, cv + sv * (d - PYLON) / 2)
                brlm.box(b, (c[0] - PYLON / 2, c[1] - PYLON / 2, z0), (c[0] + PYLON / 2, c[1] + PYLON / 2, z0 + PYLON_H),
                         LIMESTONE, TILE, skip=("-z",))


def lantern(b, p: dict, px) -> None:
    """The octagonal lantern, its openings dark (a600 and a150), and the cap to the top."""
    z0, z1, r = p["lantern"]
    lo, hi = octagon(r, z0), octagon(r, z1)
    for k in range(8):
        a0, a1, b1, b0 = lo[k], lo[(k + 1) % 8], hi[(k + 1) % 8], hi[k]
        if px == 30:
            b.face([a0, a1, b1, b0], LIMESTONE, TILE)
            continue
        s = 0.22                                            # the piers' share of a side
        h0, h1 = z0 + 1.2, z1 - 1.0                         # under and over the openings

        def at(f, z):
            return (a0[0] + (a1[0] - a0[0]) * f, a0[1] + (a1[1] - a0[1]) * f, z)

        b.face([at(0, z0), at(1, z0), at(1, h0), at(0, h0)], LIMESTONE, TILE)
        b.face([at(0, h1), at(1, h1), at(1, z1), at(0, z1)], LIMESTONE, TILE)
        b.face([at(0, h0), at(s, h0), at(s, h1), at(0, h1)], LIMESTONE, TILE)
        b.face([at(1 - s, h0), at(1, h0), at(1, h1), at(1 - s, h1)], LIMESTONE, TILE)
        b.face([at(s, h0), at(1 - s, h0), at(1 - s, h1), at(s, h1)], LIMESTONE, TILE, col=GLASS)
    rings = [octagon(apothem, z1 + share * (p["top"] - z1)) for apothem, share in CAP]
    brlm.loft(b, rings, LIMESTONE, TILE, closed=True, flip=True)
    apex = (*TOWER_AT, p["top"])
    for k in range(8):
        b.face([rings[-1][k], rings[-1][(k + 1) % 8], apex], LIMESTONE, TILE)


def steps(b, p: dict, px) -> None:
    """The grand steps: a landing from the facade (into the entrance's recess, v_back), the flight
    down to the south, its cheek walls and the pedestals at their feet (a600, a150); at a30, the
    landing and one sloped face."""
    s = p["steps"]
    u0, u1 = s["u"] - s["width"] / 2, s["u"] + s["width"] / 2
    v_land = s["v_top"] - LANDING
    count = STEPS if px == 600 else STEPS // MID_STEP
    rise, tread = (s["z_top"] - s["z_foot"]) / STEPS * (STEPS / count), TREAD * (STEPS / count)
    facing(b, [(u0, v_land, s["z_top"]), (u1, v_land, s["z_top"]), (u1, s["v_back"], s["z_top"]),
               (u0, s["v_back"], s["z_top"])], (0.0, 0.0, 1.0))
    if px == 30:
        foot = s["v_foot"]
        facing(b, [(u0, foot, s["z_foot"]), (u1, foot, s["z_foot"]), (u1, v_land, s["z_top"]),
                   (u0, v_land, s["z_top"])], (0.0, -0.5, 1.0))
        facing(b, [(u0, foot, 0.0), (u1, foot, 0.0), (u1, foot, s["z_foot"]), (u0, foot, s["z_foot"])], (0.0, -1.0, 0.0))
        return
    for k in range(count):
        v = v_land - k * tread                              # the riser's foot line
        z_hi, z_lo = s["z_top"] - k * rise, s["z_top"] - (k + 1) * rise
        bottom = 0.0 if k == count - 1 else z_lo            # the last riser goes down to the ground
        facing(b, [(u0, v, bottom), (u1, v, bottom), (u1, v, z_hi), (u0, v, z_hi)], (0.0, -1.0, 0.0))
        facing(b, [(u0, v - tread, z_lo), (u1, v - tread, z_lo), (u1, v, z_lo), (u0, v, z_lo)], (0.0, 0.0, 1.0))
    foot = v_land - count * tread
    for side in (-1, 1):                                    # the cheek walls, their tops along the steps
        inner = s["u"] + side * s["width"] / 2
        outer = inner + side * CHEEK_W
        top_hi, top_lo = s["z_top"] + CHEEK_UP, s["z_foot"] + CHEEK_UP
        for x, n in ((inner, -side), (outer, side)):
            facing(b, [(x, s["v_top"], 0.0), (x, foot, 0.0), (x, foot, top_lo), (x, v_land, top_hi),
                       (x, s["v_top"], top_hi)], (float(n), 0.0, 0.0))
        facing(b, [(inner, s["v_top"], top_hi), (inner, v_land, top_hi), (outer, v_land, top_hi),
                   (outer, s["v_top"], top_hi)], (0.0, 0.0, 1.0))
        facing(b, [(inner, v_land, top_hi), (inner, foot, top_lo), (outer, foot, top_lo), (outer, v_land, top_hi)],
               (0.0, -0.3, 1.0))
    for lo, hi in pedestals(p):
        brlm.box(b, lo, hi, LIMESTONE, TILE, skip=("-z",))


def pedestals(p: dict):
    """(lo, hi) corners of the two pedestals at the foot of the cheek walls."""
    s = p["steps"]
    across, along, high = PEDESTAL
    out = []
    for side in (-1, 1):
        inner = s["u"] + side * s["width"] / 2
        a, c = sorted((inner + side * PEDESTAL_GAP, inner + side * (PEDESTAL_GAP + across)))
        out.append(((a, s["v_foot"], 0.0), (c, s["v_foot"] + along, s["z_foot"] + high)))
    return out


# ------------------------------------------------------------------ levels of detail

def level(p: dict, px) -> brlm.Builder:
    b = brlm.Builder()
    massing(b, p, px)
    lantern(b, p, px)
    steps(b, p, px)
    return b


def collision(p: dict) -> brlm.Builder:
    b = brlm.Builder()
    ring = p["footprint"]
    brlm.extrude_ring(b, ring, 0.0, p["wings"], LIMESTONE)
    roof(b, ring, p["wings"])
    u0, v0, u1, v1 = p["centre_block"]
    brlm.box(b, (u0, v0, p["wings"]), (u1, v1, p["centre"]), LIMESTONE, skip=("-z",))
    cu, cv = TOWER_AT
    for z0, z1, w, d in p["sections"]:
        brlm.box(b, (cu - w / 2, cv - d / 2, z0), (cu + w / 2, cv + d / 2, z1), LIMESTONE, skip=("-z",))
    z0, _, r = p["lantern"]
    brlm.box(b, (cu - r, cv - r, z0), (cu + r, cv + r, p["top"]), LIMESTONE, skip=("-z",))
    s = p["steps"]
    u0, u1, v_land = s["u"] - s["width"] / 2 - CHEEK_W, s["u"] + s["width"] / 2 + CHEEK_W, s["v_top"] - LANDING
    ramp = [(u0, s["v_foot"], s["z_foot"]), (u1, s["v_foot"], s["z_foot"]), (u1, v_land, s["z_top"]),
            (u0, v_land, s["z_top"])]
    b.face(ramp, LIMESTONE)
    zt, zf, vt, vf = s["z_top"], s["z_foot"], s["v_top"], s["v_foot"]
    b.face([(u0, v_land, zt), (u1, v_land, zt), (u1, vt, zt), (u0, vt, zt)], LIMESTONE)           # the landing
    b.face([(u0, vf, 0.0), (u1, vf, 0.0), (u1, vf, zf), (u0, vf, zf)], LIMESTONE)                # the foot
    for x, n in ((u0, -1.0), (u1, 1.0)):
        facing(b, [(x, vt, 0.0), (x, vf, 0.0), (x, vf, zf), (x, v_land, zt), (x, vt, zt)], (n, 0.0, 0.0))
    for lo, hi in pedestals(p):
        brlm.box(b, lo, hi, LIMESTONE, skip=("-z",))
    return b


def parts(p: dict) -> dict[str, dict]:
    """{part: {"node": its nodes' name, "lods": [(px, triangle budget, Builder), ...],
    "collision": (budget, Builder)}}."""
    return {"main": {"node": SHAPE, "lods": [(px, BUDGETS[px], level(p, px)) for px in (600, 150, 30)],
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
