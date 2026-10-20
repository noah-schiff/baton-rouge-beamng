"""The Pete Maravich Assembly Center at LSU (plan Task 16): the folded concrete dome on its ring
wall, the glazed skirt and the podium round it with the entrances, and the north annex.

Params (out/landmarks.json, from landmarks.resolve): diameter_m (the ring wall), wall_m and
crown_m (the ring wall's top under the ribs and the dome's top, above the ground), ribs and
rib_az_deg (how many folds, and the compass bearing of one), annex_m (the annex's roof),
entrances_az_deg (the entrances' compass bearings), footprint_local and ground_dz. The model's +u
is east and +v north (the registry's yaw, azimuth 90), so a compass bearing b is the model angle
90 - b; the origin is the dome's centre. Heights below are above the ground (added to ground_dz)
and walls reach down to z 0, so the model never floats.

Two shape parts, each under validate.DAE_MAX_VERTS vertices per level of detail:
* main: the dome, its profile wall + (crown - wall) * (1 - rho^2)^DOME_EXP (rho = r / its
  radius; DOME_EXP fitted to the lidar), folded: a ridge on every rib, a valley FOLD * rho lower
  between them, its crown darker (the skylight); the ring wall from the skirt up to the dome's
  scalloped edge; a skirt of glazing sloping from SKIRT_DROP under the valleys out and down to
  the podium (BUTTRESS_OUT further out); a concrete fin under each rib from the wall's top down
  the skirt; the podium's roof out to PODIUM_OUT and its wall (PODIUM_H) with a band of windows;
  at each entrance, doors in that band and a canopy on two steel posts.
* annex: the footprint outside the podium (the annex the OSM outline joins to the dome), its
  walls to annex_m with two bands of windows, and a flat roof.
Levels of detail: pmac_a600, _a150 and _a40 and annex_a600, _a150 and _a40, and each part's
Colmesh-1: the podium's wall and roof; the annex's walls and roof.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

import brexport  # noqa: E402
import brlm  # noqa: E402

CONCRETE, STEEL = "br_lm_concrete", "br_lm_steel"
TILE = 3.0                      # m per texture repeat
DARK = (0.13, 0.14, 0.16)       # doors and the skylight
GLAZING = (0.32, 0.38, 0.44)    # tinted glass: the skirt and the window bands

# ------------------------------------------------------------------ the dome (m)
# The lidar's median dome heights above the ground (out_lsu ndsm, 2026-10-19) are 29.1, 27.1, 24.1
# and 19.8 m at r 20, 30, 40 and 50: (1 - rho^2)^0.7 between the wall's 16 m and the crown's 30.6
# fits them within 0.6 m (a sphere misses by 2 m).
DOME_EXP = 0.7
FOLD = 1.5                      # a valley's depth at the dome's edge, fading to nothing at the crown
SKIRT_DROP = 1.5                # the skirt meets the ring wall this far under the valleys (13 m)
BUTTRESS_OUT = 7.0              # the fins and the skirt reach r 60: the rib feet on NAIP 2023 end there
BUTTRESS_T = 0.9
PODIUM_OUT = 63.0               # the lidar's 4.7 m ring reaches r 62-64 all round
PODIUM_H = 4.7
WINDOWS = (0.6, 3.4)            # the podium wall's band of windows, over the ground
ENTRANCE_W, ENTRANCE_D, ENTRANCE_H = 14.0, 5.0, 3.6    # a canopy: width, depth out from the wall, underside
CANOPY_T, POST = 0.4, 0.3
ANNEX_BANDS = ((1.2, 2.8), (4.8, 6.4))   # the annex's window bands, over the ground
MIN_ANNEX = 50.0                # m2: a smaller piece of the footprint outside the podium is left out
# per level of detail: the dome's rings (rho); whether it is folded (else a corner on every third
# rib); the podium's corners round (2: a rib and a valley, 1: the ribs, 1/3: every third rib) and
# whether its wall has the windows; the fins ("full", "top": their sloped tops only, None) and the
# entrances ("full", "canopy": no posts, None); the arc step (deg) where the annex meets the podium
RINGS = {600: (0.12, 0.24, 0.36, 0.48, 0.6, 0.7, 0.8, 0.88, 0.95, 1.0), 150: (0.12, 0.4, 0.7, 1.0), 40: (0.12, 0.6, 1.0)}
FOLDED = {600: True, 150: True, 40: False}
PODIUM_CORNERS = {600: 2, 150: 1, 40: 1 / 3}
WINDOW_BAND = {600: True, 150: True, 40: False}
FINS = {600: "full", 150: "top", 40: None}
ENTRANCES = {600: "full", 150: "canopy", 40: None}
ARC_STEP = {600: 3.0, 150: 6.0, 40: 15.0, "collision": 15.0}

BUDGETS = {"main": {600: 3700, 150: 1700, 40: 200}, "annex": {600: 360, 150: 70, 40: 50}}   # ~20 % over the LSU build
COLLISION_BUDGETS = {"main": 220, "annex": 50}


# ------------------------------------------------------------------ the params

def resolve(params: dict) -> dict:
    """The dimensions the builders share, from the script's params."""
    g = float(params.get("ground_dz", 0.0))
    z_wall = g + float(params["wall_m"])
    return {"g": g, "R": float(params["diameter_m"]) / 2, "z_wall": z_wall, "z_crown": g + float(params["crown_m"]),
            "z_skirt": z_wall - FOLD - SKIRT_DROP, "z_podium": g + PODIUM_H,
            "ribs": int(params["ribs"]), "rib0": 90.0 - float(params.get("rib_az_deg", 0.0)),
            "entrances": [(90.0 - float(b)) % 360.0 for b in params.get("entrances_az_deg") or []],
            "z_annex": g + float(params["annex_m"]),
            "footprint": [tuple(map(float, q)) for q in params["footprint_local"]]}


def rib_angles(p: dict) -> list[float]:
    """The ribs' model angles (deg counter-clockwise from +u), counter-clockwise from rib_az_deg's."""
    return [(p["rib0"] + 360.0 * k / p["ribs"]) % 360.0 for k in range(p["ribs"])]


def dome_z(p: dict, rho: float) -> float:
    """The dome's height on a rib at rho (its radius's share)."""
    return p["z_wall"] + (p["z_crown"] - p["z_wall"]) * max(0.0, 1.0 - rho * rho) ** DOME_EXP


def corners(p: dict, px) -> list[tuple[float, bool]]:
    """(model angle, on a rib) round the dome at a level of detail: a rib and the valley after it,
    or (unfolded) every third rib."""
    ribs = rib_angles(p)
    if FOLDED.get(px, False):
        half = 180.0 / p["ribs"]
        return [c for a in ribs for c in ((a, True), ((a + half) % 360.0, False))]
    return [(a, True) for a in ribs[::3]]


def polar(r: float, a: float, z: float) -> tuple[float, float, float]:
    t = math.radians(a)
    return r * math.cos(t), r * math.sin(t), z


def frame(a: float):
    """P(r, z, t): the point r out along the model angle a, t across it (counter-clockwise), at z."""
    c, s = math.cos(math.radians(a)), math.sin(math.radians(a))

    def P(r: float, z: float, t: float = 0.0):
        return r * c - t * s, r * s + t * c, z

    return P, (c, s, 0.0), (-s, c, 0.0)


# ------------------------------------------------------------------ the dome, the ring and the podium

def dome(b, p: dict, px) -> None:
    cs = corners(p, px)
    rings = [[polar(rho * p["R"], a, dome_z(p, rho) - (0.0 if rib else FOLD * rho)) for a, rib in cs]
             for rho in RINGS[px]]                              # inner to outer, so the quads face up
    brlm.loft(b, rings, CONCRETE, TILE, closed=True)
    inner, apex = rings[0], (0.0, 0.0, p["z_crown"])
    for k in range(len(inner)):                                 # the crown and its skylight
        b.face([inner[k], inner[(k + 1) % len(inner)], apex], CONCRETE, TILE, col=DARK)


def ring_wall(b, p: dict, px) -> None:
    """From the skirt up to the dome's scalloped edge, facing out."""
    cs = corners(p, px)
    rim = [polar(p["R"], a, p["z_wall"] - (0.0 if rib else FOLD)) for a, rib in cs]
    foot = [polar(p["R"], a, p["z_skirt"]) for a, _ in cs]
    brlm.loft(b, [foot, rim], CONCRETE, TILE, closed=True, flip=True)


def at_entrance(p: dict, a0: float, a1: float) -> bool:
    """Whether the podium wall between the model angles a0 and a1 is behind an entrance canopy."""
    mid = (a0 + ((a1 - a0) % 360.0) / 2) % 360.0
    half = math.degrees(ENTRANCE_W / 2 / PODIUM_OUT)
    return any(abs((mid - e + 180.0) % 360.0 - 180.0) < half for e in p["entrances"])


def podium_angles(p: dict, px) -> list[float]:
    share = PODIUM_CORNERS[px]
    if share == 2:
        return [a for a, _ in corners(p, 600)]
    return rib_angles(p)[::round(1 / share)]


def podium(b, p: dict, px) -> None:
    """The skirt, the podium's roof (the skirt reaches the wall at a40) and its wall, with the
    windows (doors at the entrances) where the level has them."""
    angles = podium_angles(p, px)
    out = p["R"] + BUTTRESS_OUT if px != 40 else PODIUM_OUT
    ring = lambda r, z: [polar(r, a, z) for a in angles]   # noqa: E731
    brlm.loft(b, [ring(p["R"], p["z_skirt"]), ring(out, p["z_podium"])], CONCRETE, TILE, closed=True, col=GLAZING)
    if out < PODIUM_OUT:
        brlm.loft(b, [ring(out, p["z_podium"]), ring(PODIUM_OUT, p["z_podium"])], CONCRETE, TILE, closed=True)
    g, (w0, w1) = p["g"], WINDOWS
    for k, a0 in enumerate(angles):
        a1 = angles[(k + 1) % len(angles)]
        bands = ((0.0, g + w0, None), (g + w0, g + w1, DARK if at_entrance(p, a0, a1) else GLAZING),
                 (g + w1, p["z_podium"], None)) if WINDOW_BAND[px] else ((0.0, p["z_podium"], None),)
        for z0, z1, col in bands:
            b.face([polar(PODIUM_OUT, a0, z0), polar(PODIUM_OUT, a1, z0), polar(PODIUM_OUT, a1, z1),
                    polar(PODIUM_OUT, a0, z1)], CONCRETE, TILE, col=col)


def fins(b, p: dict, sides: bool = True) -> None:
    """A fin under each rib: from the ring wall's top down the skirt to its foot; without sides,
    its sloped top only."""
    h, out = BUTTRESS_T / 2, p["R"] + BUTTRESS_OUT
    for a in rib_angles(p):
        P, _, _ = frame(a)
        top, foot, low = (p["R"], p["z_wall"]), (out, p["z_podium"]), (p["R"], p["z_skirt"])
        if sides:
            b.face([P(*low, h), P(*top, h), P(*foot, h)], CONCRETE, TILE)
            b.face([P(*low, -h), P(*foot, -h), P(*top, -h)], CONCRETE, TILE)
        b.face([P(*top, -h), P(*foot, -h), P(*foot, h), P(*top, h)], CONCRETE, TILE)


def entrances(b, p: dict, posts: bool = True) -> None:
    """A canopy slab at each entrance, out from the podium's wall, on two steel posts."""
    zc = p["g"] + ENTRANCE_H
    for a in p["entrances"]:
        P, er, et = frame(a)
        r0 = PODIUM_OUT - 0.5                                   # into the wall, past its curve
        brlm.oriented_box(b, P((r0 + PODIUM_OUT + ENTRANCE_D) / 2, zc + CANOPY_T / 2),
                          (er, et, (0.0, 0.0, 1.0)), ((PODIUM_OUT + ENTRANCE_D - r0) / 2, ENTRANCE_W / 2, CANOPY_T / 2),
                          CONCRETE, TILE)
        for t in (-ENTRANCE_W / 2 + 0.6, ENTRANCE_W / 2 - 0.6) if posts else ():
            brlm.oriented_box(b, P(PODIUM_OUT + ENTRANCE_D - 0.4, zc / 2, t), (er, et, (0.0, 0.0, 1.0)),
                              (POST / 2, POST / 2, zc / 2), STEEL)


def main_part(p: dict, px) -> brlm.Builder:
    b = brlm.Builder()
    dome(b, p, px)
    ring_wall(b, p, px)
    podium(b, p, px)
    if FINS[px]:
        fins(b, p, sides=FINS[px] == "full")
    if ENTRANCES[px]:
        entrances(b, p, posts=ENTRANCES[px] == "full")
    return b


def main_collision(p: dict) -> brlm.Builder:
    """The podium's wall and its roof, round the ribs."""
    angles = rib_angles(p)
    b = brlm.Builder()
    brlm.loft(b, [[polar(PODIUM_OUT, a, 0.0) for a in angles], [polar(PODIUM_OUT, a, p["z_podium"]) for a in angles]],
              CONCRETE, closed=True, flip=True)
    brlm.loft(b, [[polar(p["R"], a, p["z_podium"]) for a in angles],
                  [polar(PODIUM_OUT, a, p["z_podium"]) for a in angles]], CONCRETE, closed=True)
    return b


# ------------------------------------------------------------------ the annex

def _crossings(a, c, r: float) -> list[float]:
    """Where the segment a -> c crosses the circle of radius r round the origin: t in (0, 1), sorted."""
    dx, dy = c[0] - a[0], c[1] - a[1]
    qa, qb, qc = dx * dx + dy * dy, 2 * (a[0] * dx + a[1] * dy), a[0] ** 2 + a[1] ** 2 - r * r
    disc = qb * qb - 4 * qa * qc
    if qa < 1e-12 or disc <= 0:
        return []
    root = math.sqrt(disc)
    return sorted(t for t in ((-qb - root) / (2 * qa), (-qb + root) / (2 * qa)) if 1e-9 < t < 1 - 1e-9)


def outside(ring, r: float, step_deg: float = 3.0) -> list[list[tuple[float, float]]]:
    """The parts of a counter-clockwise ring [(x, y), ...] outside the circle of radius r round the
    origin, each counter-clockwise and closed along the circle (an arc in steps of at most
    step_deg). ValueError when the circle lies inside the ring without crossing it (a hole)."""
    pts = [(float(x), float(y)) for x, y in ring]
    n = len(pts)

    def at(i: int, t: float) -> tuple[float, float]:
        (ax, ay), (cx, cy) = pts[i], pts[(i + 1) % n]
        return ax + (cx - ax) * t, ay + (cy - ay) * t

    cuts = [(i, t) for i in range(n) for t in _crossings(pts[i], pts[(i + 1) % n], r)]
    if not cuts:
        if all(math.hypot(*q) <= r for q in pts):
            return []
        if brlm._inside(pts, 0.0, 0.0):
            raise ValueError("outside: the circle lies inside the ring (a hole)")
        return [pts]
    # walk the ring once from where it first goes out of the circle
    i0, t0 = next((i, t) for i, t in cuts if math.hypot(*at(i, min(t + 1e-6, 1.0))) > r)
    parts, run = [], None
    for k in range(n + 1):
        i = (i0 + k) % n
        ts = _crossings(pts[i], pts[(i + 1) % n], r)
        ts = [t for t in ts if t >= t0] if k == 0 else [t for t in ts if t < t0] if k == n else ts
        for t in ts:
            if run is None:                            # going out
                run = [at(i, t)]
            else:                                      # coming back in: close along the circle
                parts.append(_close(run + [at(i, t)], r, step_deg))
                run = None
        if run is not None and k < n:
            run.append(pts[(i + 1) % n])
    return [q for q in parts if abs(brlm._signed_area(q)) > 1e-6]


def _close(run, r: float, step_deg: float):
    """A run outside the circle (out at run[0], back in at run[-1]) closed by the arc between them,
    clockwise round the circle from where it came back in to where it went out."""
    a_in = math.degrees(math.atan2(run[-1][1], run[-1][0]))
    a_out = math.degrees(math.atan2(run[0][1], run[0][0]))
    sweep = (a_in - a_out) % 360.0
    steps = max(1, math.ceil(sweep / step_deg))
    arc = [polar(r, a_in - sweep * k / steps, 0.0)[:2] for k in range(1, steps)]
    return list(run) + arc


def annex_rings(p: dict, level) -> list:
    return [q for q in outside(p["footprint"], PODIUM_OUT, ARC_STEP[level])
            if abs(brlm._signed_area(q)) >= MIN_ANNEX]


def annex(p: dict, px) -> brlm.Builder:
    """The annex's walls (window bands at the full level) and flat roof."""
    b = brlm.Builder()
    g, top = p["g"], p["z_annex"]
    bands = [(0.0, None)] + [(g + z, c) for lo, hi in ANNEX_BANDS for z, c in ((lo, GLAZING), (hi, None))]
    for ring in annex_rings(p, px):
        for (xa, ya), (xb, yb) in zip(ring, ring[1:] + ring[:1]):
            levels = bands if px == 600 else [(0.0, None)]
            for (z0, col), z1 in zip(levels, [z for z, _ in levels[1:]] + [top]):
                b.face([(xa, ya, z0), (xb, yb, z0), (xb, yb, z1), (xa, ya, z1)], CONCRETE, TILE, col=col)
        brlm.fill(b, ring, top, CONCRETE, TILE)
    return b


def annex_collision(p: dict) -> brlm.Builder:
    b = brlm.Builder()
    for ring in annex_rings(p, "collision"):
        brlm.extrude_ring(b, ring, 0.0, p["z_annex"], CONCRETE)
        brlm.fill(b, ring, p["z_annex"], CONCRETE)
    return b


# ------------------------------------------------------------------ the parts

def parts(p: dict) -> dict[str, dict]:
    """{part: {"node": its nodes' name, "lods": [(px, triangle budget, Builder), ...],
    "collision": (budget, Builder)}}."""
    return {"main": {"node": "pmac", "lods": [(px, BUDGETS["main"][px], main_part(p, px)) for px in (600, 150, 40)],
                     "collision": (COLLISION_BUDGETS["main"], main_collision(p))},
            "annex": {"node": "annex", "lods": [(px, BUDGETS["annex"][px], annex(p, px)) for px in (600, 150, 40)],
                      "collision": (COLLISION_BUDGETS["annex"], annex_collision(p))}}


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
