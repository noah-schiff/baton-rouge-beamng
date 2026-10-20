"""Alex Box Stadium, Skip Bertman Field at LSU (plan Task 16): the covered grandstand round home
plate, the bleachers down the lines and beyond the outfield, the field and the outfield wall.

Params (out/landmarks.json, from landmarks.resolve): rim_m (the grandstand roof's front edge,
above the ground), home_az_deg (home plate's compass bearing from the mound), bases_m and mound_m,
field_local (the field's polygons: the playing field round the infield, the infield grass round
the mound; the mound is the centre of the smallest hole), field_dz (the highest ground on the
field), outlines_local (the outfield bleachers' outlines), footprint_local and ground_dz. The
model's +u is east and +v north (the registry's yaw, azimuth 90), so a compass bearing b is the
model angle 90 - b; the origin is the ballpark's centroid. Heights below are above the ground
(added to ground_dz) and walls reach down to z 0, so the model never floats.

Four shape parts, each under validate.DAE_MAX_VERTS vertices per level of detail:
* grandstand: the arc from compass bearing 150 to 255 round the origin (the lidar's 19 m roof),
  lofted along columns (rays from the origin): the field wall where the ray leaves the field,
  the seats rising at RAKE to a walkway, the roof canopy over the upper rows from its front edge
  (ROOF_FRONT: 52 m out at the arc's ends, 58 m in its middle, at rim_m) back over the walkway,
  and behind it the concourse block (BACK_DEPTH and BACK_LOW..BACK_HIGH, deepest and tallest
  in the middle as the lidar shows); a purple fascia with a gold edge on the roof, a purple and
  gold band round the block's top; closed by end walls.
* bleachers: open stands down both lines (bearings 105-150 and 255-300, LINE_DEPTH deep up to
  LINE_TOP, as the lidar's 5-6 m ramps there) and on each outfield outline (rising away from the
  field to OUTFIELD_TOP), aluminium grey.
* field: turf over the field polygon at the highest ground on it (FIELD_LIFT over field_dz), its
  edge skirted down to z 0; the infield's clay (the fan from home to the arc INFIELD_ARC round
  the mound, between the foul lines, with the circle round home), the infield grass inside the
  base paths, the mound, the bases, home plate and the foul lines.
* wall: the outfield wall along the field polygon's edge between the foul lines (WALL_H over the
  field, green with a gold cap) and the gold foul poles at its ends.
Levels of detail: <part>_a600, _a150 and _a40, and Colmesh-1: the stands' fronts, seat slopes
and outer walls; the field (br_lm_turf, groundType GRASS); the wall.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

import brexport  # noqa: E402
import brlm  # noqa: E402

CONCRETE, SEATS, PAINT = "br_lm_concrete", "br_lm_seats", "br_lm_paint"
TURF, CLAY, LINES = "br_lm_turf", "br_lm_clay", "br_lm_field_paint"
TILE, TURF_TILE, CLAY_TILE = 3.0, 4.0, 3.0      # m per texture repeat

PURPLE = (0x46 / 255, 0x1D / 255, 0x7C / 255)
GOLD = (0xFD / 255, 0xD0 / 255, 0x23 / 255)
WALL_GREEN = (0.12, 0.30, 0.18)
ALUMINIUM = (0.74, 0.75, 0.77)
RISER_SHADE = 0.7

# ------------------------------------------------------------------ the stands (m, deg)
GRANDSTAND_AZ = (150.0, 255.0)  # compass bearings from the origin: the lidar's 18-19 m roof
LINES_AZ = ((105.0, 150.0), (255.0, 300.0))    # the stands down the lines (the replaced w1148083778)
FRONT_WALL = 1.0                # the field wall: the front row this far over the field
RAKE = 26.0                     # the seats' slope at most
ROOF_FRONT = (52.0, 58.0)       # the roof's front edge from the origin: at the arc's ends, in its middle
ROOF_DEPTH = 18.0               # the roof's front edge to the walkway's back
ROOF_FALL, ROOF_T = 1.0, 0.8    # the roof falls this much to its back; its thickness
TRIM = 0.25                     # the gold edge on the purple fascia
WALK = 2.0                      # the walkway behind the top row
HEADROOM = 4.0                  # the top row to the roof's underside, at least
BACK_DEPTH = 18.0               # the concourse block behind the roof, in the arc's middle
BACK_LOW, BACK_HIGH = 9.0, 15.0 # its roof at the arc's ends and in its middle (the lidar's 11-15 m)
BAND = 1.0                      # its purple band under the gold edge
LINE_DEPTH, LINE_TOP = 18.0, 5.5    # the line stands: depth and top row over the ground
LINE_WALK = 1.5
OUTFIELD_FRONT, OUTFIELD_TOP = 1.0, 4.5    # the outfield bleachers' front and top rows over the ground
RAIL = 1.1
STEP = {600: 1.5, 150: 5.0, 40: 15.0, "collision": 15.0}      # deg between the stands' columns
ROWS = {600: (28, 14, 10), 150: (7, 4, 3), 40: None}           # grandstand, line and outfield rows

# ------------------------------------------------------------------ the field (m)
FIELD_LIFT = 0.03               # the turf over the highest ground on the field
CLAY_DZ, GRASS_DZ, MARK_DZ = 0.01, 0.02, 0.03
INFIELD_ARC = 28.96             # 95 ft round the mound
HOME_R = 3.96                   # the dirt circle round home: 13 ft
GRASS_INSET = 1.2               # the infield grass this far inside the base lines
MOUND_R, MOUND_H = 2.74, 0.25   # 18 ft across, 10 in high
BASE, BASE_H = 0.38, 0.08       # 15 in square
PLATE = 0.43                    # home plate: 17 in across
LINE_W = 0.1
ARC_STEP = {600: 4.0, 150: 10.0, 40: 30.0}
WALL_H, WALL_T, CAP = 3.0, 0.5, 0.3    # the outfield wall: over the field, thick, its gold cap
POLE_H, POLE_W = 15.0, 0.35            # the foul poles over the field

BUDGETS = {"grandstand": {600: 11500, 150: 1350, 40: 220}, "bleachers": {600: 4750, 150: 600, 40: 110},
           "field": {600: 220, 150: 140, 40: 60}, "wall": {600: 110, 150: 60, 40: 50}}   # ~20 % over the LSU build
COLLISION_BUDGETS = {"grandstand": 80, "bleachers": 90, "field": 10, "wall": 50}


# ------------------------------------------------------------------ the params

def _v(a: float) -> tuple[float, float]:
    """The unit vector at model angle a (deg)."""
    return math.cos(math.radians(a)), math.sin(math.radians(a))


def _area(ring) -> float:
    return abs(brlm._signed_area(ring))


def resolve(params: dict) -> dict:
    """The dimensions the builders share, from the script's params."""
    g = float(params.get("ground_dz", 0.0))
    polys = [[[tuple(map(float, q)) for q in ring] for ring in poly] for poly in params["field_local"]]
    field = polys[0][0]
    holes = [ring for poly in polys for ring in poly[1:]]
    if not holes:
        raise ValueError("alex_box: the field has no hole for the mound")
    mound = brlm.centroid(min(holes, key=_area))
    home_angle = 90.0 - float(params["home_az_deg"])           # from the mound
    hx, hy = _v(home_angle)
    m = float(params["mound_m"])
    home = (mound[0] + hx * m, mound[1] + hy * m)
    cf = (home_angle + 180.0) % 360.0                           # home to second base
    bases = float(params["bases_m"])

    def from_home(turn: float, d: float):
        ux, uy = _v(cf + turn)
        return home[0] + ux * d, home[1] + uy * d

    return {"g": g, "field": field, "mound": mound, "home": home, "cf": cf,
            "first": from_home(-45.0, bases), "second": from_home(0.0, bases * math.sqrt(2)),
            "third": from_home(45.0, bases),
            "field_z": max(g, float(params.get("field_dz", g))) + FIELD_LIFT,
            "rim_z": g + float(params["rim_m"]),
            "outlines": [[tuple(map(float, q)) for q in ring] for ring in params.get("outlines_local") or []]}


def theta(bearing: float) -> float:
    """A compass bearing as a model angle in [0, 360)."""
    return (90.0 - bearing) % 360.0


def columns(lo_az: float, hi_az: float, level) -> list[float]:
    """Model angles counter-clockwise from the bearing hi_az to lo_az (both ends), STEP apart."""
    lo, hi = theta(hi_az), theta(lo_az)
    span = (hi - lo) % 360.0
    n = max(1, math.ceil(span / STEP[level] - 1e-9))
    return [lo + span * k / n for k in range(n + 1)]


def at(a: float, r: float, z: float) -> tuple[float, float, float]:
    ux, uy = _v(a)
    return ux * r, uy * r, z


def field_edge(p: dict, a: float) -> float:
    """How far out from the origin the ray at model angle a leaves the field."""
    return math.hypot(*brlm.ring_at(p["field"], (0.0, 0.0), a))


# ------------------------------------------------------------------ the grandstand

def bump(a: float) -> float:
    """0 at the grandstand arc's ends, 1 in its middle (sin of the share along it)."""
    lo, hi = theta(GRANDSTAND_AZ[1]), theta(GRANDSTAND_AZ[0])
    t = ((a - lo) % 360.0) / ((hi - lo) % 360.0)
    return math.sin(math.pi * min(max(t, 0.0), 1.0))


def grandstand_column(p: dict, a: float) -> dict:
    """The grandstand's section along the ray at model angle a: r_f the field wall, zf the front
    row; r_top, z_top the top row; r_b the walkway's back; r_c, r_e the roof's front and back
    (z_rf, z_rb on top); r_out, z_back the concourse block's outer wall and roof."""
    k = bump(a)
    r_f = field_edge(p, a)
    r_c = ROOF_FRONT[0] + (ROOF_FRONT[1] - ROOF_FRONT[0]) * k
    r_b = r_c + ROOF_DEPTH
    r_top, r_e = r_b - WALK, r_b + 1.0
    zf = p["field_z"] + FRONT_WALL
    z_rf = p["rim_z"]
    z_rb = z_rf - ROOF_FALL
    z_top = min(zf + (r_top - r_f) * math.tan(math.radians(RAKE)), z_rb - ROOF_T - HEADROOM)
    return {"a": a, "r_f": r_f, "zf": zf, "r_top": r_top, "z_top": z_top, "r_b": r_b, "r_c": r_c, "r_e": r_e,
            "z_rf": z_rf, "z_rb": z_rb, "r_out": r_e + BACK_DEPTH * k,
            "z_back": p["g"] + BACK_LOW + (BACK_HIGH - BACK_LOW) * k}


def roof_z(c: dict, r: float) -> float:
    """The roof's top at r along a column."""
    return c["z_rf"] + (c["z_rb"] - c["z_rf"]) * (r - c["r_c"]) / (c["r_e"] - c["r_c"])


def ring(cols, r_of, z_of) -> list:
    return [at(c["a"], r_of(c), z_of(c)) for c in cols]


def shade(col, k: float):
    return tuple(v * k for v in col)


def section_wall(b, a: float, section, facing_ccw: bool) -> None:
    """A wall in the plane of the ray at model angle a, its outline the (r, z) polygon section,
    facing counter-clockwise round the origin or clockwise."""
    poly = []
    for r, z in section:                                       # without repeated corners
        if not poly or math.dist(poly[-1], (r, z)) > 1e-6:
            poly.append((float(r), float(z)))
    if math.dist(poly[0], poly[-1]) <= 1e-6:
        poly.pop()
    if brlm._signed_area(poly) < 0:
        poly = poly[::-1]
    for tri in brlm.triangulate(poly):
        pts = [at(a, *poly[i]) for i in tri]                     # counter-clockwise in (r, z): faces clockwise
        b.face(pts[::-1] if facing_ccw else pts, CONCRETE, TILE)


def grandstand(p: dict, px) -> brlm.Builder:
    cols = [grandstand_column(p, a) for a in columns(*GRANDSTAND_AZ, px)]
    b = brlm.Builder()
    rows = ROWS[px]
    front = ring(cols, lambda c: c["r_f"], lambda c: c["zf"])
    top = ring(cols, lambda c: c["r_top"], lambda c: c["z_top"])
    brlm.loft(b, [ring(cols, lambda c: c["r_f"], lambda c: 0.0), front], CONCRETE, TILE)          # the field wall
    if rows:
        brlm.seat_steps(b, front, top, rows[0], SEATS, tread=PURPLE, riser=shade(PURPLE, RISER_SHADE))
    else:
        brlm.loft(b, [front, top], SEATS, col=PURPLE)
    back = ring(cols, lambda c: c["r_b"], lambda c: c["z_top"])
    brlm.loft(b, [top, back], CONCRETE, TILE)                                                   # the walkway
    under_b = ring(cols, lambda c: c["r_b"], lambda c: roof_z(c, c["r_b"]) - ROOF_T)
    brlm.loft(b, [back, under_b], CONCRETE, TILE)                                               # the back wall
    edge_lo = ring(cols, lambda c: c["r_c"], lambda c: c["z_rf"] - ROOF_T)
    edge_mid = ring(cols, lambda c: c["r_c"], lambda c: c["z_rf"] - TRIM)
    edge_top = ring(cols, lambda c: c["r_c"], lambda c: c["z_rf"])
    brlm.loft(b, [edge_lo, under_b], CONCRETE, TILE, flip=True)                                 # the roof's underside
    brlm.loft(b, [edge_lo, edge_mid], PAINT, col=PURPLE)                                        # the fascia
    brlm.loft(b, [edge_mid, edge_top], PAINT, col=GOLD)
    roof_back = ring(cols, lambda c: c["r_e"], lambda c: c["z_rb"])
    brlm.loft(b, [edge_top, roof_back], CONCRETE, TILE)                                         # the roof
    block = ring(cols, lambda c: c["r_e"], lambda c: c["z_back"])
    brlm.loft(b, [block, roof_back], CONCRETE, TILE, flip=True)                                 # the roof's back
    outer_top = ring(cols, lambda c: c["r_out"], lambda c: c["z_back"])
    brlm.loft(b, [block, outer_top], CONCRETE, TILE)                                            # the block's roof
    band_lo = ring(cols, lambda c: c["r_out"], lambda c: c["z_back"] - BAND - TRIM)
    band_hi = ring(cols, lambda c: c["r_out"], lambda c: c["z_back"] - TRIM)
    brlm.loft(b, [ring(cols, lambda c: c["r_out"], lambda c: 0.0), band_lo], CONCRETE, TILE, flip=True)
    if px != 40:
        brlm.loft(b, [band_lo, band_hi], PAINT, flip=True, col=PURPLE)
        brlm.loft(b, [band_hi, outer_top], PAINT, flip=True, col=GOLD)
    else:
        brlm.loft(b, [band_lo, outer_top], PAINT, flip=True, col=PURPLE)
    for c, facing_ccw in ((cols[0], False), (cols[-1], True)):                                # the ends
        section_wall(b, c["a"], [(c["r_f"], 0.0), (c["r_out"], 0.0), (c["r_out"], c["z_back"]),
                                 (c["r_e"], c["z_back"]), (c["r_e"], c["z_rb"]), (c["r_c"], c["z_rf"]),
                                 (c["r_c"], c["z_rf"] - ROOF_T), (c["r_b"], roof_z(c, c["r_b"]) - ROOF_T),
                                 (c["r_b"], c["z_top"]), (c["r_top"], c["z_top"]), (c["r_f"], c["zf"])], facing_ccw)
    return b


def grandstand_collision(p: dict) -> brlm.Builder:
    """The field wall, the seats' slope and walkway, the block's outer wall and the ends."""
    cols = [grandstand_column(p, a) for a in columns(*GRANDSTAND_AZ, "collision")]
    b = brlm.Builder()
    front = ring(cols, lambda c: c["r_f"], lambda c: c["zf"])
    brlm.loft(b, [ring(cols, lambda c: c["r_f"], lambda c: 0.0), front], CONCRETE)
    brlm.loft(b, [front, ring(cols, lambda c: c["r_top"], lambda c: c["z_top"]),
                  ring(cols, lambda c: c["r_b"], lambda c: c["z_top"])], CONCRETE)
    brlm.loft(b, [ring(cols, lambda c: c["r_out"], lambda c: 0.0),
                  ring(cols, lambda c: c["r_out"], lambda c: c["z_back"])], CONCRETE, flip=True)
    for c, facing_ccw in ((cols[0], False), (cols[-1], True)):
        section_wall(b, c["a"], [(c["r_f"], 0.0), (c["r_out"], 0.0), (c["r_out"], c["z_back"]),
                                 (c["r_top"], c["z_top"]), (c["r_f"], c["zf"])], facing_ccw)
    return b


# ------------------------------------------------------------------ the bleachers

def line_column(p: dict, a: float) -> dict:
    r_f = field_edge(p, a)
    return {"a": a, "r_f": r_f, "zf": p["field_z"] + FRONT_WALL, "r_top": r_f + LINE_DEPTH - LINE_WALK,
            "r_b": r_f + LINE_DEPTH, "z_top": p["g"] + LINE_TOP}


def line_stand(b, p: dict, lo_az: float, hi_az: float, level, rows, collision: bool = False) -> None:
    """Open stands down a line: field wall, seats, walkway, a rail on the back wall; the end away
    from the grandstand closed."""
    cols = [line_column(p, a) for a in columns(lo_az, hi_az, level)]
    front = ring(cols, lambda c: c["r_f"], lambda c: c["zf"])
    top = ring(cols, lambda c: c["r_top"], lambda c: c["z_top"])
    back = ring(cols, lambda c: c["r_b"], lambda c: c["z_top"])
    rail = ring(cols, lambda c: c["r_b"], lambda c: c["z_top"] + RAIL)
    brlm.loft(b, [ring(cols, lambda c: c["r_f"], lambda c: 0.0), front], CONCRETE, TILE)
    if collision:
        brlm.loft(b, [front, top, back], CONCRETE)
    elif rows:
        brlm.seat_steps(b, front, top, rows, SEATS, tread=ALUMINIUM, riser=shade(ALUMINIUM, RISER_SHADE))
        brlm.loft(b, [top, back], CONCRETE, TILE)
    else:
        brlm.loft(b, [front, top], SEATS, col=ALUMINIUM)
        brlm.loft(b, [top, back], CONCRETE, TILE)
    if not collision:
        brlm.loft(b, [back, rail], PAINT, col=PURPLE)                                       # the rail, inside
    brlm.loft(b, [ring(cols, lambda c: c["r_b"], lambda c: 0.0), rail], CONCRETE, TILE, flip=True)
    end, facing_ccw = (cols[-1], True) if lo_az < GRANDSTAND_AZ[0] else (cols[0], False)
    section_wall(b, end["a"], [(end["r_f"], 0.0), (end["r_b"], 0.0), (end["r_b"], end["z_top"] + RAIL),
                               (end["r_top"], end["z_top"]), (end["r_f"], end["zf"])], facing_ccw)


def front_edge(p: dict, ring_pts):
    """(front corners [c, a] with the field on their left, back corners [d, e]) of a bleacher's
    outline: its long side nearer the mound is the front."""
    pts = list(ring_pts)
    if brlm._signed_area(pts) < 0:
        pts = pts[::-1]
    n = len(pts)
    edges = sorted(range(n), key=lambda i: -math.dist(pts[i], pts[(i + 1) % n]))[:2]
    i = min(edges, key=lambda i: math.dist(((pts[i][0] + pts[(i + 1) % n][0]) / 2,
                                            (pts[i][1] + pts[(i + 1) % n][1]) / 2), p["mound"]))
    a, c = pts[i], pts[(i + 1) % n]
    d, e = pts[(i + 2) % n], pts[(i + 3) % n]
    return [c, a], [d, e]


def outfield_bleacher(b, p: dict, ring_pts, rows, collision: bool = False) -> None:
    (c, a), (d, e) = front_edge(p, ring_pts)
    g = p["g"]
    z1, z2 = g + OUTFIELD_FRONT, g + OUTFIELD_TOP
    front = [(*c, z1), (*a, z1)]
    back = [(*d, z2), (*e, z2)]
    brlm.loft(b, [[(*c, 0.0), (*a, 0.0)], front], CONCRETE, TILE)                      # the front wall
    if rows and not collision:
        brlm.seat_steps(b, front, back, rows, SEATS, tread=ALUMINIUM, riser=shade(ALUMINIUM, RISER_SHADE))
    else:
        brlm.loft(b, [front, back], SEATS if not collision else CONCRETE, col=None if collision else ALUMINIUM)
    brlm.loft(b, [[(*d, 0.0), (*e, 0.0)], [(*d, z2 + RAIL), (*e, z2 + RAIL)]], CONCRETE, TILE, flip=True)
    if not collision:
        brlm.loft(b, [back, [(*d, z2 + RAIL), (*e, z2 + RAIL)]], PAINT, col=PURPLE)
    b.face([(*c, 0.0), (*d, 0.0), (*d, z2), (*c, z1)], CONCRETE, TILE)                    # the sides
    b.face([(*e, 0.0), (*a, 0.0), (*a, z1), (*e, z2)], CONCRETE, TILE)


def bleachers(p: dict, px) -> brlm.Builder:
    b = brlm.Builder()
    rows = ROWS[px]
    for lo, hi in LINES_AZ:
        line_stand(b, p, lo, hi, px, rows[1] if rows else None)
    for ring_pts in p["outlines"]:
        outfield_bleacher(b, p, ring_pts, rows[2] if rows else None)
    return b


def bleachers_collision(p: dict) -> brlm.Builder:
    b = brlm.Builder()
    for lo, hi in LINES_AZ:
        line_stand(b, p, lo, hi, "collision", None, collision=True)
    for ring_pts in p["outlines"]:
        outfield_bleacher(b, p, ring_pts, None, collision=True)
    return b


# ------------------------------------------------------------------ the field

def _ray_hits(ring_pts, origin, angle: float):
    """(t, edge index) of where the ray from origin at model angle a leaves the ring (its farthest
    crossing)."""
    ox, oy = origin
    dx, dy = _v(angle)
    best = None
    n = len(ring_pts)
    for i in range(n):
        (xa, ya), (xb, yb) = ring_pts[i], ring_pts[(i + 1) % n]
        ex, ey = xb - xa, yb - ya
        den = dx * ey - dy * ex
        if abs(den) < 1e-12:
            continue
        t = ((xa - ox) * ey - (ya - oy) * ex) / den
        s = ((xa - ox) * dy - (ya - oy) * dx) / den
        if t > 1e-9 and -1e-9 <= s <= 1 + 1e-9 and (best is None or t > best[0]):
            best = (t, i)
    if best is None:
        raise ValueError("alex_box: home plate is not inside the field")
    return best


def foul_pole(p: dict, side: float):
    """(the point where the foul line on that side (-1: first base, +1: third) meets the field's
    edge, the edge's index)."""
    angle = p["cf"] + 45.0 * side
    t, i = _ray_hits(p["field"], p["home"], angle)
    ux, uy = _v(angle)
    return (p["home"][0] + ux * t, p["home"][1] + uy * t), i


def outfield_line(p: dict) -> list:
    """The field's edge counter-clockwise from the first-base foul pole to the third-base one."""
    (q1, i1), (q3, i3) = foul_pole(p, -1.0), foul_pole(p, 1.0)
    n = len(p["field"])
    pts, i = [q1], i1
    while i != i3:
        i = (i + 1) % n
        pts.append(p["field"][i])
    pts.append(q3)
    return [q for k, q in enumerate(pts) if k == 0 or math.dist(q, pts[k - 1]) > 1e-6]


def infield_clay(p: dict, step: float) -> list:
    """The clay: from the first-base line round the arc INFIELD_ARC from the mound to the
    third-base line, down it to the circle round home, round home behind the plate and back up
    the first-base line."""
    home, mound, cf = p["home"], p["mound"], p["cf"]

    def on_line(side: float, r: float):
        """Where the foul line on that side is r from the mound (the far root)."""
        ux, uy = _v(cf + 45.0 * side)
        fx, fy = home[0] - mound[0], home[1] - mound[1]
        bq = fx * ux + fy * uy
        t = -bq + math.sqrt(bq * bq - (fx * fx + fy * fy - r * r))
        return home[0] + ux * t, home[1] + uy * t

    def angle_from(c, q):
        return math.degrees(math.atan2(q[1] - c[1], q[0] - c[0]))

    def arc(c, r, a0, a1, ccw: bool):
        sweep = (a1 - a0) % 360.0 if ccw else -((a0 - a1) % 360.0)
        n = max(1, math.ceil(abs(sweep) / step))
        return [(c[0] + r * math.cos(math.radians(a0 + sweep * k / n)),
                 c[1] + r * math.sin(math.radians(a0 + sweep * k / n))) for k in range(n + 1)]

    first_out, third_out = on_line(-1.0, INFIELD_ARC), on_line(1.0, INFIELD_ARC)
    big = arc(mound, INFIELD_ARC, angle_from(mound, first_out), angle_from(mound, third_out), True)
    third_in = (home[0] + _v(cf + 45.0)[0] * HOME_R, home[1] + _v(cf + 45.0)[1] * HOME_R)
    first_in = (home[0] + _v(cf - 45.0)[0] * HOME_R, home[1] + _v(cf - 45.0)[1] * HOME_R)
    small = arc(home, HOME_R, angle_from(home, third_in), angle_from(home, first_in), True)
    return big + small


def infield_grass(p: dict) -> list:
    """The grass inside the base paths: the diamond moved GRASS_INSET in from the base lines, its
    corner at home pulled back to the dirt circle."""
    home, first, second, third = p["home"], p["first"], p["second"], p["third"]
    centre = ((home[0] + second[0]) / 2, (home[1] + second[1]) / 2)

    def toward(q, d):
        ux, uy = centre[0] - q[0], centre[1] - q[1]
        n = math.hypot(ux, uy)
        return q[0] + ux / n * d, q[1] + uy / n * d

    inset = GRASS_INSET * math.sqrt(2)
    return [toward(home, HOME_R * 1.15), toward(first, inset), toward(second, inset), toward(third, inset)]


def square(centre, side: float, angle: float) -> list:
    ux, uy = _v(angle)
    vx, vy = -uy, ux
    h = side / 2
    return [(centre[0] + sx * h * ux + sy * h * vx, centre[1] + sx * h * uy + sy * h * vy)
            for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]


def field(p: dict, px) -> brlm.Builder:
    b = brlm.Builder()
    z = p["field_z"]
    brlm.fill(b, p["field"], z, TURF, TURF_TILE)
    brlm.extrude_ring(b, p["field"], 0.0, z, CONCRETE, TILE)                              # the skirt
    brlm.fill(b, infield_clay(p, ARC_STEP[px]), z + CLAY_DZ, CLAY, CLAY_TILE)
    brlm.fill(b, infield_grass(p), z + GRASS_DZ, TURF, TURF_TILE)
    sides = {600: 12, 150: 8, 40: 6}[px]
    mx, my = p["mound"]
    rim = [(mx + MOUND_R * math.cos(2 * math.pi * k / sides), my + MOUND_R * math.sin(2 * math.pi * k / sides),
            z + GRASS_DZ + 0.01) for k in range(sides)]
    for k in range(sides):                                                                # the mound
        b.face([rim[k], rim[(k + 1) % sides], (mx, my, z + GRASS_DZ + MOUND_H)], CLAY, CLAY_TILE)
    if px == 40:
        return b
    for q in (p["first"], p["second"], p["third"]):                                      # the bases
        sq = square(q, BASE, p["cf"] + 45.0)
        brlm.extrude_ring(b, sq, z, z + BASE_H, LINES, top=LINES)
    hx, hy = p["home"]
    ux, uy = _v(p["cf"] + 180.0)                    # home plate's point is towards the catcher
    vx, vy = -uy, ux
    h = PLATE / 2
    plate = [(hx + vx * h, hy + vy * h), (hx + ux * h + vx * h, hy + uy * h + vy * h),
             (hx + ux * 2 * h, hy + uy * 2 * h), (hx + ux * h - vx * h, hy + uy * h - vy * h),
             (hx - vx * h, hy - vy * h)]
    brlm.fill(b, plate, z + MARK_DZ, LINES)
    for side in (-1.0, 1.0):                                                              # the foul lines
        q, _ = foul_pole(p, side)
        a = p["home"]
        length = math.dist(a, q)
        dx, dy = (q[0] - a[0]) / length, (q[1] - a[1]) / length
        nx, ny = -dy * LINE_W / 2, dx * LINE_W / 2
        b.face([(a[0] - nx, a[1] - ny, z + MARK_DZ), (q[0] - nx, q[1] - ny, z + MARK_DZ),
                (q[0] + nx, q[1] + ny, z + MARK_DZ), (a[0] + nx, a[1] + ny, z + MARK_DZ)], LINES)
    return b


def field_collision(p: dict) -> brlm.Builder:
    return brlm.fill(brlm.Builder(), p["field"], p["field_z"], TURF)


# ------------------------------------------------------------------ the outfield wall

def wall(p: dict, px) -> brlm.Builder:
    b = brlm.Builder()
    line = outfield_line(p)
    top = p["field_z"] + WALL_H
    if px == 600:                                       # green with a gold cap band
        brlm.wall_along(b, line, 0.0, top - CAP, WALL_T, PAINT, back=CONCRETE, col=WALL_GREEN)
        brlm.wall_along(b, line, top - CAP, top, WALL_T, PAINT, back=CONCRETE, top=PAINT, col=GOLD)
    else:                                               # green, open at its ends
        brlm.wall_along(b, line, 0.0, top, WALL_T, PAINT, back=CONCRETE, col=WALL_GREEN, ends=px == 150)
    for side in (-1.0, 1.0):                                                              # the foul poles
        q, _ = foul_pole(p, side)
        ux, uy = _v(p["cf"] + 45.0 * side)
        c = (q[0] + ux * (WALL_T + POLE_W), q[1] + uy * (WALL_T + POLE_W))
        ring_pts = square(c, POLE_W, p["cf"] + 45.0 * side)
        for (xa, ya), (xb, yb) in zip(ring_pts, ring_pts[1:] + ring_pts[:1]):
            b.face([(xa, ya, 0.0), (xb, yb, 0.0), (xb, yb, p["field_z"] + POLE_H), (xa, ya, p["field_z"] + POLE_H)],
                   PAINT, col=GOLD)
        if px != 40:
            brlm.fill(b, ring_pts, p["field_z"] + POLE_H, PAINT, col=GOLD)
    return b


def wall_collision(p: dict) -> brlm.Builder:
    return brlm.wall_along(brlm.Builder(), outfield_line(p), 0.0, p["field_z"] + WALL_H, WALL_T, CONCRETE, top=CONCRETE)


# ------------------------------------------------------------------ the parts

def parts(p: dict) -> dict[str, dict]:
    """{part: {"node": its nodes' name, "lods": [(px, triangle budget, Builder), ...],
    "collision": (budget, Builder)}}."""
    builds = {"grandstand": (grandstand, grandstand_collision), "bleachers": (bleachers, bleachers_collision),
              "field": (field, field_collision), "wall": (wall, wall_collision)}
    return {name: {"node": name, "lods": [(px, BUDGETS[name][px], build(p, px)) for px in (600, 150, 40)],
                   "collision": (COLLISION_BUDGETS[name], col(p))} for name, (build, col) in builds.items()}


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
