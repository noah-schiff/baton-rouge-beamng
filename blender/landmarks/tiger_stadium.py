"""Tiger Stadium at LSU (plan Task 15): the bowl, the field, the light towers and the scoreboard.

Params (out/landmarks.json, from landmarks.resolve): footprint_local (the bowl's outer ring),
hole_local (the field hole), rim_profile ([[angle_deg, h], ...]: the lidar rim height above the
ground, by angle counter-clockwise from +u round the footprint's centroid), rim_m ([lowest,
highest]: the profile is held to it), pitch_local and pitch_yaw (the field's centre and axis),
field_m ([length, width] with the end zones), ground_dz and field_dz (the median ground and the
highest ground on the field, above the model's origin). +u points to the north end zone
(azimuth 336), +v to the west stands. Heights below are above the ground (added to ground_dz),
and walls reach down to z 0, so the model never floats.

Seven shape parts, each under validate.DAE_MAX_VERTS vertices per level of detail:
* bowl_nw, bowl_sw, bowl_se, bowl_ne: the bowl split by angle round the centroid (0-90, 90-180,
  180-270 and 270-360 deg; +u is north). Each is lofted along columns (rays from the centroid
  through the field hole and the outer ring; the full level adds the rings' own corners) from
  the field wall (FRONT_WALL over the field) up to the rim at rim_height: the lower deck's seat
  steps to the concourse at CONCOURSE_SHARE of the rim's height, the concourse (CONCOURSE_W
  deep), the upper deck's lip OVERHANG in front of the concourse's back wall, the upper deck's
  steps to the rim, a walkway and the parapet. The tall west and east decks, the low north end
  and the south end-zone deck all follow from the rim. Seats are br_lm_seats, purple below the
  concourse and gold above it by vertex colour (risers darker). The outer facade is
  br_lm_concrete_tan: blind round arches round its foot (recessed, their backs darker by vertex
  colour), a string course over them and pilasters from it up to a cornice at the top.
* field: br_lm_turf (mowing stripes by vertex colour) over the field hole, FIELD_LIFT above the
  highest ground on it; on it, br_lm_field_paint lines (10 cm yard lines every 5 yd, hash marks,
  the sidelines' yard marks, a 6 ft border) and br_lm_field_marks decals from the field atlas
  (logos.field_atlas: the purple BATON ROUGE end zones, the numerals and arrows, the tiger at
  midfield).
* lights: four lattice towers with their lamp banks on the high corners (LIGHT_ANGLES), standing
  on the walkway up to LIGHT_TOP.
* board: the scoreboard (logos.board) on the north end, facing the field.

Levels of detail: <part>_a600, _a150 and _a40, and Colmesh-1: for the bowls the outer wall as a
prism and the concourse slabs; for the field the field plane (br_lm_turf, groundType GRASS);
boxes round the towers' masts and the board.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

import brexport  # noqa: E402
import brlm  # noqa: E402

SEATS, CONCRETE, TURF = "br_lm_seats", "br_lm_concrete_tan", "br_lm_turf"
PAINT, MARKS, BOARD = "br_lm_field_paint", "br_lm_field_marks", "br_lm_board"
STEEL, LAMP = "br_lm_steel", "br_lm_lamp"
CONCRETE_TILE, TURF_TILE, STEEL_TILE = 3.0, 4.0, 1.0       # m per texture repeat

PURPLE = (0x46 / 255, 0x1D / 255, 0x7C / 255)
GOLD = (0xFD / 255, 0xD0 / 255, 0x23 / 255)
RISER_SHADE = 0.7                       # a riser's colour, as a share of its tread's
RECESS = (0.45, 0.42, 0.40)             # the back of a blind arch (times the tan concrete)
TURF_LIGHT, TURF_DARK = (1.0, 1.0, 1.0), (0.78, 0.85, 0.76)

# ------------------------------------------------------------------ the bowl (m, deg)
FIELD_LIFT = 0.03          # the field above the highest ground on it
FRONT_WALL = 1.0           # the field wall: the front row is this far above the field
CONCOURSE_SHARE = 0.45     # the concourse's height, as a share of the rim's
CONCOURSE_W = 8.0
OVERHANG = 4.0             # the upper deck's lip stands this far in front of the concourse's back wall
CLEARANCE = 3.5            # the concourse floor to the lip's underside
FASCIA = 1.2               # the lip's face
LOWER_RAKE, UPPER_RAKE = 28.0, 32.0   # the decks' slopes, where the depth allows (else steeper)
LOWER_MAX_SHARE = 0.55     # the lower deck takes at most this share of a column's depth
TOP_WALK = 2.5             # at least this much walkway behind the top row
PARAPET, PARAPET_T = 1.1, 0.4
WALL_T = 0.6               # the collision wall's thickness
QUARTERS = {"bowl_nw": (0.0, 90.0), "bowl_sw": (90.0, 180.0), "bowl_se": (180.0, 270.0), "bowl_ne": (270.0, 360.0)}
# columns evenly round the bowl (multiples of 4) and whether the rings' own corners are columns too,
# per level of detail and for the collision mesh (whose outer wall then lies on the facade)
COLUMNS = {600: 128, 150: 64, 40: 32, "collision": 32}
RING_CORNERS = {600: True, 150: True, 40: False, "collision": True}
MERGE_DEG = 0.75           # a ring corner this close to a column already there is left out
ROWS = {600: (40, 34), 150: (8, 7), 40: None}    # lower and upper rows; None: plain slopes
# the facade
ARCADE_TOP = 8.0           # the arches' band, from the ground
BAY = 6.5                  # facade per arch, about
ARCH_SHARE = 0.55          # an arch's width, as a share of its bay
ARCH_GAP = 1.2             # wall over an arch's crown
ARCH_DEPTH = 0.5
ARCH_MIN = 3.0             # a facade segment shorter than this has no arch
ARCH_SEGMENTS = 6
PILASTER_W, PILASTER_OUT = 0.9, 0.35
COURSE_H, COURSE_OUT = 0.6, 0.3        # the string course over the arches
CORNICE_H, CORNICE_OUT = 1.0, 0.45     # the cornice at the parapet's top

# ------------------------------------------------------------------ the field (m)
YARD = 0.9144
END_ZONE = 10 * YARD
LINE_W = 0.1               # yard lines, hash marks, the sidelines' yard marks
BORDER = 6 * 0.3048        # the white border round the field
HASH_IN = 60 * 0.3048      # the hash marks' rows, in from each sideline
MARK_LEN = 2 * 0.3048
DIGIT_W, DIGIT_H = 4 * 0.3048, 6 * 0.3048
DIGIT_TOP = 9 * YARD       # a numeral's top, in from its sideline
DIGIT_GAP = 0.25           # from the yard line to a digit
ARROW = 0.9
TIGER = 10 * YARD          # the midfield mark's side
TURF_MARGIN = 1.0          # the turf reaches this far past the field hole (under the field wall)
PAINT_DZ, LINE_DZ = 0.02, 0.03   # decals and lines above the turf (lines over the decals)
# the field atlas (logos.FIELD_REGIONS as Blender uv boxes: logos.atlas_uv; tests check they match)
ATLAS = {"end_zone_a": (0.015625, 0.80859375, 0.484375, 0.984375),
         "end_zone_b": (0.015625, 0.6171875, 0.484375, 0.79296875),
         "tiger": (0.015625, 0.015625, 0.30859375, 0.6015625),
         "arrow_up": (0.32421875, 0.4453125, 0.40234375, 0.6015625),
         "numeral_0": (0.5078125, 0.515625, 0.6640625, 0.984375),
         "numeral_1": (0.671875, 0.515625, 0.828125, 0.984375),
         "numeral_2": (0.8359375, 0.515625, 0.9921875, 0.984375),
         "numeral_3": (0.5078125, 0.03125, 0.6640625, 0.5),
         "numeral_4": (0.671875, 0.03125, 0.828125, 0.5),
         "numeral_5": (0.8359375, 0.03125, 0.9921875, 0.5)}

# ------------------------------------------------------------------ the lights and the board (m over the ground)
# Round the centroid, counter-clockwise from +u: the north ends of the west and east decks (where
# their rim first reaches 42 m) and the south corners, at the lidar's tallest points on the decks
# (56.5 m at 209 deg, 55.0 m at 138 deg; out_lsu ndsm, 2026-10-19).
LIGHT_ANGLES = (63.0, 140.0, 210.0, 295.0)
LIGHT_TOP = 52.5           # the masts' top; the lamp banks reach about BANK_H higher (56.5 m)
MIN_MAST = 2.5             # over the walkway, where the rim is nearly that high
MAST_W, MAST_TOP_W = 2.4, 1.6
BANK_W, BANK_H, BANK_D = 9.0, 4.0, 0.6
LAMPS = {600: (4, 6), 150: (2, 3), 40: (1, 1)}   # lamp faces on a bank (rows, columns)
BANK_TILT = 20.0           # deg, down towards the field
BOARD_TOP = 30.9           # the lidar's peak over the north end (out_lsu ndsm, 2026-10-19)
BOARD_W, BOARD_H = 24.0, 12.0    # the face, 2:1 like logos.board()
BOARD_FRAME, BOARD_D = 0.5, 2.5
BOARD_SETBACK = 4.0        # the face this far in from the outer facade
BOARD_LEG = 1.0

BUDGETS = {                # triangles per level of detail (the full levels are about 15,000 a bowl)
    **{name: {600: 18000, 150: 3000, 40: 250} for name in QUARTERS},
    "field": {600: 1000, 150: 220, 40: 64},
    "lights": {600: 1600, 150: 600, 40: 120},
    "board": {600: 80, 150: 20, 40: 8},
}
COLLISION_BUDGETS = {**{name: 260 for name in QUARTERS}, "field": 2, "lights": 48, "board": 12}


# ------------------------------------------------------------------ the params

def resolve(params: dict) -> dict:
    """The dimensions the builders share, from the script's params."""
    outer = [tuple(map(float, p)) for p in params["footprint_local"]]
    g = float(params.get("ground_dz", 0.0))
    profile = sorted((float(a) % 360.0, float(h)) for a, h in params["rim_profile"])
    if not profile:
        raise ValueError("tiger_stadium: rim_profile is empty")
    pu, pv = map(float, params["pitch_local"])
    length, width = map(float, params["field_m"])
    return {"outer": outer, "hole": [tuple(map(float, p)) for p in params["hole_local"]],
            "centre": brlm.centroid(outer), "g": g, "profile": profile,
            "rim_m": tuple(map(float, params["rim_m"])),
            "field_z": max(g, float(params.get("field_dz", g))) + FIELD_LIFT,
            "pitch": (pu, pv), "pitch_yaw": float(params.get("pitch_yaw", 0.0)), "field": (length, width)}


def rim_height(p: dict, angle: float) -> float:
    """The rim's z at `angle` round the centre: the profile, interpolated round the circle and
    held to rim_m, over the ground."""
    prof = p["profile"]
    a = angle % 360.0
    before = [x for x in prof if x[0] <= a] or [(prof[-1][0] - 360.0, prof[-1][1])]
    after = [x for x in prof if x[0] > a] or [(prof[0][0] + 360.0, prof[0][1])]
    (a0, h0), (a1, h1) = before[-1], after[0]
    h = h0 + (h1 - h0) * (a - a0) / (a1 - a0) if a1 > a0 else h0
    lo, hi = p["rim_m"]
    return p["g"] + min(max(h, lo), hi)


def column(p: dict, angle: float) -> dict:
    """The bowl's section along the ray from the centre at `angle`: hole and outer, its points on
    the field hole and the outer ring; L the depth between them; H the rim; and along the ray
    (s, in m from the hole) and up (z): zf the front row; s1, zc the concourse's front; s2 its
    back; lip, zs and zu the upper deck's lip, its underside and top; top the top row."""
    hole, outer = brlm.ring_at(p["hole"], p["centre"], angle), brlm.ring_at(p["outer"], p["centre"], angle)
    L = math.dist(hole, outer)
    H = rim_height(p, angle)
    zf = p["field_z"] + FRONT_WALL
    zc = p["g"] + CONCOURSE_SHARE * (H - p["g"])
    s1 = min((zc - zf) / math.tan(math.radians(LOWER_RAKE)), LOWER_MAX_SHARE * L)
    s2 = s1 + CONCOURSE_W
    lip, zs = s2 - OVERHANG, zc + CLEARANCE
    zu = zs + FASCIA
    top = min(lip + (H - zu) / math.tan(math.radians(UPPER_RAKE)), L - PARAPET_T - TOP_WALK)
    if zu > H - 1.0 or top < lip + 2.0:
        raise ValueError(f"tiger_stadium: the bowl at {angle:.1f} deg (rim {H:.1f} m, {L:.1f} m deep) "
                         "has no room for two decks")
    return {"angle": angle, "hole": hole, "outer": outer, "L": L, "H": H, "zf": zf, "zc": zc, "s1": s1,
            "s2": s2, "lip": lip, "zs": zs, "zu": zu, "top": top}


def at(c: dict, s: float, z: float) -> tuple[float, float, float]:
    """The point s along a column from its hole point, at z."""
    t = s / c["L"]
    (hx, hy), (ox, oy) = c["hole"], c["outer"]
    return hx + (ox - hx) * t, hy + (oy - hy) * t, z


def angles(p: dict, level, lo: float, hi: float) -> list[float]:
    """The column angles of a level of detail (or "collision") from lo to hi (both ends):
    COLUMNS[level] evenly round the bowl, and with RING_CORNERS the outer ring's and the hole's
    own corners."""
    n = COLUMNS[level]
    even = [360.0 * k / n for k in range(n + 1)]
    out = [a for a in even if lo - 1e-9 <= a <= hi + 1e-9]
    if RING_CORNERS[level]:
        for q in p["outer"] + p["hole"]:
            a = brlm.angle_of(p["centre"], q)
            a = a if a >= lo - 1e-9 else a + 360.0
            if lo < a < hi and all(abs(a - b) >= MERGE_DEG for b in out):
                out.append(a)
    return sorted(out)


# ------------------------------------------------------------------ the bowl

def shade(col, k: float):
    return tuple(c * k for c in col)


def ring(cols, s_of, z_of) -> list:
    return [at(c, s_of(c), z_of(c)) for c in cols]


def facade(b, cols, arches: bool):
    """The outer facade from the ground to the parapet's top, facing out. With arches: a band of
    blind arches under ARCADE_TOP on every segment long enough, a string course over them,
    pilasters from it up to a cornice at the top."""
    for c0, c1 in zip(cols, cols[1:]):
        (x0, y0), (x1, y1) = c0["outer"], c1["outer"]
        top0, top1 = c0["H"] + PARAPET, c1["H"] + PARAPET
        length = math.hypot(x1 - x0, y1 - y0)
        if length < 1e-6:
            continue
        right = ((x1 - x0) / length, (y1 - y0) / length, 0.0)
        if not arches:
            brlm.quad(b, [(x0, y0, 0.0), (x1, y1, 0.0), (x1, y1, top1), (x0, y0, top0)], CONCRETE, CONCRETE_TILE)
            continue
        n = max(1, round(length / BAY)) if length >= ARCH_MIN else 0
        if n:
            bay = length / n
            width = ARCH_SHARE * bay
            spring = ARCADE_TOP - ARCH_GAP - width / 2
            openings = [((i + 0.5) * bay, width, 0.0, spring) for i in range(n)]
            brlm.blind_arches(b, (x0, y0, 0.0), right, length, 0.0, ARCADE_TOP, openings, ARCH_DEPTH, CONCRETE,
                              CONCRETE_TILE, ARCH_SEGMENTS, recess=RECESS)
        else:
            brlm.quad(b, [(x0, y0, 0.0), (x1, y1, 0.0), (x1, y1, ARCADE_TOP), (x0, y0, ARCADE_TOP)], CONCRETE,
                      CONCRETE_TILE)
        brlm.quad(b, [(x0, y0, ARCADE_TOP), (x1, y1, ARCADE_TOP), (x1, y1, top1), (x0, y0, top0)], CONCRETE,
                  CONCRETE_TILE)
        band(b, (x0, y0), (x1, y1), ARCADE_TOP, ARCADE_TOP, COURSE_H, COURSE_OUT)
        band(b, (x0, y0), (x1, y1), top0 - CORNICE_H, top1 - CORNICE_H, CORNICE_H, CORNICE_OUT)
        for i in range(n):                        # a pilaster over each pier, course to cornice
            s = max(i * bay, PILASTER_W / 2 + 0.05)
            z1 = top0 + (top1 - top0) * s / length - CORNICE_H
            pilaster(b, (x0, y0), right, s, ARCADE_TOP + COURSE_H, z1)


def band(b, p0, p1, z0: float, z1: float, height: float, proud: float):
    """A band `height` tall standing `proud` out of the facade from p0 (its foot at z0) to p1
    (at z1): its front, top and underside."""
    (x0, y0), (x1, y1) = p0, p1
    length = math.hypot(x1 - x0, y1 - y0)
    ox, oy = (y1 - y0) / length * proud, -(x1 - x0) / length * proud      # out of the facade
    lo = [(x0, y0, z0), (x1, y1, z1)]
    lo_out = [(x0 + ox, y0 + oy, z0), (x1 + ox, y1 + oy, z1)]
    hi = [(x, y, z + height) for x, y, z in lo]
    hi_out = [(x, y, z + height) for x, y, z in lo_out]
    brlm.quad(b, [lo_out[0], lo_out[1], hi_out[1], hi_out[0]], CONCRETE, CONCRETE_TILE)          # front
    brlm.quad(b, [hi_out[0], hi_out[1], hi[1], hi[0]], CONCRETE, CONCRETE_TILE)                  # top
    brlm.quad(b, [lo[0], lo[1], lo_out[1], lo_out[0]], CONCRETE, CONCRETE_TILE)                  # underside


def pilaster(b, origin, right, s: float, z0: float, z1: float):
    """A pilaster PILASTER_W wide, PILASTER_OUT proud of the wall, centred s along it."""
    out = (right[1], -right[0], 0.0)
    ox, oy = origin
    half = PILASTER_W / 2

    def p(along, proud, z):
        return (ox + right[0] * (s + along) + out[0] * proud, oy + right[1] * (s + along) + out[1] * proud, z)

    b.face([p(-half, PILASTER_OUT, z0), p(half, PILASTER_OUT, z0), p(half, PILASTER_OUT, z1),
            p(-half, PILASTER_OUT, z1)], CONCRETE, CONCRETE_TILE)
    b.face([p(-half, 0.0, z0), p(-half, PILASTER_OUT, z0), p(-half, PILASTER_OUT, z1), p(-half, 0.0, z1)],
           CONCRETE, CONCRETE_TILE)
    b.face([p(half, PILASTER_OUT, z0), p(half, 0.0, z0), p(half, 0.0, z1), p(half, PILASTER_OUT, z1)],
           CONCRETE, CONCRETE_TILE)
    b.face([p(-half, 0.0, z1), p(-half, PILASTER_OUT, z1), p(half, PILASTER_OUT, z1), p(half, 0.0, z1)],
           CONCRETE, CONCRETE_TILE)


def bowl(p: dict, lo: float, hi: float, px: int) -> brlm.Builder:
    """One quarter of the bowl, between the angles lo and hi, at a level of detail."""
    cols = [column(p, a) for a in angles(p, px, lo, hi)]
    b = brlm.Builder()
    rows = ROWS[px]
    front_foot = ring(cols, lambda c: 0.0, lambda c: 0.0)
    front = ring(cols, lambda c: 0.0, lambda c: c["zf"])
    concourse = ring(cols, lambda c: c["s1"], lambda c: c["zc"])
    concourse_back = ring(cols, lambda c: c["s2"], lambda c: c["zc"])
    back_top = ring(cols, lambda c: c["s2"], lambda c: c["zs"])
    lip_under = ring(cols, lambda c: c["lip"], lambda c: c["zs"])
    lip_top = ring(cols, lambda c: c["lip"], lambda c: c["zu"])
    top_row = ring(cols, lambda c: c["top"], lambda c: c["H"])
    walk_end = ring(cols, lambda c: c["L"] - PARAPET_T, lambda c: c["H"])
    parapet = ring(cols, lambda c: c["L"] - PARAPET_T, lambda c: c["H"] + PARAPET)
    cap = ring(cols, lambda c: c["L"], lambda c: c["H"] + PARAPET)

    brlm.loft(b, [front_foot, front], CONCRETE, CONCRETE_TILE)                       # the field wall
    if rows:
        lower, upper = rows
        brlm.seat_steps(b, front, concourse, lower, SEATS, tread=PURPLE, riser=shade(PURPLE, RISER_SHADE))
        brlm.seat_steps(b, lip_top, top_row, upper, SEATS, tread=GOLD, riser=shade(GOLD, RISER_SHADE))
    else:
        brlm.loft(b, [front, concourse], SEATS, col=PURPLE)
        brlm.loft(b, [lip_top, top_row], SEATS, col=GOLD)
    brlm.loft(b, [concourse, concourse_back], CONCRETE, CONCRETE_TILE)               # the concourse
    brlm.loft(b, [concourse_back, back_top], CONCRETE, CONCRETE_TILE)                # its back wall
    brlm.loft(b, [back_top, lip_under], CONCRETE, CONCRETE_TILE)                     # under the overhang
    brlm.loft(b, [lip_under, lip_top], CONCRETE, CONCRETE_TILE)                      # the lip's face
    brlm.loft(b, [top_row, walk_end, parapet, cap], CONCRETE, CONCRETE_TILE)         # walkway and parapet
    facade(b, cols, arches=px == 600)
    return b


def bowl_collision(p: dict, lo: float, hi: float) -> brlm.Builder:
    """The outer wall as a prism WALL_T thick to the parapet's top, and the concourse slabs."""
    cols = [column(p, a) for a in angles(p, "collision", lo, hi)]
    b = brlm.Builder()
    outside = [ring(cols, lambda c: c["L"], lambda c: 0.0), ring(cols, lambda c: c["L"], lambda c: c["H"] + PARAPET)]
    inside = [ring(cols, lambda c: c["L"] - WALL_T, lambda c: 0.0),
              ring(cols, lambda c: c["L"] - WALL_T, lambda c: c["H"] + PARAPET)]
    brlm.loft(b, outside, CONCRETE, flip=True)
    brlm.loft(b, inside, CONCRETE)
    brlm.loft(b, [inside[1], outside[1]], CONCRETE)
    brlm.loft(b, [ring(cols, lambda c: c["s1"], lambda c: c["zc"]), ring(cols, lambda c: c["s2"], lambda c: c["zc"])],
              CONCRETE)
    return b


# ------------------------------------------------------------------ the field

class Field:
    """The field's frame: l along its length (+l towards +u, the north end), w across it."""

    def __init__(self, p: dict):
        (self.cu, self.cv), yaw = p["pitch"], p["pitch_yaw"]
        self.el = (math.cos(yaw), math.sin(yaw))
        self.ew = (-math.sin(yaw), math.cos(yaw))
        self.length, self.width = p["field"]
        self.z = p["field_z"]
        pts = [self.local(q) for q in p["hole"]]
        m = TURF_MARGIN
        self.turf = (min(l for l, _ in pts) - m, min(w for _, w in pts) - m,
                     max(l for l, _ in pts) + m, max(w for _, w in pts) + m)

    def local(self, q):
        du, dv = q[0] - self.cu, q[1] - self.cv
        return du * self.el[0] + dv * self.el[1], du * self.ew[0] + dv * self.ew[1]

    def point(self, l: float, w: float, dz: float = 0.0):
        return (self.cu + l * self.el[0] + w * self.ew[0], self.cv + l * self.el[1] + w * self.ew[1], self.z + dz)

    def rect(self, b, l0, w0, l1, w1, dz, mat, tile=1.0, col=None):
        b.face([self.point(l0, w0, dz), self.point(l1, w0, dz), self.point(l1, w1, dz), self.point(l0, w1, dz)],
               mat, tile, col=col)

    def decal(self, b, corner, right, up, size, box, dz=PAINT_DZ):
        """An atlas region on the field: its bottom-left at corner (l, w), its right and up along
        the unit (l, w) vectors right and up (right x up must point up), size (w, h) m."""
        (l, w), (sw, sh) = corner, size
        pts = [(l, w), (l + right[0] * sw, w + right[1] * sw),
               (l + right[0] * sw + up[0] * sh, w + right[1] * sw + up[1] * sh), (l + up[0] * sh, w + up[1] * sh)]
        u0, v0, u1, v1 = box
        b.face([self.point(a, c, dz) for a, c in pts], MARKS, uv=[(u0, v0), (u1, v0), (u1, v1), (u0, v1)])


def yard_l(f: Field, yards: float) -> float:
    """l of the yard line `yards` from the -l goal line."""
    return -f.length / 2 + END_ZONE + yards * YARD


def turf(b, f: Field, striped: bool = True):
    """The turf over the field hole: stripes 5 yd deep between the goal lines, mowed light and dark."""
    l0, w0, l1, w1 = f.turf
    cuts = [yard_l(f, y) for y in range(0, 101, 5)] if striped else []
    edges = [l0] + [c for c in cuts if l0 < c < l1] + [l1]
    for i, (a, c) in enumerate(zip(edges, edges[1:])):
        mid = (a + c) / 2
        inside = yard_l(f, 0) < mid < yard_l(f, 100)
        dark = inside and int((mid - yard_l(f, 0)) // (5 * YARD)) % 2 == 1
        f.rect(b, a, w0, c, w1, 0.0, TURF, TURF_TILE, col=TURF_DARK if dark else TURF_LIGHT)


def lines(b, f: Field, hash_marks: bool):
    half_l, half_w = f.length / 2, f.width / 2
    for y in range(0, 101, 5):                                     # the yard lines, goal lines included
        l = yard_l(f, y)
        f.rect(b, l - LINE_W / 2, -half_w, l + LINE_W / 2, half_w, LINE_DZ, PAINT)
    # the border round the field (its inner edge is the sideline and the end line)
    f.rect(b, -half_l - BORDER, half_w, half_l + BORDER, half_w + BORDER, LINE_DZ, PAINT)
    f.rect(b, -half_l - BORDER, -half_w - BORDER, half_l + BORDER, -half_w, LINE_DZ, PAINT)
    f.rect(b, half_l, -half_w, half_l + BORDER, half_w, LINE_DZ, PAINT)
    f.rect(b, -half_l - BORDER, -half_w, -half_l, half_w, LINE_DZ, PAINT)
    if not hash_marks:
        return
    for y in range(1, 100):
        if y % 5 == 0:
            continue
        l = yard_l(f, y)
        for w in (half_w - HASH_IN, -half_w + HASH_IN):                 # the hash marks
            f.rect(b, l - LINE_W / 2, w - MARK_LEN / 2, l + LINE_W / 2, w + MARK_LEN / 2, PAINT_DZ, PAINT)
        for w in (half_w - LINE_W - MARK_LEN, -half_w + LINE_W):        # the sidelines' yard marks
            f.rect(b, l - LINE_W / 2, w, l + LINE_W / 2, w + MARK_LEN, PAINT_DZ, PAINT)


def numerals(b, f: Field, arrows: bool):
    """10 .. 50 .. 10 on both sides, 6 ft tall, tops DIGIT_TOP in from the sideline, read from
    that sideline; with arrows, one beside each number but the 50 pointing to the nearer goal."""
    half_w = f.width / 2
    for y in range(10, 100, 10):
        label = f"{min(y, 100 - y):d}"
        l = yard_l(f, y)
        for side in (1, -1):                            # +1: the +w (west) sideline's numbers
            right, up = (-side, 0.0), (0.0, -side)       # as seen from that sideline, facing the field
            bottom = side * (half_w - DIGIT_TOP + DIGIT_H)
            left_edge = l + side * (DIGIT_GAP + DIGIT_W)   # the tens digit is on the viewer's left
            for i, digit in enumerate(label):
                l0 = left_edge if i == 0 else l - side * DIGIT_GAP
                f.decal(b, (l0, bottom), right, up, (DIGIT_W, DIGIT_H), ATLAS[f"numeral_{digit}"])
            if arrows and y != 50:
                goal = 1.0 if y > 50 else -1.0           # towards the nearer goal line, along l
                tip_up, tip_right = (goal, 0.0), (0.0, -goal)
                outer = l + goal * (DIGIT_GAP + DIGIT_W + 0.3)
                top_w = side * (half_w - DIGIT_TOP + 0.1)
                corner = (outer - tip_right[0] * ARROW / 2, top_w - tip_right[1] * ARROW / 2)
                f.decal(b, corner, tip_right, tip_up, (ARROW, ARROW), ATLAS["arrow_up"])


def end_zones(b, f: Field):
    """The end zones' panels, the lettering read from the field (its foot on the goal line)."""
    half_l, half_w = f.length / 2, f.width / 2
    f.decal(b, (half_l - END_ZONE, half_w), (0.0, -1.0), (1.0, 0.0), (f.width, END_ZONE), ATLAS["end_zone_a"])
    f.decal(b, (-half_l + END_ZONE, -half_w), (0.0, 1.0), (-1.0, 0.0), (f.width, END_ZONE), ATLAS["end_zone_b"])


def tiger(b, f: Field):
    """The midfield mark, upright from the west (+w) sideline."""
    f.decal(b, (TIGER / 2, TIGER / 2), (-1.0, 0.0), (0.0, -1.0), (TIGER, TIGER), ATLAS["tiger"])


def field(p: dict, px: int) -> brlm.Builder:
    f = Field(p)
    b = brlm.Builder()
    turf(b, f)
    end_zones(b, f)
    tiger(b, f)
    if px >= 150:
        lines(b, f, hash_marks=px == 600)
        numerals(b, f, arrows=px == 600)
    return b


def field_collision(p: dict) -> brlm.Builder:
    f = Field(p)
    b = brlm.Builder()
    l0, w0, l1, w1 = f.turf
    f.rect(b, l0, w0, l1, w1, 0.0, TURF)
    return b


# ------------------------------------------------------------------ the lights and the board

def walkway(p: dict, angle: float):
    """(x, y, z) on the walkway behind the top row at `angle`."""
    c = column(p, angle)
    return at(c, (c["top"] + c["L"] - PARAPET_T) / 2, c["H"])


def lights(p: dict, px: int) -> brlm.Builder:
    b = brlm.Builder()
    target = p["pitch"]
    for angle in LIGHT_ANGLES:
        x, y, z0 = walkway(p, angle)
        z1 = max(p["g"] + LIGHT_TOP, z0 + MIN_MAST)
        facing = (target[0] - x, target[1] - y, 0.0)
        brlm.light_tower(b, (x, y), MAST_W, z0, z1, facing, STEEL, LAMP, top_width=MAST_TOP_W,
                         bank=(BANK_W, BANK_H, BANK_D), lamps=LAMPS[px],
                         tilt_deg=BANK_TILT, panels=max(1, round((z1 - z0) / 3.0)) if px >= 150 else 1,
                         braces=px == 600, lattice=px >= 150, tile=STEEL_TILE)
    return b


def lights_collision(p: dict) -> brlm.Builder:
    b = brlm.Builder()
    for angle in LIGHT_ANGLES:
        x, y, z0 = walkway(p, angle)
        z1 = max(p["g"] + LIGHT_TOP, z0 + MIN_MAST)
        h = MAST_W / 2
        brlm.box(b, (x - h, y - h, z0), (x + h, y + h, z1), STEEL, skip=("-z",))
    return b


def board_frame(p: dict):
    """(the face's centre (x, y, z), the field's axes l and w as 3-vectors, the bowl's column
    under it): on the field's axis at the north end, BOARD_SETBACK in from the outer facade, the
    face's top BOARD_FRAME under BOARD_TOP. The face looks down -l, at the field."""
    f = Field(p)
    axis = math.degrees(math.atan2(f.el[1], f.el[0]))
    ox, oy = brlm.ring_at(p["outer"], p["pitch"], axis)
    x, y = ox - f.el[0] * BOARD_SETBACK, oy - f.el[1] * BOARD_SETBACK
    top = p["g"] + BOARD_TOP - BOARD_FRAME
    el, ew = (f.el[0], f.el[1], 0.0), (f.ew[0], f.ew[1], 0.0)
    return (x, y, top - BOARD_H / 2), el, ew, column(p, brlm.angle_of(p["centre"], (x, y)))


def board_box(b, p: dict, mat=STEEL, tile=STEEL_TILE):
    """The board's box, BOARD_D deep behind its face; returns the box's centre."""
    (x, y, zc), el, ew, _ = board_frame(p)
    centre = (x + el[0] * BOARD_D / 2, y + el[1] * BOARD_D / 2, zc)
    brlm.oriented_box(b, centre, (el, ew, (0.0, 0.0, 1.0)),
                      (BOARD_D / 2, BOARD_W / 2 + BOARD_FRAME, BOARD_H / 2 + BOARD_FRAME), mat, tile)
    return centre


def board(p: dict, px: int) -> brlm.Builder:
    """The face on a steel box (at a600 with a frame round the face and legs from the walkway
    when the box clears it; at a40 the face, the back and the top only)."""
    (x, y, zc), el, ew, c = board_frame(p)
    b = brlm.Builder()
    right = (-ew[0], -ew[1])                   # the viewer's right, from the field looking down +l

    def front(su, sv, out=0.02, grow=0.0):
        return (x + right[0] * su * (BOARD_W / 2 + grow) - el[0] * out,
                y + right[1] * su * (BOARD_W / 2 + grow) - el[1] * out, zc + sv * (BOARD_H / 2 + grow))

    b.face([front(su, sv) for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))], BOARD,
           uv=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    if px == 40:
        back = [front(su, sv, -BOARD_D, BOARD_FRAME) for su, sv in ((1, -1), (-1, -1), (-1, 1), (1, 1))]
        b.face(back, STEEL, STEEL_TILE)
        b.face([front(-1, 1, 0.0, BOARD_FRAME), front(1, 1, 0.0, BOARD_FRAME), back[3], back[2]], STEEL, STEEL_TILE)
        return b
    centre = board_box(b, p)
    if px == 600:
        trim = BOARD_FRAME / 2                 # a frame standing out round the face
        for su, sv, hw, hh in ((0, 1, BOARD_W / 2 + BOARD_FRAME, trim), (0, -1, BOARD_W / 2 + BOARD_FRAME, trim),
                               (1, 0, trim, BOARD_H / 2), (-1, 0, trim, BOARD_H / 2)):
            mid = (x + right[0] * su * (BOARD_W / 2 + trim) - el[0] * 0.15,
                   y + right[1] * su * (BOARD_W / 2 + trim) - el[1] * 0.15, zc + sv * (BOARD_H / 2 + trim))
            brlm.oriented_box(b, mid, (el, ew, (0.0, 0.0, 1.0)), (0.15, hw, hh), STEEL, STEEL_TILE)
        bottom = zc - BOARD_H / 2 - BOARD_FRAME
        if bottom > c["H"] + 0.1:                  # legs from the walkway up to the box
            for su in (-1, 1):
                lx, ly = centre[0] + ew[0] * su * BOARD_W / 3, centre[1] + ew[1] * su * BOARD_W / 3
                h = BOARD_LEG / 2
                brlm.box(b, (lx - h, ly - h, c["H"]), (lx + h, ly + h, bottom), STEEL, STEEL_TILE, skip=("-z", "+z"))
    return b


def board_collision(p: dict) -> brlm.Builder:
    b = brlm.Builder()
    board_box(b, p)
    return b


# ------------------------------------------------------------------ the parts

def parts(p: dict) -> dict[str, dict]:
    """{part: {"lods": [(px, triangle budget, Builder), ...], "collision": (budget, Builder)}}."""
    out = {}
    for name, (lo, hi) in QUARTERS.items():
        out[name] = {"lods": [(px, BUDGETS[name][px], bowl(p, lo, hi, px)) for px in (600, 150, 40)],
                     "collision": (COLLISION_BUDGETS[name], bowl_collision(p, lo, hi))}
    for name, build, col in (("field", field, field_collision), ("lights", lights, lights_collision),
                             ("board", board, board_collision)):
        out[name] = {"lods": [(px, BUDGETS[name][px], build(p, px)) for px in (600, 150, 40)],
                     "collision": (COLLISION_BUDGETS[name], col(p))}
    return out


def main() -> None:
    params, out = brexport.args()
    nodes = {}
    for name, part in parts(resolve(params)).items():
        nodes[name] = brlm.lod_nodes(name, part["lods"], collision=part["collision"])
    brexport.export(nodes, out)
    for name, part_nodes in nodes.items():
        for node, ob in part_nodes:
            print(f"{name} {node}: {sum(len(f.vertices) - 2 for f in ob.data.polygons)} triangles, "
                  f"{sum(len(f.vertices) for f in ob.data.polygons)} corners")


main()
