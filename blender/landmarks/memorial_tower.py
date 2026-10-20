"""The Memorial Tower at LSU (plan Task 8): the first landmark and the in-game spike's test piece.

Params (out/landmarks.json): height_m (to the top of the roof, the cited height), shaft_m
([along +u, along +v]), footprint_local (the tower with its wings, model coordinates) and
ground_dz (the ground's height above the model's origin, which sits under the lowest ground).
The heights below are above the ground, so they are added to ground_dz; walls reach down to
z 0, so the model never floats.

The shaft, shaft_m wide, centred on the origin and aligned to +u:
* plinth to 1.2 m (br_lm_trim_stone);
* the stucco shaft with corner pilasters to 38 m, under a string course;
* the clock stage, 38-43 m: one 2.6 m dial per side in a stone frame, centred at 40.5 m. All
  four dials bind br_lm_clock;
* the belfry, 43-48.5 m: two open round arches per side through 0.5 m walls, the inner faces
  and the arches' reveals double-sided, over a string course and under the cornice;
* the cornice, then a red tile pyramid from 48.5 m to height_m with a stone finial on top.
Wings: the rest of footprint_local, split into rectangles (brlm.split_rectangles), are 6.5 m
stucco blocks with hipped tile roofs.

Levels of detail: memorial_tower_a400 (all of it, at most 6,000 triangles), _a100 (one
stucco shaft, plain clock quads, coarser arches; 800), _a25 (boxes and the pyramid; 100) and
Colmesh-1 (the shaft and wing boxes; 60).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

import brexport  # noqa: E402
import brlm  # noqa: E402

SHAPE = "memorial_tower"
STUCCO, ROOF, TRIM = "br_lm_stucco_tan", "br_lm_roof_tile", "br_lm_trim_stone"
CLOCK = "br_lm_clock"                                      # the dial on every face
STUCCO_TILE, ROOF_TILE, TRIM_TILE = 2.5, 1.5, 0.6          # m per texture repeat (trim: one block)

# heights above the ground, m
PLINTH_TOP, SHAFT_TOP, CLOCK_TOP, BELFRY_TOP = 1.2, 38.0, 43.0, 48.5
CLOCK_MID, CLOCK_SIDE = 40.5, 2.6
WING_TOP = 6.5
COURSE = 0.4            # string course height: under the clock stage and under the belfry
CORNICE = 0.45
# how far each part stands out from the shaft's face, m. A pilaster is 0.25 m proud (0.15 left a
# strip too thin to draw cleanly at a distance) and its outer face lies in no neighbour's plane,
# so nothing z-fights it: the string course above stands 0.35 m out, the stage 0.15 m
# (tests/test_memorial_tower.py).
PLINTH_OUT, PILASTER_OUT, COURSE_OUT, STAGE_OUT, CORNICE_OUT = 0.3, 0.25, 0.35, 0.15, 0.45
BELFRY_COURSE_OUT = 0.3
PILASTER_W = 0.9
FRAME_SIDE, FRAME_DEPTH, DIAL_GAP = 3.1, 0.08, 0.02
# the belfry's arches
WALL = 0.5
CORNER_PIER, MID_PIER = 1.1, 0.8
SILL, CROWN_GAP = 0.9, 0.5          # parapet under the openings; wall over the arches' crowns
ARCH_SEGMENTS = {400: 12, 100: 4}
# the wings and the finial
WING_PITCH, EAVES = 30.0, 0.5
FINIAL_R, FINIAL_SHAFT, FINIAL_CONE = 0.12, 0.5, 0.8

BUDGETS = {400: 6000, 100: 800, 25: 100}
COLLISION_BUDGET = 60


def sides(hu: float, hv: float):
    """(outward normal, half-extent along it, half-length of the face) per face."""
    return [((1.0, 0.0, 0.0), hu, hv), ((-1.0, 0.0, 0.0), hu, hv),
            ((0.0, 1.0, 0.0), hv, hu), ((0.0, -1.0, 0.0), hv, hu)]


def right_of(n):
    """The direction a viewer facing the face sees as right."""
    return brlm.cross((0.0, 0.0, 1.0), n)


def belfry(b, hu, hv, z0, z1, segments, double_inner):
    """The four arched walls of the belfry between z0 and z1."""
    for n, extent, half in sides(hu, hv):
        r = right_of(n)
        origin = brlm.add(brlm.mul(n, extent), brlm.mul(r, -half))        # the face's left end, z 0
        length = 2 * half
        width = (length - 2 * CORNER_PIER - MID_PIER) / 2
        if width <= 0.5:
            raise ValueError(f"the belfry face is {length} m long, too short for two arches")
        spring = z1 - CROWN_GAP - width / 2
        openings = [(CORNER_PIER + width / 2, width, z0 + SILL, spring),
                    (length - CORNER_PIER - width / 2, width, z0 + SILL, spring)]
        brlm.arched_wall(b, origin, r, length, z0, z1, openings, WALL, STUCCO, STUCCO_TILE, segments,
                         double_inner=double_inner)


def clocks(b, hu, hv, z, out, framed):
    """A dial on every face `out` m in front of the shaft's face; framed puts it on a stone
    frame standing FRAME_DEPTH proud."""
    for n, extent, _ in sides(hu, hv):
        face = extent + out
        if framed:
            f = FRAME_SIDE / 2
            lo, hi = [0.0, 0.0, z - f], [0.0, 0.0, z + f]
            axis = 0 if n[0] else 1
            lo[1 - axis], hi[1 - axis] = -f, f
            sign = n[axis]
            lo[axis], hi[axis] = sorted((sign * face, sign * (face + FRAME_DEPTH)))
            back = ("-" if sign > 0 else "+") + "xy"[axis]
            brlm.box(b, lo, hi, TRIM, TRIM_TILE, skip=(back,))
            face += FRAME_DEPTH
        centre = brlm.add(brlm.mul(n, face + DIAL_GAP), (0.0, 0.0, z))
        brlm.decal(b, centre, right_of(n), CLOCK_SIDE, CLOCK_SIDE, CLOCK)


def roof_and_finial(b, hu, hv, z, apex, finial):
    e = CORNICE_OUT
    brlm.pyramid_roof(b, -hu - e, -hv - e, hu + e, hv + e, z, apex, ROOF, ROOF_TILE)
    if finial:
        brlm.prism(b, (0.0, 0.0), FINIAL_R, apex - 0.3, apex + FINIAL_SHAFT, 8, TRIM, tile=TRIM_TILE)
        brlm.prism(b, (0.0, 0.0), FINIAL_R * 1.8, apex + FINIAL_SHAFT, apex + FINIAL_SHAFT + FINIAL_CONE, 8, TRIM,
                   tip=True, tile=TRIM_TILE)
        b.face([(FINIAL_R * 1.8 * c, FINIAL_R * 1.8 * s, apex + FINIAL_SHAFT) for c, s in
                [(1, 0), (0.7071, -0.7071), (0, -1), (-0.7071, -0.7071), (-1, 0), (-0.7071, 0.7071), (0, 1),
                 (0.7071, 0.7071)]], TRIM, TRIM_TILE)                      # the cone's underside


def wings(b, rects, top, soffit):
    for x0, y0, x1, y1 in rects:
        brlm.box(b, (x0, y0, 0.0), (x1, y1, top), STUCCO, STUCCO_TILE, skip=("-z", "+z"))
        brlm.hipped_roof(b, x0, y0, x1, y1, top, WING_PITCH, ROOF, ROOF_TILE, overhang=EAVES,
                         soffit=STUCCO if soffit else None)


def pilasters(b, hu, hv, z0, z1):
    """A pilaster up each corner of the shaft from z0 to z1: PILASTER_W square, PILASTER_OUT proud
    of both walls that meet there."""
    for su in (-1, 1):
        for sv in (-1, 1):
            u0, u1 = sorted((su * (hu - PILASTER_W + PILASTER_OUT), su * (hu + PILASTER_OUT)))
            v0, v1 = sorted((sv * (hv - PILASTER_W + PILASTER_OUT), sv * (hv + PILASTER_OUT)))
            brlm.box(b, (u0, v0, z0), (u1, v1, z1), STUCCO, STUCCO_TILE, skip=("-z", "+z"))


def high(p) -> brlm.Builder:
    hu, hv, g, h = p["hu"], p["hv"], p["g"], p["h"]
    b = brlm.Builder()
    o = PLINTH_OUT
    brlm.box(b, (-hu - o, -hv - o, 0.0), (hu + o, hv + o, g + PLINTH_TOP), TRIM, TRIM_TILE, skip=("-z",))
    shaft_top = g + SHAFT_TOP - COURSE
    brlm.box(b, (-hu, -hv, g + PLINTH_TOP), (hu, hv, shaft_top), STUCCO, STUCCO_TILE, skip=("-z", "+z"))
    pilasters(b, hu, hv, g + PLINTH_TOP, shaft_top)
    o = COURSE_OUT
    brlm.box(b, (-hu - o, -hv - o, shaft_top), (hu + o, hv + o, g + SHAFT_TOP), TRIM, TRIM_TILE)
    o = STAGE_OUT
    brlm.box(b, (-hu - o, -hv - o, g + SHAFT_TOP), (hu + o, hv + o, g + CLOCK_TOP), STUCCO, STUCCO_TILE,
             skip=("-z", "+z"))
    clocks(b, hu, hv, g + CLOCK_MID, STAGE_OUT, framed=True)
    o = BELFRY_COURSE_OUT                       # its top is the belfry's floor
    brlm.box(b, (-hu - o, -hv - o, g + CLOCK_TOP), (hu + o, hv + o, g + CLOCK_TOP + COURSE), TRIM, TRIM_TILE)
    belfry(b, hu, hv, g + CLOCK_TOP + COURSE, g + BELFRY_TOP - CORNICE, ARCH_SEGMENTS[400], double_inner=True)
    o = CORNICE_OUT                             # its underside is the belfry's ceiling
    brlm.box(b, (-hu - o, -hv - o, g + BELFRY_TOP - CORNICE), (hu + o, hv + o, g + BELFRY_TOP), TRIM, TRIM_TILE,
             skip=("+z",))
    roof_and_finial(b, hu, hv, g + BELFRY_TOP, g + h, finial=True)
    wings(b, p["wings"], g + WING_TOP, soffit=True)
    return b


def mid(p) -> brlm.Builder:
    hu, hv, g, h = p["hu"], p["hv"], p["g"], p["h"]
    b = brlm.Builder()
    o = PLINTH_OUT
    brlm.box(b, (-hu - o, -hv - o, 0.0), (hu + o, hv + o, g + PLINTH_TOP), TRIM, TRIM_TILE, skip=("-z",))
    brlm.box(b, (-hu, -hv, g + PLINTH_TOP), (hu, hv, g + CLOCK_TOP), STUCCO, STUCCO_TILE, skip=("-z", "+z"))
    clocks(b, hu, hv, g + CLOCK_MID, 0.0, framed=False)
    o = BELFRY_COURSE_OUT
    brlm.box(b, (-hu - o, -hv - o, g + CLOCK_TOP), (hu + o, hv + o, g + CLOCK_TOP + COURSE), TRIM, TRIM_TILE)
    belfry(b, hu, hv, g + CLOCK_TOP + COURSE, g + BELFRY_TOP - CORNICE, ARCH_SEGMENTS[100], double_inner=False)
    o = CORNICE_OUT
    brlm.box(b, (-hu - o, -hv - o, g + BELFRY_TOP - CORNICE), (hu + o, hv + o, g + BELFRY_TOP), TRIM, TRIM_TILE,
             skip=("+z",))
    roof_and_finial(b, hu, hv, g + BELFRY_TOP, g + h, finial=False)
    wings(b, p["wings"], g + WING_TOP, soffit=False)
    return b


def low(p) -> brlm.Builder:
    hu, hv, g, h = p["hu"], p["hv"], p["g"], p["h"]
    b = brlm.Builder()
    brlm.box(b, (-hu, -hv, 0.0), (hu, hv, g + BELFRY_TOP), STUCCO, STUCCO_TILE, skip=("-z", "+z"))
    roof_and_finial(b, hu, hv, g + BELFRY_TOP, g + h, finial=False)
    for x0, y0, x1, y1 in p["blocks"]:
        brlm.box(b, (x0, y0, 0.0), (x1, y1, g + WING_TOP), STUCCO, STUCCO_TILE, skip=("-z",), mats={"+z": ROOF})
    return b


def collision(p) -> brlm.Builder:
    hu, hv, g = p["hu"] + PLINTH_OUT, p["hv"] + PLINTH_OUT, p["g"]
    b = brlm.Builder()
    brlm.box(b, (-hu, -hv, 0.0), (hu, hv, g + BELFRY_TOP), STUCCO, skip=("-z",))
    for x0, y0, x1, y1 in p["blocks"]:
        brlm.box(b, (x0, y0, 0.0), (x1, y1, g + WING_TOP), STUCCO, skip=("-z",))
    return b


def resolve(params: dict) -> dict:
    """The dimensions the builders share, from the script's params."""
    hu, hv = (float(v) / 2 for v in params["shaft_m"])
    ring = [tuple(map(float, xy)) for xy in params["footprint_local"]]
    return {"hu": hu, "hv": hv, "g": float(params.get("ground_dz", 0.0)), "h": float(params["height_m"]),
            "wings": brlm.split_rectangles(ring, holes=[(-hu, -hv, hu, hv)]),   # the rest of the footprint
            "blocks": brlm.split_rectangles(ring)}                              # the whole of it, as boxes


def main() -> None:
    params, out = brexport.args()
    p = resolve(params)
    nodes = brlm.lod_nodes(SHAPE, [(400, BUDGETS[400], high(p)), (100, BUDGETS[100], mid(p)),
                                   (25, BUDGETS[25], low(p))], collision=(COLLISION_BUDGET, collision(p)))
    brexport.export({"main": nodes}, out)
    for name, ob in nodes:
        print(f"{name}: {sum(len(f.vertices) - 2 for f in ob.data.polygons)} triangles")


main()
