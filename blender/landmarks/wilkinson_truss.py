"""The Horace Wilkinson Bridge (plan Task 19): the cantilever through truss that carries I-10 over the
Mississippi, round the two decks the bridges stage already lays (its MeshRoads); the model adds the
steel and the river piers, never the roadway.

Params (out/landmarks.json): deck_local (per deck {"id", "width", "nodes": [[u, v, z], ...]}, the
deck's nodes between the truss's ends, in u order, z up from base_z), from_u and to_u (the truss's
ends), peaks_u (its two peaks, over the main piers), top_m ([at the ends, over the peaks]: the top
chord above the deck), clear_m (the least height of anything over the roadway above the deck) and
panel_m (the panel length). +u runs along the decks (west, since the yaw is taken modulo 180 degrees),
+v is left of it (south). The deck's z and v between and past its nodes are taken along straight
lines between them, carried on past the first and last node (the nodes stop short of the ends).

* Panel points: the truss from end to end, the stretches between the ends and the peaks each cut
  into panels about panel_m long, the main span (between the peaks) into an even number of them so
  its middle is a panel point too.
* Two vertical Warren truss planes, PLANE_OUT outside the combined decks' outer edges at each panel
  point (following the deck on that side): the bottom chord BOTTOM_DROP under that deck's surface,
  the top chord top_m[0] above the higher deck at the ends and in the middle of the main span and
  top_m[1] over the peaks, straight between (the cantilever's M); a vertical at every panel point
  and a diagonal in every panel, leaning one way and the other in turn. Chords CHORD square, the web
  WEB, all br_lm_truss_steel (stock metal_paint, grey-green).
* Overhead, at every panel point: a strut between the top chords and a sway frame, a strut
  SWAY_DEPTH under them with an X up to the top struts; at the two ends the frame's strut is as low
  as clear_m allows (the portals); an X of lateral bracing in the top plane in every panel. Nothing
  over the roadway is lower than clear_m above the higher deck (plus the members' half widths).
* Piers (br_lm_concrete): the two main river piers under the peaks (MAIN_PIER thick along u, cut
  water noses up and down the river) and an approach pier at each end (END_PIER thick), each
  PIER_OUT past both planes, from PIER_FOOT under base_z (the lowest ground along the footprint's
  outline: the river bed) to DECK_DEPTH under the lower deck's surface (the deck slab's underside),
  with a bearing pedestal up to each bottom chord.

Levels of detail: wilkinson_truss_a600 (all of it, at most 8,000 triangles), _a120 (the chords, the
diagonals, the posts at the ends and the peaks, a top strut at every other panel point and the piers;
3,600), _a30 (the chords between every fourth panel point and the peaks, the end and peak posts, the
piers; 700), Colmesh-1 (each plane as thin boxes, a panel each, and the piers; nothing overhead; 1,600).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

import brexport  # noqa: E402
import brlm  # noqa: E402

SHAPE = "wilkinson_truss"
STEEL, CONCRETE = "br_lm_truss_steel", "br_lm_concrete"
TILE = 2.0                  # m per texture repeat
PLANE_OUT = 1.0             # m from the outer deck edge out to a truss plane's centre
BOTTOM_DROP = 1.0           # m from the deck's surface down to the bottom chord's centre
CHORD, WEB, BRACE = 0.9, 0.5, 0.35   # m: the chords', the web's and the bracing's square sections
SWAY_DEPTH = 3.0            # m from the top struts down to a sway frame's strut
MAIN_PIER, END_PIER = 10.0, 4.0      # m: the piers' thickness along u
PIER_OUT = 2.0              # m a pier reaches past each plane
PIER_FOOT = 1.0             # m a pier reaches under base_z
DECK_DEPTH = 1.5            # m: the deck MeshRoads' thickness (beamng_format.DECK_DEPTH)
COARSE = 4                  # _a30 draws every fourth panel point (and the peaks)

BUDGETS = {600: 8000, 120: 3600, 30: 700}
COLLISION_BUDGET = 1600


# ------------------------------------------------------------------ the params

def along(nodes, u: float, k: int) -> float:
    """Column k (1: v, 2: z) of the u-sorted nodes at u, straight between them and past the ends."""
    if u <= nodes[0][0]:
        i = 0
    elif u >= nodes[-1][0]:
        i = len(nodes) - 2
    else:
        i = next(j for j in range(len(nodes) - 1) if nodes[j][0] <= u <= nodes[j + 1][0])
    a, b = nodes[i], nodes[i + 1]
    t = (u - a[0]) / (b[0] - a[0]) if b[0] != a[0] else 0.0
    return a[k] + t * (b[k] - a[k])


def panel_points(ends, peaks, panel: float) -> list[float]:
    """The panel points from end to end (see the module docstring)."""
    lo, hi = ends
    stops = [lo, *peaks, hi]
    out = [lo]
    for k, (a, b) in enumerate(zip(stops, stops[1:])):
        n = max(1, round((b - a) / panel))
        if len(stops) == 4 and k == 1:                  # the main span: an even count
            n = max(2, 2 * round((b - a) / (2 * panel)))
        out += [a + (b - a) * j / n for j in range(1, n + 1)]
    return out


def top_above(u: float, ends, peaks, top) -> float:
    """The top chord's height above the deck at u: low at the ends and mid-span, high over the peaks."""
    lo, hi = ends
    low, high = top
    marks = [(lo, low), (hi, low)] + [(p, high) for p in peaks]
    if len(peaks) == 2:
        marks.append(((peaks[0] + peaks[1]) / 2, low))
    marks.sort()
    for (u0, h0), (u1, h1) in zip(marks, marks[1:]):
        if u0 <= u <= u1:
            return h0 + (h1 - h0) * (u - u0) / (u1 - u0) if u1 != u0 else h1
    return low


def resolve(params: dict) -> dict:
    """The dimensions the builders share, from the script's params."""
    decks = [([tuple(map(float, n)) for n in d["nodes"]], float(d["width"])) for d in params["deck_local"]]
    ends = tuple(sorted((float(params["from_u"]), float(params["to_u"]))))
    peaks = sorted(float(p) for p in params["peaks_u"])
    top = tuple(float(h) for h in params["top_m"])
    panels = panel_points(ends, peaks, float(params["panel_m"]))
    planes = {"l": [], "r": []}
    deck_top, deck_low = [], []
    for u in panels:
        at = [(along(n, u, 1), along(n, u, 2), w) for n, w in decks]
        zt = max(z for _, z, _ in at) + top_above(u, ends, peaks, top)
        left = max(at, key=lambda d: d[0] + d[2] / 2)
        right = min(at, key=lambda d: d[0] - d[2] / 2)
        planes["l"].append((u, left[0] + left[2] / 2 + PLANE_OUT, left[1] - BOTTOM_DROP, zt))
        planes["r"].append((u, right[0] - right[2] / 2 - PLANE_OUT, right[1] - BOTTOM_DROP, zt))
        deck_top.append(max(z for _, z, _ in at))
        deck_low.append(min(z for _, z, _ in at))
    piers = []
    for u, thick in ((ends[0], END_PIER), *((p, MAIN_PIER) for p in peaks), (ends[1], END_PIER)):
        k = min(range(len(panels)), key=lambda j: abs(panels[j] - u))
        v0, v1 = planes["r"][k][1] - PIER_OUT, planes["l"][k][1] + PIER_OUT
        piers.append((u, thick / 2, v0, v1, -PIER_FOOT, deck_low[k] - DECK_DEPTH))
    return {"panels": panels, "planes": planes, "deck_top": deck_top, "deck_low": deck_low, "piers": piers,
            "peaks": peaks, "ends": ends, "clear": float(params["clear_m"])}


# ------------------------------------------------------------------ pieces

def chord_points(p: dict, side: str, k: int, which: str):
    u, v, zb, zt = p["planes"][side][k]
    return (u, v, zb if which == "bottom" else zt)


def truss_plane(b, p: dict, side: str, keep) -> None:
    """One plane's chords (between the panel points `keep`), and in the full level its verticals and
    diagonals."""
    for which in ("bottom", "top"):
        for k0, k1 in zip(keep, keep[1:]):
            brlm.beam(b, chord_points(p, side, k0, which), chord_points(p, side, k1, which), CHORD, STEEL, TILE)


def web(b, p: dict, side: str, verticals, diagonals: bool) -> None:
    """Verticals at the panel points `verticals`; with diagonals, one in every panel, leaning in turn."""
    for k in verticals:
        brlm.beam(b, chord_points(p, side, k, "bottom"), chord_points(p, side, k, "top"), WEB, STEEL, TILE)
    if diagonals:
        for k in range(len(p["panels"]) - 1):
            lo, hi = ("bottom", "top") if k % 2 == 0 else ("top", "bottom")
            brlm.beam(b, chord_points(p, side, k, lo), chord_points(p, side, k + 1, hi), WEB, STEEL, TILE)


def overhead(b, p: dict, struts, sway: bool, lateral: bool) -> None:
    """Top struts at the panel points `struts`; with sway, a sway frame (a portal at the ends) under
    each; with lateral, an X in the top plane in every panel."""
    last = len(p["panels"]) - 1
    for k in struts:
        (u, vl, _, zt), (_, vr, _, _) = p["planes"]["l"][k], p["planes"]["r"][k]
        brlm.beam(b, (u, vr, zt), (u, vl, zt), BRACE, STEEL, TILE)
        if not sway:
            continue
        lowest = p["deck_top"][k] + p["clear"] + BRACE
        zs = lowest if k in (0, last) else max(zt - SWAY_DEPTH, lowest)
        if zs < zt - BRACE:
            brlm.beam(b, (u, vr, zs), (u, vl, zs), BRACE, STEEL, TILE)
            brlm.beam(b, (u, vr, zs), (u, vl, zt), BRACE, STEEL, TILE)
            brlm.beam(b, (u, vl, zs), (u, vr, zt), BRACE, STEEL, TILE)
    if lateral:
        for k in range(last):
            (u0, l0, _, z0), (_, r0, _, _) = p["planes"]["l"][k], p["planes"]["r"][k]
            (u1, l1, _, z1), (_, r1, _, _) = p["planes"]["l"][k + 1], p["planes"]["r"][k + 1]
            brlm.beam(b, (u0, r0, z0), (u1, l1, z1), BRACE, STEEL, TILE)
            brlm.beam(b, (u0, l0, z0), (u1, r1, z1), BRACE, STEEL, TILE)


def piers(b, p: dict, pedestals: bool) -> None:
    """The piers (cut water noses on the main ones) and, with pedestals, a bearing block from each
    pier's top up to each bottom chord."""
    for u, half, v0, v1, z0, z1 in p["piers"]:
        if half * 2 >= MAIN_PIER:                       # pointed up and down the river
            ring = [(u, v0 - half), (u + half, v0), (u + half, v1), (u, v1 + half), (u - half, v1), (u - half, v0)]
        else:
            ring = [(u - half, v0), (u + half, v0), (u + half, v1), (u - half, v1)]
        brlm.extrude_ring(b, ring, z0, z1, CONCRETE, TILE, top=CONCRETE)
        if not pedestals:
            continue
        k = min(range(len(p["panels"])), key=lambda j: abs(p["panels"][j] - u))
        for side in ("l", "r"):
            _, v, zb, _ = p["planes"][side][k]
            foot = zb - CHORD / 2
            if foot > z1 + 0.05:
                h = CHORD / 2 + 0.2
                brlm.box(b, (u - h, v - h, z1), (u + h, v + h, foot), CONCRETE, TILE, skip=("-z",))


def coarse(p: dict) -> list[int]:
    """Every COARSE-th panel point, the peaks and the last one."""
    panels = p["panels"]
    peaks = {min(range(len(panels)), key=lambda j: abs(panels[j] - u)) for u in p["peaks"]}
    return sorted(set(range(0, len(panels), COARSE)) | peaks | {len(panels) - 1})


def ends_and_peaks(p: dict) -> list[int]:
    panels = p["panels"]
    return sorted({0, len(panels) - 1} | {min(range(len(panels)), key=lambda j: abs(panels[j] - u))
                                         for u in p["peaks"]})


# ------------------------------------------------------------------ levels of detail

def level(p: dict, px) -> brlm.Builder:
    b = brlm.Builder()
    every = list(range(len(p["panels"])))
    for side in ("l", "r"):
        truss_plane(b, p, side, every if px != 30 else coarse(p))
        web(b, p, side, every if px == 600 else ends_and_peaks(p), diagonals=px != 30)
    if px == 600:
        overhead(b, p, every, sway=True, lateral=True)
    elif px == 120:
        overhead(b, p, every[::2], sway=False, lateral=False)
    piers(b, p, pedestals=px != 30)
    return b


def collision(p: dict) -> brlm.Builder:
    """Each plane as a thin box per panel, from under its bottom chord to over its top chord, and
    the piers."""
    b = brlm.Builder()
    t = CHORD / 2
    for side in ("l", "r"):
        pts = p["planes"][side]
        for (u0, v0, b0, t0), (u1, v1, b1, t1) in zip(pts, pts[1:]):
            lo = [(u0, v0 - t, b0 - t), (u0, v0 + t, b0 - t), (u0, v0 + t, t0 + t), (u0, v0 - t, t0 + t)]
            hi = [(u1, v1 - t, b1 - t), (u1, v1 + t, b1 - t), (u1, v1 + t, t1 + t), (u1, v1 - t, t1 + t)]
            hexahedron(b, lo, hi, STEEL)
    piers(b, p, pedestals=False)
    return b


def hexahedron(b, lo, hi, mat: str) -> None:
    """The six faces of the solid between two quads (corners in the same order), facing out."""
    centre = tuple(sum(c[i] for c in lo + hi) / 8 for i in range(3))
    faces = [lo, hi] + [[lo[i], lo[(i + 1) % 4], hi[(i + 1) % 4], hi[i]] for i in range(4)]
    for pts in faces:
        mid = tuple(sum(c[i] for c in pts) / 4 for i in range(3))
        if brlm.dot(brlm.normal(pts), brlm.sub(mid, centre)) < 0:
            pts = pts[::-1]
        b.face(pts, mat, TILE)


def parts(p: dict) -> dict[str, dict]:
    """{part: {"node": its nodes' name, "lods": [(px, triangle budget, Builder), ...],
    "collision": (budget, Builder)}}."""
    return {"main": {"node": SHAPE, "lods": [(px, BUDGETS[px], level(p, px)) for px in (600, 120, 30)],
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
