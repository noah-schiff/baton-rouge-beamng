"""The bald cypress (Taxodium distichum), br_cypress_a in src/brmap/data/custom_trees.yaml: the
first custom forest item, the swamp tree of the wetlands (plan Task 9, spike S2).

Params: height_m (the crown's top above the ground), seed (every random choice), knees (how
many, 6 to 12) and moss (optional, default false: Spanish moss hangs from the branches, as on
br_cypress_b). The shape's name is the out folder's (out/models/<key>/), so one script can serve
several keys.

* A flared, fluted base: radius x FLARE at the ground, back to the trunk by FLARE_H, with
  LOBES buttress ridges FLUTE deep at the ground; below the ground it goes on at the flared
  radius to SINK, so the tree never floats on a slope.
* A straight trunk tapering from TRUNK_R x height_m above the flare to TOP_SHARE of that at
  the leader's top, LEADER x height_m.
* knees cone knees 0.3-0.9 m tall within 4 m of the trunk, clear of the flare, their ring
  KNEE_SINK below the ground.
* A narrow, flat-topped crown over the top 60 %: BRANCHES horizontal branches up the trunk
  (golden-angle turns), reaching REACH x height_m, a little shorter low down, and TOP_BRANCHES
  short ones fanning out from the leader's top; along each, fans of feathery needle cards
  (the stock t_cypress_leaves sprays, SPRIGS), mostly level, every third stood on its edge.
* With moss, MOSS_AT strands of Spanish moss under each of the BRANCHES main branches
  (brtree.moss_strand, the moss card): brtree.MOSS_LENGTH long, brtree.MOSS_WIDTH wide, their
  tops just under the branch, turned and leaning a little at random.
* Wind weights (brtree): wood near the trunk 0; r grows with the distance from the trunk up
  to BRANCH_ATTEN at ATTEN_REACH x height_m out; g one phase per branch; b the leaf-edge
  flutter. A moss strand takes its branch's r and g where it hangs, and sways from b 0 at its
  top to brtree.MOSS_FLUTTER at its foot.

Levels of detail (brtree.TREE_BUDGETS): br_cypress_a_a600 (at most 3,000 triangles), _a250
(1,200) and _a80 (400), the same skeleton each with fewer, larger cards (and MOSS_STRANDS moss
strands per branch); Colmesh-1, the octagonal trunk prism COLLISION (brtree.trunk_collision,
custom_trees.yaml's collision); and the billboard brtree.BILLBOARD, which the level asks the game
to fill with imposters (trees.IMPOSTERS). The water tupelo (water_tupelo.py) shares this layout and
the helpers in brtree.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

import brtree  # noqa: E402

BARK, LEAVES, MOSS = "br_tree_cypress_bark", "br_tree_cypress_leaves", "br_tree_moss"
BUDGETS, BILLBOARD = brtree.TREE_BUDGETS, brtree.BILLBOARD
COLLISION = (0.7, 8.0)              # m: the trunk prism's radius and top (custom_trees.yaml collision)
KNEES = (6, 12)                     # the knees param's range

# proportions, as shares of height_m unless in m
TRUNK_R = 1 / 44                    # trunk radius above the flare: 0.5 m at 22 m
FLARE, FLARE_H = 2.2, 1.5           # x radius at the ground, back to the trunk by 1.5 m
LOBES, FLUTE = 6, 0.18              # buttress ridges round the base, their depth at the ground
SINK = 0.5                          # m of trunk below the ground
LEADER, TOP_SHARE = 0.93, 0.25      # the trunk's top, and its radius there as a share of the trunk's
CROWN = (0.4, 0.9)                  # the branches' heights: the top 60 %, the leader's top flat
BRANCHES, TOP_BRANCHES = 20, 5
REACH = 0.11                        # branch length: 2.4 m at 22 m
KNEE_H, KNEE_REACH, KNEE_SINK = (0.3, 0.9), 4.0, 0.3   # m
KNEE_CLEAR = 0.3                    # m between a knee and the flared base
BRANCH_R = (0.06, 0.05, 0.015)      # m: base radius + this x (length / reach), and the tip
BRANCH_ATTEN, ATTEN_REACH = 0.2, 0.3   # r at the crown's edge, ATTEN_REACH x height_m out
CARD = 1.2                          # m: a card's size (the square root of its area) at a600
BARK_TILE = (1.0, 4.0)              # m per repeat of t_fir_bark (512 x 2048) round and along
MOSS_AT = (0.55, 0.8, 0.35, 0.95)   # where along a main branch its moss strands hang, in the order the LODs drop them
MOSS_JITTER = 0.06                  # +- share of the branch's length
MOSS_STRANDS = {600: 4, 250: 2, 80: 1}   # strands per main branch at each level of detail

# the sprays in t_cypress_leaves_b.color (1024 x 2048): (u0, v0, u1, v1) with the stem at v0,
# measured from its opacity map, and how often each is picked
SPRIGS = [((0.500, 0.725, 0.988, 0.994), 0.25), ((0.039, 0.678, 0.469, 0.990), 0.25),
          ((0.508, 0.004, 0.980, 0.367), 0.20), ((0.594, 0.539, 0.988, 0.701), 0.10),
          ((0.094, 0.488, 0.477, 0.664), 0.10), ((0.504, 0.381, 0.969, 0.521), 0.10)]

# per level of detail: trunk sides, flutes, rings below FLARE_H and the step above it; knee and
# branch sides (0: none) and branch segments; the fans along each branch (shares of its length),
# cards per fan, and the card size
LODS = {
    600: dict(sides=18, lobes=LOBES, rings=(-SINK, 0.0, 0.25, 0.55, 0.9), step=2.5, knee=6, branch=4, segments=3,
              fans=(0.3, 0.45, 0.6, 0.75, 0.9, 1.0), cards=5, size=1.0),
    250: dict(sides=9, lobes=0, rings=(-SINK, 0.0, 0.6), step=5.0, knee=4, branch=3, segments=2,
              fans=(0.4, 0.7, 1.0), cards=4, size=1.45),
    80: dict(sides=5, lobes=0, rings=(-SINK,), step=10.0, knee=0, branch=0, segments=0,
             fans=(0.55, 1.0), cards=3, size=2.0),
}


def trunk_radius(z: float, r: float, top: float) -> float:
    """The trunk's radius at z: the flare below FLARE_H, then a straight taper to the top."""
    if z <= FLARE_H:
        return brtree.flare(z, r, FLARE, FLARE_H)
    return r * (1.0 - (1.0 - TOP_SHARE) * (z - FLARE_H) / (top - FLARE_H))


def plan(params: dict) -> dict:
    """The skeleton every level of detail shares: trunk_r and top (m), the knees (x, y, base
    radius, height) and the branches (z, heading, length, rise, end, phase, seed)."""
    h, seed, knees = float(params["height_m"]), params["seed"], int(params["knees"])
    if not KNEES[0] <= knees <= KNEES[1]:
        raise ValueError(f"knees must be {KNEES[0]} to {KNEES[1]}, not {knees}")
    r, top = TRUNK_R * h, LEADER * h
    out = []
    for i in range(knees):
        rng = brtree.seeded(seed, "knee", i)
        kh = rng.uniform(*KNEE_H)
        kr = kh * rng.uniform(0.3, 0.45)
        d = rng.uniform(FLARE * r + KNEE_CLEAR + kr, KNEE_REACH - kr)
        a = 2 * math.pi * (i + rng.uniform(-0.3, 0.3)) / knees
        out.append((d * math.cos(a), d * math.sin(a), kr, kh))
    branches = []
    z0, z1 = CROWN[0] * h, CROWN[1] * h
    for i in range(BRANCHES):
        rng = brtree.seeded(seed, "branch", i)
        t = (i + 0.5 + rng.uniform(-0.35, 0.35)) / BRANCHES
        length = REACH * h * (0.72 + 0.28 * min(t / 0.45, 1.0)) * rng.uniform(0.85, 1.15)
        branches.append(dict(z=z0 + t * (z1 - z0), heading=i * brtree.GOLDEN + rng.uniform(-0.4, 0.4),
                             length=length, rise=rng.uniform(0.05, 0.25), end=rng.uniform(0.2, 0.8),
                             phase=rng.random(), seed=(seed, "branch", i)))
    for j in range(TOP_BRANCHES):
        rng = brtree.seeded(seed, "top", j)
        branches.append(dict(z=rng.uniform(0.89, LEADER) * h,
                             heading=2 * math.pi * (j + rng.uniform(-0.3, 0.3)) / TOP_BRANCHES,
                             length=REACH * h * rng.uniform(0.45, 0.65), rise=rng.uniform(0.1, 0.3),
                             end=rng.uniform(0.5, 1.0), phase=rng.random(), seed=(seed, "top", j)))
    moss = []
    if params.get("moss", False):
        lean = brtree.MOSS_LEAN
        for i in range(BRANCHES):
            for rank, at in enumerate(MOSS_AT):
                rng = brtree.seeded(seed, "moss", i, rank)
                moss.append(dict(branch=i, rank=rank, s=at + rng.uniform(-MOSS_JITTER, MOSS_JITTER),
                                 yaw=rng.uniform(0, 2 * math.pi), length=rng.uniform(*brtree.MOSS_LENGTH),
                                 width=rng.uniform(*brtree.MOSS_WIDTH), u=rng.random(),
                                 lean=(rng.uniform(-lean, lean), rng.uniform(-lean, lean))))
    return {"height": h, "trunk_r": r, "top": top, "knees": out, "branches": branches, "moss": moss}


def _branch_curve(p: dict, br: dict):
    """The branch's quadratic curve: from inside the trunk, rising at first, ending near level."""
    heading = (math.cos(br["heading"]), math.sin(br["heading"]), 0.0)
    rt = trunk_radius(br["z"], p["trunk_r"], p["top"])
    p0 = brtree.add(brtree.mul(heading, 0.3 * rt), (0.0, 0.0, br["z"]))
    p1 = brtree.add(p0, brtree.add(brtree.mul(heading, br["length"] / 2), (0.0, 0.0, br["length"] * br["rise"] / 2)))
    p2 = brtree.add(p0, brtree.add(brtree.mul(heading, br["length"]),
                                   (0.0, 0.0, br["length"] * br["rise"] * br["end"])))
    return p0, p1, p2


def _cards(br: dict, fan: int, n: int, size: float):
    """n cards for one fan: spread over +-63 degrees about the branch, mostly level, every
    third stood on its edge, a sprig each."""
    out = []
    for k in range(n):
        rng = brtree.seeded(*br["seed"], "fan", fan, "card", k)
        yaw = (1.1 * (2 * k / (n - 1) - 1) if n > 1 else 0.0) + rng.uniform(-0.15, 0.15)
        pitch = rng.uniform(-0.25, 0.35)
        roll = (1 if rng.random() < 0.5 else -1) * rng.uniform(1.0, 1.4) if k % 3 == 2 else rng.uniform(-0.45, 0.45)
        rect = brtree.pick(rng, SPRIGS)
        (u0, v0, u1, v1) = rect
        aspect = (u1 - u0) * 1024 / ((v1 - v0) * 2048)        # width / length in texels
        s = CARD * size * rng.uniform(0.8, 1.2)
        out.append((yaw, pitch, roll, s / math.sqrt(aspect), s * math.sqrt(aspect), rect))
    return out


def build(params: dict) -> dict[int, brtree.TreeBuilder]:
    """{px: TreeBuilder} for each level of detail in LODS."""
    p = plan(params)
    h, r, top = p["height"], p["trunk_r"], p["top"]
    centre = (0.0, 0.0, (CROWN[0] + 1.0) / 2 * h)
    atten = brtree.radial_atten(BRANCH_ATTEN, ATTEN_REACH * h)

    def girth(z: float) -> float:
        return trunk_radius(z, r, top)

    def taper(br: dict):
        return brtree.branch_taper(br["length"], REACH * h, BRANCH_R)

    def hang_moss(b: brtree.TreeBuilder, px: int) -> None:
        for m in p["moss"]:
            if m["rank"] >= MOSS_STRANDS[px]:
                continue
            br = p["branches"][m["branch"]]
            at = brtree.bezier(*_branch_curve(p, br), m["s"])
            point = brtree.sub(at, (0.0, 0.0, taper(br)(m["s"])))          # under the branch
            brtree.moss_strand(b, point, m["yaw"], m["length"], m["width"], m["u"], MOSS, atten=atten(point),
                               phase=br["phase"], lean=m["lean"])

    out = {}
    for px, lod in LODS.items():
        b = brtree.TreeBuilder()
        brtree.trunk(b, girth, r, top, FLARE_H, lod["rings"], lod["step"], lod["sides"], BARK, tile=BARK_TILE,
                     lobes=lod["lobes"], flute=FLUTE)
        if lod["knee"]:
            for x, y, kr, kh in p["knees"]:
                brtree.cone(b, (x, y, 0.0), kr, kh, lod["knee"], BARK, sink=KNEE_SINK, tile=BARK_TILE)
        for br in p["branches"]:
            curve = _branch_curve(p, br)
            if lod["branch"]:
                brtree.branch(b, curve, taper(br), lod["branch"], lod["segments"], BARK, atten, br["phase"],
                              tile=BARK_TILE)
            pull = (lod["size"] - 1.0) * CARD * 0.8         # larger cards start further in
            for f, s in enumerate(lod["fans"]):
                base, flat = brtree.fan_base(curve, s, pull, girth)
                cards = _cards(br, f, lod["cards"], lod["size"])
                brtree.leaf_fan(b, base, flat, cards, LEAVES, atten, br["phase"], centre)
        hang_moss(b, px)
        out[px] = b
    return out


brtree.export_tree(build, BUDGETS, collision=brtree.trunk_collision(*COLLISION, BARK), billboard=BILLBOARD)
