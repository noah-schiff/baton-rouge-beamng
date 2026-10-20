"""The water tupelo (Nyssa aquatica), br_tupelo_a in src/brmap/data/custom_trees.yaml: the bald
cypress's companion in the wetlands and on the bayou banks (plan Task 20).

Params: height_m (the crown's top above the ground) and seed (every random choice). The shape's
name is the out folder's (out/models/<key>/).

* A swollen bottle base: radius x BOTTLE at the ground, rounding in to the trunk by BOTTLE_H along a
  smoothstep (still nearly twice the trunk half way up, unlike the cypress's flare), with LOBES
  shallow ridges FLUTE deep at the ground; below the ground it goes on at the swollen radius to SINK,
  so the tree never floats on a slope.
* A straight trunk tapering from TRUNK_R x height_m above the swell to TOP_SHARE of that at the
  leader's top, LEADER x height_m.
* A narrow oval crown: BRANCHES ascending branches up the trunk between CROWN's heights
  (golden-angle turns), longest (REACH x height_m) half way up the crown and shorter below and above
  it (to STUB of that), and TOP_BRANCHES short ones from the leader's top; along each, fans of leaf
  cards, smaller toward the crown's ends (to CARD_OVAL of the size half way up) (the stock
  t_beech_leaves sprays, SPRIGS, whose ovate leaves stand in for the tupelo's), raised a little
  but never above height_m.
* Wind weights (brtree), as on the cypress: wood near the trunk 0; r grows with the distance from
  the trunk up to BRANCH_ATTEN at ATTEN_REACH x height_m out; g one phase per branch; b the
  leaf-edge flutter.

Levels of detail as the cypress's (brtree.TREE_BUDGETS): br_tupelo_a_a600 (at most 3,000
triangles), _a250 (1,200) and _a80 (400), the same skeleton each with fewer, larger cards;
Colmesh-1, the octagonal trunk prism COLLISION (brtree.trunk_collision, custom_trees.yaml's
collision); and the billboard brtree.BILLBOARD, which the level asks the game to fill with
imposters (trees.IMPOSTERS). The trunk, branches, fans, wind and nodes are brtree's, as the
cypress's.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

import brtree  # noqa: E402

BARK, LEAVES = "br_tree_tupelo_bark", "br_tree_tupelo_leaves"
BUDGETS, BILLBOARD = brtree.TREE_BUDGETS, brtree.BILLBOARD
COLLISION = (0.9, 8.0)              # m: the trunk prism's radius and top (custom_trees.yaml collision)

# proportions, as shares of height_m unless in m
TRUNK_R = 1 / 45                    # trunk radius above the swell: 0.4 m at 18 m
BOTTLE, BOTTLE_H = 2.8, 2.5         # x radius at the ground, back to the trunk by 2.5 m
LOBES, FLUTE = 5, 0.08              # ridges round the base, their depth at the ground
SINK = 0.5                          # m of trunk below the ground
LEADER, TOP_SHARE = 0.92, 0.3       # the trunk's top, and its radius there as a share of the trunk's
CROWN = (0.45, 0.9)                 # the branches' heights
BRANCHES, TOP_BRANCHES = 18, 4
REACH = 0.10                        # the longest branch: 1.8 m at 18 m, half way up the crown
STUB = 0.25                         # the lowest and highest branches' length, as a share of REACH
CARD_OVAL = 0.6                     # their cards' size, as a share of those half way up the crown
RISE = (0.35, 0.75)                 # a branch climbs this much per m out at its base ...
BEND_OUT = (0.3, 0.7)               # ... and keeps this share of the climb to its tip
BRANCH_R = (0.05, 0.05, 0.015)      # m: base radius + this x (length / reach), and the tip
BRANCH_ATTEN, ATTEN_REACH = 0.2, 0.25   # r at the crown's edge, ATTEN_REACH x height_m out
CARD = 1.2                          # m: a card's size (the square root of its area) at a600
BARK_TILE = (1.0, 4.0)              # m per repeat of t_poplar_bark (512 x 2048) round and along

# the sprays in t_beech_leaves (square; the opacity map is 2048 px, the colour 1024): (u0, v0, u1, v1)
# with the stem at v0, measured from its opacity map, and how often each is picked; the fifth spray
# (top left) lies on its side and is not used
SPRIGS = [((0.143, 0.006, 0.625, 0.628), 0.35), ((0.690, 0.006, 0.951, 0.531), 0.25),
          ((0.579, 0.535, 0.992, 0.994), 0.25), ((0.006, 0.180, 0.146, 0.575), 0.15)]

# per level of detail: trunk sides, flutes, rings below BOTTLE_H and the step above it; branch sides
# (0: none) and segments; the fans along each branch (shares of its length), cards per fan, card size
LODS = {
    600: dict(sides=16, lobes=LOBES, rings=(-SINK, 0.0, 0.5, 1.0, 1.5, 2.0), step=2.5, branch=4, segments=3,
              fans=(0.3, 0.45, 0.6, 0.75, 0.9, 1.0), cards=6, size=1.0),
    250: dict(sides=9, lobes=0, rings=(-SINK, 0.0, 0.8, 1.6), step=5.0, branch=3, segments=2,
              fans=(0.45, 0.75, 1.0), cards=4, size=1.4),
    80: dict(sides=5, lobes=0, rings=(-SINK, 0.0, 1.2), step=10.0, branch=0, segments=0,
             fans=(0.6, 1.0), cards=3, size=2.0),
}


def trunk_radius(z: float, r: float, top: float) -> float:
    """The trunk's radius at z: the bottle below BOTTLE_H (a smoothstep from BOTTLE x r at the ground
    in to r), then a straight taper to the top."""
    if z <= BOTTLE_H:
        t = min(max(z / BOTTLE_H, 0.0), 1.0)
        return r * (1.0 + (BOTTLE - 1.0) * (1.0 - t * t * (3.0 - 2.0 * t)))
    return r * (1.0 - (1.0 - TOP_SHARE) * (z - BOTTLE_H) / (top - BOTTLE_H))


def plan(params: dict) -> dict:
    """The skeleton every level of detail shares: trunk_r and top (m) and the branches (z, heading,
    length, rise, bend, oval (the length's share of the longest, by height), phase, seed)."""
    h, seed = float(params["height_m"]), params["seed"]
    r, top = TRUNK_R * h, LEADER * h
    branches = []
    z0, z1 = CROWN[0] * h, CROWN[1] * h
    for i in range(BRANCHES):
        rng = brtree.seeded(seed, "branch", i)
        t = (i + 0.5 + rng.uniform(-0.35, 0.35)) / BRANCHES
        oval = STUB + (1.0 - STUB) * math.sin(math.pi * t)          # longest half way up the crown
        branches.append(dict(z=z0 + t * (z1 - z0), heading=i * brtree.GOLDEN + rng.uniform(-0.4, 0.4),
                             length=REACH * h * oval * rng.uniform(0.85, 1.15), rise=rng.uniform(*RISE),
                             bend=rng.uniform(*BEND_OUT), oval=oval, phase=rng.random(), seed=(seed, "branch", i)))
    for j in range(TOP_BRANCHES):
        rng = brtree.seeded(seed, "top", j)
        branches.append(dict(z=rng.uniform(0.88, LEADER) * h,
                             heading=2 * math.pi * (j + rng.uniform(-0.3, 0.3)) / TOP_BRANCHES,
                             length=REACH * h * rng.uniform(0.3, 0.45), rise=rng.uniform(0.5, 0.9),
                             bend=rng.uniform(0.5, 1.0), oval=STUB, phase=rng.random(), seed=(seed, "top", j)))
    return {"height": h, "trunk_r": r, "top": top, "branches": branches}


def _branch_curve(p: dict, br: dict):
    """The branch's quadratic curve: from inside the trunk, climbing, bending toward level."""
    heading = (math.cos(br["heading"]), math.sin(br["heading"]), 0.0)
    rt = trunk_radius(br["z"], p["trunk_r"], p["top"])
    p0 = brtree.add(brtree.mul(heading, 0.3 * rt), (0.0, 0.0, br["z"]))
    climb = br["length"] * br["rise"]
    p1 = brtree.add(p0, brtree.add(brtree.mul(heading, br["length"] / 2), (0.0, 0.0, climb / 2)))
    p2 = brtree.add(p0, brtree.add(brtree.mul(heading, br["length"]), (0.0, 0.0, climb * (0.5 + br["bend"] / 2))))
    return p0, p1, p2


def _cards(br: dict, fan: int, n: int, size: float):
    """n cards for one fan: spread over +-60 degrees about the branch, raised a little, rolled
    some way off level, a spray each."""
    out = []
    for k in range(n):
        rng = brtree.seeded(*br["seed"], "fan", fan, "card", k)
        yaw = (1.05 * (2 * k / (n - 1) - 1) if n > 1 else 0.0) + rng.uniform(-0.15, 0.15)
        pitch = rng.uniform(-0.1, 0.5)
        roll = rng.uniform(-0.6, 0.6)
        rect = brtree.pick(rng, SPRIGS)
        (u0, v0, u1, v1) = rect
        aspect = (u1 - u0) / (v1 - v0)                        # width / length; the texture is square
        s = CARD * size * rng.uniform(0.8, 1.2)
        out.append((yaw, pitch, roll, s / math.sqrt(aspect), s * math.sqrt(aspect), rect))
    return out


def _max_pitch(base, length: float, width: float, h: float) -> float:
    """The steepest a card from base may rise and keep its tip corners under h."""
    return math.asin(min(max((h - base[2] - width / 2) / length, -1.0), 1.0))


def build(params: dict) -> dict[int, brtree.TreeBuilder]:
    """{px: TreeBuilder} for each level of detail in LODS."""
    p = plan(params)
    h, r, top = p["height"], p["trunk_r"], p["top"]
    centre = (0.0, 0.0, (CROWN[0] + 1.0) / 2 * h)
    atten = brtree.radial_atten(BRANCH_ATTEN, ATTEN_REACH * h)

    def girth(z: float) -> float:
        return trunk_radius(z, r, top)

    out = {}
    for px, lod in LODS.items():
        b = brtree.TreeBuilder()
        brtree.trunk(b, girth, r, top, BOTTLE_H, lod["rings"], lod["step"], lod["sides"], BARK, tile=BARK_TILE,
                     lobes=lod["lobes"], flute=FLUTE)
        for br in p["branches"]:
            curve = _branch_curve(p, br)
            if lod["branch"]:
                brtree.branch(b, curve, brtree.branch_taper(br["length"], REACH * h, BRANCH_R), lod["branch"],
                              lod["segments"], BARK, atten, br["phase"], tile=BARK_TILE)
            pull = (lod["size"] - 1.0) * CARD * 0.8         # larger cards start further in
            card = CARD_OVAL + (1.0 - CARD_OVAL) * (br["oval"] - STUB) / (1.0 - STUB)   # smaller at the crown's ends
            for f, s in enumerate(lod["fans"]):
                base, flat = brtree.fan_base(curve, s, pull, girth)
                cards = [(yaw, min(pitch, _max_pitch(base, length, width, h)), roll, length, width, rect)
                         for yaw, pitch, roll, length, width, rect in _cards(br, f, lod["cards"], lod["size"] * card)]
                brtree.leaf_fan(b, base, flat, cards, LEAVES, atten, br["phase"], centre)
        out[px] = b
    return out


brtree.export_tree(build, BUDGETS, collision=brtree.trunk_collision(*COLLISION, BARK), billboard=BILLBOARD)
