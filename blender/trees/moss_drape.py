"""Spanish moss on a live oak: the br_moss_<oak> overlays of src/brmap/data/custom_trees.yaml (plan
Task 20). The level stands one on every live oak whose trees.json entry has moss, with the oak's
position, rotation and scale (trees.level_content), so its strands hang from that oak's branches.

Params: hang_points, [[x, y, z], ...] under the oak's branches in its shape's coordinates (models.run
fills them in from src/brmap/data/moss_points.json, written by tools/extract_moss_points.py, for the
entry's hang_points_from), and seed (optional, default 0). The shape's name is the out folder's
(out/models/<key>/).

* From each point hang STRANDS[300] strands of moss (brtree.moss_strand on the generated moss
  card), brtree.MOSS_LENGTH long and brtree.MOSS_WIDTH wide, as on the mossy cypress, the longer
  first; their top edges EMBED above the point, so no gap shows under the bark; the second turned
  TURN (70 to 110 degrees) from the first, so the clump reads from every side; each leaning up to
  brtree.MOSS_LEAN m per m down.
* Wind: r 0 (the stock oaks' bark carries no branch bending: its COLOR0 is 0, so the strands' tops
  stay on it), g a phase per point, b from 0 at the top to brtree.MOSS_FLUTTER at the foot. The
  overlay's item takes the oak's own TSForestItemData fields, so it sways with its trunk.

Nodes: <shape>_a300 (every strand, at most BUDGETS[300] triangles), <shape>_a100 (the first strand
of each point) and nulldetail30. No collision mesh and no billboard: at a distance the oak's own
imposter shows.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

import brtree  # noqa: E402

MOSS = "br_tree_moss"
BUDGETS = {300: 600, 100: 300}
STRANDS = {300: 2, 100: 1}          # strands per hang point at each level of detail
NULL = "nulldetail30"
EMBED = 0.05                        # m: a strand's top edge sits this far above its point, in the bark
TURN = (math.radians(70), math.radians(110))   # between a point's two strands


def plan(params: dict) -> list[list[dict]]:
    """For each hang point, its strands (top, yaw, length, width, u, lean, phase), the longer first."""
    points, seed = params.get("hang_points") or [], params.get("seed", 0)
    if not points:
        raise ValueError("moss_drape: no hang_points (models.run fills them in from moss_points.json)")
    out, lean = [], brtree.MOSS_LEAN
    for i, (x, y, z) in enumerate(points):
        rng = brtree.seeded(seed, "point", i)
        yaw, phase = rng.uniform(0, 2 * math.pi), rng.random()
        lengths = sorted((rng.uniform(*brtree.MOSS_LENGTH) for _ in range(STRANDS[300])), reverse=True)
        strands = []
        for k, length in enumerate(lengths):
            strands.append(dict(top=(x, y, z + EMBED), yaw=yaw + k * rng.uniform(*TURN), length=length,
                                width=rng.uniform(*brtree.MOSS_WIDTH), u=rng.random(), phase=phase,
                                lean=(rng.uniform(-lean, lean), rng.uniform(-lean, lean))))
        out.append(strands)
    return out


def build(params: dict) -> dict[int, brtree.TreeBuilder]:
    """{px: TreeBuilder} for each level of detail in STRANDS."""
    points = plan(params)
    out = {}
    for px, count in STRANDS.items():
        b = brtree.TreeBuilder()
        for strands in points:
            for s in strands[:count]:
                brtree.moss_strand(b, s["top"], s["yaw"], s["length"], s["width"], s["u"], MOSS, atten=0.0,
                                   phase=s["phase"], lean=s["lean"])
        out[px] = b
    return out


brtree.export_tree(build, BUDGETS, tail=[(NULL, None)])
