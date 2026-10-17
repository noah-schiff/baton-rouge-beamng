"""Write the routes tools/ingame/brmapDrive.lua drives, from the built roads.json.

A deck route follows a whole bridge chain (as bridges.py joins it) in its traffic
direction, with the deck height at every point, so the drive log can compare the car
with the deck under it. A ground route follows one named street. Corner points are
where the car parks to look for physics jitter far from the map origin.

    python tools/ingame/drive_routes.py [config.yaml] [out.json]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from brmap.bridges import _Chain, _network, merge_bridges  # noqa: E402
from brmap.config import load_config  # noqa: E402
from brmap.roads import Road, load_roads  # noqa: E402

DECK_ROUTES = {  # route name -> an OSM way id on the chain, and the speed to drive it at (m/s)
    "i10_split_flyover_and_river_bridge": (112804121, 22.0),  # the layer-3 flyover chains into the bridge
    "i10_river_bridge_other_way": (682521449, 22.0),
    "i110_north_from_split": (112804165, 22.0),
}
STREETS = {"east_lakeshore_drive": ("East Lakeshore Drive", 14.0), "dalrymple_drive": ("Dalrymple Drive", 14.0)}
CORNERS = {"corner_ne": (7600.0, 7600.0), "corner_sw": (-7600.0, -7600.0)}


def chain_route(decks, chains, way_id):
    for ch in chains:
        if not any(way_id in decks[i].osm_ids for i, _ in ch.members):
            continue
        z = np.empty(len(ch.xy))
        for (i, rev), (a, b) in zip(ch.members, ch.spans):
            z[a:b] = decks[i].nodes[::-1, 2] if rev else decks[i].nodes[:, 2]
        xyz = np.column_stack([ch.xy, z])
        oneway_reversed = decks[ch.members[0][0]].oneway and ch.members[0][1]
        return (xyz[::-1] if oneway_reversed else xyz), ch.label
    raise SystemExit(f"no deck chain has way {way_id}")


def street_route(ground, name):
    pieces = [Road(r.osm_ids, r.kind, r.nodes.copy(), r.width, r.lanes, r.oneway, True, r.layer, name)
              for r in ground if r.name == name]
    if not pieces:
        raise SystemExit(f"no street named {name}")
    return max(merge_bridges(pieces), key=lambda r: len(r.nodes)).nodes


def main():
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "dist/brmap_routes.json")
    roads = load_roads(Path(cfg.out_dir) / "roads.json")
    decks = [r for r in roads if r.is_bridge]
    ground = [r for r in roads if not r.is_bridge]
    members, attach = _network(decks)
    chains = [_Chain(decks, m, attach) for m in members]
    routes = []
    for name, (way, speed) in DECK_ROUTES.items():
        xyz, label = chain_route(decks, chains, way)
        routes.append({"name": name, "label": label, "speed": speed, "deck": True,
                       "points": np.round(xyz, 2).tolist()})
    for name, (street, speed) in STREETS.items():
        xyz = street_route(ground, street)
        routes.append({"name": name, "label": street, "speed": speed, "deck": False,
                       "points": np.round(xyz, 2).tolist()})
    nodes = np.vstack([r.nodes for r in ground])
    parks = []
    for name, (x, y) in CORNERS.items():
        k = int(np.argmin(np.hypot(nodes[:, 0] - x, nodes[:, 1] - y)))
        parks.append({"name": name, "point": np.round(nodes[k], 2).tolist(),
                      "toward": np.round(nodes[min(k + 1, len(nodes) - 1)], 2).tolist()})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"routes": routes, "parks": parks}), encoding="utf-8")
    for r in routes:
        p = np.asarray(r["points"])
        length = float(np.hypot(*np.diff(p[:, :2], axis=0).T).sum())
        print(f"{r['name']}: {r['label']}, {len(p)} points, {length:.0f} m, z {p[:, 2].min():.1f} to "
              f"{p[:, 2].max():.1f}, from ({p[0, 0]:.0f}, {p[0, 1]:.0f}) to ({p[-1, 0]:.0f}, {p[-1, 1]:.0f})")
    print("parks:", [(p["name"], p["point"][:2]) for p in parks], "->", out)


if __name__ == "__main__":
    main()
