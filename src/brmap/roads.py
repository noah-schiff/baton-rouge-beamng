"""Stage 5: OpenStreetMap ways -> Road records in level coordinates.

Ground roads take their z from the terrain. Bridge roads keep z = NaN here;
bridges.py gives them a deck profile before roads.json is written.
"""
from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from shapely.geometry import LineString
from shapely.geometry import box as shapely_box

from .geo import Extent
from .terrain import load_heights, sample_height

log = logging.getLogger(__name__)

OUT_DIR = Path("data/out")
RAW_DIR = Path("data/raw")

MAIN_CLASSES = ("motorway", "trunk", "primary", "secondary", "tertiary")
KEPT = (set(MAIN_CLASSES) | {"unclassified", "residential", "living_street", "service"}
        | {f"{k}_link" for k in MAIN_CLASSES})
DEFAULT_LANES = {"motorway": 4, "trunk": 4, "primary": 4, "secondary": 2, "tertiary": 2,
                 "unclassified": 2, "residential": 2, "living_street": 2, "service": 1}
LANE_WIDTH = {"motorway": 3.6, "trunk": 3.6, "primary": 3.3, "secondary": 3.3}  # else 3.0
SHOULDER = {"motorway": 3.0, "trunk": 3.0}                                       # else 0.5
SERVICE_WIDTH = 4.0
IMPLIED_ONEWAY = {"motorway", "motorway_link"}
NOT_BRIDGE = {"no", "false", "0"}
MPH = 0.44704        # m/s
KMH = 1 / 3.6        # m/s; a bare OSM maxspeed number is km/h
MAX_SPACING = 20.0   # m between road nodes
SIMPLIFY_TOL = 0.5   # m
DUPLICATE_TOL = 0.05  # m


@dataclass
class Road:
    osm_ids: list[int]
    kind: str
    nodes: np.ndarray  # (N, 3) level x, y, z
    width: float
    lanes: int
    oneway: bool
    is_bridge: bool
    layer: int
    name: str | None
    maxspeed: float | None = None  # m/s, from the OSM maxspeed tag; None when untagged
    covered: bool = False          # runs under a building or roof (tunnel=building_passage, covered=yes)

    def to_dict(self) -> dict:
        return {"osm_ids": [int(i) for i in self.osm_ids], "kind": self.kind,
                "nodes": [[float(v) for v in p] for p in self.nodes],
                "width": float(self.width), "lanes": int(self.lanes), "oneway": bool(self.oneway),
                "is_bridge": bool(self.is_bridge), "layer": int(self.layer), "name": self.name,
                "maxspeed": None if self.maxspeed is None else float(self.maxspeed),
                "covered": bool(self.covered)}

    @classmethod
    def from_dict(cls, d: dict) -> "Road":
        speed = d.get("maxspeed")  # roads.json files from before maxspeed have no key
        return cls(osm_ids=[int(i) for i in d["osm_ids"]], kind=d["kind"],
                   nodes=np.array(d["nodes"], dtype=float).reshape(-1, 3), width=float(d["width"]),
                   lanes=int(d["lanes"]), oneway=bool(d["oneway"]), is_bridge=bool(d["is_bridge"]),
                   layer=int(d["layer"]), name=d.get("name"),
                   maxspeed=None if speed is None else float(speed), covered=bool(d.get("covered", False)))


def base_kind(kind: str) -> str:
    return kind[: -len("_link")] if kind.endswith("_link") else kind


def _parse_int_list(value) -> int | None:
    """Largest integer in a tag like '3' or '2;3'; None when nothing parses."""
    if value is None:
        return None
    best = None
    for part in str(value).split(";"):
        try:
            n = int(float(part.strip()))
        except ValueError:
            continue
        best = n if best is None else max(best, n)
    return best


def parse_maxspeed(value) -> float | None:
    """m/s from an OSM maxspeed tag: '35 mph', or '50' / '50 km/h' (km/h). Anything else
    ('signals', 'none', '35 mph;45 mph', zero) is None, so the road gets its class default."""
    if value is None:
        return None
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(mph|km/h|kmh|kph)?\s*", str(value).lower())
    if not m or float(m[1]) <= 0:
        return None
    return float(m[1]) * (MPH if m[2] == "mph" else KMH)


def oneway_direction(tags: dict) -> int:
    """1 = one-way along the drawn direction, -1 = one-way against it, 0 = two-way."""
    v = str(tags.get("oneway", "")).strip().lower()
    if v in ("yes", "true", "1"):
        return 1
    if v in ("-1", "reverse"):
        return -1
    if v in ("no", "false", "0"):
        return 0
    implied = tags.get("highway") in IMPLIED_ONEWAY or tags.get("junction") == "roundabout"
    return 1 if implied else 0


def road_width(tags: dict) -> tuple[float, int]:
    """(width in m, lane count) from OSM tags; a parseable lanes tag always wins."""
    kind = tags.get("highway", "")
    base = base_kind(kind)
    lanes = _parse_int_list(tags.get("lanes"))
    if lanes is not None and lanes < 1:
        lanes = None
    if lanes is None:
        if kind == "service":
            return SERVICE_WIDTH, 1
        if kind.endswith("_link"):
            lanes = 1
        else:
            lanes = DEFAULT_LANES.get(base, 2)
            if oneway_direction(tags) != 0:
                lanes = max(1, lanes // 2)
    return lanes * LANE_WIDTH.get(base, 3.0) + SHOULDER.get(base, 0.5), lanes


def parse_ways(osm_json: dict) -> list[dict]:
    """Highway ways as {'id', 'tags', 'lonlat': [(lon, lat), ...]}."""
    out = []
    for el in osm_json.get("elements", []):
        tags = el.get("tags") or {}
        if el.get("type") != "way" or "highway" not in tags:
            continue
        lonlat = [(float(g["lon"]), float(g["lat"])) for g in el.get("geometry") or [] if g]
        out.append({"id": int(el["id"]), "tags": tags, "lonlat": lonlat})
    return out


def clip_to_box(xy: np.ndarray, extent: Extent) -> list[np.ndarray]:
    """The parts of a polyline inside the map square (a way can leave and come back)."""
    if len(xy) < 2:
        return []
    line = LineString(xy)
    if line.length == 0:
        return []
    h = extent.half_size
    pieces: list[np.ndarray] = []

    def collect(g):
        if g.is_empty:
            return
        if g.geom_type == "LineString":
            if g.length > 0:
                pieces.append(np.asarray(g.coords, dtype=float)[:, :2])
        elif hasattr(g, "geoms"):
            for sub in g.geoms:
                collect(sub)

    collect(line.intersection(shapely_box(-h, -h, h, h)))
    return pieces


def _drop_duplicates(xy: np.ndarray) -> np.ndarray:
    keep = [0]
    for i in range(1, len(xy)):
        if math.dist(xy[i], xy[keep[-1]]) >= DUPLICATE_TOL:
            keep.append(i)
    return xy[keep]


def _simplify(xy: np.ndarray) -> np.ndarray:
    return np.asarray(LineString(xy).simplify(SIMPLIFY_TOL, preserve_topology=False).coords)[:, :2]


def densify(xy: np.ndarray, max_spacing: float = MAX_SPACING) -> np.ndarray:
    out = [xy[0]]
    for a, b in zip(xy[:-1], xy[1:]):
        k = max(1, math.ceil(math.dist(a, b) / max_spacing - 1e-9))
        out.extend(a + (b - a) * (j / k) for j in range(1, k + 1))
    return np.asarray(out)


def build_roads(osm_json: dict, extent: Extent, heights: np.ndarray) -> list[Road]:
    roads: list[Road] = []
    for way in parse_ways(osm_json):
        tags, wid = way["tags"], way["id"]
        kind = tags["highway"]
        if kind not in KEPT:
            continue
        if len(way["lonlat"]) < 2:
            log.warning("skipped way %d: fewer than 2 nodes", wid)
            continue
        lon, lat = np.asarray(way["lonlat"]).T
        xy = np.column_stack(extent.to_level(lon, lat))
        direction = oneway_direction(tags)
        if direction == -1:
            xy = xy[::-1]
        xy = _drop_duplicates(xy)
        if len(xy) < 2:
            log.warning("skipped way %d: all nodes in the same place", wid)
            continue
        width, lanes = road_width(tags)
        is_bridge = str(tags.get("bridge", "no")).strip().lower() not in NOT_BRIDGE
        layer = _parse_int_list(tags.get("layer")) or 0
        name = tags.get("name") or tags.get("ref")
        maxspeed = parse_maxspeed(tags.get("maxspeed"))
        covered = (str(tags.get("tunnel", "no")).strip().lower() not in NOT_BRIDGE
                   or str(tags.get("covered", "no")).strip().lower() not in NOT_BRIDGE)
        for piece in clip_to_box(xy, extent):
            piece = _drop_duplicates(piece)
            if len(piece) < 2:
                continue
            piece = densify(_drop_duplicates(_simplify(piece)))
            if len(piece) < 2:
                continue
            z = (np.full(len(piece), np.nan) if is_bridge
                 else sample_height(heights, extent, piece[:, 0], piece[:, 1]))
            roads.append(Road([wid], kind, np.column_stack([piece, z]), width, lanes,
                              direction != 0, is_bridge, layer, name, maxspeed, covered))
    return roads


def save_roads(roads: list[Road], path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([r.to_dict() for r in roads]), encoding="utf-8")
    return path


def load_roads(path=OUT_DIR / "roads.json") -> list[Road]:
    return [Road.from_dict(d) for d in json.loads(Path(path).read_text(encoding="utf-8"))]


def run(cfg, raw_dir=None, out_dir=None) -> None:
    extent = Extent.from_config(cfg)
    raw_dir, out_dir = Path(raw_dir or cfg.raw_dir), Path(out_dir or cfg.out_dir)
    osm = json.loads((raw_dir / "osm.json").read_text(encoding="utf-8"))
    heights = load_heights(out_dir=out_dir)
    roads = build_roads(osm, extent, heights)
    pieces = sum(r.is_bridge for r in roads)
    from . import bridges  # bridges imports Road from here

    # decks over water clear the water surface, so they see the terrain before carving
    roads = bridges.apply_bridges(roads, load_heights(raw=True, out_dir=out_dir), extent, cfg)
    save_roads(roads, out_dir / "roads.json")
    kinds = Counter(r.kind for r in roads)
    log.info("%d roads: %s", len(roads), ", ".join(f"{k} {v}" for k, v in kinds.most_common()))
    decks = [r for r in roads if r.is_bridge]
    log.info("%d bridge pieces merged into %d decks; highest deck %.1f m", pieces, len(decks),
             max((float(r.nodes[:, 2].max()) for r in decks), default=float("nan")))
