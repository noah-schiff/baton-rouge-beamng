"""Stage 7: assemble the BeamNG level folder from the stage outputs, write the level's licence plate
design beside it, and zip both as a mod.

Feature stages plug in through FEATURE_MODULES: for each name whose out/<name>.json
exists, brmap.<name>.level_content(cfg, data_dir, root, heights, extent) writes that
feature's own files (shapes, materials, forest files) under the level folder `root`
and returns {SimGroup name: [object, ...]}. Each object has "class" and a stable
"persistentId" and no "__parent"; the groups are listed in the MissionGroup and
written like the built-in ones. `heights` is the terrain after the cuts under the
bridge decks, the same one written to the .ter.
"""
from __future__ import annotations

import importlib
import json
import logging
import math
import shutil
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from shapely.geometry import LineString, Point, box

from . import beamng_format as bf
from . import meshcache
from .bridges import clear_under_decks
from .geo import Extent
from .imagery import LAYER_NAMES
from .roads import load_roads
from .terrain import encode_u16, sample_height
from .water import load_water

log = logging.getLogger(__name__)

DATA_DIR = Path("data/out")
DIST_DIR = Path("dist")
# Microsoft's footprints (ODbL) supply most of the buildings
CREDITS = ("Map data © OpenStreetMap contributors (ODbL). Building footprints © Microsoft (ODbL). "
           "Elevation and surface: USGS 3DEP. Imagery: USDA NAIP. Land cover: MRLC NLCD. "
           "Textures: ambientCG and Poly Haven (CC0).")
DESCRIPTION = ("Baton Rouge, Louisiana at 1:1 scale, built from public map data: real roads on "
               "real lidar terrain, the Mississippi River and its bridges, downtown, the State "
               "Capitol and LSU with its lakes. Its landmarks are modelled, from Tiger Stadium and the "
               "Memorial Tower to both State Capitols and the Horace Wilkinson Bridge's truss, and its "
               "swamps and bayous grow bald cypress and water tupelo hung with Spanish moss.")
WATER_STRIP = 512.0   # m; long water bodies are split into blocks this long so they follow bends
WATER_MARGIN = 8.0    # m added around each block to cover the shoreline cells
DECK_SNAP = 25.0      # m; a spawn this close to a bridge deck's centerline is placed on it
                      # (25 m reaches either carriageway from a point between the two)
FEATURE_MODULES = ("buildings", "styled", "trees", "markings", "landmarks", "bayous", "dressing")


def _write_json(path: Path, data, indent=1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=indent, ensure_ascii=False), encoding="utf-8")


def _write_lines(path: Path, objects: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in objects), encoding="utf-8")


def water_rects(polygon, strip: float = WATER_STRIP, margin: float = WATER_MARGIN):
    """(cx, cy, sx, sy) boxes covering a water polygon, one per strip along its long axis."""
    minx, miny, maxx, maxy = polygon.bounds
    if max(maxx - minx, maxy - miny) <= strip:
        parts = [polygon]
    else:
        along_y = (maxy - miny) >= (maxx - minx)
        lo, hi = (miny, maxy) if along_y else (minx, maxx)
        parts = []
        for k in range(math.ceil((hi - lo) / strip)):
            a, b = lo + k * strip, min(hi, lo + (k + 1) * strip)
            band = box(minx, a, maxx, b) if along_y else box(a, miny, b, maxy)
            piece = polygon.intersection(band)
            if not piece.is_empty and piece.area > 0:
                parts.append(piece)
    rects = []
    for p in parts:
        x0, y0, x1, y1 = p.bounds
        rects.append(((x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) + 2 * margin, (y1 - y0) + 2 * margin))
    return rects


def _on_deck(decks, x: float, y: float):
    """(x, y, z, heading) on the centerline of the deck nearest (x, y), if within DECK_SNAP."""
    best = None
    for r in decks:
        line = LineString(r.nodes[:, :2])
        d = line.distance(Point(x, y))
        if d < DECK_SNAP and (best is None or d < best[0]):
            best = (d, r, line)
    if best is None:
        return None
    _, r, line = best
    s = line.project(Point(x, y))
    px, py = line.interpolate(s).coords[0]
    along = np.r_[0.0, np.cumsum(np.hypot(*np.diff(r.nodes[:, :2], axis=0).T))]
    i = int(np.clip(np.searchsorted(along, s), 1, len(along) - 1))
    dx, dy = r.nodes[i, :2] - r.nodes[i - 1, :2]
    return px, py, float(np.interp(s, along, r.nodes[:, 2])) + 1.0, math.atan2(dy, dx)


def place_spawns(cfg, extent: Extent, heights: np.ndarray, roads) -> list[dict]:
    """Spawn points inside the box: on the ground, or on the centerline of a bridge deck passing
    within DECK_SNAP, facing the way its traffic runs (heading in radians from +x, else None)."""
    decks = [r for r in roads if r.is_bridge and len(r.nodes) > 1]
    out = []
    for s in cfg.spawns:
        x, y = extent.to_level(s.lon, s.lat)
        if not extent.contains(x, y):
            continue
        z, heading = float(sample_height(heights, extent, x, y)) + 0.5, None
        deck = _on_deck(decks, x, y)
        if deck is not None:
            x, y, z, heading = deck
        out.append({"name": s.name, "title": s.title, "x": x, "y": y, "z": z, "heading": heading})
    if not out:  # no configured spawn in the box: the road node nearest the center
        x = y = 0.0
        ground = [r.nodes[:, :2] for r in roads if not r.is_bridge and len(r.nodes)]
        if ground:
            pts = np.vstack(ground)
            x, y = (float(v) for v in pts[int(np.argmin(np.hypot(pts[:, 0], pts[:, 1])))])
        out.append({"name": "spawn_center", "title": "Map center", "x": x, "y": y,
                    "z": float(sample_height(heights, extent, x, y)) + 0.5, "heading": None})
    return out


def _info(cfg, half: float, spawns: list[dict], preview: str) -> dict:
    return {
        "title": cfg.level_title, "description": f"{DESCRIPTION} {CREDITS}", "authors": cfg.authors,
        "previews": [preview], "size": [int(round(2 * half))] * 2,
        "country": "levels.common.country.usa", "region": "northAmerica",
        "biome": "Mississippi River floodplain, subtropical", "roads": "Interstates, city streets, campus roads",
        "suitablefor": "Freeroam", "features": "1:1 terrain from USGS lidar, OpenStreetMap roads, NAIP aerial imagery",
        "defaultSpawnPointName": spawns[0]["name"],
        "spawnPoints": [{"translationId": s["title"], "objectname": s["name"], "preview": preview}
                        for s in spawns],
        "supportsTraffic": bool(cfg.traffic_enabled), "supportsTimeOfDay": True,
        "roadRules": {"turnOnRed": True},
    }


def _preview(basecolor: Image.Image, extent: Extent, spawns: list[dict], dest: Path) -> None:
    w, h = bf.PREVIEW_SIZE
    img = basecolor
    if img.width < w or img.height < h:
        k = max(w / img.width, h / img.height)
        img = img.resize((math.ceil(img.width * k), math.ceil(img.height * k)), Image.LANCZOS)
    col, row = extent.to_pixel(spawns[0]["x"], spawns[0]["y"], img.width)
    left = int(min(max(col - w / 2, 0), img.width - w))
    top = int(min(max(row - h / 2, 0), img.height - h))
    img.crop((left, top, left + w, top + h)).convert("RGB").save(dest, "JPEG", quality=90)


def feature_groups(cfg, data_dir: Path, root: Path, heights: np.ndarray, extent: Extent,
                   taken=()) -> dict[str, list[dict]]:
    """The SimGroups the feature modules add (see the module docstring); `taken` holds the
    group names already in use."""
    out: dict[str, list[dict]] = {}
    for name in FEATURE_MODULES:
        if not (data_dir / f"{name}.json").exists():
            continue
        module = importlib.import_module(f"brmap.{name}")
        content = module.level_content(cfg, data_dir, root, heights, extent)
        for group, objs in content.items():
            if group in taken or group in out:
                raise ValueError(f"{name}: group {group!r} is already in the level")
            for o in objs:
                if "class" not in o or "persistentId" not in o:
                    raise ValueError(f"{name}: an object in group {group!r} lacks class or persistentId")
            out[group] = list(objs)
        log.info("%s: %s", name, ", ".join(f"{len(v)} in {g}" for g, v in content.items()) or "nothing")
    return out


def write_level(cfg, out_dir=DIST_DIR / "level", data_dir=DATA_DIR) -> Path:
    """Write levels/<name>/ and the plate design's bf.PLATES_DIR/<name>/ under out_dir and return the
    level folder."""
    extent = Extent.from_config(cfg)
    out_dir, data_dir = Path(out_dir), Path(data_dir)
    name = cfg.level_name
    heights = np.load(data_dir / "heights.npy")
    layers = np.load(data_dir / "layers.npy")
    n = heights.shape[0]
    if layers.shape != heights.shape:
        raise ValueError(f"layers {layers.shape} and heights {heights.shape} differ")
    root = out_dir / "levels" / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    lvl = f"/levels/{name}"

    # terrain, cut where it would poke through a bridge deck
    roads = load_roads(data_dir / "roads.json")
    heights, cuts = clear_under_decks(heights, extent, roads)
    if cuts:
        worst = max(cuts, key=cuts.get)
        log.info("terrain cut under %d decks, deepest %.1f m (%s)", len(cuts), cuts[worst], worst)
    materials = [bf.terrain_material_name(layer) for layer in LAYER_NAMES]
    bf.write_ter(root / bf.TERRAIN_FILE, encode_u16(heights, cfg.z_min, cfg.z_max), layers, materials)
    _write_json(root / bf.TERRAIN_META, bf.terrain_meta(name, n, materials), indent=2)
    with Image.open(data_dir / "basecolor.png") as im:
        basecolor = im.convert("RGB")
    (root / bf.BASECOLOR_FILE).parent.mkdir(parents=True, exist_ok=True)
    tex = basecolor if bf.BASECOLOR_ROW0_IS_NORTH else basecolor.transpose(Image.FLIP_TOP_BOTTOM)
    tex.save(root / bf.BASECOLOR_FILE)
    if tex.width != tex.height:
        raise ValueError(f"base color is {tex.width} x {tex.height}; terrain textures must be square")
    for path, mode, color in bf.BASE_MAPS.values():
        Image.new(mode, tex.size, color).save(root / path)
    _write_json(root / bf.TERRAIN_MATERIALS_FILE,
                bf.terrain_materials(name, f"{lvl}/{bf.BASECOLOR_FILE}", n, LAYER_NAMES, tex.width))
    _write_json(root / bf.ROAD_MATERIALS_FILE, bf.road_materials())

    # objects
    water = load_water(data_dir / "water.json") if (data_dir / "water.json").exists() else []
    groups: dict[str, list[dict]] = {key: [] for key in bf.GROUPS}
    groups["terrain"].append(bf.terrain_block(name, extent.half_size, n, cfg.z_min, cfg.z_max))
    for i, r in enumerate(roads):
        if r.is_bridge:
            groups["bridges"].append(bf.mesh_road(r, i))
            groups["connectors"].append(bf.bridge_connector(r, i))
        else:
            groups["roads"].append(bf.decal_road(r, i))
    for i, body in enumerate(water):
        for j, (cx, cy, sx, sy) in enumerate(water_rects(body.polygon)):
            groups["water"].append(bf.water_block(cx, cy, sx, sy, body.level_z, body.depth, f"{i}/{j}"))
    spawns = place_spawns(cfg, extent, heights, roads)
    groups["spawns"] = [bf.spawn_sphere(s["name"], s["x"], s["y"], s["z"], s["heading"]) for s in spawns]
    groups["env"] = bf.env_objects(name)
    by_name = {g: groups[key] for key, g in bf.GROUPS.items()}
    by_name.update(feature_groups(cfg, data_dir, root, heights, extent, taken=set(by_name)))

    mission = root / "main"
    _write_lines(mission / "items.level.json", [{"name": "MissionGroup", "class": "SimGroup",
                 "persistentId": bf.persistent_id("group", name, "MissionGroup"), "enabled": "1"}])
    _write_lines(mission / "MissionGroup" / "items.level.json", [
        {"name": g, "class": "SimGroup", "persistentId": bf.persistent_id("group", name, g),
         "__parent": "MissionGroup"} for g in by_name])
    for g, members in by_name.items():
        objs = [{**{k: o[k] for k in ("name", "class") if k in o}, "persistentId": o["persistentId"],
                 "__parent": g, **{k: v for k, v in o.items()
                                   if k not in ("name", "class", "persistentId", "__parent")}}
                for o in members]
        _write_lines(mission / "MissionGroup" / g / "items.level.json", objs)

    # the game's compiled meshes and imposters, where still valid; after the feature modules wrote every
    # *.materials.json, asset.json and texture, which the imposters' fingerprints read (it logs their counts)
    used, stale = meshcache.apply(root, data_dir / "cdae")
    if used or stale:
        log.info("compiled meshes: %d shipped, %d left for the game to compile (tools/ingame/collect_meshes.py)",
                 used, len(stale))
    preview = f"{name}_preview.jpg"
    _write_json(root / "info.json", _info(cfg, extent.half_size, spawns, preview), indent=2)
    _preview(basecolor, extent, spawns, root / preview)
    plates = out_dir / bf.PLATES_DIR / name
    if plates.exists():
        shutil.rmtree(plates)
    _write_json(plates / bf.PLATE_FILE, bf.plate_design(name), indent=2)
    log.info("level %s: %d decal roads, %d bridge decks (%d AI connectors), %d water blocks, %d spawns",
             name, len(groups["roads"]), len(groups["bridges"]), len(groups["connectors"]),
             len(groups["water"]), len(spawns))
    return root


def zip_mod(stage_dir, zip_path, level_name: str) -> Path:
    """Zip stage_dir/levels/<level_name>/ and its plate design, stage_dir/<bf.PLATES_DIR>/<level_name>/,
    at the same paths from the zip root.

    Only those two folders go in, so a folder left in the stage by a build under another
    level name never ships with it. PNG and JPEG are stored (deflate gains nothing on
    them); everything else is deflated, DDS included, because block-compressed textures
    deflate well (flat areas and repeated blocks).
    """
    stage_dir, zip_path = Path(stage_dir), Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    part = zip_path.with_name(zip_path.name + ".part")
    with zipfile.ZipFile(part, "w") as z:
        for tree in (Path("levels") / level_name, Path(bf.PLATES_DIR) / level_name):
            for f in sorted((stage_dir / tree).rglob("*")):
                if f.is_file():
                    packed = f.suffix.lower() in (".png", ".jpg")
                    z.write(f, f.relative_to(stage_dir).as_posix(),
                            compress_type=zipfile.ZIP_STORED if packed else zipfile.ZIP_DEFLATED)
    part.replace(zip_path)
    return zip_path


def run(cfg, data_dir=None, dist_dir=None) -> None:
    dist_dir = Path(dist_dir or cfg.dist_dir)
    stage = dist_dir / "level"
    root = write_level(cfg, stage, Path(data_dir or cfg.out_dir))
    z = zip_mod(stage, dist_dir / f"{cfg.level_name}.zip", cfg.level_name)
    log.info("level folder %s; mod zip %s (%.1f MB)", root, z, z.stat().st_size / 1e6)
