"""Everything specific to BeamNG.drive's level files, in one place.

Checked against the stock levels shipped with BeamNG.drive 0.39.4 (template,
gridmap_v2, west_coast_usa, east_coast_usa, italy, johnson_valley, Utah):

* .ter (version 9): u8 version, u32 size, u16 heights[size*size],
  u8 layers[size*size], u32 material count, then a u8 length and the name of
  each material; nothing after. A file written here with the template's size
  and material list matches the template's header, table and length exactly.
* main/items.level.json holds the MissionGroup SimGroup. Every SimGroup has a
  folder of the same name whose items.level.json lists its children, one JSON
  object per line, each naming its group in "__parent".
* TerrainBlock: position (min corner), squareSize, maxHeight, terrainFile,
  materialTextureSet. TerrainMaterial internalName = the .ter material name.
* DecalRoad nodes are [x, y, z, width]; MeshRoad and River nodes are
  [x, y, z, width, depth, nx, ny, nz]; MeshRoad uses top/bottom/sideMaterial.
* Road and terrain textures are shared assets under /assets/materials/...;
  materials are defined per level, so this level defines its own.
* The game's lua/ge/map.lua builds the AI road network from the DecalRoads with
  drivability > 0 only; MeshRoads are left out. With autoLanes false, lanesLeft
  and lanesRight set the lanes (lanesRight runs with the node order) and
  speedLimit is a string in m/s. info.json roadRules is merged over
  {rightHandDrive: false, turnOnRed: false}.

Checked in game (0.39.4, tools/ingame): the .ter rows run south to north
(surface heights matched the grid to 1 mm, a mirrored grid was 2.4 m off); the
base color's top row is north (the Capitol, its gardens, the lakes and the street
grid line up with the decal roads); a terrain base texture's size is counted in
terrain cells, not meters; an unrotated SpawnSphere faces south. DecalRoads are
drawn in descending renderPriority order, so the lowest number ends up on top (a
stop bar at 30 vanished under a road at 16 and showed at 2). An invisible drivable
DecalRoad laid along a bridge deck with overObjects joins the AI network: a path
across the I-10 deck was found, and traffic spawned and drove.
"""
from __future__ import annotations

import json
import math
import struct
import uuid
from functools import lru_cache
from importlib import resources
from pathlib import Path

import numpy as np

TER_VERSION = 9
TER_ROW0_IS_NORTH = False        # first row in the .ter file is the south edge
BASECOLOR_ROW0_IS_NORTH = True   # top row of the base color texture is the north edge
DECK_DEPTH = 1.5                 # MeshRoad thickness below the deck surface, m
TERRAIN_FILE = "theTerrain.ter"
TERRAIN_META = "theTerrain.terrain.json"
BASECOLOR_FILE = "art/terrains/br_basecolor.png"
TERRAIN_MATERIALS_FILE = "art/terrains/main.materials.json"
ROAD_MATERIALS_FILE = "art/road/main.materials.json"
PREVIEW_SIZE = (1024, 576)

# group key -> SimGroup name (PlayerDropPoints and sky_and_sun follow the stock levels)
GROUPS = {"terrain": "terrain", "roads": "roads", "bridges": "bridges", "connectors": "ai_bridges",
          "water": "water", "spawns": "PlayerDropPoints", "env": "sky_and_sun"}

_NS = uuid.UUID("3b9a7c1e-5d2f-4e8a-9c6b-1f0d2e3a4b5c")


def persistent_id(*parts) -> str:
    """A stable UUID for an object, so re-runs don't churn every id."""
    return str(uuid.uuid5(_NS, "/".join(str(p) for p in parts)))


# ------------------------------------------------------------------ .ter

def write_ter(path, heights_u16: np.ndarray, layers: np.ndarray, material_names: list[str],
              row0_is_north: bool = TER_ROW0_IS_NORTH) -> Path:
    """Write a .ter file. Inputs are north-up grids (row 0 = north), as everywhere in brmap."""
    h = np.asarray(heights_u16, dtype="<u2")
    lay = np.asarray(layers, dtype=np.uint8)
    size = h.shape[0]
    if h.shape != (size, size) or lay.shape != (size, size):
        raise ValueError("heights and layers must be the same square size")
    if lay.max(initial=0) >= len(material_names):
        raise ValueError("layer index outside the material list")
    if not row0_is_north:
        h, lay = h[::-1], lay[::-1]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<BI", TER_VERSION, size))
        f.write(np.ascontiguousarray(h).tobytes())
        f.write(np.ascontiguousarray(lay).tobytes())
        f.write(struct.pack("<I", len(material_names)))
        for name in material_names:
            b = name.encode("ascii")
            if len(b) > 255:
                raise ValueError(f"material name too long: {name}")
            f.write(struct.pack("<B", len(b)) + b)
    return path


def read_ter(path, row0_is_north: bool = TER_ROW0_IS_NORTH) -> dict:
    """Read a .ter file back into north-up arrays."""
    data = Path(path).read_bytes()
    version, size = struct.unpack_from("<BI", data, 0)
    n = size * size
    h = np.frombuffer(data, "<u2", count=n, offset=5).reshape(size, size)
    lay = np.frombuffer(data, np.uint8, count=n, offset=5 + 2 * n).reshape(size, size)
    p = 5 + 3 * n
    (count,) = struct.unpack_from("<I", data, p)
    p += 4
    names = []
    for _ in range(count):
        ln = data[p]
        names.append(data[p + 1:p + 1 + ln].decode("ascii"))
        p += 1 + ln
    if not row0_is_north:
        h, lay = h[::-1], lay[::-1]
    return {"version": version, "size": size, "heights": h.copy(), "layers": lay.copy(),
            "materials": names}


def terrain_meta(level: str, size: int, materials: list[str]) -> dict:
    """The theTerrain.terrain.json that stock levels ship next to the .ter."""
    return {
        "binaryFormat": "version(char), size(unsigned int), heightMap(heightMapSize * "
                        "heightMapItemSize), layerMap(layerMapSize * layerMapItemSize), materialNames",
        "datafile": f"/levels/{level}/{TERRAIN_FILE}",
        "heightMapItemSize": 2, "heightMapSize": size * size,
        "layerMapItemSize": 1, "layerMapSize": size * size,
        "materials": list(materials), "size": size, "version": TER_VERSION,
    }


# ------------------------------------------------------------------ materials

@lru_cache(maxsize=1)
def templates() -> dict:
    """Stock 'template' level sky objects and terrain materials (src/brmap/data)."""
    text = resources.files("brmap").joinpath("data/beamng_templates.json").read_text(encoding="utf-8")
    return json.loads(text)


def terrain_material_name(layer: str) -> str:
    return f"br_{layer}"


# Flat stand-ins for the other base maps (values from the stock levels' base maps). The
# game drops every base texture whose pixel size differs from the texture set's
# baseTexSize, so these are written at the aerial's size and span the whole map like it.
# A base texture's *BaseTexSize is in terrain cells, not meters: the stock levels set it
# to the grid size (derby: 2048 on a 0.3 m grid), and on our 4 m grid 16384 stretched the
# aerial four times over until it was given as 4096 (seen in game, 0.39.4).
BASE_MAPS = {
    "normalBaseTex": ("art/terrains/br_base_nm.png", "RGB", (127, 127, 255)),
    "roughnessBaseTex": ("art/terrains/br_base_r.png", "L", 210),
    "aoBaseTex": ("art/terrains/br_base_ao.png", "L", 255),
    "heightBaseTex": ("art/terrains/br_base_h.png", "L", 128),
}


def terrain_materials(level: str, basecolor_path: str, grid_cells: int, layers: list[str],
                      base_px: int) -> dict:
    """TerrainMaterialTextureSet + one TerrainMaterial per layer, all draped with the aerial
    (base_px wide) and the flat BASE_MAPS, each stretched over the terrain's grid_cells."""
    set_name = f"{level}TerrainMaterialTextureSet"
    out = {set_name: {"name": set_name, "class": "TerrainMaterialTextureSet",
                      "persistentId": persistent_id("texture_set", level),
                      "baseTexSize": [int(base_px), int(base_px)], "detailTexSize": [1024, 1024],
                      "macroTexSize": [1024, 1024]}}
    base = {}
    for key, (path, _, _) in BASE_MAPS.items():
        base[key] = f"/levels/{level}/{path}"
        base[key + "Size"] = float(grid_cells)
    for layer in layers:
        name = terrain_material_name(layer)
        pid = persistent_id("terrain_material", level, name)
        out[f"{name}-{pid}"] = {"internalName": name, "class": "TerrainMaterial", "persistentId": pid,
                                **templates()["terrain_materials"][layer], **base,
                                "baseColorBaseTex": basecolor_path,
                                "baseColorBaseTexSize": float(grid_cells)}
    return out


def _material(name: str, folder: str, stem: str, maps: str, translucent: bool, annotation: str,
              opacity: str | None = None) -> dict:
    base = f"/assets/materials/decalroad/{folder}/"
    stage = {"baseColorMap": f"{base}{stem}_b.color.png"}
    if "n" in maps:
        stage["normalMap"] = f"{base}{stem}_nm.normal.png"
    if "r" in maps:
        stage["roughnessMap"] = f"{base}{stem}_r.data.png"
    if "a" in maps:
        stage["ambientOcclusionMap"] = f"{base}{stem}_ao.data.png"
    if opacity:
        stage["opacityMap"] = f"{base}{opacity}"
    m = {"name": name, "mapTo": name, "class": "Material", "persistentId": persistent_id("material", name),
         "Stages": [stage, {}, {}, {}], "annotation": annotation, "castShadows": not translucent,
         "materialTag0": "RoadAndPath", "materialTag1": "beamng", "version": 1.5}
    if translucent:
        m.update({"translucent": True, "translucentZWrite": True, "alphaRef": 150})
    return m


# texture sets from content/assets/materials/decalroad.zip (the same ones the stock
# levels' road materials point at through their .link files)
ROAD_MATERIAL_DEFS = {
    "br_road_asphalt": ("asphalt/m_asphalt_02", "t_road_asphalt1", "na", True, "ASPHALT",
                        "t_road_asphalt2_o.data.png"),
    # older, ragged-edged asphalt for minor streets (the AsphaltRoad_variation sets are
    # patch overlays: their opacity maps cover ~3 % of the road, so roads drawn with them vanish)
    "br_road_asphalt_light": ("asphalt/asphalt_decayed", "t_asphalt_decayed", "nra", True,
                              "ASPHALT", "t_asphalt_decayed_o.data.png"),
    "br_road_concrete": ("concrete/decalroad_concrete", "t_highway_concrete", "nra", True, "ASPHALT", None),
    "br_bridge_deck": ("asphalt/asphalt_mesh_road", "t_asphalt_meshroad", "nra", False, "ASPHALT", None),
    "br_bridge_concrete": ("concrete/street_concrete1", "t_street_concrete", "nra", False, "ASPHALT", None),
}


def road_materials() -> dict:
    out = {}
    for name, (folder, stem, maps, translucent, annotation, opacity) in ROAD_MATERIAL_DEFS.items():
        m = _material(name, folder, stem, maps, translucent, annotation, opacity)
        if name == "br_road_asphalt":  # this set's roughness map has its own name
            m["Stages"][0]["roughnessMap"] = "/assets/materials/decalroad/asphalt/m_asphalt_02/t_asphalt1_r.data.png"
        out[name] = m
    return out


ROAD_MATERIAL = {
    **dict.fromkeys(["motorway", "motorway_link", "trunk", "trunk_link"], "br_road_concrete"),
    **dict.fromkeys(["primary", "primary_link", "secondary", "secondary_link", "tertiary",
                     "tertiary_link"], "br_road_asphalt"),
    **dict.fromkeys(["unclassified", "residential", "living_street", "service"], "br_road_asphalt_light"),
}
BRIDGE_MATERIALS = ("br_bridge_deck", "br_bridge_concrete", "br_bridge_concrete")  # top, bottom, side
# A lower renderPriority draws on top, so major roads get the low numbers and win where
# decals overlap at junctions. Links use their base class. Paint goes under all of these
# (lines 2, stop bars and crosswalks 1).
RENDER_PRIORITY = {"motorway": 6, "trunk": 7, "primary": 8, "secondary": 9, "tertiary": 10,
                   "unclassified": 11, "residential": 11, "living_street": 11, "service": 12}
DEFAULT_RENDER_PRIORITY = 12
DRIVABILITY = {"motorway": 1.0, "trunk": 1.0, "primary": 1.0, "secondary": 1.0, "tertiary": 0.9,
               "unclassified": 0.7, "residential": 0.7, "living_street": 0.5, "service": 0.4}
# speed limit for roads without an OSM maxspeed, by base class (anything else 25 mph)
DEFAULT_SPEED_MPH = {"motorway": 65, "trunk": 55, "primary": 45, "secondary": 40, "tertiary": 35,
                     "unclassified": 25, "residential": 25, "living_street": 25, "service": 25}
OTHER_SPEED_MPH = 25
MPH = 0.44704  # m/s
# Bridge decks are MeshRoads, which the AI network ignores; an invisible drivable decal along
# each deck joins it up. road_invisible is a global material (content/art_shapes.zip,
# art/shapes/common/decalroads/main.materials.json), so the level does not define it.
CONNECTOR_MATERIAL = "road_invisible"
CONNECTOR_PRIORITY = 30  # under everything; it is invisible anyway


def _base(kind: str) -> str:
    return kind[:-5] if kind.endswith("_link") else kind


def _r(v, nd=3) -> float:
    return round(float(v), nd)


def lane_fields(road) -> dict:
    """AI lanes and speed limit for a drivable decal. Without them the game counts
    width / 3.61 lanes and guesses a speed from the width."""
    lanes = max(1, int(road.lanes))
    left, right = (0, lanes) if road.oneway else (max(1, lanes // 2),) * 2
    speed = road.maxspeed
    if speed is None or speed <= 0:
        speed = DEFAULT_SPEED_MPH.get(_base(road.kind), OTHER_SPEED_MPH) * MPH
    return {"autoLanes": False, "lanesLeft": left, "lanesRight": right, "speedLimit": f"{speed:.2f}"}


def _decal_nodes(road) -> list[list[float]]:
    return [[_r(x), _r(y), _r(z), _r(road.width, 2)] for x, y, z in road.nodes]


def _ai_fields(road) -> dict:
    """drivability plus the AI lanes and speed. A covered road (Road.covered: a passage under
    a building, which the buildings stage keeps) is a dead end for traffic, so it gets
    drivability 0, which map.lua leaves out of the AI network, and no lanes or speed."""
    if road.covered:
        return {"drivability": 0}
    return {"drivability": DRIVABILITY.get(_base(road.kind), 0.5), **lane_fields(road)}


def decal_road(road, idx: int) -> dict:
    nodes = _decal_nodes(road)
    ai = _ai_fields(road)
    return {"class": "DecalRoad", "persistentId": persistent_id("road", idx, *road.osm_ids),
            "position": nodes[0][:3], "breakAngle": 3, "distanceFade": [600, 300],
            "drivability": ai.pop("drivability"), "improvedSpline": True,
            "material": ROAD_MATERIAL.get(road.kind, "br_road_asphalt_light"), "nodes": nodes,
            "oneWay": bool(road.oneway),
            "renderPriority": RENDER_PRIORITY.get(_base(road.kind), DEFAULT_RENDER_PRIORITY),
            "startEndFade": [0, 0], "textureLength": 16, **ai}


def bridge_connector(road, idx: int) -> dict:
    """The invisible drivable DecalRoad laid along a bridge deck (same nodes, width and
    direction), projected onto the deck by overObjects, so AI traffic can cross it.

    useSubdivisions false: lua/ge/map.lua (0.39.4, about lines 642-728) then builds the AI
    path from the decal's node positions and widths (getNodePosition, getNodeWidth) instead
    of the engine's projected edge positions, which miss on the 60 m high I-10 decks
    ("centerline to edge distance exceeding 20m"). WCUSA's 78 invisible overObjects decals
    are set the same way."""
    nodes = _decal_nodes(road)
    ai = _ai_fields(road)
    return {"class": "DecalRoad", "persistentId": persistent_id("conn", idx, *road.osm_ids),
            "position": nodes[0][:3], "breakAngle": 3,
            "drivability": ai.pop("drivability"), "improvedSpline": True,
            "material": CONNECTOR_MATERIAL, "nodes": nodes, "oneWay": bool(road.oneway),
            "overObjects": True, "renderPriority": CONNECTOR_PRIORITY, "startEndFade": [0, 0],
            "textureLength": 16, "useSubdivisions": False, **ai}


def mesh_road(road, idx: int) -> dict:
    nodes = [[_r(x), _r(y), _r(z), _r(road.width, 2), DECK_DEPTH, 0, 0, 1] for x, y, z in road.nodes]
    top, bottom, side = BRIDGE_MATERIALS
    return {"class": "MeshRoad", "persistentId": persistent_id("bridge", idx, *road.osm_ids),
            "position": nodes[0][:3], "breakAngle": 3, "bottomMaterial": bottom, "nodes": nodes,
            "sideMaterial": side, "textureLength": 5, "topMaterial": top, "widthSubdivisions": 0}


def terrain_block(level: str, half: float, n: int, z_min: float, z_max: float) -> dict:
    px = 2.0 * half / n
    corner = -half + px / 2.0  # vertex (0, 0) sits on the center of the south-west grid cell
    return {"name": "theTerrain", "class": "TerrainBlock", "persistentId": persistent_id("terrain", level),
            "position": [corner, corner, float(z_min)], "baseTexSize": 4096,
            "materialTextureSet": f"{level}TerrainMaterialTextureSet",
            "maxHeight": float(z_max - z_min), "squareSize": float(px),
            "terrainFile": f"/levels/{level}/{TERRAIN_FILE}"}


WATER_STYLE = {
    "Foam": [{}, {}],
    "Ripples (texture animation)": [
        {"rippleDir": [0, 1], "rippleSpeed": 0.025, "rippleTexScale": [2, 2]},
        {"rippleDir": [-1, -1], "rippleSpeed": 0.01, "rippleTexScale": [15, 15]},
        {"rippleDir": [0.5, 0.86], "rippleMagnitude": 1, "rippleSpeed": 0.02, "rippleTexScale": [30, 30]}],
    "Waves (vertex undulation)": [
        {"waveDir": [0, 1], "waveMagnitude": 0.2, "waveSpeed": 1},
        {"waveDir": [0.707, 0.707], "waveMagnitude": 0.2, "waveSpeed": 1},
        {"waveDir": [0.5, 0.86], "waveMagnitude": 0.2, "waveSpeed": 1}],
    "baseColor": [92, 84, 60, 255], "cubemap": "cubemap_ocean_reflection", "depthGradientMax": 10,
    "depthGradientTex": "core/art/water/depthcolor_ramp.png", "foamTex": "core/art/water/foam.dds",
    "fullReflect": False, "overallRippleMagnitude": 0.2, "overallWaveMagnitude": 0.05,
    "rippleTex": "core/art/water/ripple.dds", "specularPower": 800, "waterFogDensity": 0.6,
    "waterFogDensityOffset": 0.2, "wetDarkening": 0.25,
}


def water_block(cx: float, cy: float, sx: float, sy: float, level_z: float, depth: float,
                key: str) -> dict:
    """A WaterBlock: centered on (cx, cy), water surface at position z, box reaching down."""
    return {"class": "WaterBlock", "persistentId": persistent_id("water", key),
            "position": [_r(cx, 2), _r(cy, 2), _r(level_z, 3)],
            "scale": [_r(sx, 2), _r(sy, 2), _r(depth + 5.0, 2)], **WATER_STYLE}


def spawn_sphere(name: str, x: float, y: float, z: float, heading: float | None = None) -> dict:
    """heading: direction the vehicle should face, radians from +x (east) toward +y (north).
    Measured in game (0.39.4): an unrotated sphere spawns the vehicle facing south (-y), and
    rotationMatrix [c, -s, 0, s, c, 0, 0, 0, 1] for angle a turns it to face 270 degrees - a,
    so a = 270 degrees - heading."""
    out = {"name": name, "class": "SpawnSphere", "persistentId": persistent_id("spawn", name),
           "position": [_r(x, 2), _r(y, 2), _r(z, 2)], "dataBlock": "SpawnSphereMarker",
           "enabled": "1", "homingCount": "0", "indoorWeight": "1", "isAIControlled": "0",
           "lockCount": "0", "outdoorWeight": "1", "radius": 1, "sphereWeight": "1"}
    if heading is not None:
        a = 1.5 * math.pi - heading
        c, s = math.cos(a), math.sin(a)
        out["rotationMatrix"] = [_r(v, 6) + 0.0 for v in (c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0)]
    return out


def env_objects(level: str) -> list[dict]:
    out = []
    for o in templates()["env"]:
        o = dict(o)
        name, cls = o.pop("name"), o.pop("class")
        out.append({"name": name, "class": cls, "persistentId": persistent_id("env", level, name), **o})
    return out
