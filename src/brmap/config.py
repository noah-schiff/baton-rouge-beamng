"""Pipeline settings, loaded from config.yaml."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_SOURCES = {
    "tnm_api": "https://tnmaccess.nationalmap.gov/api/v1/products",
    "overpass": "https://overpass-api.de/api/interpreter",
    "overpass_fallback": "https://overpass.kumi.systems/api/interpreter",
    "osm_extract": "https://download.geofabrik.de/north-america/us/louisiana-latest.osm.pbf",
    "stac": "https://planetarycomputer.microsoft.com/api/stac/v1",
    "nlcd_wcs": "https://www.mrlc.gov/geoserver/mrlc_download/NLCD_2021_Land_Cover_L48/wcs",
    "ms_buildings_links": "https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv",
    "dsm_collection": "3dep-lidar-dsm",   # Planetary Computer STAC collection (lidar surface heights)
    "naip_collection": "naip",            # Planetary Computer STAC collection (4-band aerials)
}

TERRAIN_SIZES = (256, 512, 1024, 2048, 4096, 8192)


@dataclass(frozen=True)
class SpawnPoint:
    name: str    # SpawnSphere object name (unique, no spaces)
    title: str   # shown in the game's spawn point list
    lat: float
    lon: float


DEFAULT_SPAWNS = (
    SpawnPoint("spawn_lsu", "LSU, Tiger Stadium lot", 30.4115, -91.1855),
    SpawnPoint("spawn_capitol", "Louisiana State Capitol", 30.4560, -91.1868),
    SpawnPoint("spawn_i10_bridge", "I-10 Mississippi River bridge, east approach", 30.4395, -91.1935),
)


@dataclass
class Config:
    center_lat: float = 30.430
    center_lon: float = -91.160
    half_size_m: float = 8192.0
    terrain_size: int = 4096
    z_min: float = -5.0
    z_max: float = 40.0
    buffer_m: float = 500.0
    clearance_m: float = 5.5
    bridge_overrides: dict[str, float] = field(
        default_factory=lambda: {"Horace Wilkinson Bridge": 52.0})
    debug_half_size_m: float | None = None
    osm_source: str = "geofabrik"  # or "overpass"; the other one is the fallback
    sources: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_SOURCES))
    level_name: str = "baton_rouge"
    level_title: str = "Baton Rouge 1:1"
    authors: str = "brmap"
    spawns: list[SpawnPoint] = field(default_factory=lambda: list(DEFAULT_SPAWNS))
    raw_dir: str = "data/raw"    # downloads (shared by every config of the same region)
    out_dir: str = "data/out"    # stage outputs
    dist_dir: str = "dist"       # level folder and mod zip
    buildings_microsoft: bool = True  # add Microsoft ML footprints where OSM has none
    building_tile_m: float = 256.0    # side of a building mesh tile, m
    trees_max: int = 250000           # cap on tree instances
    markings_enabled: bool = True     # lane markings on the major roads
    traffic_enabled: bool = True      # info.json supportsTraffic (AI traffic in Freeroam)

    def validate(self) -> None:
        if self.half_size_m <= 0:
            raise ValueError(f"half_size_m must be positive, got {self.half_size_m}")
        if self.debug_half_size_m is not None and self.debug_half_size_m <= 0:
            raise ValueError("debug_half_size_m must be positive or null")
        if self.terrain_size not in TERRAIN_SIZES:
            raise ValueError(f"terrain size must be one of {TERRAIN_SIZES}, got {self.terrain_size}")
        if self.z_max <= self.z_min:
            raise ValueError(f"z_max ({self.z_max}) must be above z_min ({self.z_min})")
        if self.buffer_m < 0 or self.clearance_m < 0:
            raise ValueError("buffer_m and clearance_m must not be negative")
        if self.osm_source not in ("geofabrik", "overpass"):
            raise ValueError(f"osm source must be geofabrik or overpass, got {self.osm_source!r}")
        if self.building_tile_m <= 0:
            raise ValueError(f"buildings tile_m must be positive, got {self.building_tile_m}")
        if self.trees_max < 0:
            raise ValueError(f"trees max_count must not be negative, got {self.trees_max}")


def _flag(value) -> bool:
    """A yes/no setting; a quoted 'false' or 'no' in the YAML still means off."""
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "on", "1")
    return bool(value)


def load_config(path: str | Path = "config.yaml") -> Config:
    """Read config.yaml. Missing keys fall back to the defaults above."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    ext = raw.get("extent") or {}
    ter = raw.get("terrain") or {}
    br = raw.get("bridges") or {}
    bld = raw.get("buildings") or {}
    d = Config()
    debug = ext.get("debug_half_size_m", d.debug_half_size_m)
    overrides = br.get("overrides")
    cfg = Config(
        center_lat=float(ext.get("center_lat", d.center_lat)),
        center_lon=float(ext.get("center_lon", d.center_lon)),
        half_size_m=float(ext.get("half_size_m", d.half_size_m)),
        buffer_m=float(ext.get("buffer_m", d.buffer_m)),
        debug_half_size_m=None if debug is None else float(debug),
        terrain_size=int(ter.get("size", d.terrain_size)),
        z_min=float(ter.get("z_min", d.z_min)),
        z_max=float(ter.get("z_max", d.z_max)),
        clearance_m=float(br.get("clearance_m", d.clearance_m)),
        bridge_overrides=(d.bridge_overrides if overrides is None
                          else {str(k): float(v) for k, v in overrides.items()}),
        osm_source=str((raw.get("osm") or {}).get("source", d.osm_source)),
        sources={**d.sources, **(raw.get("sources") or {})},
        level_name=str((raw.get("level") or {}).get("name", d.level_name)),
        level_title=str((raw.get("level") or {}).get("title", d.level_title)),
        authors=str((raw.get("level") or {}).get("authors", d.authors)),
        spawns=([SpawnPoint(str(s["name"]), str(s.get("title", s["name"])), float(s["lat"]),
                            float(s["lon"])) for s in raw["spawns"]]
                if raw.get("spawns") else list(DEFAULT_SPAWNS)),
        raw_dir=str((raw.get("paths") or {}).get("raw", d.raw_dir)),
        out_dir=str((raw.get("paths") or {}).get("out", d.out_dir)),
        dist_dir=str((raw.get("paths") or {}).get("dist", d.dist_dir)),
        buildings_microsoft=_flag(bld.get("microsoft", d.buildings_microsoft)),
        building_tile_m=float(bld.get("tile_m", d.building_tile_m)),
        trees_max=int((raw.get("trees") or {}).get("max_count", d.trees_max)),
        markings_enabled=_flag((raw.get("markings") or {}).get("enabled", d.markings_enabled)),
        traffic_enabled=_flag((raw.get("traffic") or {}).get("enabled", d.traffic_enabled)),
    )
    cfg.validate()
    return cfg
