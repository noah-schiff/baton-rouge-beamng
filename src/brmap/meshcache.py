"""Precompiled meshes (.cdae) and tree imposters shipped next to their .dae files.

BeamNG compiles every COLLADA (.dae) shape to its binary .cdae the first time a level
loads, into the user folder's temp/levels/<level>/. For the ~3,000 building tiles that took
116 s of a 165 s first load in 0.39.4, and a GeForce NOW session starts fresh every time.
Stock levels ship each .cdae beside its .dae, which the game then loads as is when the
.cdae is the newer file. The .dae stays in the zip, so a game update that changes the
.cdae format only costs a recompile.

A tree shape that asks for imposters (in its <shape>.dae.asset.json) also gets
<shape>.dae.imposter.dds and <shape>.dae.imposter_normals.dds, which the game renders at
runtime. They are the same kind of load-time work, so they travel with their mesh.

An imposter is a picture of the tree, so it also goes stale when the tree's look changes and
the .dae does not: the asset.json, a material the .dae binds, or a texture one of those uses.
The game only renders an imposter again when the shape file is newer than it, so a stale
one shipped in the zip would stay. fingerprint(level_root, key) is the SHA-1 of all of that
for one shape, and cache_dir/imposters.json ({shape key: fingerprint}, beside manifest.json,
whose format is unchanged) records what the cached imposters of each shape picture.

The game is the only compiler, so this works in two steps:

* collect(level_root, game_temp_level, cache_dir): after the level was loaded once in game,
  copy each .cdae the game compiled from this build's .dae (compiled after it was written)
  into cache_dir, with the .dae's SHA-1 in cache_dir/manifest.json, and the imposters the
  game made for that shape (made after the .dae was written) beside it, with their
  fingerprint. Imposters the game made for a .dae whose shipped .cdae it loaded are taken
  too. tools/ingame/collect_meshes.py runs it.
* apply(level_root, cache_dir): when the level is written, put each cached .cdae next to its
  .dae if that .dae is byte for byte the one it was made from, and the shape's imposters with
  it while their fingerprint is the recorded one; otherwise they are withheld and the game
  renders them at load (collect takes them next time). A shape without an asset.json has no
  fingerprint: its cached imposters, if any, travel with the .cdae as before. Written after
  the .dae, the copies are the newer files. Any other .dae is compiled by the game as before.
  It runs after every file of the level folder is written: the fingerprints read the level's
  *.materials.json files and textures.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
from pathlib import Path

log = logging.getLogger(__name__)

MANIFEST = "manifest.json"
IMPOSTER_FINGERPRINTS = "imposters.json"   # {shape key: fingerprint} of the cached imposters
# appended to the shape's file name, .dae included: art/shapes/trees/a.dae.imposter.dds
IMPOSTER_SUFFIXES = (".imposter.dds", ".imposter_normals.dds")
ASSET_SUFFIX = ".asset.json"               # <shape>.dae.asset.json asks the game for imposters
TEXTURE_SUFFIXES = (".png", ".dds", ".jpg", ".jpeg", ".tga")
_MATERIAL = re.compile(r"<material\b([^>]*)>")


def _sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def _daes(level_root: Path) -> list[Path]:
    return sorted(level_root.rglob("*.dae"))


def _read_json(path: Path) -> dict[str, str]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_manifest(cache_dir: Path) -> dict[str, str]:
    return _read_json(cache_dir / MANIFEST)


# ------------------------------------------------------------------ what an imposter pictures

def _bound_materials(dae: Path) -> list[str]:
    """The material names a .dae binds (each <material>'s name, else its id: what a level Material's
    mapTo names), sorted."""
    names = set()
    for attrs in _MATERIAL.findall(dae.read_text(encoding="utf-8", errors="replace")):
        found = re.search(r'\bname="([^"]*)"', attrs) or re.search(r'\bid="([^"]*)"', attrs)
        if found:
            names.add(found.group(1))
    return sorted(names)


def _level_materials(level_root: Path) -> dict[str, list[tuple[Path, dict]]]:
    """{mapTo: [(materials file's folder, entry), ...]} over the level's *.materials.json files, in path order."""
    out: dict[str, list[tuple[Path, dict]]] = {}
    for path in sorted(level_root.rglob("*.materials.json")):
        for name, entry in json.loads(path.read_text(encoding="utf-8")).items():
            if isinstance(entry, dict):
                out.setdefault(str(entry.get("mapTo", name)), []).append((path.parent, entry))
    return out


def _texture_paths(value) -> list[str]:
    """Every texture path a material entry names (a string ending in an image suffix), sorted."""
    if isinstance(value, str):
        return [value] if value.lower().endswith(TEXTURE_SUFFIXES) else []
    items = value.values() if isinstance(value, dict) else value if isinstance(value, list) else ()
    return sorted({path for item in items for path in _texture_paths(item)})


def _texture_file(ref: str, level_root: Path, folder: Path) -> Path | None:
    """The file in the level folder the game loads for a material's texture path, or None for a path
    elsewhere (a stock /assets/... texture). A bare file name is beside the materials file (folder). On
    0.40 a .png name loads the .dds shipped under it (NOTE_FOR_MODDERS.md; see brmap.materials)."""
    path = ref.replace("\\", "/").lstrip("/")
    prefix = f"levels/{level_root.name}/"
    if path.lower().startswith(prefix.lower()):
        file = level_root / path[len(prefix):]
    elif "/" not in path:
        file = folder / path
    else:
        return None
    return next((f for f in (file.with_suffix(".dds"), file) if f.is_file()), None)


def _fingerprint(dae: Path, asset: Path, level_root: Path, materials: dict) -> str:
    h = hashlib.sha1()

    def part(data: bytes) -> None:          # length-prefixed, so no two different inputs run together
        h.update(len(data).to_bytes(8, "little"))
        h.update(data)

    part(asset.read_bytes())
    for name in _bound_materials(dae):
        defined = materials.get(name, [])
        part(name.encode())
        part(json.dumps([entry for _, entry in defined], sort_keys=True).encode())
        for folder, entry in defined:
            for ref in _texture_paths(entry):
                file = _texture_file(ref, level_root, folder)
                part(ref.encode())
                part(file.read_bytes() if file else b"")
    return h.hexdigest()


def _fingerprints(level_root: Path):
    """fingerprint(level_root, key) for many shapes, reading the level's materials files once."""
    materials = None

    def of(key: str) -> str | None:
        nonlocal materials
        dae = level_root / key
        asset = dae.with_name(dae.name + ASSET_SUFFIX)
        if not asset.is_file():
            return None
        if materials is None:
            materials = _level_materials(level_root)
        return _fingerprint(dae, asset, level_root, materials)
    return of


def fingerprint(level_root, key: str) -> str | None:
    """SHA-1 of what the imposters of the shape `key` (its .dae path under the level folder) picture: its
    asset.json's bytes; then for each material the .dae binds (sorted), its name, its entries in the level's
    *.materials.json files (sorted keys), and for each texture path those name (sorted), the path and the
    bytes of the file the game loads for it from the level folder (none for a stock path). None for a shape
    without an asset.json: the game makes no imposters for it."""
    return _fingerprints(Path(level_root))(key)


def _cached_imposters(key: str, cache_dir: Path) -> list[str]:
    return [s for s in IMPOSTER_SUFFIXES if (cache_dir / (key + s)).is_file()]


def _ships(key: str, cache_dir: Path, now: str | None, recorded: dict[str, str]) -> bool:
    """Whether apply places the cached imposters of a shape whose .cdae it places: there are some, and they
    picture the shape as it is now (`now`, its fingerprint), or it has no asset.json (no fingerprint)."""
    return bool(_cached_imposters(key, cache_dir)) and (now is None or recorded.get(key) == now)


# ------------------------------------------------------------------ collect and apply

def _collect_imposters(key: str, dae: Path, imposter_root: Path, cache_dir: Path, dae_unchanged: bool) -> bool:
    """Cache the imposters the game made for the shape `key` (its .dae path under the level folder) since its
    .dae was written; whether it made any.

    Fresh imposters replace the cached ones (a cached file the game did not make again goes: it is of
    another render). When there are none, the cached ones stay only while the .dae is the one they were
    made from.
    """
    written = dae.stat().st_mtime
    fresh = {s: imposter_root / (key + s) for s in IMPOSTER_SUFFIXES}
    fresh = {s: f for s, f in fresh.items() if f.is_file() and f.stat().st_mtime >= written}
    for suffix in IMPOSTER_SUFFIXES:
        dest = cache_dir / (key + suffix)
        if suffix in fresh:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(fresh[suffix], dest)
        elif fresh or not dae_unchanged:
            dest.unlink(missing_ok=True)
    return bool(fresh)


def collect(level_root, game_temp_level, cache_dir, imposter_root=None) -> tuple[int, list[str]]:
    """(.cdae files collected, the .dae files with no fresh compiled copy) for a level folder.

    Each collected .cdae takes the imposters found at the same relative path under
    imposter_root (default: game_temp_level, where the game writes the .cdae) with it, and so
    does a .dae whose cached .cdae is still valid; the fingerprint of the level as it is now goes
    in imposters.json for each shape whose imposters were taken, and kept imposters keep theirs.
    """
    level_root, game_temp_level, cache_dir = Path(level_root), Path(game_temp_level), Path(cache_dir)
    imposter_root = game_temp_level if imposter_root is None else Path(imposter_root)
    old, old_prints = _read_manifest(cache_dir), _read_json(cache_dir / IMPOSTER_FINGERPRINTS)
    fingerprints = _fingerprints(level_root)
    manifest: dict[str, str] = {}
    prints: dict[str, str] = {}
    missing: list[str] = []
    for dae in _daes(level_root):
        rel = dae.relative_to(level_root)
        key, sha = rel.as_posix(), _sha1(dae)
        compiled = game_temp_level / rel.with_suffix(".cdae")
        dest = cache_dir / rel.with_suffix(".cdae")
        if compiled.exists() and compiled.stat().st_mtime >= dae.stat().st_mtime:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(compiled, dest)
        elif not (old.get(key) == sha and dest.exists()):  # else the game loaded the shipped copy: still valid
            missing.append(key)
            continue
        manifest[key] = sha
        if _collect_imposters(key, dae, imposter_root, cache_dir, dae_unchanged=old.get(key) == sha):
            now = fingerprints(key)
            if now is not None:
                prints[key] = now
        elif key in old_prints and _cached_imposters(key, cache_dir):
            prints[key] = old_prints[key]
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / MANIFEST).write_text(json.dumps(manifest, indent=0, sort_keys=True), encoding="utf-8")
    (cache_dir / IMPOSTER_FINGERPRINTS).write_text(json.dumps(prints, indent=0, sort_keys=True), encoding="utf-8")
    return len(manifest), missing


def count_imposters(cache_dir) -> int:
    """Imposter files in cache_dir that apply() can place: those of the meshes in its manifest."""
    cache_dir = Path(cache_dir)
    return sum((cache_dir / (key + suffix)).is_file()
               for key in _read_manifest(cache_dir) for suffix in IMPOSTER_SUFFIXES)


def imposter_status(level_root, cache_dir) -> tuple[list[str], list[str]]:
    """(the shapes whose cached imposters apply() would place in this level folder, the shapes with an
    asset.json whose imposters it would not: none cached, or made of something that has changed since)."""
    level_root, cache_dir = Path(level_root), Path(cache_dir)
    manifest, recorded = _read_manifest(cache_dir), _read_json(cache_dir / IMPOSTER_FINGERPRINTS)
    fingerprints = _fingerprints(level_root)
    ready, lacking = [], []
    for dae in _daes(level_root):
        key = dae.relative_to(level_root).as_posix()
        now = fingerprints(key)
        if manifest.get(key) == _sha1(dae) and _ships(key, cache_dir, now, recorded):
            ready.append(key)
        elif now is not None:
            lacking.append(key)
    return ready, lacking


def apply(level_root, cache_dir) -> tuple[int, list[str]]:
    """(.cdae files placed, the .dae files left for the game to compile) in a level folder; logs how many
    shapes' imposters were placed and which were withheld."""
    level_root, cache_dir = Path(level_root), Path(cache_dir)
    manifest, recorded = _read_manifest(cache_dir), _read_json(cache_dir / IMPOSTER_FINGERPRINTS)
    fingerprints = _fingerprints(level_root)
    used, stale, shipped, withheld = 0, [], 0, []
    for dae in _daes(level_root):
        rel = dae.relative_to(level_root).as_posix()
        cached = cache_dir / Path(rel).with_suffix(".cdae")
        if manifest.get(rel) == _sha1(dae) and cached.exists():
            shutil.copyfile(cached, dae.with_suffix(".cdae"))  # a fresh mtime: newer than the .dae
            used += 1
            if _ships(rel, cache_dir, fingerprints(rel), recorded):
                for suffix in _cached_imposters(rel, cache_dir):  # after the .cdae, so also newer than the .dae
                    shutil.copyfile(cache_dir / (rel + suffix), dae.with_name(dae.name + suffix))
                shipped += 1
            elif _cached_imposters(rel, cache_dir):
                withheld.append(rel)
        else:
            stale.append(rel)
    if shipped or withheld:
        log.info("imposters: %d pairs shipped, %d withheld%s", shipped, len(withheld),
                 f" ({', '.join(withheld)}: made of a tree that has changed since, or collected before "
                 f"{IMPOSTER_FINGERPRINTS}; the game renders them at load and tools/ingame/collect_meshes.py "
                 "takes them)" if withheld else "")
    return used, stale
