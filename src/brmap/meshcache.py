"""Precompiled building meshes (.cdae) shipped next to their .dae files.

BeamNG compiles every COLLADA (.dae) shape to its binary .cdae the first time a level
loads, into the user folder's temp/levels/<level>/. For the ~3,000 building tiles that took
116 s of a 165 s first load in 0.39.4, and a GeForce NOW session starts fresh every time.
Stock levels ship each .cdae beside its .dae, which the game then loads as is when the
.cdae is the newer file. The .dae stays in the zip, so a game update that changes the
.cdae format only costs a recompile.

The game is the only compiler, so this works in two steps:

* collect(level_root, game_temp_level, cache_dir): after the level was loaded once in game,
  copy each .cdae the game compiled from this build's .dae (compiled after it was written)
  into cache_dir, with the .dae's SHA-1 in cache_dir/manifest.json.
  tools/ingame/collect_meshes.py runs it.
* apply(level_root, cache_dir): when the level is written, put each cached .cdae next to
  its .dae if that .dae is byte for byte the one it was compiled from. Written after the
  .dae, the copy is the newer file. Any other .dae is compiled by the game as before.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path

log = logging.getLogger(__name__)

MANIFEST = "manifest.json"


def _sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def _daes(level_root: Path) -> list[Path]:
    return sorted(level_root.rglob("*.dae"))


def collect(level_root, game_temp_level, cache_dir) -> tuple[int, list[str]]:
    """(.cdae files collected, the .dae files with no fresh compiled copy) for a level folder."""
    level_root, game_temp_level, cache_dir = Path(level_root), Path(game_temp_level), Path(cache_dir)
    path = cache_dir / MANIFEST
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    manifest: dict[str, str] = {}
    missing: list[str] = []
    for dae in _daes(level_root):
        rel = dae.relative_to(level_root)
        key, sha = rel.as_posix(), _sha1(dae)
        compiled = game_temp_level / rel.with_suffix(".cdae")
        dest = cache_dir / rel.with_suffix(".cdae")
        if compiled.exists() and compiled.stat().st_mtime >= dae.stat().st_mtime:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(compiled, dest)
            manifest[key] = sha
        elif old.get(key) == sha and dest.exists():  # the game loaded the shipped copy: still valid
            manifest[key] = sha
        else:
            missing.append(key)
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / MANIFEST).write_text(json.dumps(manifest, indent=0, sort_keys=True), encoding="utf-8")
    return len(manifest), missing


def apply(level_root, cache_dir) -> tuple[int, list[str]]:
    """(.cdae files placed, the .dae files left for the game to compile) in a level folder."""
    level_root, cache_dir = Path(level_root), Path(cache_dir)
    path = cache_dir / MANIFEST
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    used, stale = 0, []
    for dae in _daes(level_root):
        rel = dae.relative_to(level_root).as_posix()
        cached = cache_dir / Path(rel).with_suffix(".cdae")
        if manifest.get(rel) == _sha1(dae) and cached.exists():
            shutil.copyfile(cached, dae.with_suffix(".cdae"))  # a fresh mtime: newer than the .dae
            used += 1
        else:
            stale.append(rel)
    return used, stale
