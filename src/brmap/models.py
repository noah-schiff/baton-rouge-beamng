"""The models stage: runs the Blender scripts of the landmarks and the custom trees headlessly
and keeps what they export, each shape's parts as npz files (the npz contract, read by
collada.read_npz), in out/models/<name>/.

* Blender is C:/Desktop/blender.exe by default (config blender.exe). A script runs as
  `blender --background --factory-startup --python-exit-code 1 --python <script> -- --params
  <json file> --out <dir>`, so an exception in it is exit code 1. Scripts get their arguments
  and write their parts with blender/lib/brexport.py.
* build() runs one script into one folder and skips the run while <folder>/key.txt holds the
  key of the script, the files in blender/lib/, the params and the Blender version. A failed
  run leaves no key, so the next one starts over.
* run() builds, one folder each:
  - every entry of out/landmarks.json whose "model" ends in .py (a path relative to the repo
    root, or absolute; "styled" entries have no script), with the entry's "params";
  - every entry of src/brmap/data/custom_trees.yaml, with the entry's "script" and "params"
    (the folder is the entry's key, which the script may take as its shape's name):
    - the Spanish moss overlays, the entries whose params hold MOSS_PARAM (hang_points_from:
      <live oak>), while cfg.moss_share is over 0 (the trees stage puts no moss on an oak
      otherwise). moss_params turns that param into hang_points, the points
      src/brmap/data/moss_points.json (tools/extract_moss_points.py) holds for that oak;
    - the others, the swamp trees, while cfg.swamp_trees is on (the trees stage places none
      when it is off).
  A missing landmarks.json or custom_trees.yaml is skipped with a log line, and Blender is
  only needed when something is built.
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import time
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]           # the repo: blender/ sits beside src/
LIB_DIR = ROOT / "blender" / "lib"                   # what every script shares, part of its key
CUSTOM_TREES = Path(__file__).resolve().parent / "data" / "custom_trees.yaml"
MOSS_POINTS = Path(__file__).resolve().parent / "data" / "moss_points.json"
MOSS_PARAM = "hang_points_from"                      # a moss overlay's param: the live oak it hangs on
LANDMARKS_FILE = "landmarks.json"
KEY_FILE = "key.txt"
PARAMS_FILE = "params.json"                          # what the script was given, kept for a look
ERROR_LINES = 20                                     # lines of stderr in a failure message

_versions: dict[tuple[str, ...], str] = {}           # blender_version's cache, per command


def blender_prefix(cfg) -> list[str]:
    """The Blender command up to the script: headless, with defaults, failing on a script error."""
    return [cfg.blender_exe, "--background", "--factory-startup", "--python-exit-code", "1"]


def blender_version(prefix: list[str]) -> str:
    """The first line of `--version`, e.g. "Blender 5.2.0 LTS". Asked once per process."""
    key = tuple(prefix)
    if key not in _versions:
        try:
            proc = subprocess.run([*prefix, "--version"], capture_output=True, text=True,
                                  encoding="utf-8", errors="replace")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Blender not found at {prefix[0]} (blender.exe in config.yaml)") from e
        lines = proc.stdout.splitlines()
        if proc.returncode or not lines:
            raise RuntimeError(f"{prefix[0]} --version failed (exit code {proc.returncode}): "
                               f"{proc.stderr.strip()[-300:]}")
        _versions[key] = lines[0].strip()
    return _versions[key]


def _lib_files() -> list[Path]:
    """The files under blender/lib/ in path order, without the folders Python compiles into."""
    files = [p for p in LIB_DIR.rglob("*") if p.is_file() and "__pycache__" not in p.relative_to(LIB_DIR).parts]
    return sorted(files, key=lambda p: p.relative_to(LIB_DIR).as_posix())


def model_key(script: Path, params: dict, version: str) -> str:
    """SHA-1 over the script's bytes, the bytes of every file under blender/lib/ (in path order,
    compiled files left out), the params as json with sorted keys, and the Blender version."""
    digest = hashlib.sha1()
    digest.update(Path(script).read_bytes())
    for path in _lib_files():
        digest.update(path.read_bytes())
    digest.update(json.dumps(params, sort_keys=True).encode("utf-8"))
    digest.update(version.encode("utf-8"))
    return digest.hexdigest()


def build(prefix: list[str], script: Path, params: dict, out_dir: Path, version: str) -> list[Path]:
    """Run `script` with `params` and keep its npz files in out_dir, unless out_dir/key.txt holds
    the key of this very run. Returns the npz paths, sorted. Raises RuntimeError when the run
    fails (with the script's name and the end of stderr) or writes no npz file."""
    script, out_dir = Path(script), Path(out_dir)
    key = model_key(script, params, version)
    key_file = out_dir / KEY_FILE
    old = sorted(out_dir.glob("*.npz"))
    if old and key_file.is_file() and key_file.read_text(encoding="utf-8").strip() == key:
        log.info("models: %s is up to date", out_dir.name)
        return old

    out_dir.mkdir(parents=True, exist_ok=True)
    key_file.unlink(missing_ok=True)       # no key until the run has worked
    for path in old:                       # parts the script no longer writes must not linger
        path.unlink()
    params_file = out_dir / PARAMS_FILE
    params_file.write_text(json.dumps(params, indent=1, sort_keys=True), encoding="utf-8")
    start = time.perf_counter()
    proc = subprocess.run([*prefix, "--python", str(script), "--", "--params", str(params_file),
                           "--out", str(out_dir)], capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode:
        tail = "\n".join((proc.stderr.strip() or proc.stdout.strip()).splitlines()[-ERROR_LINES:])
        raise RuntimeError(f"{script.name} failed (exit code {proc.returncode}); the end of stderr:\n{tail}")
    parts = sorted(out_dir.glob("*.npz"))
    if not parts:
        raise RuntimeError(f"{script.name} wrote no .npz files to {out_dir}")
    key_file.write_text(key, encoding="utf-8")
    log.info("models: %s built, %d part(s) in %.1f s", out_dir.name, len(parts), time.perf_counter() - start)
    return parts


def _script(model: str) -> Path:
    return ROOT / model        # an absolute path stays as it is


def moss_params(params: dict, points: dict | None = None) -> dict:
    """A moss overlay's params with MOSS_PARAM (a live oak) turned into hang_points, that oak's
    points in `points` (default: MOSS_POINTS)."""
    if points is None:
        points = json.loads(MOSS_POINTS.read_text(encoding="utf-8"))
    oak = params[MOSS_PARAM]
    if oak not in points:
        raise ValueError(f"models: {MOSS_POINTS.name} has no hang points for {oak} "
                         "(run tools/extract_moss_points.py)")
    return {**{k: v for k, v in params.items() if k != MOSS_PARAM}, "hang_points": points[oak]}


def _jobs(out_dir: Path, swamp_trees: bool = True, moss: bool = True) -> list[tuple[str, Path, dict]]:
    """(name, script, params) of every shape to build: the landmarks, then the custom trees: the
    swamp trees with swamp_trees, the moss overlays with moss."""
    jobs = []
    landmarks = out_dir / LANDMARKS_FILE
    if landmarks.is_file():
        for entry in json.loads(landmarks.read_text(encoding="utf-8")):
            if str(entry["model"]).endswith(".py"):
                jobs.append((entry["name"], _script(entry["model"]), entry.get("params") or {}))
    else:
        log.info("models: %s not found, no landmark models to build", landmarks)
    if CUSTOM_TREES.is_file():
        entries = yaml.safe_load(CUSTOM_TREES.read_text(encoding="utf-8")) or {}
        overlays = {k for k, e in entries.items() if MOSS_PARAM in (e.get("params") or {})}
        if not swamp_trees:
            log.info("models: trees.swamp is off, no swamp trees to build")
        if not moss and overlays:
            log.info("models: trees.moss_share is 0, no moss overlays to build")
        points = json.loads(MOSS_POINTS.read_text(encoding="utf-8")) if moss and overlays else {}
        for key, entry in entries.items():
            params = entry.get("params") or {}
            if key in overlays and moss:
                jobs.append((key, _script(entry["script"]), moss_params(params, points)))
            elif key not in overlays and swamp_trees:
                jobs.append((key, _script(entry["script"]), params))
    else:
        log.info("models: %s not found, no custom trees to build", CUSTOM_TREES)
    names = [name for name, _, _ in jobs]
    twice = sorted({name for name in names if names.count(name) > 1})
    if twice:
        raise ValueError(f"models: two shapes are called {', '.join(twice)}; each needs its own folder")
    return jobs


def run(cfg) -> None:
    out_dir = Path(cfg.out_dir)
    jobs = _jobs(out_dir, swamp_trees=cfg.swamp_trees, moss=cfg.moss_share > 0)
    if not jobs:
        log.info("models: nothing to build")
        return
    prefix = blender_prefix(cfg)
    version = blender_version(prefix)
    log.info("models: %d shape(s), %s", len(jobs), version)
    for name, script, params in jobs:
        build(prefix, script, params, out_dir / "models" / name, version)
