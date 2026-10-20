"""Material definitions (BeamNG's v1.5 Material) for the modelled shapes, from a registry.

A registry (src/brmap/data/landmark_materials.yaml for the landmarks, tree_materials.yaml for the
custom trees) maps each material name, the name the Blender model binds, to

    name: {maps: {b|nm|r|o|ao: "stock:<path>" | "tex:<tex_id>"}, factors: {...}, flags: {...}}

* material(name, stage0, **flags) -> the Material dict: name, mapTo = name, class "Material",
  version 1.5, Stages = [stage0, {}, {}, {}], persistentId bf.persistent_id("material", name)
  and the flags at the top level.
* resolve(registry, names, level_name, tex_index) -> (materials, files): the Material of each
  of `names` (in name order) and the texture files they need (the .dds files), as file names in
  out/textures/ to copy into the level's art/textures/br/.
  - A map key becomes its stage field: b -> baseColorMap, nm -> normalMap, r -> roughnessMap,
    o -> opacityMap, ao -> ambientOcclusionMap.
  - "stock:<path>" stays as it is: a texture the game ships (named .png; the game loads the
    .dds it ships under that name).
  - "tex:<id>" becomes /levels/<level_name>/art/textures/br/<stem>.png, <stem> being the name
    tex_index[id]["files"][map] gives without its .dds (out/textures/index.json, written by the
    textures stage: the shipped .dds, e.g. <id>_<suffix>.dds with suffix b.color, nm.normal,
    r.data, o.data; so clock_face_b.color.dds is referenced as clock_face_b.color.png). That .dds
    is the file to copy, and no .png ships: BeamNG.drive 0.40's NOTE_FOR_MODDERS.md says to ship
    the DDS where the PNG would be but keep referencing the .png, as the stock materials do. An
    index file that is not a .dds is refused.
  - factors go into the stage as they are (baseColorFactor, roughnessFactor, metallicFactor).
  - flags go to the top level (alphaTest, alphaRef, doubleSided, groundType, ...), except the
    per-stage switches in STAGE_FLAGS (vertColor, instanceDiffuse): every stock v1.5 material
    sets those in its stage, and the game ignores them at the top level.
* write_materials(root, rel_path, mats) writes them to the level's root/rel_path, leaving out the
  ones another *.materials.json under root already defines, so a material two feature modules
  bind (br_lm_roof_tile: the landmarks and the styled buildings) is defined once in the level.
* texture_index(data_dir) -> out/textures/index.json, or {} before the textures stage has run.
* copy_textures(files, data_dir, root): the files resolve() named, from out/textures/ into the
  level's art/textures/br/.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from . import beamng_format as bf

TEX_DIR = "art/textures/br"   # under the level folder: the repo's own textures
MAP_FIELDS = {"b": ("baseColorMap", "b.color"), "nm": ("normalMap", "nm.normal"),
              "r": ("roughnessMap", "r.data"), "o": ("opacityMap", "o.data"),
              "ao": ("ambientOcclusionMap", "ao.data")}
STAGE_FLAGS = {"vertColor", "instanceDiffuse"}
RESERVED = {"name", "mapTo", "class", "version", "Stages", "persistentId"}


def load_registry(path) -> dict[str, dict]:
    """A material registry YAML as {name: entry}."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def material(name: str, stage0: dict, /, **flags) -> dict:
    """The v1.5 Material dict for `name` with one stage; the flags go to the top level."""
    clash = sorted(RESERVED & set(flags))
    if clash:
        raise ValueError(f"material {name}: {', '.join(clash)} can't be set as a flag")
    return {"name": name, "mapTo": name, "class": "Material", "persistentId": bf.persistent_id("material", name),
            "Stages": [dict(stage0), {}, {}, {}], **flags, "version": 1.5}


def _map(name: str, key: str, ref, level_name: str, tex_index: dict) -> tuple[str, str, Path | None]:
    """(stage field, path, file to copy or None) for one map of a registry entry."""
    if key not in MAP_FIELDS:
        raise ValueError(f"material {name}: unknown map {key!r} (one of {', '.join(MAP_FIELDS)})")
    field = MAP_FIELDS[key][0]
    kind, _, value = str(ref).partition(":")
    if kind == "stock" and value:
        return field, value, None
    if kind == "tex" and value:
        try:
            file = Path(tex_index[value]["files"][key])
        except (KeyError, TypeError):
            raise ValueError(f"material {name}: {ref} has no {key} map in textures/index.json "
                             "(run the textures stage)") from None
        if file.suffix.lower() != ".dds":
            raise ValueError(f"material {name}: {ref} names {file.name} in textures/index.json; the game "
                             "finds only a .dds under a .png reference (run the textures stage again)")
        return field, f"/levels/{level_name}/{TEX_DIR}/{file.with_suffix('.png').name}", file
    raise ValueError(f"material {name}: map {key} is {ref!r}; it must be 'stock:<path>' or 'tex:<tex_id>'")


def resolve(registry: dict, names, level_name: str, tex_index: dict) -> tuple[dict[str, dict], list[Path]]:
    """(the Material of each name, the texture files to copy); see the module docstring."""
    out, files = {}, set()
    for name in sorted(set(names)):
        if name not in registry:
            raise ValueError(f"material {name} is not in the material registry")
        entry = registry[name] or {}
        stage = {}
        for key, ref in (entry.get("maps") or {}).items():
            field, path, file = _map(name, key, ref, level_name, tex_index)
            stage[field] = path
            if file is not None:
                files.add(file)
        stage.update(entry.get("factors") or {})
        flags = dict(entry.get("flags") or {})
        stage.update({k: flags.pop(k) for k in sorted(STAGE_FLAGS & set(flags))})
        out[name] = material(name, stage, **flags)
    return out, sorted(files)


def write_materials(root, rel_path, mats: dict[str, dict]) -> dict[str, dict]:
    """Write mats to root/rel_path as a materials.json, leaving out those another *.materials.json
    under root already defines (a material bound by shapes in two folders, such as the tile roof
    of the landmarks and the styled buildings, is defined once in the level); returns what was
    written."""
    root, path = Path(root), Path(root) / rel_path
    defined = set()
    for other in root.rglob("*.materials.json"):
        if other.resolve() != path.resolve():
            defined |= set(json.loads(other.read_text(encoding="utf-8")))
    out = {name: m for name, m in mats.items() if name not in defined}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def texture_index(data_dir) -> dict:
    """out/textures/index.json (the textures stage), or {} before that stage has run."""
    path = Path(data_dir) / "textures" / "index.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def copy_textures(files, data_dir, root) -> None:
    """Copy the texture files resolve() named from data_dir/textures/ into root/TEX_DIR."""
    for f in files:
        src = Path(data_dir) / "textures" / f
        if not src.exists():
            raise FileNotFoundError(f"no {src} (textures/index.json lists it); run the textures stage again")
        dest = Path(root) / TEX_DIR / Path(f).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
