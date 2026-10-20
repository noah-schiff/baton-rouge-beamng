"""The textures stage: the repo's own and the CC0 textures, written into out/textures/ for the level.

* GENERATED: {tex_id: callable} for the textures drawn by brmap.texgen and brmap.logos. Each callable
  returns {map: Image}, map one of SUFFIX (b, nm, r, o). The styled downtown walls are
  facade_<style> (texgen.facade, its metallic factor left to the material), the Old State
  Capitol's window gothic_window, the Spanish moss strands' card moss (texgen.moss_card).
* ALPHA_REF: {tex_id: alphaRef} for the generated textures a material cuts out with an alpha test (moss,
  field_atlas, gothic_window), each the alphaRef of the materials that bind it (a test checks that they
  agree). A registry entry declares its own, as `alpha_ref`. See "Alpha-tested textures" below.
* The registry, src/brmap/data/textures.yaml (load_registry): the CC0 sets, one entry each,
  `tex_id: {source, asset, maps: {b|nm|r|o: <name on the site>}, license, sha256: {map: hex}[, alpha_ref]}`.
  - source is "ambientcg" (maps named like Color, NormalGL, Roughness, Opacity) or "polyhaven"
    (diff, nor_gl, rough). The normal map is always the OpenGL variant (green = +V up), the
    convention the in-game spike S1 settled (texgen.NORMAL_CONVENTION).
  - sha256 is the hash of each downloaded file, recorded when the set was first fetched.
* fetch(tex_id, entry, raw_dir) -> {map: Path}: the 1K JPG maps of one set, in their own folder
  raw_dir/textures/<tex_id>/ (raw_dir is cfg.raw_dir, shared by every config). Cached: a file already
  there is not downloaded again. The sha256 of every file is checked against the registry, and a
  different one raises ChecksumError; a map with no recorded sha256 is logged with its hash.
  - ambientcg: the zip https://ambientcg.com/get?file=<asset>_1K-JPG.zip. A downloaded archive is
    untrusted: no member is extracted by its own name. Only the images of the registry's maps
    (<asset>_1K-JPG_<Map>.jpg) are read out of it, under that name; an absolute or ".." member
    name anywhere in the zip is refused, and the zip is deleted afterwards.
  - polyhaven: https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/<asset>/<asset>_<map>_1k.jpg,
    one file per map. brmap.fetch.download sends the User-Agent header Poly Haven wants (a 403 without).
* convert(maps, out_dir, tex_id, max_size) -> {map: file name}: the fetched images, scaled down to
  max_size and written as DDS (write_maps).
* run(cfg, raw_dir=None, out_dir=None) fetches and converts the registry entries (load_registry()) a
  material binds (bound_textures(): a `tex:<id>` map in MATERIAL_REGISTRIES, the landmark, styled and
  custom tree materials) into out_dir and writes every GENERATED texture beside them, as
  out/textures/<tex_id>_<suffix>.dds (suffix from SUFFIX: b.color, nm.normal, r.data, o.data),
  and out/textures/index.json: {tex_id: {"source": "generated" | "ambientcg" | "polyhaven", "license",
  "url": None | source_url(entry), "sha256": {map: hex}, "files": {map: file name}}}. "files" and
  "sha256" name those .dds files; brmap.materials points a material at the .png name of the file it
  finds there (<tex_id>_<suffix>.png, the stock convention) and copies the .dds.
  - A registry set no material binds is not fetched (no download, no sha256 check on every build) and
    not indexed; the log names it. Its entry stays in the registry for when a material binds it.
  - A map's sides must each be a power of two (a 2:1 rectangle such as the field atlas is fine);
    one whose longer side is over cfg.texture_max_size is scaled down to it, keeping its shape.
  - DDS (brmap.dds): BC3 for a b map with alpha, BC1 for the other b maps and the .data maps,
    BC5 for nm. b maps are marked sRGB, as the stock *_b.color.dds are.
* Alpha-tested textures: a texture with an alpha_ref (1-255, a material's alphaRef) gets
  coverage-preserving mips on its opacity (brmap.dds.write_dds's alpha_coverage_ref): BC3's alpha of the b
  map when it has alpha, and the o map. Plain halvings average thin strands into grey that crosses the
  threshold over whole texels, so the moss card turned into a solid rectangle at a distance (the full-map
  in-game check on 0.40). One declared for a texture without an opacity map is an error.
* SHIP is DDS only. The in-game spike S1 (2026-10-19) found that the game wrote no cooked .dds
  for a PNG used as a mesh texture and rendered the shipped DDS right, so a PNG would only be
  decoded and compressed at load; the spec's "Spike results (v0.2)" records it.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import zipfile
from pathlib import Path

import yaml
from PIL import Image

from . import dds, logos, texgen
from .fetch import FetchError, download
from .materials import MAP_FIELDS
from .materials import load_registry as load_material_registry

log = logging.getLogger(__name__)

SHIP = ("dds",)
SUFFIX = {key: MAP_FIELDS[key][1] for key in ("b", "nm", "r", "o")}
INDEX_FILE = "index.json"
LICENSE = "generated by brmap (src/brmap/texgen.py, src/brmap/logos.py); no third-party rights"
DATA = Path(__file__).resolve().parent / "data"
REGISTRY = DATA / "textures.yaml"
# the material registries whose tex:<id> maps bind a texture (landmarks.MATERIAL_REGISTRY, which the styled
# buildings share, and trees.TREE_MATERIALS)
MATERIAL_REGISTRIES = (DATA / "landmark_materials.yaml", DATA / "tree_materials.yaml")

AMBIENTCG_URL = "https://ambientcg.com/get?file={asset}_1K-JPG.zip"
AMBIENTCG_FILE = "{asset}_1K-JPG_{map}.jpg"
POLYHAVEN_URL = "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/{asset}/"
POLYHAVEN_FILE = "{asset}_{map}_1k.jpg"
SOURCES = ("ambientcg", "polyhaven")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
MAX_IMAGE_BYTES = 64 << 20        # the largest image read out of a zip (a 1K map is a few MB)


def _images(maps: dict) -> dict[str, Image.Image]:
    """The image maps of a texgen result, without its factors (facade's "m")."""
    return {key: img for key, img in maps.items() if isinstance(img, Image.Image)}


GENERATED = {
    "clock_face": lambda: {"b": texgen.clock_face()},
    "bevel": lambda: {"nm": texgen.bevel_normal()},
    "field_atlas": lambda: {"b": logos.field_atlas()},
    "board": lambda: {"b": logos.board()},
    "lsu_wall": texgen.lsu_wall,
    **{f"facade_{style}": (lambda style=style: _images(texgen.facade(style))) for style in texgen.FACADES},
    "gothic_window": lambda: {"b": texgen.gothic_window()},
    "moss": texgen.moss_card,
}
# the alphaRef of the materials that cut each of these out (tree_materials.yaml, landmark_materials.yaml)
ALPHA_REF = {"moss": 67, "field_atlas": 127, "gothic_window": 127}


class ChecksumError(FetchError):
    """A texture file's sha256 is not the one the registry records."""


# ------------------------------------------------------------------ registry

def _word(tex_id: str, what: str, value) -> None:
    """Ids and names go into folder names and URLs: letters, digits and underscores only."""
    if not re.fullmatch(r"[A-Za-z0-9_]+", str(value)):
        raise ValueError(f"texture {tex_id}: {what} {value!r} must be letters, digits and underscores")


def _check_alpha_ref(tex_id: str, value) -> None:
    """A texture's alpha_ref is the alphaRef of the materials that cut it out: a whole number from 1 to 255."""
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 255:
        raise ValueError(f"texture {tex_id}: alpha_ref {value!r} must be a whole number from 1 to 255 "
                         "(the alphaRef of the materials that cut it out)")


def load_registry(path=REGISTRY) -> dict[str, dict]:
    """The CC0 registry (textures.yaml) as {tex_id: entry}; an entry that can't be fetched is refused."""
    registry = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for tex_id, entry in registry.items():
        _word(tex_id, "id", tex_id)
        if entry.get("source") not in SOURCES:
            raise ValueError(f"texture {tex_id}: unknown source {entry.get('source')!r} (one of {', '.join(SOURCES)})")
        _word(tex_id, "asset", entry.get("asset"))
        if not entry.get("maps"):
            raise ValueError(f"texture {tex_id}: it has no maps")
        for key, name in entry["maps"].items():
            if key not in SUFFIX:
                raise ValueError(f"texture {tex_id}: unknown map {key!r} (one of {', '.join(SUFFIX)})")
            _word(tex_id, f"map {key}", name)
        if not entry.get("license"):
            raise ValueError(f"texture {tex_id}: it records no license")
        if "alpha_ref" in entry:
            _check_alpha_ref(tex_id, entry["alpha_ref"])
    return registry


def bound_textures() -> set[str]:
    """The tex_ids a material in MATERIAL_REGISTRIES binds (one of its maps is `tex:<tex_id>`)."""
    bound = set()
    for path in MATERIAL_REGISTRIES:
        for entry in load_material_registry(path).values():
            for ref in ((entry or {}).get("maps") or {}).values():
                kind, _, tex_id = str(ref).partition(":")
                if kind == "tex" and tex_id:
                    bound.add(tex_id)
    return bound


def source_url(entry: dict) -> str:
    """Where a set comes from: ambientCG's zip, or Poly Haven's folder of the maps."""
    template = AMBIENTCG_URL if entry["source"] == "ambientcg" else POLYHAVEN_URL
    return template.format(asset=entry["asset"])


def _file_name(entry: dict, map_name: str) -> str:
    template = AMBIENTCG_FILE if entry["source"] == "ambientcg" else POLYHAVEN_FILE
    return template.format(asset=entry["asset"], map=map_name)


# ------------------------------------------------------------------ fetch

def _has_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _unsafe(name: str) -> bool:
    """A zip member name that is absolute (or has a drive) or climbs out with "..", either slash."""
    parts = name.replace("\\", "/").split("/")
    return name.startswith(("/", "\\")) or re.match(r"[A-Za-z]:", name) is not None or ".." in parts


def _write(path: Path, data: bytes) -> None:
    part = path.with_name(path.name + ".part")
    part.write_bytes(data)
    part.replace(path)


def _from_ambientcg(tex_id: str, entry: dict, folder: Path, missing: dict[str, Path]) -> None:
    """Download the zip and keep the images of the `missing` maps, each under the name in `missing`."""
    url = source_url(entry)
    archive = folder / f"{entry['asset']}_1K-JPG.zip"
    download(url, archive)
    try:
        with zipfile.ZipFile(archive) as z:
            bad = [i.filename for i in z.infolist() if _unsafe(i.filename)]
            if bad:
                raise ValueError(f"{url}: the zip has an unsafe member name, {bad[0]!r}; nothing was extracted")
            images = {}
            for info in z.infolist():
                name = info.filename.replace("\\", "/").rsplit("/", 1)[-1]
                if not info.is_dir() and name.lower().endswith(IMAGE_SUFFIXES):
                    images.setdefault(name.lower(), info)
            for key, path in missing.items():
                info = images.get(path.name.lower())
                if info is None:
                    raise FetchError(f"{url}: the zip has no {path.name} for map {key} of {tex_id} "
                                     f"(its images: {', '.join(sorted(images)) or 'none'})")
                if info.file_size > MAX_IMAGE_BYTES:
                    raise FetchError(f"{url}: {info.filename} is {info.file_size} bytes; the limit is {MAX_IMAGE_BYTES}")
                _write(path, z.read(info))
    except zipfile.BadZipFile as exc:
        raise FetchError(f"{url}: not a zip file ({exc})") from exc
    finally:
        archive.unlink(missing_ok=True)


def _from_polyhaven(entry: dict, missing: dict[str, Path]) -> None:
    for path in missing.values():
        download(source_url(entry) + path.name, path)


def _check_sha256(tex_id: str, entry: dict, folder: Path, paths: dict[str, Path]) -> None:
    recorded = entry.get("sha256") or {}
    for key, path in sorted(paths.items()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        want = recorded.get(key)
        if want is None:
            log.warning("texture %s map %s (%s): no sha256 recorded yet; this file's is %s "
                        "(record it under sha256 in textures.yaml)", tex_id, key, path.name, digest)
        elif digest != str(want).lower():
            raise ChecksumError(f"texture {tex_id} map {key} ({path.name}): sha256 is {digest} but textures.yaml "
                                f"records {want}. If the site changed the set, check the new files and record "
                                f"the new hash; delete {folder} to download it again.")


def fetch(tex_id: str, entry: dict, raw_dir) -> dict[str, Path]:
    """The maps of one CC0 set as {map: file}, in raw_dir/textures/<tex_id>/; see the module docstring."""
    folder = Path(raw_dir) / "textures" / tex_id
    paths = {key: folder / _file_name(entry, name) for key, name in entry["maps"].items()}
    missing = {key: path for key, path in paths.items() if not _has_file(path)}
    if missing:
        folder.mkdir(parents=True, exist_ok=True)
        if entry["source"] == "ambientcg":
            _from_ambientcg(tex_id, entry, folder, missing)
        else:
            _from_polyhaven(entry, missing)
    else:
        log.info("cached: %s", folder)
    _check_sha256(tex_id, entry, folder, paths)
    return paths


# ------------------------------------------------------------------ convert

def _has_alpha(img: Image.Image) -> bool:
    return img.mode in ("RGBA", "LA") and img.getchannel("A").getextrema()[0] < 255


def _is_opacity(key: str, img: Image.Image) -> bool:
    """Whether a map holds the opacity an alpha test reads: the o map, or a b map with alpha."""
    return key == "o" or (key == "b" and _has_alpha(img))


def _dds_format(key: str, img: Image.Image) -> tuple[str, bool]:
    """(BC format, sRGB) for one map."""
    if key == "b":
        return ("BC3" if _has_alpha(img) else "BC1"), True
    if key == "nm":
        return "BC5", False
    return "BC1", False


def _capped(tex_id: str, key: str, img: Image.Image, max_size: int) -> Image.Image:
    if any(n < 4 or n & (n - 1) for n in img.size):
        raise ValueError(f"texture {tex_id} map {key} is {img.width} x {img.height}; "
                         "each side must be a power-of-two (4 or more)")
    longer = max(img.size)
    if longer > max_size:
        img = img.resize((max(4, img.width * max_size // longer), max(4, img.height * max_size // longer)),
                         Image.Resampling.LANCZOS)
    return img


def write_maps(tex_id: str, maps: dict[str, Image.Image], folder: Path, max_size: int,
               alpha_ref: int | None = None) -> tuple[dict, dict]:
    """Write one texture's maps in the SHIP format; ({map: file name}, {map: sha256}).

    alpha_ref (1-255, an alphaRef), for an alpha-tested texture, keeps the coverage of the opacity through
    the mips: the o map's, and the b map's alpha (BC3) when it has one; see the module docstring."""
    if alpha_ref is not None and not any(_is_opacity(key, img) for key, img in maps.items()):
        raise ValueError(f"texture {tex_id}: alpha_ref {alpha_ref} needs an opacity map: an o map, or a b map "
                         f"with alpha (it has {', '.join(sorted(maps))})")
    files, sha = {}, {}
    for key, img in sorted(maps.items()):
        if key not in SUFFIX:
            raise ValueError(f"texture {tex_id}: unknown map {key!r} (one of {', '.join(SUFFIX)})")
        img = _capped(tex_id, key, img, max_size)
        stem = f"{tex_id}_{SUFFIX[key]}"
        for ext in SHIP:
            if ext != "dds":
                raise ValueError(f"SHIP names an unknown format {ext!r}")
            fmt, srgb = _dds_format(key, img)
            cut_out = alpha_ref is not None and _is_opacity(key, img)
            dds.write_dds(img, folder / f"{stem}.{ext}", fmt, srgb=srgb,
                          alpha_coverage_ref=alpha_ref / 255 if cut_out else None)
        files[key] = f"{stem}.{SHIP[0]}"
        sha[key] = hashlib.sha256((folder / files[key]).read_bytes()).hexdigest()
    return files, sha


def _load(path, key: str) -> Image.Image:
    """A fetched image as the map needs it: colour and normals in RGB (RGBA when it has alpha), data in grey."""
    with Image.open(path) as img:
        img.load()
        if key in ("b", "nm"):
            return img.convert("RGBA" if "A" in img.getbands() else "RGB")
        return img.convert("L")


def convert(maps: dict[str, Path], out_dir, tex_id: str, max_size: int, alpha_ref: int | None = None) -> dict[str, str]:
    """The fetched images of one set, scaled down to max_size and written as DDS in out_dir;
    {map: file name}. The normal map is the OpenGL variant the registry names, so it goes as it is.
    alpha_ref is the entry's, for write_maps."""
    images = {key: _load(path, key) for key, path in maps.items()}
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    files, _ = write_maps(tex_id, images, folder, max_size, alpha_ref)
    return files


# ------------------------------------------------------------------ the stage

def run(cfg, raw_dir=None, out_dir=None) -> None:
    folder = Path(out_dir or cfg.out_dir) / "textures"
    folder.mkdir(parents=True, exist_ok=True)
    registry = load_registry()
    clash = sorted(set(registry) & set(GENERATED))
    if clash:
        raise ValueError(f"texture {clash[0]} is both generated and in the CC0 registry")
    bound = bound_textures()
    unbound = sorted(set(registry) - bound)
    if unbound:
        log.info("textures: %d CC0 sets not fetched, bound to no material: %s", len(unbound), ", ".join(unbound))
    index = {}
    for tex_id, entry in registry.items():
        if tex_id not in bound:
            continue
        maps = fetch(tex_id, entry, Path(raw_dir or cfg.raw_dir))
        files = convert(maps, folder, tex_id, cfg.texture_max_size, entry.get("alpha_ref"))
        sha = {key: hashlib.sha256((folder / name).read_bytes()).hexdigest() for key, name in files.items()}
        index[tex_id] = {"source": entry["source"], "license": entry["license"], "url": source_url(entry),
                         "sha256": sha, "files": files}
    downloaded = len(index)
    for tex_id, make in GENERATED.items():
        files, sha = write_maps(tex_id, make(), folder, cfg.texture_max_size, ALPHA_REF.get(tex_id))
        index[tex_id] = {"source": "generated", "license": LICENSE, "url": None, "sha256": sha, "files": files}
    path = folder / INDEX_FILE
    path.write_text(json.dumps(index, indent=1, sort_keys=True), encoding="utf-8")
    log.info("textures: %d CC0 and %d generated (%s) as %s in %s", downloaded, len(index) - downloaded,
             ", ".join(index), " and ".join(SHIP), folder)
