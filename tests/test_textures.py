"""The textures stage (textures.py): the generated textures and the CC0 downloads, written as DDS with an index.

No test here touches the network: the downloads are answered by `responses`, and the stage tests run on an
empty registry (an autouse fixture patches `textures.load_registry`) or a patched `fetch`.
"""
import hashlib
import io
import json
import logging
import re
import struct
import zipfile
from pathlib import Path

import pytest
import responses
from PIL import Image

from test_dds import level_coverages, strands

from brmap import fetch as fetch_module
from brmap import beamng_format as bf
from brmap import landmarks, materials, texgen, textures, trees
from brmap.config import Config

DXGI_BC1_SRGB, DXGI_BC3_SRGB, DXGI_BC5 = 72, 78, 83
real_registry = textures.load_registry     # the checked-in registry; the stage tests patch the module's name


def dds_format(path):
    """(fourCC, DXGI format or None, width) from a DDS header."""
    head = path.read_bytes()[:148]
    fourcc = head[84:88]
    return fourcc, struct.unpack("<I", head[128:132])[0] if fourcc == b"DX10" else None, \
        struct.unpack("<I", head[16:20])[0]


def run(tmp_path, **kw):
    textures.run(Config(out_dir=str(tmp_path), raw_dir=str(tmp_path / "raw"), **kw))
    folder = tmp_path / "textures"
    return folder, json.loads((folder / "index.json").read_text(encoding="utf-8"))


def test_generated_textures_written(tmp_path):
    folder, index = run(tmp_path)
    assert textures.SHIP == ("dds",)                  # the in-game spike (S1): the game wants DDS only
    assert (folder / "clock_face_b.color.dds").is_file()
    assert not list(folder.glob("*.png"))             # no PNG beside the DDS
    entry = index["clock_face"]
    assert entry["source"] == "generated" and entry["url"] is None and entry["license"]
    assert entry["files"] == {"b": "clock_face_b.color.dds"}
    digest = hashlib.sha256((folder / "clock_face_b.color.dds").read_bytes()).hexdigest()
    assert entry["sha256"] == {"b": digest}


def test_every_generated_texture_is_indexed(tmp_path):
    folder, index = run(tmp_path)
    assert set(index) == set(textures.GENERATED)
    assert index["bevel"]["files"] == {"nm": "bevel_nm.normal.dds"}
    for tex_id, entry in index.items():
        for key, name in entry["files"].items():
            stem, ext = name.rsplit(".", 1)
            assert (stem, ext) == (f"{tex_id}_{textures.SUFFIX[key]}", "dds")
            assert (folder / name).is_file()


def test_moss_card_is_generated(tmp_path):
    # the Spanish moss strands' card (texgen.moss_card): colour, opacity and normals, 512 x 1024
    folder, index = run(tmp_path)
    assert index["moss"]["files"] == {"b": "moss_b.color.dds", "nm": "moss_nm.normal.dds", "o": "moss_o.data.dds"}
    assert dds_format(folder / "moss_b.color.dds") == (b"DX10", DXGI_BC1_SRGB, 512)


def test_dds_formats_follow_the_map(tmp_path):
    folder, _ = run(tmp_path)
    assert dds_format(folder / "clock_face_b.color.dds")[:2] == (b"DX10", DXGI_BC1_SRGB)  # opaque colour
    assert dds_format(folder / "bevel_nm.normal.dds")[:2] == (b"DX10", DXGI_BC5)          # a normal map


def test_formats_of_other_maps(tmp_path, monkeypatch):
    opaque = Image.new("RGBA", (64, 64), (200, 100, 50, 255))
    clear = Image.new("RGBA", (64, 64), (200, 100, 50, 0))
    monkeypatch.setattr(textures, "GENERATED", {
        "plain": lambda: {"b": opaque, "r": Image.new("L", (64, 64), 128), "o": Image.new("L", (64, 64), 255)},
        "cutout": lambda: {"b": clear}})
    folder, index = run(tmp_path)
    assert dds_format(folder / "plain_b.color.dds")[:2] == (b"DX10", DXGI_BC1_SRGB)        # no alpha: BC1
    assert dds_format(folder / "cutout_b.color.dds")[:2] == (b"DX10", DXGI_BC3_SRGB)       # alpha: BC3
    assert dds_format(folder / "plain_r.data.dds")[:2] == (b"DXT1", None)                  # data: linear BC1
    assert dds_format(folder / "plain_o.data.dds")[:2] == (b"DXT1", None)
    assert index["plain"]["files"] == {"b": "plain_b.color.dds", "r": "plain_r.data.dds",
                                       "o": "plain_o.data.dds"}


def test_size_is_capped(tmp_path, monkeypatch):
    monkeypatch.setattr(textures, "GENERATED", {"big": lambda: {"b": Image.new("RGB", (1024, 1024), "red")}})
    folder, _ = run(tmp_path, texture_max_size=256)
    assert dds_format(folder / "big_b.color.dds")[2] == 256


def test_rectangle_is_capped_on_its_longer_side(tmp_path, monkeypatch):
    monkeypatch.setattr(textures, "GENERATED", {"wide": lambda: {"b": Image.new("RGB", (1024, 512), "red")}})
    folder, _ = run(tmp_path, texture_max_size=256)
    name = folder / "wide_b.color.dds"
    assert struct.unpack("<II", name.read_bytes()[12:20]) == (128, 256)       # height, width


def test_rectangle_within_the_cap_is_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(textures, "GENERATED", {"wide": lambda: {"b": Image.new("RGB", (512, 256), "red")}})
    folder, _ = run(tmp_path, texture_max_size=1024)
    assert struct.unpack("<II", (folder / "wide_b.color.dds").read_bytes()[12:20]) == (256, 512)


@pytest.mark.parametrize("size", [(100, 100), (512, 300), (2, 64)])
def test_sides_must_be_powers_of_two(tmp_path, monkeypatch, size):
    monkeypatch.setattr(textures, "GENERATED", {"odd": lambda: {"b": Image.new("RGB", size)}})
    with pytest.raises(ValueError, match="power-of-two"):
        run(tmp_path)


def test_unknown_map_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(textures, "GENERATED", {"odd": lambda: {"glow": Image.new("RGB", (8, 8))}})
    with pytest.raises(ValueError, match="glow"):
        run(tmp_path)


def test_rerun_gives_the_same_bytes(tmp_path):
    folder, first = run(tmp_path)
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    folder, second = run(tmp_path)
    assert second == first and {p.name: p.read_bytes() for p in folder.iterdir()} == before


# --- the CC0 downloads -----------------------------------------------------------------------

AMBIENT_ASSET = "RoofingTiles012A"
AMBIENT_URL = f"https://ambientcg.com/get?file={AMBIENT_ASSET}_1K-JPG.zip"
AMBIENT = {"source": "ambientcg", "asset": AMBIENT_ASSET, "license": "CC0-1.0",
           "maps": {"b": "Color", "nm": "NormalGL", "r": "Roughness"}}
POLY_URL = "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/metasequoia_bark/metasequoia_bark_{}_1k.jpg"
POLY = {"source": "polyhaven", "asset": "metasequoia_bark", "license": "CC0-1.0",
        "maps": {"b": "diff", "nm": "nor_gl", "r": "rough"}}
REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(fetch_module, "BACKOFF", [0, 0, 0, 0])
    monkeypatch.setattr(textures, "load_registry", lambda *a: {})      # the stage downloads nothing by default


def jpg(colour=(120, 80, 60), size=(32, 32)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, "JPEG")
    return buf.getvalue()


def zip_of(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return buf.getvalue()


def ambient_members() -> dict[str, bytes]:
    """What an ambientCG 1K-JPG zip holds: the maps (both normal variants), a preview, a USD and a material file."""
    maps = ("Color", "NormalDX", "NormalGL", "Roughness", "Displacement", "AmbientOcclusion")
    members = {f"{AMBIENT_ASSET}_1K-JPG_{m}.jpg": jpg((10 * i, 20, 30)) for i, m in enumerate(maps)}
    members[f"{AMBIENT_ASSET}_PREVIEW.png"] = b"\x89PNG not an image"
    members[f"{AMBIENT_ASSET}_1K-JPG.usda"] = b"#usda 1.0"
    members[f"{AMBIENT_ASSET}_1K-JPG.mtlx"] = b"<materialx/>"
    return members


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@responses.activate
def test_fetch_ambientcg_extracts_maps(tmp_path):
    members = ambient_members()
    responses.add(responses.GET, AMBIENT_URL, body=zip_of(members))
    paths = textures.fetch("roof_barrel", AMBIENT, tmp_path)
    folder = tmp_path / "textures" / "roof_barrel"                  # its own folder under raw/textures
    assert paths == {"b": folder / f"{AMBIENT_ASSET}_1K-JPG_Color.jpg",
                     "nm": folder / f"{AMBIENT_ASSET}_1K-JPG_NormalGL.jpg",
                     "r": folder / f"{AMBIENT_ASSET}_1K-JPG_Roughness.jpg"}
    assert paths["nm"].read_bytes() == members[f"{AMBIENT_ASSET}_1K-JPG_NormalGL.jpg"]   # OpenGL, not DirectX
    # only the registry's images are kept: no DirectX normals, depth, occlusion, preview, USD or zip
    assert sorted(p.name for p in folder.iterdir()) == sorted(p.name for p in paths.values())
    assert "User-Agent" in responses.calls[0].request.headers
    assert textures.fetch("roof_barrel", AMBIENT, tmp_path) == paths and len(responses.calls) == 1   # cached


@responses.activate
@pytest.mark.parametrize("unsafe", ["../evil.jpg", "/abs/evil.jpg", "sub/../../evil.jpg", "C:/evil.jpg",
                                    "..\\evil.jpg", "sub\\..\\..\\evil.jpg"])
def test_fetch_ambientcg_refuses_unsafe_members(tmp_path, unsafe):
    responses.add(responses.GET, AMBIENT_URL, body=zip_of({**ambient_members(), unsafe: jpg()}))
    raw = tmp_path / "raw"
    with pytest.raises(ValueError, match="unsafe"):
        textures.fetch("roof_barrel", AMBIENT, raw)
    assert not list(tmp_path.rglob("evil*"))
    assert not list((raw / "textures" / "roof_barrel").glob("*.jpg"))      # nothing was extracted


@responses.activate
def test_fetch_ambientcg_names_the_map_a_zip_lacks(tmp_path):
    members = ambient_members()
    del members[f"{AMBIENT_ASSET}_1K-JPG_Roughness.jpg"]
    responses.add(responses.GET, AMBIENT_URL, body=zip_of(members))
    with pytest.raises(fetch_module.FetchError, match="Roughness"):
        textures.fetch("roof_barrel", AMBIENT, tmp_path)


@responses.activate
def test_fetch_polyhaven_sends_user_agent(tmp_path):
    def reply(request):                     # Poly Haven answers 403 to a request with no real User-Agent
        agent = request.headers.get("User-Agent", "")
        if not agent or agent.startswith("python-requests"):
            return 403, {}, b"forbidden"
        return 200, {}, jpg()

    for m in POLY["maps"].values():
        responses.add_callback(responses.GET, POLY_URL.format(m), callback=reply)
    paths = textures.fetch("bark_cypress", POLY, tmp_path)
    folder = tmp_path / "textures" / "bark_cypress"
    assert paths == {key: folder / f"metasequoia_bark_{m}_1k.jpg" for key, m in POLY["maps"].items()}
    assert all(p.is_file() for p in paths.values())
    assert {c.request.url for c in responses.calls} == {POLY_URL.format(m) for m in POLY["maps"].values()}


@responses.activate
def test_sha_mismatch_raises(tmp_path):
    body = jpg()
    for m in POLY["maps"].values():
        responses.add(responses.GET, POLY_URL.format(m), body=body)
    pinned = {key: sha(body) for key in POLY["maps"]}
    textures.fetch("bark_cypress", {**POLY, "sha256": pinned}, tmp_path)               # the recorded hashes pass
    with pytest.raises(textures.ChecksumError, match=r"nm.*sha256"):
        textures.fetch("bark_cypress", {**POLY, "sha256": {**pinned, "nm": "0" * 64}}, tmp_path)


@responses.activate
def test_a_changed_cached_file_raises(tmp_path):
    body = jpg()
    for m in POLY["maps"].values():
        responses.add(responses.GET, POLY_URL.format(m), body=body)
    entry = {**POLY, "sha256": {key: sha(body) for key in POLY["maps"]}}
    paths = textures.fetch("bark_cypress", entry, tmp_path)
    paths["r"].write_bytes(b"changed")
    with pytest.raises(textures.ChecksumError, match="bark_cypress"):
        textures.fetch("bark_cypress", entry, tmp_path)
    assert len(responses.calls) == 3                                    # the cache was used, not refetched


@responses.activate
def test_a_download_without_a_recorded_sha_is_reported(tmp_path, caplog):
    body = jpg()
    for m in POLY["maps"].values():
        responses.add(responses.GET, POLY_URL.format(m), body=body)
    with caplog.at_level(logging.WARNING, logger="brmap.textures"):
        textures.fetch("bark_cypress", POLY, tmp_path)
    assert sha(body) in caplog.text and "no sha256 recorded" in caplog.text


# --- convert ---------------------------------------------------------------------------------

def test_convert_caps_size_with_full_mips(tmp_path):
    src = tmp_path / "big.jpg"
    Image.new("RGB", (2048, 2048), (90, 60, 40)).save(src)
    files = textures.convert({"b": src}, tmp_path / "out", "bark", 1024)
    assert files == {"b": "bark_b.color.dds"}
    data = (tmp_path / "out" / files["b"]).read_bytes()
    height, width = struct.unpack("<II", data[12:20])
    assert (height, width) == (1024, 1024) and struct.unpack("<I", data[28:32])[0] == 11   # 1024 down to 1
    blocks = sum(max(1, (1024 >> i) // 4) ** 2 for i in range(11))
    assert len(data) == 148 + 8 * blocks                    # DX10 header (sRGB) and BC1 blocks of every level


def test_convert_picks_the_dds_format_by_map(tmp_path):
    colours = {"b": ("RGB", (200, 100, 50)), "nm": ("RGB", (128, 128, 255)), "r": ("L", 128), "o": ("L", 255)}
    srcs = {}
    for key, (mode, colour) in colours.items():
        srcs[key] = tmp_path / f"{key}.jpg"
        Image.new(mode, (64, 64), colour).save(srcs[key])
    files = textures.convert(srcs, tmp_path / "out", "set", 1024)
    assert files == {"b": "set_b.color.dds", "nm": "set_nm.normal.dds", "r": "set_r.data.dds",
                     "o": "set_o.data.dds"}
    out = tmp_path / "out"
    assert dds_format(out / files["b"])[:2] == (b"DX10", DXGI_BC1_SRGB)       # a JPG has no alpha
    assert dds_format(out / files["nm"])[:2] == (b"DX10", DXGI_BC5)
    assert dds_format(out / files["r"])[:2] == (b"DXT1", None)
    assert dds_format(out / files["o"])[:2] == (b"DXT1", None)


# --- the stage -------------------------------------------------------------------------------

def bind(tmp_path, monkeypatch, *tex_ids):
    """Material registries in which a material binds each of tex_ids (the stage fetches only bound sets)."""
    path = tmp_path / "bound_materials.yaml"
    path.write_text("".join(f"br_test_{t}: {{maps: {{b: 'tex:{t}', r: 'stock:/assets/x_r.data.png'}}}}\n"
                            for t in tex_ids), encoding="utf-8")
    monkeypatch.setattr(textures, "MATERIAL_REGISTRIES", (path,))


def test_stage_converts_the_registry_beside_the_generated(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    for key in ("b", "nm", "r"):
        (raw / f"{key}.jpg").write_bytes(jpg())
    asked = []

    def fake_fetch(tex_id, entry, raw_dir):
        asked.append((tex_id, entry["asset"], Path(raw_dir)))
        return {key: raw / f"{key}.jpg" for key in entry["maps"]}

    monkeypatch.setattr(textures, "fetch", fake_fetch)
    monkeypatch.setattr(textures, "load_registry", lambda *a: {"roof_barrel": AMBIENT})
    bind(tmp_path, monkeypatch, "roof_barrel")
    out = tmp_path / "out"
    textures.run(Config(out_dir=str(out), raw_dir=str(raw)))
    assert asked == [("roof_barrel", AMBIENT_ASSET, raw)]
    index = json.loads((out / "textures" / "index.json").read_text(encoding="utf-8"))
    assert set(index) == set(textures.GENERATED) | {"roof_barrel"}
    entry = index["roof_barrel"]
    assert (entry["source"], entry["license"], entry["url"]) == ("ambientcg", "CC0-1.0", AMBIENT_URL)
    assert entry["files"] == {"b": "roof_barrel_b.color.dds", "nm": "roof_barrel_nm.normal.dds",
                              "r": "roof_barrel_r.data.dds"}
    for key, name in entry["files"].items():
        assert entry["sha256"][key] == sha((out / "textures" / name).read_bytes())


def test_stage_fetches_only_the_sets_a_material_binds(tmp_path, monkeypatch, caplog):
    raw = tmp_path / "raw"
    raw.mkdir()
    for key in ("b", "nm", "r"):
        (raw / f"{key}.jpg").write_bytes(jpg())
    asked = []

    def fake_fetch(tex_id, entry, raw_dir):
        asked.append(tex_id)
        return {key: raw / f"{key}.jpg" for key in entry["maps"]}

    monkeypatch.setattr(textures, "fetch", fake_fetch)
    monkeypatch.setattr(textures, "load_registry", lambda *a: {"roof_barrel": AMBIENT, "turf": AMBIENT,
                                                               "bark_cypress": POLY})
    bind(tmp_path, monkeypatch, "turf", "moss")                 # moss is generated, turf a CC0 set
    out = tmp_path / "out"
    with caplog.at_level(logging.INFO, logger="brmap.textures"):
        textures.run(Config(out_dir=str(out), raw_dir=str(raw)))
    assert asked == ["turf"]                                    # nothing downloaded for the unbound sets
    index = json.loads((out / "textures" / "index.json").read_text(encoding="utf-8"))
    assert set(index) == set(textures.GENERATED) | {"turf"}
    assert not list((out / "textures").glob("roof_barrel_*")) and not list((out / "textures").glob("bark_cypress_*"))
    assert "not fetched, bound to no material: bark_cypress, roof_barrel" in caplog.text


def test_bound_textures_are_the_tex_maps_of_the_landmark_and_tree_materials():
    assert textures.MATERIAL_REGISTRIES == (landmarks.MATERIAL_REGISTRY, trees.TREE_MATERIALS)
    bound = textures.bound_textures()
    assert {"moss", "turf", "clock_face", "lsu_wall"} <= bound
    assert bound & set(real_registry()) == {"turf"}             # the other five CC0 sets are unbound


def test_source_urls():
    assert textures.source_url(AMBIENT) == AMBIENT_URL
    assert textures.source_url(POLY) == "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/metasequoia_bark/"


# --- the registry and the docs ---------------------------------------------------------------

def test_registry_lists_the_cc0_sets():
    registry = real_registry()
    assert {k: (e["source"], e["asset"]) for k, e in registry.items()} == {
        "roof_barrel": ("ambientcg", "RoofingTiles012A"), "limestone": ("ambientcg", "Tiles143"),
        "turf": ("ambientcg", "Grass005"), "bark_cypress": ("polyhaven", "metasequoia_bark"),
        "bark_tupelo": ("ambientcg", "Bark004"), "needles": ("ambientcg", "LeafSet019")}
    assert set(registry["needles"]["maps"]) == {"b", "nm", "r", "o"}          # the leaf cut-out adds Opacity
    gl = {"ambientcg": "NormalGL", "polyhaven": "nor_gl"}
    assert texgen.NORMAL_CONVENTION == "opengl"                                # the in-game spike (S1)
    for tex_id, entry in registry.items():
        assert entry["license"] == "CC0-1.0", tex_id
        assert entry["maps"]["nm"] == gl[entry["source"]], tex_id              # the OpenGL variant
        assert {"b", "nm", "r"} <= set(entry["maps"]) <= set(textures.SUFFIX), tex_id


def test_registry_records_a_sha256_for_every_map():
    for tex_id, entry in real_registry().items():
        assert set(entry["sha256"]) == set(entry["maps"]), tex_id
        assert all(re.fullmatch(r"[0-9a-f]{64}", h) for h in entry["sha256"].values()), tex_id


def test_registry_refuses_a_bad_entry(tmp_path):
    path = tmp_path / "textures.yaml"
    path.write_text("odd: {source: flickr, asset: X, maps: {b: Color}, license: CC0-1.0}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="odd.*flickr"):
        real_registry(path)
    path.write_text("odd: {source: ambientcg, asset: X, maps: {glow: Color}, license: CC0-1.0}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="odd.*glow"):
        real_registry(path)


def test_docs_list_every_texture():
    docs = (REPO / "docs" / "textures.md").read_text(encoding="utf-8")
    registry = real_registry()
    for tex_id in [*registry, *textures.GENERATED]:
        assert f"`{tex_id}`" in docs, f"docs/textures.md does not list {tex_id}"
    for tex_id, entry in registry.items():
        assert entry["asset"] in docs and entry["license"] in docs, tex_id
    generated = docs.split("## Generated textures", 1)[1].split("\n## ", 1)[0]
    rows = {line.split("|")[1].strip(" `"): line for line in generated.splitlines() if line.startswith("| `")}
    assert set(rows) == set(textures.GENERATED)
    assert [t for t, row in rows.items() if "| original, generated by brmap |" not in row] == []  # the licence
    # every stock map the landmark, styled and custom tree materials bind: its folder and its file stem
    for registry_path in (landmarks.MATERIAL_REGISTRY, trees.TREE_MATERIALS):
        for name, entry in materials.load_registry(registry_path).items():
            for ref in (entry.get("maps") or {}).values():
                if ref.startswith("stock:"):
                    folder, file = ref.split(":", 1)[1].rsplit("/", 1)
                    tail = "/".join(folder.split("/")[-2:]) + "/"
                    stem = re.sub(r"_(b|nm|r|o|ao)\.(color|normal|data)\.png$", "", file)
                    assert tail in docs and stem in docs, f"docs/textures.md does not list {name}'s {ref}"
    assert bf.PLATE_NORMAL in docs  # the licence plate design's stock normal map


# --- alpha-tested textures: coverage-preserving mips ------------------------------------------

DATA = REPO / "src" / "brmap" / "data"


def run_only(tmp_path, monkeypatch, *tex_ids):
    """The stage on just these generated textures (the whole set takes seconds)."""
    monkeypatch.setattr(textures, "GENERATED", {tex_id: textures.GENERATED[tex_id] for tex_id in tex_ids})
    return run(tmp_path)


def declared_alpha_refs():
    """{tex_id: alpha_ref} for the generated textures and the registry's."""
    return {**textures.ALPHA_REF, **{k: e["alpha_ref"] for k, e in real_registry().items() if "alpha_ref" in e}}


def test_alpha_tested_textures_declare_the_alphaRef_of_their_materials():
    """The texture's mips keep the share of texels at or above its alpha_ref, so it has to be the alphaRef of
    each alphaTest material that cuts it out; and a texture only declares one if such a material binds it."""
    bound = {}
    for name in ("landmark_materials.yaml", "tree_materials.yaml"):
        for entry in materials.load_registry(DATA / name).values():
            flags = entry.get("flags") or {}
            for key, ref in (entry.get("maps") or {}).items():
                kind, _, tex_id = str(ref).partition(":")
                if flags.get("alphaTest") and kind == "tex" and key in ("b", "o"):
                    bound.setdefault(tex_id, set()).add(flags.get("alphaRef"))
    assert bound == {tex_id: {ref} for tex_id, ref in declared_alpha_refs().items()}
    assert declared_alpha_refs() == {"moss": 67, "field_atlas": 127, "gothic_window": 127}
    assert set(textures.ALPHA_REF) <= set(textures.GENERATED)


def test_moss_opacity_mips_keep_the_cards_coverage(tmp_path, monkeypatch):
    """The moss card is cut out at alphaRef 67: plain halvings turn its strands into solid grey far away."""
    folder, _ = run_only(tmp_path, monkeypatch, "moss")
    shares = level_coverages((folder / "moss_o.data.dds").read_bytes(), "BC1", 512, 1024)
    assert len(shares) == 11 and max(abs(s - shares[0]) for s in shares[:7]) <= 0.05        # down to 8 x 16
    plain, _ = textures.write_maps("moss", {"o": texgen.moss_card()["o"]}, tmp_path / "plain", 1024)
    drift = level_coverages((tmp_path / "plain" / plain["o"]).read_bytes(), "BC1", 512, 1024)
    assert max(abs(s - drift[0]) for s in drift[:7]) > 0.1                                   # the test bites


@pytest.mark.parametrize("tex_id, size", [("field_atlas", (1024, 512)), ("gothic_window", (256, 256))])
def test_decals_are_written_with_their_alphaRef(tmp_path, monkeypatch, tex_id, size):
    """The decals are mostly solid, so their coverage barely drifts; the stage still hands them their alpha_ref."""
    folder, index = run_only(tmp_path, monkeypatch, tex_id)
    name = index[tex_id]["files"]["b"]
    alpha_ref = textures.ALPHA_REF[tex_id]
    shares = level_coverages((folder / name).read_bytes(), "BC3", *size, threshold=alpha_ref)
    assert max(abs(s - shares[0]) for s in shares[:6]) <= 0.05
    plain, _ = textures.write_maps(tex_id, textures.GENERATED[tex_id](), tmp_path / "plain", 1024)
    assert (folder / name).read_bytes() != (tmp_path / "plain" / plain["b"]).read_bytes()


def test_write_maps_applies_alpha_ref_to_the_opacity_maps_only(tmp_path):
    a = strands()
    grey = Image.fromarray(a)
    maps = {"b": Image.merge("RGBA", [Image.new("L", grey.size, 200)] * 3 + [grey]), "o": grey, "r": grey,
            "nm": Image.new("RGB", grey.size, (128, 128, 255))}
    _, plain = textures.write_maps("t", maps, tmp_path / "plain", 1024)
    files, kept = textures.write_maps("t", maps, tmp_path / "kept", 1024, alpha_ref=67)
    assert kept["b"] != plain["b"] and kept["o"] != plain["o"]
    assert kept["r"] == plain["r"] and kept["nm"] == plain["nm"]           # roughness and normals have no opacity
    base = float((a >= 67).mean())
    for key, fmt in (("b", "BC3"), ("o", "BC1")):
        shares = level_coverages((tmp_path / "kept" / files[key]).read_bytes(), fmt, 256, 256)
        assert max(abs(s - base) for s in shares[:6]) <= 0.05, key


def test_write_maps_without_alpha_ref_is_unchanged(tmp_path):
    maps = {"b": Image.new("RGBA", (64, 64), (200, 100, 50, 90)), "o": Image.fromarray(strands(64))}
    assert textures.write_maps("t", maps, tmp_path / "a", 1024)[1] == \
        textures.write_maps("t", maps, tmp_path / "b", 1024, alpha_ref=None)[1]


@pytest.mark.parametrize("maps", [{"b": Image.new("RGB", (16, 16))}, {"b": Image.new("RGBA", (16, 16), (1, 2, 3, 255))},
                                  {"r": Image.new("L", (16, 16))}, {"nm": Image.new("RGB", (16, 16))}])
def test_write_maps_refuses_an_alpha_ref_without_an_opacity_map(tmp_path, maps):
    with pytest.raises(ValueError, match="alpha_ref.*opacity"):
        textures.write_maps("t", maps, tmp_path, 1024, alpha_ref=67)
    assert not list(tmp_path.iterdir())                                      # refused before anything is written


def test_convert_passes_the_alpha_ref_on(tmp_path):
    src = tmp_path / "strands.png"
    Image.fromarray(strands()).save(src)
    files = textures.convert({"o": src}, tmp_path / "out", "set", 1024, alpha_ref=67)
    shares = level_coverages((tmp_path / "out" / files["o"]).read_bytes(), "BC1", 256, 256)
    assert max(abs(s - shares[0]) for s in shares[:6]) <= 0.05


def test_registry_entry_can_declare_an_alpha_ref(tmp_path, monkeypatch):
    """An entry's alpha_ref reaches its opacity map's mips in the stage."""
    raw = tmp_path / "raw"
    raw.mkdir()
    Image.new("RGB", (256, 256), (60, 90, 40)).save(raw / "b.png")
    Image.fromarray(strands()).save(raw / "o.png")
    monkeypatch.setattr(textures, "fetch", lambda tex_id, entry, raw_dir: {k: raw / f"{k}.png" for k in entry["maps"]})
    bind(tmp_path, monkeypatch, "leafy")
    entry = {**AMBIENT, "maps": {"b": "Color", "o": "Opacity"}}
    shares = {}
    for label, extra in (("plain", {}), ("kept", {"alpha_ref": 67})):
        monkeypatch.setattr(textures, "load_registry", lambda *a, e={**entry, **extra}: {"leafy": e})
        folder, _ = run_only(tmp_path / label, monkeypatch)
        shares[label] = level_coverages((folder / "leafy_o.data.dds").read_bytes(), "BC1", 256, 256)
    assert max(abs(s - shares["kept"][0]) for s in shares["kept"][:6]) <= 0.05
    assert max(abs(s - shares["plain"][0]) for s in shares["plain"][:6]) > 0.1


@pytest.mark.parametrize("bad", [0, 256, -5, 67.5, "67", True, None])
def test_registry_refuses_a_bad_alpha_ref(tmp_path, bad):
    path = tmp_path / "textures.yaml"
    path.write_text(f"odd: {{source: ambientcg, asset: X, maps: {{b: Color, o: Opacity}}, license: CC0-1.0, "
                    f"alpha_ref: {json.dumps(bad)}}}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="odd.*alpha_ref"):
        real_registry(path)


def test_registry_accepts_an_alpha_ref(tmp_path):
    path = tmp_path / "textures.yaml"
    path.write_text("ok: {source: ambientcg, asset: X, maps: {b: Color, o: Opacity}, license: CC0-1.0, alpha_ref: 90}\n",
                    encoding="utf-8")
    assert real_registry(path)["ok"]["alpha_ref"] == 90
