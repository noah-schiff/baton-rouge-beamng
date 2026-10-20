"""The DDS writer: block-compressed textures with a full mip chain (brmap.dds)."""
import struct

import numpy as np
import pytest
from PIL import Image

from brmap import dds

BLOCK_BYTES = {"BC1": 8, "BC3": 16, "BC5": 16}
PIXEL_FORMAT = {"BC1": (1, "BC1"), "BC3": (3, "BC3"), "BC5": (5, "BC5")}
DDSD_FLAGS = 0x1 | 0x2 | 0x4 | 0x1000 | 0x20000 | 0x80000  # CAPS HEIGHT WIDTH PIXELFORMAT MIPMAPCOUNT LINEARSIZE
CAPS = 0x1000 | 0x400000 | 0x8  # TEXTURE | MIPMAP | COMPLEX


def u32(b, offset):
    return struct.unpack_from("<I", b, offset)[0]


def level_sizes(side):
    sizes = [side]
    while sizes[-1] > 1:
        sizes.append(sizes[-1] // 2)
    return sizes


def payload_len(fmt, side):
    return BLOCK_BYTES[fmt] * sum(max(1, s // 4) ** 2 for s in level_sizes(side))


def decode_level(b, fmt, side, level, header_len):
    """Decode mip `level` of the file bytes `b` (the mip levels follow one header, largest first)."""
    sizes = level_sizes(side)
    start = header_len + BLOCK_BYTES[fmt] * sum(max(1, s // 4) ** 2 for s in sizes[:level])
    s = sizes[level]
    data = b[start:start + BLOCK_BYTES[fmt] * max(1, s // 4) ** 2]
    return Image.frombytes("RGB" if fmt == "BC5" else "RGBA", (s, s), data, "bcn", PIXEL_FORMAT[fmt])


def level_dims(width, height):
    dims = [(width, height)]
    while dims[-1] != (1, 1):
        dims.append((max(1, dims[-1][0] // 2), max(1, dims[-1][1] // 2)))
    return dims


def blocks(width, height):
    return max(1, (width + 3) // 4) * max(1, (height + 3) // 4)


def test_bc1_header_and_mips(tmp_path):
    p = dds.write_dds(Image.new("RGB", (256, 256), (200, 120, 40)), tmp_path / "a.dds", "BC1")
    b = p.read_bytes()
    assert b[:4] == b"DDS " and b[84:88] == b"DXT1"
    assert struct.unpack_from("<I", b, 28)[0] == 9          # dwMipMapCount
    blocks = sum(max(1, s // 4) ** 2 for s in (256, 128, 64, 32, 16, 8, 4, 2, 1))
    assert len(b) == 128 + 8 * blocks


def test_bc3_keeps_alpha(tmp_path):
    im = Image.new("RGBA", (64, 64), (255, 255, 255, 255)); im.paste((0, 0, 0, 0), (0, 0, 32, 64))
    back = Image.open(dds.write_dds(im, tmp_path / "a.dds", "BC3")).convert("RGBA")
    assert back.getpixel((8, 32))[3] < 16 and back.getpixel((56, 32))[3] > 240


def test_bc5_uses_dx10_header(tmp_path):
    b = dds.write_dds(Image.new("RGB", (64, 64), (128, 128, 255)), tmp_path / "n.dds", "BC5").read_bytes()
    assert b[84:88] == b"DX10" and struct.unpack_from("<I", b, 128)[0] == 83


def test_rejects_non_power_of_two(tmp_path):
    with pytest.raises(ValueError):
        dds.write_dds(Image.new("RGB", (100, 100)), tmp_path / "x.dds", "BC1")


@pytest.mark.parametrize("size", [(2, 2), (1, 1), (8192, 8192), (0, 0), (64, 2), (8192, 4), (96, 32), (32, 100)])
def test_rejects_other_sizes(tmp_path, size):
    """Each side a power of two from 4 to 4096."""
    with pytest.raises(ValueError, match="power of two"):
        dds.write_dds(Image.new("RGB", size), tmp_path / "x.dds", "BC1")
    assert not (tmp_path / "x.dds").exists()


def test_rejects_unknown_format_and_srgb_bc5(tmp_path):
    im = Image.new("RGB", (16, 16))
    with pytest.raises(ValueError, match="BC7"):
        dds.write_dds(im, tmp_path / "x.dds", "BC7")
    with pytest.raises(ValueError, match="sRGB"):
        dds.write_dds(im, tmp_path / "x.dds", "BC5", srgb=True)
    assert not (tmp_path / "x.dds").exists()


@pytest.mark.parametrize("fmt, fourcc", [("BC1", b"DXT1"), ("BC3", b"DXT5")])
def test_legacy_header_fields(tmp_path, fmt, fourcc):
    side = 128
    p = dds.write_dds(Image.new("RGBA", (side, side), (10, 20, 30, 255)), tmp_path / "a.dds", fmt)
    b = p.read_bytes()
    assert u32(b, 4) == 124 and u32(b, 8) == DDSD_FLAGS
    assert u32(b, 12) == side and u32(b, 16) == side            # height, width
    assert u32(b, 20) == BLOCK_BYTES[fmt] * (side // 4) ** 2    # the top level's size in bytes
    assert u32(b, 28) == 8                                      # 128 .. 1
    assert u32(b, 76) == 32 and u32(b, 80) == 0x4 and b[84:88] == fourcc  # DDPF_FOURCC
    assert u32(b, 108) == CAPS
    assert len(b) == 128 + payload_len(fmt, side)


@pytest.mark.parametrize("fmt, srgb, dxgi", [("BC1", True, 72), ("BC3", True, 78), ("BC5", False, 83)])
def test_dx10_header_fields(tmp_path, fmt, srgb, dxgi):
    side = 64
    p = dds.write_dds(Image.new("RGB", (side, side), (10, 20, 30)), tmp_path / "a.dds", fmt, srgb=srgb)
    b = p.read_bytes()
    assert b[:4] == b"DDS " and u32(b, 8) == DDSD_FLAGS
    assert u32(b, 12) == side and u32(b, 16) == side
    assert u32(b, 20) == BLOCK_BYTES[fmt] * (side // 4) ** 2
    assert u32(b, 28) == 7
    assert u32(b, 80) == 0x4 and b[84:88] == b"DX10"
    assert u32(b, 108) == CAPS
    assert u32(b, 128) == dxgi and u32(b, 132) == 3             # D3D10_RESOURCE_DIMENSION_TEXTURE2D
    assert u32(b, 136) == 0 and u32(b, 140) == 1                # no misc flags, array size 1
    assert len(b) == 148 + payload_len(fmt, side)


@pytest.mark.parametrize("fmt", ["BC1", "BC3"])
def test_srgb_changes_the_header_only(tmp_path, fmt):
    """The same blocks follow either header; sRGB only tells the sampler how to read them."""
    im = Image.linear_gradient("L").resize((64, 64)).convert("RGB")
    plain = dds.write_dds(im, tmp_path / "plain.dds", fmt).read_bytes()
    srgb = dds.write_dds(im, tmp_path / "srgb.dds", fmt, srgb=True).read_bytes()
    assert plain[128:] == srgb[148:]


def test_every_mip_level_is_encoded_from_the_one_above(tmp_path):
    # left half black, right half white: levels 0-4 keep the split, the smallest ones average it
    im = Image.new("RGB", (32, 32), (255, 255, 255)); im.paste((0, 0, 0), (0, 0, 16, 32))
    b = dds.write_dds(im, tmp_path / "a.dds", "BC1").read_bytes()
    top = decode_level(b, "BC1", 32, 0, 128)
    assert top.getpixel((4, 16))[0] < 8 and top.getpixel((28, 16))[0] > 247
    half = decode_level(b, "BC1", 32, 1, 128)       # 16 x 16
    assert half.getpixel((2, 8))[0] < 8 and half.getpixel((14, 8))[0] > 247
    one = decode_level(b, "BC1", 32, 5, 128)        # 1 x 1: the whole image averaged
    assert 100 < one.getpixel((0, 0))[0] < 155


def test_bc3_mips_keep_alpha(tmp_path):
    im = Image.new("RGBA", (16, 16), (255, 255, 255, 255)); im.paste((255, 255, 255, 0), (0, 0, 8, 16))
    b = dds.write_dds(im, tmp_path / "a.dds", "BC3").read_bytes()
    quarter = decode_level(b, "BC3", 16, 2, 128)    # 4 x 4
    assert quarter.getpixel((0, 2))[3] < 16 and quarter.getpixel((3, 2))[3] > 240


def test_bc5_stores_red_and_green(tmp_path):
    b = dds.write_dds(Image.new("RGB", (16, 16), (60, 200, 255)), tmp_path / "n.dds", "BC5").read_bytes()
    px = decode_level(b, "BC5", 16, 0, 148).getpixel((5, 5))
    assert abs(px[0] - 60) <= 2 and abs(px[1] - 200) <= 2


@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA"])
@pytest.mark.parametrize("fmt", ["BC1", "BC3", "BC5"])
def test_accepts_any_common_mode(tmp_path, mode, fmt):
    p = dds.write_dds(Image.new(mode, (16, 16)), tmp_path / "a.dds", fmt)
    assert p.stat().st_size == 128 + payload_len(fmt, 16) + (20 if fmt == "BC5" else 0)


def test_bc1_ignores_alpha(tmp_path):
    """BC1's one-bit alpha would punch holes in the texture; an RGBA source is written opaque."""
    b = dds.write_dds(Image.new("RGBA", (16, 16), (90, 90, 90, 0)), tmp_path / "a.dds", "BC1").read_bytes()
    px = decode_level(b, "BC1", 16, 0, 128).getpixel((3, 3))
    assert px[3] == 255 and abs(px[0] - 90) <= 8


def test_returns_the_path_and_makes_the_folder(tmp_path):
    target = tmp_path / "art" / "textures" / "br" / "a.dds"
    p = dds.write_dds(Image.new("RGB", (4, 4)), target, "BC1")
    assert p == target and p.is_file()


def test_the_smallest_and_largest_sizes(tmp_path):
    small = dds.write_dds(Image.new("RGB", (4, 4)), tmp_path / "s.dds", "BC1")
    assert u32(small.read_bytes(), 28) == 3 and small.stat().st_size == 128 + 3 * 8
    big = dds.write_dds(Image.new("RGB", (4096, 4096), (9, 9, 9)), tmp_path / "b.dds", "BC1")
    assert u32(big.read_bytes(), 28) == 13 and big.stat().st_size == 128 + payload_len("BC1", 4096)


def test_pillow_opens_what_it_writes(tmp_path):
    im = Image.new("RGB", (64, 64), (200, 120, 40))
    with Image.open(dds.write_dds(im, tmp_path / "a.dds", "BC1")) as back:
        assert back.size == (64, 64)
        px = back.convert("RGB").getpixel((10, 10))
    assert all(abs(a - b) <= 8 for a, b in zip(px, (200, 120, 40)))


@pytest.mark.parametrize("fmt", ["BC1", "BC3", "BC5"])
@pytest.mark.parametrize("width, height", [(2048, 1024), (64, 128), (4, 256)])
def test_rectangular_texture(tmp_path, fmt, width, height):
    """A power-of-two rectangle (the 2:1 field atlas): width and height in the header, each mip
    level halves both sides down to 1 x 1, and the blocks follow in that order."""
    srgb = fmt != "BC5"
    p = dds.write_dds(Image.new("RGB", (width, height), (10, 20, 30)), tmp_path / "r.dds", fmt, srgb=srgb)
    b = p.read_bytes()
    dims = level_dims(width, height)
    assert u32(b, 12) == height and u32(b, 16) == width         # dwHeight, dwWidth
    assert u32(b, 20) == BLOCK_BYTES[fmt] * blocks(width, height)
    assert u32(b, 28) == len(dims) == max(width, height).bit_length()
    assert len(b) == 148 + BLOCK_BYTES[fmt] * sum(blocks(w, h) for w, h in dims)


def test_rectangular_texture_keeps_its_layout(tmp_path):
    """Left half black, right half white on a 2:1 image: it opens with the same size and the split in place."""
    im = Image.new("RGB", (128, 64), (255, 255, 255)); im.paste((0, 0, 0), (0, 0, 64, 64))
    with Image.open(dds.write_dds(im, tmp_path / "r.dds", "BC1")) as back:
        assert back.size == (128, 64)
        rgb = back.convert("RGB")
        assert rgb.getpixel((16, 32))[0] < 8 and rgb.getpixel((112, 32))[0] > 247


# ------------------------------------------------------------------ coverage-preserving mips

REF = 67 / 255          # the moss card's alphaRef, as a share of full opacity


def strands(side=256, share=0.2, seed=5):
    """An opacity image of 1-2 px wide vertical strands (255 on 0) that cover about `share` of its texels."""
    rng = np.random.default_rng(seed)
    a = np.zeros((side, side), np.uint8)
    while (a > 0).mean() < share:
        x, w = int(rng.integers(0, side - 1)), int(rng.integers(1, 3))
        y0 = int(rng.integers(0, side // 2))
        a[y0:y0 + int(rng.integers(side // 4, side // 2 + 1)), x:x + w] = 255
    return a


def opacity_image(a, fmt):
    """The strands as the image the textures stage hands write_dds: a grey `o` map for BC1, the alpha of a
    white RGBA colour map for BC3."""
    grey = Image.fromarray(a)
    return grey if fmt == "BC1" else Image.merge("RGBA", [Image.new("L", grey.size, 255)] * 3 + [grey])


def level_coverages(b, fmt, width, height, threshold=67):
    """The share of texels at or above the threshold in each mip level of the file bytes `b`, largest first:
    BC3's alpha, or BC1's red (the grey `o` data map), as the blocks decode."""
    start = 148 if b[84:88] == b"DX10" else 128
    shares = []
    for w, h in level_dims(width, height):
        size = BLOCK_BYTES[fmt] * blocks(w, h)
        px = np.asarray(Image.frombytes("RGBA", (w, h), b[start:start + size], "bcn", PIXEL_FORMAT[fmt]))
        shares.append(float((px[..., 3 if fmt == "BC3" else 0] >= threshold).mean()))
        start += size
    return shares


@pytest.mark.parametrize("fmt", ["BC1", "BC3"])
def test_coverage_preserving_mips_keep_sparse_strands(tmp_path, fmt):
    a = strands()
    base = float((a >= 67).mean())
    assert 0.18 < base < 0.23
    im = opacity_image(a, fmt)
    kept = dds.write_dds(im, tmp_path / "kept.dds", fmt, alpha_coverage_ref=REF).read_bytes()
    plain = dds.write_dds(im, tmp_path / "plain.dds", fmt).read_bytes()
    for level, share in enumerate(level_coverages(kept, fmt, 256, 256)[:6]):        # 256 down to 8
        assert abs(share - base) <= 0.05, level
    # without it the averaged strands drift past the threshold: the test bites
    assert abs(level_coverages(plain, fmt, 256, 256)[4] - base) > 0.1               # the 16 x 16 level


@pytest.mark.parametrize("fmt", ["BC1", "BC3"])
def test_coverage_ref_leaves_the_top_level_alone(tmp_path, fmt):
    im = opacity_image(strands(64), fmt)
    kept = dds.write_dds(im, tmp_path / "kept.dds", fmt, alpha_coverage_ref=REF).read_bytes()
    plain = dds.write_dds(im, tmp_path / "plain.dds", fmt).read_bytes()
    top = BLOCK_BYTES[fmt] * (64 // 4) ** 2
    assert kept[:128 + top] == plain[:128 + top] and kept != plain


def test_coverage_ref_scales_the_alpha_only(tmp_path):
    """BC3 keeps the colour of every mip level: only each block's alpha half changes."""
    im = opacity_image(strands(64), "BC3")
    kept = dds.write_dds(im, tmp_path / "kept.dds", "BC3", alpha_coverage_ref=REF).read_bytes()
    plain = dds.write_dds(im, tmp_path / "plain.dds", "BC3").read_bytes()
    assert len(kept) == len(plain)
    colour = lambda b: [b[i + 8:i + 16] for i in range(128, len(b), 16)]
    assert colour(kept) == colour(plain)


@pytest.mark.parametrize("fmt", ["BC1", "BC3"])
def test_coverage_ref_keeps_a_solid_texture_as_it_is(tmp_path, fmt):
    """Nothing to preserve: a fully opaque (or fully clear) map is written byte for byte as without."""
    for value in (255, 0):
        im = opacity_image(np.full((64, 64), value, np.uint8), fmt)
        kept = dds.write_dds(im, tmp_path / "kept.dds", fmt, alpha_coverage_ref=REF).read_bytes()
        assert kept == dds.write_dds(im, tmp_path / "plain.dds", fmt).read_bytes()


def test_coverage_ref_none_changes_nothing(tmp_path):
    im = opacity_image(strands(64), "BC3")
    assert (dds.write_dds(im, tmp_path / "a.dds", "BC3", alpha_coverage_ref=None).read_bytes()
            == dds.write_dds(im, tmp_path / "b.dds", "BC3").read_bytes())


def test_coverage_ref_works_on_a_rectangle(tmp_path):
    a = np.concatenate([strands(128, seed=2), strands(128, seed=3)], axis=1)       # 256 wide, 128 high
    base = float((a >= 67).mean())
    b = dds.write_dds(Image.fromarray(a), tmp_path / "r.dds", "BC1", alpha_coverage_ref=REF).read_bytes()
    for level, share in enumerate(level_coverages(b, "BC1", 256, 128)[:6]):         # 256 x 128 down to 8 x 4
        assert abs(share - base) <= 0.05, level


@pytest.mark.parametrize("ref", [0, -0.1, 1.5, 67])
def test_coverage_ref_must_be_a_share_of_full_opacity(tmp_path, ref):
    with pytest.raises(ValueError, match="alpha_coverage_ref"):
        dds.write_dds(Image.new("RGBA", (16, 16)), tmp_path / "x.dds", "BC3", alpha_coverage_ref=ref)
    assert not (tmp_path / "x.dds").exists()


def test_coverage_ref_is_refused_for_a_normal_map(tmp_path):
    with pytest.raises(ValueError, match="BC5"):
        dds.write_dds(Image.new("RGB", (16, 16)), tmp_path / "x.dds", "BC5", alpha_coverage_ref=REF)
    assert not (tmp_path / "x.dds").exists()


def share_at(values, threshold=67):
    """share(factor): the share of `values` at or above the threshold once scaled, clipped and rounded to 8 bits."""
    values = np.asarray(values, float)
    return lambda factor: float((np.rint(np.minimum(values * factor, 255)) >= threshold).mean())


def test_coverage_scale_finds_a_factor_with_the_wanted_share():
    share = share_at([20, 40, 60, 80, 100, 120, 140, 160])
    assert share(1.0) == 5 / 8
    assert dds.coverage_scale(share, 5 / 8) == 1.0              # already the wanted share: unchanged
    down = dds.coverage_scale(share, 3 / 8)
    assert share(down) == 3 / 8 and 0.55 < down < 0.67
    up = dds.coverage_scale(share, 1.0)
    assert share(up) == 1.0 and 3.3 < up <= 4.0


def test_coverage_scale_overshoots_when_no_factor_gives_the_share():
    share = share_at([10, 100, 100, 100])                       # the three 100s pass together or not at all
    s = dds.coverage_scale(share, 0.5)
    assert share(s) == 0.75 and 0.665 <= s < 0.67               # thin strands may gain, never vanish


def test_coverage_scale_is_clamped_to_zero_and_four():
    assert dds.coverage_scale(share_at([5, 10, 15, 16]), 1.0) == 4.0        # would need 13x
    share = share_at([200, 255])
    s = dds.coverage_scale(share, 0.0)                          # nothing may pass: scaled right down
    assert share(s) == 0.0 and 0.0 <= s < 0.34
