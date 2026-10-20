"""DDS files for BeamNG.drive: BC1, BC3 and BC5 textures with a full mip chain.

A texture is a power-of-two rectangle, square or not; each mip level halves both sides until
they reach 1 x 1 (a side that is already 1 stays 1).

Pillow 12.3 encodes the blocks (its "bcn" encoder; BC4 comes out wrong and BC7 doesn't exist,
so neither is offered). Each mip level is encoded on its own, and the levels follow one header:

* BC1 and BC3 in linear colour get the legacy header (fourCC DXT1 / DXT5).
* BC5, and anything sRGB, get the DX10 extension header, because only that header can name
  BC1_UNORM_SRGB (72) and BC3_UNORM_SRGB (78) or BC5_UNORM (83).
* Both forms set DDSD_MIPMAPCOUNT and DDSCAPS_MIPMAP | DDSCAPS_COMPLEX.

The sRGB flag only changes the header. The blocks are the same either way, so the game decides
whether the sampler converts them to linear light.

An alpha-tested map (a material with alphaTest and an alphaRef) can ask for coverage-preserving
mips with `alpha_coverage_ref`. Plain halvings average thin strands into grey: with a low alphaRef the
averaged opacity rises past the threshold over whole texels (a solid card at a distance), and with a
high one it falls short everywhere (the strands vanish). So each mip level's opacity is scaled by the
factor (coverage_scale) that puts the same share of its texels at or above the threshold as the top
level has, measured on the blocks as they decode.
"""
from __future__ import annotations

import struct
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image

MIN_SIDE, MAX_SIDE = 4, 4096
MAX_COVERAGE_SCALE = 4.0        # coverage_scale never brightens opacity more than this
COVERAGE_STEPS = 24             # its bisection: the factor is found to within 4 / 2**24

MAGIC = b"DDS "
HEADER_SIZE = 124
DDSD_CAPS, DDSD_HEIGHT, DDSD_WIDTH, DDSD_PIXELFORMAT = 0x1, 0x2, 0x4, 0x1000
DDSD_MIPMAPCOUNT, DDSD_LINEARSIZE = 0x20000, 0x80000
DDPF_FOURCC = 0x4
DDSCAPS_COMPLEX, DDSCAPS_TEXTURE, DDSCAPS_MIPMAP = 0x8, 0x1000, 0x400000
D3D10_RESOURCE_DIMENSION_TEXTURE2D = 3


class _Format(NamedTuple):
    bcn: int  # the encoder argument: 1 for BC1, 3 for BC3, 5 for BC5
    mode: str  # the Pillow mode the encoder is fed
    block_bytes: int  # one 4 x 4 block
    fourcc: bytes | None  # the legacy header's fourCC; None when only the DX10 header can say it
    dxgi: int  # DXGI_FORMAT_*_UNORM
    dxgi_srgb: int | None  # DXGI_FORMAT_*_UNORM_SRGB; None when there is no such format


FORMATS = {
    "BC1": _Format(1, "RGB", 8, b"DXT1", 71, 72),  # opaque colour: an RGBA source loses its alpha
    "BC3": _Format(3, "RGBA", 16, b"DXT5", 77, 78),
    "BC5": _Format(5, "RGB", 16, None, 83, None),  # two channels (red, green): normal maps
}


def _mip_chain(img: Image.Image) -> list[Image.Image]:
    """The image and each LANCZOS halving of the one above it (a side stops at 1), down to 1 x 1."""
    levels = [img]
    while levels[-1].size != (1, 1):
        size = tuple(max(1, n // 2) for n in levels[-1].size)
        levels.append(levels[-1].resize(size, Image.Resampling.LANCZOS))
    return levels


def _scaled(values: np.ndarray, factor: float) -> np.ndarray:
    """values times factor, clipped to 255 and rounded: the 8-bit opacity the scaled level stores."""
    return np.rint(np.minimum(values * factor, 255.0))


def coverage_scale(share_at: Callable[[float], float], share: float) -> float:
    """The factor in [0, MAX_COVERAGE_SCALE] to scale a mip level's opacity by, given share_at(factor), the
    share of its texels at or above the alpha threshold once scaled, so that the share is `share`.

    A bisection that starts at 1, so a level whose share is already right comes back unscaled (an opaque
    map stays opaque). A share is a step function of the factor: when no factor gives exactly `share`,
    the smallest factor tried that gives more is returned, so a thin strand grows rather than vanishes;
    MAX_COVERAGE_SCALE if none does."""
    low, high, factor = 0.0, MAX_COVERAGE_SCALE, 1.0
    for _ in range(COVERAGE_STEPS):
        got = share_at(factor)
        if got == share:
            return factor
        if got < share:
            low = factor
        else:
            high = factor
        factor = (low + high) / 2
    return high


def _preserve_coverage(levels: list[Image.Image], f: _Format, ref: float) -> list[Image.Image]:
    """levels with the opacity of every one below the first scaled to the first's share of texels at or
    above ref (a share of full opacity).

    The opacity is BC3's alpha channel (the rest is left alone) or, for BC1, the red channel of a grey
    data map, where all three channels are scaled alike so it stays grey. The shares are those of the
    levels as the sampler reads them, after block compression: BC1's four-colour blocks snap values
    that sit between a strand and its gap, which moves a level's share by several percent.
    """
    alpha = f.mode == "RGBA"
    threshold = max(1, round(ref * 255))

    def decoded_share(px: np.ndarray) -> float:
        img = Image.fromarray(px)
        data = img.tobytes("bcn", (f.bcn,))
        out = np.asarray(Image.frombytes("RGBA", img.size, data, "bcn", (f.bcn, f"BC{f.bcn}")))
        return float(np.mean(out[..., 3 if alpha else 0] >= threshold))

    def scaled(px: np.ndarray, factor: float) -> np.ndarray:
        out = px.copy()
        if alpha:
            out[..., 3] = _scaled(px[..., 3], factor)
        else:
            out[:] = _scaled(px, factor)
        return out

    share = decoded_share(np.asarray(levels[0]))
    out = [levels[0]]
    for level in levels[1:]:
        px = np.asarray(level)
        factor = coverage_scale(lambda k: decoded_share(scaled(px, k)), share)
        out.append(Image.fromarray(scaled(px, factor)))
    return out


def _header(width: int, height: int, mips: int, f: _Format, srgb: bool) -> bytes:
    top_level_bytes = f.block_bytes * (width // 4) * (height // 4)
    dx10 = srgb or f.fourcc is None
    flags = (DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT | DDSD_MIPMAPCOUNT
             | DDSD_LINEARSIZE)
    head = (MAGIC
            + struct.pack("<7I", HEADER_SIZE, flags, height, width, top_level_bytes, 0, mips)
            + bytes(44)  # dwReserved1[11]
            # DDS_PIXELFORMAT: size, flags, fourCC, RGB bit count, four colour masks
            + struct.pack("<II4s5I", 32, DDPF_FOURCC, b"DX10" if dx10 else f.fourcc, 0, 0, 0, 0, 0)
            + struct.pack("<5I", DDSCAPS_TEXTURE | DDSCAPS_MIPMAP | DDSCAPS_COMPLEX, 0, 0, 0, 0))
    if dx10:
        dxgi = f.dxgi_srgb if srgb else f.dxgi
        # DXGI format, 2-D texture, no misc flags, array size 1, alpha mode unknown
        head += struct.pack("<5I", dxgi, D3D10_RESOURCE_DIMENSION_TEXTURE2D, 0, 1, 0)
    return head


def write_dds(img: Image.Image, path, fmt: str, srgb: bool = False,
              alpha_coverage_ref: float | None = None) -> Path:
    """Write img as a DDS with a full mip chain and return the path.

    img's sides are each a power of two from 4 to 4096 (a rectangle such as the 2:1 field atlas
    is fine). fmt is "BC1" (RGB, 8 bytes per block), "BC3" (RGBA) or "BC5" (red and green).
    srgb=True marks BC1 and BC3 colour as sRGB.

    alpha_coverage_ref, for an alpha-tested map, is the material's alphaRef as a share of full
    opacity (67 / 255 for alphaRef 67): the opacity of each mip level is scaled so that the share of
    its texels at or above it matches the top level's (see the module docstring). The opacity is
    BC3's alpha channel, or the red (grey) channel of a BC1 data map such as `o`; BC5 has none.
    """
    f = FORMATS.get(fmt)
    if f is None:
        raise ValueError(f"unsupported DDS format {fmt!r}: use one of {', '.join(FORMATS)}")
    if srgb and f.dxgi_srgb is None:
        raise ValueError(f"{fmt} has no sRGB variant")
    if any(not MIN_SIDE <= n <= MAX_SIDE or n & (n - 1) for n in img.size):
        raise ValueError(f"each side of a DDS texture must be a power of two from {MIN_SIDE} to {MAX_SIDE} "
                         f"pixels; got {img.width} x {img.height}")
    if alpha_coverage_ref is not None:
        if fmt == "BC5":
            raise ValueError("alpha_coverage_ref needs an opacity: BC3's alpha or a BC1 data map's red; "
                             "BC5 holds a normal map")
        if not 0 < alpha_coverage_ref <= 1:
            raise ValueError(f"alpha_coverage_ref is the alphaRef as a share of full opacity, above 0 and at "
                             f"most 1 (67 / 255 for alphaRef 67); got {alpha_coverage_ref}")
    levels = _mip_chain(img.convert(f.mode))
    if alpha_coverage_ref is not None:
        levels = _preserve_coverage(levels, f, alpha_coverage_ref)
    data = b"".join(level.tobytes("bcn", (f.bcn,)) for level in levels)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_header(img.width, img.height, len(levels), f, srgb) + data)
    return path
