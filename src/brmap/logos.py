"""Original logos and field graphics, drawn from polygons and a polyline table.

IP rule (spec): no real logo, wordmark or trademark, and clear of LSU's marks: no side-profile
roaring tiger head, no single "tiger eye", no italic or interlocking letters. So the letters are
upright, generic block capitals and numerals from GLYPHS (no font file is ever opened); the tiger
is a front-facing, left-right symmetric head of flat polygons with two eyes; purple #461D7C and
gold #FDD023 are the only team colours, and a colour is not a mark.

* GLYPHS: {char: [polyline, ...]}, each polyline a list of (x, y) on a 4 x 6 grid (y runs down).
  The strokes are STROKE wide, mitred at the bends and square at the ends. The grid is the
  glyph's ink box, so a glyph is exactly 4 : 6, the proportions of a 4 ft by 6 ft field numeral.
* text(s, height_px, fill, outline=None, stroke=None) -> RGBA: the string in GLYPHS capitals,
  digits and spaces; its ink is height_px tall. An outline adds a margin of `stroke` pixels
  (default OUTLINE_SHARE of the height) on every side.
* tiger_mark(size), red_stick(size) -> RGBA squares on a clear ground.
* field_atlas() -> 2048 x 1024 RGBA: the midfield mark, two "BATON ROUGE" end-zone panels,
  numerals 0 to 5 and four direction arrows, laid out as FIELD_REGIONS (pixel boxes, y down,
  GUTTER apart so the mip levels do not bleed). Clear pixels are alpha 0 over the TURF colour, so
  a material that ignores alpha shows turf, not black. atlas_uv(name) gives a region as a Blender
  UV box (u0, v0, u1, v1), v up.
* board() -> 1024 x 512 opaque RGBA: the scoreboard face, laid out as BOARD_REGIONS (board_uv).
* review_sheet() -> RGB: everything above on one sheet, for docs/screenshots/v02/logos.png
  (python -B -m brmap.logos [path]).

Every shape is drawn SUPERSAMPLE times larger and reduced with LANCZOS; the same call gives the
same bytes every time.
"""
from __future__ import annotations

import functools
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageOps
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

SUPERSAMPLE = 4

PURPLE = (0x46, 0x1D, 0x7C)
GOLD = (0xFD, 0xD0, 0x23)
WHITE = (255, 255, 255)
TURF = (52, 96, 44)            # the RGB under clear pixels of the atlas

# --- the glyphs ----------------------------------------------------------------------------

GLYPH_W, GLYPH_H = 4, 6        # the grid, and the ink box of every glyph
STROKE = 0.9                   # stroke width, in grid units
LETTER_GAP = 0.8               # between the ink boxes of neighbouring glyphs
SPACE_ADVANCE = 2.4            # a space moves the pen this far (the gap after the last glyph included)
MITRE_LIMIT = 2.5              # sharper bends are bevelled
OUTLINE_SHARE = 0.08           # the default outline width, as a share of the text height

GLYPHS: dict[str, list[list[tuple[float, float]]]] = {
    "A": [[(0, 6), (0, 1), (1, 0), (3, 0), (4, 1), (4, 6)], [(0, 3.5), (4, 3.5)]],
    "B": [[(0, 0), (0, 6)], [(0, 0), (3, 0), (4, 1), (4, 2), (3, 3), (0, 3)],
          [(3, 3), (4, 4), (4, 5), (3, 6), (0, 6)]],
    "C": [[(4, 0), (1, 0), (0, 1), (0, 5), (1, 6), (4, 6)]],
    "D": [[(0, 0), (0, 6)], [(0, 0), (2.5, 0), (4, 1.5), (4, 4.5), (2.5, 6), (0, 6)]],
    "E": [[(4, 0), (0, 0), (0, 6), (4, 6)], [(0, 3), (3, 3)]],
    "G": [[(4, 0), (1, 0), (0, 1), (0, 5), (1, 6), (4, 6), (4, 3), (2, 3)]],
    "I": [[(2, 0), (2, 6)], [(0.5, 0), (3.5, 0)], [(0.5, 6), (3.5, 6)]],
    "K": [[(0, 0), (0, 6)], [(4, 0), (0, 3), (4, 6)]],
    "N": [[(0, 6), (0, 0)], [(0, 0), (4, 6)], [(4, 6), (4, 0)]],
    "O": [[(1, 0), (3, 0), (4, 1), (4, 5), (3, 6), (1, 6), (0, 5), (0, 1), (1, 0)]],
    "R": [[(0, 6), (0, 0)], [(0, 0), (3, 0), (4, 1), (4, 2), (3, 3), (0, 3)], [(2.2, 3.3), (4, 6)]],
    "S": [[(4, 0), (1, 0), (0, 1), (0, 2), (1, 3), (3, 3), (4, 4), (4, 5), (3, 6), (0, 6)]],
    "T": [[(0, 0), (4, 0)], [(2, 0), (2, 6)]],
    "U": [[(0, 0), (0, 5), (1, 6), (3, 6), (4, 5), (4, 0)]],
    "0": [[(1.2, 0), (2.8, 0), (4, 1.2), (4, 4.8), (2.8, 6), (1.2, 6), (0, 4.8), (0, 1.2), (1.2, 0)]],
    "1": [[(0.5, 1.5), (2, 0), (2, 6)], [(0.5, 6), (3.5, 6)]],
    "2": [[(0, 1), (1, 0), (3, 0), (4, 1), (4, 2.2), (0, 6)], [(0, 6), (4, 6)]],
    "3": [[(0, 0), (3, 0), (4, 1), (4, 2), (3, 3), (1.5, 3)], [(3, 3), (4, 4), (4, 5), (3, 6), (0, 6)]],
    "4": [[(2.7, 0.4), (0, 4), (4, 4)], [(3, 0), (3, 6)]],
    "5": [[(4, 0), (0, 0), (0, 3), (3, 3), (4, 4), (4, 5), (3, 6), (0, 6)]],
    "6": [[(4, 0), (1, 0), (0, 1), (0, 5), (1, 6), (3, 6), (4, 5), (4, 4), (3, 3), (0, 3)]],
    "7": [[(0, 0), (4, 0), (1.5, 6)]],
    "8": [[(1, 0), (3, 0), (4, 1), (4, 2), (3, 3), (1, 3), (0, 2), (0, 1), (1, 0)],
          [(1, 3), (3, 3), (4, 4), (4, 5), (3, 6), (1, 6), (0, 5), (0, 4), (1, 3)]],
    "9": [[(0, 6), (3, 6), (4, 5), (4, 1), (3, 0), (1, 0), (0, 1), (0, 2), (1, 3), (4, 3)]],
}


@functools.lru_cache(maxsize=None)
def _ink(ch: str):
    """One glyph's ink as a shapely geometry in grid units: its strokes, cut flush to the 4 x 6 box."""
    half = STROKE / 2
    kx, ky = (GLYPH_W - STROKE) / GLYPH_W, (GLYPH_H - STROKE) / GLYPH_H    # centre lines sit inside the box
    strokes = [LineString([(half + x * kx, half + y * ky) for x, y in line])
               .buffer(half, cap_style="square", join_style="mitre", mitre_limit=MITRE_LIMIT)
               for line in GLYPHS[ch]]
    return unary_union(strokes).intersection(box(0, 0, GLYPH_W, GLYPH_H))


def _polygons(geom):
    return [g for g in getattr(geom, "geoms", [geom]) if g.geom_type == "Polygon" and not g.is_empty]


def _fill(draw: ImageDraw.ImageDraw, geom, k: float, value, dx: float = 0.0, dy: float = 0.0, hole=None):
    """Draw a shapely geometry scaled by k and moved by (dx, dy); `hole` paints its holes, if given."""
    for poly in _polygons(geom):
        draw.polygon([(x * k + dx, y * k + dy) for x, y in poly.exterior.coords], fill=value)
        if hole is not None:
            for ring in poly.interiors:
                draw.polygon([(x * k + dx, y * k + dy) for x, y in ring.coords], fill=hole)


def _rgba(colour) -> tuple[int, int, int, int]:
    return ImageColor.getcolor(colour, "RGBA") if isinstance(colour, str) else \
        tuple(colour) + (255,) * (4 - len(colour))


def _layer(mask: Image.Image, colour) -> Image.Image:
    """A flat colour whose alpha is the mask (straight alpha: the colour is exact at every edge)."""
    r, g, b, a = _rgba(colour)
    layer = Image.new("RGBA", mask.size, (r, g, b, 0))
    layer.putalpha(mask if a == 255 else mask.point(lambda v: v * a // 255))
    return layer


def _reduce(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    return img.resize(size, Image.Resampling.LANCZOS)


def _layout(s: str) -> tuple[list[tuple[str, float]], float]:
    """[(glyph, x)] in grid units and the width of the ink."""
    x, placed, right = 0.0, [], 0.0
    for ch in s:
        if ch == " ":
            x += SPACE_ADVANCE
            continue
        if ch not in GLYPHS:
            raise ValueError(f"no glyph for {ch!r}; GLYPHS has {''.join(sorted(GLYPHS))}")
        placed.append((ch, x))
        right = x + GLYPH_W
        x += GLYPH_W + LETTER_GAP
    if not placed:
        raise ValueError(f"nothing to draw in {s!r}")
    return placed, right


def text(s: str, height_px: int, fill, outline=None, stroke: int | None = None) -> Image.Image:
    """The string as RGBA on a clear ground: GLYPHS capitals and digits, spaces between words.

    The ink is exactly height_px tall; the width follows from the glyphs, so doubling the height
    doubles the width. `outline` (a colour) surrounds the letters with a band `stroke` pixels wide
    (default round(OUTLINE_SHARE * height_px)); the image then grows by that on every side.
    """
    placed, right = _layout(s)
    scale = height_px / GLYPH_H
    margin = 0 if outline is None else (round(height_px * OUTLINE_SHARE) if stroke is None else int(stroke))
    width = max(1, round(right * scale))
    size = (width + 2 * margin, height_px + 2 * margin)
    k = scale * SUPERSAMPLE
    big = (size[0] * SUPERSAMPLE, size[1] * SUPERSAMPLE)

    def mask(grow: float = 0.0) -> Image.Image:
        m = Image.new("L", big, 0)
        draw = ImageDraw.Draw(m)
        for ch, x in placed:
            ink = affinity.translate(_ink(ch), x)
            if grow:
                ink = ink.buffer(grow, join_style="round", quad_segs=8)
            _fill(draw, ink, k, 255, margin * SUPERSAMPLE, margin * SUPERSAMPLE, hole=0)
        return _reduce(m, size)

    letters = _layer(mask(), fill)
    if outline is None:
        return letters
    return Image.alpha_composite(_layer(mask(margin / scale), outline), letters)


# --- flat shapes -----------------------------------------------------------------------------

def _render(size: int, layers, half: bool = False) -> Image.Image:
    """Paint (colour, geometry) layers, geometry in unit-square coordinates (y down), in order.

    half=True keeps only the left half of the square and mirrors it onto the right, so the
    result is left-right symmetric to the pixel; the geometry then only has to be right on x <= 0.5.
    """
    big = size * SUPERSAMPLE
    left_only = box(-1, -1, 0.5, 2)
    canvas = Image.new("RGBA", (big // 2 if half else big, big), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    for colour, geom in layers:
        _fill(draw, geom.intersection(left_only) if half else geom, big, _rgba(colour))
    if half:
        whole = Image.new("RGBA", (big, big))
        whole.paste(canvas, (0, 0))
        whole.paste(ImageOps.mirror(canvas), (big // 2, 0))
        canvas = whole
    return _reduce(canvas, (size, size))


def _mirrored(chain: list[tuple[float, float]]) -> Polygon:
    """The polygon whose left half is chain (top centre down to bottom centre) and whose right half mirrors it."""
    return Polygon(chain + [(1 - x, y) for x, y in reversed(chain[1:-1])])


def _bar(p0, p1, width: float):
    return LineString([p0, p1]).buffer(width / 2, cap_style="flat")


def tiger_mark(size: int = 512) -> Image.Image:
    """A front-facing tiger head of flat polygons, symmetric about the vertical axis, with two eyes."""
    head = _mirrored([
        (0.50, 0.15), (0.39, 0.17), (0.28, 0.24), (0.20, 0.33), (0.12, 0.40), (0.06, 0.47), (0.14, 0.50),
        (0.05, 0.60), (0.15, 0.62), (0.09, 0.73), (0.20, 0.72), (0.18, 0.82), (0.30, 0.78), (0.36, 0.90),
        (0.50, 0.95)])
    ear = Polygon([(0.11, 0.17), (0.15, 0.085), (0.25, 0.055), (0.35, 0.10), (0.41, 0.18), (0.20, 0.33)])
    ear_in = Polygon([(0.175, 0.17), (0.20, 0.115), (0.25, 0.095), (0.31, 0.125), (0.345, 0.18), (0.215, 0.27)])
    both = unary_union([head, ear, affinity.scale(ear, -1, 1, origin=(0.5, 0))])
    muzzle = _mirrored([(0.50, 0.54), (0.41, 0.55), (0.33, 0.60), (0.31, 0.68), (0.37, 0.77), (0.50, 0.80)])
    nose = _mirrored([(0.50, 0.57), (0.445, 0.575), (0.47, 0.63), (0.50, 0.655)])
    forehead = _mirrored([(0.50, 0.17), (0.475, 0.22), (0.485, 0.30), (0.50, 0.35)])
    eye_socket = Polygon([(0.205, 0.425), (0.445, 0.46), (0.435, 0.545), (0.25, 0.54)])
    eye_white = Polygon([(0.245, 0.448), (0.43, 0.474), (0.42, 0.53), (0.275, 0.525)])
    iris = Point(0.35, 0.497).buffer(0.026, quad_segs=2)
    pupil = Polygon([(0.35, 0.472), (0.358, 0.497), (0.35, 0.522), (0.342, 0.497)])
    stripes = [
        Polygon([(0.43, 0.20), (0.455, 0.22), (0.445, 0.32)]),                      # forehead
        Polygon([(0.35, 0.22), (0.395, 0.235), (0.40, 0.36)]),
        Polygon([(0.27, 0.30), (0.32, 0.29), (0.355, 0.40)]),
        Polygon([(0.20, 0.40), (0.44, 0.44), (0.44, 0.457), (0.20, 0.425)]),        # brow
        Polygon([(0.07, 0.49), (0.24, 0.50), (0.12, 0.54)]),                        # cheeks
        Polygon([(0.09, 0.61), (0.25, 0.58), (0.15, 0.65)]),
        Polygon([(0.14, 0.72), (0.27, 0.67), (0.20, 0.77)]),
    ]
    mouth = unary_union([_bar((0.50, 0.65), (0.50, 0.70), 0.016), _bar((0.50, 0.70), (0.45, 0.75), 0.016),
                         _bar((0.45, 0.75), (0.39, 0.735), 0.016)])
    whiskers = [Polygon([(x, y), (x + 0.014, y), (x + 0.014, y + 0.014), (x, y + 0.014)])
                for x, y in ((0.375, 0.635), (0.345, 0.665), (0.405, 0.665))]
    layers = [
        (WHITE, both.buffer(0.026, join_style="mitre")),
        (PURPLE, both.buffer(0.014, join_style="mitre")),
        (PURPLE, ear.buffer(0.010, join_style="mitre")),
        (GOLD, ear),
        (WHITE, ear_in),
        (PURPLE, head.buffer(0.010, join_style="mitre")),
        (GOLD, head),
        *[(PURPLE, s) for s in stripes],
        (PURPLE, forehead),
        (PURPLE, eye_socket), (WHITE, eye_white), (GOLD, iris), (PURPLE, pupil),
        (PURPLE, muzzle.buffer(0.008, join_style="mitre")), (WHITE, muzzle),
        (PURPLE, nose), (PURPLE, mouth), *[(PURPLE, w) for w in whiskers],
    ]
    return _render(size, layers, half=True)


RED = (0xC4, 0x24, 0x24)
DARK_RED = (0x8C, 0x1A, 0x1E)
NAVY = (0x1B, 0x3A, 0x5C)
SKY = (0xF1, 0xE7, 0xCE)
WAVES = ((0x6F, 0xA8, 0xD8), (0x4A, 0x88, 0xBE), (0x2D, 0x68, 0x9F))


def _wave(y: float, amplitude: float, phase: float, period: float = 0.22):
    """A band whose top edge is a sine wave about height y and which runs down past the roundel."""
    xs = np.linspace(0.0, 1.0, 81)
    top = [(float(x), y + amplitude * math.sin(2 * math.pi * (x - phase) / period)) for x in xs]
    return Polygon(top + [(1.0, 1.2), (0.0, 1.2)])


def red_stick(size: int = 512) -> Image.Image:
    """A roundel: a red pole with three notched pennants, standing in three waves of the river, in a ring."""
    disc = Point(0.5, 0.5).buffer(0.41, quad_segs=32)
    pole = box(0.4825, 0.17, 0.5175, 0.80)
    pennants = [Polygon([(0.5175, y), (0.5175 + length, y), (0.5175 + length - 0.045, y + 0.035),
                         (0.5175 + length, y + 0.07), (0.5175, y + 0.07)])
                for y, length in ((0.235, 0.20), (0.335, 0.17), (0.435, 0.14))]
    layers = [
        (NAVY, Point(0.5, 0.5).buffer(0.49, quad_segs=32)),
        (SKY, disc),
        (WAVES[0], _wave(0.66, 0.018, 0.0).intersection(disc)),
        (DARK_RED, pole),
        (RED, unary_union(pennants)),
        (GOLD, Point(0.5, 0.15).buffer(0.026, quad_segs=4)),
        (WAVES[1], _wave(0.75, 0.020, 0.07).intersection(disc)),
        (WAVES[2], _wave(0.84, 0.022, 0.14).intersection(disc)),
    ]
    return _render(size, layers)


# --- the field atlas -------------------------------------------------------------------------

ATLAS_SIZE = (2048, 1024)
GUTTER = 16                    # clear pixels kept between two regions

_NUMERAL_W, _NUMERAL_H = 320, 480       # 4 ft by 6 ft
_ARROW = 160
FIELD_REGIONS: dict[str, tuple[int, int, int, int]] = {
    # an end zone is 160 ft wide and 30 ft deep
    "end_zone_a": (32, 16, 992, 196),
    "end_zone_b": (32, 212, 992, 392),
    "tiger": (32, 408, 632, 1008),
    "arrow_up": (664, 408, 664 + _ARROW, 408 + _ARROW),
    "arrow_down": (840, 408, 840 + _ARROW, 408 + _ARROW),
    "arrow_left": (664, 584, 664 + _ARROW, 584 + _ARROW),
    "arrow_right": (840, 584, 840 + _ARROW, 584 + _ARROW),
    **{f"numeral_{n}": (x, y, x + _NUMERAL_W, y + _NUMERAL_H)
       for n, (x, y) in enumerate([(1040, 16), (1376, 16), (1712, 16), (1040, 512), (1376, 512), (1712, 512)])},
}


def _uv(box_px, size) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box_px
    return x0 / size[0], 1 - y1 / size[1], x1 / size[0], 1 - y0 / size[1]


def atlas_uv(name: str) -> tuple[float, float, float, float]:
    """A field-atlas region as a Blender UV box (u0, v0, u1, v1); v runs up, so the image top is v = 1."""
    return _uv(FIELD_REGIONS[name], ATLAS_SIZE)


def _shape(size: int, polygons, colour) -> Image.Image:
    """Flat polygons (unit-square coordinates) in one colour, on a clear square."""
    return _render(size, [(colour, Polygon(p)) for p in polygons])


def _arrow(direction: str, size: int) -> Image.Image:
    a, b = 0.12, 0.88
    tri = {"up": [(0.5, a), (a, b), (b, b)], "down": [(0.5, b), (a, a), (b, a)],
           "left": [(a, 0.5), (b, a), (b, b)], "right": [(b, 0.5), (a, a), (a, b)]}[direction]
    return _shape(size, [tri], WHITE)


def _end_zone(width: int, height: int) -> Image.Image:
    """A purple panel with gold rules and BATON ROUGE across it in gold."""
    panel = Image.new("RGBA", (width, height), PURPLE + (255,))
    draw = ImageDraw.Draw(panel)
    rule, inset = max(2, height // 30), height // 12
    for y in (inset, height - inset - rule):
        draw.rectangle([inset, y, width - inset - 1, y + rule - 1], fill=GOLD)
    lettering = text("BATON ROUGE", round(height * 0.5), GOLD, outline=WHITE, stroke=max(2, height // 40))
    panel.alpha_composite(lettering, ((width - lettering.width) // 2, (height - lettering.height) // 2))
    return panel


def field_atlas() -> Image.Image:
    """The 2048 x 1024 atlas; see the module docstring and FIELD_REGIONS for the layout."""
    atlas = Image.new("RGBA", ATLAS_SIZE, (0, 0, 0, 0))

    def put(name: str, tile: Image.Image) -> None:
        x0, y0, x1, y1 = FIELD_REGIONS[name]
        assert tile.size == (x1 - x0, y1 - y0), (name, tile.size)
        atlas.alpha_composite(tile, (x0, y0))

    for name in ("end_zone_a", "end_zone_b"):
        x0, y0, x1, y1 = FIELD_REGIONS[name]
        put(name, _end_zone(x1 - x0, y1 - y0))
    x0, y0, x1, y1 = FIELD_REGIONS["tiger"]
    put("tiger", tiger_mark(x1 - x0))
    for direction in ("up", "down", "left", "right"):
        put(f"arrow_{direction}", _arrow(direction, _ARROW))
    for n in range(6):
        put(f"numeral_{n}", text(str(n), _NUMERAL_H, WHITE))
    px = np.array(atlas)
    px[px[..., 3] == 0, :3] = TURF
    return Image.fromarray(px, "RGBA")


# --- the scoreboard --------------------------------------------------------------------------

BOARD_SIZE = (1024, 512)
BOARD_REGIONS: dict[str, tuple[int, int, int, int]] = {
    "title": (56, 36, 968, 140),
    "score_a": (56, 176, 328, 448),
    "tiger": (376, 176, 648, 448),
    "score_b": (696, 176, 968, 448),
}
SCORE_PANEL = (14, 10, 22)


def board_uv(name: str) -> tuple[float, float, float, float]:
    """A scoreboard region as a Blender UV box (u0, v0, u1, v1), v up."""
    return _uv(BOARD_REGIONS[name], BOARD_SIZE)


def board() -> Image.Image:
    """The scoreboard face: BATON ROUGE across the top, the tiger between two blank score panels."""
    face = Image.new("RGBA", BOARD_SIZE, PURPLE + (255,))
    draw = ImageDraw.Draw(face)
    w, h = BOARD_SIZE
    draw.rectangle([10, 10, w - 11, h - 11], outline=GOLD, width=6)
    x0, y0, x1, y1 = BOARD_REGIONS["title"]
    lettering = text("BATON ROUGE", round((y1 - y0) * 0.7), GOLD)
    face.alpha_composite(lettering, (x0 + (x1 - x0 - lettering.width) // 2, y0 + (y1 - y0 - lettering.height) // 2))
    draw.rectangle([x0, y1 + 12, x1 - 1, y1 + 15], fill=GOLD)
    for name in ("score_a", "score_b"):
        x0, y0, x1, y1 = BOARD_REGIONS[name]
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=SCORE_PANEL, outline=GOLD, width=3)
    x0, y0, x1, y1 = BOARD_REGIONS["tiger"]
    face.alpha_composite(tiger_mark(x1 - x0), (x0, y0))
    return face


# --- the review sheet ------------------------------------------------------------------------

def review_sheet() -> Image.Image:
    """Every glyph, sample lettering, both marks, the atlas and the board on one sheet (RGB)."""
    pad, ground = 32, (38, 42, 50)
    width = ATLAS_SIZE[0] + 2 * pad
    sheet = Image.new("RGBA", (width, 3200), ground + (255,))
    y = pad

    def put(img: Image.Image, x: int, top: int, under=None) -> None:
        if under is not None:
            sheet.paste(Image.new("RGBA", img.size, under + (255,)), (x, top))
        sheet.alpha_composite(img, (x, top))

    glyphs = sorted(GLYPHS, key=lambda c: (c.isdigit(), c))
    cell = (width - 2 * pad) // 12
    for i, ch in enumerate(glyphs):
        put(text(ch, 120, WHITE), pad + (i % 12) * cell, y + (i // 12) * 170)
    y += 2 * 170 + pad
    for sample, h, colour in (("BATON ROUGE", 120, GOLD), ("BATON ROUGE", 60, WHITE), ("012345", 90, PURPLE)):
        img = text(sample, h, colour, outline=WHITE if colour == PURPLE else None)
        put(img, pad, y, under=(120, 120, 130) if colour == PURPLE else None)
        y += img.height + pad
    put(tiger_mark(512), pad, y, under=TURF)
    put(red_stick(512), pad + 512 + pad, y, under=(150, 150, 150))
    put(text("BATON ROUGE", 80, GOLD, outline=PURPLE), pad + 2 * (512 + pad), y)
    y += 512 + pad
    put(field_atlas(), pad, y, under=TURF)
    y += ATLAS_SIZE[1] + pad
    put(board(), pad, y)
    y += BOARD_SIZE[1] + pad
    return sheet.crop((0, 0, width, y)).convert("RGB")


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/screenshots/v02/logos.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    review_sheet().save(out)
    print(out)
