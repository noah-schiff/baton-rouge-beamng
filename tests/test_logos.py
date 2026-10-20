"""Original logos and field graphics (logos.py): block glyphs, the tiger and red-stick marks,
the field atlas and the scoreboard face. Everything is drawn from polygons and the GLYPHS table."""
import json

import numpy as np
import pytest
from PIL import ImageDraw, ImageFont

from brmap import logos, textures
from brmap.config import Config

PURPLE, GOLD, WHITE = (0x46, 0x1D, 0x7C), (0xFD, 0xD0, 0x23), (255, 255, 255)


def arr(img):
    return np.asarray(img, float)


def share(img, colour, tol=24):
    """The share of the opaque pixels within tol of colour (per channel)."""
    a = np.asarray(img.convert("RGBA"), int)
    opaque = a[..., 3] == 255
    near = (np.abs(a[..., :3] - colour) <= tol).all(axis=2) & opaque
    return near.sum() / max(1, opaque.sum())


def is_power_of_two(n):
    return n > 0 and n & (n - 1) == 0


def boom(*args, **kwargs):
    raise AssertionError("a font was loaded")


# --- GLYPHS --------------------------------------------------------------------------------

def test_glyphs_cover_the_text():
    assert set("BATONRUGE") | set("012345") <= set(logos.GLYPHS)


def test_glyphs_are_exactly_the_planned_set():
    assert set(logos.GLYPHS) == set("ABCDEGIKNORSTU") | set("0123456789")


def test_glyphs_are_polylines_on_a_4_by_6_grid():
    for ch, lines in logos.GLYPHS.items():
        assert lines, ch
        for line in lines:
            assert len(line) >= 2, ch
            for x, y in line:
                assert 0 <= x <= 4 and 0 <= y <= 6, (ch, x, y)


# --- text ----------------------------------------------------------------------------------

def test_no_font_files(monkeypatch):
    monkeypatch.setattr(ImageFont, "truetype", boom)
    monkeypatch.setattr(ImageFont, "load_default", boom)
    monkeypatch.setattr(ImageDraw.ImageDraw, "text", boom)
    assert logos.field_atlas().size == (2048, 1024)
    assert logos.board().size == (1024, 512)
    assert logos.text("BATON ROUGE", 40, GOLD).height == 40


def test_text_scales():
    wide, narrow = logos.text("BATON ROUGE", 100, GOLD), logos.text("BATON ROUGE", 50, GOLD)
    assert abs(wide.width - 2 * narrow.width) <= 4
    assert (wide.height, narrow.height) == (100, 50)


def test_text_is_rgba_in_the_fill_colour_on_clear():
    img = logos.text("BATON", 60, GOLD)
    a = np.asarray(img)
    assert img.mode == "RGBA"
    assert (a[..., 3] == 0).mean() > 0.3                              # clear round the letters
    solid = a[a[..., 3] == 255][:, :3]
    assert len(solid) > 500 and (solid == GOLD).all()                # the strokes are the fill colour


def test_text_edges_are_antialiased():
    a = np.asarray(logos.text("BATON", 60, GOLD))
    assert ((a[..., 3] > 0) & (a[..., 3] < 255)).sum() > 100
    soft = a[(a[..., 3] > 40) & (a[..., 3] < 215)][:, :3]
    assert (soft == GOLD).all()                                      # straight alpha: no dark fringe


def test_text_outline_adds_a_margin_in_the_outline_colour():
    plain = logos.text("BATON", 60, GOLD)
    framed = logos.text("BATON", 60, GOLD, outline=WHITE, stroke=5)
    assert framed.size == (plain.width + 10, 60 + 10)
    a = np.asarray(framed)
    assert (a[a[..., 3] == 255][:, :3] == WHITE).all(axis=1).sum() > 200
    assert (a[a[..., 3] == 255][:, :3] == GOLD).all(axis=1).sum() > 500
    inner = np.asarray(framed)[5:65, 5:5 + plain.width]
    solid = np.asarray(plain)[..., 3] == 255
    solid[:2], solid[-2:], solid[:, :2], solid[:, -2:] = False, False, False, False   # the image edge filters differently
    assert solid.sum() > 500
    assert np.abs(inner[solid][:, :3].astype(int) - GOLD).max() <= 3  # the letters sit in the middle


def test_text_outline_default_stroke_is_a_share_of_the_height():
    framed = logos.text("U", 100, GOLD, outline=WHITE)
    assert framed.height == 100 + 2 * round(100 * logos.OUTLINE_SHARE)


def test_text_space_is_a_gap_and_unknown_characters_are_refused():
    assert logos.text("A A", 60, GOLD).width > logos.text("AA", 60, GOLD).width
    with pytest.raises(ValueError, match="'Z'"):
        logos.text("BAZ", 60, GOLD)
    with pytest.raises(ValueError, match="'a'"):
        logos.text("a", 60, GOLD)


def test_a_digit_is_a_four_by_six_cell():
    for ch in "012345":
        img = logos.text(ch, 300, WHITE)
        assert img.size == (200, 300), ch


def test_text_is_deterministic():
    assert logos.text("BATON ROUGE", 80, GOLD, WHITE).tobytes() == logos.text("BATON ROUGE", 80, GOLD, WHITE).tobytes()


def test_every_glyph_draws_ink():
    for ch in logos.GLYPHS:
        a = np.asarray(logos.text(ch, 120, WHITE))
        assert (a[..., 3] > 128).mean() > 0.12, ch                   # a bold stroke, not a hairline


# --- the tiger mark ------------------------------------------------------------------------

def test_tiger_mark_is_front_facing():
    img = logos.tiger_mark(256)
    assert img.size == (256, 256) and img.mode == "RGBA"
    a = arr(img)
    left, right = a[:, :128], a[:, 128:][:, ::-1]
    assert np.abs(left - right).mean() < 2


@pytest.mark.parametrize("size", [64, 99, 301])
def test_tiger_mark_is_symmetric_at_any_size(size):
    a = arr(logos.tiger_mark(size))
    assert np.abs(a - a[:, ::-1]).mean() < 2


def test_tiger_mark_is_purple_gold_and_white():
    img = logos.tiger_mark(400)
    assert np.asarray(img)[0, 0, 3] == 0 and np.asarray(img)[-1, -1, 3] == 0   # clear corners
    assert share(img, GOLD) > 0.25
    assert share(img, PURPLE) > 0.08
    assert share(img, WHITE) > 0.08
    assert share(img, GOLD) + share(img, PURPLE) + share(img, WHITE) > 0.9      # nothing else


def test_tiger_mark_fills_most_of_the_square():
    alpha = np.asarray(logos.tiger_mark(400))[..., 3] > 0
    rows, cols = np.flatnonzero(alpha.any(axis=1)), np.flatnonzero(alpha.any(axis=0))
    assert rows[0] < 400 * 0.15 and rows[-1] > 400 * 0.85
    assert cols[0] < 400 * 0.15 and cols[-1] > 400 * 0.85


def test_tiger_mark_has_two_eyes_not_one():
    # white eye patches sit left and right of the axis at the same height; the muzzle is on the axis
    a = np.asarray(logos.tiger_mark(400)).astype(int)
    white = (a[..., :3] > 235).all(axis=2) & (a[..., 3] == 255)
    band = white[160:215]                                            # the eye line
    left, middle, right = band[:, :160].sum(), band[:, 160:240].sum(), band[:, 240:].sum()
    assert left > 150 and right > 150
    assert middle < left


def test_tiger_mark_is_deterministic():
    assert logos.tiger_mark(128).tobytes() == logos.tiger_mark(128).tobytes()


# --- the red stick -------------------------------------------------------------------------

def test_red_stick_is_a_roundel():
    img = logos.red_stick(300)
    a = np.asarray(img)
    assert img.size == (300, 300) and img.mode == "RGBA"
    assert a[0, 0, 3] == 0 and a[0, -1, 3] == 0 and a[-1, 0, 3] == 0   # clear outside the ring
    assert a[150, 4, 3] == 255 and a[4, 150, 3] == 255                 # the ring touches the edge
    assert (a[..., 3] == 255).sum() > 0.7 * np.pi * 150 ** 2 * 0.95


def test_red_stick_has_a_red_pole_over_blue_waves():
    img = logos.red_stick(300)
    a = np.asarray(img).astype(int)
    red = (a[..., 0] > 120) & (a[..., 1] < 70) & (a[..., 2] < 70) & (a[..., 3] == 255)
    blue = (a[..., 2] > a[..., 0] + 80) & (a[..., 3] == 255)       # the waves; the navy ring is less blue
    assert red[:, 135:165].sum() > 800                                 # the pole is up the middle
    ys = np.flatnonzero(blue.any(axis=1))
    assert ys.min() > 150 and ys.max() > 240                           # water is in the lower half
    pole_rows = np.flatnonzero(red[:, 140:160].any(axis=1))
    assert pole_rows.min() < 90 and pole_rows.max() > 190              # the pole is tall
    right = red[:, 165:230].sum()
    left = red[:, 70:135].sum()
    assert right > 400 and right > 3 * left                            # pennants stream to one side


def test_red_stick_pennants_are_notched():
    a = np.asarray(logos.red_stick(600)).astype(int)
    red = (a[..., 0] > 120) & (a[..., 1] < 70) & (a[..., 2] < 70) & (a[..., 3] == 255)
    # along the pennants' free end a notch makes the red shorter at mid height than at the corners
    rows = np.flatnonzero(red[:, 330:].any(axis=1))
    assert len(rows) > 20
    ends = [np.flatnonzero(red[y, 330:]).max() for y in rows]
    assert max(ends) - min(ends) > 20


# --- the field atlas -----------------------------------------------------------------------

def test_atlas_power_of_two():
    atlas = logos.field_atlas()
    assert atlas.size == (2048, 1024) and atlas.mode == "RGBA"
    assert all(is_power_of_two(n) for n in atlas.size)


def test_atlas_regions_are_inside_and_apart():
    w, h = logos.ATLAS_SIZE
    names = list(logos.FIELD_REGIONS)
    for name, (x0, y0, x1, y1) in logos.FIELD_REGIONS.items():
        assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h, name
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ax0, ay0, ax1, ay1 = logos.FIELD_REGIONS[a]
            bx0, by0, bx1, by1 = logos.FIELD_REGIONS[b]
            gap_x = max(bx0 - ax1, ax0 - bx1)
            gap_y = max(by0 - ay1, ay0 - by1)
            assert max(gap_x, gap_y) >= logos.GUTTER, (a, b)         # room for the mip levels


def test_atlas_holds_what_the_brief_lists():
    names = set(logos.FIELD_REGIONS)
    assert {"tiger", "end_zone_a", "end_zone_b"} <= names
    assert {f"numeral_{n}" for n in range(6)} <= names
    assert {"arrow_up", "arrow_down", "arrow_left", "arrow_right"} <= names


def test_atlas_numerals_are_six_foot_proportions():
    for n in range(6):
        x0, y0, x1, y1 = logos.FIELD_REGIONS[f"numeral_{n}"]
        assert 3 * (x1 - x0) == 2 * (y1 - y0)                       # 4 ft wide to 6 ft tall


def test_atlas_end_zone_panels_are_the_shape_of_an_end_zone():
    for name in ("end_zone_a", "end_zone_b"):
        x0, y0, x1, y1 = logos.FIELD_REGIONS[name]
        assert (x1 - x0) * 3 == (y1 - y0) * 16                       # 160 ft by 30 ft


def test_atlas_regions_have_content():
    atlas = logos.field_atlas()
    alpha = np.asarray(atlas)[..., 3]
    for name, (x0, y0, x1, y1) in logos.FIELD_REGIONS.items():
        assert (alpha[y0:y1, x0:x1] > 0).mean() > 0.05, name
    x0, y0, x1, y1 = logos.FIELD_REGIONS["end_zone_a"]
    panel = atlas.crop((x0, y0, x1, y1))
    assert share(panel, PURPLE) > 0.4 and share(panel, GOLD) > 0.05  # purple panel, gold lettering


def test_atlas_gutters_are_clear_with_turf_colour_under_them():
    atlas = np.asarray(logos.field_atlas())
    alpha = atlas[..., 3]
    assert alpha[0, 0] == 0 and alpha[-1, -1] == 0 and alpha[500, 1030] == 0
    clear = atlas[alpha == 0][:, :3]
    assert (clear == logos.TURF).all()                               # not black, if alpha is ignored


def test_atlas_numerals_read_as_numerals():
    atlas = logos.field_atlas()
    for n in range(6):
        x0, y0, x1, y1 = logos.FIELD_REGIONS[f"numeral_{n}"]
        tile = atlas.crop((x0, y0, x1, y1))
        expected = logos.text(str(n), y1 - y0, WHITE)
        assert tile.size == expected.size
        assert np.abs(arr(tile)[..., 3] - arr(expected)[..., 3]).mean() < 4


def test_atlas_arrows_point_the_way():
    atlas = np.asarray(logos.field_atlas())[..., 3] > 128
    def mass(name):
        x0, y0, x1, y1 = logos.FIELD_REGIONS[name]
        ys, xs = np.nonzero(atlas[y0:y1, x0:x1])
        return (xs.mean() - (x1 - x0) / 2) / (x1 - x0), (ys.mean() - (y1 - y0) / 2) / (y1 - y0)
    # a triangle's tip is its light end: the centre of mass sits towards the base, away from the way it points
    assert mass("arrow_up")[1] > 0.05 and abs(mass("arrow_up")[0]) < 0.02       # image y runs down
    assert mass("arrow_down")[1] < -0.05 and abs(mass("arrow_down")[0]) < 0.02
    assert mass("arrow_left")[0] > 0.05 and abs(mass("arrow_left")[1]) < 0.02
    assert mass("arrow_right")[0] < -0.05 and abs(mass("arrow_right")[1]) < 0.02


def test_atlas_midfield_mark_is_the_tiger():
    atlas = logos.field_atlas()
    x0, y0, x1, y1 = logos.FIELD_REGIONS["tiger"]
    crop = atlas.crop((x0, y0, x1, y1))
    assert crop.size[0] == crop.size[1]
    mark = logos.tiger_mark(crop.width)
    assert np.abs(arr(crop)[..., 3] - arr(mark)[..., 3]).mean() < 3
    solid = np.asarray(mark)[..., 3] == 255
    assert (np.abs(np.asarray(crop).astype(int) - np.asarray(mark))[solid] <= 2).all()


def test_atlas_is_deterministic():
    assert logos.field_atlas().tobytes() == logos.field_atlas().tobytes()


def test_atlas_uv_rect_uses_blenders_v_up():
    u0, v0, u1, v1 = logos.atlas_uv("end_zone_a")
    x0, y0, x1, y1 = logos.FIELD_REGIONS["end_zone_a"]
    assert (u0, u1) == (x0 / 2048, x1 / 2048)
    assert (v0, v1) == (1 - y1 / 1024, 1 - y0 / 1024) and v1 > 0.9      # the top of the image is v = 1


# --- the scoreboard ------------------------------------------------------------------------

def test_board_size_and_content():
    img = logos.board()
    assert img.size == (1024, 512) and img.mode == "RGBA"
    assert all(is_power_of_two(n) for n in img.size)
    a = np.asarray(img)
    assert (a[..., 3] == 255).all()                                   # a solid face
    assert share(img, GOLD) > 0.03 and share(img, PURPLE) > 0.15


def test_board_score_panels_are_blank():
    img = logos.board()
    a = np.asarray(img).astype(int)
    for name in ("score_a", "score_b"):
        x0, y0, x1, y1 = logos.BOARD_REGIONS[name]
        inside = a[y0 + 8:y1 - 8, x0 + 8:x1 - 8, :3]
        assert inside.std(axis=(0, 1)).max() < 2                      # one flat dark colour
        assert inside.mean() < 40


def test_board_title_and_tiger_regions():
    img = logos.board()
    x0, y0, x1, y1 = logos.BOARD_REGIONS["title"]
    assert share(img.crop((x0, y0, x1, y1)), GOLD, tol=10) > 0.1       # the lettering
    x0, y0, x1, y1 = logos.BOARD_REGIONS["tiger"]
    crop = img.crop((x0, y0, x1, y1))
    assert share(crop, GOLD) > 0.15 and share(crop, WHITE, tol=10) > 0.03
    a = arr(crop)
    assert np.abs(a - a[:, ::-1]).mean() < 6                           # front-facing on the board too


def test_board_regions_are_inside_and_do_not_overlap():
    boxes = list(logos.BOARD_REGIONS.values())
    for x0, y0, x1, y1 in boxes:
        assert 0 <= x0 < x1 <= 1024 and 0 <= y0 < y1 <= 512
    for i, (ax0, ay0, ax1, ay1) in enumerate(boxes):
        for bx0, by0, bx1, by1 in boxes[i + 1:]:
            assert ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0


def test_board_uv_rect():
    u0, v0, u1, v1 = logos.board_uv("title")
    assert v1 > 0.8 and u0 < 0.1 and u1 > 0.9


# --- the textures stage ----------------------------------------------------------------------

def test_textures_register_the_atlas_and_the_board():
    maps = textures.GENERATED["field_atlas"]()
    assert set(maps) == {"b"} and maps["b"].size == (2048, 1024)
    maps = textures.GENERATED["board"]()
    assert set(maps) == {"b"} and maps["b"].size == (1024, 512)


def test_textures_stage_ships_them_as_dds(tmp_path, monkeypatch):
    monkeypatch.setattr(textures, "load_registry", lambda *a: {})     # no CC0 downloads in a test
    textures.run(Config(out_dir=str(tmp_path), raw_dir=str(tmp_path / "raw"), texture_max_size=2048))
    folder = tmp_path / "textures"
    index = json.loads((folder / "index.json").read_text(encoding="utf-8"))
    assert index["field_atlas"]["files"] == {"b": "field_atlas_b.color.dds"}
    assert index["board"]["files"] == {"b": "board_b.color.dds"}
    head = (folder / "field_atlas_b.color.dds").read_bytes()
    assert head[:4] == b"DDS " and int.from_bytes(head[12:16], "little") == 1024 \
        and int.from_bytes(head[16:20], "little") == 2048
    assert head[84:88] == b"DX10" and int.from_bytes(head[128:132], "little") == 78   # BC3, sRGB: it has alpha
    board = (folder / "board_b.color.dds").read_bytes()
    assert int.from_bytes(board[128:132], "little") == 72                              # opaque: BC1, sRGB


# --- the review sheet --------------------------------------------------------------------

def test_review_sheet_shows_everything():
    sheet = logos.review_sheet()
    assert sheet.mode == "RGB" and sheet.width >= 1600
    a = np.asarray(sheet).astype(int)
    assert (a.std(axis=(0, 1)) > 20).all()                             # not blank
