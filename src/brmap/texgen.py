"""Textures drawn by the repo's own code (Pillow and numpy, deterministic), for the textures stage.

* clock_face(size=512) -> RGBA: a cream dial with a dark ring, 60 minute ticks, 12 hour bars
  (a double bar at 12, so the dial shows which way is up) and the hands at 10:10, in a square of
  light stone (STONE, close to the tower's trim). Every pixel is opaque: with the material's
  alpha test off, clear corners rendered black round the dial in the game (spike S1).
* bevel_normal(size=256) -> RGB: a tangent-space normal map of one raised square block whose
  edges bevel down to the border, so a tiled surface reads as blocks with V joints. +X points
  right; +Y points up in the image when NORMAL_CONVENTION is "opengl", down for "directx". The
  in-game spike S1 (2026-10-19) showed the game reads the OpenGL one as raised.
* lsu_wall(size=512) -> {"b", "nm", "r"}: the LSU campus core's wall (styled.yaml's lsu style),
  one bay (LSU_BAY_M) of one floor (LSU_FLOOR_M) per repeat: tan stucco with a round-arched
  window in a cast-stone surround and sill, white frames and muntins, dark glass; its normal map
  (NORMAL_CONVENTION) sets the window back and the stone forward, and its roughness map makes
  the glass smooth. The image's top row is the top of the floor (v up, as the game shows
  Blender's uv), and it tiles both ways.
* facade(style, size=512) -> {"b", "r"} (+ "m"): the wall of each styled.yaml style. lsu is
  lsu_wall; the downtown standouts' styles (FACADES: ribbon_dark, precast_punched, brick_hotel,
  precast_arena, glass_blue) hold `floors` floors and `bays` bays per repeat (styled.yaml's
  floors_per_tile and bays_per_tile), each bay's window centred in it, the top row up, tiling
  both ways; "m" is the metallic factor of the styles whose glass needs one (the material's
  metallicFactor).
* gothic_window(size=256) -> RGBA: the Old State Capitol's pointed-arch window for its decal
  quads, drawn GOTHIC_ASPECT times as tall as wide, clear outside the arch.
* moss_card(size=(512, 1024), seed=0) -> {"b", "o", "nm"}: hanging Spanish moss for the moss strands
  (brtree.hanging_card): grey-green random-walk filaments 1-3 px wide, swaying and branching, dense at
  the top edge (where a strand hangs from) and thinning toward the foot; the opacity is the strokes'
  mask and the normal map (NORMAL_CONVENTION) the blurred mask as height.

They are drawn SUPERSAMPLE times larger, or computed per pixel (the stucco's mottling from a
seeded generator), so the same call gives the same bytes every time.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

NORMAL_CONVENTION = "opengl"   # "opengl": green is +Y up; "directx": green is +Y down
SUPERSAMPLE = 4

# the clock, as fractions of the image side, measured from the centre
DIAL_R = 0.49                  # the outer edge of the ring
RING_W = 0.035
TICK_R, TICK_LEN, TICK_W = 0.445, 0.02, 0.006
BAR_R, BAR_LEN, BAR_W = 0.445, 0.09, 0.032
BAR_MID = BAR_R - BAR_LEN / 2  # the middle of the hour bars
DOUBLE_GAP = 0.012             # between the two bars at 12
HOUR_HAND = (0.24, 0.035)      # length, width at the hub
MINUTE_HAND = (0.37, 0.022)
TAIL = 0.16                    # how far a hand reaches back past the hub, as a share of its length
HUB_R = 0.025
CREAM = (238, 226, 196)
DARK = (36, 32, 30)
STONE = (226, 218, 200)        # the square round the dial: light stone, near the trim

BEVEL = 1 / 8                  # bevel width, as a share of the side
BEVEL_SLOPE = math.tan(math.radians(40))

# the LSU wall: one bay (LSU_BAY_M wide) of one floor (LSU_FLOOR_M high), as styled.yaml's lsu
# style maps it (u = metres along the wall / bay, v = metres up / floor); sizes in metres
LSU_BAY_M, LSU_FLOOR_M = 3.0, 3.6
LSU_WINDOW = (1.5, 1.3)        # centre along the bay, width
LSU_SILL, LSU_SPRING = 0.9, 2.45   # the opening's bottom; where its round arch springs (crown 3.1)
LSU_FRAME, LSU_BAR, LSU_MUNTIN = 0.07, 0.06, 0.03  # frame, mullion and transom, muntin widths
LSU_MUNTINS_Z = (1.45, 1.95)   # the lower sash's horizontal muntins
LSU_TRIM = 0.12                # the cast-stone band round the arch, and the sill's height
LSU_SILL_OUT = 0.1             # the sill reaches past the opening on each side
LSU_STUCCO = (214, 190, 150)   # tan
LSU_STONE = (226, 214, 190)    # cast stone
LSU_PAINT = (236, 232, 222)    # the window frames
LSU_GLASS = (46, 56, 64)
LSU_ROUGH = {"stucco": 0.88, "stone": 0.75, "paint": 0.5, "glass": 0.12}
LSU_DEPTH = {"stucco": 0.0, "stone": 0.03, "paint": -0.10, "glass": -0.14}  # m out of the wall
LSU_NOISE = 4.0                # stucco mottling, levels of 255 (standard deviation)
LSU_SEED = 17

# The downtown standouts' walls (styled.yaml's other styles): one repeat holds `floors` floors of
# floor_m and `bays` bays of bay_m (styled.yaml's floors_per_tile, bays_per_tile, floor_m and bay_m;
# a group with a floor height of its own stretches it). Within a floor and a bay, sizes are metres:
# z up from the floor line, x from the bay's left edge. Every bay's window is centred in it, so a
# repeat's edges fall between two windows. "m" is the metallic factor the glass needs.
FACADES = {
    # dark tinted glass ribbons between dark bronze spandrels, a mullion at every bay
    "ribbon_dark": {"floors": 4, "bays": 10, "floor_m": 3.8, "bay_m": 1.5, "m": 0.3,
                    "window": (0.0, 1.5, 0.95, 2.85), "frame": 0.05,
                    "colour": {"wall": (56, 52, 50), "frame": (82, 78, 72), "glass": (24, 31, 38),
                               "blind": (132, 130, 124)},
                    "rough": {"wall": 0.5, "frame": 0.45, "glass": 0.06, "blind": 0.75}},
    # light precast concrete panels with punched windows (a mullion in each), the reveal's shadow over them
    "precast_punched": {"floors": 4, "bays": 8, "floor_m": 3.8, "bay_m": 2.0,
                        "window": (0.35, 1.65, 0.9, 2.6), "frame": 0.06, "mullion": True, "joints": 2,
                        "reveal": 0.12,
                        "colour": {"wall": (206, 200, 186), "joint": (158, 153, 143), "reveal": (128, 124, 116),
                                   "frame": (92, 94, 97), "glass": (38, 46, 54), "blind": (150, 148, 140)},
                        "rough": {"wall": 0.88, "joint": 0.92, "reveal": 0.9, "frame": 0.45, "glass": 0.06,
                                  "blind": 0.75}},
    # 1920s red-brown brick, double-hung windows with stone sills and lintels
    "brick_hotel": {"floors": 4, "bays": 6, "floor_m": 3.8, "bay_m": 2.4,
                    "window": (0.65, 1.75, 0.85, 2.75), "frame": 0.07, "sash": 0.05,
                    "sill": (0.1, 0.15), "lintel": (0.05, 0.25), "brick": (10, 49),
                    "colour": {"wall": (150, 82, 62), "trim": (212, 202, 182), "frame": (228, 226, 220),
                               "glass": (36, 40, 44), "blind": (186, 172, 146)},
                    "rough": {"wall": 0.92, "trim": 0.8, "frame": 0.45, "glass": 0.06, "blind": 0.75}},
    # tall beige precast panels, fluted, a joint at every bay and floor; no windows
    "precast_arena": {"floors": 4, "bays": 5, "floor_m": 3.8, "bay_m": 3.0, "flute": (0.6, 0.9), "joints": 1,
                      "colour": {"wall": (196, 186, 166), "flute": (174, 164, 146), "joint": (148, 140, 126)},
                      "rough": {"wall": 0.88, "flute": 0.9, "joint": 0.92}},
    # a blue-green glass curtain wall: light mullions, an opaque glass spandrel over each slab
    "glass_blue": {"floors": 4, "bays": 10, "floor_m": 3.8, "bay_m": 1.5, "m": 0.5,
                   "window": (0.0, 1.5, 1.0, 3.8), "frame": 0.05, "spandrel": (0.05, 0.95),
                   "colour": {"wall": (74, 96, 112), "frame": (180, 186, 192), "glass": (46, 86, 110)},
                   "rough": {"wall": 0.3, "frame": 0.45, "glass": 0.06}},
}
FACADE_SEED = 23
BLINDS = 0.4                   # the share of windows with a blind down, ...
BLIND_DROP = (0.15, 0.8)       # ... over this share of the glass, from its top
PANE_SPREAD = 0.12             # each pane's brightness varies by up to this share
SKY = 0.25                     # the glass is this much brighter at its top (the sky in it) than at its foot
FACADE_NOISE = 3.0             # the cladding's mottling, levels of 255 (standard deviation)
BRICK_NOISE = 9.0              # the bricks' shades, levels of 255 (standard deviation)
JOINT = 0.03                   # m: a panel joint's width

# The Old State Capitol's pointed-arch window (gothic_window), a decal on its white stucco walls; sizes
# in half-widths of the window
GOTHIC_ASPECT = 2.2            # the decal's height over its width
GOTHIC_FRAME = 0.12            # the stone frame
GOTHIC_MULLION = 0.1           # between the two lancets
GOTHIC_LANCET_DROP = 0.5       # the lancets spring this far under the outer arch
GOTHIC_LEAD_STEP, GOTHIC_LEAD_W = 0.32, 0.035   # the leading's spacing and width, up the glass
GOTHIC_STONE = (232, 228, 218)
GOTHIC_GLASS = (40, 48, 60)
GOTHIC_LEAD = (78, 80, 84)

# Spanish moss (moss_card): sizes in pixels of the finished card, shares of its height. The greens were
# #8F9682 .. #BFC4AE until the full-map check on 0.40, where lit by the sun they read near white.
MOSS_COLOURS = ((0x6F, 0x78, 0x65), (0xA2, 0xA8, 0x94))   # grey-green, the darkest and the palest
MOSS_FILAMENTS = 240           # filaments hanging from the top edge
MOSS_START = 0.06              # they start |N(0, this)| of the height below the top edge
MOSS_LENGTH = (0.1, 2.5)       # length = lo + (1 - lo) u^hi of the card's height: most stop well short of the foot
MOSS_WIDTHS = ((1, 0.45), (2, 0.4), (3, 0.15))   # stroke width (px) and how often
MOSS_STEP = 4.0                # px down per point of a filament
MOSS_DRIFT = 0.7               # px of random sideways drift per step (standard deviation)
MOSS_SWAY = ((1.0, 4.0), (60.0, 220.0))   # the sway's amplitude (px) and wavelength (px) ranges
MOSS_BRANCH = 0.035            # chance per step of a side branch ...
MOSS_BRANCH_LEN = (0.03, 0.14)  # ... this long (share of the height) ...
MOSS_BRANCH_SLANT = (0.25, 0.7)  # ... leaving at this slope (px across per px down)
MOSS_BLUR = 1.5                # px: the mask's blur before it is used as height for the normal map
MOSS_RELIEF = 3.0              # the normal map's strength


def _bar(c: float, angle: float, r0: float, r1: float, width: float, offset: float = 0.0):
    """The corners of a bar from radius r0 to r1 along a clock angle (radians clockwise from
    12), shifted sideways by offset; image coordinates (y down)."""
    d = (math.sin(angle), -math.cos(angle))     # outwards
    p = (math.cos(angle), math.sin(angle))      # clockwise across it
    out = []
    for r, side in ((r0, -1), (r1, -1), (r1, 1), (r0, 1)):
        s = offset + side * width / 2
        out.append((c + d[0] * r + p[0] * s, c + d[1] * r + p[1] * s))
    return out


def _hand(c: float, angle: float, length: float, width: float):
    """A tapered hand from TAIL behind the hub to its tip."""
    d = (math.sin(angle), -math.cos(angle))
    p = (math.cos(angle), math.sin(angle))
    tail, tip = -TAIL * length, length
    pts = [(tail, -width / 2), (tip * 0.85, -width / 3), (tip, 0.0), (tip * 0.85, width / 3), (tail, width / 2)]
    return [(c + d[0] * r + p[0] * s, c + d[1] * r + p[1] * s) for r, s in pts]


def clock_face(size: int = 512) -> Image.Image:
    """The clock dial; see the module docstring."""
    big = size * SUPERSAMPLE
    c = big / 2
    img = Image.new("RGBA", (big, big), STONE + (255,))
    draw = ImageDraw.Draw(img)

    def disc(r: float, colour) -> None:
        draw.ellipse([c - r * big, c - r * big, c + r * big, c + r * big], fill=colour + (255,))

    disc(DIAL_R, DARK)
    disc(DIAL_R - RING_W, CREAM)
    for k in range(60):
        angle = math.radians(6 * k)
        if k % 5:
            draw.polygon(_bar(c, angle, (TICK_R - TICK_LEN) * big, TICK_R * big, TICK_W * big), fill=DARK)
        elif k == 0:
            shift = (BAR_W + DOUBLE_GAP) / 2 * big
            for off in (-shift, shift):
                draw.polygon(_bar(c, angle, (BAR_R - BAR_LEN) * big, BAR_R * big, BAR_W * big, off), fill=DARK)
        else:
            draw.polygon(_bar(c, angle, (BAR_R - BAR_LEN) * big, BAR_R * big, BAR_W * big), fill=DARK)
    minute = math.radians(6 * 10)
    hour = math.radians(30 * (10 + 10 / 60))
    draw.polygon(_hand(c, hour, HOUR_HAND[0] * big, HOUR_HAND[1] * big), fill=DARK)
    draw.polygon(_hand(c, minute, MINUTE_HAND[0] * big, MINUTE_HAND[1] * big), fill=DARK)
    disc(HUB_R, DARK)
    return img.resize((size, size), Image.Resampling.LANCZOS)


def _wrap_blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """a blurred by a Gaussian of sigma pixels, wrapping round its edges (so a tile stays seamless)."""
    f0, f1 = np.fft.fftfreq(a.shape[0])[:, None], np.fft.fftfreq(a.shape[1])[None, :]
    return np.real(np.fft.ifft2(np.fft.fft2(a) * np.exp(-2 * (math.pi * sigma) ** 2 * (f0 ** 2 + f1 ** 2))))


def _lsu_parts(size: int) -> dict[str, np.ndarray]:
    """{stucco|stone|paint|glass: (size, size) share of each pixel}, drawn SUPERSAMPLE times larger
    and averaged down. Row 0 is the top of the floor (v = 1, as the game shows Blender's uv)."""
    big = size * SUPERSAMPLE
    centres = (np.arange(big) + 0.5) / big
    x = (centres * LSU_BAY_M)[None, :]
    z = ((1.0 - centres) * LSU_FLOOR_M)[:, None]
    cx, half = LSU_WINDOW[0], LSU_WINDOW[1] / 2
    dx, r = np.abs(x - cx), np.hypot(x - cx, z - LSU_SPRING)

    def opening(shrink: float) -> np.ndarray:
        return (dx <= half - shrink) & (z >= LSU_SILL + shrink) & ((z <= LSU_SPRING) | (r <= half - shrink))

    hole, glazed = opening(0.0), opening(LSU_FRAME)
    pane = (LSU_BAR / 2 + half - LSU_FRAME) / 2                         # the middle of a sash
    bars = ((dx <= LSU_BAR / 2) | (np.abs(z - LSU_SPRING) <= LSU_BAR / 2)
            | ((z < LSU_SPRING) & (np.abs(dx - pane) <= LSU_MUNTIN / 2)))
    for mz in LSU_MUNTINS_Z:
        bars = bars | (np.abs(z - mz) <= LSU_MUNTIN / 2)
    crown = LSU_SPRING + half
    stone = (((r <= half + LSU_TRIM) & (z >= LSU_SPRING))                # the arch's band
             | ((dx <= 0.09) & (z >= crown) & (z <= crown + LSU_TRIM + 0.06))   # its keystone
             | ((dx <= half + LSU_SILL_OUT) & (z >= LSU_SILL - LSU_TRIM) & (z < LSU_SILL)))  # the sill
    glass = glazed & ~bars
    paint = hole & ~glass
    stone = stone & ~hole
    parts = {"stucco": ~(stone | hole), "stone": stone, "paint": paint, "glass": glass}
    k = SUPERSAMPLE
    return {name: m.reshape(size, k, size, k).mean(axis=(1, 3)) for name, m in parts.items()}


def lsu_wall(size: int = 512) -> dict[str, Image.Image]:
    """The LSU campus core's wall, one bay of one floor: tan stucco with a round-arched window in a
    cast-stone surround and sill, white frames and muntins, dark glass. {"b": RGB colour, "nm": RGB
    normal map (OpenGL, the window recessed, the stone proud), "r": L roughness}."""
    parts = _lsu_parts(size)
    colours = {"stucco": LSU_STUCCO, "stone": LSU_STONE, "paint": LSU_PAINT, "glass": LSU_GLASS}
    rgb = sum(parts[n][..., None] * np.array(c, float) for n, c in colours.items())
    noise = _wrap_blur(np.random.default_rng(LSU_SEED).normal(size=(size, size)), 1.5)
    rgb += (parts["stucco"] * noise * (LSU_NOISE / noise.std()))[..., None]
    b = Image.fromarray(np.clip(np.round(rgb), 0, 255).astype(np.uint8), "RGB")

    height = _wrap_blur(sum(parts[n] * d for n, d in LSU_DEPTH.items()), 1.0)
    gx = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) / (2 * LSU_BAY_M / size)
    gz = (np.roll(height, 1, axis=0) - np.roll(height, -1, axis=0)) / (2 * LSU_FLOOR_M / size)  # row 0 is up
    n = np.stack([-gx, -gz if NORMAL_CONVENTION == "opengl" else gz, np.ones_like(gx)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    nm = Image.fromarray(np.round((n * 0.5 + 0.5) * 255).astype(np.uint8), "RGB")

    rough = sum(parts[k] * v for k, v in LSU_ROUGH.items())
    r = Image.fromarray(np.round(rough * 255).astype(np.uint8), "L")
    return {"b": b, "nm": nm, "r": r}


def _facade_grid(spec: dict, n: int):
    """For an n x n drawing of one repeat: x and z (m within a bay and a floor; x a row vector,
    z a column, row 0 the repeat's top) and each pixel's floor and bay (column and row vectors)."""
    centres = (np.arange(n) + 0.5) / n
    across, up = centres * spec["bays"] * spec["bay_m"], (1.0 - centres) * spec["floors"] * spec["floor_m"]
    bay = np.minimum(across // spec["bay_m"], spec["bays"] - 1).astype(int)
    floor = np.minimum(up // spec["floor_m"], spec["floors"] - 1).astype(int)
    return ((across - bay * spec["bay_m"])[None, :], (up - floor * spec["floor_m"])[:, None],
            floor[:, None], bay[None, :])


def _facade_masks(spec: dict, n: int, drop: np.ndarray) -> dict[str, np.ndarray]:
    """{part: (n, n) bool} of one repeat drawn n px wide; drop is each window's blind, (floors, bays)."""
    x, z, floor, bay = _facade_grid(spec, n)
    full = np.zeros((n, n), bool)
    parts: dict[str, np.ndarray] = {}
    if "window" in spec:
        x0, x1, z0, z1 = spec["window"]
        f = spec["frame"]
        hole = (x >= x0) & (x < x1) & (z >= z0) & (z < z1)
        glass = (x >= x0 + f) & (x < x1 - f) & (z >= z0 + f) & (z < z1 - f)
        if spec.get("mullion"):
            glass &= np.abs(x - (x0 + x1) / 2) >= f / 2
        if "sash" in spec:                                       # the meeting rails of a double-hung window
            glass &= np.abs(z - (z0 + z1) / 2) >= spec["sash"] / 2
        blind = glass & (z >= z1 - f - drop[floor, bay] * (z1 - z0 - 2 * f))
        parts |= {"glass": glass & ~blind, "blind": blind, "frame": hole & ~glass}
        full |= hole
    if "spandrel" in spec:                                       # glass_blue: the transoms over and under it
        s0, s1 = spec["spandrel"]
        transom = ((z < s0) | ((z >= s1) & (z < spec["window"][2]))) & ~full
        edge = ((x < spec["frame"]) | (x >= spec["bay_m"] - spec["frame"])) & ~full
        parts["frame"] = parts["frame"] | transom | edge
        full |= transom | edge
    if "reveal" in spec:                                         # the shadow under the window head
        x0, x1, z0, z1 = spec["window"]
        reveal = (x >= x0) & (x < x1) & (z >= z1) & (z < z1 + spec["reveal"])
        parts["reveal"] = reveal & ~full
        full |= reveal
    if "sill" in spec:
        x0, x1, z0, z1 = spec["window"]
        (out, sill), (lout, lintel) = spec["sill"], spec["lintel"]
        trim = (((x >= x0 - out) & (x < x1 + out) & (z >= z0 - sill) & (z < z0))
                | ((x >= x0 - lout) & (x < x1 + lout) & (z >= z1) & (z < z1 + lintel)))
        parts["trim"] = trim & ~full
        full |= trim
    if "flute" in spec:                                          # a fluted strip in each bay
        a, w = spec["flute"]
        centre = spec["bay_m"] / 2
        flute = (np.abs(x - centre) >= a / 2) & (np.abs(x - centre) < a / 2 + w) & np.ones_like(z, bool)
        parts["flute"] = flute & ~full
        full |= flute
    if "joints" in spec:                                         # panel joints: each floor line, every k bays
        k = spec["joints"]
        joint = (z < JOINT) | ((bay % k == 0) & (x < JOINT))
        parts["joint"] = joint & ~full
        full |= joint
    parts["wall"] = ~full
    return parts


def facade(style: str, size: int = 512) -> dict:
    """One repeat of a styled building's wall, `size` px square: {"b": RGB colour, "r": L roughness},
    with "m" (the metallic factor) for the styles with glass that needs one (FACADES). The lsu style is
    lsu_wall(size). Glass gets a brighter top (the sky in it) and a slightly different shade per pane,
    some windows a blind; the cladding a fine mottling (brick: one shade per brick)."""
    if style == "lsu":
        return lsu_wall(size)
    if style not in FACADES:
        raise ValueError(f"no facade for style {style!r} (one of lsu, {', '.join(FACADES)})")
    spec = FACADES[style]
    rng = np.random.default_rng(FACADE_SEED)
    shape = (spec["floors"], spec["bays"])
    drop = np.where(rng.random(shape) < BLINDS, rng.uniform(*BLIND_DROP, shape), 0.0)
    if "blind" not in spec["colour"]:                            # glass_blue: no blinds behind the curtain wall
        drop[:] = 0.0
    pane = 1.0 + rng.uniform(-PANE_SPREAD, PANE_SPREAD, shape)
    k = SUPERSAMPLE
    parts = {name: m.reshape(size, k, size, k).mean(axis=(1, 3))
             for name, m in _facade_masks(spec, size * k, drop).items()}

    rgb = sum(parts[name][..., None] * np.array(c, float) for name, c in spec["colour"].items() if name in parts)
    x, z, floor, bay = _facade_grid(spec, size)
    if "window" in spec:
        _, _, z0, z1 = spec["window"]
        sky = 1.0 + SKY * np.clip((z - z0) / (z1 - z0), 0.0, 1.0) - SKY / 2
        rgb += (parts["glass"] * (pane[floor, bay] * sky - 1.0))[..., None] * np.array(spec["colour"]["glass"], float)
    noise = _wrap_blur(rng.normal(size=(size, size)), 1.0)
    level = FACADE_NOISE
    if "brick" in spec:                       # one shade per brick, each course offset by half a brick
        across, courses = spec["brick"]       # bricks per bay, courses per floor (whole, so the repeat tiles)
        course = np.floor((z + floor * spec["floor_m"]) / (spec["floor_m"] / courses)).astype(int)
        index = np.floor((x + bay * spec["bay_m"]) / (spec["bay_m"] / across) + 0.5 * (course % 2)).astype(int)
        noise = rng.normal(size=(spec["floors"] * courses, spec["bays"] * across))[
            course % (spec["floors"] * courses), index % (spec["bays"] * across)]
        level = BRICK_NOISE
    rgb += (parts["wall"] * noise * (level / max(float(np.std(noise)), 1e-9)))[..., None]
    b = Image.fromarray(np.clip(np.round(rgb), 0, 255).astype(np.uint8), "RGB")
    rough = sum(parts[name] * v for name, v in spec["rough"].items() if name in parts)
    out = {"b": b, "r": Image.fromarray(np.round(rough * 255).astype(np.uint8), "L")}
    if "m" in spec:
        out["m"] = float(spec["m"])
    return out


def gothic_window(size: int = 256) -> Image.Image:
    """The Old State Capitol's window, RGBA, size px square, drawn for a decal GOTHIC_ASPECT times
    as tall as it is wide: two pointed lancets and a round light over them in a light stone frame,
    under an equilateral pointed arch whose apex is the top edge; dark leaded glass; alpha 0 outside
    the arch (the wall shows round its head; the material's alpha test cuts it), coloured as the
    stone there so the mipmaps keep no dark fringe."""
    big = size * SUPERSAMPLE
    c = (np.arange(big) + 0.5) / big
    x = ((c - 0.5) * 2.0)[None, :]                          # -1 .. 1 across, in half-widths
    y = ((1.0 - c) * 2.0 * GOTHIC_ASPECT)[:, None]          # up from the foot, in half-widths

    def arch(cx: float, half: float, spring: float) -> np.ndarray:
        """Inside the pointed arch `half` wide each side of cx, springing at y = spring: its two arcs
        have the arch's span as radius, centred on the opposite springing points."""
        r2 = (2.0 * half) ** 2
        head = (((x - cx + half) ** 2 + (y - spring) ** 2 <= r2) & ((x - cx - half) ** 2 + (y - spring) ** 2 <= r2))
        return (np.abs(x - cx) <= half) & (y >= 0.0) & ((y <= spring) | head)

    spring = 2.0 * GOTHIC_ASPECT - math.sqrt(3.0)           # the outer arch's apex at the top edge
    f, m = GOTHIC_FRAME, GOTHIC_MULLION
    outer = arch(0.0, 1.0, spring)
    inner = arch(0.0, 1.0 - f, spring) & (y >= f)
    half = (1.0 - f - m / 2) / 2                            # each lancet's half-width
    low = spring - GOTHIC_LANCET_DROP
    lancets = arch(-(m / 2 + half), half, low) | arch(m / 2 + half, half, low)
    apex, top = low + half * math.sqrt(3.0), spring + (1.0 - f) * math.sqrt(3.0)
    r = (top - apex) / 2 - m
    light = x ** 2 + (y - (apex + top) / 2) ** 2 <= r * r
    glass = (lancets | light) & inner
    lead = glass & (np.abs((y / GOTHIC_LEAD_STEP) % 1.0 - 0.5) > 0.5 - GOTHIC_LEAD_W / GOTHIC_LEAD_STEP / 2)
    rgb = ((outer & ~glass)[..., None] * np.array(GOTHIC_STONE, float)
           + (glass & ~lead)[..., None] * np.array(GOTHIC_GLASS, float) + lead[..., None] * np.array(GOTHIC_LEAD, float))
    k = SUPERSAMPLE
    alpha = outer.reshape(size, k, size, k).mean(axis=(1, 3))
    rgb = rgb.reshape(size, k, size, k, 3).mean(axis=(1, 3))
    rgb = np.where(alpha[..., None] > 0, rgb / np.maximum(alpha, 1e-9)[..., None], np.array(GOTHIC_STONE, float))
    rgba = np.concatenate([rgb, alpha[..., None] * 255.0], axis=-1)
    return Image.fromarray(np.clip(np.round(rgba), 0, 255).astype(np.uint8), "RGBA")


def _moss_walk(rng, x: float, y: float, length: float, slant: float, sway: tuple[float, float, float]):
    """The points of one filament (supersampled px): down from (x, y) for `length` px, drifting sideways
    at random, swaying with sway = (amplitude, wavelength, phase) and leaning `slant` px per px down."""
    amp, wave, phase = sway
    n = max(2, int(length / (MOSS_STEP * SUPERSAMPLE)) + 1)
    dy = length / (n - 1)
    drift = np.cumsum(rng.normal(0.0, MOSS_DRIFT * SUPERSAMPLE, n))
    t = np.arange(n) * dy
    xs = x + drift + slant * t + amp * SUPERSAMPLE * np.sin(2 * math.pi * t / (wave * SUPERSAMPLE) + phase)
    return list(zip(xs.tolist(), (y + t).tolist()))


def moss_card(size: tuple[int, int] = (512, 1024), seed: int = 0) -> dict[str, Image.Image]:
    """A card of hanging Spanish moss, `size` = (width, height) px, its top edge where it hangs from:
    {"b": RGB colour, "o": L opacity, "nm": RGB normal map (OpenGL)}.

    Random-walk filaments 1-3 px wide hang from near the top with a sinusoidal sway and short side
    branches; their lengths make it dense at the top and thin toward the foot. Each is one grey-green
    between MOSS_COLOURS; the colour map is that green everywhere, the empty parts too (no dark fringe in
    the mipmaps). The opacity is the strokes' mask; the normal map treats the mask, blurred by MOSS_BLUR
    px, as height. Drawn SUPERSAMPLE times larger and averaged down; the same seed gives the same bytes."""
    w, h = size
    k = SUPERSAMPLE
    rng = np.random.default_rng(seed)
    lo, hi = (np.array(c, float) for c in MOSS_COLOURS)
    mid = tuple(int(round(v)) for v in (lo + hi) / 2)
    colour = Image.new("RGB", (w * k, h * k), mid)
    mask = Image.new("L", (w * k, h * k), 0)
    paint, cut = ImageDraw.Draw(colour), ImageDraw.Draw(mask)
    widths, odds = zip(*MOSS_WIDTHS)

    def stroke(points, width: int, rgb) -> None:
        paint.line(points, fill=rgb, width=width * k, joint="curve")
        cut.line(points, fill=255, width=width * k, joint="curve")

    for _ in range(MOSS_FILAMENTS):
        x, y = rng.uniform(0, w * k), abs(rng.normal(0.0, MOSS_START)) * h * k
        length = (MOSS_LENGTH[0] + (1 - MOSS_LENGTH[0]) * rng.random() ** MOSS_LENGTH[1]) * h * k
        width = int(rng.choice(widths, p=odds))
        rgb = tuple(int(round(v)) for v in lo + rng.random() * (hi - lo))
        sway = (rng.uniform(*MOSS_SWAY[0]), rng.uniform(*MOSS_SWAY[1]), rng.uniform(0, 2 * math.pi))
        points = _moss_walk(rng, x, y, length, 0.0, sway)
        stroke(points, width, rgb)
        for bx, by in points[1:-1]:
            if rng.random() < MOSS_BRANCH:
                side = 1 if rng.random() < 0.5 else -1
                twig = _moss_walk(rng, bx, by, rng.uniform(*MOSS_BRANCH_LEN) * h * k,
                                  side * rng.uniform(*MOSS_BRANCH_SLANT), (sway[0] / 2, sway[1] / 2, sway[2]))
                stroke(twig, max(1, width - 1), rgb)

    b = colour.resize(size, Image.Resampling.BOX)
    o = mask.resize(size, Image.Resampling.BOX)
    height = np.asarray(o.filter(ImageFilter.GaussianBlur(MOSS_BLUR)), float) / 255
    gx = np.gradient(height, axis=1)
    g_up = -np.gradient(height, axis=0)                      # row 0 is the top
    n = np.stack([-MOSS_RELIEF * gx, -MOSS_RELIEF * g_up if NORMAL_CONVENTION == "opengl" else MOSS_RELIEF * g_up,
                  np.ones_like(gx)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    nm = Image.fromarray(np.round((n * 0.5 + 0.5) * 255).astype(np.uint8), "RGB")
    return {"b": b, "o": o, "nm": nm}


def bevel_normal(size: int = 256) -> Image.Image:
    """The bevelled block's normal map; see the module docstring."""
    centre = np.arange(size) + 0.5
    x, y = np.meshgrid(centre, centre)                       # y runs down the image
    dist = np.stack([x, size - x, y, size - y])              # to the left, right, top, bottom edge
    edge = dist.argmin(axis=0)
    bevel = dist.min(axis=0) < BEVEL * size
    # the height rises away from the nearest edge; the normal leans against the rise
    gx = np.select([edge == 0, edge == 1], [1.0, -1.0], 0.0)
    gy_up = np.select([edge == 2, edge == 3], [-1.0, 1.0], 0.0)
    nx = np.where(bevel, -gx * BEVEL_SLOPE, 0.0)
    ny = np.where(bevel, -gy_up * BEVEL_SLOPE, 0.0)
    if NORMAL_CONVENTION == "directx":
        ny = -ny
    elif NORMAL_CONVENTION != "opengl":
        raise ValueError(f"NORMAL_CONVENTION is {NORMAL_CONVENTION!r}; it must be 'opengl' or 'directx'")
    n = np.stack([nx, ny, np.ones_like(nx)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return Image.fromarray(np.round((n * 0.5 + 0.5) * 255).astype(np.uint8), "RGB")
