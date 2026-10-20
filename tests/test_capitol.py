"""The Louisiana State Capitol's model script (blender/landmarks/capitol.py): its plain-Python geometry
on the real params, built without Blender."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from landmark_scripts import check_parts, close, fixture_params, load_script

import brlm  # noqa: E402  (landmark_scripts puts blender/lib on the path)

PARAMS = fixture_params("capitol")   # landmarks.resolve on the full map (data/out), 2026-10-19
LODS = (600, 150, 30)
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def capitol():
    return load_script("capitol")


@pytest.fixture(scope="module")
def dims(capitol):
    return capitol.resolve(PARAMS)


@pytest.fixture(scope="module")
def built(capitol, dims):
    return capitol.parts(dims)


def corners_of(b):
    return [p for pts, _, _ in b.faces for p in pts]


def test_parts_and_levels(built):
    assert set(built) == {"main"}
    check_parts(built, LODS)


def test_every_level_reaches_the_cited_top(dims, built):
    top = PARAMS["ground_dz"] + PARAMS["height_m"]
    for px, _, b in built["main"]["lods"]:
        assert max(p[2] for p in corners_of(b)) == pytest.approx(top, abs=0.01), px


def test_the_tower_rises_through_the_setbacks(capitol, dims):
    g = PARAMS["ground_dz"]
    sections = dims["sections"]
    assert sections[0][0] == pytest.approx(g + PARAMS["centre_m"])          # from the centre block's roof
    for (_, top, _, _), (bottom, _, _, _) in zip(sections, sections[1:]):
        assert top == pytest.approx(bottom)                                   # stacked
    for start, w, d in PARAMS["setbacks"]:
        (match,) = [s for s in sections if s[0] == pytest.approx(g + start)]
        assert (match[2], match[3]) == (w, d)
    widths = [w for _, _, w, _ in sections]
    assert widths == sorted(widths, reverse=True) and widths[0] > PARAMS["setbacks"][0][1]   # a wider base
    z0, z1, r = dims["lantern"]
    assert z0 == pytest.approx(sections[-1][1]) and z1 < g + PARAMS["height_m"] and 2 * r <= sections[-1][2]


def test_the_collision_mesh_has_the_tower_boxes(dims, built):
    col = built["main"]["collision"][1]
    cu, cv = dims["tower_at"]
    pts = corners_of(col)
    for z0, z1, w, d in dims["sections"]:
        top = [p for p in pts if abs(p[2] - z1) < 1e-6
               and abs(p[0] - cu) <= w / 2 + 1e-6 and abs(p[1] - cv) <= d / 2 + 1e-6]
        assert max(abs(p[0] - cu) for p in top) == pytest.approx(w / 2), (z0, w)
        assert max(abs(p[1] - cv) for p in top) == pytest.approx(d / 2), (z0, d)


def test_wings_and_centre_block(dims, built):
    g = PARAMS["ground_dz"]
    low = built["main"]["lods"][-1][2]
    wings = [p for p in corners_of(low) if abs(p[0]) > 30.0 and p[1] > -20.0]
    assert max(p[2] for p in wings) == pytest.approx(g + PARAMS["wings_m"])
    x0, y0, x1, y1 = dims["centre_block"]
    centre = [p for p in corners_of(low) if x0 - 1e-6 <= p[0] <= x1 + 1e-6 and y0 - 1e-6 <= p[1] <= y1 + 1e-6
              and abs(p[0] - dims["tower_at"][0]) > 15.0]
    assert max(p[2] for p in centre) == pytest.approx(g + PARAMS["centre_m"])


def test_the_lantern_is_octagonal(dims, built):
    z0, z1, _ = dims["lantern"]
    for px, _, b in built["main"]["lods"][:2]:
        sides = sorted({round(math.degrees(math.atan2(n[1], n[0])) % 360.0, 1)
                        for pts, _, _ in b.faces if all(z0 - 1e-6 <= p[2] <= z1 + 1e-6 for p in pts)
                        for n in [brlm.normal(pts)] if abs(n[2]) < 1e-6})
        assert len(sides) == 8, (px, sides)
        assert all(q - p == pytest.approx(45.0, abs=0.2) for p, q in zip(sides, sides[1:])), px


def risers(capitol, dims, b):
    """The steps' risers: limestone faces facing -v (south), south of the facade, no higher than the
    landing, in the flight's width."""
    s = dims["steps"]
    out = []
    for pts, _, mat in b.faces:
        n = brlm.normal(pts)
        if mat == capitol.LIMESTONE and n[1] < -0.999 and max(p[1] for p in pts) < s["v_top"] - 1e-6 \
                and max(p[2] for p in pts) <= s["z_top"] + 1e-6 \
                and all(abs(p[0] - s["u"]) <= s["width"] / 2 + 1e-6 for p in pts):
            out.append(pts)
    return out


def test_the_grand_steps_face_south(capitol, dims, built):
    s = dims["steps"]
    assert s["width"] == 30.0 and capitol.STEPS == 48
    assert s["v_top"] == pytest.approx(min(v for u, v in PARAMS["footprint_local"] if abs(u - s["u"]) <= 15.0))
    b = built["main"]["lods"][0][2]
    rs = risers(capitol, dims, b)
    assert len(rs) == capitol.STEPS
    spans = {(round(min(p[0] for p in r), 3), round(max(p[0] for p in r), 3)) for r in rs}
    assert spans == {(round(s["u"] - 15.0, 3), round(s["u"] + 15.0, 3))}
    tops = sorted((max(p[1] for p in r), max(p[2] for p in r)) for r in rs)   # from the south up to the facade
    assert all(z1 > z0 for (_, z0), (_, z1) in zip(tops, tops[1:]))
    assert len(risers(capitol, dims, built["main"]["lods"][1][2])) >= 1     # the mid level keeps a flight
    col = corners_of(built["main"]["collision"][1])                          # a ramp to drive up, the pedestals
    assert min(p[1] for p in col) == pytest.approx(s["v_foot"])
    assert sum(abs(p[2] - (s["z_foot"] + capitol.PEDESTAL[2])) < 1e-6 for p in col) >= 2 * 4


def test_window_bands_are_dark(capitol, built):
    # glass with a spandrel at each floor, close up; one shade of both further off
    high, mid = built["main"]["lods"][0][2], built["main"]["lods"][1][2]
    assert any(close(c, capitol.GLASS) for c in high.cols) and any(close(c, capitol.SPANDREL) for c in high.cols)
    assert any(close(c, capitol.BAND) for c in mid.cols)
    assert max(capitol.GLASS) < max(capitol.BAND) < max(capitol.SPANDREL) < 0.6


def test_ribs_stand_out_of_the_tower(capitol, dims, built):
    b = built["main"]["lods"][0][2]
    cu, cv = dims["tower_at"]
    z0, z1, w, d = dims["sections"][1]
    out = [p for p in corners_of(b) if z0 - 1e-6 <= p[2] <= z1 + 1e-6 and abs(p[1] - cv) < d / 2 - 1
           and abs(p[0] - cu) <= w / 2 + 1]                                 # the shaft, not the base's roof
    assert max(abs(p[0] - cu) for p in out) == pytest.approx(w / 2 + capitol.RIB_OUT)


REAL = ROOT / "data" / "out" / "heights.npy"
REAL_OSM = ROOT / "data" / "raw" / "osm_features.json"


@pytest.mark.skipif(not (REAL.exists() and REAL_OSM.exists()), reason="no full-map terrain (data/out/heights.npy)")
def test_the_steps_stand_on_the_real_terrain(capitol):
    # the DEM rises with the real steps; every tread corner stands over the full map's terrain
    from brmap import landmarks
    from brmap.config import load_config
    from brmap.geo import Extent
    from brmap.terrain import load_heights, sample_height

    extent = Extent.from_config(load_config(ROOT / "config.yaml"))
    reg = {"capitol": landmarks.load_registry()["capitol"]}
    (lm,) = landmarks.resolve(reg, json.loads(REAL_OSM.read_text(encoding="utf-8")), extent,
                              load_heights(out_dir=ROOT / "data" / "out"))
    p = capitol.resolve(lm["params"])
    b = brlm.Builder()
    capitol.steps(b, p, 600)
    heights = load_heights(out_dir=ROOT / "data" / "out")
    c, s = math.cos(lm["yaw"]), math.sin(lm["yaw"])
    treads = [pts for pts, _, _ in b.faces if brlm.normal(pts)[2] > 0.999]
    assert len(treads) >= capitol.STEPS
    gap = []
    for pts in treads:
        for u, v, z in pts:
            x, y = lm["x"] + u * c - v * s, lm["y"] + u * s + v * c
            gap.append(lm["base_z"] + z - float(sample_height(heights, extent, x, y)))
    assert min(gap) >= 0.05
    assert np.median(gap) <= 0.8                                            # and not far above it
