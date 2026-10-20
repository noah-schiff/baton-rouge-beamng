"""The Horace Wilkinson Bridge truss's model script (blender/landmarks/wilkinson_truss.py): its
plain-Python geometry on the real params, built without Blender."""
import numpy as np
import pytest

from landmark_scripts import check_parts, fixture_params, levels, load_script

PARAMS = fixture_params("wilkinson_truss")   # landmarks.resolve on the full map (data/out), 2026-10-19
LODS = (600, 120, 30)


@pytest.fixture(scope="module")
def truss():
    return load_script("wilkinson_truss")


@pytest.fixture(scope="module")
def dims(truss):
    return truss.resolve(PARAMS)


@pytest.fixture(scope="module")
def built(truss, dims):
    return truss.parts(dims)


def interp(u, us, values):
    """values at u along the nodes' us, carried on straight past the first and the last node (the deck
    nodes stop short of the truss's ends)."""
    scalar, u = np.ndim(u) == 0, np.atleast_1d(np.asarray(u, float))
    out = np.interp(u, us, values)
    before, after = u < us[0], u > us[-1]
    out[before] = values[0] + (u[before] - us[0]) * (values[1] - values[0]) / (us[1] - us[0])
    out[after] = values[-1] + (u[after] - us[-1]) * (values[-1] - values[-2]) / (us[-1] - us[-2])
    return out[0] if scalar else out


def deck_at(u, which="max"):
    """The highest (or lowest) of the decks' z at u, by their nodes in model coordinates."""
    zs = []
    for d in PARAMS["deck_local"]:
        n = np.asarray(d["nodes"], float)
        zs.append(interp(u, n[:, 0], n[:, 2]))
    return np.max(zs, axis=0) if which == "max" else np.min(zs, axis=0)


def edges_at(u):
    """(the combined decks' edge on -v, on +v) at u."""
    lo, hi = [], []
    for d in PARAMS["deck_local"]:
        n = np.asarray(d["nodes"], float)
        v = interp(u, n[:, 0], n[:, 1])
        lo.append(v - d["width"] / 2)
        hi.append(v + d["width"] / 2)
    return np.min(lo, axis=0), np.max(hi, axis=0)


def test_parts_and_levels(built):
    assert set(built) == {"main"}
    check_parts(built, LODS)


def test_the_truss_spans_its_ends_with_the_peaks_on_panel_points(dims):
    u = np.asarray(dims["panels"])
    lo, hi = sorted((PARAMS["from_u"], PARAMS["to_u"]))
    assert (u[0], u[-1]) == (pytest.approx(lo), pytest.approx(hi))
    for peak in PARAMS["peaks_u"]:
        assert np.min(np.abs(u - peak)) < 1e-6
    step = np.diff(u)
    assert step.min() > 0.8 * PARAMS["panel_m"] and step.max() < 1.25 * PARAMS["panel_m"]


def test_planes_stand_just_outside_the_decks(dims):
    u = np.asarray(dims["panels"])
    low, high = edges_at(u)
    for side, edge, sign in (("r", low, -1), ("l", high, 1)):
        v = np.array([p[1] for p in dims["planes"][side]])
        gap = sign * (v - edge)
        assert gap.min() > 0.5 and gap.max() < 2.0, side         # clear of the deck, just outside it


def test_top_chord_rises_from_the_ends_to_the_peaks(dims):
    u = np.asarray(dims["panels"])
    top = np.array([p[3] for p in dims["planes"]["l"]])
    above = top - deck_at(u)
    low, high = PARAMS["top_m"]
    assert above[0] == pytest.approx(low, abs=0.01) and above[-1] == pytest.approx(low, abs=0.01)
    for peak in PARAMS["peaks_u"]:
        assert above[np.argmin(np.abs(u - peak))] == pytest.approx(high, abs=0.01)
    assert above.min() == pytest.approx(low, abs=0.01) and above.max() == pytest.approx(high, abs=0.01)


def test_bottom_chords_follow_their_decks(dims):
    u = np.asarray(dims["panels"])
    for side in ("l", "r"):
        z = np.array([p[2] for p in dims["planes"][side]])
        assert np.abs(z - deck_at(u, "min")).max() < 2.5, side


def edge_points(b, n=21):
    """Points every 1/(n - 1) along every edge of every face: the members span the roadway with no
    corner over it."""
    t = np.linspace(0.0, 1.0, n)[:, None]
    out = []
    for pts, _, _ in b.faces:
        p = np.asarray(pts, float)
        for a, c in zip(p, np.roll(p, -1, axis=0)):
            out.append(a + t * (c - a))
    return np.vstack(out)


def test_overhead_bracing_clears_the_deck(built):
    # over the roadway (between the decks' outer edges) nothing is lower than clear_m above the deck,
    # except what stands under it (the piers)
    for name, b in levels(built["main"]):
        if name == "Colmesh-1":
            continue
        c = edge_points(b)
        low, high = edges_at(c[:, 0])
        over = (c[:, 1] > low + 0.05) & (c[:, 1] < high - 0.05) & (c[:, 2] > deck_at(c[:, 0], "min") - 0.5)
        assert over.any() or name == "a30", name                     # the far level has nothing overhead
        assert (c[over, 2] >= deck_at(c[over, 0]) + PARAMS["clear_m"] - 1e-6).all(), name


def test_collision_is_the_planes_and_the_piers(built):
    c = edge_points(built["main"]["collision"][1])
    low, high = edges_at(c[:, 0])
    over = (c[:, 1] > low) & (c[:, 1] < high)
    assert (c[over, 2] < deck_at(c[over, 0], "min")).all()        # nothing over the roadway: piers only
    assert c[:, 2].min() <= 0.0                                     # down to the river bed


def test_piers_under_the_peaks_and_at_the_ends(dims):
    centres = sorted(p[0] for p in dims["piers"])
    lo, hi = sorted((PARAMS["from_u"], PARAMS["to_u"]))
    want = sorted([lo, hi, *PARAMS["peaks_u"]])
    assert len(centres) == 4 and np.allclose(centres, want, atol=3.0)
    for u, _, v0, v1, z0, z1 in dims["piers"]:
        assert z0 <= 0.0 and z1 < deck_at(u, "min")
        low, high = edges_at(u)
        assert v0 < low and v1 > high                                 # under both planes


def test_painted_steel_and_concrete_piers(truss, built):
    for name, b in levels(built["main"]):
        assert truss.STEEL in b.materials and b.materials <= {truss.STEEL, truss.CONCRETE}, name


def test_deck_depth_matches_the_deck_meshroads(truss):
    from brmap import beamng_format as bf

    assert truss.DECK_DEPTH == bf.DECK_DEPTH    # the script can't import brmap: the two must agree
