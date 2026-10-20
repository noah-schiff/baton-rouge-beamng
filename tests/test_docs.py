"""Guards on the public text (README.md, docs/repo-submission.md, docs/known-issues.md) and the decisions record.

Ruling 24 of the v0.2 build: the text must say what the dressing stage really builds. Barriers stand on
all 252 decks, but the two 6 m ramp stubs get only a short piece, and piers stand under only the 116
decks that are 2.5 m or more over the ground or the river bed. A claim that both stand on every deck,
or that the piers go "under them" (the 252), must not creep back in."""
import re
from pathlib import Path

import pytest

from brmap import dressing
from brmap.roads import load_roads

REPO = Path(__file__).resolve().parents[1]
PUBLIC = ("README.md", "docs/repo-submission.md", "docs/known-issues.md")
COUNTED = ("README.md", "docs/repo-submission.md")   # the files that state the deck counts
SPEC = "docs/superpowers/specs/2026-10-19-landmarks-street-life-ground-design.md"
ROADS = REPO / "data" / "out" / "roads.json"
DRESSING = REPO / "data" / "out" / "dressing.json"

DECKS, PIER_DECKS = 252, 116
STUBS = ("w40007900", "w112804149")   # the two 6 m ramp stubs: a short piece of barrier at one edge


def prose(name: str) -> str:
    """A file's text on one line, its blockquote marks and line breaks gone, so a wrapped phrase matches."""
    text = (REPO / name).read_text(encoding="utf-8")
    return " ".join(re.sub(r"^\s*>\s?", "", text, flags=re.M).split())


@pytest.mark.parametrize("name", PUBLIC)
def test_public_text_does_not_claim_barriers_and_piers_on_every_deck(name):
    text = prose(name)
    for claim in (r"every (bridge )?deck",                     # "along both edges of every bridge deck"
                  r"the 252 (bridge )?decks,? and piers",      # "the 252 bridge decks, and piers under them"
                  r"decks with concrete barriers and piers"):  # "solid decks with concrete barriers and piers"
        found = re.search(claim, text, re.I)
        assert found is None, f"{name} says {found.group(0)!r}"


@pytest.mark.parametrize("name", COUNTED)
def test_the_deck_counts_are_stated(name):
    text = prose(name)
    assert re.search(rf"\b{DECKS} bridge decks\b", text) or re.search(rf"\b{DECKS} bridges and elevated roads", text)
    assert f"piers under the {PIER_DECKS} decks that stand 2.5 m or more over the ground or the river bed" in text
    assert "the two 6 m ramp stubs get only a short piece" in text


def test_known_issues_name_the_two_stubs_and_the_pier_decks():
    text = prose("docs/known-issues.md")
    assert all(stub in text for stub in STUBS)
    assert f"{PIER_DECKS} of the {DECKS} decks" in text


@pytest.mark.skipif(not (ROADS.exists() and DRESSING.exists()), reason="no data/out/roads.json and dressing.json")
def test_the_counts_in_the_text_are_the_maps():
    roads = load_roads(ROADS)
    decks = [r for r in roads if r.is_bridge]
    assert len(decks) == DECKS
    length = {}
    for bar in dressing.barriers(decks, roads):
        pts = bar["points"]
        length[bar["deck"]] = length.get(bar["deck"], 0.0) + sum(
            ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 for a, b in zip(pts, pts[1:]))
    assert sorted(length) == sorted(dressing.deck_id(d) for d in decks)              # a barrier on all of them ...
    assert sorted(d for d, m in length.items() if m < 5.0) == sorted(STUBS)          # ... but a short piece on two
    assert len({p["deck"] for p in dressing.load_dressing(DRESSING)["piers"]}) == PIER_DECKS


# ------------------------------------------------------------------ the mod type, the decisions

def test_the_mod_type_trade_off_is_recorded():
    text = prose("docs/repo-submission.md")
    assert "core/modmanager.lua" in text and 'type "unknown"' in text
    assert "licence plate" in text and "0.40" in text   # what dropping the plate folder would bring back
    assert "check the type" in text                      # in the v0.2 steps


def test_the_fresh_install_check_really_starts_from_a_fresh_install():
    # 0.40's mod manager keeps the type of an entry it already knows and drops a deleted zip's entry only when
    # the game starts without it: the old zip has to go, and the game has to run once, before the new zip goes in
    text = prose("docs/repo-submission.md")
    assert "updateZIPEntry" in text and "keeps an entry it already knows" in text
    assert "out of the mods folder" in text and "quit it once" in text
    assert "an existing install keeps its type" in text   # also in the guidelines table's row


def test_the_decisions_are_numbered_in_order():
    text = (REPO / "docs" / "decisions-v0.2.md").read_text(encoding="utf-8")
    numbers = [int(n) for n in re.findall(r"^(\d+)\. ", text, re.M)]
    assert numbers == list(range(1, len(numbers) + 1)) and len(numbers) >= 34


def decision(n: int) -> str:
    found = re.search(rf"^{n}\. (.*)$", (REPO / "docs" / "decisions-v0.2.md").read_text(encoding="utf-8"), re.M)
    assert found, f"no decision {n}"
    return " ".join(found.group(1).split())


def test_the_decisions_name_the_whole_fix_wave_and_the_pools_after_task_22h():
    text = prose("docs/decisions-v0.2.md")
    assert "22a to 22h" in text and "22a to 22g" not in text
    assert "61 %" in decision(31) and "69 %" not in text   # 8,797 of the 14,380 low wetland cells, not 22g's 69 %
    assert "centreline" in decision(33) and "lane_cut.through" in decision(33)
    assert "SHOW_PCT" in decision(34) and "218" in decision(34)


@pytest.mark.skipif(not (REPO / SPEC).exists(), reason="the spec is not in the public snapshot")
def test_the_spec_records_the_landmark_road_rule_and_the_final_pools():
    text = prose(SPEC)
    note = "Corrected (Rulings 28, 32, 33; Tasks 22b, 22f, 22h)"
    assert text.count(note) == 2                            # the anti-goal and the landmark rules
    assert text.count("LANE_CLEAR") >= 1 and "lane_cut" in text and "uncut footprint" in text
    assert "234 `WaterBlock`s" not in text and "7 pools in 218" in text and "Tasks 22a, 22g and 22h" in text


def test_known_issues_state_the_pool_edges():
    text = prose("docs/known-issues.md")
    assert "49 cells" in text and "780 m2" in text and "mostly on the Mississippi's bank" in text
    assert "stops in the air" in text and "1.2 km of edges" in text
    assert "version 0.3" in text


def test_the_readme_says_only_the_textures_a_material_binds_are_fetched():
    text = prose("README.md")
    assert "six CC0 texture sets" not in text and "(about 18 MB)" not in text
    assert "the CC0 texture sets a material binds (today only `turf`" in text
    assert "not fetched" in text                              # the others stay in textures.yaml


@pytest.mark.parametrize("name", COUNTED)
def test_the_release_recipe_requires_every_imposter_shipped_and_a_clean_collect(name):
    text = prose(name)
    assert re.search(r"imposters: (N|\d+) pairs shipped, 0 withheld", text)
    assert "no warnings" in text


def test_the_readme_says_what_the_imposter_fingerprint_leaves_out():
    text = prose("README.md")
    assert "hashed by its path only" in text and "game version is not part of it" in text
    assert "after a game update" in text


def test_the_readme_recollect_after_a_game_update_rebuilds_the_zip_before_loading_it():
    """Deleting imposters.json only withholds the old imposters when `level` runs on it: the zip that step 2
    loads must be built after the delete, or it still ships the old imposters, which the game never renders
    again (a shipped imposter is the newer file)."""
    text = prose("README.md")
    found = re.search(r"Collect the imposters again after a game update \((.*?)\)\. The `level` stage logs", text)
    assert found, "README no longer says how to collect the imposters again after a game update"
    recipe = found.group(1)
    delete, rebuild, steps = (recipe.find(part) for part in
                              ("delete `data/out/cdae/imposters.json`", r"-m brmap level`", "steps 2 to 4"))
    assert 0 <= delete < rebuild < steps, recipe
    assert "ships no imposters" in recipe
