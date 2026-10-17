"""Precompiled building meshes (.cdae): collected from the game's cache, reused while the .dae is unchanged."""
import os
import time

from brmap import meshcache


def _write(path, data: bytes, mtime=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def test_collect_takes_only_meshes_compiled_from_this_build(tmp_path):
    level, temp, cache = tmp_path / "level", tmp_path / "temp", tmp_path / "cache"
    now = time.time()
    _write(level / "art/shapes/buildings/a.dae", b"<a/>", now - 100)
    _write(level / "art/shapes/buildings/b.dae", b"<b/>", now - 100)
    _write(level / "art/shapes/buildings/c.dae", b"<c/>", now - 100)
    _write(temp / "art/shapes/buildings/a.cdae", b"A", now - 50)    # compiled after the build: fresh
    _write(temp / "art/shapes/buildings/b.cdae", b"B", now - 200)   # compiled before it: stale
    got, missing = meshcache.collect(level, temp, cache)
    assert got == 1 and missing == ["art/shapes/buildings/b.dae", "art/shapes/buildings/c.dae"]
    assert (cache / "art/shapes/buildings/a.cdae").read_bytes() == b"A"
    assert not (cache / "art/shapes/buildings/b.cdae").exists()


def test_apply_reuses_only_while_the_dae_is_unchanged(tmp_path):
    level, temp, cache = tmp_path / "level", tmp_path / "temp", tmp_path / "cache"
    now = time.time()
    _write(level / "art/shapes/buildings/a.dae", b"<a/>", now - 100)
    _write(level / "art/shapes/buildings/b.dae", b"<b/>", now - 100)
    _write(temp / "art/shapes/buildings/a.cdae", b"A", now - 50)
    _write(temp / "art/shapes/buildings/b.cdae", b"B", now - 50)
    assert meshcache.collect(level, temp, cache) == (2, [])
    # a rebuild writes the same a.dae again (newer, same bytes) and a changed b.dae
    _write(level / "art/shapes/buildings/a.dae", b"<a/>")
    _write(level / "art/shapes/buildings/b.dae", b"<b changed/>")
    used, stale = meshcache.apply(level, cache)
    assert (used, stale) == (1, ["art/shapes/buildings/b.dae"])
    a = level / "art/shapes/buildings/a.cdae"
    assert a.read_bytes() == b"A"
    assert a.stat().st_mtime >= (level / "art/shapes/buildings/a.dae").stat().st_mtime  # the game must see it as newer
    assert not (level / "art/shapes/buildings/b.cdae").exists()


def test_collect_keeps_cached_meshes_still_valid(tmp_path):
    # after a rebuild that changed one tile, the game loads the shipped .cdae for the rest and
    # compiles only the changed one: collecting again must keep the rest, not forget them
    level, temp, cache = tmp_path / "level", tmp_path / "temp", tmp_path / "cache"
    now = time.time()
    _write(level / "a.dae", b"<a/>", now - 100)
    _write(level / "b.dae", b"<b/>", now - 100)
    _write(temp / "a.cdae", b"A", now - 50)
    _write(temp / "b.cdae", b"B", now - 50)
    assert meshcache.collect(level, temp, cache) == (2, [])
    (temp / "a.cdae").unlink()
    (temp / "b.cdae").unlink()
    _write(level / "a.dae", b"<a/>", now - 10)          # rewritten, same bytes
    _write(level / "b.dae", b"<b changed/>", now - 10)  # changed
    _write(temp / "b.cdae", b"B2", now)                 # only b compiled again
    assert meshcache.collect(level, temp, cache) == (2, [])
    assert (cache / "a.cdae").read_bytes() == b"A" and (cache / "b.cdae").read_bytes() == b"B2"
    assert meshcache.apply(level, cache) == (2, [])


def test_apply_without_a_cache_does_nothing(tmp_path):
    _write(tmp_path / "level/x.dae", b"<x/>")
    assert meshcache.apply(tmp_path / "level", tmp_path / "nothing") == (0, ["x.dae"])
