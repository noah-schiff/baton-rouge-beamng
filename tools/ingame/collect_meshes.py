"""Collect the meshes the game compiled (.cdae) and the tree imposters it made, so the next level build ships them.

Run after loading the built level once in BeamNG.drive (the first load compiles every .dae
into the user folder's temp/levels/<level>/), before rebuilding the level:

    .venv\\Scripts\\python tools\\ingame\\collect_meshes.py [config.yaml]

Then `python -m brmap level` puts each .cdae, and the imposters made for the same shape,
beside its unchanged .dae (see brmap.meshcache). It prints how many shapes' imposter pairs the next
level build can ship and warns for each imposter tree (a shape with an .asset.json) it has none for.
"""
import os
import sys
from pathlib import Path

from brmap.config import load_config
from brmap.meshcache import collect, imposter_status


def main(config="config.yaml") -> int:
    cfg = load_config(config)
    level_root = Path(cfg.dist_dir) / "level" / "levels" / cfg.level_name
    temp = Path(os.environ["LOCALAPPDATA"]) / "BeamNG" / "BeamNG.drive" / "current" / "temp" / "levels" / cfg.level_name
    imposter_root = temp  # until the Task 9 spike finds where the game writes imposters, beside the .cdae
    cache = Path(cfg.out_dir) / "cdae"
    got, missing = collect(level_root, temp, cache, imposter_root=imposter_root)
    ready, lacking = imposter_status(level_root, cache)
    print(f"collected {got} compiled meshes and {len(ready)} imposter pairs into {cache}; "
          f"{len(missing)} not compiled from this build")
    for rel in missing[:10]:
        print("  not compiled:", rel)
    for rel in lacking:
        print(f"  warning: {rel} has no cached imposters for this build (the game renders them at load; "
              "load the level once more, then collect again)")
    return 0 if got and not missing else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
