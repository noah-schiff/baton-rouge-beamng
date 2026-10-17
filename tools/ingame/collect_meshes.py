"""Collect the building meshes the game compiled (.cdae) so the next level build ships them.

Run after loading the built level once in BeamNG.drive (the first load compiles every .dae
into the user folder's temp/levels/<level>/), before rebuilding the level:

    .venv\\Scripts\\python tools\\ingame\\collect_meshes.py [config.yaml]

Then `python -m brmap level` puts each .cdae beside its unchanged .dae (see brmap.meshcache).
"""
import os
import sys
from pathlib import Path

from brmap.config import load_config
from brmap.meshcache import collect


def main(config="config.yaml") -> int:
    cfg = load_config(config)
    level_root = Path(cfg.dist_dir) / "level" / "levels" / cfg.level_name
    temp = Path(os.environ["LOCALAPPDATA"]) / "BeamNG" / "BeamNG.drive" / "current" / "temp" / "levels" / cfg.level_name
    got, missing = collect(level_root, temp, Path(cfg.out_dir) / "cdae")
    print(f"collected {got} compiled meshes into {Path(cfg.out_dir) / 'cdae'}; {len(missing)} not compiled from this build")
    for rel in missing[:10]:
        print("  not compiled:", rel)
    return 0 if got and not missing else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
