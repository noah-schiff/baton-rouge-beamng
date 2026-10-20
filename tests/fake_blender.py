"""Stands in for blender.exe in tests/test_models.py, so the runner's logic can be tested without Blender.

It takes the arguments models.build passes after the prefix (`--python <script> -- --params <json>
--out <dir>`) and, by the params read from the json file:

* writes one-triangle shape parts in the npz contract to <dir>, one for each name in "parts"
  (default ["main"]), or none when "empty" is true;
* appends a line to <dir>/count.txt: the script path and the params it received, as json;
* with "fail" true, prints 30 numbered lines and "fake blender: boom" to stderr and exits 1.

`--version` prints a two-line version text like Blender's. Only numpy and the standard library
are used, since it runs as a separate process.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def write_part(path: Path) -> None:
    """One triangle in node x_a100, as Blender writes it (the npz contract)."""
    pos = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    np.savez_compressed(
        path, **{"meta/nodes": np.array(["x_a100"]), "meta/billboard": np.array(""),
                 "meta/blender": np.array("fake"), "meta/script_sha1": np.array("0" * 40),
                 "x_a100/pos": pos, "x_a100/nrm": np.tile([0.0, 0.0, 1.0], (3, 1)),
                 "x_a100/uv": pos[:, :2], "x_a100/col": np.ones((3, 4)),
                 "x_a100/mat": np.zeros(1, np.int32), "x_a100/materials": np.array(["br_x"])})


def main(argv: list[str]) -> int:
    if "--version" in argv:
        print("Blender 9.9.9 (fake)\n\tbuild date: never")
        return 0
    script = argv[argv.index("--python") + 1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True)
    parser.add_argument("--out", required=True)
    ns = parser.parse_args(argv[argv.index("--") + 1:])
    params = json.loads(Path(ns.params).read_text(encoding="utf-8"))
    out = Path(ns.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "count.txt").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"script": script, "params": params}) + "\n")
    if params.get("fail"):
        for i in range(30):
            print(f"noise {i}", file=sys.stderr)
        print("fake blender: boom", file=sys.stderr)
        return 1
    if not params.get("empty"):
        for part in params.get("parts", ["main"]):
            write_part(out / f"{part}.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
