"""Command line: python -m brmap <stage> [--config config.yaml]."""
from __future__ import annotations

import argparse
import importlib
import logging
import sys
import time
from pathlib import Path

from .config import load_config

FEATURE_STAGES = ["features", "canopy", "textures", "landmarks", "models", "bayous", "buildings", "trees",
                  "markings", "dressing"]
STAGES = ["fetch", "terrain", "water", "imagery", "roads", *FEATURE_STAGES, "level", "validate"]
# terrain runs twice: once raw, then again to carve the water bodies into it.
ALL_ORDER = ["fetch", "terrain", "water", "terrain", "imagery", "roads", *FEATURE_STAGES, "level",
             "validate"]
LOG_DIR = Path("data/logs")
FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(stage: str, log_dir: Path = LOG_DIR) -> logging.Logger:
    """Log to the console and to data/logs/<stage>.log (appending)."""
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("brmap")
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(FORMAT, "%H:%M:%S"))
    file = logging.FileHandler(log_dir / f"{stage}.log", encoding="utf-8")
    file.setFormatter(logging.Formatter(FORMAT))
    root.addHandler(console)
    root.addHandler(file)
    return root


def run_stage(stage: str, cfg) -> None:
    importlib.import_module(f"brmap.{stage}").run(cfg)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="brmap", description=__doc__)
    p.add_argument("stage", choices=STAGES + ["all"])
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    order = ALL_ORDER if args.stage == "all" else [args.stage]
    for stage in order:
        log = setup_logging(stage)
        start = time.perf_counter()
        log.info("stage %s: start", stage)
        run_stage(stage, cfg)
        log.info("stage %s: done in %.1f s", stage, time.perf_counter() - start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
