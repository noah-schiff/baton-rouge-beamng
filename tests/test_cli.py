import re
from pathlib import Path

import pytest

from brmap import __main__ as cli

README = Path(__file__).resolve().parents[1] / "README.md"


def test_unknown_stage_rejected():
    with pytest.raises(SystemExit):
        cli.main(["nonsense"])


def test_stage_dispatches_to_module_run(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(cli, "run_stage", lambda stage, cfg: calls.append(stage))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")
    assert cli.main(["terrain"]) == 0
    assert calls == ["terrain"]
    assert (tmp_path / "data" / "logs" / "terrain.log").exists()


def test_all_runs_stages_in_pipeline_order(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(cli, "run_stage", lambda stage, cfg: calls.append(stage))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")
    cli.main(["all"])
    assert calls == cli.ALL_ORDER


def test_all_order_has_feature_stages():
    assert cli.ALL_ORDER == ["fetch", "terrain", "water", "terrain", "imagery", "roads", "features",
                             "canopy", "textures", "landmarks", "models", "bayous", "buildings", "trees", "markings",
                             "dressing", "level", "validate"]
    assert set(cli.ALL_ORDER) == set(cli.STAGES)


@pytest.mark.parametrize("stage", ["features", "canopy", "textures", "landmarks", "bayous", "buildings", "trees",
                                   "markings", "dressing"])
def test_feature_stage_is_a_module_with_run(stage):
    import importlib
    import inspect

    mod = importlib.import_module(f"brmap.{stage}")
    assert mod.__doc__  # the module docstring states its contract
    assert list(inspect.signature(mod.run).parameters) == ["cfg", "raw_dir", "out_dir"]


def test_readme_lists_every_stage_in_order():
    readme = README.read_text(encoding="utf-8")
    stages = readme.split("\n## Stages", 1)[1].split("\n## ", 1)[0]
    assert re.findall(r"^\| `(\w+)` \|", stages, re.M) == cli.STAGES
    loop = re.search(r"for %s in \(([^)]*)\) do", readme)
    assert loop.group(1).split() == cli.ALL_ORDER[1:]  # every stage but fetch, on the cached downloads
