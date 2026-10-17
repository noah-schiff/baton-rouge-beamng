import pytest

from brmap import __main__ as cli


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
                             "canopy", "buildings", "trees", "markings", "level", "validate"]
    assert set(cli.ALL_ORDER) == set(cli.STAGES)


@pytest.mark.parametrize("stage", ["features", "canopy", "buildings", "trees", "markings"])
def test_feature_stage_is_a_module_with_run(stage):
    import importlib
    import inspect

    mod = importlib.import_module(f"brmap.{stage}")
    assert mod.__doc__  # the module docstring states its contract
    assert list(inspect.signature(mod.run).parameters) == ["cfg", "raw_dir", "out_dir"]
