"""The repository installs and builds from a fresh clone."""
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEV_ONLY = {"pytest", "responses"}


def _names(specs) -> set[str]:
    return {re.split(r"[<>=!~;\[ ]", s.strip(), maxsplit=1)[0].lower() for s in specs
            if s.strip() and not s.strip().startswith("#")}


def test_pyproject_installs_the_runtime_requirements():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    needed = _names((ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines())
    assert needed - DEV_ONLY <= _names(project.get("dependencies", []))
    assert DEV_ONLY <= _names(project.get("optional-dependencies", {}).get("dev", []))


def test_package_data_is_tracked_not_git_ignored():
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    data = "src/brmap/data/beamng_templates.json"
    ignored = subprocess.run(["git", "check-ignore", "-q", data], cwd=ROOT).returncode == 0
    assert not ignored, f"{data} is git-ignored, so a clone cannot build the level"
