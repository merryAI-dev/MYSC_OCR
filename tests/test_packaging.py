import tomllib
from pathlib import Path


def test_pyproject_limits_package_discovery_to_settlement_tool():
    pyproject = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())

    find_config = pyproject["tool"]["setuptools"]["packages"]["find"]

    assert find_config["include"] == ["settlement_tool*"]
    assert "models*" in find_config["exclude"]
    assert "output*" in find_config["exclude"]
