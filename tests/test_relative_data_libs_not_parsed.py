from pathlib import Path

import pytest

from hydromt.model.components.config import ConfigComponent
from hydromt.model.model import Model
from hydromt.plugins import PLUGINS
from hydromt.readers import read_workflow_yaml


@pytest.fixture
def workflow_yaml(tmp_path: Path) -> tuple[Path, Path, Path]:
    data_lib_rel = "examples/local_sources_relative.yml"
    data_lib_abs = tmp_path / "examples/local_sources_absolute.yml"
    yaml_content = f"""
modeltype: model
global:
  data_libs:
    - {data_lib_rel}
    - {data_lib_abs}
    - artifact_data
  components:
    config:
      type: ConfigComponent
      filename: run_config.toml
steps:
  - config.create:
      template: run_config.toml
  - config.update:
      data:
        starttime: 2010-01-01
        model.type: model
  - write:
      components:
        - config
"""
    yaml_path = tmp_path / "workflow.yml"
    yaml_path.write_text(yaml_content)
    for dc in [tmp_path / data_lib_rel, data_lib_abs]:
        dc.parent.mkdir(parents=True, exist_ok=True)
        dc.write_text("dummy: content")

    return yaml_path, data_lib_abs, data_lib_rel, tmp_path


def test_read_workflow_yaml(workflow_yaml: tuple[Path, Path, Path, Path]):
    yaml_path, data_lib_abs, data_lib_rel, tmp_path = workflow_yaml

    modeltype, model_init, steps = read_workflow_yaml(
        yaml_path, skip_abspath_sections=None
    )
    assert modeltype == Model.__name__
    assert data_lib_abs in model_init["data_libs"]
    assert yaml_path.parent / data_lib_rel in model_init["data_libs"]
    assert "artifact_data" in model_init["data_libs"]
    assert len(steps) == 3


def test_missing_modeltype(tmp_path: Path):
    yaml_path = tmp_path / "workflow.yml"
    yaml_path.write_text("""
global:
  data_libs: []
steps: []
""")
    with pytest.raises(ValueError, match="Model type not specified"):
        read_workflow_yaml(yaml_path)


def test_empty_steps(tmp_path: Path):
    yaml_path = tmp_path / "workflow.yml"
    yaml_path.write_text("""
modeltype: model
global:
  data_libs: []
""")
    # Should return empty list of steps without error
    modeltype, model_init, steps = read_workflow_yaml(yaml_path)
    assert steps == []


def test_absolute_and_relative_paths(tmp_path: Path):
    yaml_path = tmp_path / "workflow.yml"
    rel_path = "data/file.yml"
    abs_path = tmp_path / "absolute_file.yml"
    abs_path.write_text("content")
    (tmp_path / "data").mkdir()
    (tmp_path / rel_path).write_text("content")
    yaml_path.write_text(f"""
modeltype: model
global:
  data_libs:
    - {rel_path}
    - {abs_path.as_posix()}
steps: []
""")
    modeltype, model_init, steps = read_workflow_yaml(yaml_path)
    # Relative path resolved to absolute
    assert any(
        isinstance(x, Path) and x.is_absolute() and x.name == "file.yml"
        for x in model_init["data_libs"]
    )
    # Absolute path stays absolute
    assert any(
        isinstance(x, Path) and x.is_absolute() and x.name == "absolute_file.yml"
        for x in model_init["data_libs"]
    )


def test_predefined_catalogs(tmp_path: Path):
    yaml_path = tmp_path / "workflow.yml"
    catalog_name = next(iter(PLUGINS.catalog_plugins))
    yaml_path.write_text(f"""
modeltype: model
global:
  data_libs:
    - {catalog_name}
steps: []
""")
    modeltype, model_init, steps = read_workflow_yaml(yaml_path)
    # Should keep catalog string as-is
    assert any(
        isinstance(x, str) and x == catalog_name for x in model_init["data_libs"]
    )


def test_component_dict_syntax(tmp_path: Path):
    yaml_path = tmp_path / "workflow.yml"
    yaml_path.write_text("""
modeltype: model
global:
  components:
    config:
      type: ConfigComponent
      filename: run_config.toml
steps: []
""")
    modeltype, model_init, steps = read_workflow_yaml(yaml_path)
    # Component should be a list of HydromtComponentConfig-like dicts
    comp = model_init["components"][0]
    assert comp["type"] == ConfigComponent
    assert comp["name"] == "config"
    assert comp["filename"] == "run_config.toml"


def test_invalid_step_name(tmp_path: Path):
    yaml_path = tmp_path / "workflow.yml"
    yaml_path.write_text("""
modeltype: model
global:
  components:
    config:
      type: ConfigComponent
steps:
  - invalid.step.name.too.long: {}
""")
    with pytest.raises(ValueError, match="Invalid step name"):
        read_workflow_yaml(yaml_path)


def test_defaults_merging(workflow_yaml: tuple[Path, Path, Path, Path]):
    yaml_path, data_lib_abs, data_lib_rel, tmp_path = workflow_yaml

    defaults = {"global": {"new_option": 42}}
    _, model_init, _ = read_workflow_yaml(
        yaml_path, defaults=defaults, skip_abspath_sections=["global"]
    )
    assert model_init.get("new_option") == 42
    # Ensure original data_libs preserved
    assert "artifact_data" in model_init["data_libs"]


def test_abs_path_behavior(workflow_yaml: tuple[Path, Path, Path, Path]):
    yaml_path, data_lib_abs, data_lib_rel, tmp_path = workflow_yaml

    # abs_path=True should resolve relative paths
    _, model_init, _ = read_workflow_yaml(
        yaml_path, abs_path=True, skip_abspath_sections=None
    )
    resolved_paths = [p for p in model_init["data_libs"] if isinstance(p, Path)]
    assert any(p.is_absolute() and p.exists() for p in resolved_paths)
    # The catalog string should remain a string
    assert "artifact_data" in model_init["data_libs"]

    # abs_path=False should keep relative paths
    _, model_init_rel, _ = read_workflow_yaml(
        yaml_path, abs_path=False, skip_abspath_sections=None
    )
    assert any(
        isinstance(p, Path) and not p.is_absolute()
        for p in model_init_rel["data_libs"]
        if p != "artifact_data"
    )


def test_skip_abspath_sections(workflow_yaml: tuple[Path, Path, Path, Path]):
    yaml_path, data_lib_abs, data_lib_rel, tmp_path = workflow_yaml

    # Skip global section, relative paths should remain relative
    _, model_init, _ = read_workflow_yaml(
        yaml_path, abs_path=True, skip_abspath_sections=["global"]
    )
    data_libs = model_init["data_libs"]
    rel_paths = [p for p in data_libs if isinstance(p, Path) and not p.is_absolute()]
    assert any(rel_paths)
    # Catalog string still preserved
    assert "artifact_data" in data_libs
