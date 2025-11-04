from os.path import abspath, isabs
from pathlib import Path

import pytest
from tomli_w import dump as toml_dump
from yaml import dump as yaml_dump

from hydromt._utils.path import _make_config_paths_absolute, _make_config_paths_relative
from hydromt.io.readers import _config_read, read_yaml
from hydromt.io.writers import write_yaml
from hydromt.model import Model
from hydromt.model.components.config import ConfigComponent

ABS_PATH = Path(abspath(__name__))


@pytest.fixture
def test_config_dict():
    return {
        "section1": {
            "list": [1, 2, 3],
            "bool": True,
            "str": "test",
            "int": 1,
            "float": 2.3,
        },
        "section2": {
            "path": "config.yml",  # path exists -> Path
            "path1": "config1.yml",  # path does not exist -> str
        },
        # evaluation skipped by default for setup_config
        "setup_config": {
            "path": "config.yaml",
            "float": 2.3,
        },
    }


def test_rejects_non_yaml_format(tmp_path: Path):
    config_file = tmp_path / "config.toml"
    # hydromt just checks the extension, so an empty file is ok
    open(config_file, "w").close()

    with pytest.raises(ValueError, match="Unknown extension"):
        _ = _config_read(config_file, abs_path=True)


def test_config_create_always_reads(tmp_path: Path):
    filename = "myconfig.yaml"
    config_path = tmp_path / filename
    config_data = {"a": 1, "b": 3.14, "c": None, "d": {"e": {"f": True}}}
    write_yaml(config_path, config_data)
    # notice the write mode
    model = Model(root=tmp_path, mode="w")
    config_component = ConfigComponent(
        model, filename=filename, default_template_filename=str(config_path)
    )
    model.add_component("config", config_component)
    config_component.create()
    # we use _data here to avoid initializing it through lazy loading
    assert config_component._data == config_data


def test_config_does_not_read_at_lazy_init(tmp_path: Path):
    filename = "myconfig.yaml"
    config_path = tmp_path / filename
    config_data = {"a": 1, "b": 3.14, "c": None, "d": {"e": {"f": True}}}
    write_yaml(config_path, config_data)
    # notice the write mode
    model = Model(root=tmp_path, mode="w")
    config_component = ConfigComponent(
        model, filename=filename, default_template_filename=str(config_path)
    )
    model.add_component("config", config_component)
    assert config_component.data == {}


def test_raises_on_no_config_template_found(tmp_path: Path):
    model = Model(root=tmp_path, mode="w")
    config_component = ConfigComponent(model)
    model.add_component("config", config_component)
    with pytest.raises(FileNotFoundError, match="No template file was provided"):
        config_component.create()


@pytest.mark.parametrize("extension", ["yaml", "yml"])
def test_loads_from_config_template_yaml(tmp_path: Path, extension, test_config_dict):
    template_name = "default_config_template"
    template_path = tmp_path / f"{template_name}.{extension}"
    with open(template_path, "w") as fp:
        yaml_dump(test_config_dict, fp)

    model = Model(root=tmp_path, mode="w")
    config_component = ConfigComponent(
        model, default_template_filename=str(template_path)
    )
    model.add_component("config", config_component)
    config_component.read()

    assert config_component._data == test_config_dict


def test_loads_from_config_template_toml(tmp_path: Path, test_config_dict):
    template_path = tmp_path / "default_config_template.toml"
    with open(template_path, "wb") as fp:
        toml_dump(test_config_dict, fp)

    model = Model(root=tmp_path, mode="w")
    config_component = ConfigComponent(
        model, default_template_filename=str(template_path)
    )
    config_component.read()

    assert config_component._data == test_config_dict


def test_make_config_abs(tmp_path: Path, test_config_dict):
    p = tmp_path / "config.yml"
    # create file so it will get parsed correctly
    open(p, "w").close()
    test_config_dict["section2"]["path"] = p
    test_config_dict["section2"]["path2"] = abspath(p)
    parsed_config = _make_config_paths_absolute(test_config_dict, tmp_path)
    assert all(
        isabs(p) for p in parsed_config["section2"].values() if isinstance(p, Path)
    ), parsed_config["section2"]


def test_make_rel_abs(tmp_path: Path, test_config_dict):
    p = tmp_path / "config.yml"
    # create file so it will get parsed correctly
    open(p, "w").close()
    test_config_dict["section2"]["path"] = p
    test_config_dict["section2"]["path2"] = abspath(p)
    parsed_config = _make_config_paths_relative(test_config_dict, tmp_path)
    assert all(
        not isabs(p) for p in parsed_config["section2"].values() if isinstance(p, Path)
    ), parsed_config["section2"]


def test_set_config(tmp_path: Path):
    model = Model(root=tmp_path)
    config_component = ConfigComponent(model)
    model.add_component("config", config_component)
    config_component.set("global.name", "test")
    assert config_component._data is not None
    assert "name" in config_component._data["global"]
    assert config_component.get_value("global.name") == "test"


def test_write_config(tmp_path: Path):
    model = Model(root=tmp_path)
    config_component = ConfigComponent(model)
    model.add_component("config", config_component)
    config_component.set("global.name", "test")
    write_path = tmp_path / "config.yaml"
    assert not write_path.exists()
    config_component.write()
    assert write_path.exists()
    read_contents = read_yaml(write_path)
    assert read_contents == {"global": {"name": "test"}}


def test_get_config_abs_path(tmp_path: Path):
    model = Model(root=tmp_path)
    config_component = ConfigComponent(model)
    model.add_component("config", config_component)
    abs_path = tmp_path / "test.file"
    config_component.set("global.file", "test.file")
    assert str(config_component.get_value("global.file")) == "test.file"
    assert config_component.get_value("global.file", abs_path=True) == abs_path
