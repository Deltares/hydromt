from os.path import abspath, isabs, isfile, join
from pathlib import Path

import pytest
from tomli_w import dump as toml_dump
from yaml import dump as yaml_dump

from hydromt._io.readers import _config_read, _read_yaml
from hydromt._io.writers import _write_yaml
from hydromt._utils.path import _make_config_paths_abs, _make_config_paths_relative
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


def test_rejects_non_yaml_format(tmpdir):
    config_file = tmpdir.join("config.toml")
    # hydromt just checks the extension, so an empty file is ok
    open(config_file, "w").close()

    with pytest.raises(ValueError, match="Unknown extension"):
        _ = _config_read(config_file, abs_path=True)


def test_config_create_always_reads(tmpdir):
    filename = "myconfig.yaml"
    config_path = join(tmpdir, filename)
    config_data = {"a": 1, "b": 3.14, "c": None, "d": {"e": {"f": True}}}
    _write_yaml(config_path, config_data)
    # notice the write mode
    model = Model(root=tmpdir, mode="w")
    config_component = ConfigComponent(
        model, filename=filename, default_template_filename=config_path
    )
    model.add_component("config", config_component)
    config_component.create()
    # we use _data here to avoid initializing it through lazy loading
    assert config_component._data == config_data


def test_config_does_not_read_at_lazy_init(tmpdir):
    filename = "myconfig.yaml"
    config_path = join(tmpdir, filename)
    config_data = {"a": 1, "b": 3.14, "c": None, "d": {"e": {"f": True}}}
    _write_yaml(config_path, config_data)
    # notice the write mode
    model = Model(root=tmpdir, mode="w")
    config_component = ConfigComponent(
        model, filename=filename, default_template_filename=config_path
    )
    model.add_component("config", config_component)
    assert config_component.data == {}


def test_raises_on_no_config_template_found(tmpdir):
    model = Model(root=tmpdir, mode="w")
    config_component = ConfigComponent(model)
    model.add_component("config", config_component)
    with pytest.raises(FileNotFoundError, match="No template file was provided"):
        config_component.create()


@pytest.mark.parametrize("extention", ["yaml", "yml"])
def test_loads_from_config_template_yaml(tmpdir, extention, test_config_dict):
    template_name = "default_config_template"
    template_path = tmpdir / f"{template_name}.{extention}"
    with open(template_path, "w") as fp:
        yaml_dump(test_config_dict, fp)

    model = Model(root=tmpdir, mode="w")
    config_component = ConfigComponent(model, default_template_filename=template_path)
    model.add_component("config", config_component)
    config_component.read()

    assert config_component._data == test_config_dict


def test_loads_from_config_template_toml(tmpdir, test_config_dict):
    template_path = join(tmpdir, "default_config_template.toml")
    with open(template_path, "wb") as fp:
        toml_dump(test_config_dict, fp)

    model = Model(root=tmpdir, mode="w")
    config_component = ConfigComponent(model, default_template_filename=template_path)
    config_component.read()

    assert config_component._data == test_config_dict


def test_make_config_abs(tmpdir, test_config_dict):
    p = join(tmpdir, "config.yml")
    # create file so it will get parsed correctly
    open(p, "w").close()
    test_config_dict["section2"]["path"] = p
    test_config_dict["section2"]["path2"] = abspath(p)
    parsed_config = _make_config_paths_abs(test_config_dict, tmpdir)
    assert all(
        [isabs(p) for p in parsed_config["section2"].values() if isinstance(p, Path)]
    ), parsed_config["section2"]


def test_make_rel_abs(tmpdir, test_config_dict):
    p = join(tmpdir, "config.yml")
    # create file so it will get parsed correctly
    open(p, "w").close()
    test_config_dict["section2"]["path"] = p
    test_config_dict["section2"]["path2"] = abspath(p)
    parsed_config = _make_config_paths_relative(test_config_dict, tmpdir)
    assert all(
        [
            not isabs(p)
            for p in parsed_config["section2"].values()
            if isinstance(p, Path)
        ]
    ), parsed_config["section2"]


def test_set_config(tmpdir):
    model = Model(root=tmpdir)
    config_component = ConfigComponent(model)
    model.add_component("config", config_component)
    config_component.set("global.name", "test")
    assert config_component._data is not None
    assert "name" in config_component._data["global"]
    assert config_component.get_value("global.name") == "test"


def test_write_config(tmpdir):
    model = Model(root=tmpdir)
    config_component = ConfigComponent(model)
    model.add_component("config", config_component)
    config_component.set("global.name", "test")
    write_path = join(tmpdir, "config.yaml")
    assert not isfile(write_path)
    config_component.write()
    assert isfile(write_path)
    read_contents = _read_yaml(write_path)
    assert read_contents == {"global": {"name": "test"}}


def test_get_config_abs_path(tmpdir):
    model = Model(root=tmpdir)
    config_component = ConfigComponent(model)
    model.add_component("config", config_component)
    abs_path = str(tmpdir.join("test.file"))
    config_component.set("global.file", "test.file")
    assert str(config_component.get_value("global.file")) == "test.file"
    assert str(config_component.get_value("global.file", abs_path=True)) == abs_path
