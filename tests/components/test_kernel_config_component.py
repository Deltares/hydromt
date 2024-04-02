from os.path import abspath, isabs, isfile, join
from pathlib import Path

import pytest

from hydromt.components.kernel_config import (
    DEFAULT_KERNEL_CONFIG_PATH,
    KernelConfigComponent,
)
from hydromt.io.path import make_config_paths_abs, make_config_paths_relative
from hydromt.io.readers import configread, read_yaml
from hydromt.io.writers import write_yaml
from hydromt.models import Model

ABS_PATH = Path(abspath(__name__))


@pytest.fixture()
def test_config_dict():
    return {
        "section1": {
            "list": [1, 2, 3],
            "bool": True,
            "str": "test",
            "int": 1,
            "float": 2.3,
            "None": None,
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
    # hydromt just cheks the extention, so an empty file is ok
    with open(config_file, "w"):
        pass

    with pytest.raises(ValueError, match="Unknown extention"):
        _ = configread(config_file, abs_path=True)


def test_config_always_reads(tmpdir):
    kernel_config_path = join(tmpdir, DEFAULT_KERNEL_CONFIG_PATH)
    config_data = {"a": 1, "b": 3.14, "c": None, "d": {"e": {"f": True}}}
    write_yaml(kernel_config_path, config_data)
    # notice the write mode
    model = Model(root=tmpdir, mode="w")
    model.add_component("kernel_config", KernelConfigComponent(model))
    comp = model.get_component("kernel_config", KernelConfigComponent)
    assert comp.data == config_data


def test_raises_warning_on_no_config_template_found(tmpdir, caplog):
    model = Model(root=tmpdir, mode="w")
    model.add_component("kernel_config", KernelConfigComponent(model))
    comp = model.get_component("kernel_config", KernelConfigComponent)
    assert comp.data == {}
    assert "No default kernel config was found " in caplog.text


def test_make_config_abs(tmpdir, test_config_dict):
    p = join(tmpdir, "config.yml")
    # create file so it will get parsed correctly
    with open(p, "w"):
        pass
    test_config_dict["section2"]["path"] = p
    test_config_dict["section2"]["path2"] = abspath(p)
    parsed_config = make_config_paths_abs(test_config_dict, tmpdir)
    assert all(
        [isabs(p) for p in parsed_config["section2"].values() if isinstance(p, Path)]
    ), parsed_config["section2"]


def test_make_rel_abs(tmpdir, test_config_dict):
    p = join(tmpdir, "config.yml")
    # create file so it will get parsed correctly
    with open(p, "w"):
        pass
    test_config_dict["section2"]["path"] = p
    test_config_dict["section2"]["path2"] = abspath(p)
    parsed_config = make_config_paths_relative(test_config_dict, tmpdir)
    assert all(
        [
            not isabs(p)
            for p in parsed_config["section2"].values()
            if isinstance(p, Path)
        ]
    ), parsed_config["section2"]


def test_set_config(tmpdir):
    model = Model(root=tmpdir)
    model.add_component("config", KernelConfigComponent(model))
    config_component = model.get_component("config", KernelConfigComponent)
    config_component.set_value("global.name", "test")
    assert config_component._data is not None
    assert "name" in config_component._data["global"]
    assert config_component.get_value("global.name") == "test"


def test_write_config(tmpdir):
    model = Model(root=tmpdir)
    model.add_component("config", KernelConfigComponent(model))
    config_component = model.get_component("config", KernelConfigComponent)
    config_component.set_value("global.name", "test")
    write_path = join(tmpdir, DEFAULT_KERNEL_CONFIG_PATH)
    assert not isfile(write_path)
    config_component.write()
    assert isfile(write_path)
    read_contents = read_yaml(write_path)
    assert read_contents == {"global": {"name": "test"}}


def test_get_config_abs_path(tmpdir):
    model = Model(root=tmpdir)
    model.add_component("config", KernelConfigComponent(model))
    config_component = model.get_component("config", KernelConfigComponent)
    abs_path = str(tmpdir.join("test.file"))
    config_component.set_value("global.file", "test.file")
    assert str(config_component.get_value("global.file")) == "test.file"
    assert str(config_component.get_value("global.file", abs_path=True)) == abs_path
