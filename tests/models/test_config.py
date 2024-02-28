from os.path import abspath, isabs, join
from pathlib import Path

import pytest

from hydromt.io.path import make_config_paths_abs, make_config_paths_relative
from hydromt.io.readers import configread
from hydromt.io.writers import configwrite

ABS_PATH = Path(abspath(__name__))


@pytest.fixture()
def test_config_dict():
    return {
        "section1": {
            "list": [1, 2, 3],
            # "tuple": (1, "b"), # yaml cannot deal with tuple
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


def test_config(tmpdir, test_config_dict):
    ext = "yml"
    config_fn = tmpdir.join(f"config.{ext}")
    configwrite(config_fn, test_config_dict)
    cfdict1 = configread(config_fn, abs_path=True)
    assert test_config_dict["section1"] == cfdict1["section1"]
    assert isinstance(cfdict1["section2"]["path"], Path)
    assert isinstance(cfdict1["section2"]["path1"], str)
    # by default paths in setup_config are not evaluated
    assert isinstance(cfdict1["setup_config"]["path"], str)
    assert isinstance(cfdict1["setup_config"]["float"], float)


def test_rejects_non_yaml_format(tmpdir):
    config_file = tmpdir.join("config.toml")
    # hydromt just cheks the extention, so an empty file is ok
    with open(config_file, "w"):
        pass

    with pytest.raises(ValueError, match="Unknown extention"):
        _ = configread(config_file, abs_path=True)


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
