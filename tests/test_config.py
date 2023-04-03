from pathlib import Path

import pytest
import yaml

from hydromt import config


def test_config(tmpdir):
    cfdict = {
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
            "path": "config.ini",  # path exists -> Path
            "path1": "config1.ini",  # path does not exist -> str
        },
        # evaluation skipped by default for setup_config
        "setup_config": {
            "path": "config.ini",
            "float": 2.3,
        },
    }
    config_fn = tmpdir.join("config.ini")
    config_fn2 = tmpdir.join("config.yaml")
    with open(config_fn2, "w") as yaml_file:
        yaml.dump(cfdict, yaml_file)
    config.configwrite(config_fn, cfdict)
    # test for deprecation warning
    with pytest.deprecated_call():
        config.configread(config_fn=config_fn)
    cfdict1 = config.configread(config_fn, abs_path=True)
    assert cfdict["section1"] == cfdict1["section1"]
    assert isinstance(cfdict1["section2"]["path"], Path)
    assert isinstance(cfdict1["section2"]["path1"], str)
    # by default paths in setup_config are not evaluated
    assert isinstance(cfdict1["setup_config"]["path"], str)
    assert isinstance(cfdict1["setup_config"]["float"], float)
    # return only str if skip_eval=True
    cfdict1 = config.configread(config_fn, skip_eval=True)
    for section in cfdict1:
        print([val for val in cfdict1[section].values()])
        print([type(val) for val in cfdict1[section].values()])
        assert all([isinstance(val, str) for val in cfdict1[section].values()])
    # do not evaluate a specific section
    cfdict1 = config.configread(config_fn, skip_eval_sections=["setup_config"])
    assert isinstance(cfdict1["setup_config"]["float"], str)
    cfdict2 = config.configread(config_fn2, abs_path=True)
    # cfdict["section1"].pop(
    #     "None", None
    # )  # None is dropped from the dictionary when writing to toml
    assert cfdict["section1"] == cfdict2["section1"]
    assert isinstance(cfdict2["section2"]["path"], Path)
    assert isinstance(cfdict2["section2"]["path1"], str)
    # by default paths in setup_config are not evaluated
    assert isinstance(cfdict2["setup_config"]["path"], str)
    assert isinstance(cfdict2["setup_config"]["float"], float)
