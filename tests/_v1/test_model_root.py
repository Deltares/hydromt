from pathlib import PurePath
from typing import TypedDict

import pytest

from hydromt.models._v1.model_root import ModelRoot
from hydromt.typing import ModelMode


class RootModelTestCase(TypedDict):
    name: str
    path: str
    num_parents: int
    drive: str


def case_name(case):
    return case["name"]


# these tests below are all designed to have the same target folder and parents
# (3 segments in total) but different "mount points". We all want these to be able
# to point to essentially the same place
def windows_absolute_root():
    return PurePath()


def windows_network_drive_root():
    return PurePath()


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "name": "posix_absolute",
            "platform": "posix",
            "path": "/home/user/hydromt/test/model_root",
            "num_parents": 5,
            "drive": "",
        },
        {
            "name": "posix_mounted",
            "platform": "posix",
            "path": "/mnt/d/hydromt/test/model_root",
            "num_parents": 5,
            "drive": "",
        },
        {
            "name": "posix_relative",
            "platform": "posix",
            "path": "hydromt/test/model_root",
            "num_parents": 3,
            "drive": "",
        },
        {
            "name": "windows_absolute",
            "platform": "windows",
            "path": "D:\\hydromt\\test\\model_root",
            "num_parents": 3,
            "drive": "D:",
        },
        {
            "name": "windows_network_drive",
            "platform": "windows",
            "path": "\\\\hydromt\\test\\model_root",
            "num_parents": 3,
            "drive": "//",
        },
        {
            "name": "windows_relative",
            "platform": "windows",
            "path": "hydromt\\test\\model_root",
            "num_parents": 3,
            "drive": "",
        },
    ],
    ids=case_name,
)
def test_cross_platform_paths(test_case):
    model_root = ModelRoot(test_case["path"], "r")
    if test_case["platform"] == "posix":
        assert model_root.describes_posix_path(test_case["path"])
    elif test_case["platform"] == "windows":
        assert model_root.describes_windows_path(test_case["path"])

    assert len(model_root._path.parents) == test_case["num_parents"]
    assert model_root._path.drive == test_case["drive"]


@pytest.mark.parametrize("mode", ["r", "r+", ModelMode.READ, ModelMode.APPEND])
def test_assert_reading_modes(mode):
    assert ModelMode.from_str_or_mode(mode).is_reading_mode()


@pytest.mark.parametrize("mode", ["w", "w+", ModelMode.WRITE, ModelMode.FORCED_WRITE])
def test_assert_writing_modes(mode):
    assert ModelMode.from_str_or_mode(mode).is_writing_mode()


@pytest.mark.parametrize(
    "mode",
    ["a", "w2", "\\w", "ww", "", "+w", "lorum ipsum", None, 1, -8, 3.14, "⽀", "ðŸ˜Š"],
)
def test_errors_on_unknown_modes(mode):
    with pytest.raises(ValueError, match="Unknown mode"):
        _ = ModelMode.from_str_or_mode(mode)
