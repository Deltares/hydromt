from platform import system
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


@pytest.mark.skipif(
    system() != "Windows", reason="not running windows tests on non windows platform"
)
@pytest.mark.parametrize(
    "test_case",
    [
        {
            "name": "absolute",
            "path": "D:\\hydromt\\test\\model_root",
            "abs_path": "D:\\hydromt\\test\\model_root",
            "num_parents": 3,
            "drive": "D:",
        },
        {
            "name": "relative",
            "path": "hydromt\\test\\model_root",
            "abs_path": "hydromt\\test\\model_root",
            "num_parents": 3,
            "drive": "",
        },
        {
            "name": "network_drive",
            "path": "\\\\p-drive\\hydromt\\test\\model_root",
            "abs_path": "\\\\p-drive\\hydromt\\test\\model_root",
            "num_parents": 3,
            "drive": "\\\\p-drive",
        },
    ],
    ids=case_name,
)
def test_windows_paths_on_windows(test_case):
    model_root = ModelRoot(test_case["path"], "r")
    assert model_root._describes_windows_path(test_case["path"])
    assert len(model_root._path.parents) == test_case["num_parents"]
    assert model_root.path.drive == test_case["drive"]
    assert str(model_root.path.drive) == test_case["abs_path"]


@pytest.mark.skipif(
    system() in ["Linux", "Darwin"],
    reason="not running posix tests on non posix platform",
)
@pytest.mark.parametrize(
    "test_case",
    [
        {
            "name": "absolute",
            "path": "D:\\hydromt\\test\\model_root",
            "abs_path": "/mnt/d/hydromt/test/model_root",
            "num_parents": 5,
            "drive": "",
        },
        {
            "name": "relative",
            "path": "hydromt\\test\\model_root",
            "abs_path": "/hydromt/test/model_root",
            "num_parents": 3,
            "drive": "",
        },
        {
            "name": "network_drive",
            "path": "\\\\p-drive\\hydromt\\test\\model_root",
            "abs_path": "/mnt/p-drive/hydromt/test/model_root",
            "num_parents": 5,
            "drive": "",
        },
    ],
    ids=case_name,
)
def test_windows_paths_on_linux(test_case):
    model_root = ModelRoot(test_case["path"], "r")
    assert model_root._describes_windows_path(test_case["path"])
    assert len(model_root._path.parents) == test_case["num_parents"]
    assert model_root.path.drive == test_case["drive"]
    assert str(model_root.path.drive) == test_case["abs_path"]


@pytest.mark.skipif(
    system() != "Windows", reason="not running windows tests on non windows platform"
)
@pytest.mark.parametrize(
    "test_case",
    [
        {
            "name": "absolute",
            "path": "/home/user/hydromt/test/model_root",
            "abs_path": "C:\\Users\\user\\hydromt\\test\\model_root",
            "num_parents": 3,
        },
        {
            "name": "relative",
            "path": "hydromt/test/model_root",
            "abs_path": "C:\\hydromt\\test\\model_root",
            "num_parents": 3,
        },
        {
            "name": "mounted",
            "path": "/mnt/p-drive\\hydromt\\test\\model_root",
            "abs_path": "\\\\p-drive\\hydromt\\test\\model_root",
            "num_parents": 3,
        },
    ],
    ids=case_name,
)
def test_posix_paths_on_windows(test_case):
    model_root = ModelRoot(test_case["path"], "r")
    assert model_root._describes_windows_path(test_case["path"])
    assert len(model_root._path.parents) == test_case["num_parents"]
    assert model_root.path.drive == test_case["drive"]
    assert str(model_root.path.drive) == test_case["abs_path"]


@pytest.mark.skipif(
    system() in ["Linux", "Darwin"],
    reason="not running posix tests on non posix platform",
)
@pytest.mark.parametrize(
    "test_case",
    [
        {
            "name": "posix_absolute",
            "platform": "posix",
            "path": "/home/user/hydromt/test/model_root",
            "abs_path": "/home/user/hydromt/test/model_root",
            "num_parents": 5,
        },
        {
            "name": "posix_mounted",
            "platform": "posix",
            "path": "/mnt/d/hydromt/test/model_root",
            "abs_path": "/mnt/d/hydromt/test/model_root",
            "num_parents": 5,
        },
        {
            "name": "posix_relative",
            "platform": "posix",
            "path": "hydromt/test/model_root",
            "abs_path": "hydromt/test/model_root",
            "num_parents": 3,
        },
    ],
    ids=case_name,
)
def test_posix_path_on_linux(test_case):
    model_root = ModelRoot(test_case["path"], "r")
    assert model_root._describes_windows_path(test_case["path"])
    assert len(model_root._path.parents) == test_case["num_parents"]
    assert model_root.path.drive == test_case["drive"]
    assert str(model_root.path.drive) == test_case["abs_path"]


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
