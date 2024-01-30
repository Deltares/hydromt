from os.path import abspath, join
from platform import system

import pytest

from hydromt.models._v1.model_root import ModelRoot
from hydromt.typing import ModelMode

# we need to compensate for where the repo is located when
# we run the tests
CURRENT_PATH = abspath(".")


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
            "path": "D:\\data\\test\\model_root",
            "abs_path": "D:\\data\\test\\model_root",
        },
        {
            "name": "relative",
            "path": "data\\test\\model_root",
            "abs_path": join(CURRENT_PATH, "data", "test", "model_root"),
        },
    ],
    ids=case_name,
)
def test_windows_paths_on_windows(test_case):
    model_root = ModelRoot(test_case["path"], "r")
    assert model_root._describes_windows_path(test_case["path"])
    assert str(model_root._path) == test_case["abs_path"]


@pytest.mark.skipif(
    system() not in ["Linux", "Darwin"],
    reason="not running posix tests on non posix platform",
)
@pytest.mark.parametrize(
    "test_case",
    [
        {
            "name": "absolute",
            "path": "D:\\data\\test\\model_root",
            "abs_path": "/mnt/d/data/test/model_root",
        },
        {
            "name": "relative",
            "path": "data\\test\\model_root",
            "abs_path": join(CURRENT_PATH, "data", "test", "model_root"),
        },
    ],
    ids=case_name,
)
def test_windows_paths_on_linux(test_case):
    model_root = ModelRoot(test_case["path"], "r")
    assert model_root._describes_windows_path(test_case["path"])
    assert str(model_root._path) == test_case["abs_path"]


@pytest.mark.skipif(
    system() != "Windows", reason="not running windows tests on non windows platform"
)
@pytest.mark.parametrize(
    "test_case",
    [
        {
            "name": "absolute",
            "path": "/home/user/hydromt/data/test/model_root",
            "abs_path": join(CURRENT_PATH, "data", "test", "model_root"),
        },
        {
            "name": "relative",
            "path": "data/test/model_root",
            "abs_path": join(CURRENT_PATH, "data", "test", "model_root"),
        },
    ],
    ids=case_name,
)
def test_posix_paths_on_windows(test_case):
    model_root = ModelRoot(test_case["path"], "r")
    assert model_root._describes_posix_path(test_case["path"])
    assert str(model_root._path) == test_case["abs_path"]


@pytest.mark.skipif(
    system() not in ["Linux", "Darwin"],
    reason="not running posix tests on non posix platform",
)
@pytest.mark.parametrize(
    "test_case",
    [
        {
            "name": "absolute",
            "path": "/home/user/data/test/model_root",
            "abs_path": "/home/user/data/test/model_root",
        },
        {
            "name": "mounted",
            "path": "/mnt/d/data/test/model_root",
            "abs_path": "/mnt/d/data/test/model_root",
        },
        {
            "name": "relative",
            "path": "data/test/model_root",
            "abs_path": join(CURRENT_PATH, "data", "test", "model_root"),
        },
    ],
    ids=case_name,
)
def test_posix_path_on_linux(test_case):
    model_root = ModelRoot(test_case["path"], "r")
    assert model_root._describes_posix_path(test_case["path"])
    assert str(model_root._path) == test_case["abs_path"]


@pytest.mark.parametrize("mode", ["r", "r+", ModelMode.READ, ModelMode.APPEND])
def test_assert_reading_modes(mode):
    assert ModelMode.from_str_or_mode(mode).is_reading_mode()


@pytest.mark.parametrize("mode", ["w", "w+", ModelMode.WRITE, ModelMode.FORCED_WRITE])
def test_assert_writing_modes(mode):
    assert ModelMode.from_str_or_mode(mode).is_writing_mode()


@pytest.mark.parametrize(
    "mode",
    [
        "a",
        "wr",
        "rw",
        "r++",
        "w2",
        "\\w",
        "ww",
        "",
        "+w",
        "lorum ipsum",
        None,
        1,
        -8,
        3.14,
        "⽀",
        "ðŸ˜Š",
    ],
)
def test_errors_on_unknown_modes(mode):
    with pytest.raises(ValueError, match="Unknown mode"):
        _ = ModelMode.from_str_or_mode(mode)
