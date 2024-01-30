from logging import WARNING
from os import listdir
from os.path import abspath, exists, join
from pathlib import Path
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
    model_root = ModelRoot(test_case["path"], "w")
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
    model_root = ModelRoot(test_case["path"], "w")
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
    model_root = ModelRoot(test_case["path"], "w")
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
    model_root = ModelRoot(test_case["path"], "w")
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


def test_new_root_creates_dirs_and_log_files(tmpdir):
    model_root_path = Path(join(tmpdir, "root_folder"))
    _ = ModelRoot(model_root_path, "w", ["asdf", "qwery"], True)
    assert exists(model_root_path)
    # two for the folders, one for the log file
    assert len(listdir(model_root_path)) == 3


def test_non_forced_write_errors_on_existing_dir(tmpdir):
    model_root_path = Path(join(tmpdir, "root_folder"))
    _ = ModelRoot(model_root_path, "w", ["asdf", "qwery"], True)
    tmp_file = Path(join(model_root_path, "asdf", "aaaaaaaaa.txt"))
    tmp_file.touch()
    with pytest.raises(IOError, match="Model dir already exists and cannot be "):
        _ = ModelRoot(model_root_path, "w", ["asdf", "qwery"], True)
    assert exists(tmp_file)


def test_read_errors_on_non_existing_dir(tmpdir):
    model_root_path = Path(join(tmpdir, "root_folder"))
    with pytest.raises(IOError, match="model root not found "):
        _ = ModelRoot(model_root_path, "r", ["asdf", "qwery"])


def test_forced_write_warns_on_existing_dir(tmpdir, caplog):
    model_root_path = Path(join(tmpdir, "root_folder"))
    _ = ModelRoot(model_root_path, "w", ["asdf", "qwery"], True)
    tmp_file = Path(join(model_root_path, "asdf", "aaaaaaaaa.txt"))
    tmp_file.touch()

    with caplog.at_level(WARNING):
        _ = ModelRoot(model_root_path, "w+", ["asdf", "qwery"], True)

    assert "Model dir already exists" in caplog.text
    assert exists(tmp_file)
