from os.path import abspath
from pathlib import Path

import pytest

from hydromt._typing import ModelMode
from hydromt._utils.log import get_hydromt_logger
from hydromt.model.root import ModelRoot

# we need to compensate for where the repo is located when
# we run the tests
CURRENT_PATH = abspath(".")


def case_name(case):
    return case["name"]


@pytest.mark.parametrize("mode", ["r", "r+", ModelMode.READ, ModelMode.APPEND])
def test_assert_reading_modes(mode):
    assert ModelMode.from_str_or_mode(mode).is_reading_mode()


@pytest.mark.parametrize(
    "mode", ["w", "w+", "r+", ModelMode.APPEND, ModelMode.WRITE, ModelMode.FORCED_WRITE]
)
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
        1,
        None,
        -8,
        3.14,
        "⽀",
        "ðŸ˜Š",
    ],
)
def test_errors_on_unknown_modes(mode):
    with pytest.raises(ValueError, match="Unknown mode"):
        _ = ModelMode.from_str_or_mode(mode)


def test_root_init_creates_logfile(tmp_path: Path):
    root = tmp_path / "create_me"
    log_path = root / "hydromt.log"
    assert not log_path.exists()

    _ = ModelRoot(path=root)
    assert root.exists()
    assert log_path.exists()


@pytest.mark.parametrize(
    (
        "mode",
        "should_clear",
    ),
    [
        ("w", True),
        ("w+", True),
        ("r", False),
        ("r+", False),
        (ModelMode.WRITE, True),
        (ModelMode.FORCED_WRITE, True),
        (ModelMode.READ, False),
        (ModelMode.APPEND, False),
    ],
)
def test_root_init_should_clear_logfile(tmp_path: Path, mode, should_clear):
    logger = get_hydromt_logger("fake_module")
    path = tmp_path / "create_me"
    log_path = path / "hydromt.log"
    path.mkdir(parents=True, exist_ok=True)
    log_path.write_text("existing log messages")

    _ = ModelRoot(path=path, mode=mode)
    logger.warning("message from test_root_init_logging_modes")

    with open(log_path, "r") as file:
        log_str = file.read()

    assert "message from test_root_init_logging_modes" in log_str, log_str
    assert ("existing log messages" in log_str) is not should_clear, log_str


@pytest.mark.parametrize(
    (
        "mode",
        "should_clear",
    ),
    [
        ("w", True),
        ("w+", True),
        ("r", False),
        ("r+", False),
        (ModelMode.WRITE, True),
        (ModelMode.FORCED_WRITE, True),
        (ModelMode.READ, False),
        (ModelMode.APPEND, False),
    ],
)
def test_root_set_with_reading_modes_copies_old_logfile(
    tmp_path: Path, mode, should_clear
):
    logger = get_hydromt_logger("fake_module")
    path = tmp_path / "create_me"
    log_path = path / "hydromt.log"
    path.mkdir(parents=True, exist_ok=True)
    log_path.write_text("existing log messages")

    root = ModelRoot(path=path, mode=mode)
    new_path = tmp_path / "another"
    new_path.mkdir(parents=True, exist_ok=True)

    root.set(new_path, mode=mode)
    logger.warning("message from new path")

    with open(new_path / "hydromt.log", "r") as file:
        log_str = file.read()

    assert ("existing log messages" in log_str) is not should_clear, log_str
    assert "message from new path" in log_str, log_str


def test_root_close_removes_filehandler_and_closes_logfile(tmp_path: Path):
    root = tmp_path / "create_me"
    log_path = root / "hydromt.log"
    assert not log_path.exists()

    model_root = ModelRoot(path=root)
    assert model_root._filehandler is not None
    assert root.exists()
    assert log_path.exists()

    model_root.close()
    log_path.unlink()  # try to delete to verify filehandler is removed
    assert model_root._filehandler is None


def test_root_add_and_remove_filehandler(tmp_path: Path):
    path = tmp_path / "create_me"
    log_path = path / "hydromt.log"

    root = ModelRoot(path=path, mode="w")
    assert root._filehandler is not None
    assert log_path.exists()

    root._remove_filehandler()
    assert root._filehandler is None
    log_path.unlink()  # try to delete to verify filehandler is removed


def test_root_add_filehandler_maximum_one_handler(tmp_path: Path):
    path = tmp_path / "create_me"
    log_path = path / "hydromt.log"

    root = ModelRoot(path=path, mode="w")
    assert root._filehandler is not None
    assert log_path.exists()

    # adding a new filehandler should remove the old one first
    old_handler = root._filehandler
    root._add_filehandler(path=log_path.with_name("another.log"))
    assert root._filehandler is not None
    assert root._filehandler is not old_handler
    log_path.unlink()  # try to delete to verify filehandler is removed
