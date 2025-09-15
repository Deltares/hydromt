from logging import FileHandler, Logger, getLogger
from os.path import abspath
from pathlib import Path

import pytest

from hydromt._typing import ModelMode
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


def test_root_creates_logs_and_dir(tmp_path: Path):
    p = tmp_path / "one"
    log_path = p / "hydromt.log"
    assert not log_path.exists()
    _ = ModelRoot(p)
    assert p.exists()
    assert log_path.is_file()


def test_new_root_copies_old_file(tmp_path: Path):
    logger: Logger = getLogger("hydromt.fake_module")
    first_path = tmp_path / "one"
    assert not first_path.exists()

    r = ModelRoot(first_path, "w")
    logger.warning("hey! this is a secret you should really remember")

    assert first_path.exists()

    with open(first_path / "hydromt.log", "r") as file:
        first_log_str = file.read()
    assert "hey!" in first_log_str, first_log_str

    second_path = tmp_path / "two"
    assert not second_path.exists()

    r.set(second_path)

    assert second_path.exists()
    second_log_path = second_path / "hydromt.log"
    assert second_log_path.exists()

    with open(second_log_path, "r") as file:
        second_log_str = file.read()
    assert "hey!" in second_log_str, second_log_str


def test_new_root_closes_old_log(tmp_path: Path):
    main_logger = getLogger("hydromt")

    first_path = tmp_path / "one"
    second_path = tmp_path / "two"

    r = ModelRoot(first_path, "w")
    assert any(
        h
        for h in main_logger.handlers
        if isinstance(h, FileHandler) and h.baseFilename.startswith(str(first_path))
    ), main_logger.handlers

    r.set(second_path)
    assert not any(
        h
        for h in main_logger.handlers
        if isinstance(h, FileHandler) and h.baseFilename.startswith(str(first_path))
    ), main_logger.handlers

    assert any(
        h
        for h in main_logger.handlers
        if isinstance(h, FileHandler) and h.baseFilename.startswith(str(second_path))
    ), main_logger.handlers


def test_root_overwrite_deletes_old_log(tmp_path: Path):
    logger: Logger = getLogger("hydromt.fake_module")
    path = tmp_path / "one"
    assert not path.exists()
    root = ModelRoot(path, "w")
    logger.warning("hey!, this is a secret you should really remember")
    log_path = path / "hydromt.log"

    with open(log_path, "r") as file:
        first_log_str = file.read()

    assert "hey!" in first_log_str, first_log_str
    root.set(path, "w+")
    logger.warning("what were we talking about again?")
    with open(log_path, "r") as file:
        second_log_str = file.read()
    assert "hey!" not in second_log_str, second_log_str
    assert path.exists()
    assert log_path.is_file()
