from logging import FileHandler, Logger
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


def test_root_creates_logs_and_dir(tmp_path: Path):
    root1 = tmp_path / "one"
    filename = Path("hydromt.log")
    log_path1 = root1 / filename
    assert not log_path1.exists()

    _ = ModelRoot(path=root1)
    assert root1.exists()
    assert not log_path1.exists()

    root2 = tmp_path / "two"
    log_path2 = root2 / filename
    _ = ModelRoot(path=root2, log_file=filename)
    assert root2.exists()
    assert log_path2.exists()


def test_new_root_copies_old_file(tmp_path: Path):
    logger: Logger = get_hydromt_logger("fake_module")
    first_path = tmp_path / "one"
    assert not first_path.exists()
    filename = Path("hydromt.log")
    log_path = first_path / filename
    r = ModelRoot(first_path, "w", log_file=filename)
    logger.warning("hey! this is a secret you should really remember")

    assert first_path.exists()

    with open(log_path, "r") as file:
        first_log_str = file.read()
    assert "hey!" in first_log_str, first_log_str

    second_path = tmp_path / "two"
    assert not second_path.exists()

    r.set(second_path)

    assert second_path.exists()
    second_log_path = second_path / filename
    assert second_log_path.exists()

    with open(second_log_path, "r") as file:
        second_log_str = file.read()
    assert "hey!" in second_log_str, second_log_str


def test_new_root_closes_old_log(tmp_path: Path):
    main_logger = get_hydromt_logger()
    filename = Path("hydromt.log")
    first_path = tmp_path / "one"
    second_path = tmp_path / "two"

    root = ModelRoot(first_path, "w", log_file=filename)

    # assert there exists 1 file handler with the correct path
    file_handlers = [h for h in main_logger.handlers if isinstance(h, FileHandler)]
    open_logfiles = [
        h for h in file_handlers if h.baseFilename.startswith(str(first_path))
    ]
    assert len(open_logfiles) == 1, (
        f"Expected 1 file handler for first root, but got {len(open_logfiles)}"
    )

    # set to new root, which should close the old log file, remove the handler, and create a new one
    root.set(second_path)

    # assert there exists 1 file handler with the correct path, and 0 with the old path
    file_handlers = [h for h in main_logger.handlers if isinstance(h, FileHandler)]
    open_logfiles_first = [
        h for h in file_handlers if h.baseFilename.startswith(str(first_path))
    ]
    open_logfiles_second = [
        h for h in file_handlers if h.baseFilename.startswith(str(second_path))
    ]
    assert len(open_logfiles_first) == 0, (
        f"Expected 0 file handlers for old root, but got {len(open_logfiles_first)}"
    )
    assert len(open_logfiles_second) == 1, (
        f"Expected 1 file handler for new root, but got {len(open_logfiles_second)}"
    )


def test_root_overwrite_deletes_old_log(tmp_path: Path):
    logger: Logger = get_hydromt_logger("fake_module")
    path = tmp_path / "one"
    filename = Path("hydromt.log")
    log_path = path / filename

    assert not path.exists()

    root = ModelRoot(path, "w", log_file=filename)
    logger.warning("hey!, this is a secret you should really remember")

    with open(log_path, "r") as file:
        first_log_str = file.read()
    assert "hey!" in first_log_str, first_log_str

    root.set(path, "w+")
    logger.warning("what were we talking about again?")
    with open(log_path, "r") as file:
        second_log_str = file.read()
    assert "hey!" not in second_log_str, second_log_str
