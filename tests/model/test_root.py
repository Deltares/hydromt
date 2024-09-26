from logging import FileHandler, Logger, getLogger
from os.path import abspath, exists, join

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


def test_root_creates_logs_and_dir(tmpdir):
    p = join(tmpdir, "one")
    assert not exists(p)
    _ = ModelRoot(p)
    assert exists(p)
    assert exists(join(p, "hydromt.log"))


def test_new_root_copies_old_file(tmpdir):
    logger: Logger = getLogger("hydromt.fake_module")
    first_path = join(tmpdir, "one")
    assert not exists(first_path)

    r = ModelRoot(first_path, "w")
    logger.warning("hey! this is a secret you should really remember")

    assert exists(first_path)

    with open(join(first_path, "hydromt.log"), "r") as file:
        first_log_str = file.read()
    assert "hey!" in first_log_str, first_log_str

    second_path = join(tmpdir, "two")
    assert not exists(second_path)

    r.set(second_path)

    assert exists(second_path)
    assert exists(join(second_path, "hydromt.log"))

    with open(join(second_path, "hydromt.log"), "r") as file:
        second_log_str = file.read()
    assert "hey!" in second_log_str, second_log_str


def test_new_root_closes_old_log(tmpdir):
    main_logger = getLogger("hydromt")

    first_path = join(tmpdir, "one")
    second_path = join(tmpdir, "two")

    r = ModelRoot(first_path, "w")
    assert any(
        [
            h
            for h in main_logger.handlers
            if isinstance(h, FileHandler) and h.baseFilename.startswith(first_path)
        ]
    ), main_logger.handlers

    r.set(second_path)
    assert not any(
        [
            h
            for h in main_logger.handlers
            if isinstance(h, FileHandler) and h.baseFilename.startswith(first_path)
        ]
    ), main_logger.handlers

    assert any(
        [
            h
            for h in main_logger.handlers
            if isinstance(h, FileHandler) and h.baseFilename.startswith(second_path)
        ]
    ), main_logger.handlers


def test_root_overwrite_deletes_old_log(tmpdir):
    logger: Logger = getLogger("hydromt.fake_module")
    path = join(tmpdir, "one")
    assert not exists(path)
    root = ModelRoot(path, "w")
    logger.warning("hey!, this is a secret you should really remember")

    with open(join(path, "hydromt.log"), "r") as file:
        first_log_str = file.read()

    assert "hey!" in first_log_str, first_log_str
    root.set(path, "w+")
    logger.warning("what were we talking about again?")
    with open(join(root.path, "hydromt.log"), "r") as file:
        second_log_str = file.read()
    assert "hey!" not in second_log_str, second_log_str
    assert exists(path)
    assert exists(join(path, "hydromt.log"))
