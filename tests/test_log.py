import logging
from pathlib import Path

import pytest

from hydromt.log import _ROOT_LOGGER, to_file


def test_to_file_logs_and_reraises(tmp_path: Path):
    log_file = tmp_path / "test.log"

    logger = _ROOT_LOGGER
    logger.setLevel(logging.DEBUG)

    # Ensure clean state
    initial_handlers = list(logger.handlers)

    with pytest.raises(RuntimeError, match="test error"):  # noqa: PT012
        with to_file(path=log_file):
            logger.info("before error")
            raise RuntimeError("test error")

    # File should exist
    assert log_file.exists()

    content = log_file.read_text()

    # 1. normal log message is written
    assert "before error" in content

    # 2. exception is logged
    assert "Unhandled exception" in content
    assert "RuntimeError: test error" in content

    # 3. traceback is present
    assert "Traceback" in content

    # 4. handler cleanup
    assert logger.handlers == initial_handlers


def test_to_file_append_false_overwrites(tmp_path: Path):
    log_file = tmp_path / "test.log"

    logger = _ROOT_LOGGER
    logger.setLevel(logging.INFO)

    with to_file(path=log_file):
        logger.info("first")

    with to_file(path=log_file, append=False):
        logger.info("second")

    content = log_file.read_text()

    assert "first" not in content
    assert "second" in content
