"""Implementations related to logging."""

import logging
import sys
from contextlib import contextmanager
from pathlib import Path

from hydromt import __version__

__all__ = [
    "initialize_logging",
    "set_log_level",
    "add_filehandler",
    "remove_filehandler",
]

_ROOT_LOGGER = logging.getLogger("hydromt")
_LOG_FORMAT = "%(asctime)s - %(name)s - %(module)s - %(levelname)s - %(message)s"
_DEFAULT_FORMATTER = logging.Formatter(_LOG_FORMAT)


def initialize_logging(
    log_level: int = logging.INFO, formatter: logging.Formatter = _DEFAULT_FORMATTER
) -> None:
    """Initialize the hydromt root logger with a console handler, formatter and a log level.

    Example
    -------
    ```python
    from hydromt._utils.log import initialize_logging

    # Initialize with log level (INFO)
    initialize_logging(log_level=logging.INFO)

    # Change log level to ERROR
    logger = logging.getLogger("hydromt")
    logger.setLevel(logging.ERROR)
    """
    logging.captureWarnings(True)
    _ROOT_LOGGER.setLevel(log_level)
    if not _ROOT_LOGGER.hasHandlers():
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(log_level)
        console.setFormatter(formatter)
        _ROOT_LOGGER.addHandler(console)
    _ROOT_LOGGER.info(f"HydroMT version: {__version__}")


def set_log_level(log_level: int, logger: logging.Logger = _ROOT_LOGGER) -> None:
    """Set the log level of the hydromt root logger.

    This also affects all child loggers (e.g., ``hydromt.core``),
    unless they have their own log level explicitly set.

    Example
    -------
    ```python
    from hydromt.log import set_log_level

    # Set log level to ERROR
    set_log_level(logging.ERROR)
    ```
    """
    logger.setLevel(log_level)


def add_filehandler(
    path: Path,
    log_level: int | None = None,
    formatter: logging.Formatter = _DEFAULT_FORMATTER,
    logger: logging.Logger = _ROOT_LOGGER,
) -> logging.FileHandler:
    """Add file handler to logger."""
    path.parent.mkdir(parents=True, exist_ok=True)
    filehandler = logging.FileHandler(path)
    filehandler.setFormatter(formatter)
    if log_level is not None:
        filehandler.setLevel(log_level)
    logger.addHandler(filehandler)
    if path.exists():
        logger.debug(f"Appending log messages to file {path}.")
    else:
        logger.debug(f"Writing log messages to new file {path}.")
    return filehandler


def remove_filehandler(path: Path, logger: logging.Logger = _ROOT_LOGGER) -> None:
    """Remove file handler from logger."""
    for h in logger.handlers:
        if not isinstance(h, logging.FileHandler):
            continue
        if Path(h.baseFilename) == path.resolve():
            logger.removeHandler(h)
            h.close()
            logger.debug(f"Removed log file handler {path}.")


@contextmanager
def to_file(
    path: Path,
    log_level: int | None = None,
    formatter: logging.Formatter = _DEFAULT_FORMATTER,
    logger: logging.Logger = _ROOT_LOGGER,
    append: bool = True,
):
    """Context manager that attaches a file handler for the duration of the block."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not append and path.exists():
        path.unlink()
    filehandler = logging.FileHandler(path)
    if formatter is not None:
        filehandler.setFormatter(formatter)
    if log_level is not None:
        filehandler.setLevel(log_level)
    logger.addHandler(filehandler)
    try:
        yield
    finally:
        logger.removeHandler(filehandler)
        filehandler.close()


def shutdown_logging():
    """Shut down the logging system by removing all handlers and closing them."""
    for h in _ROOT_LOGGER.handlers:
        _ROOT_LOGGER.removeHandler(h)
        h.close()

    for child in _ROOT_LOGGER.manager.loggerDict.values():
        if isinstance(child, logging.Logger):
            for h in child.handlers:
                child.removeHandler(h)
                h.close()

    logging.shutdown()
