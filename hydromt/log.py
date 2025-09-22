"""Implementations related to logging."""

import logging
import sys
from contextlib import contextmanager
from pathlib import Path

from hydromt import __version__

__all__ = [
    "initialize_logging",
    "add_filehandler",
    "remove_filehandler",
]

_ROOT_LOGGER = logging.getLogger("hydromt")
FMT = "%(asctime)s - %(name)s - %(module)s - %(levelname)s - %(message)s"
_DEFAULT_FORMATTER = logging.Formatter(FMT)


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
    logger = logging.getLogger("hydromt")
    logging.captureWarnings(True)
    logger.setLevel(log_level)
    if not logger.hasHandlers():
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(log_level)
        console.setFormatter(formatter or logging.Formatter(FMT))
        logger.addHandler(console)
    logger.info(f"HydroMT version: {__version__}")


def add_filehandler(
    path: Path,
    log_level: int | None = None,
    fmt: logging.Formatter = _DEFAULT_FORMATTER,
    logger: logging.Logger = _ROOT_LOGGER,
) -> logging.FileHandler:
    """Add file handler to logger."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(path)

    fh.setFormatter(fmt)

    if log_level is not None:
        fh.setLevel(log_level)

    logger.addHandler(fh)

    if path.exists():
        logger.debug(f"Appending log messages to file {path}.")
    else:
        logger.debug(f"Writing log messages to new file {path}.")

    return fh


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
    fmt: str | None = None,
    logger: logging.Logger = _ROOT_LOGGER,
    append: bool = True,
):
    """Context manager that attaches a file handler for the duration of the block."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not append and path.exists():
        path.unlink()

    fh = logging.FileHandler(path)
    if fmt is not None:
        fh.setFormatter(logging.Formatter(fmt))
    if log_level is not None:
        fh.setLevel(log_level)

    logger.addHandler(fh)

    try:
        yield
    finally:
        logger.removeHandler(fh)
        fh.close()


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
