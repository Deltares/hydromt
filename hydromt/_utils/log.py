"""Implementations related to logging."""

import logging
import sys
from contextlib import contextmanager
from pathlib import Path

from hydromt import __version__

__all__ = [
    "initialize_logging",
    "set_log_level",
    "to_file",
]

_ROOT_LOGGER = logging.getLogger("hydromt")
_LOG_FORMAT = "%(asctime)s - %(name)s - %(module)s - %(levelname)s - %(message)s"
_DEFAULT_FORMATTER = logging.Formatter(_LOG_FORMAT)


def initialize_logging() -> None:
    """Initialize the hydromt root logger with a formatter, a log level and an optional console handler.

    This function should be called once, at the start of the program.
    For HydroMT, this is achieved by calling it in the `hydromt.__init__.py` file, so it is
    automatically called when importing the hydromt package.
    If the root logger has a handler before hydromt is imported, no new handler will be added.
    """
    logging.captureWarnings(True)
    _ROOT_LOGGER.setLevel(logging.INFO)
    if not _ROOT_LOGGER.hasHandlers():
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(_DEFAULT_FORMATTER)
        _ROOT_LOGGER.addHandler(console)
    _ROOT_LOGGER.info(f"HydroMT version: {__version__}")


def set_log_level(log_level: int) -> None:
    """Set the log level of the hydromt root logger.

    This also affects all child loggers (e.g., ``hydromt.core``),
    unless they have their own log level explicitly set.

    Example
    -------
    .. code-block:: python
        from hydromt.log import set_log_level

        # Set log level to ERROR
        set_log_level(logging.ERROR)
    """
    _ROOT_LOGGER.setLevel(log_level)


def _add_filehandler(
    path: Path,
    *,
    log_level: int | None = None,
    formatter: logging.Formatter = _DEFAULT_FORMATTER,
    logger: logging.Logger = _ROOT_LOGGER,
) -> logging.FileHandler:
    """Add file handler to the hydromt root logger.

    Parameters
    ----------
    path : Path
        Path to the log file.
    log_level : int, optional
        Log level for the file handler, by default None (inherits from logger)
    formatter : logging.Formatter, optional
        Log formatter, by default _DEFAULT_FORMATTER
    logger : logging.Logger, optional
        Logger to add the file handler to, by default _ROOT_LOGGER

    Returns
    -------
    logging.FileHandler
        The created file handler.
    """
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


@contextmanager
def to_file(
    path: Path,
    *,
    log_level: int | None = None,
    formatter: logging.Formatter = _DEFAULT_FORMATTER,
    logger: logging.Logger = _ROOT_LOGGER,
    append: bool = True,
):
    """Context manager that attaches a file handler for the duration of the block.

    Parameters
    ----------
    path : Path
        Path to the log file.
    log_level : int, optional
        Log level for the file handler, by default None (inherits from logger)
    formatter : logging.Formatter, optional
        Log formatter, by default _DEFAULT_FORMATTER
    logger : logging.Logger, optional
        Logger to add the file handler to, by default _ROOT_LOGGER
    append : bool, optional
        Whether to append to the log file if it exists, by default True.

    Example
    -------
    .. code-block:: python
        from hydromt import log
        from pathlib import Path
        log_file = Path.cwd() / "my_log.log"
        with log.to_file(log_file):
            # log messages in this block will be written to `log_file`
            logger = logging.getLogger("hydromt.my_module")
            logger.info("This is an info message.")
            logger.error("This is an error message.")

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not append and path.exists():
        path.unlink()
    handler = _add_filehandler(
        path, log_level=log_level, formatter=formatter, logger=logger
    )
    try:
        yield
    finally:
        logger.removeHandler(handler)
        handler.close()
