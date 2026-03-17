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


def initialize_logging(
    *,
    file_path: Path | None = None,
    console: bool = True,
    level: int = logging.INFO,
    capture_warnings: bool = True,
) -> logging.FileHandler | None:
    """Initialize the logging configuration for the hydromt root logger.

    If you want logging statements to appear in the console, call this function at the start of your script or application with the ``console`` parameter set to True.
    If you want to write all log messages to a file, provide a path to the log file via the ``file_path`` argument.

    Parameters
    ----------
    file_path : Path, optional
        Optional path to a log file. If provided, a file handler will be added that writes
        log messages to this file, by default None.
        Note that if you provide a file path, you are responsible for ensuring that the file
        is properly closed when your application exits. You can use the ``to_file``
        context manager to automatically handle this.
    console : bool, optional
        Whether to add a console handler that writes log messages to the console, by default True
    level : int, optional
        Log level to set for the logger, by default logging.INFO
    capture_warnings : bool, optional
        Whether to capture warnings issued by the warnings module and redirect them to the logging system, by default True

    Returns
    -------
    logging.FileHandler | None
        The file handler that was added if a file path was provided, otherwise None.
    """
    logging.captureWarnings(capture_warnings)
    _ROOT_LOGGER.setLevel(level)
    file_handler = None
    if file_path is not None:
        file_handler = _add_filehandler(
            file_path,
            log_level=level,
            formatter=_DEFAULT_FORMATTER,
            logger=_ROOT_LOGGER,
        )

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(_DEFAULT_FORMATTER)
        _ROOT_LOGGER.addHandler(console_handler)

    return file_handler


def set_log_level(log_level: int) -> None:
    """Set the log level of the hydromt root logger.

    This also affects all child loggers (e.g., ``hydromt.core``),
    unless they have their own log level explicitly set.

    Parameters
    ----------
    log_level : int
        Log level to set (e.g., logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, etc)

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
        Log level for the file handler, by default None
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


def log_version() -> None:
    """Log the current version of HydroMT at INFO level."""
    _ROOT_LOGGER.info(f"HydroMT version: {__version__}")


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
    except Exception:
        logger.exception("Unhandled exception occured.")
        raise
    finally:
        handler.flush()
        logger.removeHandler(handler)
        handler.close()
