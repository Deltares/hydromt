"""Implementations related to logging."""

import logging
import sys
from pathlib import Path
from typing import Optional

from hydromt import __version__

FMT = "%(asctime)s - %(name)s - %(module)s - %(levelname)s - %(message)s"

__all__ = [
    "_setuplog",
    "get_hydromt_logger",
    "remove_hydromt_file_handlers",
]


def get_hydromt_logger(name: str | None = None) -> logging.Logger:
    """Get a logger with the given name, or the root hydromt logger if name is None."""
    if name is None:
        return logging.getLogger("hydromt")
    else:
        return logging.getLogger(f"hydromt.{name}")


ROOT_LOGGER = get_hydromt_logger()


def remove_hydromt_file_handlers(
    path_or_filename: str | Path | None = None,
    include_children: bool = True,
) -> None:
    """Remove only file handlers that match a given path or filename.

    Parameters
    ----------
    path_or_name : str | Path | None, optional
        Either a full path to the log file, the filename (e.g., "hydromt.log"), or None to remove all hydromt file handlers, by default None.
    include_children : bool, optional
        Whether to also remove handlers from child loggers (hydromt.*), by default True.
    """
    if path_or_filename is None:
        match_str = None
    elif isinstance(path_or_filename, str):
        match_str = path_or_filename
    else:
        match_str = str(path_or_filename)

    def _remove_matching(logger: logging.Logger):
        for handler in list(logger.handlers):
            if isinstance(handler, logging.FileHandler):
                handler_path = Path(handler.baseFilename).resolve()
                if match_str is not None and match_str in {
                    str(handler_path),
                    handler_path.name,
                }:
                    logger.removeHandler(handler)
                    handler.flush()
                    handler.close()

    # main hydromt logger
    _remove_matching(ROOT_LOGGER)

    if include_children:
        for name, lg in logging.Logger.manager.loggerDict.items():
            if isinstance(lg, logging.Logger) and name.startswith("hydromt."):
                _remove_matching(lg)


def _setuplog(
    path: Optional[Path] = None,
    log_level: int = logging.INFO,
    fmt: str = FMT,
    append: bool = True,
    logger: logging.Logger = ROOT_LOGGER,
):
    """Set up the logging for hydromt with a console handler and optionally a file handler.

    Parameters
    ----------
    path : Path, optional
        Path to model root, where the log file "hydromt.log" will be created.
        If None is provided, no file handler is added. By default None.
    log_level : int, optional
        Log level [0-50], by default 20 (info)
    fmt : str, optional
        log message formatter
    append : bool, optional
        Whether to append (True) or overwrite (False) to a logfile at path,
        by default True
    """
    if path is not None:
        remove_hydromt_file_handlers(path_or_filename=path)
    logging.captureWarnings(True)
    logger.setLevel(log_level)

    # Remove all stream handlers
    logger.handlers = [
        h for h in logger.handlers if not isinstance(h, logging.StreamHandler)
    ]
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(log_level)
    console.setFormatter(logging.Formatter(fmt))
    logger.addHandler(console)

    if path is not None:
        if not append:
            path.unlink(missing_ok=True)
        _add_filehandler(path=path, log_level=log_level, fmt=fmt, logger=logger)
    logger.info(f"HydroMT version: {__version__}")


def _add_filehandler(
    path: Path,
    log_level: int = logging.INFO,
    fmt: str = FMT,
    logger: logging.Logger = ROOT_LOGGER,
):
    """Add file handler to logger."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ch = logging.FileHandler(path)
    ch.setFormatter(logging.Formatter(fmt))
    ch.setLevel(log_level)
    logger.addHandler(ch)

    if path.exists():
        logger.debug(f"Appending log messages to file {path}.")
    else:
        logger.debug(f"Writing log messages to new file {path}.")
