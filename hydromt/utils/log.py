"""Implementations related to logging."""

import os
import sys
from logging import (
    FileHandler,
    Formatter,
    Logger,
    StreamHandler,
    captureWarnings,
    getLogger,
)
from typing import Optional

from hydromt import __version__

FMT = "%(asctime)s - %(name)s - %(module)s - %(levelname)s - %(message)s"

__all__ = ["setuplog"]


def wait_and_remove_file_handlers(logger: Logger):
    for handler in logger.handlers:
        if isinstance(handler, FileHandler):
            # remove handler first, as otherwise new calls to the logger may open the
            # same filename again.
            logger.removeHandler(handler)
            # wait for all messages to be processed
            handler.flush()
            # then close the handler
            handler.close()


def setuplog(
    path: Optional[str] = None,
    log_level: int = 20,
    fmt: str = FMT,
    append: bool = True,
):
    """Set up the logging on sys.stdout and file if path is given.

    Parameters
    ----------
    path : str, optional
        path to logfile, by default None
    log_level : int, optional
        Log level [0-50], by default 20 (info)
    fmt : str, optional
        log message formatter
    append : bool, optional
        Whether to append (True) or overwrite (False) to a logfile at path,
        by default True
    """
    main_logger: Logger = getLogger("hydromt")
    wait_and_remove_file_handlers(main_logger)
    captureWarnings(True)
    main_logger.setLevel(log_level)
    console = StreamHandler(sys.stdout)
    console.setLevel(log_level)
    console.setFormatter(Formatter(fmt))
    main_logger.addHandler(console)
    if path is not None:
        if append is False and os.path.isfile(path):
            os.unlink(path)
        add_filehandler(main_logger, path, log_level=log_level, fmt=fmt)
    main_logger.info(f"HydroMT version: {__version__}")


def add_filehandler(logger, path, log_level=20, fmt=FMT):
    """Add file handler to logger."""
    if not os.path.isdir(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    isfile = os.path.isfile(path)
    ch = FileHandler(path)
    ch.setFormatter(Formatter(fmt))
    ch.setLevel(log_level)
    logger.addHandler(ch)
    if isfile:
        logger.debug(f"Appending log messages to file {path}.")
    else:
        logger.debug(f"Writing log messages to new file {path}.")
