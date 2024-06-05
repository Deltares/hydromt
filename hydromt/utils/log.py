"""Implementations related to logging."""

import logging
import logging.handlers
import os
import sys
from functools import wraps

from hydromt import __version__

FMT = "%(asctime)s - %(name)s - %(module)s - %(levelname)s - %(message)s"

__all__ = ["setuplog"]


def setuplog(
    name: str = "hydromt",
    path: str = None,
    log_level: int = 20,
    fmt: str = FMT,
    append: bool = True,
) -> logging.Logger:
    """Set up the logging on sys.stdout and file if path is given.

    Parameters
    ----------
    name : str, optional
        logger name, by default "hydromt"
    path : str, optional
        path to logfile, by default None
    log_level : int, optional
        Log level [0-50], by default 20 (info)
    fmt : str, optional
        log message formatter
    append : bool, optional
        Whether to append (True) or overwrite (False) to a logfile at path,
        by default True

    Returns
    -------
    logging.Logger
        _description_
    """
    logger = logging.getLogger(name)
    for _ in range(len(logger.handlers)):
        logger.handlers.pop().close()  # remove and close existing handlers
    logging.captureWarnings(True)
    logger.setLevel(log_level)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(log_level)
    console.setFormatter(logging.Formatter(fmt))
    logger.addHandler(console)
    if path is not None:
        if append is False and os.path.isfile(path):
            os.unlink(path)
        add_filehandler(logger, path, log_level=log_level, fmt=fmt)
    logger.info(f"HydroMT version: {__version__}")

    return logger


def add_filehandler(logger, path, log_level=20, fmt=FMT):
    """Add file handler to logger."""
    if not os.path.isdir(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    isfile = os.path.isfile(path)
    ch = logging.FileHandler(path)
    ch.setFormatter(logging.Formatter(fmt))
    ch.setLevel(log_level)
    logger.addHandler(ch)
    if isfile:
        logger.debug(f"Appending log messages to file {path}.")
    else:
        logger.debug(f"Writing log messages to new file {path}.")


def logged(logger):
    """Define a decorator that logs the execution of a function.

    This decorator logs the input arguments, output, and any raised exceptions of
    the decorated function using the provided logger.

    Parameters
    ----------
    logger : Logger
        The logger object used to log the function execution.

    Returns
    -------
    wrap : callable
        The decorator function that wraps the decorated function.
    """

    def wrap(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            f = function.__name__
            logger.debug(f"{f} - args={args} kwargs={kwargs}")
            try:
                response = function(*args, **kwargs)
            except Exception as error:
                e = error.__class__.__name__
                emsg = str(error)
                logger.error(f"{f} - raised {e} with error '{emsg}'")
                raise
            logger.debug(f"{f} - return={response}")
            return response

        return wrapper

    return wrap
