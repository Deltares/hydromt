#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import wraps
import logging
import logging.handlers
import sys
import os
import logging

FMT = "%(asctime)s - %(name)s - %(module)s - %(levelname)s - %(message)s"
from . import __version__


def setuplog(name, path=None, log_level=20, fmt=FMT, append=True):
    """Set-up the logging on sys.stdout"""
    logger = logging.getLogger(name)
    logger.handlers = []  # remove earlier handlers
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
    def wrap(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            # logger = logging.getLogger(log)
            f = function.__name__
            # logger.debug(f"Calling '{f}'")
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
