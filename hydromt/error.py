"""All of the types for handeling errors within HydroMT."""

import inspect
import logging
from enum import Enum


class DeprecatedError(Exception):
    """Simple custom class to raise an error for something that is now deprecated."""

    def __init__(self, msg: str):
        """Initialise the object."""
        self.base = "DeprecationError"
        self.message = msg

    def __str__(self):
        return f"{self.base}: {self.message}"


class NoDataStrategy(Enum):
    """Strategy to handle nodata values."""

    RAISE = "raise"
    WARN = "warn"
    IGNORE = "ignore"


class NoDataException(Exception):
    """Exception raised for errors in the input.

    Attributes
    ----------
        message -- explanation of the error
    """

    def __init__(self, message="No data available"):
        self.message = message
        super().__init__(self.message)


def exec_nodata_strat(msg: str, strategy: NoDataStrategy) -> None:
    """Execute nodata strategy.

    Uses the logger from the calling module if it has a logger.
    Otherwise creates a new logger with the calling module's name.
    Otherwise uses a backup logger from this current module.
    """
    frame = inspect.currentframe()
    caller_frame = frame.f_back if frame else None
    module = inspect.getmodule(caller_frame) if caller_frame else None
    logger_name = getattr(module, "__name__", __name__)
    logger = getattr(module, "logger", logging.getLogger(logger_name))

    if strategy == NoDataStrategy.RAISE:
        raise NoDataException(msg)
    elif strategy == NoDataStrategy.WARN:
        logger.warning(msg)
    elif strategy == NoDataStrategy.IGNORE:
        # do nothing
        pass
