"""All of the types for handeling errors within HydroMT."""
from enum import Enum
from logging import Logger


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


def _exec_nodata_strat(msg: str, strategy: NoDataStrategy, logger: Logger) -> None:
    """Execute nodata strategy."""
    if strategy == NoDataStrategy.RAISE:
        raise NoDataException(msg)
    elif strategy == NoDataStrategy.IGNORE:
        logger.warning(msg)
    else:
        raise NotImplementedError(f"NoDataStrategy '{strategy}' not implemented.")


class ErrorHandleMethod(Enum):
    """Strategies for error handling within hydromt."""

    RAISE = 1
    SKIP = 2
    COERCE = 3
