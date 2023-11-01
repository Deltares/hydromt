"""Functions to handle nodata values."""
from enum import Enum
from logging import Logger

from .exceptions import NoDataException


class NoDataStrategy(Enum):
    """Strategy to handle nodata values."""

    RAISE = "raise"
    IGNORE = "ignore"


def _exec_nodata_strat(msg: str, strategy: NoDataStrategy, logger: Logger) -> None:
    """Execute nodata strategy."""
    if strategy == NoDataStrategy.RAISE:
        raise NoDataException(msg)
    elif strategy == NoDataStrategy.IGNORE:
        logger.warning(msg)
    else:
        raise NotImplementedError(f"NoDataStrategy '{strategy}' not implemented.")
