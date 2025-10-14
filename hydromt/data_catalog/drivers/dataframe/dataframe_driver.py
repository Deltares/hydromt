"""Generic driver for reading and writing DataFrames."""

import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import pandas as pd

from hydromt.data_catalog.drivers import BaseDriver
from hydromt.error import NoDataStrategy
from hydromt.typing import (
    SourceMetadata,
    StrPath,
    TimeRange,
    Variables,
)

logger = logging.getLogger(__name__)


class DataFrameDriver(BaseDriver, ABC):
    """Abstract Driver to read DataFrames."""

    supports_writing: ClassVar[bool] = False

    @abstractmethod
    def read(
        self,
        uris: list[str],
        *,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        kwargs_for_open: dict[str, Any] | None = None,
        variables: Variables | None = None,
        time_range: TimeRange | None = None,
        metadata: SourceMetadata | None = None,
    ) -> pd.DataFrame:
        """Read in any compatible data source to a pandas `DataFrame`."""
        ...

    def write(
        self,
        path: StrPath,
        df: pd.DataFrame,
        **kwargs,
    ) -> str:
        """
        Write out a DataFrame to file.

        Not all drivers should have a write function, so this method is not
        abstract.

        args:
        """
        raise NotImplementedError(
            f"Writing using driver '{self.name}' is not supported."
        )
