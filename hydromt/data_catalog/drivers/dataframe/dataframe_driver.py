"""Generic driver for reading and writing DataFrames."""

from abc import ABC, abstractmethod
from logging import Logger, getLogger
from typing import ClassVar, List, Optional

import pandas as pd

from hydromt._typing import (
    NoDataStrategy,
    SourceMetadata,
    StrPath,
    TimeRange,
    Variables,
)
from hydromt.data_catalog.drivers import BaseDriver

logger: Logger = getLogger(__name__)


class DataFrameDriver(BaseDriver, ABC):
    """Abstract Driver to read DataFrames."""

    supports_writing: ClassVar[bool] = False

    @abstractmethod
    def read(
        self,
        uris: List[str],
        *,
        variables: Optional[Variables] = None,
        time_range: Optional[TimeRange] = None,
        metadata: Optional[SourceMetadata] = None,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
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
