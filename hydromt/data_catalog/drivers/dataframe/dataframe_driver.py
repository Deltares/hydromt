"""Generic driver for reading and writing DataFrames."""
from abc import ABC, abstractmethod
from logging import Logger, getLogger
from typing import List, Optional

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

    def read(
        self,
        uri: str,
        *,
        variables: Optional[Variables] = None,
        time_range: Optional[TimeRange] = None,
        metadata: Optional[SourceMetadata] = None,
        logger: Logger = logger,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        # TODO: https://github.com/Deltares/hydromt/issues/802
    ) -> pd.DataFrame:
        """
        Read in any compatible data source to a pandas `DataFrame`.

        args:
        """
        # Merge static kwargs from the catalog with dynamic kwargs from the query.
        uris = self.metadata_resolver.resolve(
            uri,
            self.filesystem,
            variables=variables,
            time_range=time_range,
            handle_nodata=handle_nodata,
        )
        df = self.read_data(
            uris,
            logger=logger,
            variables=variables,
            time_range=time_range,
            metadata=metadata,
            handle_nodata=handle_nodata,
        )
        return df

    @abstractmethod
    def read_data(
        self,
        uris: List[str],
        *,
        variables: Optional[Variables] = None,
        time_range: Optional[TimeRange] = None,
        metadata: Optional[SourceMetadata] = None,
        logger: Logger = logger,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
    ) -> pd.DataFrame:
        """Read in any compatible data source to a pandas `DataFrame`."""
        ...

    def write(
        self,
        path: StrPath,
        df: pd.DataFrame,
        **kwargs,
    ) -> None:
        """
        Write out a DataFrame to file.

        Not all drivers should have a write function, so this method is not
        abstract.

        args:
        """
        raise NotImplementedError(
            f"Writing using driver '{self.name}' is not supported."
        )
