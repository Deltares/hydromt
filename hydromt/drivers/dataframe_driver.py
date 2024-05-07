"""Generic driver for reading and writing DataFrames."""
from abc import ABC, abstractmethod
from logging import Logger, getLogger
from typing import TYPE_CHECKING, List, Optional

import pandas as pd

from hydromt._typing import NoDataStrategy, StrPath, TimeRange, Variables
from hydromt.drivers import BaseDriver

if TYPE_CHECKING:
    from hydromt.data_source import SourceMetadata


logger: Logger = getLogger(__name__)


class DataFrameDriver(BaseDriver, ABC):
    """Abstract Driver to read DataFrames."""

    def read(
        self,
        uri: str,
        metadata: "SourceMetadata",
        *,
        variables: Optional[Variables] = None,
        time_range: Optional[TimeRange] = None,
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
            metadata,
            logger=logger,
            variables=variables,
            time_range=time_range,
            handle_nodata=handle_nodata,
        )
        return df

    @abstractmethod
    def read_data(
        self,
        uris: List[str],
        metadata: "SourceMetadata",
        *,
        variables: Optional[Variables] = None,
        time_range: Optional[TimeRange] = None,
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
