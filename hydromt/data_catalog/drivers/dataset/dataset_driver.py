"""Driver for Datasets."""

from abc import ABC, abstractmethod
from logging import Logger, getLogger
from typing import List, Optional

import xarray as xr

from hydromt._typing import (
    SourceMetadata,
    StrPath,
    TimeRange,
    Variables,
)
from hydromt._typing.error import NoDataStrategy
from hydromt.data_catalog.drivers.base_driver import BaseDriver

logger: Logger = getLogger(__name__)


class DatasetDriver(BaseDriver, ABC):
    """Abstract Driver to read Datasets."""

    @abstractmethod
    def read(
        self,
        uris: List[str],
        *,
        variables: Optional[Variables] = None,
        time_range: Optional[TimeRange] = None,
        metadata: Optional[SourceMetadata] = None,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
    ) -> xr.Dataset:
        """
        Read in any compatible data source to an xarray Dataset.

        args:
        """
        ...

    def write(
        self,
        path: StrPath,
        ds: xr.Dataset,
        **kwargs,
    ) -> str:
        """
        Write out a Dataset to file.

        Not all drivers should have a write function, so this method is not
        abstract.

        args:
        """
        raise NotImplementedError(
            f"Writing using driver '{self.name}' is not supported."
        )
