"""Driver for RasterDatasets."""

from abc import ABC, abstractmethod
from logging import Logger, getLogger
from typing import List, Optional

import xarray as xr

from hydromt._typing import (
    Geom,
    SourceMetadata,
    StrPath,
    TimeRange,
    Variables,
    Zoom,
)
from hydromt._typing.error import NoDataStrategy
from hydromt.data_catalog.drivers.base_driver import BaseDriver

logger = getLogger(__name__)


logger: Logger = getLogger(__name__)


class RasterDatasetDriver(BaseDriver, ABC):
    """Abstract Driver to read GeoDataFrames."""

    @abstractmethod
    def read(
        self,
        uris: List[str],
        *,
        mask: Optional[Geom] = None,
        variables: Optional[Variables] = None,
        time_range: Optional[TimeRange] = None,
        zoom_level: Optional[Zoom] = None,
        chunks: Optional[dict] = None,
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
        Write out a RasterDataset to file.

        Not all drivers should have a write function, so this method is not
        abstract.

        args:
        """
        raise NotImplementedError(
            f"Writing using driver '{self.name}' is not supported."
        )
