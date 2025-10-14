"""Driver for RasterDatasets."""

import logging
from abc import ABC, abstractmethod
from typing import Any

import xarray as xr

from hydromt.data_catalog.drivers.base_driver import (
    BaseDriver,
)
from hydromt.error import NoDataStrategy
from hydromt.typing import (
    Geom,
    SourceMetadata,
    StrPath,
    TimeRange,
    Variables,
    Zoom,
)

logger = logging.getLogger(__name__)


class RasterDatasetDriver(BaseDriver, ABC):
    """Abstract Driver to read GeoDataFrames."""

    @abstractmethod
    def read(
        self,
        uris: list[str],
        *,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        kwargs_for_open: dict[str, Any] | None = None,
        mask: Geom | None = None,
        variables: Variables | None = None,
        time_range: TimeRange | None = None,
        zoom: Zoom | None = None,
        chunks: dict[str, Any] | None = None,
        metadata: SourceMetadata | None = None,
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
