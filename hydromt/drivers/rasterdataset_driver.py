"""Driver for RasterDatasets."""

from abc import ABC, abstractmethod
from logging import Logger
from typing import List, Optional

import xarray as xr

from hydromt._typing import Geom, StrPath, TimeRange, ZoomLevel
from hydromt._typing.error import NoDataStrategy

from .base_driver import BaseDriver


class RasterDatasetDriver(BaseDriver, ABC):
    """Abstract Driver to read GeoDataFrames."""

    def read(
        self,
        uri: str,
        *,
        mask: Optional[Geom] = None,
        variables: Optional[List[str]] = None,
        time_range: Optional[TimeRange] = None,
        zoom_level: Optional[ZoomLevel] = None,
        logger: Optional[Logger] = None,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        # TODO: https://github.com/Deltares/hydromt/issues/802
        **kwargs,
    ) -> xr.Dataset:
        """Read in any compatible data source to an xarray Dataset."""
        uris = self.metadata_resolver.resolve(
            uri,
            self.filesystem,
            mask=mask,
            time_range=time_range,
            variables=variables,
            zoom_level=zoom_level,
            handle_nodata=handle_nodata,
            **kwargs,
        )
        return self.read_data(
            uris,
            mask=mask,
            time_range=time_range,
            zoom_level=zoom_level,
            logger=logger,
            handle_nodata=handle_nodata,
            **kwargs,
        )

    @abstractmethod
    def read_data(
        self,
        uris: List[str],
        *,
        mask: Optional[Geom] = None,
        time_range: Optional[TimeRange] = None,
        zoom_level: Optional[ZoomLevel] = None,
        logger: Logger,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        **kwargs,
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
    ) -> None:
        """
        Write out a RasterDataset to file.

        Not all drivers should have a write function, so this method is not
        abstract.

        args:
        """
        raise NotImplementedError(
            f"Writing using driver '{self.name}' is not supported."
        )
