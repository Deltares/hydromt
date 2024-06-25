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
    ZoomLevel,
)
from hydromt._typing.error import NoDataStrategy
from hydromt.data_catalog.drivers.base_driver import BaseDriver

logger = getLogger(__name__)


logger: Logger = getLogger(__name__)


class RasterDatasetDriver(BaseDriver, ABC):
    """Abstract Driver to read GeoDataFrames."""

    def read(
        self,
        uri: str,
        *,
        mask: Optional[Geom] = None,
        variables: Optional[Variables] = None,
        time_range: Optional[TimeRange] = None,
        zoom_level: Optional[ZoomLevel] = None,
        metadata: Optional[SourceMetadata] = None,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        # TODO: https://github.com/Deltares/hydromt/issues/802
    ) -> xr.Dataset:
        """
        Read in any compatible data source to an xarray Dataset.

        args:
            mask: Optional[Geom]. Mask for features to match the predicate, preferably
                in the same CRS.
        """
        # Merge static kwargs from the catalog with dynamic kwargs from the query.
        uris = self.uri_resolver.resolve(
            uri,
            self.filesystem,
            mask=mask,
            time_range=time_range,
            variables=variables,
            zoom_level=zoom_level,
            handle_nodata=handle_nodata,
            options=self.options,
        )
        return self.read_data(
            uris,
            mask=mask,
            time_range=time_range,
            variables=variables,
            zoom_level=zoom_level,
            metadata=metadata,
            handle_nodata=handle_nodata,
        )

    @abstractmethod
    def read_data(
        self,
        uris: List[str],
        *,
        mask: Optional[Geom] = None,
        variables: Optional[Variables] = None,
        time_range: Optional[TimeRange] = None,
        zoom_level: Optional[ZoomLevel] = None,
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
