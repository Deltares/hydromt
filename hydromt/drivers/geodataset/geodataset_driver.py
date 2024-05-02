"""Driver for RasterDatasets."""

from abc import ABC, abstractmethod
from logging import Logger, getLogger
from typing import List, Optional

import xarray as xr

from hydromt._typing import Geom, StrPath, TimeRange
from hydromt._typing.error import NoDataStrategy
from hydromt._typing.type_def import Bbox, GeomBuffer, Predicate, Variables
from hydromt.drivers import BaseDriver

logger = getLogger(__name__)


class GeoDatasetDriver(BaseDriver, ABC):
    """Abstract Driver to read GeoDatasets."""

    def read(
        self,
        uri: str,
        *,
        bbox: Optional[Bbox] = None,
        geom: Optional[Geom] = None,
        buffer: GeomBuffer = 0,
        predicate: Predicate = "intersects",
        variables: Optional[List[str]] = None,
        time_range: Optional[TimeRange] = None,
        single_var_as_array: bool = True,
        logger: Logger = logger,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        # TODO: https://github.com/Deltares/hydromt/issues/802
    ) -> Optional[xr.Dataset]:
        """
        Read in any compatible data source to an xarray Dataset.

        args:
            mask: Optional[Geom]. Mask for features to match the predicate, preferably
                in the same CRS.
        """
        # Merge static kwargs from the catalog with dynamic kwargs from the query.
        uris = self.metadata_resolver.resolve(
            uri,
            fs=self.filesystem,
            time_range=time_range,
            mask=geom,
            variables=variables,
            handle_nodata=handle_nodata,
        )
        return self.read_data(
            uris,
            bbox=bbox,
            geom=geom,
            buffer=buffer,
            predicate=predicate,
            variables=variables,
            time_range=time_range,
            single_var_as_array=single_var_as_array,
            logger=logger,
            handle_nodata=handle_nodata,
        )

    @abstractmethod
    def read_data(
        self,
        uris: List[str],
        *,
        bbox: Optional[Bbox] = None,
        geom: Optional[Geom] = None,
        buffer: GeomBuffer = 0,
        predicate: Predicate = "intersects",
        variables: Optional[Variables] = None,
        time_range: Optional[TimeRange] = None,
        single_var_as_array: bool = True,
        logger: Logger,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        **kwargs,
    ) -> Optional[xr.Dataset]:
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
