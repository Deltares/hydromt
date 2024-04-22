"""Driver for GeoDataFrames."""

from abc import ABC, abstractmethod
from logging import Logger, getLogger
from typing import Any, Dict, List, Optional

import geopandas as gpd
from pyproj import CRS

from hydromt._typing import Geom, StrPath
from hydromt._typing.error import NoDataStrategy
from hydromt.drivers import BaseDriver

logger: Logger = getLogger(__name__)


class GeoDataFrameDriver(BaseDriver, ABC):
    """Abstract Driver to read GeoDataFrames."""

    def read(
        self,
        uri: str,
        *,
        mask: Optional[Geom] = None,
        crs: Optional[CRS] = None,
        variables: Optional[List[str]] = None,
        predicate: str = "intersects",
        logger: Logger = logger,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        # TODO: https://github.com/Deltares/hydromt/issues/802
        **kwargs,
    ) -> gpd.GeoDataFrame:
        """
        Read in any compatible data source to a geopandas `GeoDataFrame`.

        args:
        """
        # Merge static kwargs from the catalog with dynamic kwargs from the query.
        driver_kwargs: Dict[str, Any] = self.options | kwargs
        uris = self.metadata_resolver.resolve(
            uri,
            self.filesystem,
            mask=mask,
            variables=variables,
            handle_nodata=handle_nodata,
            **kwargs,
        )
        gdf = self.read_data(
            uris,
            mask=mask,
            crs=crs,
            predicate=predicate,
            logger=logger,
            handle_nodata=handle_nodata,
            **driver_kwargs,
        )
        return gdf

    @abstractmethod
    def read_data(
        self,
        uris: List[str],
        *,
        mask: Optional[Geom] = None,
        crs: Optional[CRS] = None,
        predicate: str = "intersects",
        logger: Logger = logger,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        **kwargs,
    ) -> gpd.GeoDataFrame:
        """Read in any compatible data source to a geopandas `GeoDataFrame`."""
        ...

    def write(
        self,
        path: StrPath,
        gdf: gpd.GeoDataFrame,
        **kwargs,
    ) -> None:
        """
        Write out a GeoDataFrame to file.

        Not all drivers should have a write function, so this method is not
        abstract.

        args:
        """
        raise NotImplementedError(
            f"Writing using driver '{self.name}' is not supported."
        )
