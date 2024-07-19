"""Driver for GeoDataFrames."""

from abc import ABC, abstractmethod
from logging import Logger, getLogger
from typing import List, Optional

import geopandas as gpd

from hydromt._typing import Geom, SourceMetadata, StrPath
from hydromt._typing.error import NoDataStrategy
from hydromt.data_catalog.drivers import BaseDriver

logger: Logger = getLogger(__name__)


class GeoDataFrameDriver(BaseDriver, ABC):
    """Abstract Driver to read GeoDataFrames."""

    @abstractmethod
    def read(
        self,
        uris: List[str],
        *,
        mask: Optional[Geom] = None,
        predicate: str = "intersects",
        variables: Optional[List[str]] = None,
        metadata: Optional[SourceMetadata] = None,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
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
