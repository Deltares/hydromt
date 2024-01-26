"""Driver for GeoDataFrames."""
from abc import ABC, abstractmethod
from logging import Logger

import geopandas as gpd
from pydantic import BaseModel
from pyproj import CRS

# from hydromt.nodata import NoDataStrategy


class GeoDataFrameDriver(ABC, BaseModel):
    """Abstract Driver to read GeoDataFrames."""

    @abstractmethod
    def read(
        self,
        uri: str,
        bbox: list[int] | None = None,
        mask: gpd.GeoDataFrame | None = None,
        buffer: float = 0.0,
        crs: CRS | None = None,
        predicate: str = "intersects",
        logger: Logger | None = None,
        # handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
    ) -> gpd.GeoDataFrame:
        """
        Read in any compatible data source to a geopandas `GeoDataFrame`.

        args:
        """
        ...
