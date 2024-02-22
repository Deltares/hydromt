"""Driver for GeoDataFrames."""
from abc import ABC, abstractmethod
from logging import Logger, getLogger
from typing import List, Optional

import geopandas as gpd
from pydantic import BaseModel
from pyproj import CRS

# from hydromt.nodata import NoDataStrategy

logger: Logger = getLogger(__name__)


class GeoDataFrameDriver(ABC, BaseModel):
    """Abstract Driver to read GeoDataFrames."""

    @abstractmethod
    def read(
        self,
        uri: str,
        bbox: Optional[List[int]] = None,
        mask: Optional[gpd.GeoDataFrame] = None,
        buffer: float = 0.0,
        crs: Optional[CRS] = None,
        predicate: str = "intersects",
        logger: Optional[Logger] = logger,
    ) -> gpd.GeoDataFrame:
        """
        Read in any compatible data source to a geopandas `GeoDataFrame`.

        args:
        """
        ...
