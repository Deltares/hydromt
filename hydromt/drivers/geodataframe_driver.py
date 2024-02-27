"""Driver for GeoDataFrames."""
from abc import ABC, abstractmethod
from logging import Logger, getLogger
from typing import List, Optional

import geopandas as gpd
from pydantic import BaseModel
from pyproj import CRS

from hydromt._typing import Bbox, Geom

logger: Logger = getLogger(__name__)


class GeoDataFrameDriver(ABC, BaseModel):
    """Abstract Driver to read GeoDataFrames."""

    @abstractmethod
    def read(
        self,
        uris: List[str],
        *,
        bbox: Optional[Bbox] = None,
        mask: Optional[Geom] = None,
        buffer: float = 0.0,
        crs: Optional[CRS] = None,
        predicate: str = "intersects",
        logger: Logger = logger,
        # handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,  # TODO
        **kwargs,
    ) -> gpd.GeoDataFrame:
        """
        Read in any compatible data source to a geopandas `GeoDataFrame`.

        args:
        """
        ...
