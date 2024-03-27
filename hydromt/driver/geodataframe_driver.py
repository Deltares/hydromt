"""Driver for GeoDataFrames."""

from abc import ABC, abstractmethod
from logging import Logger, getLogger
from typing import List, Optional

import geopandas as gpd
from pyproj import CRS

from hydromt._typing import Bbox, Geom
from hydromt._typing.error import NoDataStrategy
from hydromt.driver import BaseDriver

logger: Logger = getLogger(__name__)


class GeoDataFrameDriver(BaseDriver, ABC):
    """Abstract Driver to read GeoDataFrames."""

    @abstractmethod
    def read(
        self,
        uri: str,
        *,
        bbox: Optional[Bbox] = None,
        mask: Optional[Geom] = None,
        buffer: float = 0.0,
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
        ...
