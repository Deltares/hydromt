"""Driver for GeoDataFrames."""
from abc import ABC, abstractmethod
from logging import Logger
from typing import Optional

import geopandas as gpd
from pydantic import BaseModel

from hydromt._typing.type_def import Predicate
from hydromt.region import Region

# from hydromt.nodata import NoDataStrategy


class GeoDataFrameDriver(ABC, BaseModel):
    """Abstract Driver to read GeoDataFrames."""

    @abstractmethod
    def read(
        self,
        uri: str,
        region: Optional[Region] = None,
        predicate: Predicate = "intersects",
        logger: Optional[Logger] = None,
    ) -> gpd.GeoDataFrame:
        """
        Read in any compatible data source to a geopandas `GeoDataFrame`.

        args:
        """
        ...
