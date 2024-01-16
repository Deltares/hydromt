"""Driver for GeoDataFrames."""
from abc import abstractmethod
from logging import Logger

import geopandas as gpd
from pyproj import CRS

# from hydromt.nodata import NoDataStrategy
from .abstract_driver import AbstractDriver


class GeoDataFrameDriver(AbstractDriver):
    """Abstract Driver to read GeoDataFrames."""

    _crs: CRS

    def __init__(self, uri: str, crs: CRS):
        super.__init__(uri)
        self._crs = crs

    @property
    def crs(self) -> CRS:
        """Getter for CRS."""
        return self._crs

    @abstractmethod
    def read(
        self,
        bbox: list[int] | None = None,
        mask: gpd.GeoDataFrame | None = None,
        buffer: float = 0.0,
        predicate: str = "intersects",  # ??
        logger: Logger | None = None,
        # handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
    ) -> gpd.GeoDataFrame:
        """
        Read in any compatible data source to a geopandas `GeoDataFrame`.

        args:
        """
        ...
