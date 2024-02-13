"""Driver for RasterDataSets."""
from abc import ABC, abstractmethod
from logging import Logger
from typing import Optional

import xarray as xr
from pydantic import BaseModel
from pyproj import CRS

from hydromt._typing import Bbox, Geom


class RasterDataSetDriver(ABC, BaseModel):
    """Abstract Driver to read GeoDataFrames."""

    @abstractmethod
    def read(
        self,
        uri: str,
        bbox: Optional[Bbox] = None,
        mask: Optional[Geom] = None,
        buffer: float = 0.0,
        crs: Optional[CRS] = None,
        predicate: str = "intersects",
        zoom_level: int = 0,
        logger: Optional[Logger] = None,
        # handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,  # TODO:
        **kwargs,
    ) -> xr.Dataset:
        """
        Read in any compatible data source to an xarray Dataset.

        args:
        """
        ...
