"""Driver for RasterDatasets."""
from abc import ABC, abstractmethod
from logging import Logger
from typing import List, Optional

import xarray as xr
from pyproj import CRS

from hydromt._typing import Bbox, Geom
from hydromt._typing.error import NoDataStrategy

from .base_driver import BaseDriver


class RasterDatasetDriver(ABC, BaseDriver):
    """Abstract Driver to read GeoDataFrames."""

    @abstractmethod
    def read(
        self,
        uris: str,
        *,
        bbox: Optional[Bbox] = None,
        mask: Optional[Geom] = None,
        buffer: float = 0.0,
        crs: Optional[CRS] = None,
        variables: Optional[List[str]] = None,
        predicate: str = "intersects",
        zoom_level: int = 0,
        logger: Optional[Logger] = None,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        # TODO: https://github.com/Deltares/hydromt/issues/802
        **kwargs,
    ) -> xr.Dataset:
        """
        Read in any compatible data source to an xarray Dataset.

        args:
        """
        ...
