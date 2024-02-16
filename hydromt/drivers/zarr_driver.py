"""RasterDataSetDriver for zarr data."""
from logging import Logger
from typing import List, Optional

import xarray as xr
from pyproj import CRS

from hydromt._typing import Bbox, Geom
from hydromt.drivers.rasterdataset_driver import RasterDataSetDriver


class ZarrDriver(RasterDataSetDriver):
    """RasterDataSetDriver for zarr data."""

    def read(
        self,
        uris: List[str],
        bbox: Optional[Bbox] = None,
        mask: Optional[Geom] = None,
        buffer: float = 0,
        crs: Optional[CRS] = None,
        predicate: str = "intersects",
        zoom_level: int = 0,
        logger: Optional[Logger] = None,
        **kwargs,
    ) -> xr.Dataset:
        """
        Read zarr data to an xarray DataSet.

        Args:
        """
        return xr.merge([xr.open_zarr(uri, **kwargs) for uri in uris])
