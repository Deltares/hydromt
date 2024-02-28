"""RasterDatasetDriver for zarr data."""
from functools import partial
from logging import Logger
from typing import Callable, List, Optional

import xarray as xr
from pyproj import CRS

from hydromt._typing import Bbox, Geom
from hydromt.drivers.preprocessing import PREPROCESSORS
from hydromt.drivers.rasterdataset_driver import RasterDatasetDriver


class ZarrDriver(RasterDatasetDriver):
    """RasterDatasetDriver for zarr data."""

    def read(
        self,
        uris: List[str],
        *,
        bbox: Optional[Bbox] = None,
        mask: Optional[Geom] = None,
        buffer: float = 0,
        crs: Optional[CRS] = None,
        predicate: str = "intersects",
        zoom_level: int = 0,
        logger: Optional[Logger] = None,
        # handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        # TODO: https://github.com/Deltares/hydromt/issues/802
        **kwargs,
    ) -> xr.Dataset:
        """
        Read zarr data to an xarray DataSet.

        Args:
        """
        preprocess: str = kwargs.get("preprocess")
        if preprocess:
            preprocess: Callable = PREPROCESSORS.get(preprocess)

        opn: Callable = partial(xr.open_zarr, **kwargs)
        return xr.merge(
            [preprocess(opn(uri)) if preprocess else opn(uri) for uri in uris]
        )
