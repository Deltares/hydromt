"""RasterDatasetDriver for zarr data."""

from functools import partial
from logging import Logger
from typing import Callable, List, Optional

import xarray as xr
from pyproj import CRS

from hydromt._typing import Bbox, Geom
from hydromt._typing.error import NoDataStrategy
from hydromt.drivers.preprocessing import PREPROCESSORS
from hydromt.drivers.rasterdataset_driver import RasterDatasetDriver


class ZarrDriver(RasterDatasetDriver):
    """RasterDatasetDriver for zarr data."""

    name = "zarr"

    def read(
        self,
        uri: str,
        *,
        bbox: Optional[Bbox] = None,
        mask: Optional[Geom] = None,
        buffer: float = 0,
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
        Read zarr data to an xarray DataSet.

        Args:
        """
        uris = self.metadata_resolver.resolve(
            uri,
            bbox=bbox,
            mask=mask,
            buffer=buffer,
            predicate=predicate,
            variables=variables,
            zoom_level=zoom_level,
            handle_nodata=handle_nodata,
            **kwargs,
        )
        preprocessor: Optional[Callable] = None
        preprocessor_name: Optional[str] = kwargs.get("preprocess")
        if preprocessor_name:
            preprocessor = PREPROCESSORS.get(preprocessor_name)
            if not preprocessor:
                raise ValueError(f"unknown preprocessor: '{preprocessor_name}'")

        opn: Callable = partial(xr.open_zarr, **kwargs)

        return xr.merge(
            [preprocessor(opn(_uri)) if preprocessor else opn(_uri) for _uri in uris]
        )
