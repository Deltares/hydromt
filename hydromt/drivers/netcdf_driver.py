"""Module for the netcdf driver."""

from logging import Logger
from typing import Callable, List, Optional

import xarray as xr
from pyproj import CRS

from hydromt._typing import Bbox, Geom, StrPath, TimeRange
from hydromt._typing.error import NoDataStrategy
from hydromt.drivers.preprocessing import PREPROCESSORS
from hydromt.drivers.rasterdataset_driver import RasterDatasetDriver


class NetcdfDriver(RasterDatasetDriver):
    """Driver for netcdf files."""

    name = "netcdf"

    def read(
        self,
        uri: str,
        *,
        bbox: Optional[Bbox] = None,
        mask: Optional[Geom] = None,
        buffer: float = 0,
        crs: Optional[CRS] = None,
        variables: Optional[List[str]] = None,
        time_range: Optional[TimeRange] = None,
        predicate: str = "intersects",
        zoom_level: int = 0,
        logger: Optional[Logger] = None,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        **kwargs,
    ) -> xr.Dataset:
        """Read netcdf data."""
        preprocessor: Optional[Callable] = None
        preprocessor_name: Optional[str] = kwargs.get("preprocess")
        if preprocessor_name:
            preprocessor = PREPROCESSORS.get(preprocessor_name)
            if not preprocessor:
                raise ValueError(f"unknown preprocessor: '{preprocessor_name}'")
            kwargs.update({"preprocess": preprocessor})

        uris: List[str] = self.metadata_resolver.resolve(
            uri,
            self.filesystem,
            bbox=bbox,
            mask=mask,
            buffer=buffer,
            predicate=predicate,
            variables=variables,
            time_range=time_range,
            zoom_level=zoom_level,
            handle_nodata=handle_nodata,
            **kwargs,
        )

        return xr.open_mfdataset(uris, decode_coords="all", **kwargs)

    def write(
        self,
        path: StrPath,
        ds: xr.Dataset,
        **kwargs,
    ):
        """
        Write a dataset to netcdf.

        args:
        """
        ds.to_netcdf(path, **kwargs)
