"""Module for the netcdf driver."""

from logging import Logger
from typing import Callable, List, Optional

import xarray as xr

from hydromt._typing import Geom, StrPath, TimeRange, ZoomLevel
from hydromt._typing.error import NoDataStrategy
from hydromt.drivers.preprocessing import PREPROCESSORS
from hydromt.drivers.rasterdataset_driver import RasterDatasetDriver


class NetcdfDriver(RasterDatasetDriver):
    """Driver for netcdf files."""

    name = "netcdf"

    def read_data(
        self,
        uris: List[str],
        *,
        mask: Optional[Geom] = None,
        time_range: Optional[TimeRange] = None,
        zoom_level: Optional[ZoomLevel] = None,
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

        # TODO: add **self.options, see https://github.com/Deltares/hydromt/issues/899
        return xr.open_mfdataset(uris, decode_coords="all", preprocess=preprocessor)

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
