"""The module of HydroMT handeling file and path interactions."""

from .io import (
    configread,
    configwrite,
    netcdf_writer,
    open_geodataset,
    open_mfcsv,
    open_mfraster,
    open_raster,
    open_raster_from_tindex,
    open_timeseries_from_table,
    open_vector,
    open_vector_from_table,
    read_ini_config,
    write_ini_config,
    write_xy,
    zarr_writer,
)
from .path import (
    make_path_abs_and_cross_platform,
)

__all__ = [
    "make_path_abs_and_cross_platform",
    "open_raster",
    "open_mfraster",
    "open_mfcsv",
    "open_raster_from_tindex",
    "open_geodataset",
    "open_timeseries_from_table",
    "open_vector",
    "open_vector_from_table",
    "write_xy",
    "configread",
    "configwrite",
    "write_ini_config",
    "read_ini_config",
    "zarr_writer",
    "netcdf_writer",
]
