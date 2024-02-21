"""The module of HydroMT handeling file and path interactions."""

from .path import (
    make_path_abs_and_cross_platform,
)
from .readers import (
    configread,
    open_geodataset,
    open_mfcsv,
    open_mfraster,
    open_raster,
    open_raster_from_tindex,
    open_timeseries_from_table,
    open_vector,
    open_vector_from_table,
)
from .writers import (
    configwrite,
    netcdf_writer,
    write_xy,
    zarr_writer,
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
    "configwrite",
    "configread",
    "netcdf_writer",
    "zarr_writer",
]
