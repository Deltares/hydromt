"""HydroMT: Automated and reproducible model building and analysis."""

# version number without 'v' at start
__version__ = "1.3.0-rc3"

# This is only here to suppress the bug described in
# https://github.com/pydata/xarray/issues/7259
# We have to make sure that netcdf4 is imported before
# numpy is imported for the first time, e.g. also via
# importing xarray

import warnings  # noqa: F401

import netCDF4  # noqa: F401

# submodules
from hydromt import data_catalog, gis, io, model, stats

# high-level methods
from hydromt._utils.log import initialize_logging
from hydromt.data_catalog import DataCatalog
from hydromt.gis import raster, vector
from hydromt.model import Model, hydromt_step
from hydromt.plugins import PLUGINS

initialize_logging()

__all__ = [
    # high-level classes
    "DataCatalog",
    "Model",
    # submodules
    "data_catalog",
    "gis",
    "io",
    "model",
    "stats",
    # raster and vector accessor
    "raster",
    "vector",
    # high-level functions
    "hydromt_step",
    # plugins
    "PLUGINS",
]
