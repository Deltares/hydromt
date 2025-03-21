"""HydroMT: Automated and reproducible model building and analysis."""

# version number without 'v' at start
__version__ = "1.0.1"

# This is only here to suppress the bug described in
# https://github.com/pydata/xarray/issues/7259
# We have to make sure that netcdf4 is imported before
# numpy is imported for the first time, e.g. also via
# importing xarray
import warnings

import netCDF4  # noqa: F401

# submodules
from . import _io, data_catalog, gis, model, stats

# high-level methods
from .data_catalog import DataCatalog
from .gis import raster, vector
from .model import Model, hydromt_step
from .plugins import PLUGINS

__all__ = [
    # high-level classes
    "DataCatalog",
    "Model",
    # submodules
    "data_catalog",
    "gis",
    "_io",
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
