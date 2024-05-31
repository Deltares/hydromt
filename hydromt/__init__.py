"""HydroMT: Automated and reproducible model building and analysis."""

# version number without 'v' at start
__version__ = "1.0.0-alpha"

# This is only here to suppress the bug described in
# https://github.com/pydata/xarray/issues/7259
# We have to make sure that netcdf4 is imported before
# numpy is imported for the first time, e.g. also via
# importing xarray
import netCDF4  # noqa: F401

# required for accessor style documentation
from xarray import DataArray, Dataset  # noqa: F401

from hydromt.hydromt_step import hydromt_step

from . import cli, gis, stats, workflows
from .data_catalog import *

# submodules
from .gis import raster, vector
from .io import *

# high-level methods
from .model import *
