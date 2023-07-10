"""HydroMT: Automated and reproducible model building and analysis."""

# version number without 'v' at start
__version__ = "0.7.2.dev0"

# Set environment variables (this will be temporary)
# to use shapely 2.0 in favor of pygeos (if installed)
import os

os.environ["USE_PYGEOS"] = "0"

# pkg_resource deprication warnings originate from dependencies
# so silence them for now
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import dask
dask.config.set(scheduler='single-threaded')

# required for accessor style documentation
from xarray import DataArray, Dataset

# submodules
from . import cli, flw, raster, stats, vector, workflows
from .data_catalog import *
from .io import *

# high-level methods
from .models import *
