"""HydroMT: Automated and reproducible model building and analysis"""

# version number without 'v' at start
__version__ = "0.7.1.dev"

# Set environment variables (this will be temporary)
# to use shapely 2.0 in favor of pygeos (if installed)
import os

os.environ["USE_PYGEOS"] = "0"

# submodules
from . import cli, workflows, stats, flw, raster, vector

# high-level methods
from .models import *
from .io import *
from .data_catalog import *

# required for accessor style documentation
from xarray import DataArray, Dataset
