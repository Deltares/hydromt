"""HydroMT: Automated and reproducible model building and analysis."""

# version number without 'v' at start
__version__ = "1.0.0-alpha"

# pkg_resource deprication warnings originate from dependencies
# so silence them for now
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


# required for accessor style documentation
from xarray import DataArray, Dataset  # noqa: F401

from . import cli, gis, stats, workflows
from .data_catalog import *

# submodules
from .gis import raster, vector
from .io import *

# high-level methods
from .models import *
