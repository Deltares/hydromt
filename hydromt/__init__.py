"""HydroMT: Automated and reproducible model building and analysis."""

# version number without 'v' at start
__version__ = "0.9.2.dev"

# pkg_resource deprication warnings originate from dependencies
# so silence them for now
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


# required for accessor style documentation
from xarray import DataArray, Dataset

# submodules
from . import cli, flw, raster, stats, vector, workflows
from .data_catalog import *
from .io import *

# high-level methods
from .models import *
