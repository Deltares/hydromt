"""HydroMT: Build and analyze models like a data-wizard!"""

__version__ = "0.4.1.dev"

import geopandas as gpd
from os.path import join, isdir, dirname, basename, isfile, abspath
import glob

# required for accessor style documentation
from xarray import DataArray, Dataset

try:
    import pygeos

    gpd.options.use_pygeos = True
except ImportError:
    pass

try:
    import pcraster as pcr

    HAS_PCRASTER = True
except ImportError:
    HAS_PCRASTER = False


# submoduls
from . import cli, workflows, stats, flw, raster, vector

# high-level methods
from .models import *
from .io import *
from .data_adapter import *
