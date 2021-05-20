"""HydroMT: Build and analyze models like a data-wizard!"""

__version__ = "0.4.1"

import geopandas as gpd
import warnings

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

# TODO remove after fixed in all plugins
def __getattr__(name):
    if name in PLUGINS:
        ep = ENTRYPOINTS[PLUGINS[name]]
        plugin_name = ep.module_name.split(".")[0]
        warnings.warn(
            f'"hydromt.{name}" will be deprecated, use "{plugin_name}.{name}" instead.',
            DeprecationWarning,
        )
        return model_plugins.load(ep)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
