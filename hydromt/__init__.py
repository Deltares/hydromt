"""HydroMT: Automated and reproducible model building and analysis"""

# version number without 'v' at start
__version__ = "0.5.1.dev"

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


def _has_xugrid():
    try:
        import xugrid

        return True
    except ImportError:
        return False


# submoduls
from . import cli, workflows, stats, flw, raster, vector

# high-level methods
from .models import *
from .io import *
from .data_catalog import *

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
