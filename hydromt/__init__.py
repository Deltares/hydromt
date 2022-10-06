"""HydroMT: Automated and reproducible model building and analysis"""

# version number without 'v' at start
__version__ = "0.5.1.dev"

import warnings

# required for accessor style documentation
from xarray import DataArray, Dataset

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
