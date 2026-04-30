"""HydroMT: Automated and reproducible model building and analysis."""

# version number without 'v' at start
__version__ = "1.4.0.dev0"


import warnings

try:
    # Rasterio 1.5 installs a broken sys.excepthook that recurses infinitely
    # on interpreter shutdown. Reset it to a safe version that prints the exception
    # and then calls the original hook wrapped in a try-except.
    # See: https://github.com/rasterio/rasterio/issues/3563
    import sys as _sys
    import traceback as _traceback

    import rasterio as _rasterio

    _original_excepthook = _sys.excepthook

    def safe_excepthook(exc_type, exc_value, exc_tb):
        try:
            _original_excepthook(exc_type, exc_value, exc_tb)
        except Exception:
            _traceback.print_exception(exc_type, exc_value, exc_tb, file=_sys.stderr)

    _sys.excepthook = safe_excepthook
except Exception:
    warnings.warn(
        "Failed to patch the broken sys.excepthook installed by rasterio 1.5. "
        "This may cause a harmless, but annoying recursive error on interpreter "
        "shutdown. To fix this, upgrade rasterio to a version that resolves the issue "
        "or downgrade to rasterio <1.5. You can safely ignore this warning if you do "
        "not experience shutdown errors.",
        category=RuntimeWarning,
        stacklevel=2,
    )


# This is only here to suppress the bug described in
# https://github.com/pydata/xarray/issues/7259
# We have to make sure that netcdf4 is imported before
# numpy is imported for the first time, e.g. also via
# importing xarray

import warnings  # noqa: F401

import netCDF4  # noqa: F401

# submodules
from hydromt import data_catalog, gis, model, stats

# high-level methods
from hydromt.data_catalog import DataCatalog
from hydromt.gis import raster, vector
from hydromt.model import Model
from hydromt.model.steps import hydromt_step
from hydromt.plugins import PLUGINS

__all__ = [
    # high-level classes
    "DataCatalog",
    "Model",
    # submodules
    "data_catalog",
    "gis",
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
