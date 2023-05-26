"""HydroMT: Automated and reproducible model building and analysis."""

# version number without 'v' at start
__version__ = "0.7.2.dev0"

# Set environment variables (this will be temporary)
# to use shapely 2.0 in favor of pygeos (if installed)
import os

os.environ["USE_PYGEOS"] = "0"

# required for accessor style documentation
from xarray import DataArray, Dataset

# submodules
from . import cli, flw, raster, stats, vector, workflows
from .data_catalog import (
    DataAdapter,
    DataCatalog,
    DataFrameAdapter,
    GeoDataFrameAdapter,
    GeoDatasetAdapter,
)
from .io import (
    open_geodataset,
    open_mfraster,
    open_raster,
    open_raster_from_tindex,
    open_timeseries_from_table,
    open_vector,
    open_vector_from_table,
    write_xy,
)

# high-level methods
from .models import (
    MODELS,
    GridModel,
    LumpedModel,
    MeshModel,
    Model,
    ModelCatalog,
    NetworkModel,
    model_api,
    model_grid,
    model_lumped,
    model_mesh,
    model_network,
    model_plugins,
)

__all__ = [
    "DataAdapter",
    "DataArray",
    "DataCatalog",
    "DataFrameAdapter",
    "Dataset",
    "GeoDataFrameAdapter",
    "GeoDatasetAdapter",
    "GridModel",
    "LumpedModel",
    "MODELS",
    "MeshModel",
    "Model",
    "ModelCatalog",
    "NetworkModel",
    "cli",
    "flw",
    "model_api",
    "model_grid",
    "model_lumped",
    "model_mesh",
    "model_network",
    "model_plugins",
    "open_geodataset",
    "open_mfraster",
    "open_raster",
    "open_raster_from_tindex",
    "open_timeseries_from_table",
    "open_vector",
    "open_vector_from_table",
    "raster",
    "stats",
    "vector",
    "workflows",
    "write_xy",
]
