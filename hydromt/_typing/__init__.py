"""A module for all of the type definitions used in HydroMT."""

from .crs import CRS
from .error import (
    NoDataException,
    NoDataStrategy,
    exec_nodata_strat,
)
from .fsspec_types import FS
from .metadata import SourceMetadata
from .model_mode import ModelMode
from .type_def import (
    Bbox,
    Crs,
    Data,
    DataType,
    DeferedFileClose,
    ExportConfigDict,
    GeoDataframeSource,
    GeoDatasetSource,
    Geom,
    GeomBuffer,
    GpdShapeGeom,
    ModeLike,
    Number,
    Pathdantic,
    Predicate,
    RasterDatasetSource,
    SourceSpecDict,
    StrPath,
    TimeRange,
    TotalBounds,
    Variables,
    XArrayDict,
    Zoom,
)

__all__ = [
    "Bbox",
    "Crs",
    "CRS",
    "StrPath",
    "DeferedFileClose",
    "ExportConfigDict",
    "FS",
    "GeoDataframeSource",
    "GeoDatasetSource",
    "ModeLike",
    "Number",
    "Pathdantic",
    "RasterDatasetSource",
    "SourceSpecDict",
    "TimeRange",
    "TotalBounds",
    "XArrayDict",
    "ModelMode",
    "NoDataStrategy",
    "NoDataException",
    "exec_nodata_strat",
    "Variables",
    "Geom",
    "GpdShapeGeom",
    "Data",
    "DataType",
    "GeomBuffer",
    "Predicate",
    "Zoom",
    "SourceMetadata",
]
