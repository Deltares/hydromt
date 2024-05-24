"""A module for all of the type definitions used in HydroMT."""

from .crs import CRS
from .error import (
    ErrorHandleMethod,
    NoDataException,
    NoDataStrategy,
    _exec_nodata_strat,
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
    ZoomLevel,
)

__all__ = [
    "Bbox",
    "Crs",  # TODO: Unify usage of CRS in the code
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
    "ErrorHandleMethod",
    "NoDataStrategy",
    "NoDataException",
    "_exec_nodata_strat",
    "Variables",
    "Geom",
    "GpdShapeGeom",
    "Data",
    "DataType",
    "GeomBuffer",
    "Predicate",
    "ZoomLevel",
    "SourceMetadata",
]
