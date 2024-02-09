"""A module for all of the type definitions used in HydroMT."""

from .error import (
    ErrorHandleMethod,
    NoDataException,
    NoDataStrategy,
    _exec_nodata_strat,
)
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
    Predicate,
    RasterDatasetSource,
    SourceSpecDict,
    StrPath,
    TimeRange,
    TotalBounds,
    Variables,
    XArrayDict,
)

__all__ = [
    "Bbox",
    "Crs",
    "StrPath",
    "DeferedFileClose",
    "ExportConfigDict",
    "GeoDataframeSource",
    "GeoDatasetSource",
    "ModeLike",
    "Number",
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
]
