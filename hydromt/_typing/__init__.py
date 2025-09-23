"""A module for all of the type definitions used in HydroMT."""

from .crs import CRS
from .deferred_file_close import DeferredFileClose
from .fsspec_types import FS
from .metadata import SourceMetadata
from .model_mode import ModelMode
from .type_def import (
    Bbox,
    Crs,
    Data,
    DataType,
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
    "DeferredFileClose",
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
