"""A module for all of the type definitions used in HydroMT."""

from hydromt._typing.crs import CRS
from hydromt._typing.fsspec_types import FSSpecFileSystem
from hydromt._typing.metadata import SourceMetadata
from hydromt._typing.model_mode import ModelMode
from hydromt._typing.type_def import (
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
    "ExportConfigDict",
    "FSSpecFileSystem",
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
