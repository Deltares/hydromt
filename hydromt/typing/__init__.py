"""A module for all of the type definitions used in HydroMT."""

from hydromt.typing.crs import CRS
from hydromt.typing.fsspec_types import FSSpecFileSystem
from hydromt.typing.metadata import SourceMetadata
from hydromt.typing.model_mode import ModelMode
from hydromt.typing.type_def import (
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
