"""A module for all of the type definitions used in HydroMT."""

from .error import ErrorHandleMethod
from .model_mode import ModelMode
from .type_def import (
    Bbox,
    Crs,
    DeferedFileClose,
    ExportConfigDict,
    GeoDataframeSource,
    GeoDatasetSource,
    ModeLike,
    Number,
    RasterDatasetSource,
    SourceSpecDict,
    TimeRange,
    TotalBounds,
    XArrayDict,
)

__all__ = [
    "Bbox",
    "Crs",
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
]
