"""All the definitions of type aliases used in HydroMT."""
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple, TypedDict, Union

from geopandas import GeoDataFrame, GeoSeries
from xarray import DataArray, Dataset

from hydromt._compat import HAS_XUGRID
from hydromt._typing.model_mode import ModelMode

if HAS_XUGRID:
    from xugrid import Ugrid1d, Ugrid2d, UgridDataArray, UgridDataset

    BasinIdType = str
    UgridData = Union[UgridDataArray, UgridDataset, Ugrid1d, Ugrid2d]

StrPath = Union[str, Path]
GeoDataframeSource = StrPath
GeoDatasetSource = StrPath
RasterDatasetSource = StrPath
DatasetSource = StrPath

Bbox = Tuple[float, float, float, float]
Crs = int
TotalBounds = Tuple[Bbox, Crs]
TimeRange = Tuple[datetime, datetime]
Number = Union[int, float]
SourceSpecDict = TypedDict(
    "SourceSpecDict", {"source": str, "provider": str, "version": Union[str, int]}
)

DeferedFileClose = TypedDict(
    "DeferedFileClose",
    {"ds": Dataset, "org_fn": str, "tmp_fn": str, "close_attempts": int},
)
XArrayDict = Dict[str, Union[DataArray, Dataset]]

ExportConfigDict = TypedDict(
    "ExportConfigDict",
    {"args": Dict[str, Any], "meta": Dict[str, Any], "sources": List[SourceSpecDict]},
)

Predicate = Literal[
    "intersects", "within", "contains", "overlaps", "crosses", "touches"
]

Geom = Union[GeoDataFrame, GeoSeries]

Data = Union[Dataset, DataArray]

Variables = Union[str, List[str]]

GeomBuffer = int


ModeLike = Union[ModelMode, str]
