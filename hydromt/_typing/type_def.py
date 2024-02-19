"""All the definitions of type aliases used in HydroMT."""
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple, TypedDict, Union

import geopandas as gpd
from dateutil.parser import parse
from pydantic.functional_validators import AfterValidator, BeforeValidator
from shapely.geometry.base import BaseGeometry
from typing_extensions import Annotated
from xarray import DataArray, Dataset
from xugrid import Ugrid1d, Ugrid2d, UgridDataArray, UgridDataset

from hydromt._typing.model_mode import ModelMode

BasinIdType = str
UgridData = Union[UgridDataArray, UgridDataset, Ugrid1d, Ugrid2d]


def _time_tuple_from_str(
    t: Tuple[Union[str, datetime], Union[str, datetime]],
) -> "TimeRange":
    if isinstance(t[0], str):
        t0 = parse(t[0])
    else:
        t0 = t[0]
    if isinstance(t[1], str):
        t1 = parse(t[1])
    else:
        t1 = t[1]
    return (t0, t1)


def _timerange_validate(tr: tuple[datetime, datetime]) -> tuple[datetime, datetime]:
    assert tr[0] >= tr[1], f"timerange t0: '{tr[0]}' should be less than t1: '{tr[1]}'"
    return tr


def _validate_bbox(
    bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    assert (
        bbox[0] < bbox[2]
    ), f"bbox minx: '{bbox[0]}' should be less than maxx: '{bbox[2]}'."
    assert (
        bbox[1] < bbox[3]
    ), f"bbox miny: '{bbox[1]}' should be less than maxy: '{bbox[3]}'."
    return bbox


DataType = Literal[
    "DataFrame", "DataSet", "GeoDataFrame", "GeoDataSet", "RasterDataSet"
]
GeoDataframeSource = Union[str, Path]
GeoDatasetSource = Union[str, Path]
RasterDatasetSource = Union[str, Path]
Bbox = Annotated[Tuple[float, float, float, float], _validate_bbox]

StrPath = Union[str, Path]
GeoDataframeSource = StrPath
GeoDatasetSource = StrPath
RasterDatasetSource = StrPath
DatasetSource = StrPath

Bbox = Tuple[float, float, float, float]
Crs = int
TotalBounds = Tuple[Bbox, Crs]
TimeRange = Annotated[
    Tuple[datetime, datetime],
    BeforeValidator(_time_tuple_from_str),
    AfterValidator(_timerange_validate),
]
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

Geom = Union[gpd.GeoDataFrame, gpd.GeoSeries]
GpdShapeGeom = Union[Geom, BaseGeometry]
DataLike = Union[str, SourceSpecDict, Path, Dataset, DataArray]

Data = Union[Dataset, DataArray]

Variables = Union[str, List[str]]

GeomBuffer = int

ModeLike = Union[ModelMode, str]
