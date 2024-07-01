"""All the definitions of type aliases used in HydroMT."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple, TypedDict, Union

import geopandas as gpd
from dateutil.parser import parse
from pydantic import ValidationInfo, ValidatorFunctionWrapHandler
from pydantic.functional_validators import (
    AfterValidator,
    BeforeValidator,
    WrapValidator,
)
from shapely.geometry.base import BaseGeometry
from typing_extensions import Annotated
from xarray import DataArray, Dataset

from hydromt._typing.model_mode import ModelMode


def _time_range_from_str(
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


def _time_range_validate(tr: tuple[datetime, datetime]) -> tuple[datetime, datetime]:
    assert tr[0] >= tr[1], f"time range t0: '{tr[0]}' should be less than t1: '{tr[1]}'"
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
    "DataFrame", "DataSet", "GeoDataFrame", "GeoDataSet", "RasterDataset"
]
Bbox = Annotated[Tuple[float, float, float, float], _validate_bbox]


def _validate_path(
    path: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo
):
    if isinstance(path, str):
        path = Path(str)
    return handler(path, info)


Pathdantic = Annotated[Path, WrapValidator(_validate_path)]

StrPath = Union[str, Path]
GeoDataframeSource = StrPath
GeoDatasetSource = StrPath
RasterDatasetSource = StrPath
DatasetSource = StrPath

Crs = int
TotalBounds = Tuple[Bbox, Crs]
TimeRange = Annotated[
    Tuple[datetime, datetime],
    BeforeValidator(_time_range_from_str),
    AfterValidator(_time_range_validate),
]
ZoomLevel = Union[int, Tuple[float, str]]  # level OR (scale, resolution)
Number = Union[int, float]
SourceSpecDict = TypedDict(
    "SourceSpecDict", {"source": str, "provider": str, "version": Union[str, int]}
)

DeferedFileClose = TypedDict(
    "DeferedFileClose",
    {
        "ds": Union[Dataset, DataArray],
        "original_path": str,
        "temp_path": str,
        "close_attempts": int,
    },
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

Data = Union[Dataset, DataArray]

Variables = Union[str, List[str]]

GeomBuffer = int

ModeLike = Union[ModelMode, str]
