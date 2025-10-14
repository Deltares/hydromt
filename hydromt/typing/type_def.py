"""All the definitions of type aliases used in HydroMT."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple, TypedDict, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from dateutil.parser import parse
from pydantic import (
    BaseModel,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)
from shapely.geometry.base import BaseGeometry
from typing_extensions import Annotated
from xarray import DataArray, Dataset

from hydromt.typing.model_mode import ModelMode


def _validate_bbox(
    bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    assert bbox[0] < bbox[2], (
        f"bbox minx: '{bbox[0]}' should be less than maxx: '{bbox[2]}'."
    )
    assert bbox[1] < bbox[3], (
        f"bbox miny: '{bbox[1]}' should be less than maxy: '{bbox[3]}'."
    )
    return bbox


DataType = Literal[
    "DataFrame", "DataSet", "GeoDataFrame", "GeoDataSet", "RasterDataset"
]
Bbox = Annotated[Tuple[float, float, float, float], _validate_bbox]


StrPath = Union[str, Path]
GeoDataframeSource = StrPath
GeoDatasetSource = StrPath
RasterDatasetSource = StrPath
DatasetSource = StrPath

Crs = int
TotalBounds = Tuple[Bbox, Crs]
Zoom = Union[int, Tuple[float, str]]  # level OR (resolution, unit)
Number = Union[int, float]
SourceSpecDict = TypedDict(
    "SourceSpecDict", {"source": str, "provider": str, "version": Union[str, int]}
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

DATETIME_FORMAT: str = "%Y-%m-%d_%H:%M:%S"


class TimeRange(BaseModel):
    """A time range with start and end datetime."""

    start: datetime = Field(..., description="Start of the time range.")
    end: datetime = Field(..., description="End of the time range.")

    @field_validator("start", "end", mode="before")
    def parse_datetime(cls, v: Union[str, datetime]) -> datetime:
        """Parse a datetime from string or numpy datetime64."""
        if isinstance(v, str):
            try:
                # first try to parse as exact format
                return datetime.strptime(v, DATETIME_FORMAT)
            except ValueError:
                # fallback to more flexible parsing
                return parse(v)
        elif isinstance(v, np.datetime64):
            return pd.to_datetime(v).to_pydatetime()
        return v

    @model_validator(mode="after")
    def validate(self) -> "TimeRange":
        """Validate the time range."""
        if self.start > self.end:
            raise ValueError(
                f"time range start: '{self.start}' should be less than end: '{self.end}'"
            )
        return self

    @field_serializer("start", "end", mode="plain")
    def serialize_datetime(self, v: datetime) -> str:
        """Serialize a datetime to string."""
        return v.strftime(DATETIME_FORMAT)

    @staticmethod
    def create(data: Any) -> "TimeRange":
        """Create a TimeRange from various input types."""
        if isinstance(data, TimeRange):
            return data
        elif isinstance(data, dict):
            return TimeRange(**data)
        elif isinstance(data, (list, tuple)) and len(data) == 2:
            return TimeRange(start=data[0], end=data[1])
        else:
            raise ValueError(
                f"Cannot create TimeRange from data: '{data}' of type '{type(data)}'."
            )
