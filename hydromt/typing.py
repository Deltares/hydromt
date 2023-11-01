"""Type aliases used by hydromt."""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Tuple, TypedDict, Union

from xarray import DataArray, Dataset

GeoDataframeSource = Union[str, Path]
GeoDatasetSource = Union[str, Path]
RasterDatasetSource = Union[str, Path]
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


class AdapterType(Enum):
    GEODATAFRAME = 1
    DATAFRAME = 2
    RASTERDATSET = 3
    GEODATASET = 4


class ErrorHandleMethod(Enum):
    """Strategies for error handling withing hydromt."""

    RAISE = 1
    SKIP = 2
    COERCE = 3
