"""Type aliases used by hydromt."""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple, TypedDict, Union

GeoDataframeSource = Union[str, Path]
GeoDatasetSource = Union[str, Path]
RasterDatasetSource = Union[str, Path]
Bbox = Tuple[float, float, float, float]
Crs = int
TotalBounds = Tuple[Bbox, Crs]
TimeRange = Tuple[datetime, datetime]

SourceSpecDict = TypedDict(
    "SourceSpecDict", {"source": str, "provider": str, "version": Union[str, int]}
)
Extent = TypedDict(
    "Extent",
    {
        "bbox": Tuple[float, float, float, float],
        "time_range": Tuple[datetime, datetime],
    },
)

ExportConfigDict = TypedDict(
    "ExportConfigDict",
    {"args": Dict[str, Any], "meta": Dict[str, Any], "sources": List[SourceSpecDict]},
)


class ErrorHandleMethod(Enum):
    """Strategies for error handling withing hydromt."""

    RAISE = 1
    SKIP = 2
    COERCE = 3
