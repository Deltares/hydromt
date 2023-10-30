"""Type aliases used by hydromt."""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Tuple, Union

GeoDataframeSource = Union[str, Path]
GeoDatasetSource = Union[str, Path]
RasterDatasetSource = Union[str, Path]
Bbox = Tuple[float, float, float, float]
Crs = int
TotalBounds = Tuple[Bbox, Crs]
TimeRange = Tuple[datetime, datetime]


class ErrorHandleMethod(Enum):
    """Strategies for error handling withing hydromt."""

    RAISE = 1
    SKIP = 2
    COERCE = 3
