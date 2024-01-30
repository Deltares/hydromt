"""Type aliases used by hydromt."""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple, TypedDict, Union

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

ExportConfigDict = TypedDict(
    "ExportConfigDict",
    {"args": Dict[str, Any], "meta": Dict[str, Any], "sources": List[SourceSpecDict]},
)


class ErrorHandleMethod(Enum):
    """Strategies for error handling within hydromt."""

    RAISE = 1
    SKIP = 2
    COERCE = 3


class ModelMode(Enum):
    """Modes that the model can be in."""

    READ = "r"
    WRITE = "w"
    FORCED_WRITE = "w+"
    APPEND = "r+"

    @staticmethod
    def from_str_or_mode(s: Union["ModelMode", str]) -> "ModelMode":
        """Construct a model mode from either a string or return provided if it's already a mode."""
        if isinstance(s, ModelMode):
            return s

        if s == "r":
            return ModelMode.READ
        elif s == "r+":
            return ModelMode.APPEND
        elif s == "w":
            return ModelMode.WRITE
        elif s == "w+":
            return ModelMode.FORCED_WRITE
        else:
            raise ValueError(f"Unknown mode: {s}, options are: r, r+, w, w+")

    def is_writing_mode(self):
        """Asster whether mode is writing or not."""
        return self in [ModelMode.WRITE, ModelMode.FORCED_WRITE]

    def is_reading_mode(self):
        """Asster whether mode is reading or not."""
        return self in [ModelMode.READ, ModelMode.APPEND]


ModeLike = Union[ModelMode, str]
