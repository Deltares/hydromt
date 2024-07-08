"""Pydantic models for the validation of region specifications."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, model_validator

from hydromt._typing.type_def import Bbox


class PathRegion(BaseModel):
    """A validation model for a region loaded from a file."""

    path: Path

    @staticmethod
    def from_path(path: Union[Path, str]) -> "PathRegion":
        """Create a region that will be loaded from a file."""
        if isinstance(path, Path):
            return PathRegion(path=path)
        else:
            return PathRegion(path=Path(path))

    @model_validator(mode="after")
    def _check_path_exists(self) -> "PathRegion":
        if not self.path.exists():
            raise ValueError(f"Path not found at {self.path}")
        return self


class BoundingBoxRegion(BaseModel):
    """A validation model for a region described by a bounding box in WGS84 space."""

    xmin: float = Field(ge=-180)
    xmax: float = Field(le=180)
    ymin: float = Field(ge=-90)
    ymax: float = Field(le=90)

    @staticmethod
    def from_list(
        input: Union[Tuple[float, float, float, float], List[float]],
    ) -> "BoundingBoxRegion":
        """Create a region specification from a [xmin,ymin,xmax,ymax] list."""
        xmin, ymin, xmax, ymax = input
        return BoundingBoxRegion(
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
        )

    @staticmethod
    def from_dict(input_dict: Dict[str, Bbox]) -> "BoundingBoxRegion":
        """Create a region specification from dictionary specifying values for xmin, ymin, xmax, ymax."""
        xmin, ymin, xmax, ymax = input_dict["bbox"]

        return BoundingBoxRegion(
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
        )

    @model_validator(mode="after")
    def _check_bounds_ordering(self) -> "BoundingBoxRegion":
        # pydantic will turn these assertion errors into validation errors for us
        assert self.xmin <= self.xmax
        assert self.ymin <= self.ymax
        return self


Region = Union[
    BoundingBoxRegion,
    PathRegion,
]


def validate_region(input: Dict[str, Any]) -> Optional[Region]:
    """Create a validated model from a dictionary as allowed by the CLI."""
    if "bbox" in input:
        val = input["bbox"]
        if isinstance(val, list):
            return BoundingBoxRegion.from_list(val)
        elif isinstance(val, dict):
            return BoundingBoxRegion.from_dict(val)
        else:
            raise ValueError(f"bbox value {val} is not a list or dict")
    elif "geom" in input:
        val = input["geom"]
        return PathRegion.from_path(val)
    else:
        for region_type in ["grid", "mesh", "basin", "subbasin", "interbasin"]:
            if region_type in input:
                raise NotImplementedError(
                    f"region kind {region_type} is not supported in region validation yet, but is recognized by HydroMT."
                )
        raise NotImplementedError(f"Unknown region kind: {input}")
