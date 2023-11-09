"""Pydantic models for the validation of region specifications."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, ValidationError, model_validator


# bare types
class WGS84Point(BaseModel):
    """A validation model for a point in WSG84 space."""

    x: float = Field(ge=-180, le=180)
    y: float = Field(ge=-90, le=90)

    @staticmethod
    def from_dict(input_dict: Dict) -> "WGS84Point":
        """Create a WGS84Point from a {x:,y:} dictionary."""
        return WGS84Point(**input_dict)

    @staticmethod
    def from_list(input_list: List) -> "WGS84Point":
        """Create a WGS84Point from a [x,y] list."""
        return WGS84Point(x=input_list[0], y=input_list[1])

    @staticmethod
    def from_xy(x: float, y: float) -> "WGS84Point":
        """Create a WGS84Point from two coordinates."""
        return WGS84Point(x=x, y=y)


class VariableThreshold(BaseModel):
    """A threshold to be applied to a region specification."""

    name: str
    threshold: float = Field(ge=0)


class PathlikeRegion(BaseModel):
    """A validation model for a region loaded from a file."""

    path: Path
    threshold: Optional[VariableThreshold] = None

    @staticmethod
    def from_path(path: Union[Path, str]) -> "PathlikeRegion":
        """Create a region that will be loaded from a file."""
        if isinstance(path, Path):
            return PathlikeRegion(path=path)
        else:
            return PathlikeRegion(path=Path(path))

    @model_validator(mode="after")
    def _check_path_exists(self) -> "PathlikeRegion":
        if not self.path.exists():
            raise ValueError(f"Path not found at {self.path}")
        return self


class BoundingBoxLikeRegion(BaseModel):
    """A validation model for a region described by a bounding box in WGS84 space."""

    xmin: float = Field(ge=-180)
    xmax: float = Field(le=180)
    ymin: float = Field(ge=-90)
    ymax: float = Field(le=90)

    @staticmethod
    def from_list(
        input: Union[Tuple[float, float, float, float], List[float]]
    ) -> "BoundingBoxLikeRegion":
        """Create a region specification from a [xmin,ymin,xmax,ymax] list."""
        xmin, ymin, xmax, ymax = input
        return BoundingBoxLikeRegion(
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
        )

    @staticmethod
    def from_dict(input_dict: Dict) -> "BoundingBoxLikeRegion":
        """Create a region specification from dictionary specifying values for xmin, ymin, xmax, ymax."""
        xmin, ymin, xmax, ymax = input_dict["bbox"]

        return BoundingBoxLikeRegion(
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
        )

    @model_validator(mode="after")
    def _check_bounds_ordering(self) -> "BoundingBoxLikeRegion":
        # pydantic will turn these asserion errors into validation errors for us
        assert self.xmin <= self.xmax
        assert self.ymin <= self.ymax
        return self


class PointLikeRegion(BaseModel):
    """A validation model for a region described by a point in WGS84 space."""

    points: List[WGS84Point]

    @staticmethod
    def from_dict(input_dict: Dict) -> "PointLikeRegion":
        """Create a region specification from dictionary specifying values for x and y."""
        return PointLikeRegion(points=[WGS84Point.from_dict(input_dict)])

    @staticmethod
    def from_list(input_list: Union[List, Tuple[float, float]]) -> "PointLikeRegion":
        """Create a region specification from a [x,y] list or tuple."""
        if isinstance(input_list, Tuple):
            return PointLikeRegion(points=[WGS84Point.from_list(list(input_list))])
        else:
            return PointLikeRegion(points=[WGS84Point.from_list(input_list)])

    @staticmethod
    def from_xy(x: float, y: float) -> "PointLikeRegion":
        """Create a region specification by specifying values for x and y."""
        return PointLikeRegion(points=[WGS84Point.from_xy(x=x, y=y)])

    @staticmethod
    def from_points(l: List[WGS84Point]) -> "PointLikeRegion":
        """Create a region specification by specifying values for x and y."""
        return PointLikeRegion(points=l)

    @staticmethod
    def from_xy_lists(xs: list[float], ys: list[float]) -> "PointLikeRegion":
        """Create a region specification from lists specifying [x1,...,xn] and [y1,...,yn] respectively."""
        tups = zip(xs, ys)

        return PointLikeRegion(points=[WGS84Point.from_xy(x=x, y=y) for (x, y) in tups])


GeometryRegion = PathlikeRegion
GridRegion = PathlikeRegion
MeshRegion = PathlikeRegion
GeometryBasinRegion = PathlikeRegion
GeometrySubBasinRegion = PathlikeRegion
PointSubBasinRegion = PointLikeRegion
MultiPointSubBasinRegion = PointLikeRegion
BoundingBoxSubBasinRegion = BoundingBoxLikeRegion
BoundingBoxInterBasinRegion = BoundingBoxLikeRegion
GeometryInterBasinRegion = PathlikeRegion

BoundingBoxRegion = BoundingBoxLikeRegion
PointBasinRegion = PointLikeRegion
MultiPointBasinRegion = PointLikeRegion
BoundingBoxBasinRegion = BoundingBoxLikeRegion


Region = Union[
    BoundingBoxRegion,
    GeometryRegion,
    GridRegion,
    MeshRegion,
    PointBasinRegion,
    MultiPointBasinRegion,
    BoundingBoxBasinRegion,
    GeometryBasinRegion,
    PointSubBasinRegion,
    MultiPointSubBasinRegion,
    BoundingBoxSubBasinRegion,
    GeometrySubBasinRegion,
    BoundingBoxInterBasinRegion,
    GeometryInterBasinRegion,
]


def validate_region(input: Dict[str, Any]) -> Optional[Region]:
    """Create a validated model from a dictionary as allowed by the CLI."""
    if "bbox" in input:
        val = input["bbox"]
        if isinstance(val, list):
            return BoundingBoxRegion.from_list(val)
        elif isinstance(val, dict):
            return BoundingBoxRegion.from_dict(val)
    elif "geom" in input:
        val = input["geom"]
        return GeometryRegion.from_path(val)
    elif "grid" in input:
        val = input["grid"]
        return GridRegion.from_path(val)
    elif "mesh" in input:
        val = input["mesh"]
        return MeshRegion.from_path(val)
    elif "basin" in input:
        val = input["basin"]
        if isinstance(val, list):
            if isinstance(val[0], (float, int)) and len(val) == 2:  # [x,y]
                return PointBasinRegion.from_xy(x=val[0], y=val[1])
            elif (
                isinstance(val[0], (float, int)) and len(val) == 4
            ):  # [xmin,ymin, xmaxn ymax]
                return BoundingBoxBasinRegion.from_list(val)
            elif isinstance(val[0], list):  # [[x1,...,xn], [y1,..,yn]]
                return MultiPointBasinRegion.from_xy_lists(xs=val[0], ys=val[1])
            else:
                raise ValidationError(f"Unknown subbasin kind: {val}")
        elif isinstance(val, str):
            return GeometryBasinRegion.from_path(path=val)
        else:
            raise ValidationError(f"Unknown subbasin kind: {val}")

    elif "subbasin" in input:
        val = input["subbasin"]
        if isinstance(val, list):
            if isinstance(val[0], (float, int)) and len(val) == 2:  # [x,y]
                return PointSubBasinRegion.from_xy(x=val[0], y=val[1])
            elif (
                isinstance(val[0], (float, int)) and len(val) == 4
            ):  # [xmin, ymin, xmaxn, ymax]
                return BoundingBoxSubBasinRegion.from_list(val)
            elif isinstance(val[0], list):  # [[x1,...,xn], [y1,..,yn]]
                return MultiPointSubBasinRegion.from_xy_lists(xs=val[0], ys=val[1])
            else:
                raise ValidationError(f"Unknown subbasin kind: {val}")
        elif isinstance(val, str):
            return GeometrySubBasinRegion.from_path(path=val)
        else:
            raise ValidationError(f"Unknown subbasin kind: {val}")
    elif "interbasin" in input:
        val = input["interbasin"]
        if isinstance(val, list):
            if (
                isinstance(val[0], (float, int)) and len(val) == 4
            ):  # [xmin, ymin, xmaxn, ymax]
                return BoundingBoxInterBasinRegion.from_list(val)
            else:
                raise ValidationError(f"Unknown subbasin kind: {val}")
        elif isinstance(val, str):
            return GeometryInterBasinRegion.from_path(path=val)
        else:
            raise ValidationError(f"Unknown subbasin kind: {val}")
    else:
        key, val = next(iter(input.items()))
        raise ValueError(f"Unknown region kind: {val}")
        # model name
