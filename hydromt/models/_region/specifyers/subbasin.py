from logging import getLogger
from os.path import exists
from pathlib import Path
from typing import Dict, List, Literal, Optional

from geopandas import GeoDataFrame
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pyproj import CRS

WGS84 = CRS.from_epsg(4326)

logger = getLogger(__name__)


class SubBasinXYSpecifyer(BaseModel):
    kind: Literal["subbasin_xy"]
    # sub_kind: Literal["xy"]
    x: float
    y: float
    variables: Dict[str, float] = Field(default_factory=dict)
    crs: CRS = Field(default=WGS84)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def construct(self) -> GeoDataFrame:
        raise NotImplementedError("not yet implemented")


class SubBasinXYsSpecifyer(BaseModel):
    kind: Literal["subbasin_xys"]
    # sub_kind: Literal["xys"]
    variables: Dict[str, float] = Field(default_factory=dict)
    xs: List[float]
    ys: List[float]
    xmin: Optional[float] = None
    ymin: Optional[float] = None
    xmax: Optional[float] = None
    ymax: Optional[float] = None
    crs: CRS = Field(default=WGS84)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def construct(self) -> GeoDataFrame:
        raise NotImplementedError("not yet implemented")

    @model_validator(mode="after")
    def _check_equal_length(self) -> "SubBasinXYsSpecifyer":
        if len(self.xs) != len(self.ys):
            raise ValueError("number of x coords is not equal to number of y coords")
        return self


class SubBasinGeomFileSpecifyer(BaseModel):
    kind: Literal["subbasin_file"]
    # sub_kind: Literal["file"]
    path: Path
    variables: Dict[str, float] = Field(default_factory=dict)
    crs: CRS = Field(default=WGS84)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def construct(self) -> GeoDataFrame:
        raise NotImplementedError("not yet implemented")

    @model_validator(mode="after")
    def _check_file_exists(self) -> "SubBasinGeomFileSpecifyer":
        if not exists(self.path):
            raise ValueError(f"Could not find file at {self.path}")
        return self


class SubBasinBboxSpecifyer(BaseModel):
    kind: Literal["subbasin_bbox"]
    # sub_kind: Literal["bbox"]
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    outlets: bool = False
    variables: Dict[str, float] = Field(default_factory=dict)
    crs: CRS = Field(default=WGS84)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def construct(self) -> GeoDataFrame:
        raise NotImplementedError("not yet implemented")

    @model_validator(mode="after")
    def _check_bounds_ordering(self) -> "SubBasinBboxSpecifyer":
        # pydantic will turn these asserion errors into validation errors for us
        if self.xmin >= self.xmax:
            raise ValueError(
                f"xmin ({self.xmin}) should be strictly less than xmax ({self.xmax})"
            )
        if self.ymin >= self.ymax:
            raise ValueError(
                f"ymin ({self.ymin}) should be strictly less than ymax ({self.ymax}) "
            )
        return self
