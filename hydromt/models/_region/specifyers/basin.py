from logging import getLogger
from os.path import exists
from pathlib import Path
from typing import Dict, List, Literal

from geopandas import GeoDataFrame
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pyproj import CRS

from hydromt._typing.type_def import BasinIdType

WGS84 = CRS.from_epsg(4326)

logger = getLogger(__name__)


class BasinIDSpecifyer(BaseModel):
    kind: Literal["basin_id"]
    id: BasinIdType
    crs: CRS = Field(default=WGS84)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def construct(self) -> GeoDataFrame:
        raise NotImplementedError("not yet implemented")


class BasinIDsSpecifyer(BaseModel):
    kind: Literal["basin_ids"]
    ids: List[BasinIdType]
    crs: CRS = Field(default=WGS84)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def construct(self) -> GeoDataFrame:
        raise NotImplementedError("not yet implemented")


class BasinXYSpecifyer(BaseModel):
    kind: Literal["basin_xy"]
    x: float
    y: float
    crs: CRS = Field(default=WGS84)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def construct(self) -> GeoDataFrame:
        raise NotImplementedError("not yet implemented")


class BasinXYsSpecifyer(BaseModel):
    kind: Literal["basin_xys"]
    xs: List[float]
    ys: List[float]
    crs: CRS = Field(default=WGS84)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def construct(self) -> GeoDataFrame:
        raise NotImplementedError("not yet implemented")

    @model_validator(mode="after")
    def _check_equal_length(self) -> "BasinXYsSpecifyer":
        if len(self.xs) != len(self.ys):
            raise ValueError("number of x coords is not equal to number of y coords")
        return self


class BasinGeomFileSpecifyer(BaseModel):
    kind: Literal["basin_file"]
    path: Path
    crs: CRS = Field(default=WGS84)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def construct(self) -> GeoDataFrame:
        raise NotImplementedError("not yet implemented")

    @model_validator(mode="after")
    def _check_file_exists(self) -> "BasinGeomFileSpecifyer":
        if not exists(self.path):
            raise ValueError(f"Could not find file at {self.path}")
        return self


class BasinBboxSpecifyer(BaseModel):
    kind: Literal["basin_bbox"]
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
    def _check_bounds_ordering(self) -> "BasinBboxSpecifyer":
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
