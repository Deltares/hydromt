from logging import getLogger
from os.path import exists
from pathlib import Path
from typing import Dict, Literal

from geopandas import GeoDataFrame
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pyproj import CRS

WGS84 = CRS.from_epsg(4326)

logger = getLogger(__name__)


class InterBasinGeomFileSpecifyer(BaseModel):
    kind: Literal["interbasin_file"]
    path: Path
    variables: Dict[str, float] = Field(default_factory=dict)
    crs: CRS = Field(default=WGS84)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def construct(self) -> GeoDataFrame:
        raise NotImplementedError("not yet implemented")

    @model_validator(mode="after")
    def _check_file_exists(self) -> "InterBasinGeomFileSpecifyer":
        if not exists(self.path):
            raise ValueError(f"Could not find file at {self.path}")
        return self


class InterBasinGeomSpecifyer(BaseModel):
    kind: Literal["interbasin_geom"]
    data: GeoDataFrame
    variables: Dict[str, float] = Field(default_factory=dict)
    crs: CRS = Field(default=WGS84)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def construct(self) -> GeoDataFrame:
        raise NotImplementedError("not yet implemented")


class InterBasinBboxVarSpecifyer(BaseModel):
    kind: Literal["interbasin_bbox_var"]
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
    def _check_bounds_ordering(self) -> "InterBasinBboxVarSpecifyer":
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


class InterBasinBboxXYSpecifyer(BaseModel):
    kind: Literal["interbasin_bbox_xy"]
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
    def _check_bounds_ordering(self) -> "InterBasinBboxXYSpecifyer":
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
