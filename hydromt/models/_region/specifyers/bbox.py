from logging import getLogger
from typing import Literal, cast

from geopandas import GeoDataFrame
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pyproj import CRS
from shapely import box

WGS84 = CRS.from_epsg(4326)

logger = getLogger(__name__)


class BboxRegionSpecifyer(BaseModel):
    """A region specified by a bounding box."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float
    kind: Literal["bbox"]
    buffer: float = 0.0
    crs: CRS = Field(default=WGS84)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def construct(self) -> GeoDataFrame:
        """Calculate the actual geometry based on the specification."""
        geom = GeoDataFrame(
            geometry=[
                box(xmin=self.xmin, ymin=self.ymin, xmax=self.xmax, ymax=self.ymax)
            ],
            crs=self.crs,
        )

        if self.buffer > 0:
            if geom.crs.is_geographic:
                geom = cast(GeoDataFrame, geom.to_crs(3857))
            geom = geom.buffer(self.buffer)

        return geom

    @model_validator(mode="after")
    def _check_bounds_ordering(self) -> "BboxRegionSpecifyer":
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
