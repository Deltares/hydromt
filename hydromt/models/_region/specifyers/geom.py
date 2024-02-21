from logging import getLogger
from typing import Literal, cast

from geopandas import GeoDataFrame
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pyproj import CRS

WGS84 = CRS.from_epsg(4326)

logger = getLogger(__name__)


class GeomRegionSpecifyer(BaseModel):
    """A region specified by another geometry."""

    geom: GeoDataFrame
    kind: Literal["geom_data"]
    crs: CRS = Field(default=WGS84)
    buffer: float = Field(0.0, ge=0)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def construct(self) -> GeoDataFrame:
        """Calculate the actual geometry based on the specification."""
        geom = self.geom
        if self.buffer > 0:
            if geom.crs.is_geographic:
                geom = cast(GeoDataFrame, geom.to_crs(3857))
            geom = geom.buffer(self.buffer)

        if geom.crs is None:
            geom.set_crs(self.crs)
        return geom

    @model_validator(mode="after")
    def _check_crs_consistent(self) -> "GeomRegionSpecifyer":
        if self.geom.crs != self.crs:
            raise ValueError("geom crs and provided crs do not correspond")
        return self

    @model_validator(mode="after")
    def _check_valid_geom(self) -> "GeomRegionSpecifyer":
        # if we have a buffer points will be turned in polygons
        if any(self.geom.geometry.type == "Point") and self.buffer <= 0:
            raise ValueError(
                "Region based on points are not supported. Provide a geom with polygons, or specify a buffer greater than 0"
            )
        return self
