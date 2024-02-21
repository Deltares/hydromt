from logging import getLogger
from typing import Any, Union

from geopandas import GeoDataFrame
from pydantic import BaseModel, Field
from pyproj import CRS

from hydromt.models._region.specifyers.bbox import BboxRegionSpecifyer
from hydromt.models._region.specifyers.file import GeomFileRegionSpecifyer
from hydromt.models._region.specifyers.geom import GeomRegionSpecifyer

__all__ = ["GeomFileRegionSpecifyer"]
logger = getLogger(__name__)

WGS84 = CRS.from_epsg(4326)


class RegionSpecifyer(BaseModel):
    """A specification of a region that can be turned into a geometry."""

    # make typechecking a bit easier
    def __instancecheck__(self, instance: Any) -> bool:
        return isinstance(self.spec, instance)

    def __subclass_check(self, subclass: Any) -> bool:
        return issubclass(self.spec.__class__, subclass)

    spec: Union[
        BboxRegionSpecifyer,
        GeomRegionSpecifyer,
        GeomFileRegionSpecifyer,
        # GridRegionSpecifyer,
    ] = Field(..., discriminator="kind")

    def construct(self) -> GeoDataFrame:
        """Calculate the actual geometry based on the specification."""
        return self.spec.construct()
