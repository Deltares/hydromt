from logging import getLogger
from typing import Any, Union

from geopandas import GeoDataFrame
from pydantic import BaseModel, ConfigDict, Field

from hydromt.models._region.specifyers.basin import (
    BasinBboxSpecifyer,
    BasinGeomFileSpecifyer,
    BasinIDSpecifyer,
    BasinIDsSpecifyer,
    BasinXYSpecifyer,
    BasinXYsSpecifyer,
)
from hydromt.models._region.specifyers.bbox import BboxRegionSpecifyer
from hydromt.models._region.specifyers.catalog import (
    GeomCatalogRegionSpecifyer,
    GridCatalogRegionSpecifyer,
)
from hydromt.models._region.specifyers.file import GeomFileRegionSpecifyer
from hydromt.models._region.specifyers.geom import GeomRegionSpecifyer
from hydromt.models._region.specifyers.grid import (
    GridDataRegionSpecifyer,
    GridPathRegionSpecifyer,
)
from hydromt.models._region.specifyers.interbasin import (
    InterBasinBboxVarSpecifyer,
    InterBasinBboxXYSpecifyer,
    InterBasinGeomFileSpecifyer,
    InterBasinGeomSpecifyer,
)
from hydromt.models._region.specifyers.model import ModelRegionSpecifyer
from hydromt.models._region.specifyers.subbasin import (
    SubBasinBboxSpecifyer,
    SubBasinGeomFileSpecifyer,
    SubBasinXYSpecifyer,
    SubBasinXYsSpecifyer,
)

__all__ = ["RegionSpecifyer"]
logger = getLogger(__name__)


class RegionSpecifyer(BaseModel):
    """A specification of a region that can be turned into a geometry."""

    # make typechecking a bit easier
    def __instancecheck__(self, instance: Any) -> bool:
        return isinstance(self.spec, instance)

    spec: Union[
        BasinBboxSpecifyer,
        BasinGeomFileSpecifyer,
        BasinIDSpecifyer,
        BasinIDsSpecifyer,
        BasinXYSpecifyer,
        BasinXYsSpecifyer,
        BboxRegionSpecifyer,
        GeomFileRegionSpecifyer,
        GeomRegionSpecifyer,
        GridDataRegionSpecifyer,
        GridPathRegionSpecifyer,
        GridCatalogRegionSpecifyer,
        GeomCatalogRegionSpecifyer,
        InterBasinBboxVarSpecifyer,
        InterBasinBboxXYSpecifyer,
        InterBasinGeomFileSpecifyer,
        InterBasinGeomSpecifyer,
        ModelRegionSpecifyer,
        SubBasinBboxSpecifyer,
        SubBasinGeomFileSpecifyer,
        SubBasinXYSpecifyer,
        SubBasinXYsSpecifyer,
    ] = Field(..., discriminator="kind")
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def construct(self) -> GeoDataFrame:
        """Calculate the actual geometry based on the specification."""
        return self.spec.construct()
