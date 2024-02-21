from typing import Literal

from geopandas import GeoDataFrame
from pydantic import BaseModel


class GeomCatalogRegionSpecifyer(BaseModel):
    """A region specified by another geometry."""

    source: str
    kind: Literal["geom_catalog"]

    def construct(self) -> GeoDataFrame:
        raise NotImplementedError("todo")


class GridCatalogRegionSpecifyer(BaseModel):
    """A region specified by another geometry."""

    source: str
    kind: Literal["grid_catalog"]

    def construct(self) -> GeoDataFrame:
        raise NotImplementedError("todo")
