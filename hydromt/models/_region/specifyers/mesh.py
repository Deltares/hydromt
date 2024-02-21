from typing import Literal

from geopandas import GeoDataFrame
from pydantic import BaseModel


class MeshRegionSpecifyer(BaseModel):
    pass

    kind: Literal["mesh"]

    def construct(self) -> GeoDataFrame:
        raise NotImplementedError("todo")
