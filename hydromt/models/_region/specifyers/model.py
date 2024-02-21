from typing import Literal

from geopandas import GeoDataFrame
from pydantic import BaseModel


class ModelRegionSpecifyer(BaseModel):
    # TODO
    kind: Literal["model"]

    def construct(self) -> GeoDataFrame:
        raise NotImplementedError("todo")
