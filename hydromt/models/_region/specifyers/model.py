from typing import Literal

from geopandas import GeoDataFrame
from pydantic import BaseModel


class ModelRegionSpecifyer(BaseModel):
    # TODO
    pass
    kind: Literal["model"]

    def construct(self) -> GeoDataFrame:
        raise NotImplementedError("todo")
