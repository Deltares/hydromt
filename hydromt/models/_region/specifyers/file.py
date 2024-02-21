from logging import getLogger
from os.path import exists
from pathlib import Path
from typing import Literal, cast

from geopandas import GeoDataFrame
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pyproj import CRS

from hydromt.models._region.specifyers import WGS84

logger = getLogger(__name__)


class GeomFileRegionSpecifyer(BaseModel):
    """A region specified by a geometry read from a file."""

    kind: Literal["geom_file"]
    path: Path
    buffer: float = Field(0.0, ge=0)
    crs: CRS = Field(default=WGS84)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def construct(self) -> GeoDataFrame:
        """Calculate the actual geometry based on the specification."""
        geom = GeoDataFrame.from_file(self.path)
        if self.buffer > 0:
            if geom.crs.is_geographic:
                geom = cast(GeoDataFrame, geom.to_crs(3857))
            geom = geom.buffer(self.buffer)
        if geom.crs is None:
            geom.set_crs(self.crs)
        return geom

    @model_validator(mode="after")
    def _check_file_exists(self) -> "GeomFileRegionSpecifyer":
        if not exists(self.path):
            raise ValueError(f"Provided path does not exist: {self.path}")

        return self
