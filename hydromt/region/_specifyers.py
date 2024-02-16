"""Pydantic models for the validation of region specifications."""
from logging import getLogger
from os.path import exists
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union, cast

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pyproj import CRS
from shapely import box
from hydromt._typing import Predicate

logger = getLogger(__name__)

WGS84 = CRS.from_epsg(4326)


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
        )
        if self.buffer > 0:
            if geom.crs.is_geographic:
                geom = cast(GeoDataFrame, geom.to_crs(3857))
            geom = geom.buffer(self.buffer)
        return geom

    @model_validator(mode="after")
    def _check_bounds_ordering(self) -> "BboxRegionSpecifyer":
        # pydantic will turn these asserion errors into validation errors for us
        assert self.xmin < self.xmax
        assert self.ymin < self.ymax
        return self


class GeomFileRegionSpecifyer(BaseModel):
    """A region specified by a geometry read from a file."""

    kind: Literal["geom_file"]
    path: Path
    buffer: float = 0.0
    crs: CRS = Field(default=WGS84)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def construct(self) -> GeoDataFrame:
        """Calculate the actual geometry based on the specification."""
        geom = GeoDataFrame.from_file(self.path)
        if self.buffer > 0:
            if geom.crs.is_geographic:
                geom = cast(GeoDataFrame, geom.to_crs(3857))
            geom = geom.buffer(self.buffer)
        return geom

    @model_validator(mode="after")
    def _check_file_exists(self) -> "GeomFileRegionSpecifyer":
        assert exists(self.path)
        return self


class GeomRegionSpecifyer(BaseModel):
    """A region specified by another geometry."""

    geom: GeoDataFrame
    kind: Literal["geom"]
    crs: CRS = Field(default=WGS84)
    buffer: float = 0.0
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def construct(self) -> GeoDataFrame:
        """Calculate the actual geometry based on the specification."""
        geom = self.geom
        if self.buffer > 0:
            if geom.crs.is_geographic:
                geom = cast(GeoDataFrame, geom.to_crs(3857))
            geom = geom.buffer(self.buffer)
        return geom

    @model_validator(mode="after")
    def _check_crs_consistent(self) -> "GeomRegionSpecifyer":
        assert self.geom.crs == self.crs
        return self


# class GridRegionSpecifyer(BaseModel):
#     """A region specified by a Rasterdataset."""

#     kind: Literal["grid"]
#     source: str
#     buffer: float = 0.0
#     driver_kwargs: Optional[Dict[str, Any]]

#     crs: CRS = Field(default=WGS84)
#     model_config = ConfigDict(arbitrary_types_allowed=True)

#     def construct(self, DataCatalog) -> GeoDataFrame:
#         """Calculate the actual geometry based on the specification."""
#         raster_dataset = self.data_catalog.get_rasterdataset(
#             self.source, driver_kwargs=self.driver_kwargs
#         )
#         if raster_dataset is None:
#             raise ValueError("raster dataset was not found")
#         crs = raster_dataset.raster.crs
#         coord_index = cast(pd.Index, raster_dataset.coords.to_index())
#         dims_max = cast(np.ndarray, coord_index.max())
#         dims_min = cast(np.ndarray, coord_index.min())

#         # in raster datasets it is guaranteed that y_dim is penultimate dim and x_dim is last dim
#         geom: GeoDataFrame = GeoDataFrame(
#             geometry=[
#                 box(
#                     xmin=dims_min[-1],
#                     ymin=dims_min[-2],
#                     xmax=dims_max[-1],
#                     ymax=dims_max[-2],
#                 )
#             ],
#             crs=crs,
#         )

#         if self.buffer > 0:
#             if geom.crs.is_geographic:
#                 geom = cast(GeoDataFrame, geom.to_crs(3857))
#             geom = geom.buffer(self.buffer)

#         return geom

#     @model_validator(mode="after")
#     def _check_has_source(self) -> "GridRegionSpecifyer":
#         assert self.data_catalog.contains_source(self.source)
#         return self


# TODO still add basins, subbasins, interbasins and other models
class RegionSpecifyer(BaseModel):
    """A specification of a region that can be turned into a geometry."""

    spec: Union[
        BboxRegionSpecifyer,
        GeomRegionSpecifyer,
        GeomFileRegionSpecifyer,
        # GridRegionSpecifyer,
    ] = Field(..., discriminator="kind")

    def construct(self) -> GeoDataFrame:
        """Calculate the actual geometry based on the specification."""
        return self.spec.construct()
