"""Pydantic models for the validation of region specifications."""
from os import listdir
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from typing_extensions import Annotated
from os.path import exists, isdir
from geopandas import GeoDataFrame

from pydantic import BaseModel, Field, model_validator
from pyproj import CRS
from shapely import box
from xarray import Dataset
from hydromt.data_catalog import DataCatalog

from hydromt.models._v1.model_region import ModelRegion


BasinIdType = str


class BboxRegionSpecifyer(BaseModel):
    crs: CRS = CRS.from_epsg(4326)
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    kind: Literal["bbox"]

    def construct(self) -> GeoDataFrame:
        return GeoDataFrame(
            geometry=[
                box(xmin=self.xmin, ymin=self.ymin, xmax=self.xmax, ymax=self.ymax)
            ],
            crs=self.crs,
        )

    @model_validator(mode="after")
    def _check_bounds_ordering(self) -> "BboxRegionSpecifyer":
        # pydantic will turn these asserion errors into validation errors for us
        assert self.xmin < self.xmax
        assert self.ymin < self.ymax
        return self


class GeomRegionSpecifyer(BaseModel):
    kind: Literal["geom"]
    path: Path

    def construct(self) -> GeoDataFrame:
        return GeoDataFrame.from_file(self.path)

    @model_validator(mode="after")
    def _check_file_exists(self) -> "GeomRegionSpecifyer":
        assert exists(self.path)
        return self


class DerivedRegionSpecifyer(BaseModel):
    """i.e. a region that is derived from the region of another model."""

    # TODO: use entrypoints here to figure out if we need more
    kind: Literal["grid_model", "network_model", "vector_model", "mesh_model"]
    path: Path

    def construct(self) -> GeoDataFrame:
        return GeoDataFrame.from_file(self.path)

    @model_validator(mode="after")
    def _check_root_not_empty(self) -> "DerivedRegionSpecifyer":
        assert exists(self.path)
        assert isdir(self.path)
        assert len(listdir(self.path)) > 0
        return self


class GridRegionSpecifyer(BaseModel):
    kind: Literal["grid"]
    source: str
    data_catalog: DataCatalog
    driver_kwargs: Optional[Dict[str, Any]]

    def construct(self) -> Dataset:
        return self.data_catalog.get_rasterdataset(
            self.source, driver_kwargs=self.driver_kwargs
        )


class MeshRegionSpecifyer(BaseModel):
    kind: Literal["mesh"]
    path: Path

    def construct(self) -> GeoDataFrame:
        if _compat.HAS_XUGRID:
            if isinstance(value, (str, Path)) and isfile(value):
                kwarg = dict(mesh=xu.open_dataset(value))
            elif isinstance(value, (xu.UgridDataset, xu.UgridDataArray)):
                kwarg = dict(mesh=value)
            elif isinstance(value, (xu.Ugrid1d, xu.Ugrid2d)):
                kwarg = dict(
                    mesh=xu.UgridDataset(value.to_dataset(optional_attributes=True))
                )
            else:
                raise ValueError(
                    f"Unrecognised type {type(value)}."
                    "Should be a path, data catalog key or xugrid object."
                )
            kwargs.update(kwarg)
        else:
            raise ImportError("xugrid is required to read mesh files.")
        return GeoDataFrame.from_file(self.path)

    @model_validator(mode="after")
    def _check_file_exists(self) -> "MeshRegionSpecifyer":
        assert exists(self.path)
        return self


class BasinIdRegionSpecifyer(BaseModel):
    kind: Literal["basin"]
    sub_kind: Literal["id"]
    id: BasinIdType

    def construct(self) -> GeoDataFrame:
        return self.spec.construct()


class BasinMultipleIdsRegionSpecifyer(BaseModel):
    kind: Literal["basin"]
    sub_kind: Literal["ids"]
    ids: List[BasinIdType]

    def construct(self) -> GeoDataFrame:
        return self.spec.construct()


class BasinPointRegionSpecifyer(BaseModel):
    kind: Literal["basin"]
    sub_kind: Literal["point"]
    x_coord: float
    y_coord: float

    def construct(self) -> GeoDataFrame:
        return self.spec.construct()


class BasinPointListRegionSpecifyer(BaseModel):
    kind: Literal["basin"]
    sub_kind: Literal["points"]
    x_coords: List[float]
    y_coords: List[float]

    def construct(self) -> GeoDataFrame:
        return self.spec.construct()

    @model_validator(mode="after")
    def _check_lengths(self) -> "BasinPointListRegionSpecifyer":
        assert len(self.x_coords) == len(self.y_coords)
        return self


class BasinPointGeomRegionSpecifyer(BaseModel):
    kind: Literal["basin"]
    sub_kind: Literal["point_geom"]
    path: Path

    def construct(self) -> GeoDataFrame:
        return self.spec.construct()


class BasinPointBboxRegionSpecifyer(BaseModel):
    kind: Literal["basin"]
    sub_kind: Literal["bbox"]
    bbox: BboxRegionSpecifyer
    outlets: bool = False
    thresholds: Optional[Dict[str, float]]

    def construct(self) -> GeoDataFrame:
        return self.spec.construct()


BasinRegionSpecifyer = Annotated[
    Union[
        BasinIdRegionSpecifyer,
        BasinMultipleIdsRegionSpecifyer,
        BasinPointGeomRegionSpecifyer,
        BasinPointListRegionSpecifyer,
        BasinPointRegionSpecifyer,
        BasinPointBboxRegionSpecifyer,
    ],
    Field(discriminator="sub_kind"),
]
"""            Dictionary describing region of interest.

            Subbasins are defined by its outlet locations and include all area upstream
            from these points. The outlet locations can be passed as xy coordinate pairs,
            but also derived from the most downstream cell(s) within a area of interest
            defined by a bounding box or geometry, optionally refined by stream threshold
            arguments.

            The method can be speed up by providing an additional ``bounds`` argument which
            should contain all upstream cell. If cells upstream of the subbasin are not
            within the provide bounds a warning will be raised. Common use-cases include:
                * {'subbasin': [x, y], '<variable>': threshold}
                * {
                    'subbasin': [[x1, x2, ..], [y1, y2, ..]],
                    '<variable>': threshold, 'bounds': [xmin, ymin, xmax, ymax]
                    }
                * {'subbasin': /path/to/point_geometry, '<variable>': threshold}
                * {'subbasin': [xmin, ymin, xmax, ymax], '<variable>': threshold}
                * {'subbasin': /path/to/polygon_geometry, '<variable>': threshold}
            Interbasins are similar to subbasins but are bounded by a bounding box or
            geometry and do not include all upstream area. Common use-cases include:
                * {'interbasin': [xmin, ymin, xmax, ymax], '<variable>': threshold}
                * {'interbasin': [xmin, ymin, xmax, ymax], 'xy': [x, y]}
                * {'interbasin': /path/to/polygon_geometry, 'outlets': true}
"""


# TODO still add subbasins and interbasins
class RegionSpecifyer(BaseModel):
    spec: Union[
        BboxRegionSpecifyer,
        GeomRegionSpecifyer,
        DerivedRegionSpecifyer,
        GridRegionSpecifyer,
        MeshRegionSpecifyer,
        BasinRegionSpecifyer,
    ] = Field(..., discriminator="kind")

    def construct(self) -> Union[GeoDataFrame, Dataset]:
        return self.spec.construct()
