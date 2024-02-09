"""Pydantic models for the validation of region specifications."""
from logging import Logger, getLogger
from os import listdir
from os.path import exists, isdir, isfile
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from geopandas import GeoDataFrame
from pydantic import BaseModel, ConfigDict, Field, model_validator
from shapely import box
from typing_extensions import Annotated
from xarray import Dataset

from hydromt._compat import HAS_XUGRID
from hydromt._typing import Bbox
from hydromt.data_catalog import DataCatalog
from hydromt.workflows.basin_mask import get_basin_geometry

logger = getLogger(__name__)

if HAS_XUGRID:
    import xugrid as xu


class BboxRegionSpecifyer(BaseModel):
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
        )

    @property
    def bounds(self) -> Bbox:
        return (self.xmin, self.ymin, self.xmax, self.ymax)

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

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def construct(self) -> Dataset:
        return self.data_catalog.get_rasterdataset(
            self.source, driver_kwargs=self.driver_kwargs
        )


class MeshRegionSpecifyer(BaseModel):
    kind: Literal["mesh"]
    source: Union[str, Path, UgridData]
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def construct(self) -> UgridData:
        if HAS_XUGRID:
            if isinstance(self.source, (str, Path)) and isfile(self.source):
                data = xu.open_dataset(self.source)
            elif isinstance(self.source, (xu.UgridDataset, xu.UgridDataArray)):
                data = self.source
            elif isinstance(self.source, (xu.Ugrid1d, xu.Ugrid2d)):
                data = xu.UgridDataset(self.source.to_dataset(optional_attributes=True))
            else:
                raise ValueError("This should never happen.")
        else:
            raise ImportError("xugrid is required to read mesh files.")

        return data

    @model_validator(mode="after")
    def _check_file_exists(self) -> "MeshRegionSpecifyer":
        if isinstance(self.source, (str, Path)):
            assert exists(self.source)
        return self


class BasinIdRegionSpecifyer(BaseModel):
    kind: Literal["basin"]
    sub_kind: Literal["id"]
    id: BasinIdType
    data_catalog: DataCatalog
    hydrography_source: str
    basin_index_source: str
    logger: Logger
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def construct(self) -> Tuple[GeoDataFrame, GeoDataFrame]:
        ds_org = self.data_catalog.get_rasterdataset(self.hydrography_source)
        basin_index = self.data_catalog.get_source(self.basin_index_source)
        basin_geom, outlet_geom = get_basin_geometry(
            ds=ds_org,
            basid=self.id,
            basin_index=basin_index,
            kind="basin",
            logger=self.logger,
        )
        return basin_geom, outlet_geom


class BasinMultipleIdsRegionSpecifyer(BaseModel):
    kind: Literal["basin"]
    sub_kind: Literal["ids"]
    ids: List[BasinIdType]
    data_catalog: DataCatalog
    hydrography_source: str
    basin_index_source: str
    logger: Logger
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def construct(self) -> Tuple[GeoDataFrame, GeoDataFrame]:
        ds_org = self.data_catalog.get_rasterdataset(self.hydrography_source)
        basin_index = self.data_catalog.get_source(self.basin_index_source)
        basin_geom, outlet_geom = get_basin_geometry(
            ds=ds_org,
            basid=self.ids,
            basin_index=basin_index,
            kind="basin",
            logger=self.logger,
        )
        return basin_geom, outlet_geom


class BasinPointRegionSpecifyer(BaseModel):
    kind: Literal["basin"]
    sub_kind: Literal["point"]
    x_coord: float
    y_coord: float
    data_catalog: DataCatalog
    hydrography_source: str
    basin_index_source: str
    logger: Logger
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def construct(self) -> Tuple[GeoDataFrame, GeoDataFrame]:
        ds_org = self.data_catalog.get_rasterdataset(self.hydrography_source)
        basin_index = self.data_catalog.get_source(self.basin_index_source)
        basin_geom, outlet_geom = get_basin_geometry(
            ds=ds_org,
            xy=([self.x_coord], [self.y_coord]),
            basin_index=basin_index,
            kind="basin",
            logger=self.logger,
        )
        return basin_geom, outlet_geom


class BasinPointListRegionSpecifyer(BaseModel):
    kind: Literal["basin"]
    sub_kind: Literal["points"]
    x_coords: List[float]
    y_coords: List[float]
    data_catalog: DataCatalog
    hydrography_source: str
    basin_index_source: str
    logger: Logger
    # should be removed once we can makd eDataCatalog a fully validated model
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def construct(self) -> Tuple[GeoDataFrame, GeoDataFrame]:
        ds_org = self.data_catalog.get_rasterdataset(self.hydrography_source)
        basin_index = self.data_catalog.get_source(self.basin_index_source)
        # get basin geometry
        basin_geom, outlet_geom = get_basin_geometry(
            ds=ds_org,
            xy=(self.x_coords, self.y_coords),
            basin_index=basin_index,
            kind="basin",
            logger=self.logger,
        )
        return basin_geom, outlet_geom

    @model_validator(mode="after")
    def _check_lengths(self) -> "BasinPointListRegionSpecifyer":
        assert len(self.x_coords) == len(self.y_coords)
        return self


class BasinPointGeomRegionSpecifyer(BaseModel):
    kind: Literal["basin"]
    sub_kind: Literal["point_geom"]
    path: Path
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def construct(self) -> GeoDataFrame:
        return GeoDataFrame.from_file(self.path)


class BasinPointBboxRegionSpecifyer(BaseModel):
    kind: Literal["basin"]
    sub_kind: Literal["bbox"]
    bbox: BboxRegionSpecifyer
    outlets: bool = False
    thresholds: Optional[Dict[str, float]]
    data_catalog: DataCatalog
    hydrography_source: str
    basin_index_source: str
    logger: Logger
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def construct(self) -> Tuple[GeoDataFrame, GeoDataFrame]:
        ds_org = self.data_catalog.get_rasterdataset(self.hydrography_source)
        basin_index = self.data_catalog.get_source(self.basin_index_source)
        # get basin geometry
        basin_geom, outlet_geom = get_basin_geometry(
            ds=ds_org,
            bbox=self.bbox.bounds,
            basin_index=basin_index,
            kind="basin",
            outlets=self.outlets,
            thresholds=self.thresholds,
            logger=self.logger,
        )
        return basin_geom, outlet_geom


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

    def construct(
        self,
    ) -> Union[
        GeoDataFrame, Dataset, UgridData, Dataset, Tuple[GeoDataFrame, GeoDataFrame]
    ]:
        return self.spec.construct()
