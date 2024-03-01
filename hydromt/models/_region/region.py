from logging import Logger, getLogger
from os.path import join
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, cast
from weakref import ReferenceType, ref

from geopandas import GeoDataFrame, gpd
from pyproj import CRS
from shapely import box
from xugrid import UgridDataArrayAccessor

from hydromt._typing.model_mode import ModelMode
from hydromt._typing.type_def import StrPath
from hydromt.data_catalog import DataCatalog
from hydromt.models._region.utils import _parse_region
from hydromt.models.root import ModelRoot

if TYPE_CHECKING:
    from hydromt.models import Model

logger = getLogger(__name__)


class ModelRegion:
    def __init__(
        self,
        model: "Model",
        logger: Logger = logger,
    ) -> None:
        self.model_ref: ReferenceType["Model"] = ref(model)
        self._data: Optional[GeoDataFrame] = None

    def create(
        self, region_dict: Dict[str, Any], catalog: Optional[DataCatalog] = None
    ):
        self._kind, data = _parse_region(region_dict, catalog, logger)
        if self._kind == "mesh":
            self._data = cast(UgridDataArrayAccessor, data["mesh"]).to_geodataframe()

        elif self._kind == "bbox":
            self._data = gpd.GeoDataFrame(
                geometry=[box(*data["bbox"])], crs=CRS.from_epsg(4326)
            )
        elif self._kind == "geom":
            self._data = data["geom"]
        elif self._kind == "grid":
            raise NotImplementedError("TODO")
        elif self._kind in ["basin", "subbasin", "interbasin"]:
            raise NotImplementedError("TODO")
        else:
            raise ValueError(f"Could not understand kind {self._kind}")

    # def _rasterdataset_to_geom(self, raster_dataset) -> GeoDataFrame:
    # crs = raster_dataset.raster.crs
    # coord_index = cast(pd.Index, raster_dataset.raster.coords.to_index())
    # dims_max = cast(np.ndarray, coord_index.max())
    # dims_min = cast(np.ndarray, coord_index.min())

    # # in raster datasets it is guaranteed that y_dim is penultimate dim and x_dim is last dim
    # geom: GeoDataFrame = GeoDataFrame(
    #     geometry=[
    #         box(
    #             xmin=dims_min[-1],
    #             ymin=dims_min[-2],
    #             xmax=dims_max[-1],
    #             ymax=dims_max[-2],
    #         )
    #     ],
    #     crs=crs,
    # )
    # return geom

    @property
    def total_bounds(self):
        return self.data.total_bounds

    @property
    def data(self) -> GeoDataFrame:
        if self._data is None:
            # cast is necessary because technically the model could have been
            # dealocated, but it shouldn't be in this case.
            root: Optional[ModelRoot] = cast("Model", self.model_ref()).root

            # cannot read geom files for purely in memory models
            if root is None:
                raise ValueError("Root was not set, cannot read region file")
            else:
                self.read("region.geojson")

        return cast(GeoDataFrame, self._data)

    @property
    def crs(self) -> CRS:
        return self.data.crs

    def read(
        self,
        rel_path: StrPath = Path("region.geojson"),
        model_mode: ModelMode = ModelMode.READ,
        **read_kwargs,
    ):
        if self._data is None:
            if model_mode.is_reading_mode():
                root: Optional[ModelRoot] = cast("Model", self.model_ref()).root

                # cannot read geom files for purely in memory models
                if root is None:
                    raise ValueError("Root was not set, cannot read region file")
                else:
                    self._data = cast(
                        GeoDataFrame,
                        gpd.read_file(join(root.path, rel_path), **read_kwargs),
                    )
                    self._kind = "geom"
            else:
                raise ValueError("Cannot read while not in read mode")

    def write(
        self,
        rel_path: Path = Path("region.geojson"),
        model_mode: ModelMode = ModelMode.WRITE,
        **write_kwargs,
    ):
        if model_mode.is_writing_mode():
            root: Optional[ModelRoot] = cast("Model", self.model_ref()).root

            # cannot read geom files for purely in memory models
            if root is None:
                raise ValueError("Root was not set, cannot read region file")
            else:
                self.read()
            self.data.to_file(join(root.path, rel_path), **write_kwargs)
