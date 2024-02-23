from logging import Logger, getLogger
from os.path import join
from pathlib import Path
from typing import Any, Dict, Optional, cast

from geopandas import GeoDataFrame, gpd
from pyproj import CRS
from shapely import box
from xugrid import UgridDataArrayAccessor

from hydromt._typing.model_mode import ModelMode
from hydromt.data_catalog import DataCatalog
from hydromt.models._region.utils import _parse_region

logger = getLogger(__name__)


class ModelRegion:
    def __init__(
        self,
        region_dict: Optional[Dict[str, Any]] = None,
        root: Optional[Path] = None,
        rel_path: Optional[Path] = None,
        catalog: Optional[DataCatalog] = None,
        model_mode: ModelMode = ModelMode.READ,
        logger: Logger = logger,
        **read_kwargs,
    ) -> None:
        self._spec = region_dict
        if region_dict is not None:
            kind, data = _parse_region(region_dict, catalog, logger)
            self.set(kind, data)

        elif rel_path is not None and root is not None:
            self.read(root, rel_path, model_mode, **read_kwargs)

    def set(self, kind, parsed_dict):
        if kind == "mesh":
            self._data: GeoDataFrame = cast(
                UgridDataArrayAccessor, parsed_dict["mesh"]
            ).to_geodataframe()

        elif kind == "bbox":
            self._data = gpd.GeoDataFrame(
                geometry=[box(*parsed_dict["bbox"])], crs=CRS.from_epsg(4326)
            )
        elif kind == "geom":
            self._data = parsed_dict["geom"]
        elif kind == "grid":
            raise NotImplementedError("TODO")
            #  raster_dataset = self.data_catalog.get_rasterdataset(
            #     self.source, driver_kwargs=self.driver_kwargs
            # )
            # if raster_dataset is None:
            #     raise ValueError("raster dataset was not found")
            # self._data = self._rasterdataset_to_geom(parsed_dict["grid"])
        elif kind in ["basin", "subbasin", "interbasin"]:
            raise NotImplementedError("TODO")
        else:
            raise ValueError(f"Could not understand kind {kind}")

    def _rasterdataset_to_geom(self, raster_dataset) -> GeoDataFrame:
        raise NotImplementedError("TODO")
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

    def read(self, root: Path, rel_path: Path, model_mode: ModelMode, **read_kwargs):
        if model_mode.is_reading_mode():
            self._data = cast(
                GeoDataFrame, gpd.read_file(join(root, rel_path), **read_kwargs)
            )
            self._kind = "geom"
        else:
            raise ValueError("Cannot read while not in read mode")

    def write(self, root: Path, rel_path: Path, model_mode: ModelMode):
        if self._data is None:
            raise ValueError("Region was not initialised, so cannot read")

        if model_mode.is_writing_mode():
            self._data.to_file(join(root, rel_path))
