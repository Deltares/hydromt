"""A class to handle all functionalities to do with model regions."""

from logging import Logger, getLogger
from os.path import isdir, isfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from geopandas import GeoDataFrame
from pyproj import CRS
from xarray import DataArray

from hydromt._compat import HAS_XUGRID
from hydromt.typing import Bbox
from hydromt.validators.region import BasinIdType, RegionSpecifyer
from hydromt.workflows import get_basin_geometry
from hydromt.models import MODELS
from hydromt.data_catalog import DataCatalog

if HAS_XUGRID:
    import xugrid as xu

logger = getLogger(__name__)


class ModelRegion:
    """A class to handle all operations to do with the model region."""

    def __init__(
        self,
        bbox: Optional[Bbox] = None,
        geoms: Optional[GeoDataFrame] = None,
        crs: Optional[CRS] = None,
        logger: Logger = logger,
    ):
        self.logger = logger
        self.set(bbox=bbox, geoms=geoms, crs=crs)

    def set(
        self,
        bbox: Optional[Bbox] = None,
        geoms: Optional[GeoDataFrame] = None,
        basin_index: Optional[GeoDataFrame] = None,
        hydrography: Optional[GeoDataFrame] = None,
        crs: Optional[CRS] = None,
    ):
        """Set the the model region."""
        if bbox is not None and geoms is not None:
            raise ValueError("Only supply one of bbox and geoms.")

        self.bbox = bbox
        self.geoms = geoms
        self.crs = crs

    def read(self):
        """Read the model geom from a file."""
        pass

    def write(self):
        """Write the model geom to a file."""
        pass

    # def setup_region(
    #     self,
    ##     region: dict,
    #     hydrography_fn: str = "merit_hydro",
    #     basin_index_fn: str = "merit_hydro_index",
    # ) -> dict:
    #     hydrography_ds : str
    #         Name of data source for hydrography data.
    #     basin_index_fn : str
    #         Name of data source with basin (bounding box) geometries associated with
    #         the 'basins' layer of `hydrography_fn`. Only required if the `region` is
    #         based on a (sub)(inter)basins without a 'bounds' argument.

    #     kind, region = self._parse_region(region, logger=self.logger)
    #     if kind in ["basin", "subbasin", "interbasin"]:
    #         # retrieve global hydrography data (lazy!)
    #         ds_org = self.data_catalog.get_rasterdataset(hydrography_fn)
    #         if "bounds" not in region:
    ##             region.update(basin_index=self.data_catalog.get_source(basin_index_fn))
    #         # get basin geometry
    #         geom, xy = get_basin_geometry(
    #             ds=ds_org,
    #             kind=kind,
    #             logger=self.logger,
    #             **region,
    #         )
    ##         region.update(xy=xy)
    #     elif "bbox" in region:
    #         bbox = region["bbox"]
    #     elif "geom" in region:
    #         geom = region["geom"]
    #         if geom.crs is None:
    #             raise ValueError('Model region "geom" has no CRS')
    #     elif "grid" in region:  # Grid specific - should be removed in the future
    #         geom = region["grid"].raster.box
    #     elif "model" in region:
    #         geom = region["model"].region
    #     else:
    #         raise ValueError(f"model region argument not understood: {region}")

    def _parse_region(
        self, region: Dict[str, Any], data_catalog: Optional[DataCatalog], logger=logger
    ) -> RegionSpecifyer:
        # popitem returns last inserted, we want first
        kind = next(iter(region.keys()))
        value = region.pop(kind)
        if isinstance(value, np.ndarray):
            value = value.tolist()  # array to list
        flat_region_dict = {}
        if kind == "bbox":
            flat_region_dict = {
                "kind": "bbox",
                **dict(zip(["xmin", "ymin", "xmax", "ymax"], value)),
            }
        elif kind in MODELS:
            model_class = MODELS.load(kind)
            flat_region_dict = {
                "kind": kind,
                "mod": model_class.__init__(root=value, mode="r", logger=logger),
            }
        elif isinstance(value, (Path, str)):
            flat_region_dict = {"kind": kind, "path": Path(value)}
            if kind == "basin":
                flat_region_dict["sub_kind"] = "point_geom"
        elif kind == "basin":
            if isinstance(value, BasinIdType):
                flat_region_dict = {"kind": "basin", "sub_kind": "id", "id": value}
            elif isinstance(value, (list, tuple)):
                if isinstance(value[0], int):
                    flat_region_dict = {
                        "kind": "basin",
                        "sub_kind": "ids",
                        "ids": value,
                    }
                elif isinstance(value[0], float) and len(value) == 2:
                    flat_region_dict = {
                        "kind": "basin",
                        "sub_kind": "point",
                        "x_coord": value[0],
                        "y_coord": value[1],
                    }
                elif isinstance(value[0], float) and len(value) == 4:
                    flat_region_dict = {
                        "kind": "basin",
                        "sub_kind": "bbox",
                        **dict(zip(["xmin", "ymin", "xmax", "ymax"], value)),
                    }
                elif isinstance(value[0], list) and len(value) == 2:
                    flat_region_dict = {
                        "kind": "basin",
                        "sub_kind": "points",
                        "x_coords": value[0],
                        "y_coords": value[1],
                    }
                elif isinstance(value, Path):
                    flat_region_dict = {
                        "kind": "basin",
                        "sub_kind": "point_geom",
                        "path": value,
                    }

            flat_region_dict = {**flat_region_dict, **region}
        else:
            flat_region_dict = region

        return RegionSpecifyer(spec=flat_region_dict)  # type: ignore

    """
    * {'basin': [xmin, ymin, xmax, ymax]}

    * {'basin': [xmin, ymin, xmax, ymax], 'outlets': true}

    * {'basin': [xmin, ymin, xmax, ymax], '<variable>': threshold}

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


    * {'subbasin': [xmin, ymin, xmax, ymax], '<variable>': threshold}


    Interbasins are similar to subbasins but are bounded by a bounding box or
    geometry and do not include all upstream area. Common use-cases include:

    * {'interbasin': [xmin, ymin, xmax, ymax], '<variable>': threshold}

    * {'interbasin': [xmin, ymin, xmax, ymax], 'xy': [x, y]}
    """

    #     if "geom" in kwarg and np.all(kwarg["geom"].geometry.type == "Point"):
    #         xy = (
    #             kwarg["geom"].geometry.x.values,
    #             kwarg["geom"].geometry.y.values,
    #         )
    #         kwarg = dict(xy=xy)
    #     return kwarg
