"""A class to handle all functionalities to do with model regions."""

from logging import Logger, getLogger
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from geopandas import GeoDataFrame, GeoSeries
from pyproj import CRS

from hydromt._compat import HAS_XUGRID
from hydromt._typing.type_def import BasinIdType
from hydromt.data_catalog import DataCatalog
from hydromt.models._region.specifyers import RegionSpecifyer

if HAS_XUGRID:
    pass

logger = getLogger(__name__)


class Region:
    """A class to handle all operations to do with the model region."""

    def __init__(
        self,
        region: Dict[str, Any],
        data_catalog: Optional[DataCatalog] = None,
        basin_index: Optional[Union[str, GeoDataFrame]] = None,
        hydrography: Optional[Union[str, GeoDataFrame]] = None,
        crs: Optional[CRS] = None,
        logger: Logger = logger,
    ):
        self.logger = logger

        self.set(
            region=region,
            data_catalog=data_catalog,
            hydrography=hydrography,
            basin_index=basin_index,
        )

    def set(
        self,
        region: Dict[str, Any],
        data_catalog: Optional[DataCatalog] = None,
        basin_index: Optional[Union[str, GeoDataFrame]] = None,
        hydrography: Optional[Union[str, GeoDataFrame]] = None,
        crs: Optional[CRS] = None,
    ):
        """Set the the model region."""
        self.crs = crs
        self.region_specifyer = self._parse_region(
            region=region,
            data_catalog=data_catalog,
            hydrography=hydrography,
            basin_index=basin_index,
            logger=logger,
        )

    @staticmethod
    def _parse_region(
        region: Dict[str, Any],
        data_catalog: Optional[DataCatalog] = None,
        basin_index: Optional[Union[str, GeoDataFrame]] = None,
        hydrography: Optional[Union[str, GeoDataFrame]] = None,
        logger=logger,
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
        # TODO: fimplement this in a way that does not result in circular imports
        # elif kind in MODELS:
        #     model_class = MODELS.load(kind)
        #     flat_region_dict = {
        #         "kind": kind,
        #         "mod": model_class.__init__(root=value, mode="r", logger=logger),
        #     }
        elif isinstance(value, (GeoDataFrame, GeoSeries)):
            flat_region_dict = {"kind": kind, "geom": value}
        elif isinstance(value, (Path, str)):
            if kind == "geom":
                flat_region_dict = {"kind": "geom_file", "path": Path(value)}
            else:
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
            raise ValueError(f"Unknown region kind: {kind}")

        return RegionSpecifyer(spec=flat_region_dict)  # type: ignore
