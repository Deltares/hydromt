"""A class to handle all functionalities to do with model regions."""

from logging import Logger, getLogger
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from geopandas import GeoDataFrame, GeoSeries
from pyproj import CRS

from hydromt._compat import HAS_XUGRID
from hydromt._typing.type_def import BasinIdType
from hydromt.models._region.specifyers import RegionSpecifyer

if HAS_XUGRID:
    pass

logger = getLogger(__name__)


class Region:
    """A class to handle all operations to do with the model region."""

    def __init__(
        self,
        region: Dict[str, Any],
        basin_index: Optional[Union[str, GeoDataFrame]] = None,
        hydrography: Optional[Union[str, GeoDataFrame]] = None,
        crs: Optional[CRS] = None,
        logger: Logger = logger,
    ):
        self.logger = logger

        self._data = None
        self.set(
            region=region,
            hydrography=hydrography,
            basin_index=basin_index,
        )

    def set(
        self,
        region: Dict[str, Any],
        basin_index: Optional[Union[str, GeoDataFrame]] = None,
        hydrography: Optional[Union[str, GeoDataFrame]] = None,
        crs: Optional[CRS] = None,
    ):
        """Set the the model region."""
        self.crs = crs
        self.region_specifyer = self._parse_region(
            region=region,
            hydrography=hydrography,
            basin_index=basin_index,
            logger=logger,
        )

    def get_geom(self):
        pass

    @property
    def crs(self):
        return self

    @staticmethod
    def _parse_region(
        region: Dict[str, Any],
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


def filter_gdf(gdf: GeoDataFrame, region: Region, predicate: Predicate = "intersects"):
    """Filter GeoDataFrame geometries based on geometry mask or bounding box."""

    geom = region.get_geom()
    if not isinstance(geom, BaseGeometry):
        # reproject
        if geom.crs is None and gdf.crs is not None:
            geom = geom.set_crs(gdf.crs)
        elif gdf.crs is not None and geom.crs != gdf.crs:
            geom = geom.to_crs(gdf.crs)
        # convert geopandas to geometry
        geom = geom.unary_union
    idx = np.sort(gdf.sindex.query(geom, predicate=predicate))
    return idx


def parse_geom_bbox_buffer(
    geom: Optional[Geom] = None,
    bbox: Optional[Bbox] = None,
    buffer: float = 0.0,
    crs: Optional[CRS] = None,
):
    """Parse geom or bbox to a (buffered) geometry.

    Arguments
    ---------
    geom : geopandas.GeoDataFrame/Series, optional
        A geometry defining the area of interest.
    bbox : array-like of floats, optional
        (xmin, ymin, xmax, ymax) bounding box of area of interest
        (in WGS84 coordinates).
    buffer : float, optional
        Buffer around the `bbox` or `geom` area of interest in meters. By default 0.
    crs: pyproj.CRS, optional
        projection of the bbox or geometry. If the geometry already has a crs, this
        argument is ignored. Defaults to EPSG:4236.

    Returns
    -------
    geom: geometry
        the actual geometry
    """
    if crs is None:
        crs = CRS("EPSG:4326")
    if geom is None and bbox is not None:
        # convert bbox to geom with crs EPGS:4326 to apply buffer later
        geom = gpd.GeoDataFrame(geometry=[box(*bbox)], crs=crs)
    elif geom is None:
        raise ValueError("No geom or bbox provided.")
    elif geom.crs is None:
        geom = geom.set_crs(crs)

    if buffer > 0:
        # make sure geom is projected > buffer in meters!
        if geom.crs.is_geographic:
            geom = geom.to_crs(3857)
        geom = geom.buffer(buffer)
    return geom
