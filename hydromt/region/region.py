"""A class to handle all functionalities to do with model regions."""

from logging import Logger, getLogger
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from geopandas import GeoDataFrame, GeoSeries
from pyproj import CRS
from shapely import box

from hydromt._typing import Bbox, Geom, Predicate
from hydromt.region._specifyers import RegionSpecifyer

logger = getLogger(__name__)


class Region:
    """A class to handle all operations to do with the model region."""

    def __init__(
        self,
        region_dict: Dict[str, Any],
        logger: Logger = logger,
    ):
        self._dict: Dict[str, Any] = region_dict
        self._data: Optional[GeoDataFrame] = None
        self._spec = self._parse_region(region_dict, logger=logger)
        self.logger = logger

    def write(self, path):
        """Write the geometry to a file."""
        if self._data is None:
            raise ValueError(
                "Region is not yet initialised. use the construct() method."
            )
        else:
            self._data.to_file(path)

    def construct(self) -> GeoDataFrame:
        """Calculate the actual geometry based on the specification."""
        self._data = self._spec.construct()
        return self._data

    @staticmethod
    def _parse_region(
        region: Dict[str, Any],
        logger=logger,
    ) -> RegionSpecifyer:
        # popitem returns last inserted, we want first
        kind = next(iter(region.keys()))
        value = region.pop(kind)
        if isinstance(value, np.ndarray):
            value = value.tolist()

        flat_region_dict: Dict[str, Any] = {}
        if kind == "bbox":
            flat_region_dict = {
                "kind": "bbox",
                **dict(zip(["xmin", "ymin", "xmax", "ymax"], value)),
            }
        elif isinstance(value, (GeoDataFrame, GeoSeries)):
            flat_region_dict = {"kind": kind, "geom": value}
        elif isinstance(value, (Path, str)):
            if kind == "geom":
                flat_region_dict = {"kind": "geom_file", "path": Path(value)}
            else:
                flat_region_dict = {"kind": kind, "path": Path(value)}
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
        geom = GeoDataFrame(geometry=[box(*bbox)], crs=crs)
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
