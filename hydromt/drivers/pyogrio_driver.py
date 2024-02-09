"""Driver to read geodataframes using Pyogrio."""
from logging import Logger
from typing import Optional, Union

import geopandas as gpd
from pyogrio import read_dataframe, read_info
from pyproj import CRS
from shapely.geometry.base import BaseGeometry

from hydromt._typing import Bbox, Geom
from hydromt.gis import parse_geom_bbox_buffer

from .geodataframe_driver import GeoDataFrameDriver

GEOM_TYPES = Union[Geom, BaseGeometry]


class PyogrioDriver(GeoDataFrameDriver):
    """Driver to read GeoDataFrames using the `pyogrio` package."""

    def read(
        self,
        uri: str,
        bbox: Optional[Bbox] = None,
        mask: Optional[gpd.GeoDataFrame] = None,
        buffer: float = 0,
        crs: Optional[CRS] = None,
        predicate: str = "intersects",
        logger: Optional[Logger] = None,
        # handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
    ) -> gpd.GeoDataFrame:
        """
        Read data using pyogrio.

        args:
        """
        if bbox is not None:  # buffer bbox
            bbox: Geom = parse_geom_bbox_buffer(mask, bbox, buffer, crs)
        if mask is not None:  # buffer mask
            mask: Geom = parse_geom_bbox_buffer(mask, bbox, buffer, crs)
        bbox_reader = bbox_from_file_and_filters(uri, bbox, mask, crs)
        return read_dataframe(uri, bbox=bbox_reader, mode="r")


def bbox_from_file_and_filters(
    uri: str,
    bbox: Optional[GEOM_TYPES] = None,
    mask: Optional[GEOM_TYPES] = None,
    crs: Optional[CRS] = None,
) -> Optional[Bbox]:
    """Create a bbox from the file metadata and filter options.

    Pyogrio does not accept a mask, and requires a bbox in the same CRS as the data.
    This function takes the possible bbox filter, mask filter and crs of the input data
    and returns a bbox in the same crs as the data based on the input filters.
    As pyogrio currently does not support filtering using a mask, the mask is converted
    to a bbox and the bbox is returned so that the data has some geospatial filtering.

    Parameters
    ----------
    uri: str,
        URI of the data.
    bbox: GeoDataFrame | GeoSeries | BaseGeometry
        bounding box to filter the data while reading.
    mask: GeoDataFrame | GeoSeries | BaseGeometry
        mask to filter the data while reading.
    crs: pyproj.CRS
        coordinate reference system of the bounding box or geometry. If already set,
        this argument is ignored.
    """
    if bbox is not None and mask is not None:
        raise ValueError(
            "Both 'bbox' and 'mask' are provided. Please provide only one."
        )
    if bbox is None and mask is None:
        return None
    if source_crs_str := read_info(uri).get("crs"):
        source_crs = CRS(source_crs_str)
    elif crs:
        source_crs = crs
    else:  # assume WGS84
        source_crs = CRS("EPSG:4326")

    if mask is not None:
        bbox = mask

    # convert bbox to geom with input crs (assume WGS84 if not provided)
    crs = crs if crs is not None else CRS.from_user_input(4326)
    if issubclass(type(bbox), BaseGeometry):
        bbox = gpd.GeoSeries(bbox, crs=crs)
    bbox = bbox if bbox.crs is not None else bbox.set_crs(crs)
    return tuple(bbox.to_crs(source_crs).total_bounds)
