"""Driver to read geodataframes using Pyogrio."""
from logging import Logger

import geopandas as gpd
from pyogrio import read_dataframe, read_info
from pyproj import CRS
from shapely.geometry.base import BaseGeometry

from hydromt.gis_utils import parse_geom_bbox_buffer
from hydromt.typing import GEOM_TYPES, GPD_TYPES

# from hydromt.nodata import NoDataStrategy
from .geodataframe_driver import GeoDataFrameDriver


class PyogrioDriver(GeoDataFrameDriver):
    """Driver to read GeoDataFrames using the `pyogrio` package."""

    def read(
        self,
        bbox: list[int] | None = None,
        mask: gpd.GeoDataFrame | None = None,
        buffer: float = 0,
        predicate: str = "intersects",
        logger: Logger | None = None,
        # handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
    ) -> gpd.GeoDataFrame:
        """
        Read data using pyogrio.

        args:
        """
        if bbox:  # buffer bbox
            bbox_gpd: GPD_TYPES = parse_geom_bbox_buffer(bbox, mask, buffer, self.crs)
        bbox_reader = bbox_from_file_and_filters(self.uri, bbox_gpd, mask, self.crs)
        return read_dataframe(self.uri, bbox=bbox_reader, mode="r")


def bbox_from_file_and_filters(
    uri: str,
    bbox: GEOM_TYPES | None = None,
    mask: GEOM_TYPES | None = None,
    crs: CRS | None = None,
) -> tuple[float, float, float, float] | None:
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
