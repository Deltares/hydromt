"""Driver to read geodataframes using Pyogrio."""
from logging import Logger
from typing import Optional

import geopandas as gpd
from geopandas import GeoDataFrame
from pyogrio import read_dataframe, read_info
from pyproj import CRS

from hydromt._typing import Bbox

from .geodataframe_driver import GeoDataFrameDriver


class PyogrioDriver(GeoDataFrameDriver):
    """Driver to read GeoDataFrames using the `pyogrio` package."""

    def read(
        self,
        uri: str,
        region: Optional[gpd.GeoDataFrame] = None,
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
        bbox_reader = bbox_from_file_and_filters(uri, region, crs)
        return read_dataframe(uri, bbox=bbox_reader, mode="r")


def bbox_from_file_and_filters(
    uri: str,
    geom: Optional[GeoDataFrame],
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
    if geom is None:
        return None
    if source_crs_str := read_info(uri).get("crs"):
        source_crs = CRS(source_crs_str)
    elif crs:
        source_crs = crs
    else:  # assume WGS84
        source_crs = CRS("EPSG:4326")

    # convert bbox to geom with input crs (assume WGS84 if not provided)
    crs = crs if crs is not None else CRS.from_user_input(4326)
    return tuple(geom.to_crs(source_crs).total_bounds)
