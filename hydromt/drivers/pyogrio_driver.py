"""Driver to read geodataframes using Pyogrio."""

from logging import Logger, getLogger
from typing import List, Optional

import geopandas as gpd
from pyogrio import read_dataframe, read_info, write_dataframe
from pyproj import CRS
from shapely.geometry.base import BaseGeometry

from hydromt._typing import Bbox, Geom, GpdShapeGeom, StrPath
from hydromt._typing.error import NoDataStrategy
from hydromt.drivers.geodataframe_driver import GeoDataFrameDriver

logger: Logger = getLogger(__name__)


class PyogrioDriver(GeoDataFrameDriver):
    """Driver to read GeoDataFrames using the `pyogrio` package."""

    name = "pyogrio"

    def read_data(
        self,
        uris: List[str],
        *,
        mask: Optional[Geom] = None,
        crs: Optional[CRS] = None,
        predicate: str = "intersects",
        logger: Logger = logger,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        **kwargs,
    ) -> gpd.GeoDataFrame:
        """
        Read data using pyogrio.

        args:
        """
        if len(uris) != 1:
            raise ValueError("Length of uris for Pyogrio Driver must be 1.")
        _uri = uris[0]
        if mask is not None:
            bbox = bbox_from_file_and_filters(_uri, mask=mask, crs=crs)
        else:
            bbox = None
        # TODO: add **self.options, see see https://github.com/Deltares/hydromt/issues/899
        return read_dataframe(_uri, bbox=bbox)

    def write(
        self,
        gdf: gpd.GeoDataFrame,
        path: StrPath,
        **kwargs,
    ) -> None:
        """
        Write out a GeoDataFrame to file using pyogrio.

        args:
        """
        write_dataframe(gdf, path, **kwargs)


def bbox_from_file_and_filters(
    uri: str,
    mask: GpdShapeGeom,
    crs: Optional[CRS] = None,
) -> Optional[Bbox]:
    """Create a bbox from the file metadata and filter options.

    Pyogrio does not accept a mask, and requires a bbox in the same CRS as the data.
    This function takes the mask filter and crs of the input data
    and returns a bbox in the same crs as the data based on the input filters.

    Parameters
    ----------
    uri: str,
        URI of the data.
    mask: GeoDataFrame | GeoSeries | BaseGeometry
        mask to filter the data while reading.
    crs: pyproj.CRS | None
        coordinate reference system of the file. If already set,
        this argument is ignored.
    """
    if issubclass(type(mask), BaseGeometry):
        mask = gpd.GeoSeries(mask, crs=crs)

    source_crs = None
    if source_crs_str := read_info(uri).get("crs"):
        source_crs = CRS.from_user_input(source_crs_str)
    elif crs:
        source_crs = CRS.from_user_input(crs)
    else:
        raise ValueError("CRS must be set either in the file or as an argument.")

    if source_crs is not None and source_crs != mask.crs:
        mask = mask.to_crs(source_crs)

    return tuple(mask.total_bounds)
