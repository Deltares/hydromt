"""Driver to read geodataframes using Pyogrio."""
from enum import Enum
from functools import wraps
from inspect import signature
from logging import Logger, getLogger
from pathlib import Path
from typing import Callable, Optional

import geopandas as gpd
from pyogrio import read_dataframe, read_info
from pyproj import CRS
from shapely.geometry.base import BaseGeometry

from hydromt._typing import Bbox, Geom, GpdShapeGeom
from hydromt.drivers.geodataframe_driver import GeoDataFrameDriver
from hydromt.gis import parse_geom_bbox_buffer

logger: Logger = getLogger(__name__)


class PyogrioExtension(Enum):
    """All compatible extensions for Pyogrio."""

    shapefile = ".shp"
    geopackage = ".gpkg"
    geojson = ".geojson"
    flatgeobuf = ".fgb"


def _read_df_default(
    extension: PyogrioExtension,
    uri: str,
    bbox: Optional[Bbox],
    mode: str,
    logger: Logger = logger,
) -> gpd.GeoDataFrame:
    """`read_dataframe` for different gdal drivers.

    Warnings are thrown for invalid kwargs for different drivers, so this makes pyogrio explicit.
    So far only geobuf accepts the mode argument.
    """
    return read_dataframe(uri, bbox=bbox)


class _EnumDispatchRegistry:
    def __init__(self, enum: Enum):
        self._enum = enum
        self._impls = {}

    def register(self, func: Callable) -> Callable:
        try:
            annotation = signature(func).parameters.get("extension")._annotation
            if isinstance(annotation, Enum):
                self._impls[annotation.name] = func
            else:
                raise ValueError(
                    f"First argument of function {func} must be PyogrioExtension."
                )
        except AttributeError:
            raise ValueError(f"Registered function {func} needs annotations.")

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    def _read_df(self, ext: PyogrioExtension, *args, **kwargs):
        return self._impls.get(ext.name, _read_df_default)(ext, *args, **kwargs)


_read_df_registry = _EnumDispatchRegistry(PyogrioExtension)


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
        logger: Logger = logger,
        # handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
    ) -> gpd.GeoDataFrame:
        """
        Read data using pyogrio.

        args:
        """
        extension = PyogrioExtension(Path(uri).suffix)
        if bbox is not None:  # buffer bbox
            bbox: Geom = parse_geom_bbox_buffer(mask, bbox, buffer, crs)
        if mask is not None:  # buffer mask
            mask: Geom = parse_geom_bbox_buffer(mask, bbox, buffer, crs)
        bbox_reader = bbox_from_file_and_filters(uri, bbox, mask, crs)
        logger.debug(
            f"Reading datafrom from uri: '{uri}' with extension: '{extension.name}'"
        )
        return _read_df_registry._read_df(extension, uri, bbox=bbox_reader, mode="r")


def bbox_from_file_and_filters(
    uri: str,
    bbox: Optional[GpdShapeGeom] = None,
    mask: Optional[GpdShapeGeom] = None,
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
