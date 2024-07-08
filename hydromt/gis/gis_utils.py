#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""GIS related convenience functions."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import geopandas as gpd
import numpy as np
from pyogrio import read_info
from pyproj import CRS
from pyproj.transformer import Transformer
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from hydromt._typing import Bbox, Geom, GpdShapeGeom, Zoom
from hydromt.gis.raster_utils import cellres

__all__ = [
    "axes_attrs",
    "bbox_from_file_and_filters",
    "parse_crs",
    "parse_geom_bbox_buffer",
    "zoom_to_overview_level",
    "to_geographic_bbox",
    "utm_crs",
]

logger = logging.getLogger(__name__)


# REPROJ
def utm_crs(bbox):
    """Return wkt string of nearest UTM projects.

    Parameters
    ----------
    bbox : array-like of floats
        (xmin, ymin, xmax, ymax) bounding box in latlon WGS84 (EPSG:4326) coordinates

    Returns
    -------
    crs: pyproj.CRS
        CRS of UTM projection
    """
    left, bottom, right, top = bbox
    x = (left + right) / 2
    y = (top + bottom) / 2
    kwargs = dict(zone=int(np.ceil((x + 180) / 6)))
    # BUGFIX hydroMT v0.3.5: south=False doesn't work only add south=True if y<0
    if y < 0:
        kwargs.update(south=True)
    # BUGFIX hydroMT v0.4.6: add datum
    epsg = CRS(proj="utm", datum="WGS84", ellps="WGS84", **kwargs).to_epsg()
    return CRS.from_epsg(epsg)


def parse_crs(crs: Any, bbox: List[float] = None) -> CRS:
    """Parse crs string to pyproj.CRS.

    Parameters
    ----------
    crs: Any
        crs int, wkt string, or pyproj.CRS object
        if crs is 'utm' the best utm zone is calculated based on the bbox
    bbox: List[float], optional
        bounding box of the data, required for 'utm' crs

    Returns
    -------
    crs: pyproj.CRS
        coordinate reference system
    """
    if crs == "utm":
        if bbox is not None:
            crs = utm_crs(bbox)
        else:
            raise ValueError('CRS "utm" requires bbox')
    else:
        crs = CRS.from_user_input(crs)
    return crs


def axes_attrs(crs):
    """Provide CF-compliant variable names and metadata for axes.

    Parameters
    ----------
    crs: pyproj.CRS
        coordinate reference system

    Returns
    -------
    x_dim: str - variable name of x dimension (e.g. 'x')
    y_dim: str - variable name of y dimension (e.g. 'lat')
    x_attr: dict - attributes of variable x
    y_attr: dict - attributes of variable y
    """
    # # check for type of crs
    if not isinstance(crs, CRS):
        crs = CRS.from_user_input(crs)
    if crs.is_geographic:
        x_dim, y_dim = "longitude", "latitude"
    else:
        x_dim, y_dim = "x", "y"
    cf_coords = crs.cs_to_cf()
    x_attrs = [c for c in cf_coords if c["axis"] == "X"][0]
    y_attrs = [c for c in cf_coords if c["axis"] == "Y"][0]
    return x_dim, y_dim, x_attrs, y_attrs


def parse_geom_bbox_buffer(
    geom: Optional[Geom] = None,
    bbox: Optional[Bbox] = None,
    buffer: float = 0.0,
    crs: Optional[CRS] = None,
) -> Geom:
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


def to_geographic_bbox(
    bbox: List[float], source_crs: Optional[CRS] = None
) -> List[float]:
    """Convert a bbox to geographic coordinates (EPSG:4326).

    Parameters
    ----------
    bbox: List[float]
        (xmin, ymin, xmax, ymax) bounding box in the source crs.
    source_crs: pyproj.CRS, optional
        coordinate reference system of the bounding box. If not provided, the
        bounding box is assumed to be in WGS84 (EPSG:4326).

    Returns
    -------
    bbox: List[float]
        (xmin, ymin, xmax, ymax) bounding box in geographic coordinates (EPSG:4326).
    """
    target_crs = CRS.from_user_input(4326)
    if source_crs is None:
        logger.warning("No CRS was set. Skipping CRS conversion")
    elif source_crs != target_crs:
        bbox = Transformer.from_crs(source_crs, target_crs).transform_bounds(*bbox)

    return bbox


def bbox_from_file_and_filters(
    path: str,
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
    path: IOBase,
        path to the opened file.
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
    if source_crs_str := read_info(path).get("crs"):
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


def zoom_to_overview_level(
    zoom: Zoom,
    mask: Optional[Geom] = None,
    zls_dict: Optional[Dict[int, float]] = None,
    source_crs: Union[CRS, int, None] = None,
) -> Optional[int]:
    """Return overview level of data corresponding to zoom level.

    Parameters
    ----------
    zoom: Zoom
        Overview level or tuple with resolution and unit
    mask: gpd.GeoSeries, optional
        Geometry to determine res if zoom_level or source in degree
    zls_dict: Dict[int, float], optional
        Dictionary translating overview level to resolution.
    source_crs: pyproj.CRS, optional
        Source crs to determine res if zoom_level tuple is provided
        with different unit than source_crs
    """
    # check zoom level
    if zls_dict is None or len(zls_dict) == 0:
        return None
    elif isinstance(zoom, int):
        overview_level = zoom
        if overview_level not in zls_dict:
            raise ValueError(
                f"Overview level {overview_level} not defined."
                f"Select from {zls_dict}."
            )
        dst_res = zls_dict[overview_level]
    elif (
        isinstance(zoom, tuple)
        and isinstance(zoom[0], (int, float))
        and isinstance(zoom[1], str)
        and len(zoom) == 2
        and source_crs is not None
    ):
        src_res, src_res_unit = zoom
        # convert res if different unit than crs
        source_crs = CRS.from_user_input(source_crs)
        dst_crs_unit = source_crs.axis_info[0].unit_name
        dst_res = src_res
        if dst_crs_unit != src_res_unit:
            known_units = ["degree", "metre", "US survey foot", "meter", "foot"]
            if src_res_unit not in known_units:
                raise TypeError(
                    f"zoom_level unit {src_res_unit} not understood;"
                    f" should be one of {known_units}"
                )
            if dst_crs_unit not in known_units:
                raise NotImplementedError(
                    f"no conversion available for {src_res_unit} to {dst_crs_unit}"
                )
            conversions = {
                "foot": 0.3048,
                "metre": 1,  # official pyproj units
                "US survey foot": 0.3048,  # official pyproj units
            }  # to meter
            if src_res_unit == "degree" or dst_crs_unit == "degree":
                lat = 0
                if mask is not None:
                    lat = mask.to_crs(4326).centroid.y.item()
                conversions["degree"] = cellres(lat=lat)[1]
            fsrc = conversions.get(src_res_unit, 1)
            fdst = conversions.get(dst_crs_unit, 1)
            dst_res = src_res * fsrc / fdst
        # find nearest zoom level
        res = list(zls_dict.values())[0] / 2
        zls = list(zls_dict.keys())
        smaller = [x < (dst_res + res * 0.01) for x in zls_dict.values()]
        overview_level = (
            zls[-1] if all(smaller) else zls[max(smaller.index(False) - 1, 0)]
        )
    elif source_crs is None:
        raise ValueError("No CRS defined, hence no zoom level can be determined.")
    else:
        raise TypeError(f"zoom_level not understood: {zoom}")
    logger.debug(f"Using overview level {overview_level} ({dst_res:.2f})")
    return overview_level
