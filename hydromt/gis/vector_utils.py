#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""GIS related vector convenience functions."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import geopandas as gpd
import numpy as np
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

logger = logging.getLogger(__name__)

__all__ = [
    "nearest",
    "nearest_merge",
    "filter_gdf",
]


def nearest_merge(
    gdf1: gpd.GeoDataFrame,
    gdf2: gpd.GeoDataFrame,
    columns: Optional[list] = None,
    max_dist: Optional[float] = None,
    overwrite: bool = False,
    inplace: bool = False,
    logger=logger,
) -> gpd.GeoDataFrame:
    """Merge attributes of gdf2 with the nearest feature of gdf1.

    Output is optionally bounded by a maximum distance `max_dist`.
    Unless `overwrite = True`, gdf2 values are only
    merged where gdf1 has missing values.

    Parameters
    ----------
    gdf1, gdf2: geopandas.GeoDataFrame
        Source `gdf1` and destination `gdf2` geometries.
    columns : list of str, optional
        Names of columns in `gdf2` to merge, by default None
    max_dist : float, optional
        Maximum distance threshold for merge, by default None, i.e.: no threshold.
    overwrite : bool, optional
        If False (default) gdf2 values are only merged where gdf1 has missing values,
        i.e. NaN values for existing columns or missing columns.
    inplace : bool,
        If True, apply the merge to gdf1, otherwise return a new object.
    logger:
        The logger to use.

    Returns
    -------
    gpd.GeoDataFrame
        Merged GeoDataFrames
    """
    # Get nearest index right
    idx_nn, dst = nearest(gdf1, gdf2)
    if not inplace:
        gdf1 = gdf1.copy()
    valid = dst < max_dist if max_dist is not None else np.ones_like(idx_nn, dtype=bool)
    gdf1["distance_right"] = dst
    gdf1["index_right"] = -1
    gdf1.loc[valid, "index_right"] = idx_nn[valid]

    if not overwrite:
        new_cols = [c for c in gdf2.columns if c not in gdf1.columns]
        gdf1.loc[:, new_cols] = np.nan
        return gdf1.combine_first(gdf2)
    else:
        left_only_cols = [c for c in gdf1.columns if c not in gdf2.columns]
        return gdf1[:, left_only_cols].join(gdf2, join="left")


def nearest(
    gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the index of and distance [m] to the nearest geometry.

    For Line geometries in `gdf1` the nearest geometry is based line center point
    and for polygons on its representative point. Mixed geometry types are not
    yet supported.

    Note: Since geopandas v0.10.0 it contains a sjoin_nearest method which is very
    similar and should.


    Parameters
    ----------
    gdf1, gdf2: geopandas.GeoDataFrame
        Source `gdf1` and destination `gdf2` geometries.

    Returns
    -------
    index: ndarray
        index of nearest `gdf2` geometry
    dst: ndarray of float
        distance to the nearest `gdf2` geometry
    """
    if np.all(gdf1.type == "Point"):
        pnts = gdf1.geometry.copy()
    elif np.all(np.isin(gdf1.type, ["LineString", "MultiLineString"])):
        pnts = gdf1.geometry.interpolate(0.5, normalized=True)  # mid point
    elif np.all(np.isin(gdf1.type, ["Polygon", "MultiPolygon"])):
        pnts = gdf1.geometry.representative_point()  # inside polygon
    else:
        raise NotImplementedError("Mixed geometry dataframes are not yet supported.")
    if gdf1.crs != gdf2.crs:
        pnts = pnts.to_crs(gdf2.crs)
    # find nearest
    other = pnts.geometry.values
    idx = gdf2.sindex.nearest(other, return_all=False)[1]
    # get distance in meters
    gdf2_nearest = gdf2.iloc[idx]
    if gdf2_nearest.crs.is_geographic:
        pnts = pnts.to_crs(3857)  # web mercator
        gdf2_nearest = gdf2_nearest.to_crs(3857)
    dst = gdf2_nearest.distance(pnts, align=False).values
    return gdf2.index.values[idx], dst


def filter_gdf(gdf, geom=None, bbox=None, crs=None, predicate="intersects"):
    """Filter GeoDataFrame geometries based on geometry mask or bounding box."""
    gtypes = (gpd.GeoDataFrame, gpd.GeoSeries, BaseGeometry)
    if bbox is not None and geom is None:
        if crs is None:
            crs = gdf.crs
        geom = gpd.GeoSeries([box(*bbox)], crs=crs)
    elif geom is not None and not isinstance(geom, gtypes):
        raise ValueError(
            f"Unknown geometry mask type {type(geom).__name__}. "
            "Provide geopandas GeoDataFrame, GeoSeries or shapely geometry."
        )
    elif bbox is None and geom is None:
        raise ValueError("Either geom or bbox is required.")
    if not isinstance(geom, BaseGeometry):
        # reproject
        if geom.crs is None and gdf.crs is not None:
            geom = geom.set_crs(gdf.crs)
        elif gdf.crs is not None and geom.crs != gdf.crs:
            geom = geom.to_crs(gdf.crs)
        # convert geopandas to geometry
        geom = geom.union_all()
    idx = np.sort(gdf.sindex.query(geom, predicate=predicate))
    return idx
