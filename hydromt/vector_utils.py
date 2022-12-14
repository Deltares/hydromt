from shapely.ops import split, snap, unary_union
from shapely.geometry import LineString, Point, MultiLineString, MultiPoint
import numpy as np
import geopandas as gpd
import shapely
import hydromt  # to load raster dataset accessor
import xarray as xr
from typing import Tuple


def split_line_by_point(
    line: LineString, point: Point, tolerance: float = 1.0e-5
) -> MultiLineString:
    """Split a single line with a point.

    Parameters
    ----------
    line : LineString
        line
    point : Point
        point
    tolerance : float, optional
        tolerance to snap the point to the line, by default 1.0e-5

    Returns
    -------
    MultiLineString
        splitted line
    """
    return split(snap(line, point, tolerance), point)


def split_line_equal(
    line: LineString, approx_length: float, tolerance: float = 1.0e-5
) -> MultiLineString:
    """Split line into segments with equal length.

    Parameters
    ----------
    line : LineString
        line to split
    approx_length : float
        Based in this approximate length the number of line segments is determined.
    tolerance : float, optional
        tolerance to snap the point to the line, by default 1.0e-5

    Returns
    -------
    MultiLineString
        line splitted in segments of equal length
    """
    return split_line_by_point(
        line, approx_dist_points(line, approx_length), tolerance=tolerance
    )


def line_to_points(line: LineString, dist: float = None, n: int = None) -> MultiPoint:
    """Get points along line based on a distance `dist` or number of points `n`.

    Parameters
    ----------
    line : LineString
        line
    dist : float
        distance between points
    n: integer
        numer of points

    Returns
    -------
    MultiPoint
        points
    """
    if dist is not None:
        distances = np.arange(0, line.length, dist)
    elif n is not None:
        distances = np.linspace(0, line.length, n)
    else:
        ValueError('Either "dist" or "n" should be provided')
    points = unary_union(
        [line.interpolate(distance) for distance in distances[:-1]] + [line.boundary[1]]
    )
    return points


def connect_line(
    p0: Point,
    l0: LineString,
    endpoints: gpd.GeoDataFrame,
    lines: gpd.GeoDataFrame,
    tolerance=1,
    lines_index="level_0",
):
    line = lines.loc[l0].geometry
    start = endpoints.loc[p0].geometry
    bounds = line.boundary
    assert start in bounds
    end = bounds[-1] if start == bounds[0] else bounds[0]
    dist, idxs = endpoints.sindex.nearest(end, 2)
    lev0 = endpoints.iloc[idxs][lines_index].values
    select = lev0 != l0
    if dist[select] > tolerance or not np.any(select):
        return p0, l0
    p0 = idxs[select][0]
    l0 = lev0[select][0]
    return p0, l0


def connect_lines(
    l0: LineString, lines: gpd.GeoDataFrame, tolerance=1
) -> Tuple[np.ndarray]:
    endpoints = (
        lines.boundary.explode()
        .rename("geometry")
        .reset_index()
        .set_geometry("geometry")
    )
    lines_index = "level_0" if lines.index.name is None else lines.index.name
    _, idxs, cnts = np.unique(
        np.stack([endpoints.geometry.x, endpoints.geometry.y]),
        axis=1,
        return_index=True,
        return_counts=True,
    )
    boundaries = endpoints.iloc[idxs[cnts == 1]]
    p0 = boundaries.index[boundaries[lines_index] == l0].values[0]

    idx, idx_ds = [], []
    while True:
        p1, l1 = connect_line(p0, l0, endpoints, lines, tolerance, lines_index)
        idx.append(l0)
        idx_ds.append(l1)
        if l0 == l1:
            break
        p0, l0 = p1, l1
    return idx, idx_ds


def gdf_lines_to_points(gdf_lines: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    points = gpd.GeoDataFrame(
        gdf_lines.assign(
            geometry=gdf_lines.apply(
                lambda x: gpd.points_from_xy(*zip(*x.geometry.coords)), axis=1
            )
        ).explode()
    )
    points.set_crs(gdf_lines.crs, inplace=True)
    return points


def gdf_split_line_equal(gdf: gpd.GeoDataFrame, dist: float) -> gpd.GeoDataFrame:
    gdf_splitted = (
        gdf.assign(
            geometry=gdf.apply(lambda x: split_line_equal(x.geometry, dist), axis=1)
        )
        .explode()
        .reset_index(drop=True)
    )
    return gdf_splitted


def sample_line_transect(
    da: xr.DataArray,
    gdf: gpd.GeoDataFrame,
    dist: float = None,
    n: int = None,  # fstat=np.nanmean
) -> gpd.GeoDataFrame:
    """Sample raster along line transect based on a fixed distance `dist` or number of points `n`

    Parameters
    ----------
    da : xr.DataArray
        raster
    gdf : gpd.GeoDataFrame
        lines
    dist : float
        distance between points
    n: integer
        numer of points

    Returns
    -------
    gpd.GeoDataFrame
        point GeoDataframe with sampled values
    """
    points = gdf.assign(
        geometry=gdf.apply(lambda x: line_to_points(x.geometry, dist), axis=1)
    ).explode()
    points["values"] = da.raster.sample(points, wdw=0).values
    return points  # ["values"].groupby(level=0).apply(fstat)
