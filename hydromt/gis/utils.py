#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""GIS related convenience functions."""
from __future__ import annotations

import glob
import logging
import os
from os.path import dirname
from typing import Optional, Tuple

import geopandas as gpd
import numpy as np
import xarray as xr
from pyflwdir import gis_utils as gis
from pyogrio import read_info
from pyproj import CRS
from pyproj.transformer import Transformer
from rasterio.transform import Affine
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from hydromt import _compat
from hydromt._typing import Bbox, Geom, GpdShapeGeom

__all__ = ["spread2d", "nearest", "nearest_merge"]

logger = logging.getLogger(__name__)

_R = 6371e3  # Radius of earth in m. Use 3956e3 for miles
GDAL_DRIVER_CODE_MAP = {
    "asc": "AAIGrid",
    "blx": "BLX",
    "bmp": "BMP",
    "bt": "BT",
    "dat": "ZMap",
    "dem": "USGSDEM",
    "gen": "ADRG",
    "gif": "GIF",
    "gpkg": "GPKG",
    "grd": "NWT_GRD",
    "gsb": "NTv2",
    "gtx": "GTX",
    "hdr": "MFF",
    "hf2": "HF2",
    "hgt": "SRTMHGT",
    "img": "HFA",
    "jpg": "JPEG",
    "kro": "KRO",
    "lcp": "LCP",
    "mbtiles": "MBTiles",
    "mpr/mpl": "ILWIS",
    "ntf": "NITF",
    "pix": "PCIDSK",
    "png": "PNG",
    "pnm": "PNM",
    "rda": "R",
    "rgb": "SGI",
    "rst": "RST",
    "rsw": "RMF",
    "sdat": "SAGA",
    "sqlite": "Rasterlite",
    "ter": "Terragen",
    "tif": "GTiff",
    "vrt": "VRT",
    "xpm": "XPM",
    "xyz": "XYZ",
}
GDAL_EXT_CODE_MAP = {v: k for k, v in GDAL_DRIVER_CODE_MAP.items()}


## GEOM functions


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

    Output is optionally bounded by a maximumum distance `max_dist`.
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
    idx_nn, dst = nearest(gdf1, gdf2)
    if not inplace:
        gdf1 = gdf1.copy()
    valid = dst < max_dist if max_dist is not None else np.ones_like(idx_nn, dtype=bool)
    columns = gdf2.columns if columns is None else columns
    gdf1["distance_right"] = dst
    gdf1["index_right"] = -1
    gdf1.loc[valid, "index_right"] = idx_nn[valid]
    skip = ["geometry"]
    for col in columns:
        if col in skip or col not in gdf2:
            if col not in gdf2:
                logger.warning(f"Column {col} not found in gdf2 and skipped.")
            continue
        new_vals = gdf2.loc[idx_nn[valid], col].values
        if col in gdf1 and not overwrite:
            old_vals = gdf1.loc[valid, col]
            replace = np.logical_or(old_vals.isnull(), old_vals.eq(""))
            new_vals = np.where(replace, new_vals, old_vals)
        gdf1.loc[valid, col] = new_vals
    return gdf1


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
    # NOTE: requires shapely v2.0; changed in v0.6.1
    if not _compat.HAS_SHAPELY20:
        raise ImportError("Shapely >= 2.0.0 is required for execution")
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


def parse_crs(crs, bbox=None):
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


def meridian_offset(ds, bbox=None):
    """Shift data along the x-axis of global datasets to avoid issues along the 180 meridian.

    Without a bbox the data is shifted to span 180W to 180E.
    With bbox the data is shifted to at least span the bbox west to east,
    also if the bbox crosses the 180 meridian.

    Note that this method is only applicable to data that spans 360 degrees longitude
    and is set in a global geographic CRS (WGS84).

    Parameters
    ----------
    ds: xarray.Dataset
        input dataset
    bbox: tuple of float
        bounding box (west, south, east, north) in degrees

    Returns
    -------
    ds: xarray.Dataset
        dataset with x dim re-arranged if needed
    """
    w, _, e, _ = ds.raster.bounds
    if (
        ds.raster.crs is None
        or ds.raster.crs.is_projected
        or not np.isclose(e - w, 360)  # grid should span 360 degrees!
    ):
        raise ValueError(
            "This method is only applicable to data that spans 360 degrees "
            "longitude and is set in a global geographic CRS"
        )
    x_name = ds.raster.x_dim
    lons = np.copy(ds[x_name].values)
    if bbox is not None:  # bbox west and east
        bbox_w, bbox_e = bbox[0], bbox[2]
    else:  # global west and east in case of no bbox
        bbox_w, bbox_e = -180, 180
    if bbox_w < w:  # shift lons east of x0 by 360 degrees west
        x0 = 180 if bbox_w >= -180 else 0
        lons = np.where(lons > max(bbox_e, x0), lons - 360, lons)
    elif bbox_e > e:  # shift lons west of x0 by 360 degrees east
        x0 = -180 if bbox_e <= 180 else 0
        lons = np.where(lons < min(bbox_w, x0), lons + 360, lons)
    else:
        return ds
    ds = ds.copy(deep=False)  # make sure not to overwrite original ds
    ds[x_name] = xr.Variable(ds[x_name].dims, lons)
    return ds.sortby(x_name)


# TRANSFORM


def affine_to_coords(transform, shape, x_dim="x", y_dim="y"):
    """Return a raster axis with pixel center coordinates based on the transform.

    Parameters
    ----------
    transform : affine transform
        Two dimensional affine transform for 2D linear mapping
    shape : tuple of int
        The height, width  of the raster.
    x_dim, y_dim: str
        The name of the x and y dimensions

    Returns
    -------
    x, y coordinate arrays : dict of tuple with dims and coords
    """
    if not isinstance(transform, Affine):
        transform = Affine(*transform)
    height, width = shape
    if np.isclose(transform.b, 0) and np.isclose(transform.d, 0):
        x_coords, _ = transform * (np.arange(width) + 0.5, np.zeros(width) + 0.5)
        _, y_coords = transform * (np.zeros(height) + 0.5, np.arange(height) + 0.5)
        coords = {
            y_dim: (y_dim, y_coords),
            x_dim: (x_dim, x_coords),
        }
    else:
        x_coords, y_coords = (
            transform
            * transform.translation(0.5, 0.5)
            * np.meshgrid(np.arange(width), np.arange(height))
        )
        coords = {
            "yc": ((y_dim, x_dim), y_coords),
            "xc": ((y_dim, x_dim), x_coords),
        }
    return coords


def affine_to_meshgrid(transform, shape):
    """Return a meshgrid of pixel center coordinates based on the transform.

    Parameters
    ----------
    transform : affine transform
        Two dimensional affine transform for 2D linear mapping
    shape : tuple of int
        The height, width  of the raster.

    Returns
    -------
    x_coords, y_coords: ndarray
        2D arrays of x and y coordinates
    """
    if not isinstance(transform, Affine):
        transform = Affine(*transform)
    height, width = shape
    x_coords, y_coords = (
        transform
        * transform.translation(0.5, 0.5)
        * np.meshgrid(np.arange(width), np.arange(height))
    )
    return x_coords, y_coords


## CELLAREAS
def reggrid_area(lats, lons):
    """Return the cell area [m2] for a regular grid based on its cell centres lat, lon."""  # noqa: E501
    xres = np.abs(np.mean(np.diff(lons)))
    yres = np.abs(np.mean(np.diff(lats)))
    area = np.ones((lats.size, lons.size), dtype=lats.dtype)
    return cellarea(lats, xres, yres)[:, None] * area


def cellarea(lat, xres=1.0, yres=1.0):
    """Return the area [m2] of cell based on its center latitude and resolution in degrees.

    Resolution is in measured degrees.
    """  # noqa: E501
    l1 = np.radians(lat - np.abs(yres) / 2.0)
    l2 = np.radians(lat + np.abs(yres) / 2.0)
    dx = np.radians(np.abs(xres))
    return _R**2 * dx * (np.sin(l2) - np.sin(l1))


def cellres(lat, xres=1.0, yres=1.0):
    """Return the cell (x, y) resolution [m].

    Based on cell center latitude and its resolution measured in degrees.
    """
    m1 = 111132.92  # latitude calculation term 1
    m2 = -559.82  # latitude calculation term 2
    m3 = 1.175  # latitude calculation term 3
    m4 = -0.0023  # latitude calculation term 4
    p1 = 111412.84  # longitude calculation term 1
    p2 = -93.5  # longitude calculation term 2
    p3 = 0.118  # longitude calculation term 3

    radlat = np.radians(lat)  # numpy cos work in radians!
    # Calculate the length of a degree of latitude and longitude in meters
    dy = (
        m1
        + (m2 * np.cos(2.0 * radlat))
        + (m3 * np.cos(4.0 * radlat))
        + (m4 * np.cos(6.0 * radlat))
    )
    dx = (
        (p1 * np.cos(radlat))
        + (p2 * np.cos(3.0 * radlat))
        + (p3 * np.cos(5.0 * radlat))
    )

    return dx * xres, dy * yres


## SPREAD


def spread2d(
    da_obs: xr.DataArray,
    da_mask: Optional[xr.DataArray] = None,
    da_friction: Optional[xr.DataArray] = None,
    nodata: Optional[float] = None,
) -> xr.Dataset:
    """Return values of `da_obs` spreaded to cells with `nodata` value within `da_mask`.

    powered by :py:meth:`pyflwdir.gis_utils.spread2d`.

    Parameters
    ----------
    da_obs : xarray.DataArray
        Input raster with observation values and background/nodata values which are
        filled by the spreading algorithm.
    da_mask :  xarray.DataArray, optional
        Mask of cells to fill with the spreading algorithm, by default None
    da_friction :  xarray.DataArray, optional
        Friction values used by the spreading algorithm to calcuate the friction
        distance, by default None
    nodata : float, optional
        Nodata or background value. Must be finite numeric value. If not given the
        raster nodata value is used.

    Returns
    -------
    ds_out: xarray.Dataset
        Dataset with spreaded source values, linear index of the source cell
        "source_idx" and friction distance to the source cell "source_dst".
    """
    nodata = da_obs.raster.nodata if nodata is None else nodata
    if nodata is None or np.isnan(nodata):
        raise ValueError(f'"nodata" must be a finite value, not {nodata}')
    msk, frc = None, None
    if da_mask is not None:
        assert da_obs.raster.identical_grid(da_mask)
        msk = da_mask.values
    if da_friction is not None:
        assert da_obs.raster.identical_grid(da_friction)
        frc = da_friction.values
    out, src, dst = gis.spread2d(
        obs=da_obs.values,
        msk=msk,
        frc=frc,
        nodata=nodata,
        latlon=da_obs.raster.crs.is_geographic,
        transform=da_obs.raster.transform,
    )
    # combine outputs and return as dataset
    dims = da_obs.raster.dims
    coords = da_obs.raster.coords
    name = da_obs.name if da_obs.name else "source_value"
    da_out = xr.DataArray(dims=dims, coords=coords, data=out, name=name)
    da_out.raster.attrs.update(**da_obs.attrs)  # keep attrs incl nodata and unit
    da_src = xr.DataArray(dims=dims, coords=coords, data=src, name="source_idx")
    da_src.raster.set_nodata(-1)
    da_dst = xr.DataArray(dims=dims, coords=coords, data=dst, name="source_dst")
    da_dst.raster.set_nodata(-1)
    da_dst.attrs.update(unit="m")
    ds_out = xr.merge([da_out, da_src, da_dst])
    ds_out.raster.set_crs(da_obs.raster.crs)
    return ds_out


def create_vrt(
    vrt_path: str,
    files: list = None,
    files_path: str = None,
):
    r"""Create a .vrt file from a list op raster datasets.

    Either a list of files (`files`) or a path containing wildcards
    (`files_path`) to infer the list of files is required.

    Parameters
    ----------
    vrt_path : str
        Path of the output vrt
    files : list, optional
        List of raster datasets filenames, by default None
    files_path : str, optional
        Unix style path containing a pattern using wildcards (*)
        n.b. this is without an extension
        e.g. c:\\temp\\*\\*.tif for all tif files in subfolders of 'c:\temp'
    """
    if files is None and files_path is None:
        raise ValueError("Either 'files' or 'files_path' is required")

    if not _compat.HAS_RIO_VRT:
        raise ImportError(
            "rio-vrt is required for execution, install with 'pip install rio-vrt'"
        )
    import rio_vrt

    if files is None and files_path is not None:
        files = glob.glob(files_path)
        if len(files) == 0:
            raise IOError(f"No files found at {files_path}")

    outdir = dirname(vrt_path)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    rio_vrt.build_vrt(vrt_path, files=files, relative=True)
    return None


def to_geographic_bbox(bbox, source_crs):
    target_crs = CRS.from_user_input(4326)
    if source_crs is None:
        logger.warning("No CRS was set. Skipping CRS conversion")
    elif source_crs != target_crs:
        bbox = Transformer.from_crs(source_crs, target_crs).transform_bounds(*bbox)

    return bbox


def bbox_from_file_and_filters(
    fn: str,
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
    fn: IOBase,
        opened file.
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
    if source_crs_str := read_info(fn).get("crs"):
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
