#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""gis related convience functions. More in pyflwdir.gis_utils"""
from os.path import dirname, join, isfile
import os
import glob
import sys
import subprocess
import numpy as np
import xarray as xr
import rasterio
from pyproj import CRS
from rasterio.transform import Affine
import geopandas as gpd
from shapely.geometry.base import BaseGeometry
from shapely.geometry import box
import logging
from pyflwdir import core_conversion, core_d8, core_ldd
from pyflwdir import gis_utils as gis
from typing import Optional, Tuple
from . import _compat


__all__ = ["spread2d", "nearest", "nearest_merge"]

logger = logging.getLogger(__name__)

_R = 6371e3  # Radius of earth in m. Use 3956e3 for miles
XATTRS = {
    "geographic": {
        "standard_name": "longitude",
        "long_name": "longitude coordinate",
        "short_name": "lon",
        "units": "degrees_east",
    },
    "projected": {
        "standard_name": "projection_x_coordinate",
        "long_name": "x coordinate of projection",
        "short_name": "x",
        "units": "m",
    },
}
YATTRS = {
    "geographic": {
        "standard_name": "latitude",
        "long_name": "latitude coordinate",
        "short_name": "lat",
        "units": "degrees_north",
    },
    "projected": {
        "standard_name": "projection_y_coordinate",
        "long_name": "y coordinate of projection",
        "short_name": "y",
        "units": "m",
    },
}
PCR_VS_MAP = {"ldd": "ldd"}
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
    "map": "PCRaster",
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
    """Merge attributes of gdf2 with the nearest feature of gdf1, optionally bounded by
    a maximumum distance `max_dist`. Unless `overwrite = True`, gdf2 values are only
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
    """Return the index of and distance [m] to the nearest geometry
    in `gdf2` for each geometry of `gdf1`. For Line geometries in `gdf1` the nearest
    geometry is based line center point and for polygons on its representative point.
    Mixed geometry types are not yet supported.

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
        if gdf.crs is not None and geom.crs != gdf.crs:
            geom = geom.to_crs(gdf.crs)
        # convert geopandas to geometry
        geom = geom.unary_union
    idx = gdf.sindex.query(geom, predicate=predicate)
    return idx


# REPROJ
def utm_crs(bbox):
    """Returns wkt string of nearest UTM projects

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
    """
    Provide CF-compliant variable names and metadata for axes

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
    # check for type of crs
    crs_type = "geographic" if crs.is_geographic else "projected"
    y_dim = YATTRS[crs_type]["short_name"]
    x_dim = XATTRS[crs_type]["short_name"]
    y_attrs = YATTRS[crs_type]
    x_attrs = XATTRS[crs_type]
    return x_dim, y_dim, x_attrs, y_attrs


def meridian_offset(ds, x_name="x", bbox=None):
    """re-arange data along x dim"""
    if ds.raster.crs is None or ds.raster.crs.is_projected:
        raise ValueError("The method is only applicable to geographic CRS")
    lons = np.copy(ds[x_name].values)
    w, e = lons.min(), lons.max()
    if bbox is not None and bbox[0] < w and bbox[0] < -180:  # 180W - 180E > 360W - 0W
        lons = np.where(lons > 0, lons - 360, lons)
    elif bbox is not None and bbox[2] > e and bbox[2] > 180:  # 180W - 180E > 0E-360E
        lons = np.where(lons < 0, lons + 360, lons)
    elif e > 180:  # 0E-360E > 180W - 180E
        lons = np.where(lons > 180, lons - 360, lons)
    else:
        return ds
    ds = ds.copy(deep=False)  # make sure not to overwrite original ds
    ds[x_name] = xr.Variable(ds[x_name].dims, lons)
    return ds.sortby(x_name)


# TRANSFORM


def affine_to_coords(transform, shape, x_dim="x", y_dim="y"):
    """Returns a raster axis with pixel center coordinates based on the transform.

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
    if transform.b == 0:
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
    """Returns a mesgrid of pixel center coordinates based on the transform.

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
    """Returns the cell area [m2] for a regular grid based on its cell centres
    lat, lon coordinates."""
    xres = np.abs(np.mean(np.diff(lons)))
    yres = np.abs(np.mean(np.diff(lats)))
    area = np.ones((lats.size, lons.size), dtype=lats.dtype)
    return cellarea(lats, xres, yres)[:, None] * area


def cellarea(lat, xres=1.0, yres=1.0):
    """Return the area [m2] of cell based on the cell center latitude and its resolution
    in measured in degrees."""
    l1 = np.radians(lat - np.abs(yres) / 2.0)
    l2 = np.radians(lat + np.abs(yres) / 2.0)
    dx = np.radians(np.abs(xres))
    return _R**2 * dx * (np.sin(l2) - np.sin(l1))


def cellres(lat, xres=1.0, yres=1.0):
    """Return the cell (x, y) resolution [m] based on cell center latitude and its
    resolution measured in degrees."""
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
    """Returns values of `da_obs` spreaded to cells with `nodata` value within `da_mask`,
    powered by :py:meth:`pyflwdir.gis_utils.spread2d`

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
        Dataset with spreaded source values, linear index of the source cell "source_idx"
        and friction distance to the source cell "source_dst".
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
        nodata=da_obs.raster.nodata if nodata is None else nodata,
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


## PCRASTER


def write_clone(tmpdir, gdal_transform, wkt_projection, shape):
    """write pcraster clone file to a tmpdir using gdal"""
    from osgeo import gdal

    gdal.AllRegister()
    driver1 = gdal.GetDriverByName("GTiff")
    driver2 = gdal.GetDriverByName("PCRaster")
    fn = join(tmpdir, "clone.map")
    # create temp tif file
    fn_temp = join(tmpdir, "clone.tif")
    TempDataset = driver1.Create(fn_temp, shape[1], shape[0], 1, gdal.GDT_Float32)
    TempDataset.SetGeoTransform(gdal_transform)
    if wkt_projection is not None:
        TempDataset.SetProjection(wkt_projection)
    # TODO set csr
    # copy to pcraster format
    outDataset = driver2.CreateCopy(fn, TempDataset, 0)
    # close and cleanup
    TempDataset = None
    outDataset = None
    return fn


def write_map(
    data,
    raster_path,
    nodata,
    transform,
    crs=None,
    clone_path=None,
    pcr_vs="scalar",
    **kwargs,
):
    """Write pcraster map files using pcr.report functionality.

    A PCRaster clone map is written to a temporary directory if not provided.
    For PCRaster types see https://www.gdal.org/frmt_various.html#PCRaster

    Parameters
    ----------
    data : ndarray
        Raster data
    raster_path : str
        Path to output map
    nodata : int, float
        no data value
    transform : affine transform
        Two dimensional affine transform for 2D linear mapping
    clone_path : str, optional
        Path to PCRaster clone map, by default None
    pcr_vs : str, optional
        pcraster type, by default "scalar"

    Raises
    ------
    ImportError
        pcraster package is required
    ValueError
        if invalid ldd
    """
    if not _compat.HAS_PCRASTER:
        raise ImportError("The pcraster package is required to write map files")
    import tempfile
    import pcraster as pcr

    with tempfile.TemporaryDirectory() as tmpdir:
        # deal with pcr clone map
        if clone_path is None:
            clone_path = write_clone(
                tmpdir,
                gdal_transform=transform.to_gdal(),
                wkt_projection=None if crs is None else CRS.from_user_input(crs).wkt,
                shape=data.shape,
            )
        elif not isfile(clone_path):
            raise IOError(f'clone_path: "{clone_path}" does not exist')
        pcr.setclone(clone_path)
        if nodata is None and pcr_vs != "ldd":
            raise ValueError("nodata value required to write PCR map")
        # write to pcrmap
        if pcr_vs == "ldd":
            # if d8 convert to ldd
            data = data.astype(np.uint8)  # force dtype
            if core_d8.isvalid(data):
                data = core_conversion.d8_to_ldd(data)
            elif not core_ldd.isvalid(data):
                raise ValueError("LDD data not understood")
            mv = int(core_ldd._mv)
            ldd = pcr.numpy2pcr(pcr.Ldd, data.astype(int), mv)
            # make sure it is pcr sound
            # NOTE this should not be necessary
            pcrmap = pcr.lddrepair(ldd)
        elif pcr_vs == "bool":
            pcrmap = pcr.numpy2pcr(pcr.Boolean, data.astype(np.bool), np.bool(nodata))
        elif pcr_vs == "scalar":
            pcrmap = pcr.numpy2pcr(pcr.Scalar, data.astype(float), float(nodata))
        elif pcr_vs == "ordinal":
            pcrmap = pcr.numpy2pcr(pcr.Ordinal, data.astype(int), int(nodata))
        elif pcr_vs == "nominal":
            pcrmap = pcr.numpy2pcr(pcr.Nominal, data.astype(int), int(nodata))
        pcr.report(pcrmap, raster_path)
        # set crs (pcrmap ignores this info from clone ??)
        if crs is not None:
            with rasterio.open(raster_path, "r+") as dst:
                dst.crs = crs


def create_vrt(
    fname: str,
    file_list_path: str = None,
    files_path: str = None,
    ext: list = [".tif"],
    output: str = None,
    **kwargs,
):
    """Creates a .vrt file from a list op raster datasets by either
    passing the list directly (file_list_path) or by inferring it by passing
    a path containing wildcards (files_path) of the location(s) of the
    raster datasets

    Parameters
    ----------
    fname : str
        Name of the output vrt
    file_list_path : str, optional
        Path to the text file containing the paths to the raster files
    files_path : str, optional
        Unix style path containing a pattern using wildcards (*)
        n.b. this is without an extension
        e.g. c:\\temp\\*\\* for all files in subfolders of 'c:\temp'
    ext : list, optional
        List of extensions to be sought after
    output : str, optional
        Output directory
        if not given a directory will be inferred from either 'file_list_path' of 'files_path'
    kwargs : optional
        Extra keyword arguments for glob (combined with files_path)
        e.g. recursive=True

    Raises
    ------
    ValueError
        A Path is needed, either file_list_path or files_path
    """

    if file_list_path is None and files_path is None:
        raise ValueError(
            "Either 'file_list_path' or 'files_path' is required -> None was given"
        )

    def create_folder(path):
        if not os.path.exists(path):
            os.makedirs(path)

    if file_list_path is None:
        files = []
        for e in ext:
            files += glob.glob(f"{files_path}{e}", **kwargs)
        if output is None:
            output = files_path.split("*")[0]
        create_folder(output)
        file_list_path = join(output, "filelist.txt")
        with open(file_list_path, "w") as w:
            for line in files:
                w.write(f"{line}\n")

    if output is None:
        output = dirname(file_list_path)

    create_folder(output)

    subprocess.run(
        [
            "gdalbuildvrt",
            "-input_file_list",
            file_list_path,
            join(output, f"{fname}.vrt"),
        ]
    )
    return None
