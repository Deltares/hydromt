#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Credits: This script is based on the rioxarray package (Apache License, Version 2.0)
# source file: https://github.com/corteva/rioxarray
# license file: https://github.com/corteva/rioxarray/blob/master/LICENSE

"""
This module is an extension for xarray to provide rasterio capabilities
to xarray datasets/dataarrays.
"""

import os
from os.path import join, basename, dirname, isdir
import numpy as np
from shapely.geometry import box
import geopandas as gpd
import xarray as xr
import dask
from affine import Affine
from rasterio.crs import CRS
import rasterio.warp
import rasterio.fill
from rasterio import features
from rasterio.enums import Resampling
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from scipy import ndimage
import tempfile
import pyproj
import logging
import rioxarray

from . import gis_utils, HAS_PCRASTER

logger = logging.getLogger(__name__)
XDIMS = ("x", "longitude", "lon", "long")
YDIMS = ("y", "latitude", "lat")
GEO_MAP_COORD = "spatial_ref"


def full_like(other, nodata=None, lazy=False):
    """Return a full object with the same grid and geospatial attributes as ``other``.

    Arguments
    ---------
    other: DataArray
        DataArray from which coordinates and attributes are taken
    nodata: float, int, optional
        Fill value for new DataArray, defaults to other.nodata or if not set np.nan
    lazy: bool, optional
        If True return DataArray with a dask rather than numpy array.

    Returns
    -------
    da: DataArray
        Filled DataArray
    """
    if not isinstance(other, xr.DataArray):
        raise ValueError("other should be xarray.DataArray.")
    if nodata is None:
        nodata = other.raster.nodata if other.raster.nodata is not None else np.nan
    da = full(
        coords={d: c for d, c in other.coords.items() if d in other.dims},
        nodata=nodata,
        dtype=other.dtype,
        name=other.name,
        attrs=other.attrs,
        crs=other.raster.crs,
        lazy=lazy,
    )
    da.raster.set_attrs(**other.raster.attrs)
    return da


def full(
    coords, nodata=np.nan, dtype=np.float32, name=None, attrs={}, crs=None, lazy=False
):
    """Return a full DataArray based on a geospatial coords dictionary.

    Arguments
    ---------
    coords: sequence or dict of array_like, optional
        Coordinates (tick labels) to use for indexing along each dimension (max 3).
        The coordinate sequence should be (dim0, y, x) of which the first is optional.
    nodata: float, int, optional
        Fill value for new DataArray, defaults to other.nodata or if not set np.nan
    dtype: numpy.dtype, optional
        Data type
    name: str, optional
        DataArray name
    attrs : dict, optional
        additional attributes
    crs: int, dict, or str, optional
        Coordinate Reference System. Accepts EPSG codes (int or str); proj (str or dict)
    lazy: bool, optional
        If True return DataArray with a dask rather than numpy array.

    Returns
    -------
    da: DataArray
        Filled DataArray
    """
    f = dask.array.empty if lazy else np.full
    dims = tuple([d for d in coords])
    shape = tuple([coords[dim].size for dim in dims])
    data = f(shape, nodata, dtype=dtype)
    da = xr.DataArray(data, coords, dims, name, attrs)
    da.raster.set_nodata(nodata)
    da.raster.set_crs(crs)
    return da


def full_from_transform(
    transform,
    shape,
    nodata=np.nan,
    dtype=np.float32,
    name=None,
    attrs={},
    crs=None,
    lazy=False,
):
    """Return a full DataArray based on a geospatial transform and shape.
    See :py:meth:`~hydromt.raster.full` for all options.

    Arguments
    ---------
    transform : affine transform
        Two dimensional affine transform for 2D linear mapping
    shape: tuple of int
        Length along (dim0, x, y) dimensions, of which the first is optional.

    Returns
    -------
    da: DataArray
        Filled DataArray
    """
    if len(shape) not in [2, 3]:
        raise ValueError("Only 2D and 3D data arrays supported.")
    _xcoords, _ycoords = gis_utils.affine_to_coords(transform, shape[-2:])
    coords = {"y": _ycoords, "x": _xcoords}
    if len(shape) == 3:
        coords = {"dim0": np.arange(shape[0], dtype=int), **coords}
    da = full(
        coords=coords,
        nodata=nodata,
        dtype=dtype,
        name=name,
        attrs=attrs,
        crs=crs,
        lazy=lazy,
    )
    return da


class XGeoBase(object):
    """This is the base class for the GIS extensions for xarray"""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        # create new coordinate with attributes in which to save x_dim, y_dim and crs.
        # other spatial properties are always calculated on the fly to ensure consistency with data
        if GEO_MAP_COORD not in self._obj.coords:
            self._obj.coords[GEO_MAP_COORD] = xr.Variable((), 1)

    @property
    def attrs(self):
        """Return dictionary of spatial attributes"""
        return self._obj.coords[GEO_MAP_COORD].attrs

    def set_attrs(self, **kwargs):
        """Update spatial attributes. Usage raster.set_attr(key=value)."""
        self._obj.coords[GEO_MAP_COORD].attrs.update(**kwargs)

    def get_attrs(self, key, placeholder=None):
        """Return single spatial attribute."""
        return self._obj.coords[GEO_MAP_COORD].attrs.get(key, placeholder)

    @property
    def crs(self):
        """Return Coordinate Reference System as :py:meth:`pyproj.CRS` object."""
        return self._obj.rio.crs

    @property
    def x_dim(self):
        """Return the x dimension name"""
        if self.get_attrs("x_dim") not in self._obj.dims:
            self.set_spatial_dims()
        return self.attrs["x_dim"]

    @property
    def y_dim(self):
        """Return the y dimension name"""
        if self.get_attrs("y_dim") not in self._obj.dims:
            self.set_spatial_dims()
        return self.attrs["y_dim"]

    @property
    def xcoords(self):
        """Return the x coordinates"""
        return self._obj[self.x_dim]

    @property
    def ycoords(self):
        """Return the y coordinates"""
        return self._obj[self.y_dim]

    def set_spatial_dims(self, x_dim=None, y_dim=None):
        """Set the geospatial dimensions of the object.

        Arguments
        ----------
        x_dim: str, optional
            The name of the x dimension.
        y_dim: str, optional
            The name of the y dimension.
        """
        _dims = self._obj.dims
        if x_dim is None:
            for dim in XDIMS:
                if dim in _dims:
                    x_dim = dim
                    break
        if x_dim and x_dim in _dims:
            self.set_attrs(x_dim=x_dim)
        else:
            raise ValueError(
                "x dimension not found. Use 'set_spatial_dims'"
                + " functions with correct x_dim argument provided."
            )

        if y_dim is None:
            for dim in YDIMS:
                if dim in _dims:
                    y_dim = dim
                    break
        if y_dim and y_dim in _dims:
            self.set_attrs(y_dim=y_dim)
        else:
            raise ValueError(
                "y dimension not found. Use 'set_spatial_dims'"
                + " functions with correct y_dim argument provided."
            )

        check_x = np.all(np.isclose(np.diff(np.diff(self._obj[x_dim])), 0, atol=1e-4))
        check_y = np.all(np.isclose(np.diff(np.diff(self._obj[y_dim])), 0, atol=1e-4))
        if check_x == False or check_y == False:
            raise ValueError("raster only applies to regular grids")

    def set_crs(self, input_crs=None):
        """Set the Coordinate Reference System.

        Arguments
        ----------
        input_crs: int, dict, or str, optional
            Coordinate Reference System. Accepts EPSG codes (int or str); proj (str or dict)
        """
        crs_names = ["crs_wkt", "crs", "epsg"]
        names = list(self._obj.coords.keys())
        if isinstance(self._obj, xr.Dataset):
            names = names + self.vars
        # user defined
        if input_crs is not None:
            input_crs = CRS.from_user_input(input_crs)
        # look in grid_mapping and data variable attributes
        else:
            for name in crs_names:
                # check default > GEO_MAP_COORDS attrs
                crs = self._obj.coords[GEO_MAP_COORD].attrs.get(name, None)
                if crs is None:  # global attrs
                    crs = self._obj.attrs.pop(name, None)
                for var in names:  # data var and coords attrs
                    if name in self._obj[var].attrs:
                        crs = self._obj[var].attrs.pop(name)
                        break
                if crs is not None:
                    # avoid Warning 1: +init=epsg:XXXX syntax is deprecated
                    crs = crs.strip("+init=") if isinstance(crs, str) else crs
                    try:
                        input_crs = CRS.from_user_input(crs)
                        break
                    except:
                        pass
        if input_crs is not None:
            self._obj.rio.write_crs(input_crs, inplace=True)

    def reset_spatial_dims_attrs(self):
        """Reset spatial dimension names and attributes to make CF-compliant
        Requires CRS attribute."""
        if self.crs is None:
            raise ValueError("CRS is missing. Use set_crs function to resolve.")
        _da = self._obj
        x_dim, y_dim, x_attrs, y_attrs = gis_utils.axes_attrs(self.crs)
        _da = _da.rename({self.x_dim: x_dim, self.y_dim: y_dim})
        _da[x_dim].attrs.update(x_attrs)
        _da[y_dim].attrs.update(y_attrs)
        _da.raster.set_spatial_dims(x_dim=x_dim, y_dim=y_dim)
        return _da


class XRasterBase(XGeoBase):
    """This is the base class for a Raster GIS extensions for xarray"""

    def __init__(self, xarray_obj):
        super(XRasterBase, self).__init__(xarray_obj)

    @property
    def dim0(self):
        """Return the non geospatial dimension name."""
        if self.get_attrs("dim0") not in self._obj.dims:
            self._check_dimensions()
        return self.get_attrs("dim0")

    @property
    def dims(self):
        """Return tuple of geospatial dimensions names."""
        # if self.dim0 is not None:
        #     return self.dim0, self.y_dim, self.x_dim
        # else:
        return self.y_dim, self.x_dim

    @property
    def coords(self):
        """Return dict of geospatial dimensions coordinates."""
        return {dim: self._obj.coords[dim] for dim in list(self.dims)}

    @property
    def shape(self):
        """Return shape of geospatial dimension (height, width)."""
        # return tuple([self._obj.coords[d].size for d in list(self.dims)])
        return self.height, self.width

    @property
    def size(self):
        """Return size of geospatial grid."""
        return np.multiply(*self.shape)

    @property
    def width(self):
        """Return the width of the object (x dimension size)."""
        return self.xcoords.size

    @property
    def height(self):
        """Return the height of the object (y dimension size)."""
        return self.ycoords.size

    @property
    def transform(self):
        """Return the affine transform of the object."""
        transform = rasterio.transform.from_bounds(
            *self.internal_bounds, self.width, self.height
        )
        return transform

    @property
    def internal_bounds(self):
        """Return the internal bounds of the object."""
        res_x, res_y = self.res
        left = float(self.xcoords[0]) - res_x / 2.0
        right = float(self.xcoords[-1]) + res_x / 2.0
        top = float(self.ycoords[0]) - res_y / 2.0
        bottom = float(self.ycoords[-1]) + res_y / 2.0
        return left, bottom, right, top

    @property
    def bounds(self):
        """Return the bounds (xmin, ymin, xmax, ymax) of the object."""
        left, bottom, right, top = self.internal_bounds
        xmin, xmax = min(left, right), max(left, right)
        ymin, ymax = min(top, bottom), max(top, bottom)
        return xmin, ymin, xmax, ymax

    @property
    def box(self):
        """Return :py:meth:`~geopandas.GeoDataFrame` of bounding box"""
        crs = self.crs
        if crs is None and crs.to_epsg() is not None:
            crs = crs.to_epsg()  # not all CRS have an EPSG code
        return gpd.GeoDataFrame(geometry=[box(*self.bounds)], crs=crs)

    @property
    def res(self):
        """Return resolution (x, y) tuple."""
        left, right = float(self.xcoords[0]), float(self.xcoords[-1])
        top, bottom = float(self.ycoords[0]), float(self.ycoords[-1])
        res_x = (right - left) / max(self.width - 1, 1)
        res_y = (bottom - top) / max(self.height - 1, 1)
        return res_x, res_y

    def _check_dimensions(self):
        """Validates the dimensions number of dimensions and dimension order."""
        dims = (self.y_dim, self.x_dim)
        da = self._obj[self.vars[0]] if isinstance(self._obj, xr.Dataset) else self._obj
        extra_dims = [dim for dim in da.dims if dim not in dims]
        if len(extra_dims) == 1:
            dims = tuple(extra_dims) + dims
            self.set_attrs(dim0=extra_dims[0])
        elif len(extra_dims) == 0:
            self.set_attrs(dim0=None)
        else:
            raise ValueError("Only 2D and 3D data arrays supported.")
        if isinstance(self._obj, xr.Dataset):
            check = np.all([self._obj[name].dims == dims for name in self.vars])
        else:
            check = self._obj.dims == dims
        if check == False:
            raise ValueError(
                f"Invalid dimension order ({da.dims}). "
                f"You can use `obj.transpose({dims}) to reorder your dimensions."
            )

    def identical_grid(self, other):
        """Return True if other has an same grid as object (crs, transform, shape)."""
        return (
            (
                self.crs is None
                or other.raster.crs is None
                or self.crs == other.raster.crs
            )
            and np.allclose(self.transform, other.raster.transform, atol=1e-06)
            and np.allclose(self.shape, other.raster.shape)
        )

    def aligned_grid(self, other):
        """Return True if other grid aligns with object grid (crs, resolution, origin),
        but with a smaller extent"""
        w, s, e, n = self.bounds
        w1, s1, e1, n1 = other.raster.bounds
        dx = (w - w1) % self.res[0]
        dy = (n - n1) % self.res[1]
        return (
            (
                self.crs is None
                or other.raster.crs is None
                or self.crs == other.raster.crs
            )
            and np.allclose(self.res, other.raster.res)
            and (np.isclose(dx, 0) or np.isclose(dx, 1))
            and (np.isclose(dy, 0) or np.isclose(dy, 1))
            and np.logical_and.reduce((w <= w1, s <= s1, e >= e1, n >= n1))
        )

    def gdal_compliant(self, rename_dims=True, force_sn=False):
        """Updates attributes to get GDAL compliant NetCDF files.

        Arguments
        ----------
        rename_dims: bool, optional
            If True, rename x_dim and y_dim to standard names depending on the CRS
            (x/y for projected and lat/lon for geographic).
        force_sn: bool, optional
            If True, forces the dataset to have South -> North orientation.

        Returns
        -------
        ojb_out: xr.Dataset or xr.DataArray
            GDAL compliant object
        """
        obj_out = self._obj
        crs = obj_out.raster.crs
        if (
            obj_out.raster.res[1] < 0 and force_sn
        ):  # write data with South -> North orientation
            obj_out = obj_out.raster.flipud()
        x_dim, y_dim, x_attrs, y_attrs = gis_utils.axes_attrs(crs)
        if rename_dims:
            obj_out = obj_out.rename(
                {obj_out.raster.x_dim: x_dim, obj_out.raster.y_dim: y_dim}
            )
        else:
            x_dim = obj_out.raster.x_dim
            y_dim = obj_out.raster.y_dim
        obj_out[x_dim].attrs.update(x_attrs)
        obj_out[y_dim].attrs.update(y_attrs)
        obj_out = obj_out.drop_vars(["spatial_ref"], errors="ignore")
        obj_out.rio.write_crs(crs, inplace=True)
        obj_out.rio.write_transform(obj_out.raster.transform, inplace=True)
        obj_out.raster.set_spatial_dims()

        return obj_out

    def transform_bounds(self, dst_crs, densify_pts=21):
        """Transform bounds from object to destination CRS.

        Optionally densifying the edges (to account for nonlinear transformations
        along these edges) and extracting the outermost bounds.

        Note: this does not account for the antimeridian.

        Arguments
        ----------
        dst_crs: CRS, str, int, or dict
            Target coordinate reference system, input to
            :py:meth:`rasterio.crs.CRS.from_user_input`
        densify_pts: uint, optional
            Number of points to add to each edge to account for nonlinear
            edges produced by the transform process.  Large numbers will produce
            worse performance.  Default: 21 (gdal default).

        Returns
        -------
        bounds: list of float
            Outermost coordinates in target coordinate reference system.
        """
        if self.crs != dst_crs:
            bounds = rasterio.warp.transform_bounds(
                self.crs, dst_crs, *self.bounds, densify_pts=densify_pts
            )
        else:
            bounds = self.bounds
        return bounds

    def flipud(self):
        """Returns raster flipped along y-axis"""
        yrev = list(reversed(self.ycoords))
        return self._obj.reindex({self.y_dim: yrev})

    def rowcol(self, xs, ys, mask=None, mask_outside=False, nodata=-1):
        """Return row, col indices of x, y coordinates

        Arguments
        ----------
        xs, ys: ndarray of float
            x, y coordinates
        mask : ndarray of bool, optional
            data mask of valid values, by default None
        mask_outside : boolean, optional
            mask xy points outside domain (i.e. set nodata), by default False
        nodata : int, optional
            nodata value, used for output array, by default -1

        Returns
        -------
        ndarray of int
            linear indices
        """
        nrow, ncol = self.shape
        xs = np.atleast_1d(xs)
        ys = np.atleast_1d(ys)
        if mask is None:
            mask = np.logical_and(np.isfinite(xs), np.isfinite(ys))
        r = np.full(mask.shape, nodata, dtype=int)
        c = np.full(mask.shape, nodata, dtype=int)
        r[mask], c[mask] = rasterio.transform.rowcol(self.transform, xs[mask], ys[mask])
        points_inside = np.logical_and.reduce((r >= 0, r < nrow, c >= 0, c < ncol))
        if mask_outside:
            invalid = ~np.logical_and(mask, points_inside)
            r[invalid], c[invalid] = nodata, nodata
        elif np.any(points_inside[mask] == False):
            raise ValueError("Coordinates outside domain.")
        return r, c

    def xy(self, r, c, mask=None, mask_outside=False, nodata=np.nan):
        """Return x,y coordinates at cell center of row, col indices

        Arguments
        ----------
        r, c : ndarray of int
            index of row, column
        mask : ndarray of bool, optional
            data mask of valid values, by default None
        mask_outside : boolean, optional
            mask xy points outside domain (i.e. set nodata), by default False
        nodata : int, optional
            nodata value, used for output array, by default np.nan

        Returns
        -------
        Tuple of ndarray of float
            x, y coordinates
        """
        nrow, ncol = self.shape
        r = np.atleast_1d(r)
        c = np.atleast_1d(c)
        points_inside = np.logical_and.reduce((r >= 0, r < nrow, c >= 0, c < ncol))
        if mask is None:
            mask = np.ones(r.shape, dtype=bool)  # all valid
        if mask_outside:
            mask[points_inside == False] = False
        elif np.any(points_inside[mask] == False):
            raise ValueError("Linear indices outside domain.")
        y = np.full(r.shape, nodata, dtype=np.float64)
        x = np.full(r.shape, nodata, dtype=np.float64)
        y[mask] = self.ycoords.values[r[mask]]
        x[mask] = self.xcoords.values[c[mask]]
        return x, y

    def idx_to_xy(self, idx, mask=None, mask_outside=False, nodata=np.nan):
        """Return x,y coordinates at linear index

        Arguments
        ----------
        idx : ndarray of int
            linear index
        mask : ndarray of bool, optional
            data mask of valid values, by default None
        mask_outside : boolean, optional
            mask xy points outside domain (i.e. set nodata), by default False
        nodata : int, optional
            nodata value, used for output array, by default np.nan

        Returns
        -------
        Tuple of ndarray of float
            x, y coordinates
        """
        idx = np.atleast_1d(idx)
        nrow, ncol = self.shape
        r, c = idx // ncol, idx % ncol
        return self.xy(r, c, mask=mask, mask_outside=mask_outside, nodata=nodata)

    def xy_to_idx(self, xs, ys, mask=None, mask_outside=False, nodata=-1):
        """Return linear index of x, y coordinates

        Arguments
        ----------
        xs, ys: ndarray of float
            x, y coordinates
        mask : ndarray of bool, optional
            data mask of valid values, by default None
        mask_outside : boolean, optional
            mask xy points outside domain (i.e. set nodata), by default False
        nodata : int, optional
            nodata value, used for output array, by default -1

        Returns
        -------
        ndarray of int
            linear indices
        """
        _, ncol = self.shape
        r, c = self.rowcol(xs, ys, mask=mask, mask_outside=mask_outside, nodata=nodata)
        mask = r != nodata
        idx = np.full(r.shape, nodata, dtype=int)
        idx[mask] = r[mask] * ncol + c[mask]
        return idx

    def sample(self, gdf, wdw=0):
        """Sample from map at point locations with optional window around the points.

        Arguments
        ---------
        gdf: geopandas.GeoDataFrame
            GeoDataFrame with Point geometries
        wdw: int
            Number of cells around point location to sample from

        Returns
        -------
        ojb_out: xr.Dataset or xr.DataArray
            Output sample data
        """
        # TODO: add method for line geometries
        if not np.all(gdf.geometry.type == "Point"):
            raise ValueError("Only point geometries accepted")

        if gdf.crs is not None and self.crs is not None and gdf.crs != self.crs:
            gdf = gdf.to_crs(self.crs)

        pnts = gdf.geometry
        r, c = self.rowcol(pnts.x.values, pnts.y.values, mask_outside=True, nodata=-1)
        if wdw > 0:
            ar_wdw = np.arange(-wdw, wdw + 1)
            rwdw = np.add.outer(r, np.repeat(ar_wdw, ar_wdw.size))
            cwdw = np.add.outer(c, np.tile(ar_wdw, ar_wdw.size))
            nrow, ncol = self.shape
            mask = np.logical_or(
                np.logical_or(rwdw < 0, rwdw >= nrow),
                np.logical_or(cwdw < 0, cwdw >= ncol),
            )
            rwdw[mask] = -1
            cwdw[mask] = -1
            ds_sel = xr.Dataset(
                {
                    "index": xr.IndexVariable("index", gdf.index.values),
                    "mask": xr.Variable(("index", "wdw"), ~mask),
                    self.x_dim: xr.Variable(("index", "wdw"), cwdw),
                    self.y_dim: xr.Variable(("index", "wdw"), rwdw),
                }
            )
        else:
            ds_sel = xr.Dataset(
                {
                    "index": xr.IndexVariable("index", gdf.index.values),
                    "mask": xr.Variable("index", np.logical_and(r != -1, c != -1)),
                    self.x_dim: xr.Variable("index", c),
                    self.y_dim: xr.Variable("index", r),
                }
            )
        obj_out = self._obj.isel(ds_sel[[self.y_dim, self.x_dim]])
        if np.any(~ds_sel["mask"]):  # mask out of domain points
            obj_out = obj_out.raster.mask(ds_sel["mask"])
        return obj_out

    def zonal_stats(self, gdf, stats, all_touched=False):
        """Calculate zonal statistics of raster samples aggregated for geometries.

        Arguments
        ---------
        gdf: geopandas.GeoDataFrame
            GeoDataFrame with geometries
        stats: list of str, callable
            Statistics to compute from raster values, options include
            {'count', 'min', 'max', 'sum', 'mean', 'std', 'median', 'q##'}.
            Multiple percentiles can be calculated using comma-seperated values, e.g.: 'q10,50,90'
            Statistics ignore the nodata value and are applied along the x and y dimension.
            By default ['mean']
        all_touched : bool, optional
            If True, all pixels touched by geometries will used to define the sample.
            If False, only pixels whose center is within the geometry or that are
            selected by Bresenham's line algorithm will be used. By default False.

        Returns
        -------
        ojb_out: xr.Dataset
            Output dataset with a variable for each combination of input variable
            and statistic.
        """
        _ST = ["count", "min", "max", "sum", "mean", "std", "median"]

        def rmd(ds, stat):
            return {var: f"{var}_{stat}" for var in ds.raster.vars}

        def gen_zonal_stat(ds, geoms, stats, all_touched=False):
            a, b, _, d, e, _, _, _, _ = tuple(ds.raster.transform)
            dims = (ds.raster.x_dim, ds.raster.y_dim)
            for i, geom in enumerate(geoms):
                # add buffer to work with point geometries
                ds1 = ds.raster.clip_bbox(geom.bounds, buffer=1).raster.mask_nodata()
                if np.any(np.asarray(ds1.raster.shape) == 0):
                    continue
                # transform based on ds resolution as ds1 resolution cannot be
                # calculated in case of x/y dimension with length one
                c, f = (
                    ds1.raster.xcoords.values[0] - a / 2.0,
                    ds1.raster.ycoords.values[0] - e / 2.0,
                )
                transform = Affine(a, b, c, d, e, f)
                mask = full(ds1.raster.coords, nodata=0, dtype=np.uint8)
                features.rasterize(
                    [(geom, 1)],
                    out_shape=mask.raster.shape,
                    fill=0,
                    transform=transform,
                    out=mask.data,
                    all_touched=all_touched,
                )
                ds1 = ds1.where(mask == 1)
                dss = []
                for stat in stats:
                    if stat in _ST:
                        ds1_stat = getattr(ds1, stat)(dims)
                        dss.append(ds1_stat.rename(rmd(ds1, stat)))
                    elif isinstance(stat, str) and stat.startswith("q"):
                        qs = np.array([float(q) for q in stat.strip("q").split(",")])
                        dss.append(
                            ds1.quantile(qs / 100, dims).rename(rmd(ds1, "quantile"))
                        )
                    elif callable(stat):
                        dss.append(
                            ds1.reduce(stat, dims).rename(rmd(ds1, stat.__name__))
                        )
                    else:
                        raise ValueError(f"Stat {stat} not valid.")
                yield xr.merge(dss), i

        if isinstance(stats, str):
            stats = stats.split()
        elif callable(stats):
            stats = list([stats])

        if gdf.crs is not None and self.crs is not None and gdf.crs != self.crs:
            gdf = gdf.to_crs(self.crs)
        geoms = gdf["geometry"].values

        ds = self._obj.copy()
        if isinstance(ds, xr.DataArray):
            if ds.name is None:
                ds.name = "values"
            ds = ds.to_dataset()

        out = list(gen_zonal_stat(ds, geoms, stats, all_touched))
        if len(out) == 0:
            raise IndexError("All geometries outside raster domain")

        dss, idx = zip(*out)
        ds_out = xr.concat(dss, "index")
        ds_out["index"] = xr.IndexVariable("index", gdf.index.values[np.array(idx)])

        return ds_out

    def clip_bbox(self, bbox, align=None, buffer=0):
        """Clip object based on a bounding box.

        Arguments
        ----------
        bbox : array-like of floats
            (xmin, ymin, xmax, ymax) bounding box
        align : float, optional
            Resolution to align the bounding box, by default None
        buffer : int, optional
            Buffer around the bounding box expressed in resolution multiplicity,
            by default 0

        Returns
        -------
        xarray.DataSet or DataArray
            Data clipped to bbox
        """
        w, s, e, n = bbox
        left, bottom, right, top = self.internal_bounds
        xres, yres = [np.abs(res) for res in self.res]
        if align is not None:
            align = abs(align)
            # align to grid
            w = (w // align) * align
            s = (s // align) * align
            e = (e // align + 1) * align
            n = (n // align + 1) * align
        if top > bottom:
            n = min(top, n + buffer * yres)
            s = max(bottom, s - buffer * yres)
            y_slice = slice(n, s)
        else:
            n = min(bottom, n + buffer * yres)
            s = max(top, s - buffer * yres)
            y_slice = slice(s, n)
        if left > right:
            e = min(left, e + buffer * xres)
            w = max(right, w - buffer * xres)
            x_slice = slice(e, w)
        else:
            e = min(right, e + buffer * xres)
            w = max(left, w - buffer * xres)
            x_slice = slice(w, e)
        return self._obj.sel({self.x_dim: x_slice, self.y_dim: y_slice})

    def clip_mask(self, mask):
        """Clip object to region with mask values greater than zero.

        Arguments
        ---------
        mask : xarray.DataArray
            Mask array.

        Returns
        -------
        xarray.DataSet or DataArray
            Data clipped to mask
        """
        if not isinstance(mask, xr.DataArray):
            raise ValueError("Mask should be xarray.DataArray type.")
        if not mask.raster.shape == self.shape:
            raise ValueError("Mask shape invalid.")
        mask_bin = (mask.values != 0).astype(np.uint8)
        if not np.any(mask_bin):
            raise ValueError("Invalid mask.")
        row_slice, col_slice = ndimage.find_objects(mask_bin)[0]
        self._obj.coords["mask"] = xr.Variable(self.dims, mask_bin)
        return self._obj.isel({self.x_dim: col_slice, self.y_dim: row_slice})

    def clip_geom(self, geom, align=None, buffer=0, mask=False):
        """Clip object to the bounding box of the geometry and add geometry 'mask' coordinate.

        Arguments
        ---------
        geom : geopandas.GeoDataFrame/Series,
            A geometry defining the area of interest.
        align : float, optional
            Resolution to align the bounding box, by default None
        buffer : int, optional
            Buffer around the bounding box expressed in resolution multiplicity,
            by default 0
        mask: bool, optional
            Mask values outside geometry with the

        Returns
        -------
        xarray.DataSet or DataArray
            Data clipped to geometry
        """
        # TODO make common geom to gdf with correct crs parsing
        if not hasattr(geom, "crs"):
            raise ValueError("geom should be geopandas GeoDataFrame object.")
        bbox = geom.total_bounds
        if geom.crs is not None and self.crs is not None and geom.crs != self.crs:
            bbox = rasterio.warp.transform_bounds(geom.crs, self.crs, *bbox)
        obj_clip = self.clip_bbox(bbox, align=align, buffer=buffer)
        obj_clip.coords["mask"] = obj_clip.raster.geometry_mask(geom)
        if mask:
            obj_clip = obj_clip.raster.mask(obj_clip.coords["mask"])
        return obj_clip

    def rasterize(
        self,
        gdf,
        col_name="index",
        nodata=0,
        all_touched=False,
        dtype=None,
        sindex=False,
        **kwargs,
    ):
        """Return an object with input geometry values burned in.

        Arguments
        ---------
        gdf : geopandas.GeoDataFrame
            GeoDataFrame of shapes and values to burn.
        col_name : str, optional
            GeoDataFrame column name to use for burning, by default 'index'
        nodata : int or float, optional
            Used as fill value for all areas not covered by input geometries, by default 0.
        all_touched : bool, optional
            If True, all pixels touched by geometries will be burned in. If false, only
            pixels whose center is within the polygon or that are selected by
            Bresenham's line algorithm will be burned in.
        dtype : numpy dtype, optional
            Used as data type for results, by default it is derived from values.
        sindex : bool, optional
            Create a spatial index to select overlapping geometries before rasterizing,
            by default False.

        Returns
        -------
        xarray.DataArray
            DataArray with burned geometries

        Raises
        ------
        ValueError
            If no geometries are found inside the bounding box.
        """
        if gdf.crs is not None and self.crs is not None and gdf.crs != self.crs:
            gdf = gdf.to_crs(self.crs)

        if sindex:
            idx = list(gdf.sindex.intersection(self.bounds))
            gdf = gdf.iloc[idx, :]

        if len(gdf.index) > 0:
            geoms = gdf.geometry.values
            values = gdf.reset_index()[col_name].values
            dtype = values.dtype if dtype is None else dtype
            if dtype == np.int64:
                dtype = np.int32  # max integer accuracy accepted
            shapes = list(zip(geoms, values))
            raster = np.full(self.shape, nodata, dtype=dtype)
            features.rasterize(
                shapes,
                out_shape=self.shape,
                fill=nodata,
                transform=self.transform,
                out=raster,
                all_touched=all_touched,
                **kwargs,
            )
        else:
            raise ValueError("No shapes found within raster bounding box")
        attrs = self._obj.attrs.copy()
        da_out = xr.DataArray(
            name=col_name, dims=self.dims, coords=self.coords, data=raster, attrs=attrs
        )
        da_out.raster.set_nodata(nodata)
        da_out.raster.set_attrs(**self.attrs)
        return da_out

    def geometry_mask(self, gdf, all_touched=False, invert=False, **kwargs):
        """Return a grid with True values where shapes overlap pixels.

        Arguments
        ---------
        gdf : geopandas.GeoDataFrame
            GeoDataFrame of shapes and values to burn.
        all_touched : bool, optional
            If True, all pixels touched by geometries will masked. If false, only
            pixels whose center is within the polygon or that are selected by
            Bresenham's line algorithm will be burned in. By default False.
        invert : bool, optional
            If True, the mask will be False where shapes overlap pixels, by default False

        Returns
        -------
        xarray.DataArray
            Geometry mask
        """
        gdf1 = gdf.copy()
        gdf1["mask"] = np.full(gdf.index.size, (not invert), dtype=np.uint8)
        da_out = self.rasterize(
            gdf1,
            col_name="mask",
            all_touched=all_touched,
            nodata=np.uint8(invert),
            **kwargs,
        )
        # remove nodata value before converting to boolean
        da_out.attrs.pop("_FillValue", None)
        return da_out.astype(bool)

    def vector_grid(self):
        """Return a geopandas GeoDataFrame with a box for each grid cell."""
        w, _, _, n = self.bounds
        dx, dy = self.res
        nrow, ncol = self.shape
        cells = []
        for i in range(nrow):
            top = n + i * dy
            bottom = n + (i + 1) * dy
            for j in range(ncol):
                left = w + j * dx
                right = w + (j + 1) * dx
                cells.append(box(left, bottom, right, top))
        return gpd.GeoDataFrame(geometry=cells, crs=self.crs)

    def area_grid(self, dtype=np.float32):
        """Returns the grid cell area [m2].

        Returns
        -------
        da_area : xarray.DataArray
            Grid cell surface area [m2].
        """
        if self.crs.is_geographic:
            data = gis_utils.reggrid_area(self.ycoords.values, self.xcoords.values)
        elif self.crs.is_projected:
            xres = abs(self.res[0]) * self.crs.linear_units_factor[1]
            yres = abs(self.res[1]) * self.crs.linear_units_factor[1]
            data = np.full(self.shape, xres * yres)
        da_area = xr.DataArray(
            data=data.astype(dtype), coords=self.coords, dims=self.dims
        )
        da_area.raster.set_nodata(0)
        da_area.raster.set_crs(self.crs)
        da_area.attrs.update(unit="m2")
        return da_area.rename("area")

    def density_grid(self):
        """Returns the density in [unit/m2] of raster(s). The cell areas are calculated
        using :py:meth:`~hydromt.raster.XRasterBase.area_grid`.

        Returns
        -------
        ds_out: xarray.DataArray or xarray.DataSet
            The density in [unit/m2] of the raster.
        """

        # Create a grid that contains the area in m2 per grid cell.
        if self.crs.is_geographic:
            area = self.area_grid()

        elif self.crs.is_projected:
            xres = abs(self.res[0]) * self.crs.linear_units_factor[1]
            yres = abs(self.res[1]) * self.crs.linear_units_factor[1]
            area = xres * yres

        # Create a grid that contains the density in unit/m2 per grid cell.
        unit = self._obj.attrs.get("unit", "")
        ds_out = self._obj / area
        ds_out.attrs.update(unit=f"{unit}.m-2")
        return ds_out

    def _dst_transform(
        self,
        dst_crs=None,
        dst_res=None,
        dst_transform=None,
        dst_width=None,
        dst_height=None,
        align=False,
    ):
        if dst_transform is None or dst_width is None or dst_height is None:
            (
                dst_transform,
                dst_width,
                dst_height,
            ) = rasterio.warp.calculate_default_transform(
                self.crs,
                dst_crs,
                self.width,
                self.height,
                *self.internal_bounds,
                resolution=dst_res,
                dst_width=dst_width,
                dst_height=dst_height,
            )
        if align:
            dst_transform, dst_width, dst_height = rasterio.warp.aligned_target(
                dst_transform, dst_width, dst_height, dst_res
            )
        return dst_transform, dst_width, dst_height

    def _dst_crs(self, dst_crs=None):
        # check CRS and transform set destination crs if missing
        if self.crs is None:
            raise ValueError("CRS is missing. Use set_crs function to resolve.")
        if dst_crs == "utm":
            # make sure bounds are in EPSG:4326
            dst_crs = gis_utils.utm_crs(self.box.to_crs(4326).total_bounds)
        else:
            dst_crs = CRS.from_user_input(dst_crs) if dst_crs is not None else self.crs
        return dst_crs

    def nearest_index(
        self,
        dst_crs=None,
        dst_res=None,
        dst_transform=None,
        dst_width=None,
        dst_height=None,
        align=False,
    ):
        """Prepare nearest index mapping for the reprojection of a gridded timeseries
        file, powered by pyproj and k-d tree lookup.

        Index mappings typically are used in reprojection workflows of time series,
        or combinations of time series

        ... Note: Is used by :py:meth:`~hydromt.raster.RasterDataArray.reproject` if method equals 'nearest_index'

        Arguments
        ----------
        dst_crs: int, dict, or str, optional
            Target CRS. Accepts EPSG codes (int or str); proj (str or dict) or wkt (str)
            "utm" is accepted and will return the centroid utm zone CRS
        dst_res: tuple (x resolution, y resolution) or float, optional
            Target resolution, in units of target CRS.
        dst_transform: affine.Affine(), optional
            Target affine transformation. Will be calculated if None.
        dst_width, dst_height: int, optional
            Output file size in pixels and lines. Cannot be used together with
            resolution (dst_res).
        align: boolean, optional
            If True, align target transform to resolution

        Returns
        -------
        index: xarray.DataArray of intp
            DataArray with flat indices of source DataArray
        """
        # parse and check destination grid and crs
        dst_crs = self._dst_crs(dst_crs)
        dst_transform, dst_width, dst_height = self._dst_transform(
            dst_crs, dst_res, dst_transform, dst_width, dst_height, align
        )
        # Transform the destination grid points to the source CRS.
        reproj2src = pyproj.transformer.Transformer.from_crs(
            crs_from=dst_crs, crs_to=self.crs, always_xy=True
        )
        # Create destination coordinate pairs in source CRS.
        dst_xs, dst_ys = gis_utils.affine_to_coords(
            dst_transform, (dst_height, dst_width)
        )
        dst_yy, dst_xx = np.meshgrid(dst_ys, dst_xs, indexing="ij")
        dst_yy, dst_xx = dst_yy.ravel(), dst_xx.ravel()
        dst_xx_reproj, dst_yy_reproj = reproj2src.transform(xx=dst_xx, yy=dst_yy)
        dst_coords_reproj = np.vstack([dst_xx_reproj, dst_yy_reproj]).transpose()
        # Create source coordinate pairs.
        src_yy, src_xx = np.meshgrid(self.ycoords, self.xcoords, indexing="ij")
        src_yy, src_xx = src_yy.ravel(), src_xx.ravel()
        src_coords = np.vstack([src_xx, src_yy]).transpose()
        # Build a KD-tree with the source grid cell center coordinate pairs.
        # For each destination grid cell coordinate pairs, search for the nearest
        # source grid cell in the KD-tree.
        # TODO: benchmark against RTree or S2Index https://github.com/benbovy/pys2index
        tree = cKDTree(src_coords)
        _, indices = tree.query(dst_coords_reproj)
        # filter destination cells with center outside source bbox
        w, s, e, n = self.bounds
        valid = np.logical_and(
            np.logical_and(dst_xx_reproj > w, dst_xx_reproj < e),
            np.logical_and(dst_yy_reproj > s, dst_yy_reproj < n),
        )
        indices[~valid] = -1  # nodata value
        # create 2D remapping dataset
        dims = (self.y_dim, self.x_dim)
        dst_shape = (dst_ys.size, dst_xs.size)
        index = xr.DataArray(
            data=indices.reshape(dst_shape),
            dims=dims,
            coords={self.y_dim: dst_ys, self.x_dim: dst_xs},
        )
        index.raster.set_crs(dst_crs)
        index.raster.set_nodata(-1)
        return index


@xr.register_dataarray_accessor("raster")
class RasterDataArray(XRasterBase):
    """This is the GIS extension for xarray.DataArray"""

    def __init__(self, xarray_obj):
        super(RasterDataArray, self).__init__(xarray_obj)

    @staticmethod
    def from_numpy(data, transform, nodata=None, attrs={}, crs=None):
        """Transform a 2D/3D numpy array into a DataArray with geospatial attributes.
        The data dimensions should have the y and x on the second last and last dimensions.

        Arguments
        ---------
        data : numpy.array, 2-dimensional
            values to parse into DataArray
        transform : affine transform
            Two dimensional affine transform for 2D linear mapping
        nodata : float or int, optional
            nodata value
        attrs : dict, optional
            additional attributes
        crs: int, dict, or str, optional
            Coordinate Reference System. Accepts EPSG codes (int or str); proj (str or dict)
            or wkt (str)

        Returns
        -------
        da : RasterDataArray
            xarray.DataArray with geospatial information
        """
        nrow, ncol = data.shape[-2:]
        dims = ("y", "x")
        if len(data.shape) == 3:
            dims = ("dim0",) + dims
        elif len(data.shape) != 2:
            raise ValueError("Only 2D and 3D arrays supported")
        _xcoords, _ycoords = gis_utils.affine_to_coords(transform, (nrow, ncol))
        da = xr.DataArray(
            data,
            dims=dims,
            coords={"y": _ycoords, "x": _xcoords},
        )
        da.raster.set_spatial_dims(x_dim="x", y_dim="y")
        da.raster.set_nodata(nodata=nodata)  # set  _FillValue attr
        if attrs:
            da.attrs.update(attrs)
        if crs is not None:
            da.raster.set_crs(input_crs=crs)
        return da

    @property
    def nodata(self):
        """Nodata value of the DataArray."""
        # first check attrs, then encoding
        nodata = self._obj.rio.nodata
        if nodata is None:
            nodata = self._obj.rio.encoded_nodata
            if nodata is not None:
                self.set_nodata(nodata)
        return nodata

    def set_nodata(self, nodata=None, logger=logger):
        """Set the nodata value as CF compliant attribute of the DataArray.

        Arguments
        ----------
        nodata: float, integer
            Nodata value for the DataArray.
            If the nodata property and argument are both None, the _FillValue
            attribute will be removed.
        """
        if nodata is None:
            nodata = self._obj.rio.nodata
            if nodata is None:
                nodata = self._obj.rio.encoded_nodata
        # Only numerical nodata values are supported
        if np.issubdtype(type(nodata), np.number):
            self._obj.rio.set_nodata(nodata, inplace=True)
            self._obj.rio.write_nodata(nodata, inplace=True)
        else:
            logger.warning("No numerical nodata value found, skipping set_nodata")
            self._obj.attrs.pop("_FillValue", None)

    def mask_nodata(self, fill_value=np.nan):
        """Mask nodata values with fill_value (default np.nan).
        Note that masking with np.nan will change integer dtypes to float.
        """
        _da = self._obj
        if self.nodata is not None and self.nodata != fill_value:
            mask = _da.notnull() if np.isnan(self.nodata) else _da != self.nodata
            _da = _da.where(mask, fill_value)
            _da.raster.set_nodata(fill_value)
        return _da

    def mask(self, mask, logger=logger):
        """Mask cells where mask equals False with the data nodata value.
        A warning is raised if no the data has no nodata value."""
        if self.nodata is not None:
            da_masked = self._obj.where(mask != 0, self.nodata)
        else:
            logger.warning("Nodata value missing, skipping mask")
            da_masked = self._obj
        return da_masked

    def _reproject(
        self,
        dst_crs,
        dst_transform,
        dst_width,
        dst_height,
        dst_nodata=np.nan,
        method="nearest",
    ):
        """Reproject a DataArray, powered by :py:meth:`rasterio.warp.reproject`."""
        resampling = getattr(Resampling, method, None)
        if resampling is None:
            raise ValueError(f"Resampling method unknown: {method}.")
        # create new DataArray for output
        xs, ys = gis_utils.affine_to_coords(dst_transform, (dst_height, dst_width))
        dst_coords = {d: self._obj.coords[d] for d in self._obj.dims}
        dst_coords.update({self.y_dim: ys, self.x_dim: xs})
        da_reproject = full(
            dst_coords,
            nodata=dst_nodata,
            dtype=self._obj.dtype,
            name=self._obj.name,
            attrs=self._obj.attrs,
            crs=dst_crs,
        )
        # apply rasterio warp reproject
        rasterio.warp.reproject(
            source=self._obj.load().data,
            destination=da_reproject.data,
            src_transform=self.transform,
            src_crs=self.crs,
            src_nodata=self.nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=da_reproject.raster.nodata,
            resampling=resampling,
        )
        return da_reproject

    def _reindex2d(self, index, dst_nodata=np.nan):
        """Return reindexed (reprojected) object"""
        # create new DataArray for output
        dst_coords = {d: self._obj.coords[d] for d in self._obj.dims}
        ys, xs = index.raster.ycoords, index.raster.xcoords
        dst_coords.update({self.y_dim: ys, self.x_dim: xs})
        da_reproject = full(
            dst_coords,
            nodata=dst_nodata,
            dtype=self._obj.dtype,
            name=self._obj.name,
            attrs=self._obj.attrs,
            crs=index.raster.crs,
        )
        # reproject by indexing
        shape2d = (self._obj.shape[0] if self.dim0 else 1, self.size)
        src_data = self._obj.load().data.reshape(shape2d)
        idxs = index.values
        valid = idxs >= 0
        if self.dim0:
            da_reproject.data[:, valid] = src_data[:, idxs[valid]]
        else:
            da_reproject.data[valid] = src_data[:, idxs[valid]].squeeze()
        return da_reproject

    def reproject(
        self,
        dst_crs=None,
        dst_res=None,
        dst_transform=None,
        dst_width=None,
        dst_height=None,
        dst_nodata=None,
        method="nearest",
        align=False,
    ):
        """Reproject a DataArray with geospatial coordinates, powered
        by :py:meth:`rasterio.warp.reproject`.

        Arguments
        ----------
        dst_crs: int, dict, or str, optional
            Target CRS. Accepts EPSG codes (int or str); proj (str or dict) or wkt (str)
            "utm" is accepted and will return the centroid utm zone CRS
        dst_res: tuple (x resolution, y resolution) or float, optional
            Target resolution, in units of target CRS.
        dst_transform: affine.Affine(), optional
            Target affine transformation. Will be calculated if None.
        dst_width, dst_height: int, optional
            Output file size in pixels and lines. Cannot be used together with
            resolution (dst_res).
        dst_nodata: int or float, optional
            The nodata value used to initialize the destination; it will
            remain in all areas not covered by the reprojected source. If None, the
            source nodata value will be used.
        method: str, optional
            See rasterio.warp.reproject for existing methods, by default nearest.
            Additionally "nearest_index" can be used for KDTree based downsampling.
        align: boolean, optional
            If True, align target transform to resolution

        Returns
        -------
        da_reproject : xarray.DataArray
            A reprojected DataArray.
        """

        def _reproj(da, **kwargs):
            return da.raster._reproject(**kwargs)

        # parse and check destination grid and crs
        dst_crs = self._dst_crs(dst_crs)
        dst_transform, dst_width, dst_height = self._dst_transform(
            dst_crs, dst_res, dst_transform, dst_width, dst_height, align
        )
        reproj_kwargs = dict(
            dst_crs=dst_crs,
            dst_transform=dst_transform,
            dst_width=dst_width,
            dst_height=dst_height,
        )
        # gdal resampling method with exception for index based resampling
        method = method.lower()
        if method == "nearest_index":
            index = self.nearest_index(**reproj_kwargs)
            return self.reindex2d(index, dst_nodata)
        # update reproject settings
        if dst_nodata is None:
            dst_nodata = self.nodata if self.nodata is not None else np.nan
        reproj_kwargs.update(method=method, dst_nodata=dst_nodata)
        if self._obj.chunks is None:
            da_reproj = _reproj(self._obj, **reproj_kwargs)
        else:
            # create template with dask data
            xs, ys = gis_utils.affine_to_coords(dst_transform, (dst_height, dst_width))
            dst_coords = {d: self._obj.coords[d] for d in self._obj.dims}
            dst_coords.update({self.y_dim: ys, self.x_dim: xs})
            da_temp = full(
                dst_coords,
                nodata=dst_nodata,
                dtype=self._obj.dtype,
                name=self._obj.name,
                attrs=self._obj.attrs,
                crs=dst_crs,
                lazy=True,
            )
            # chunk time and set reset chunks on other dims
            chunksize = max(self._obj.chunks[0])
            chunks = {d: chunksize if d == self.dim0 else -1 for d in self._obj.dims}
            _da = self._obj.chunk(chunks)
            da_temp = da_temp.chunk(chunks)
            da_reproj = _da.map_blocks(_reproj, kwargs=reproj_kwargs, template=da_temp)
        da_reproj.raster.set_crs(dst_crs)
        return da_reproj.raster.reset_spatial_dims_attrs()

    def reproject_like(self, other, method="nearest"):
        """Reproject a object to match the grid of ``other``.

        Arguments
        ----------
        other : xarray.DataArray or Dataset
            DataArray of the target resolution and projection.
        method : str, optional
            See :py:meth:`~hydromt.raster.RasterDataArray.reproject` for existing methods,
            by default 'nearest'.

        Returns
        --------
        da : xarray.DataArray
            Reprojected object.
        """
        # clip first; then reproject
        da = self._obj
        if self.aligned_grid(other):
            da = self.clip_bbox(other.raster.bounds)
        elif not self.identical_grid(other):
            dst_bbox = other.raster.transform_bounds(self.crs)
            da = self.clip_bbox(dst_bbox, buffer=2).raster.reproject(
                dst_crs=other.raster.crs,
                dst_transform=other.raster.transform,
                dst_width=other.raster.width,
                dst_height=other.raster.height,
                method=method,
            )
        if (
            da.raster.x_dim != other.raster.x_dim
            or da.raster.y_dim != other.raster.y_dim
        ):
            # overwrite spatial dimension names which might have been changed
            rm = {
                da.raster.x_dim: other.raster.x_dim,
                da.raster.y_dim: other.raster.y_dim,
            }
            da = da.rename(rm)
            da.raster.set_spatial_dims(
                x_dim=other.raster.x_dim, y_dim=other.raster.y_dim
            )
        # make sure coordinates are identical!
        da[other.raster.x_dim] = other.raster.xcoords
        da[other.raster.y_dim] = other.raster.ycoords
        return da

    def reindex2d(self, index, dst_nodata=None):
        """Return reprojected DataArray object based on simple reindexing using
        linear indices in ``index``, which can be calculated with
        :py:meth:`~hydromt.raster.RasterDataArray.nearest_index`.

        This is typically used to downscale time series data.

        Arguments
        ----------
        index: xarray.DataArray of intp
            DataArray with flat indices of source DataArray

        Returns
        -------
        da_reproject : xarray.DataArray
            The reindexed DataArray.
        """

        def _reindex2d(da, index, dst_nodata):
            return da.raster._reindex2d(index=index, dst_nodata=dst_nodata)

        if dst_nodata is None:
            dst_nodata = self.nodata if self.nodata is not None else np.nan
        kwargs = dict(index=index, dst_nodata=dst_nodata)
        if self._obj.chunks is None:
            da_reproj = _reindex2d(self._obj, **kwargs)
        else:
            # create template with dask data
            dst_coords = {d: self._obj.coords[d] for d in self._obj.dims}
            ys, xs = index.raster.ycoords, index.raster.xcoords
            dst_coords.update({self.y_dim: ys, self.x_dim: xs})
            da_temp = full(
                dst_coords,
                nodata=dst_nodata,
                dtype=self._obj.dtype,
                name=self._obj.name,
                attrs=self._obj.attrs,
                crs=index.raster.crs,
                lazy=True,
            )
            # chunk along first dim
            chunksize = max(self._obj.chunks[0])
            chunks = {d: chunksize if d == self.dim0 else -1 for d in self._obj.dims}
            _da = self._obj.chunk(chunks)
            da_temp = da_temp.chunk(chunks)
            # map blocks
            da_reproj = _da.map_blocks(_reindex2d, kwargs=kwargs, template=da_temp)
        da_reproj.raster.set_nodata(dst_nodata)
        return da_reproj.raster.reset_spatial_dims_attrs()

    def _interpolate_na(
        self, src_data: np.ndarray, method: str = "nearest", **kwargs
    ) -> np.ndarray:
        """Returns interpolated array"""
        data_isnan = True if self.nodata is None else np.isnan(self.nodata)
        mask = ~np.isnan(src_data) if data_isnan else src_data != self.nodata
        if not mask.any() or mask.all():
            return src_data
        if method == "rio_idw":  # NOTE: modifies src_data inplace
            # NOTE this method might also extrapolate
            interp_data = rasterio.fill.fillnodata(src_data.copy(), mask, **kwargs)
        else:
            # get valid cells D4-neighboring nodata cells to setup triangulation
            valid = np.logical_and(mask, ndimage.binary_dilation(~mask))
            xs, ys = np.meshgrid(self.xcoords.values, self.ycoords.values)
            # interpolate data at nodata cells only
            interp_data = src_data.copy()
            interp_data[~mask] = griddata(
                points=(xs[valid], ys[valid]),
                values=src_data[valid],
                xi=(xs[~mask], ys[~mask]),
                method=method,
                fill_value=self.nodata,
            )
        return interp_data

    def interpolate_na(self, method: str = "nearest", **kwargs):
        """Interpolate missing data

        Arguments
        ----------
        method: {'linear', 'nearest', 'cubic', 'rio_idw'}, optional
            {'linear', 'nearest', 'cubic'} use :py:meth:`scipy.interpolate.griddata`;
            'rio_idw' applies inverse distance weighting based on :py:meth:`rasterio.fill.fillnodata`.
        **kwargs
            Additional key-word arguments are passed to :py:meth:`rasterio.fill.fillnodata`,
            only used in combination with `method='rio_idw'`

        Returns
        -------
        xarray.DataArray
            Filled object
        """
        dim0 = self.dim0
        if dim0:
            interp_data = np.empty(self._obj.shape, dtype=self._obj.dtype)
            for i, (_, sub_xds) in enumerate(self._obj.groupby(dim0)):
                interp_data[i, ...] = self._interpolate_na(
                    sub_xds.load().data, method=method, **kwargs
                )
        else:
            interp_data = self._interpolate_na(
                self._obj.load().data, method=method, **kwargs
            )
        interp_array = xr.DataArray(
            name=self._obj.name,
            dims=self._obj.dims,
            coords=self._obj.coords,
            data=interp_data,
            attrs=self._obj.attrs,
        )
        return interp_array

    def to_raster(
        self,
        raster_path,
        driver="GTiff",
        dtype=None,
        tags=None,
        windowed=False,
        mask=False,
        logger=logger,
        **profile_kwargs,
    ):
        """Write DataArray object to a gdal-writable raster file.

        Arguments
        ----------
        raster_path: str
            The path to output the raster to.
        driver: str, optional
            The name of the GDAL/rasterio driver to use to export the raster.
            Default is "GTiff".
        dtype: str, optional
            The data type to write the raster to. Default is the datasets dtype.
        tags: dict, optional
            A dictionary of tags to write to the raster.
        windowed: bool, optional
            If True, it will write using the windows of the output raster.
            Default is False.
        mask: bool, optional
            If True, set nodata values where 'mask' coordinate equals False.
        **profile_kwargs
            Additional keyword arguments to pass into writing the raster. The
            nodata, transform, crs, count, width, and height attributes
            are ignored.

        """
        for k in ["height", "width", "count", "transform"]:
            if k in profile_kwargs:
                msg = f"{k} will be set based on the DataArray, remove the argument"
                raise ValueError(msg)
        da_out = self._obj
        # set nodata, mask, crs and dtype
        if "nodata" in profile_kwargs:
            da_out.raster.set_nodata(profile_kwargs.pop("nodata"))
        nodata = da_out.raster.nodata
        if nodata is not None and not np.isnan(nodata):
            da_out = da_out.fillna(nodata)
        elif nodata is None:
            logger.warning(f"nodata value missing for {raster_path}")
        if mask and "mask" in da_out.coords and nodata is not None:
            da_out = da_out.where(da_out.coords["mask"] != 0, nodata)
        if dtype is not None:
            da_out = da_out.astype(dtype)
        if "crs" in profile_kwargs:
            da_out.raster.set_crs(profile_kwargs.pop("crs"))
        # check dimensionality
        dim0 = da_out.raster.dim0
        count = 1
        if dim0 is not None:
            count = da_out[dim0].size
            da_out = da_out.sortby(dim0)
        # write
        if driver.lower() == "pcraster" and HAS_PCRASTER:
            for i in range(count):
                if dim0:
                    bname = basename(raster_path).split(".")[0]
                    bname = f"{bname[:8]:8s}".replace(" ", "0")
                    raster_path = join(dirname(raster_path), f"{bname}.{i+1:03d}")
                    data = da_out.isel({dim0: i}).load().squeeze().data
                else:
                    data = da_out.load().data
                gis_utils.write_map(
                    data,
                    raster_path,
                    crs=da_out.raster.crs,
                    transform=da_out.raster.transform,
                    nodata=nodata,
                    **profile_kwargs,
                )
        else:
            profile = dict(
                driver=driver,
                height=da_out.raster.height,
                width=da_out.raster.width,
                count=count,
                dtype=str(da_out.dtype),
                crs=da_out.raster.crs,
                transform=da_out.raster.transform,
                nodata=nodata,
                **profile_kwargs,
            )
            with rasterio.open(raster_path, "w", **profile) as dst:
                if windowed:
                    window_iter = dst.block_windows(1)
                else:
                    window_iter = [(None, None)]
                for _, window in window_iter:
                    if window is not None:
                        row_slice, col_slice = window.toslices()
                        sel = {self.x_dim: col_slice, self.y_dim: row_slice}
                        data = da_out.isel(sel).load().values
                    else:
                        data = da_out.load().values
                    if data.ndim == 2:
                        dst.write(data, 1, window=window)
                    else:
                        dst.write(data, window=window)
                if tags is not None:
                    dst.update_tags(**tags)

    def vectorize(self, connectivity=8):
        """Return geometry of grouped pixels with the same value in a DataArray object.

        Arguments
        ---------
        connectivity : int, optional
            Use 4 or 8 pixel connectivity for grouping pixels into features, by default 8

        Returns
        -------
        gdf : geopandas.GeoDataFrame
            Geometry of grouped pixels.
        """
        data = self._obj.values
        data_isnan = True if self.nodata is None else np.isnan(self.nodata)
        mask = ~np.isnan(data) if data_isnan else data != self.nodata
        feats_gen = features.shapes(
            data,
            mask=mask,
            transform=self.transform,
            connectivity=connectivity,
        )
        feats = [
            {"geometry": geom, "properties": {"value": idx}}
            for geom, idx in list(feats_gen)
        ]
        if len(feats) == 0:  # return empty GeoDataFrame
            return gpd.GeoDataFrame()
        crs = self.crs
        if crs is None and crs.to_epsg() is not None:
            crs = crs.to_epsg()  # not all CRS have an EPSG code
        gdf = gpd.GeoDataFrame.from_features(feats, crs=crs)
        gdf.index = gdf.index.astype(self._obj.dtype)
        return gdf


@xr.register_dataset_accessor("raster")
class RasterDataset(XRasterBase):
    """This is the GIS extension for :class:`xarray.Dataset`"""

    @property
    def vars(self):
        """list: Returns non-coordinate varibles"""
        return list(self._obj.data_vars.keys())

    def mask_nodata(self):
        """Mask nodata values with np.nan.
        Note this will change integer dtypes to float.
        """
        ds_out = self._obj
        for var in self.vars:
            ds_out[var] = ds_out[var].raster.mask_nodata()
        return ds_out

    def mask(self, mask):
        """Mask cells where mask equals False with the data nodata value.
        A warning is raised if no the data has no nodata value."""
        ds_out = self._obj
        for var in self.vars:
            ds_out[var] = ds_out[var].raster.mask(mask)
        return ds_out

    @staticmethod
    def from_numpy(data_vars, transform, attrs=None, crs=None):
        """Transform multiple numpy arrays to a Dataset object.
        The arrays should have identical shape.

        Arguments
        ---------
        data_vars: - dict-like
            A mapping from variable names to numpy arrays. The following notations
            are accepted:

            * {var_name: array-like}
            * {var_name: (array-like, nodata)}
            * {var_name: (array-like, nodata, attrs)}
        transform : affine transform
            Two dimensional affine transform for 2D linear mapping
        attrs : dict, optional
            additional global attributes
        crs: int, dict, or str, optional
            Coordinate Reference System. Accepts EPSG codes (int or str); proj (str or dict)

        Returns
        -------
        ds : xr.Dataset
            Dataset of data_vars arrays
        """
        da_lst = list()
        for i, (name, data) in enumerate(data_vars.items()):
            args = ()
            if isinstance(data, tuple):
                data, args = data[0], data[1:]
            da = RasterDataArray.from_numpy(data, transform, *args)
            da.name = name
            if i > 0 and da.shape[-2:] != da_lst[0].shape[-2:]:
                raise xr.MergeError(f"Data shapes do not match.")
            da_lst.append(da)
        ds = xr.merge(da_lst)
        if attrs is not None:
            ds.attrs.update(attrs)
        if crs is not None:
            ds.raster.set_crs(input_crs=crs)
            ds = ds.raster.reset_spatial_dims_attrs()
        return ds

    def reproject(
        self,
        dst_crs=None,
        dst_res=None,
        dst_transform=None,
        dst_width=None,
        dst_height=None,
        method="nearest",
        align=False,
    ):
        """Reproject a Dataset object, powered by :py:meth:`rasterio.warp.reproject`.

        Arguments
        ----------
        dst_crs: int, dict, or str, optional
            Target CRS. Accepts EPSG codes (int or str); proj (str or dict) or wkt (str)
            "utm" is accepted and will return the centroid utm zone CRS
        dst_res: tuple (x resolution, y resolution) or float, optional
            Target resolution, in units of target CRS.
        dst_transform: affine.Affine(), optional
            Target affine transformation. Will be calculated if None.
        dst_width, dst_height: int, optional
            Output file size in pixels and lines. Cannot be used together with
            resolution (dst_res).
        method: str, optional
            See :py:meth:`rasterio.warp.reproject` for existing methods, by default nearest.
            Additionally "nearest_index" can be used for KDTree based downsampling.
        align: boolean, optional
            If True, align target transform to resolution

        Returns
        --------
        ds_out : xarray.Dataset
            A reprojected Dataset.
        """
        reproj_kwargs = dict(
            dst_crs=dst_crs,
            dst_res=dst_res,
            dst_transform=dst_transform,
            dst_width=dst_width,
            dst_height=dst_height,
            align=align,
        )
        if isinstance(method, str) and method == "nearest_index":
            index = self.nearest_index(**reproj_kwargs)  # reuse same index !
            ds = self.reindex2d(index)
        else:
            if isinstance(method, str):
                method = {var: method for var in self.vars}
            elif not isinstance(method, dict):
                raise ValueError("Method should be a dictionary mapping or string.")
            ds = xr.Dataset(attrs=self._obj.attrs)
            for var in method:
                ds[var] = self._obj[var].raster.reproject(
                    method=method[var], **reproj_kwargs
                )
        return ds

    def reproject_like(self, other, method="nearest"):
        """Reproject a Dataset object to match the resolution, projection,
        and region of ``other``.

        Arguments
        ----------
        other: :xarray.DataArray of Dataset
            DataArray of the target resolution and projection.
        method: dict, optional
            Reproject method mapping. If a string is provided all variables are
            reprojecte with the same method. See
            :py:meth:`~hydromt.raster.RasterDataArray.reproject` for existing methods,
            by default nearest.

        Returns
        --------
        ds_out : xarray.Dataset
            Reprojected Dataset
        """
        ds = self._obj
        if self.aligned_grid(other):
            ds = self.clip_bbox(other.raster.bounds)
        elif not self.identical_grid(other):
            dst_bbox = other.raster.transform_bounds(self.crs)
            ds = self.clip_bbox(dst_bbox, buffer=2).raster.reproject(
                dst_crs=other.raster.crs,
                dst_transform=other.raster.transform,
                dst_width=other.raster.width,
                dst_height=other.raster.height,
                method=method,
            )
        if (
            ds.raster.x_dim != other.raster.x_dim
            or ds.raster.y_dim != other.raster.y_dim
        ):
            # overwrite spatial dimension names which might have been changed
            rm = {
                ds.raster.x_dim: other.raster.x_dim,
                ds.raster.y_dim: other.raster.y_dim,
            }
            ds = ds.rename(rm)
            ds.raster.set_spatial_dims(
                x_dim=other.raster.x_dim, y_dim=other.raster.y_dim
            )
        # make sure coordinates are identical!
        ds[other.raster.x_dim] = other.raster.xcoords
        ds[other.raster.y_dim] = other.raster.ycoords
        return ds

    def reindex2d(self, index):
        """Return reprojected Dataset object based on simple reindexing using
        linear indices in ``index``, which can be calculated with
        :py:meth:`~hydromt.raster.RasterDataArray.nearest_index`.

        Arguments
        ----------
        index: xarray.DataArray of intp
            DataArray with flat indices of source DataArray

        Returns
        --------
        ds_out : xarray.Dataset
            The reindexed dataset
        """
        ds_out = xr.Dataset(attrs=self._obj.attrs)
        for var in self.vars:
            ds_out[var] = self._obj[var].raster.reindex2d(index=index)
        return ds_out

    def to_mapstack(
        self,
        root,
        driver="GTiff",
        dtype=None,
        tags=None,
        windowed=False,
        mask=False,
        prefix="",
        postfix="",
        pcr_vs_map=gis_utils.PCR_VS_MAP,
        logger=logger,
        **profile_kwargs,
    ):
        """Write the Dataset object to one gdal-writable raster files per variable.
        The files are written to the ``root`` directory using the following filename
        ``<prefix><variable_name><postfix>.<ext>``.

        Arguments
        ----------
        root : str
            The path to output the raster to. It is created if it does not yet exist.
        driver : str, optional
            The name of the GDAL/rasterio driver to use to export the raster.
            Default is "GTiff".
        dtype : str, optional
            The data type to write the raster to. Default is the datasets dtype.
        tags : dict, optional
            A dictionary of tags to write to the raster.
        windowed : bool, optional
            If True, it will write using the windows of the output raster.
            Default is False.
        prefix : str, optional
            Prefix to filenames in mapstack
        postfix : str, optional
            Postfix to filenames in mapstack
        pcr_vs_map : dict, optional
            Only for PCRaster driver: <variable name> : <PCRaster type> key-value pairs
            e.g.: {'dem': 'scalar'}, see https://www.gdal.org/frmt_various.html#PCRaster
        **profile_kwargs
            Additional keyword arguments to pass into writing the raster. The
            nodata, transform, crs, count, width, and height attributes
            are ignored.

        """
        if driver not in gis_utils.GDAL_EXT_CODE_MAP:
            raise ValueError(f"Extension unknown for driver: {driver}")
        ext = gis_utils.GDAL_EXT_CODE_MAP.get(driver)
        if not isdir(root):
            os.makedirs(root)
        with tempfile.TemporaryDirectory() as tmpdir:
            if driver == "PCRaster" and HAS_PCRASTER:
                clone_path = gis_utils.write_clone(
                    tmpdir,
                    gdal_transform=self.transform.to_gdal(),
                    wkt_projection=None if self.crs is None else self.crs.to_wkt(),
                    shape=self.shape,
                )
                profile_kwargs.update({"clone_path": clone_path})
            for var in self.vars:
                if "/" in var:
                    # variables with in subfolders
                    folders = "/".join(var.split("/")[:-1])
                    if not isdir(join(root, folders)):
                        os.makedirs(join(root, folders))
                    var0 = var.split("/")[-1]
                    raster_path = join(root, folders, f"{prefix}{var0}{postfix}.{ext}")
                else:
                    raster_path = join(root, f"{prefix}{var}{postfix}.{ext}")
                if driver == "PCRaster":
                    profile_kwargs.update({"pcr_vs": pcr_vs_map.get(var, "scalar")})
                self._obj[var].raster.to_raster(
                    raster_path,
                    driver=driver,
                    dtype=dtype,
                    tags=tags,
                    windowed=windowed,
                    mask=mask,
                    logger=logger,
                    **profile_kwargs,
                )
