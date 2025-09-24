#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""GIS related raster functions."""

from __future__ import annotations

import logging
from typing import Optional

import dask
import numpy as np
import rasterio
import xarray as xr
from pyflwdir import gis_utils as gis
from pyproj import CRS
from rasterio.transform import Affine

__all__ = [
    "_affine_to_coords",
    "_affine_to_meshgrid",
    "_cellarea",
    "_meridian_offset",
    "_reggrid_area",
    "cellres",
    "full",
    "full_from_transform",
    "full_like",
    "merge",
    "spread2d",
]

logger = logging.getLogger(__name__)

_R = 6371e3  # Radius of earth in m. Use 3956e3 for miles

# FULL


def full_like(
    other: xr.DataArray, *, nodata: float = None, lazy: bool = False
) -> xr.DataArray:
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
        shape=other.shape,
        dims=other.dims,
    )
    da.raster.set_attrs(**other.raster.attrs)
    da.raster._transform = other.raster.transform
    return da


def full(
    coords,
    *,
    nodata=np.nan,
    dtype=np.float32,
    name=None,
    attrs=None,
    crs=None,
    lazy=False,
    shape=None,
    dims=None,
) -> xr.DataArray:
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
    shape: tuple, optional
        Length along (dim0, y, x) dimensions, of which the first is optional.
    dims: tuple, optional
        Name(s) of the data dimension(s).

    Returns
    -------
    da: DataArray
        Filled DataArray
    """
    attrs = attrs or {}
    f = dask.array.empty if lazy else np.full
    if dims is None:
        dims = tuple([d for d in coords])
    if shape is None:
        cs = next(iter(coords.values()))  # get first coordinate
        if cs.ndim == 1:
            shape = tuple([coords[dim].size for dim in dims])
        else:  # rotated
            shape = cs.shape
            if hasattr(cs, "dims"):
                dims = cs.dims
    data = f(shape, nodata, dtype=dtype)
    da = xr.DataArray(data, coords, dims, name, attrs)
    da.raster.set_nodata(nodata)
    da.raster.set_crs(crs)
    return da


def full_from_transform(
    transform,
    shape,
    *,
    nodata=np.nan,
    dtype=np.float32,
    name=None,
    attrs=None,
    crs=None,
    lazy=False,
):
    """Return a full DataArray based on a geospatial transform and shape.

    See :py:meth:`~hydromt.raster_utils.full` for all options.

    Parameters
    ----------
    transform : affine transform
        Two dimensional affine transform for 2D linear mapping.
    shape : tuple of int
        Length along (dim0, y, x) dimensions, of which the first is optional.
    nodata : optional
        The nodata value to assign to the DataArray. Defaults to np.nan.
    dtype : optional
        The data type to use for the DataArray. Defaults to np.float32.
    name : optional
        The name of the DataArray. Defaults to None.
    attrs : optional
        Additional attributes to assign to the DataArray. Empty by default.
    crs : optional
        The coordinate reference system (CRS) of the DataArray. Defaults to None.
    lazy : bool, optional
        Whether to create a lazy DataArray. Defaults to False.

    Returns
    -------
    da : DataArray
        Filled DataArray
    """
    attrs = attrs or {}
    if len(shape) not in [2, 3]:
        raise ValueError("Only 2D and 3D data arrays supported.")
    if not isinstance(transform, Affine):
        transform = Affine(*transform)
    coords = _affine_to_coords(transform, shape[-2:], x_dim="x", y_dim="y")
    dims = ("y", "x")
    if len(shape) == 3:
        coords = {"dim0": ("dim0", np.arange(shape[0], dtype=int)), **coords}
        dims = ("dim0", "y", "x")
    da = full(
        coords=coords,
        nodata=nodata,
        dtype=dtype,
        name=name,
        attrs=attrs,
        crs=crs,
        lazy=lazy,
        shape=shape,
        dims=dims,
    )
    da.raster._transform = transform
    return da


# TRANSFORM


def _affine_to_coords(transform, shape, x_dim="x", y_dim="y"):
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


def _affine_to_meshgrid(transform, shape):
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


def _meridian_offset(ds, bbox=None):
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


## CELLAREAS
def _reggrid_area(lats, lons):
    """Return the cell area [m2] for a regular grid based on its cell centres lat, lon."""  # noqa: E501
    xres = np.abs(np.mean(np.diff(lons)))
    yres = np.abs(np.mean(np.diff(lats)))
    area = np.ones((lats.size, lons.size), dtype=lats.dtype)
    return _cellarea(lats, xres, yres)[:, None] * area


def _cellarea(lat, xres=1.0, yres=1.0):
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


## MERGE


def merge(
    data_arrays,
    dst_crs=None,
    dst_bounds=None,
    dst_res=None,
    align=True,
    mask=None,
    merge_method="first",
    **kwargs,
):
    """Merge multiple tiles to a single DataArray.

    If mismatching grid CRS or resolution, tiles are reprojected
    to match the output DataArray grid. If no destination
    grid is defined it is based on the first DataArray in the list.

    Based on :py:meth:`rasterio.merge.merge`.

    Arguments
    ---------
    data_arrays: list of xarray.DataArray
        Tiles to merge
    dst_crs: pyproj.CRS, int
        CRS (or EPSG code) of destination grid
    dst_bounds: list of float
        Bounding box [xmin, ymin, xmax, ymax] of destination grid
    dst_res: float
        Resolution of destination grid
    align : bool, optional
        If True, align grid with dst_res
    mask: geopands.GeoDataFrame, optional
        Mask of destination area of interest.
        Used to determine dst_crs, dst_bounds and dst_res if missing.
    merge_method: {'first','last','min','max','new'}, callable
        Merge method:

        * first: reverse painting
        * last: paint valid new on top of existing
        * min: pixel-wise min of existing and new
        * max: pixel-wise max of existing and new
        * new: assert no pixel overlap
    **kwargs:
        Key-word arguments passed to
        :py:meth:`~hydromt.raster.RasterDataArray.reproject`

    Returns
    -------
    da_out: xarray.DataArray
        Merged tiles.
    """
    # define merge method
    if merge_method == "first":

        def copyto(old_data, new_data, old_nodata, new_nodata):
            dmask = np.logical_and(old_nodata, ~new_nodata)
            old_data[dmask] = new_data[dmask]

    elif merge_method == "last":

        def copyto(old_data, new_data, old_nodata, new_nodata):
            dmask = ~new_nodata
            old_data[dmask] = new_data[dmask]

    elif merge_method == "min":

        def copyto(old_data, new_data, old_nodata, new_nodata):
            dmask = np.logical_and(~old_nodata, ~new_nodata)
            old_data[dmask] = np.minimum(old_data[dmask], new_data[dmask])
            dmask = np.logical_and(old_nodata, ~new_nodata)
            old_data[dmask] = new_data[dmask]

    elif merge_method == "max":

        def copyto(old_data, new_data, old_nodata, new_nodata):
            dmask = np.logical_and(~old_nodata, ~new_nodata)
            old_data[dmask] = np.maximum(old_data[dmask], new_data[dmask])
            dmask = np.logical_and(old_nodata, ~new_nodata)
            old_data[dmask] = new_data[dmask]

    elif merge_method == "new":

        def copyto(old_data, new_data, old_nodata, new_nodata):
            data_overlap = np.logical_and(~old_nodata, ~new_nodata)
            assert not np.any(data_overlap)
            mask = ~new_nodata
            old_data[mask] = new_data[mask]

    elif callable(merge_method):
        copyto = merge_method

    # resolve arguments
    if data_arrays[0].ndim != 2:
        raise ValueError("Mosaic is only implemented for 2D DataArrays.")
    if dst_crs is None and (dst_bounds is not None or dst_res is not None):
        raise ValueError("dst_bounds and dst_res not understood without dst_crs.")
    idx0 = 0
    if mask is not None and (dst_res is None or dst_crs is None):
        # select array with largest overlap of valid cells if mask
        areas = []
        for da in data_arrays:
            da_clipped = da.raster.clip_geom(geom=mask, mask=True)
            n = da_clipped.raster.mask_nodata().notnull().sum().load().item()
            areas.append(n)
        idx0 = np.argmax(areas)
        # if single layer with overlap of valid cells: return clip
        if np.sum(np.array(areas) > 0) == 1:
            return data_arrays[idx0].raster.clip_geom(mask)
    da0 = data_arrays[idx0]  # used for default dst_crs and dst_res
    # dst CRS
    if dst_crs is None:
        dst_crs = da0.raster.crs
    else:
        dst_crs = CRS.from_user_input(dst_crs)
    # dst res
    if dst_res is None:
        dst_res = da0.raster.res
    if isinstance(dst_res, float):
        x_res, y_res = dst_res, -dst_res
    elif isinstance(dst_res, (tuple, list)) and len(dst_res) == 2:
        x_res, y_res = dst_res
    else:
        raise ValueError("dst_res not understood.")
    assert x_res > 0
    assert y_res < 0
    dst_res = (x_res, -y_res)  # NOTE: y_res is multiplied with -1 in rasterio!
    # dst bounds
    if dst_bounds is None and mask is not None:
        w, s, e, n = mask.to_crs(dst_crs).buffer(2 * abs(x_res)).total_bounds
        # align with da0 grid
        w0, s0, e0, n0 = da0.raster.bounds
        w = w0 + np.round((w - w0) / abs(x_res)) * abs(x_res)
        n = n0 + np.round((n - n0) / abs(y_res)) * abs(y_res)
        e = w + int(round((e - w) / abs(x_res))) * abs(x_res)
        s = n - int(round((n - s) / abs(y_res))) * abs(y_res)
        align = False  # don't align with resolution
    elif dst_bounds is None:
        for i, da in enumerate(data_arrays):
            if i == 0:
                w, s, e, n = da.raster.transform_bounds(dst_crs)
            else:
                w1, s1, e1, n1 = da.raster.transform_bounds(dst_crs)
                w = min(w, w1)
                s = min(s, s1)
                e = max(e, e1)
                n = max(n, n1)
    else:
        w, s, e, n = dst_bounds
    # check N>S orientation
    top, bottom = (n, s) if y_res < 0 else (s, n)
    dst_bounds = w, bottom, e, top
    # dst transform
    width = int(round((e - w) / abs(x_res)))
    height = int(round((n - s) / abs(y_res)))
    transform = rasterio.transform.from_bounds(*dst_bounds, width, height)
    # align with dst_res
    if align:
        transform, width, height = rasterio.warp.aligned_target(
            transform, width, height, dst_res
        )

    # creat output array
    nodata = da0.raster.nodata
    isnan = np.isnan(nodata)
    dtype = da0.dtype
    da_out = full_from_transform(
        transform=transform,
        shape=(height, width),
        nodata=nodata,
        dtype=dtype,
        name=da0.name,
        attrs=da0.attrs,
        crs=dst_crs,
    )
    ys = da_out.raster.ycoords.values
    xs = da_out.raster.xcoords.values
    dest = da_out.values
    kwargs.update(dst_crs=dst_crs, dst_res=dst_res, align=True)
    sf_lst = []
    for _, da in enumerate(data_arrays):
        if not da.raster.aligned_grid(da_out):
            # clip with buffer
            src_bbox = da_out.raster.transform_bounds(da.raster.crs)
            da = da.raster.clip_bbox(src_bbox, buffer=10)
            if np.any([da[dim].size == 0 for dim in da.raster.dims]):
                continue  # out of bounds
            # reproject
            da = da.raster.reproject(**kwargs)
        # clip to dst_bounds
        da = da.raster.clip_bbox(dst_bounds)
        if np.any([da[dim].size == 0 for dim in da.raster.dims]):
            continue  # out of bounds
        if "source_file" in da.attrs:
            sf_lst.append(da.attrs["source_file"])
        # merge overlap
        w0, s0, e0, n0 = da.raster.bounds
        if y_res < 0:
            top = np.nonzero(ys <= n0)[0][0] if n0 < ys[0] else 0
            bottom = np.nonzero(ys < s0)[0][0] if s0 > ys[-1] else None
        else:
            top = np.nonzero(ys > n0)[0][0] if n0 < ys[-1] else 0
            bottom = np.nonzero(ys >= s0)[0][0] if s0 > ys[0] else None
        left = np.nonzero(xs >= w0)[0][0] if w0 > xs[0] else 0
        right = np.nonzero(xs > e0)[0][0] if e0 < xs[-1] else None
        y_slice = slice(top, bottom)
        x_slice = slice(left, right)
        region = dest[y_slice, x_slice]
        temp = np.ma.masked_equal(da.values.astype(dtype), nodata)
        if isnan:
            region_nodata = np.isnan(region)
            temp_nodata = np.isnan(temp)
        else:
            region_nodata = region == nodata
            temp_nodata = temp.mask
        copyto(region, temp, region_nodata, temp_nodata)

    # set attrs
    da_out.attrs.update(source_file="; ".join(sf_lst))
    da_out.raster.set_crs(dst_crs)
    return da_out.raster.reset_spatial_dims_attrs()
