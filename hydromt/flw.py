#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""hydrological methods powered by pyFlwDir"""

import os
from os.path import join, isdir, dirname, basename, isfile
import warnings
import logging
import numpy as np
import xarray as xr
from rasterio.enums import Resampling
import pyflwdir
from pyflwdir import FlwdirRaster

from . import gis_utils

logger = logging.getLogger(__name__)


def flwdir_from_da(da, ftype="infer", check_ftype=True, mask=None):
    """Parse dataarray to flow direction raster object. If a mask coordinate is present
    this will be passed on the the pyflwdir.from_array method.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray containing flow direction raster
    ftype : {'d8', 'ldd', 'nextxy', 'nextidx', 'infer'}, optional
        name of flow direction type, infer from data if 'infer', by default is 'infer'
    check_ftype : bool, optional
        check if valid flow direction raster if ftype is not 'infer', by default True
    mask : xr.DataArray of bool, optional
        Mask for gridded flow direction data, by default None.

    Returns
    -------
    flwdir : pyflwdir.FlwdirRaster
        Flow direction raster object
    """
    if not isinstance(da, xr.DataArray):
        raise TypeError("da should be instance xarray.DataArray type")

    crs = da.raster.crs
    latlon = False
    if crs is not None and crs.to_epsg() == 4326:
        latlon = True
    elif crs is None or da.raster.crs.is_projected:
        warnings.warn("Assuming a projected CRS with unit meter.")
    elif crs is not None and crs.is_geographic:
        raise NotImplementedError("unknown geographic CRS unit")
    if isinstance(mask, xr.DataArray):
        mask = mask.values
    elif isinstance(mask, bool) and mask and "mask" in da.coords:
        # backwards compatibility for mask = True
        mask = da["mask"].Values
    elif not isinstance(mask, np.ndarray):
        mask = None
    flwdir = pyflwdir.from_array(
        data=da.squeeze().values,
        ftype=ftype,
        check_ftype=check_ftype,
        mask=mask,
        transform=da.raster.transform,
        latlon=latlon,
    )
    return flwdir


def gaugemap(ds, idxs=None, xy=None, ids=None, mask=None, flwdir=None, logger=logger):
    # Snap if mask and flwdir are not None
    if xy is not None:
        idxs = ds.raster.xy_to_idx(xs=xy[0], ys=xy[1])
    elif idxs is None:
        raise ValueError("Either idxs or xy required")
    if ids is None:
        ids = np.arange(1, idxs.size + 1, dtype=np.int32)
    # Snapping
    # TODO: should we do the snapping similar to basin_map ??
    if mask is not None and flwdir is not None:
        idxs, dist = flwdir.snap(idxs=idxs, mask=mask, unit="m")
        if np.any(dist > 10000):
            far = len(dist[dist > 10000])
            logger.warn(f"Snapping distance of {far} gauge(s) is > 10km")
    gauges = np.zeros(ds.raster.shape, dtype=np.int32)
    gauges.flat[idxs] = ids
    da_gauges = xr.DataArray(
        dims=ds.raster.dims,
        coords=ds.raster.coords,
        data=gauges,
        attrs=dict(_FillValue=0),
    )
    return da_gauges, idxs, ids


def outlet_map(da_flw, ftype="infer"):
    """Returns a mask of basin outlets/pits from a flow direction raster.

    Parameters
    ----------
    da_flw: xr.DataArray
        Flow direction data array
    ftype : {'d8', 'ldd', 'nextxy', 'nextidx', 'infer'}, optional
        name of flow direction type, infer from data if 'infer', by default is 'infer'

    Returns
    -------
    da_basin : xarray.DataArray of int32
        basin ID map
    """
    if ftype == "infer":
        ftype = pyflwdir.pyflwdir._infer_ftype(da_flw.values)
    elif ftype not in pyflwdir.pyflwdir.FTYPES:
        raise ValueError(f"Unknown pyflwdir ftype: {ftype}")
    pit_values = pyflwdir.pyflwdir.FTYPES[ftype]._pv
    mask = np.isin(da_flw.values, pit_values)
    return xr.DataArray(mask, dims=da_flw.raster.dims, coords=da_flw.raster.coords)


def stream_map(ds, stream=None, **stream_kwargs):
    """Return a stream mask DataArray

    Parameters
    ----------
    ds : xarray.Dataset
        dataset containing flow direction data
    stream: 2D array of bool, optional
        Initial mask of stream cells used to snap outlets to, by default None
    stream_kwargs : dict, optional
        Parameter-treshold pairs to define streams. Multiple threshold will be combined
        using a logical_and operation. If a stream if provided, it is combined with the
        threhold based map as well.

    Returns
    -------
    stream : xarray.DataArray of bool
        stream mask
    """
    if stream is None or isinstance(stream, np.ndarray):
        data = np.full(ds.raster.shape, True, dtype=bool) if stream is None else stream
        stream = xr.DataArray(
            coords=ds.raster.coords, dims=ds.raster.dims, data=data, name="mask"
        )  # all True
    for name, value in stream_kwargs.items():
        stream = stream.where(
            np.logical_and(ds[name] != ds[name].raster.nodata, ds[name] >= value), False
        )
    if not np.any(stream):
        raise ValueError("Stream criteria resulted in invalid mask.")
    return stream


def basin_map(
    ds,
    flwdir,
    xy=None,
    idxs=None,
    outlets=False,
    ids=None,
    stream=None,
    **stream_kwargs,
):
    """Return a (sub)basin ID DataArray

    Parameters
    ----------
    ds : xarray.Dataset
        dataset containing flow direction data
    flwdir : pyflwdir.FlwdirRaster
        Flow direction raster object
    idxs : 1D array or int, optional
        linear indices of sub(basin) outlets, by default is None.
    xy : tuple of 1D array of float, optional
        x, y coordinates of sub(basin) outlets, by default is None.
    outlets : bool, optional
        If True and xy and idxs are None, the basin map is derived for basin outlets
        only, excluding pits at the edge of the domain of incomplete basins.
    ids : 1D array of int32, optional
        IDs of (sub)basins, must be larger than zero, by default None
    stream: 2D array of bool, optional
        Mask of stream cells used to snap outlets to, by default None
    stream_kwargs : dict, optional
        Parameter-treshold pairs to define streams. Multiple threshold will be combined
        using a logical_and operation. If a stream if provided, it is combined with the
        threhold based map as well.

    Returns
    -------
    da_basin : xarray.DataArray of int32
        basin ID map
    xy : tuple of array_like of float
        snapped x, y coordinates of sub(basin) outlets
    """
    if not np.all(flwdir.shape == ds.raster.shape):
        raise ValueError("flwdir and ds dimensions do not match")
    # get stream map
    locs = xy is not None or idxs is not None
    if locs and (stream is not None or len(stream_kwargs) > 0):
        # snap provided xy/idxs to streams
        stream = stream_map(ds, stream=stream, **stream_kwargs)
        idxs = flwdir.snap(xy=xy, idxs=idxs, mask=stream.values)[0]
        xy = None
    elif not locs and outlets:
        # get idxs from real outlets excluding pits at the domain edge
        idxs = flwdir.idxs_outlet
        if idxs is None or len(idxs) == 0:
            raise ValueError(
                "No outlets found in domain. "
                "Provide 'xy' or 'idxs' outlet locations or set 'outlets' to False."
            )
        ids = None
    da_basins = xr.DataArray(
        data=flwdir.basins(idxs=idxs, xy=xy, ids=ids).astype(np.int32),
        dims=ds.raster.dims,
        coords=ds.raster.coords,
    )
    da_basins.raster.set_nodata(0)
    if idxs is not None:
        xy = flwdir.xy(idxs)
    return da_basins, xy


def basin_shape(ds, flwdir, basin_name="basins", mask=True, **kwargs):
    """Return a shape of the basin.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset containing flow direction data
    flwdir : pyflwdir.FlwdirRaster
        Flow direction raster object
    basin_name : str, optional
        Name of data variable with basin array, by default "basins". If not found in
        ds it is derived on the fly.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with basin shapes.
    """
    if not np.all(flwdir.shape == ds.raster.shape):
        raise ValueError("flwdir and ds dimensions do not match")
    if basin_name not in ds:
        ds[basin_name] = basin_map(ds, flwdir, **kwargs)[0]
    da_basins = ds[basin_name]
    nodata = da_basins.raster.nodata
    if mask and "mask" in da_basins.coords and nodata is not None:
        da_basins = da_basins.where(da_basins.coords["mask"] != 0, nodata)
        da_basins.raster.set_nodata(nodata)
    gdf = da_basins.raster.vectorize().set_index("value").sort_index()
    gdf.index.name = basin_name
    return gdf


def clip_basins(ds, flwdir, xy, flwdir_name="flwdir", **stream_kwargs):
    """Clip a dataset to a subbasin.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset containing flow direction data
    flwdir : pyflwdir.FlwdirRaster
        Flow direction raster object
    xy : tuple of array_like of float
        x, y coordinates of (sub)basin outlet locations
    flwdir_name : str, optional
        name of flow direction DataArray, by default 'dir'
    stream_kwargs : key-word arguments
        name of variable in ds and threshold value

    Returns
    -------
    xarray.Dataset
        clipped dataset
    """
    da_basins, xy = basin_map(ds, flwdir, xy, **stream_kwargs)
    idxs_pit = flwdir.index(*xy)
    # set pit values in DataArray
    pit_value = flwdir._core._pv
    if isinstance(pit_value, np.ndarray):
        pit_value = pit_value[0]
    dir_arr = ds[flwdir_name].values.copy()
    dir_arr.flat[idxs_pit] = pit_value
    attrs = ds[flwdir_name].attrs.copy()
    ds[flwdir_name] = xr.Variable(dims=ds.raster.dims, data=dir_arr, attrs=attrs)
    # clip data
    ds.coords["mask"] = da_basins
    return ds.raster.clip_mask(da_basins)


def upscale_flwdir(
    ds,
    flwdir,
    scale_ratio,
    method="com2",
    uparea_name=None,
    flwdir_name="flwdir",
    logger=logger,
    **kwargs,
):
    """Upscale flow direction network to lower resolution and resample other data
    variables in dataset to the same resolution.

    Note: This method only works for D8 or LDD flow directon data.

    # TODO add refs

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset flow direction and auxiliry data data
    flwdir : pyflwdir.FlwdirRaster
        Flow direction raster object.
    scale_ratio: int
        Size of upscaled (coarse) grid cells.
    uparea_name : str, optional
        Name of upstream area DataArray, by default None
    flwdir_name : str, optional
        Name of upscaled flow direction raster DataArray, by default "flwdir"
    method : {'com2', 'com', 'eam', 'dmm'}
        Upscaling method for flow direction data, by default 'com2'.

    Returns
    -------
    ds_out = xarray.Dataset
        Upscaled Dataset
    flwdir_out : pyflwdir.FlwdirRaster
        Upscaled flow direction raster object.
    """
    if not np.all(flwdir.shape == ds.raster.shape):
        raise ValueError("Flwdir and ds dimensions do not match.")
    uparea = None
    if uparea_name is not None:
        if uparea_name in ds.data_vars:
            uparea = ds[uparea_name].values
        else:
            logger.warning(f'Upstream area map "{uparea_name}" not in dataset.')
    flwdir_out, idxs_out = flwdir.upscale(
        scale_ratio, method=method, uparea=uparea, **kwargs
    )
    # setup output DataArray
    ftype = flwdir.ftype
    dims = ds.raster.dims
    xs, ys = gis_utils.affine_to_coords(flwdir_out.transform, flwdir_out.shape)
    coords = {ds.raster.y_dim: ys, ds.raster.x_dim: xs}
    da_flwdir = xr.DataArray(
        name=flwdir_name,
        data=flwdir_out.to_array(ftype),
        coords=coords,
        dims=dims,
        attrs=dict(long_name=f"{ftype} flow direction", _FillValue=flwdir._core._mv),
    )
    # translate outlet indices to global x,y coordinates
    x_out, y_out = ds.raster.idx_to_xy(idxs_out, mask=idxs_out != flwdir._mv)
    da_flwdir.coords["x_out"] = xr.Variable(
        dims=dims,
        data=x_out,
        attrs=dict(long_name="subgrid outlet x coordinate", _FillValue=np.nan),
    )
    da_flwdir.coords["y_out"] = xr.Variable(
        dims=dims,
        data=y_out,
        attrs=dict(long_name="subgrid outlet y coordinate", _FillValue=np.nan),
    )
    # outlet indices
    da_flwdir.coords["idx_out"] = xr.DataArray(
        data=idxs_out,
        dims=dims,
        attrs=dict(long_name="subgrid outlet index", _FillValue=flwdir._mv),
    )
    return da_flwdir, flwdir_out
