#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""hydrological methods powered by pyFlwDir"""

import warnings
import logging
import numpy as np
import xarray as xr
from rasterio.transform import from_origin
import geopandas as gpd
import pyflwdir

from . import gis_utils

logger = logging.getLogger(__name__)

__all__ = [
    "flwdir_from_da",
    "d8_from_dem",
    "reproject_hydrography_like",
    "stream_map",
    "basin_map",
    "outlet_map",
    "clip_basins",
]

### FLWDIR METHODS ###


def flwdir_from_da(da, ftype="infer", check_ftype=True, mask=None, logger=logger):
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
    if crs is not None and crs.is_geographic:
        latlon = True
        _crs = "geographic CRS with unit degree"
    elif crs is None or da.raster.crs.is_projected:
        _crs = "projected CRS with unit meter"
    logger.debug(f"Initializing flwdir with {_crs}.")
    if isinstance(mask, xr.DataArray):
        mask = mask.values
    elif isinstance(mask, bool) and mask and "mask" in da.coords:
        # backwards compatibility for mask = True
        mask = da["mask"].values
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


def d8_from_dem(da_elv, gdf_stream=None, max_depth=-1.0, outlets="edge"):
    """Derive D8 flow directions grid from an elevation grid.

    Outlets occur at the edge of the data or at the interface with nodata values.
    A local depressions is filled based on its lowest pour point level if the pour point
    depth is smaller than the maximum pour point depth `max_depth`, otherwise the lowest
    elevation in the depression becomes a pit.

    Based on: Wang, L., & Liu, H. (2006). https://doi.org/10.1080/13658810500433453

    Parameters
    ----------
    da_elv: 2D xarray.DataArray
        elevation raster
    gdf_stream: geopandas.GeoDataArray, optional
        stream vector layer with 'uparea' [km2] column which is used to burn
        the river in the elevation data.
    max_depth: float, optional
        Maximum pour point depth. Depressions with a larger pour point
        depth are set as pit. A negative value (default) equals an infitely
        large pour point depth causing all depressions to be filled.
    outlets: {'edge', 'min'}
        Position for basin outlet(s) at the all valid elevation edge cell ('edge')
        or only the minimum elevation edge cell ('min')

    Returns
    -------
    da_flw: 2D xarray.DataArray
        D8 flow direction data
    """
    nodata = da_elv.raster.nodata
    crs = da_elv.raster.crs
    assert da_elv.raster.res[1] < 0
    assert nodata is not None and ~np.isnan(nodata)
    # burn in river if
    if gdf_stream is not None and "uparea" in gdf_stream.columns:
        gdf_stream = gdf_stream.sort_values(by="uparea")
        dst_rivupa = da_elv.raster.rasterize(gdf_stream, col_name="uparea", nodata=0)
        # make sure the rivers have a slope and are below all other elevation cells.
        # river elevation = min(elv) - log10(uparea[m2]) from rasterized river uparea.
        elvmin = da_elv.where(da_elv != nodata).min()
        elvriv = elvmin - np.log10(np.maximum(1.0, dst_rivupa * 1e3))
        # synthetic elevation with river burned in
        da_elv = elvriv.where(np.logical_and(da_elv != nodata, dst_rivupa > 0), da_elv)
        da_elv.raster.set_nodata(nodata)
        da_elv.raster.set_crs(crs)
    # derive new flow directions from (synthetic) elevation
    d8 = pyflwdir.dem.fill_depressions(
        da_elv.values.astype(np.float32),
        max_depth=max_depth,
        nodata=da_elv.raster.nodata,
        outlets=outlets,
    )[1]
    # return xarray data array
    da_flw = xr.DataArray(
        dims=da_elv.raster.dims,
        coords=da_elv.raster.coords,
        data=d8,
        name="flwdir",
    )
    da_flw.raster.set_nodata(247)
    da_flw.raster.set_crs(crs)
    return da_flw


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

    Note: This method only works for D8 data.

    # TODO add doi
    Based on: Eilander et al. (2021).

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


def reproject_hydrography_like(
    ds_hydro,
    da_elv,
    river_upa=5,
    method="bilinear",
    uparea_name="uparea",
    flwdir_name="flwdir",
    logger=logger,
):
    """Reproject flow direction and upstream area data to the `da_elv` crs and grid.
    Note that the resolution of `da_elv` and `ds_hydro` should be similar or smaller
    for good results.

    The reprojection is based on a synthetic elevation grid defined by the
    destination elevation minus the reprojected log10 upstream area [km2] grids.
    Additionally, rivers are vectorized and burned into the synthetic elevation grid
    for better results.

    If not all upstream area is inlcuded, the upstream area of rivers entering the
    domain will be used as boundary conditions to the reprojected upsteam area raster.

    NOTE: this method is still experimental and might change in the future!

    Parameters
    ----------
    ds_hydro: xarray.Dataset
        Dataset with gridded flow directions named `flwdir_name` and upstream area
        named `uparea_name` [km2].
    da_elv: xarray.DataArray
        DataArray with elevation on destination grid.
    river_upa: float, optional
        Minimum upstream area threshold for rivers, by default 5 [km2]
    method:
        Interpolation method for the upstream area grid.

    Returns
    -------
    xarray.Dataset
        Reprojected flow direction and upstream area grids.
    """
    # TODO fix for case without ds_hydro, but with gdf_stream
    # check N->S orientation
    assert da_elv.raster.res[1] < 0
    assert ds_hydro.raster.res[1] < 0
    crs = da_elv.raster.crs
    bbox = np.asarray(da_elv.raster.bounds)
    # pad da_elv to avoid boundary problems
    buf0 = 2
    nrow, ncol = da_elv.raster.shape
    t = da_elv.raster.transform
    dst_transform = from_origin(
        t[2] - buf0 * t[0], t[5] + buf0 * abs(t[4]), t[0], abs(t[4])
    )
    dst_shape = nrow + buf0 * 2, ncol + buf0 * 2
    xcoords, ycoords = gis_utils.affine_to_coords(dst_transform, dst_shape)
    da_elv = xr.DataArray(
        dims=da_elv.raster.dims,
        coords={da_elv.raster.x_dim: xcoords, da_elv.raster.y_dim: ycoords},
        data=np.pad(da_elv.values, buf0, "edge"),
        attrs=da_elv.attrs,
    )
    da_elv.raster.set_crs(crs)
    # reproject uparea & elevation with buffer
    da_upa = ds_hydro[uparea_name].raster.reproject_like(da_elv, method=method)
    max_upa = da_upa.where(da_upa != da_upa.raster.nodata).max().values
    nodata = da_elv.raster.nodata
    # vectorize and reproject river uparea
    mask = ds_hydro[uparea_name] > river_upa
    flwdir_src = flwdir_from_da(ds_hydro[flwdir_name], mask=mask)
    feats = flwdir_src.vectorize(uparea=ds_hydro[uparea_name].values)
    gdf_stream = gpd.GeoDataFrame.from_features(feats, crs=ds_hydro.raster.crs)
    gdf_stream = gdf_stream.sort_values(by="uparea")
    # only area with upa otherwise the outflows are not resolved!
    # synthetic elevation -> reprojected elevation - log10(reprojected uparea[m2])
    elvsyn = xr.where(
        np.logical_and(da_elv != nodata, da_upa != da_upa.raster.nodata),
        da_elv - np.log10(np.maximum(1.0, da_upa * 1e3)),
        nodata,
    )
    elvsyn.raster.set_nodata(nodata)
    elvsyn.raster.set_crs(crs)
    # get flow directions
    da_flw = d8_from_dem(elvsyn, gdf_stream).raster.clip_bbox(bbox)
    # calculate upstream area with uparea from rivers at edge
    flwdir = flwdir_from_da(da_flw, ftype="d8")
    da_flw.data = flwdir.to_array()  # to set outflow pits after clip
    area = flwdir.area / 1e6  # area [km2]
    # get inflow cells: headwater river cells at edge
    rivupa = da_flw.raster.rasterize(gdf_stream, col_name="uparea", nodata=0)
    _edge = pyflwdir.gis_utils.get_edge(da_flw.values == 247)
    headwater = np.logical_and(
        rivupa.values > 0, flwdir.upstream_sum(rivupa.values > 0) == 0
    )
    inflow_idxs = np.where(np.logical_and(headwater, _edge).ravel())[0]
    if inflow_idxs.size > 0:
        # use nearest mapping to avoid duplicating uparea when reprojecting to higher res.
        gdf0 = gpd.GeoDataFrame(
            index=inflow_idxs,
            geometry=gpd.points_from_xy(*flwdir.xy(inflow_idxs)),
            crs=crs,
        )
        gdf0["idx2"], gdf0["dst2"] = gis_utils.nearest(gdf0, gdf_stream)
        gdf0 = gdf0.sort_values(by="dst2").drop_duplicates("idx2")
        gdf0["uparea"] = gdf_stream.loc[gdf0["idx2"].values, "uparea"].values
        # set stream uparea to selected inflow cells and calculate total uparea
        area.flat[gdf0.index.values] = gdf0["uparea"].values
        logger.info(
            f"Calculating upstream area with {gdf0.index.size} input cell at the domain edge."
        )
    da_upa = xr.DataArray(
        dims=da_flw.raster.dims,
        coords=da_flw.raster.coords,
        data=flwdir.accuflux(area).astype(np.float32),
        name="uparea",
    )
    da_upa.raster.set_nodata(-9999)
    da_upa.raster.set_crs(crs)
    max_upa1 = da_upa.max().values
    logger.info(
        f"Reprojected maximum upstream area: {max_upa1:.2f} km2 ({max_upa:.2f} km2)"
    )
    return xr.merge([da_flw, da_upa])


### hydrography maps ###


def gaugemap(ds, idxs=None, xy=None, ids=None, mask=None, flwdir=None, logger=logger):
    """This method has been deprecated. See :py:meth:`~hydromt.flw.gauge_map`"""
    warnings.warn(
        'The "gaugemap" method has been deprecated, use  "gauge_map" instead.',
        DeprecationWarning,
    )
    return gauge_map(
        ds=ds,
        idxs=idxs,
        xy=xy,
        ids=ids,
        stream=mask,
        flwdir=flwdir,
        logger=logger,
    )


def gauge_map(ds, idxs=None, xy=None, ids=None, mask=None, flwdir=None, logger=logger):
    """Return map with unique gauge IDs. Initial gauge locations are snapped to the
    nearest downstream river defined by mask if both `flwdir` and `mask` are provided.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset containing flow direction data
    idxs : 1D array or int, optional
        linear indices of gauges, by default is None.
    xy : tuple of 1D array of float, optional
        x, y coordinates of gauges, by default is None.
    outlets : bool, optional
        If True and xy and idxs are None, the basin map is derived for basin outlets
        only, excluding pits at the edge of the domain of incomplete basins.
    ids : 1D array of int32, optional
        IDs of gauges, values must be larger than zero, by default None.
    flwdir : pyflwdir.FlwdirRaster
        Flow direction raster object
    stream: 2D array of bool, optional
        Mask of stream cells used to snap gauges to, by default None

    Returns
    -------
    xarray.DataArray
        Map with unique gauge IDs
    """
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
        Initial mask of stream cells. If a stream if provided, it is combined with the
        threshold based map using a logical AND operation..
    stream_kwargs : dict, optional
        Parameter-treshold pairs to define streams. Multiple threshold will be combined
        using a logical AND operation.

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
    """Return a (sub)basin map, with unique non-zero IDs for each subbasin.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset used for output grid definition and containing `stream_kwargs` variables.
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
    """This method  will be deprecated. Use :py:meth:`~hydromt.flw.basin_map` in combination
    with :py:meth:`~hydromt.raster.RasterDataArray.vectorize` instead.
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
