#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""hydrological methods powered by pyFlwDir"""

import warnings
import logging
import numpy as np
import xarray as xr
import geopandas as gpd
import pyflwdir
from typing import Tuple, Union, Optional
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
    "upscale_flwdir",
]

### FLWDIR METHODS ###


def flwdir_from_da(
    da: xr.DataArray,
    ftype: str = "infer",
    check_ftype: bool = True,
    mask: Union[xr.DataArray, bool, None] = None,
    logger=logger,
):
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
    mask : xr.DataArray, bool, optional
        Mask for gridded flow direction data, by default None.
        If True, use the mask coordinate of `da`.

    Returns
    -------
    flwdir : pyflwdir.FlwdirRaster
        Flow direction raster object
    """
    if not isinstance(da, xr.DataArray):
        raise TypeError("da should be an instance of xarray.DataArray")
    crs = da.raster.crs
    if crs is None:
        raise ValueError("da is missing CRS property, set using `da.raster.set_crs`")
    latlon = crs.is_geographic
    _crs = "geographic" if latlon else "projected"
    _unit = "degree" if latlon else "meter"
    logger.debug(f"Initializing flwdir with {_crs} CRS with unit {_unit}.")
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


def d8_from_dem(
    da_elv: xr.DataArray,
    gdf_stream: Optional[gpd.GeoDataFrame] = None,
    max_depth: float = -1.0,
    outlets: str = "edge",
    idxs_pit: Optional[np.ndarray] = None,
) -> xr.DataArray:
    """Derive D8 flow directions grid from an elevation grid.

    Outlets occur at the edge of valid data or at user defined cells (if `idxs_pit` is provided).
    A local depressions is filled based on its lowest pour point level if the pour point
    depth is smaller than the maximum pour point depth `max_depth`, otherwise the lowest
    elevation in the depression becomes a pit.

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
    idxs_pit: 1D array of int
        Linear indices of outlet cells.

    Returns
    -------
    da_flw: xarray.DataArray
        D8 flow direction grid

    See Also
    --------
    pyflwdir.dem.fill_depressions
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
        idxs_pit=idxs_pit,
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
    ds: xr.Dataset,
    flwdir: pyflwdir.FlwdirRaster,
    scale_ratio: int,
    method: str = "com2",
    uparea_name: Optional[str] = None,
    flwdir_name: str = "flwdir",
    logger=logger,
    **kwargs,
) -> Tuple[xr.DataArray, pyflwdir.FlwdirRaster]:
    """Upscale flow direction network to lower resolution.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset flow direction.
    flwdir : pyflwdir.FlwdirRaster
        Flow direction raster object.
    scale_ratio: int
        Size of upscaled (coarse) grid cells.
    uparea_name : str, optional
        Name of upstream area DataArray, by default None and derived on the fly.
    flwdir_name : str, optional
        Name of upscaled flow direction raster DataArray, by default "flwdir"
    method : {'com2', 'com', 'eam', 'dmm'}
        Upscaling method for flow direction data, by default 'com2'.

    Returns
    -------
    da_flwdir = xarray.DataArray
        Upscaled D8 flow direction grid.
    flwdir_out : pyflwdir.FlwdirRaster
        Upscaled pyflwdir flow direction raster object.

    See Also
    --------
    pyflwdir.FlwdirRaster.upscale
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
    ds_hydro: xr.Dataset,
    da_elv: xr.DataArray,
    river_upa: float = 5.0,
    river_len: float = 1e3,
    uparea_name: str = "uparea",
    flwdir_name: str = "flwdir",
    logger=logger,
    **kwargs,
) -> xr.Dataset:
    """Reproject flow direction and upstream area data to the `da_elv` crs and grid.

    Flow directions are derived from a reprojected grid of synthetic elevation,
    based on the log10 upstream area [m2]. For regions without upstream area, the original
    elevation is used assuming these elevation values are <= 0 (i.e. offshore bathymetry).

    The upstream area on the reprojected grid is based on the new flow directions and
    rivers entering the domain, defined by the minimum upstream area `river_upa` [km2].

    NOTE: the resolution of `ds_hydro` should be similar or smaller than the resolution
    of `da_elv` for good results.
    NOTE: this method is still experimental and might change in the future!

    Parameters
    ----------
    ds_hydro: xarray.Dataset
        Dataset with gridded flow directions named `flwdir_name` and upstream area
        named `uparea_name` [km2].
    da_elv: xarray.DataArray
        DataArray with elevation on destination grid.
    river_upa: float, optional
        Minimum upstream area threshold [km2] for inflowing rivers, by default 5 km2
    river_len: float, optional
        Mimimum distance from river outlet for inflowing river location, by default 1000 m.
    uparea_name, flwdir_name : str, optional
        Name of upstream area (default "uparea") and flow direction ("flwdir") variables
        in `ds_hydro`.
    kwargs: key-word arguments
        key-word arguments are passed to `d8_from_dem`

    Returns
    -------
    xarray.Dataset
        Reprojected gridded dataset with flow direction and upstream area variables.

    See Also
    --------
    d8_from_dem
    """
    # check N->S orientation
    assert da_elv.raster.res[1] < 0
    assert ds_hydro.raster.res[1] < 0
    for name in [uparea_name, flwdir_name]:
        if name not in ds_hydro:
            raise ValueError(f"{name} variable not found in ds_hydro")
    crs = da_elv.raster.crs
    da_upa = ds_hydro[uparea_name]
    nodata = da_upa.raster.nodata
    upa_mask = da_upa != nodata
    rivmask = da_upa > river_upa
    # synthetic elevation -> max(log10(uparea[m2])) - log10(uparea[m2])
    elvsyn = np.log10(np.maximum(1.0, da_upa * 1e3))
    elvsyn = da_upa.where(~upa_mask, elvsyn.max() - elvsyn)
    # take minimum with rank to ensure pits of main rivers have zero syn. elevation
    if np.any(rivmask):
        flwdir_src = flwdir_from_da(ds_hydro[flwdir_name], mask=rivmask)
        elvsyn = elvsyn.where(flwdir_src.rank < 0, np.minimum(flwdir_src.rank, elvsyn))
    # reproject with 'min' to preserve rivers
    elv_mask = da_elv != da_elv.raster.nodata
    elvsyn_reproj = elvsyn.raster.reproject_like(da_elv, method="min")
    # in regions without uparea use elevation, assuming the elevation < 0 (i.e. offshore bathymetry)
    elvsyn_reproj = elvsyn_reproj.where(
        np.logical_or(elvsyn_reproj != nodata, ~elv_mask),
        da_elv - da_elv.where(elvsyn_reproj == nodata).max() - 0.1,  # make sure < 0
    )
    elvsyn_reproj = elvsyn_reproj.where(da_elv != da_elv.raster.nodata, nodata)
    elvsyn_reproj.raster.set_crs(crs)
    elvsyn_reproj.raster.set_nodata(nodata)
    # get flow directions based on reprojected synthetic elevation
    # initiate new flow direction at edge with syn elv <= 0 + inland pits if no kwargs given
    _edge = pyflwdir.gis_utils.get_edge(elv_mask.values)
    if not kwargs:
        _msk = np.logical_and(_edge, elvsyn_reproj <= 0)
        _msk = np.logical_or(_msk, elvsyn_reproj == 0)
        if np.any(_msk):  # False if all pits outside domain
            kwargs.update(idxs_pit=np.where(_msk.values.ravel())[0])
    logger.info(f"Deriving flow direction from reprojected synthethic elevation.")
    da_flw1 = d8_from_dem(elvsyn_reproj, **kwargs)
    flwdir = flwdir_from_da(da_flw1, ftype="d8", mask=elv_mask)
    # find source river cells outside destination grid bbox
    outside_dst = da_upa.raster.geometry_mask(da_elv.raster.box, invert=True)
    area = flwdir.area / 1e6  # area [km2]
    # If any river cell outside the destination grid, vectorize and reproject river segments(!) uparea
    # to set as boundary condition to the upstream area map.
    nriv = 0
    if np.any(np.logical_and(rivmask, outside_dst)):
        feats = flwdir_src.streams(uparea=da_upa.values, mask=rivmask)
        gdf_stream = gpd.GeoDataFrame.from_features(feats, crs=ds_hydro.raster.crs)
        gdf_stream = gdf_stream.sort_values(by="uparea")
        # calculate upstream area with uparea from inflowing rivers at edge
        # get edge river cells indices
        rivupa = da_flw1.raster.rasterize(gdf_stream, col_name="uparea", nodata=0)
        rivmsk = np.logical_and(flwdir.distnc > river_len, rivupa > 0).values
        inflow_idxs = np.where(np.logical_and(rivmsk, _edge).ravel())[0]
        if inflow_idxs.size > 0:
            # map nearest segment to each river edge cell;
            # keep cell which longest distance to outlet per river segment to avoid duplicating uparea
            gdf0 = gpd.GeoDataFrame(
                index=inflow_idxs,
                geometry=gpd.points_from_xy(*flwdir.xy(inflow_idxs)),
                crs=crs,
            )
            gdf0["distnc"] = flwdir.distnc.flat[inflow_idxs]
            gdf0["idx2"], gdf0["dst2"] = gis_utils.nearest(gdf0, gdf_stream)
            gdf0 = gdf0.sort_values("distnc", ascending=False).drop_duplicates("idx2")
            gdf0["uparea"] = gdf_stream.loc[gdf0["idx2"].values, "uparea"].values
            # set stream uparea to selected inflow cells and calculate total uparea
            nriv = gdf0.index.size
            area.flat[gdf0.index.values] = gdf0["uparea"].values
    logger.info(f"Calculating upstream area with {nriv} river inflows.")
    da_upa1 = xr.DataArray(
        dims=da_flw1.raster.dims,
        coords=da_flw1.raster.coords,
        data=flwdir.accuflux(area).astype(np.float32),
        name="uparea",
        attrs=dict(units="km2", _FillValue=-9999),
    ).where(da_elv != nodata, -9999)
    da_upa1.raster.set_crs(crs)
    if logger.getEffectiveLevel() <= 10:
        upa_reproj_max = da_upa.raster.reproject_like(da_elv, method="max")
        max_upa = upa_reproj_max.where(elv_mask).max().values
        max_upa1 = da_upa1.max().values
        logger.debug(f"New/org max upstream area: {max_upa1:.2f}/{max_upa:.2f} km2")
    return xr.merge([da_flw1, da_upa1])


### hydrography maps ###


def gaugemap(
    ds: xr.Dataset,
    idxs: Optional[np.ndarray] = None,
    xy: Optional[Tuple] = None,
    ids: Optional[np.ndarray] = None,
    mask: Optional[xr.DataArray] = None,
    flwdir: Optional[pyflwdir.FlwdirRaster] = None,
    logger=logger,
) -> xr.DataArray:
    """This method is deprecated. See :py:meth:`~hydromt.flw.gauge_map`"""
    warnings.warn(
        'The "gaugemap" method is deprecated, use  "hydromt.flw.gauge_map" instead.',
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


def gauge_map(
    ds: Union[xr.Dataset, xr.DataArray],
    idxs: Optional[np.ndarray] = None,
    xy: Optional[Tuple] = None,
    ids: Optional[np.ndarray] = None,
    stream: Optional[xr.DataArray] = None,
    flwdir: Optional[pyflwdir.FlwdirRaster] = None,
    max_dist: float = 10e3,
    logger=logger,
) -> Tuple[xr.DataArray, np.ndarray, np.ndarray]:
    """Return map with unique gauge IDs.

    Gauge locations should be provided by either x,y coordinates (`xy`) or
    linear indices (`idxs`). Gauge labels (`ids`) can optionally be provided,
    but are by default numbered starting at one.

    If `flwdir` and `stream` are provided, the gauge locations are snapped to the
    nearest downstream river defined by the boolean `stream` mask. Else, the gauge
    locations

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset or Dataarray with destination grid.
    idxs : 1D array or int, optional
        linear indices of gauges, by default is None.
    xy : tuple of 1D array of float, optional
        x, y coordinates of gauges, by default is None.
    ids : 1D array of int32, optional
        IDs of gauges, values must be larger than zero.
        By default None and numbered on the fly.
    flwdir : pyflwdir.FlwdirRaster, optional
        Flow direction raster object, by default None.
    stream: 2D array of bool, optional
        Mask of stream cells used to snap gauges to, by default None
    max_dist: float, optional
        Maximum distance between original and snapped point location.
        A warning is logged if exceeded. By default 10 km.

    Returns
    -------
    da_gauges: xarray.DataArray
        Map with unique gauge IDs
    idxs: 1D array or int
        linear indices of gauges
    ids: 1D array of int
        IDs of gauges
    """
    # Snap if mask and flwdir are not None
    if xy is not None:
        idxs = ds.raster.xy_to_idx(xs=xy[0], ys=xy[1])
    elif idxs is None:
        raise ValueError("Either idxs or xy required")
    if ids is None:
        idxs = np.atleast_1d(idxs)
        ids = np.arange(1, idxs.size + 1, dtype=np.int32)
    # Snapping
    # TODO: should we do the snapping similar to basin_map ??
    if stream is not None and flwdir is not None:
        idxs, dist = flwdir.snap(idxs=idxs, mask=stream, unit="m")
        if np.any(dist > max_dist):
            far = len(dist[dist > max_dist])
            msg = f"Snapping distance of {far} gauge(s) exceeds {max_dist} m"
            warnings.warn(msg, UserWarning)
            logger.warning(msg)
    gauges = np.zeros(ds.raster.shape, dtype=np.int32)
    gauges.flat[idxs] = ids
    da_gauges = xr.DataArray(
        dims=ds.raster.dims,
        coords=ds.raster.coords,
        data=gauges,
        attrs=dict(_FillValue=0),
    )
    return da_gauges, idxs, ids


def outlet_map(da_flw: xr.DataArray, ftype: str = "infer") -> xr.DataArray:
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
        basin outlets/pits ID map
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
        dataset containing all maps for stream criteria
    stream: 2D array of bool, optional
        Initial mask of stream cells. If a stream if provided, it is combined with the
        threshold based map using a logical AND operation.
    stream_kwargs : dict, optional
        Parameter: minimum threshold pairs to define streams.
        Multiple threshold will be combined using a logical AND operation.

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
    ds: xr.Dataset,
    flwdir: pyflwdir.FlwdirRaster,
    xy: Optional[Tuple] = None,
    idxs: Optional[np.ndarray] = None,
    outlets: bool = False,
    ids: Optional[np.ndarray] = None,
    stream: Optional[xr.DataArray] = None,
    **stream_kwargs,
) -> Union[xr.DataArray, Tuple]:
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

    See Also
    --------
    stream_map
    outlet_map
    """
    if not np.all(flwdir.shape == ds.raster.shape):
        raise ValueError("Flwdir and ds dimensions do not match")
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
                "No basin outlets found in domain."
                "Provide 'xy' or 'idxs' outlet locations or set 'outlets=False'"
            )
    da_basins = xr.DataArray(
        data=flwdir.basins(idxs=idxs, xy=xy, ids=ids).astype(np.int32),
        dims=ds.raster.dims,
        coords=ds.raster.coords,
    )
    da_basins.raster.set_nodata(0)
    if idxs is not None:
        xy = flwdir.xy(idxs)
    return da_basins, xy


def basin_shape(
    ds: xr.Dataset,
    flwdir: pyflwdir.FlwdirRaster,
    basin_name: str = "basins",
    mask: bool = True,
    **kwargs,
) -> gpd.GeoDataFrame:
    """This method is be deprecated. Use :py:meth:`~hydromt.flw.basin_map` in combination
    with :py:meth:`~hydromt.raster.RasterDataArray.vectorize` instead.
    """
    warnings.warn(
        "basin_shape is deprecated, use a combination of hydromt.flw.basin_map"
        " and hydromt.raster.RasterDataArray.vectorize instead.",
        DeprecationWarning,
    )
    if basin_name not in ds:
        ds[basin_name] = basin_map(ds, flwdir, **kwargs)[0]
    elif not np.all(flwdir.shape == ds.raster.shape):
        raise ValueError("flwdir and ds dimensions do not match")
    da_basins = ds[basin_name]
    nodata = da_basins.raster.nodata
    if mask and "mask" in da_basins.coords and nodata is not None:
        da_basins = da_basins.where(da_basins.coords["mask"] != 0, nodata)
        da_basins.raster.set_nodata(nodata)
    gdf = da_basins.raster.vectorize().set_index("value").sort_index()
    gdf.index.name = basin_name
    return gdf


def clip_basins(
    ds: xr.Dataset,
    flwdir: pyflwdir.FlwdirRaster,
    xy: Optional[Tuple],
    flwdir_name: str = "flwdir",
    **kwargs,
) -> xr.Dataset:
    """Clip a dataset to a subbasin.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to be clipped, containing flow direction (`flwdir_name`) data
    flwdir : pyflwdir.FlwdirRaster
        Flow direction raster object
    xy : tuple of array_like of float, optional
        x, y coordinates of (sub)basin outlet locations
    flwdir_name : str, optional
        name of flow direction DataArray, by default 'flwdir'
    kwargs : key-word arguments
        Keyword arguments based to the :py:meth:`~hydromt.flw.basin_map` method.

    Returns
    -------
    xarray.Dataset
        clipped dataset

    See Also
    --------
    basin_map
    hydromt.RasterDataArray.clip_mask
    """
    da_basins, xy = basin_map(ds, flwdir, xy, **kwargs)
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


def dem_adjust(
    da_elevtn: xr.DataArray,
    da_flwdir: xr.DataArray,
    da_rivmsk: Optional[xr.DataArray] = None,
    flwdir: Optional[pyflwdir.FlwdirRaster] = None,
    connectivity: int = 4,
    river_d8: bool = False,
    logger=logger,
) -> xr.DataArray:
    """Returns hydrologically conditioned elevation.

    The elevation is conditioned to D4 (`connectivity=4`) or D8 (`connectivity=8`)
    flow directions.

    The method assumes the orignal flow directions are in D8. Therefore, if
    `connectivity=4`, an intermediate D4 conditioned elevation raster is derived
    first, based on which new D4 flow directions are obtained used to condition the
    original elvation.

    Parameters
    ----------
    da_elevtn, da_flwdir, da_rivmsk : xr.DataArray
        elevation [m+REF]
        D8 flow directions [-]
        binary river mask [-], optional
    flwdir : pyflwdir.FlwdirRaster, optional
        D8 flow direction raster object. If None it is derived on the fly from `da_flwdir`.
    connectivity: {4, 8}
        D4 or D8 flow connectivity.
    river_d8 : bool
        If True and `connectivity==4`, additionally condition river cells to D8.
        Requires `da_rivmsk`.

    Returns
    -------
    xr.Dataset
        Dataset with hydrologically adjusted elevation ('elevtn') [m+REF]

    See Also
    --------
    pyflwdir.FlwdirRaster.dem_adjust
    pyflwdir.FlwdirRaster.dem_dig_d4
    """
    # get flow directions for entire domain and for rivers
    if flwdir is None:
        flwdir = flwdir_from_da(da_flwdir, mask=False)
    if connectivity == 4 and river_d8 and da_rivmsk is None:
        raise ValueError('Provide "da_rivmsk" in combination with "river_d8"')
    elevtn = da_elevtn.values
    nodata = da_elevtn.raster.nodata

    logger.info(f"Condition elevation to D{connectivity} flow directions.")
    # get D8 conditioned elevation
    elevtn = flwdir.dem_adjust(elevtn)
    # get D4 conditioned elevation (based on D8 conditioned!)
    if connectivity == 4:
        rivmsk = da_rivmsk.values == 1 if da_rivmsk is not None else None
        # derive D4 flow directions with forced pits at original locations
        d4 = pyflwdir.dem.fill_depressions(
            elevtn=flwdir.dem_dig_d4(elevtn, rivmsk=rivmsk, nodata=nodata),
            nodata=nodata,
            connectivity=connectivity,
            idxs_pit=flwdir.idxs_pit,
        )[1]
        # condition the DEM to the new D4 flow dirs
        flwdir_d4 = pyflwdir.from_array(
            d4, ftype="d8", transform=flwdir.transform, latlon=flwdir.latlon
        )
        elevtn = flwdir_d4.dem_adjust(elevtn)
        # condition river cells to D8
        if river_d8:
            flwdir_river = flwdir_from_da(da_flwdir, mask=rivmsk)
            elevtn = flwdir_river.dem_adjust(elevtn)
        # assert np.all((elv2 - flwdir_d4.downstream(elv2))>=0)

    # save to dataarray
    da_out = xr.DataArray(
        data=elevtn,
        coords=da_elevtn.raster.coords,
        dims=da_elevtn.raster.dims,
    )
    da_out.raster.set_nodata(nodata)
    da_out.raster.set_crs(da_elevtn.raster.crs)
    return da_out
